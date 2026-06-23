"""MV Act 1988 (as amended 2019) penalty engine.

Maps violation types to the statutory provision and base fine used by the
eChallan penalty schedule. The console books offences against these exact
sections, so the figures here are the single source of truth for challan
totals.

Reference: Motor Vehicles (Amendment) Act 2019 penalty schedule.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Penalty:
    code: str            # internal violation type
    section: str         # MV Act provision string (as shown on the challan)
    base_fine: int       # statutory base fine in INR
    per_rider: bool      # helmet/triple fines scale with number of riders
    label: str


# Single source of truth. Keyed by the violation `code` emitted by the
# detection modules in the report (violation_detection_modules.tex).
SCHEDULE: dict[str, Penalty] = {
    "helmet_absence":   Penalty("helmet_absence",   "MV Act §194D",            1000, True,  "Helmet absence"),
    "triple_riding":    Penalty("triple_riding",    "MV Act §194C",            1000, False, "Triple-riding"),
    "seatbelt_driver":  Penalty("seatbelt_driver",  "MV Act §194B",            1000, False, "Seatbelt non-compliance (driver)"),
    "seatbelt_passenger": Penalty("seatbelt_passenger", "MV Act §194B",        1000, False, "Seatbelt non-compliance (passenger)"),
    "phone_use":        Penalty("phone_use",        "MV Act §184",             5000, False, "Handheld phone use"),
    "wrong_side":       Penalty("wrong_side",       "MV Act §184",             5000, False, "Wrong-side driving"),
    "red_light":        Penalty("red_light",        "MV Act §177 r/w §119",    5000, False, "Red-light running"),
    "stop_line":        Penalty("stop_line",        "MV Act §177",              500, False, "Stop-line crossing"),
    "illegal_parking":  Penalty("illegal_parking",  "MV Act §15 r/w §127",      500, False, "Illegal parking"),
}


def fine_for(code: str, rider_count: int = 1) -> int:
    """Statutory fine for one booked offence.

    Helmet/triple offences detected across multiple riders are booked as a
    single offence whose fine scales per rider, matching enforcement practice.
    """
    p = SCHEDULE.get(code)
    if p is None:
        return 0
    if p.per_rider and rider_count > 1:
        return p.base_fine * rider_count
    return p.base_fine


def section_for(code: str) -> str:
    p = SCHEDULE.get(code)
    return p.section if p else "—"


def challan_total(offences: list[dict]) -> int:
    """Sum the `fine` field across booked offences (defensive against gaps)."""
    return sum(int(o.get("fine", 0)) for o in offences)
