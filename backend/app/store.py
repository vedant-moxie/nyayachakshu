"""In-memory application state.

On boot it replays the seed events into the evidence hash chain so the ledger
is populated and verifiable from the first request. A real deployment swaps
this for the PostgreSQL schema in the report (§Database Schema); the public
surface (functions below) stays identical.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import alpr, penalties, seed
from .hashchain import HashChain, sha256_hex

DATE_UTC = "2026-06-23"

# Anonymised registered-owner names (DPDP Act data-minimisation — see report).
OWNERS = ["R█████ K████", "S████ R██", "M█████ P███", "V█████ N████",
          "A███ S█████", "K████ B██", "D█████ J███"]


class Store:
    def __init__(self) -> None:
        self.cases: list[dict] = seed.CASES
        self.ledger = HashChain()
        self._index = {c["id"]: c for c in self.cases}
        self._seed_ledger()

    # ---------------------------------------------------------------- ledger
    def _seed_ledger(self) -> None:
        """Replay each seed event into the per-camera hash chain."""
        for idx, c in enumerate(self.cases):
            ts = f"{DATE_UTC}T{c['sig']}+05:30"
            # On-device digest of the raw (unannotated) primary frame.
            raw_sha = sha256_hex(f"{c['id']}|{c['cam']}|{c['frame']}|raw".encode())
            for v in c["viols"]:
                payload = {
                    "event_id": c["id"],
                    "camera_id": c["cam"],
                    "timestamp_utc": ts,
                    "violation_type": v["code"],
                    "track_id": int(c["tracks"]),
                    "confidence": float(v["sc"]),
                    "plate_string": c["plate"] if c["pconf"] else None,
                    "plate_confidence": c["pconf"],
                    "zone_id": c["cam"],
                }
                self.ledger.append(c["cam"], c["id"], raw_sha, payload, ts)

    # ------------------------------------------------------------- accessors
    def list_cases(self) -> list[dict]:
        return self.cases

    def get_case(self, case_id: str) -> dict | None:
        return self._index.get(case_id)

    def case_index(self, case_id: str) -> int:
        for i, c in enumerate(self.cases):
            if c["id"] == case_id:
                return i
        return -1

    # --------------------------------------------------------------- challan
    def build_challan(self, case_id: str) -> dict | None:
        c = self.get_case(case_id)
        if c is None:
            return None
        idx = self.case_index(case_id)
        offences = []
        for v in c["viols"]:
            fine = penalties.fine_for(v["code"], v.get("riders", 1))
            offences.append({
                "offence": v["nm"],
                "section": penalties.section_for(v["code"]),
                "code": v["code"],
                "ai_confidence": float(v["sc"]),
                "fine": fine,
            })
        total = penalties.challan_total(offences)
        challan_no = f"TS/CYB/2026/00{41187 + idx}"
        # Evidence-bundle integrity: chain head for this camera.
        entries = self.ledger.entries(c["cam"])
        evidence_hash = entries[-1].record_hash if entries else None
        return {
            "challan_no": challan_no,
            "issued_on": DATE_UTC,
            "issued_time_ist": c["sig"],
            "authority": "Telangana Traffic Police",
            "event_id": c["id"],
            "camera_id": c["cam"],
            "location": c["loc"],
            "gps": c["gps"],
            "registration": c["plate"] if c["pconf"] else "under verification",
            "plate_confidence": c["pconf"],
            "registered_owner": OWNERS[idx % len(OWNERS)],
            "offences": offences,
            "total_payable_inr": total,
            "evidence_frame_sha256": evidence_hash,
            "ledger_camera": c["cam"],
            "status": "PENDING_REVIEW" if c["pconf"] and c["pconf"] < 0.85 else "AUTO_CLEARED",
        }

    # ---------------------------------------------------------- alpr passthru
    def run_alpr(self, candidates: list[tuple[str, float]]) -> dict:
        r = alpr.resolve(candidates)
        return r.__dict__


store = Store()
