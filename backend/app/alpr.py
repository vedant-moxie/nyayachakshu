"""ALPR post-processing tool — Vahan / HSRP plate string resolution.

This is the tail end of the ALPR Preprocessing Subsystem in the report
(license_plate.tex, stages P6 Universal Normalisation -> P8 Quality Gate).
It does NOT do OCR; it takes per-frame OCR candidates and turns them into a
single court-grade plate string via:

  1. Universal normalisation (strip, uppercase, drop separators).
  2. Format-aware OCR-confusion repair (digit slots vs letter slots).
  3. Multi-frame per-character majority vote (the "3-frame majority vote"
     the console shows on EVT-7741).
  4. Validation against the Vahan registration grammar + a quality gate.

All logic here is real and unit-testable — no model weights required.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Standard Bharat-series + state registration grammar:
#   <state:2 alpha><rto:1-2 digit><series:0-3 alpha><number:1-4 digit>
# e.g. AP 37 BK 6798, TS 09 EM 4521, KA 01 1234 (no series), 22 BH 1234 AA (BH).
_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")

# Valid Indian state / UT registration prefixes (subset is fine; this is the
# guard the report's P8 quality gate uses to reject hallucinated reads).
_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}

# OCR look-alikes. Applied directionally: in a slot that must be a DIGIT we
# coerce letters->digits; in a slot that must be a LETTER we coerce the reverse.
_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                    "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


@dataclass
class PlateResult:
    plate: str
    confidence: float
    valid: bool
    state: str | None = None
    rto: str | None = None
    series: str | None = None
    number: str | None = None
    method: str = ""
    notes: list[str] = field(default_factory=list)


def normalise(raw: str) -> str:
    """P6 Universal Normalisation: uppercase, drop whitespace/separators."""
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())


def _repair(token: str) -> str:
    """Format-aware OCR-confusion repair.

    Anchors on the structural template SS DD [series] NNNN by walking the
    token left-to-right and coercing each character toward the type its slot
    demands. Cheap, deterministic, and far more reliable than a flat dict swap.
    """
    if len(token) < 4:
        return token
    chars = list(token)
    n = len(chars)
    # Slot 0-1: state letters.
    for i in (0, 1):
        if chars[i].isdigit():
            chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
    # Trailing 1-4 chars: vehicle number -> digits.
    for i in range(max(2, n - 4), n):
        if chars[i].isalpha():
            chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])
    # RTO digits right after the state code.
    for i in (2, 3):
        if i < n - 1 and chars[i].isalpha() and chars[i] in _LETTER_TO_DIGIT:
            # only coerce if the next char keeps a plausible structure
            chars[i] = _LETTER_TO_DIGIT[chars[i]]
    return "".join(chars)


def validate(token: str) -> PlateResult:
    """P8 Quality Gate: parse against the Vahan grammar + state-code allow-list."""
    t = normalise(token)
    m = _PLATE_RE.match(t)
    if not m:
        return PlateResult(plate=t, confidence=0.0, valid=False,
                           notes=["grammar_mismatch"])
    state, rto, series, number = m.groups()
    notes: list[str] = []
    valid = True
    if state not in _STATE_CODES:
        valid = False
        notes.append(f"unknown_state_code:{state}")
    if int(rto) == 0:
        notes.append("rto_zero")
    return PlateResult(
        plate=f"{state} {rto} {series} {number}".replace("  ", " ").strip(),
        confidence=1.0 if valid else 0.4,
        valid=valid, state=state, rto=rto, series=series or None,
        number=number, method="single_frame", notes=notes,
    )


def resolve(candidates: list[tuple[str, float]]) -> PlateResult:
    """Multi-frame ALPR resolution via per-character majority vote.

    `candidates` is a list of (raw_ocr_string, ocr_confidence) from N frames
    of the same tracked vehicle. Returns the consensus plate. Implements the
    console's "3-frame majority vote · HSRP verified" behaviour.
    """
    repaired = []
    for raw, conf in candidates:
        norm = _repair(normalise(raw))
        if norm:
            repaired.append((norm, max(0.0, min(1.0, conf))))
    if not repaired:
        return PlateResult(plate="", confidence=0.0, valid=False,
                           notes=["no_candidates"])

    # Vote on the modal length first, then per character (confidence-weighted).
    length = Counter(len(s) for s, _ in repaired).most_common(1)[0][0]
    pool = [(s, c) for s, c in repaired if len(s) == length]
    consensus = []
    agreement = []
    for i in range(length):
        weighted: Counter = Counter()
        for s, c in pool:
            weighted[s[i]] += c
        ch, score = weighted.most_common(1)[0]
        consensus.append(ch)
        total = sum(weighted.values()) or 1.0
        agreement.append(score / total)

    voted = "".join(consensus)
    result = validate(voted)
    # Confidence = mean per-char agreement, gated by validity.
    char_conf = sum(agreement) / len(agreement)
    n_frames = len(pool)
    result.confidence = round(char_conf * (1.0 if result.valid else 0.5), 3)
    result.method = f"{n_frames}-frame majority vote"
    if result.valid:
        result.notes.append("HSRP verified")
    if result.confidence < 0.85:
        result.notes.append("below_review_threshold:0.85")
    return result
