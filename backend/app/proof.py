"""Self-verification harness.

Everything here runs live on request and returns *checkable* evidence — not a
claim that something works, but the inputs and outputs so a sceptic can
re-derive the result themselves (e.g. recompute a SHA-256 by hand). Used by
GET /api/proof and the /proof.html page.
"""
from __future__ import annotations

import json

from . import alpr, penalties
from .hashchain import HashChain, _canonical, sha256_hex


def _check(name: str, ok: bool, expected, actual, detail: str = "") -> dict:
    return {"name": name, "pass": bool(ok), "expected": expected,
            "actual": actual, "detail": detail}


def prove_hashchain_recompute() -> dict:
    """A skeptic can recompute this digest with `sha256` over the shown bytes."""
    chain = HashChain()
    e = chain.append(
        camera_id="CAM-PROOF", event_id="EVT-PROOF-1",
        raw_frame_sha256=sha256_hex(b"raw-frame-bytes"),
        payload={"violation_type": "red_light", "confidence": 0.97},
        recorded_at="2026-06-23T14:23:28+05:30",
    )
    body = {
        "seq": e.seq, "camera_id": e.camera_id, "event_id": e.event_id,
        "recorded_at": e.recorded_at, "raw_frame_sha256": e.raw_frame_sha256,
        "payload": e.payload, "prev_hash": e.prev_hash,
    }
    canonical = _canonical(body).decode()
    recomputed = sha256_hex(canonical.encode())
    return _check(
        "Hash chain digest is reproducible SHA-256",
        recomputed == e.record_hash, e.record_hash, recomputed,
        detail=(f"Run this yourself: printf '%s' '{canonical}' | shasum -a 256  "
                f"→ must equal {e.record_hash}"),
    )


def prove_tamper_detection() -> dict:
    """Build a clean chain, confirm intact, tamper one record, confirm caught."""
    chain = HashChain()
    for i in range(3):
        chain.append("CAM-PROOF", f"EVT-{i}", sha256_hex(f"f{i}".encode()),
                     {"violation_type": "stop_line", "confidence": 0.9 + i / 100})
    before = chain.verify("CAM-PROOF")
    # Retroactively alter a stored record's payload (an attacker editing history).
    chain.entries("CAM-PROOF")[1].payload["confidence"] = 0.01
    after = chain.verify("CAM-PROOF")
    return _check(
        "Tampering a sealed record is detected",
        before["intact"] is True and after["intact"] is False,
        {"clean_chain_intact": True, "after_tamper_intact": False},
        {"clean_chain_intact": before["intact"],
         "after_tamper_intact": after["intact"],
         "breaks": after["breaks"]},
        detail="One record's confidence was changed from ~0.91 to 0.01 after "
               "sealing; verify() recomputes every link and flags it.",
    )


def prove_alpr_consensus() -> dict:
    """Noisy OCR across frames must resolve to the correct, valid plate."""
    candidates = [("AP37BK6798", 0.80), ("4P37BK6798", 0.60),
                  ("AP37BK6798", 0.90), ("APE7BK6798", 0.55)]
    r = alpr.resolve(candidates)
    return _check(
        "Multi-frame ALPR corrects OCR noise to a valid Vahan plate",
        r.plate == "AP 37 BK 6798" and r.valid,
        {"plate": "AP 37 BK 6798", "valid": True},
        {"plate": r.plate, "valid": r.valid,
         "confidence": r.confidence, "method": r.method},
        detail=f"Inputs (with errors): {[c[0] for c in candidates]} "
               "— note '4P37...' and 'APE7...' are wrong; consensus fixes them.",
    )


def prove_alpr_rejects_garbage() -> dict:
    """An invalid string must NOT be accepted as a real plate."""
    r = alpr.validate("ZZ99ZZ9999")  # ZZ is not a real state code
    return _check(
        "ALPR rejects an invalid registration",
        r.valid is False, {"valid": False}, {"valid": r.valid, "notes": r.notes},
        detail="ZZ is not a recognised Indian state/UT code → quality gate fails.",
    )


def prove_penalty_math() -> dict:
    """Challan total = sum of statutory fines; helmet scales per rider."""
    offences = [
        {"code": "triple_riding", "fine": penalties.fine_for("triple_riding", 1)},
        {"code": "helmet_absence", "fine": penalties.fine_for("helmet_absence", 3)},
    ]
    total = penalties.challan_total(offences)
    return _check(
        "MV Act penalty totals are computed, not hardcoded",
        total == 4000,
        {"triple_riding": 1000, "helmet_absence(3 riders)": 3000, "total": 4000},
        {"triple_riding": offences[0]["fine"],
         "helmet_absence(3 riders)": offences[1]["fine"], "total": total},
        detail="₹1000 (§194C) + 3 × ₹1000 (§194D, per un-helmeted rider) = ₹4000.",
    )


def run_all() -> dict:
    checks = [
        prove_hashchain_recompute(),
        prove_tamper_detection(),
        prove_alpr_consensus(),
        prove_alpr_rejects_garbage(),
        prove_penalty_math(),
    ]
    passed = sum(1 for c in checks if c["pass"])
    return {
        "suite": "NyayaChakshu live self-verification",
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
        "note": "These run live on the server at request time. The hash-chain "
                "check prints the exact bytes so you can recompute the SHA-256 "
                "yourself and confirm nothing is faked.",
    }
