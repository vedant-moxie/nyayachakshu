"""Evidence hash chain — tamper-evident legal chain of custody.

Implements the report's "running hash chain" (starter.tex, §Cryptographic
Evidence Integrity): each violation record stored for a camera carries the
SHA-256 of the previous record for that camera, forming a blockchain-like
structure. Any retrospective deletion, insertion or reordering breaks the
chain and is caught by `verify()`.

This is fully functional, not a mock — the digests are real SHA-256 over a
canonical JSON serialisation, and the verifier recomputes the entire chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

GENESIS = "0" * 64


def _canonical(payload: dict) -> bytes:
    """Deterministic serialisation so the digest is reproducible."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class LedgerEntry:
    seq: int
    camera_id: str
    event_id: str
    recorded_at: str          # UTC ISO-8601
    raw_frame_sha256: str     # P-on-device digest of the UNANNOTATED frame
    payload: dict             # the ViolationRecord (minus hash fields)
    prev_hash: str
    record_hash: str = ""

    def compute_hash(self) -> str:
        body = {
            "seq": self.seq,
            "camera_id": self.camera_id,
            "event_id": self.event_id,
            "recorded_at": self.recorded_at,
            "raw_frame_sha256": self.raw_frame_sha256,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        return sha256_hex(_canonical(body))


@dataclass
class HashChain:
    """One append-only chain per camera, plus a global verifier."""
    _by_camera: dict[str, list[LedgerEntry]] = field(default_factory=dict)

    def append(self, camera_id: str, event_id: str, raw_frame_sha256: str,
               payload: dict, recorded_at: str | None = None) -> LedgerEntry:
        chain = self._by_camera.setdefault(camera_id, [])
        prev_hash = chain[-1].record_hash if chain else GENESIS
        entry = LedgerEntry(
            seq=len(chain),
            camera_id=camera_id,
            event_id=event_id,
            recorded_at=recorded_at or datetime.now(timezone.utc).isoformat(),
            raw_frame_sha256=raw_frame_sha256,
            payload=payload,
            prev_hash=prev_hash,
        )
        entry.record_hash = entry.compute_hash()
        chain.append(entry)
        return entry

    def entries(self, camera_id: str | None = None) -> list[LedgerEntry]:
        if camera_id:
            return list(self._by_camera.get(camera_id, []))
        out: list[LedgerEntry] = []
        for c in self._by_camera.values():
            out.extend(c)
        return out

    def verify(self, camera_id: str | None = None) -> dict:
        """Recompute every chain and report integrity.

        Detects: hash tampering, broken prev->record links, and seq gaps
        (insertion / deletion). Returns a court-presentable proof-of-completeness
        summary for the requested window.
        """
        cameras = [camera_id] if camera_id else list(self._by_camera)
        breaks: list[dict] = []
        total = 0
        for cam in cameras:
            chain = self._by_camera.get(cam, [])
            prev = GENESIS
            for i, e in enumerate(chain):
                total += 1
                recomputed = e.compute_hash()
                if e.seq != i:
                    breaks.append({"camera_id": cam, "seq": e.seq,
                                   "issue": "sequence_gap", "expected_seq": i})
                if e.prev_hash != prev:
                    breaks.append({"camera_id": cam, "seq": e.seq,
                                   "issue": "broken_link"})
                if recomputed != e.record_hash:
                    breaks.append({"camera_id": cam, "seq": e.seq,
                                   "issue": "hash_mismatch"})
                prev = e.record_hash
        return {
            "intact": len(breaks) == 0,
            "cameras_checked": len(cameras),
            "records_checked": total,
            "breaks": breaks,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
