"""End-to-end perception → routing → violation → evidence orchestrator.

Chains the three independently-built and -tested pipeline stages into the
single sequence the report describes (perception.tex Layers 1-5 →
violation_detection_modules.tex → evidence packaging):

    detections  ──Layer 1-4──►  TrackedObject metadata
                ──Layer 5──────►  active violation modules (routing)
                ──detectors────►  fired ViolationRecords
                ──hash chain───►  tamper-evident evidence ledger entry

Everything here is stdlib-only and deterministic, so the whole stack runs
with no model weights and no GPU — suitable for a demo / CI smoke test.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import perception as _perception
from . import violations as _violations


def run_scene(frames: list[list[dict]], scene: dict,
              camera_id: str = "CAM-07A") -> dict:
    """Run a multi-frame scene end-to-end.

    `frames` is a list of frames; each frame is a list of detection dicts
    ({class_id, class_name, bbox, score, zone_ids}). `scene` carries the
    one-time geometry (lane_dir_deg, stop_line_y, signal_state, no_park_zone_ids).
    Returns the full per-track trace plus the violation records produced.
    """
    pipe = _perception.PerceptionPipeline()
    last_tracks: list[dict] = []
    last_dets: list[dict] = []
    for i, dets in enumerate(frames):
        last_tracks = pipe.process_frame(i, dets)
        last_dets = dets

    # Carry per-object attributes (helmet/seatbelt/phone) from the source
    # detection onto its track — perception only emits geometric/kinematic
    # schema fields, so match by best-IoU and merge the attribute keys.
    _ATTR_KEYS = ("riders", "driver_belted", "passenger_belted", "phone_in_use")
    for t in last_tracks:
        best, best_iou = None, 0.0
        for d in last_dets:
            i_ou = _perception.iou(t["bbox"], d["bbox"])
            if i_ou > best_iou:
                best, best_iou = d, i_ou
        if best and best_iou > 0.3:
            for k in _ATTR_KEYS:
                if k in best:
                    t[k] = best[k]

    trace = []
    records = []
    ts = datetime.now(timezone.utc).isoformat()
    for t in last_tracks:
        active = pipe.route(t)
        fired = _violations.run_modules(t, scene, active)
        trace.append({
            "track_id": t["track_id"],
            "class_name": t.get("class_name"),
            "track_state": t["track_state"],
            "speed_kmh": round(t.get("speed_kmh", -1), 1),
            "heading_deg": round(t.get("heading_deg", 0), 1),
            "is_stationary": t.get("is_stationary"),
            "zone_ids": t.get("zone_ids", []),
            "routed_modules": active,
            "violations": fired,
        })
        for f in fired:
            records.append(_violations.build_violation_record(
                t, scene, f, camera_id=camera_id,
                event_id=f"EVT-{t['track_id']:04d}", timestamp_utc=ts,
                plate=scene.get("plate"), plate_conf=scene.get("plate_conf", 0.0)))

    return {
        "camera_id": camera_id,
        "frames_processed": len(frames),
        "tracks": trace,
        "violation_records": records,
    }
