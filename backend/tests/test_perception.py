"""Tests for the NyayaChakshu perception pipeline (Layers 2-5).

Runnable two ways:
    cd backend && python3 -m pytest tests/test_perception.py
    cd backend && python3 tests/test_perception.py     # plain assert harness

No third-party deps required (stdlib + pytest-optional).
"""
from __future__ import annotations

import os
import sys

# Make ``app`` importable whether run from backend/ or via pytest rootdir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.pipelines.perception import PerceptionPipeline, iou  # noqa: E402


def _find(results, track_id):
    for r in results:
        if r["track_id"] == track_id:
            return r
    return None


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0
    # half-overlap on x: intersection 5x10=50, union 100+100-50=150
    assert abs(iou([0, 0, 10, 10], [5, 0, 10, 10]) - (50.0 / 150.0)) < 1e-9


def test_track_id_persists_across_three_frames():
    p = PerceptionPipeline(fps=25.0, px_per_metre=8.0)
    det = lambda x: [{"class_id": 3, "class_name": "car", "bbox": [x, 100, 40, 30],
                      "score": 0.9, "zone_ids": []}]
    r0 = p.process_frame(0, det(100))
    tid = r0[0]["track_id"]
    r1 = p.process_frame(1, det(104))
    r2 = p.process_frame(2, det(108))
    assert _find(r1, tid) is not None, "track lost on frame 1"
    assert _find(r2, tid) is not None, "track lost on frame 2"
    # exactly one track should exist throughout
    assert len(r1) == 1 and len(r2) == 1


def test_tentative_to_confirmed_promotion():
    p = PerceptionPipeline(fps=25.0)
    det = lambda x: [{"class_id": 3, "class_name": "car", "bbox": [x, 50, 40, 30],
                      "score": 0.9, "zone_ids": []}]
    r0 = p.process_frame(0, det(200))
    assert r0[0]["track_state"] == "TENT", "new track must start TENT"
    r1 = p.process_frame(1, det(203))
    assert r1[0]["track_state"] == "TENT", "still TENT after 2 hits"
    r2 = p.process_frame(2, det(206))
    assert r2[0]["track_state"] == "CONF", "3rd hit must promote to CONF"


def test_track_goes_lost_when_unmatched():
    p = PerceptionPipeline(fps=25.0)
    det = [{"class_id": 3, "class_name": "car", "bbox": [10, 10, 40, 30],
            "score": 0.9, "zone_ids": []}]
    r0 = p.process_frame(0, det)
    tid = r0[0]["track_id"]
    last = None
    # 6 empty frames > MAX_MISSED_FRAMES (5) -> LOST emitted
    for f in range(1, 8):
        last = p.process_frame(f, [])
    # the LOST transition should have been emitted at some point
    # (track no longer present in steady state after going LOST)
    assert all(r["track_id"] != tid or r["track_state"] == "LOST"
               for batch in [last] for r in batch)


def test_speed_and_heading_on_moving_box():
    # 8 px/metre, 25 fps. Box is 100px wide; move +20 px/frame in x so the
    # boxes still overlap (IoU > 0.3) and the tracker keeps the same id.
    # 20 px = 2.5 m per frame; per frame = 1/25 s -> 62.5 m/s -> 225 km/h.
    p = PerceptionPipeline(fps=25.0, px_per_metre=8.0)
    det = lambda x: [{"class_id": 3, "class_name": "car", "bbox": [x, 100, 100, 60],
                      "score": 0.9, "zone_ids": []}]
    r0 = p.process_frame(0, det(0))
    tid = r0[0]["track_id"]
    r1 = p.process_frame(1, det(20))
    assert _find(r1, tid) is not None, "track id changed -- IoU match failed"
    obj = _find(r1, tid)
    assert abs(obj["speed_kmh"] - 225.0) < 1.0, obj["speed_kmh"]
    # moving purely +x (rightwards) -> heading 0 deg
    assert abs(obj["heading_deg"] - 0.0) < 1e-6 or abs(obj["heading_deg"] - 360.0) < 1e-6
    assert obj["velocity_px"][0] > 0 and abs(obj["velocity_px"][1]) < 1e-9
    assert obj["is_stationary"] is False


def test_stationary_on_still_box():
    p = PerceptionPipeline(fps=25.0, px_per_metre=8.0, v_stat_kmh=2.0)
    det = [{"class_id": 3, "class_name": "car", "bbox": [300, 300, 50, 40],
            "score": 0.9, "zone_ids": ["no_park_zone_A"]}]
    p.process_frame(0, det)
    last = None
    for f in range(1, 6):
        last = p.process_frame(f, det)
    obj = last[0]
    assert obj["is_stationary"] is True, obj["speed_kmh"]
    # 5 update frames at 40 ms each = 200 ms accumulated
    assert obj["stationary_ms"] >= 150, obj["stationary_ms"]


def test_routing_two_wheeler():
    p = PerceptionPipeline()
    tracked = {
        "track_state": "CONF", "class_name": "motorcycle",
        "speed_kmh": 30.0, "track_age": 12, "zone_ids": [],
        "is_stationary": False, "stationary_ms": 0, "stop_line_rel": None,
    }
    mods = p.route(tracked)
    assert "helmet" in mods and "triple_riding" in mods, mods


def test_routing_car_windshield():
    p = PerceptionPipeline()
    tracked = {
        "track_state": "CONF", "class_name": "car",
        "speed_kmh": 25.0, "track_age": 12,
        "zone_ids": ["windshield_roi"],
        "is_stationary": False, "stationary_ms": 0, "stop_line_rel": None,
    }
    mods = p.route(tracked)
    assert "seatbelt_driver" in mods and "phone_use" in mods, mods


def test_routing_red_light_and_stop_line():
    p = PerceptionPipeline()
    tracked = {
        "track_state": "CONF", "class_name": "car", "speed_kmh": 20.0,
        "track_age": 5, "zone_ids": ["intersection", "signal_red"],
        "is_stationary": False, "stationary_ms": 0, "stop_line_rel": "AFTER",
    }
    mods = p.route(tracked)
    assert "stop_line" in mods and "red_light" in mods, mods


def test_routing_illegal_parking():
    p = PerceptionPipeline()
    tracked = {
        "track_state": "CONF", "class_name": "car", "speed_kmh": 0.0,
        "track_age": 50, "zone_ids": ["no_parking_main_st"],
        "is_stationary": True, "stationary_ms": 120_000, "stop_line_rel": None,
    }
    mods = p.route(tracked)
    assert mods == ["illegal_parking"], mods


def test_routing_ignores_unconfirmed():
    p = PerceptionPipeline()
    tracked = {
        "track_state": "TENT", "class_name": "motorcycle", "speed_kmh": 30.0,
        "track_age": 1, "zone_ids": [], "is_stationary": False,
        "stationary_ms": 0, "stop_line_rel": None,
    }
    assert p.route(tracked) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
