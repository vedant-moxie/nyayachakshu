"""Unit tests for app/pipelines/violations.py.

Plain-assert tests runnable with bare `python3 tests/test_violations.py`
(no pytest required). Covers a positive and a negative case for each
detector, run_modules gating, and build_violation_record field shaping.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipelines import violations as V  # noqa: E402
from app.penalties import SCHEDULE  # noqa: E402


# --------------------------- wrong-side -----------------------------------

def test_wrong_side_positive():
    # Lane flows east (0 deg); track heads west (180 deg) -> 180 deg angle.
    scene = {"lane_dir_deg": 0.0}
    track = {"track_id": 1, "class_name": "car", "heading_deg": 180.0,
             "speed_kmh": 30.0, "bbox": [0, 0, 10, 10]}
    r = V.detect_wrong_side(track, scene)
    assert r is not None, "full opposition should flag wrong-side"
    assert r["code"] == "wrong_side"
    assert r["confidence"] == 1.0, r["confidence"]


def test_wrong_side_negative_aligned():
    # Heading roughly with flow -> no violation.
    scene = {"lane_dir_deg": 0.0}
    track = {"track_id": 1, "class_name": "car", "heading_deg": 10.0,
             "speed_kmh": 30.0}
    assert V.detect_wrong_side(track, scene) is None


def test_wrong_side_negative_too_slow():
    # Opposes flow but below min-speed gate -> suppressed.
    scene = {"lane_dir_deg": 0.0}
    track = {"track_id": 1, "class_name": "car", "heading_deg": 180.0,
             "speed_kmh": 2.0}
    assert V.detect_wrong_side(track, scene) is None


def test_wrong_side_vector_form():
    scene = {"lane_dir_vec": [1.0, 0.0]}  # east
    track = {"track_id": 1, "class_name": "car", "heading_deg": 175.0,
             "speed_kmh": 25.0}
    r = V.detect_wrong_side(track, scene)
    assert r is not None and r["code"] == "wrong_side"


# --------------------------- stop-line ------------------------------------

def test_stop_line_positive_red():
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    track = {"track_id": 2, "class_name": "car", "stop_line_rel": "AFTER",
             "bbox": [0, 90, 10, 30]}  # bottom at 120 > 100
    r = V.detect_stop_line(track, scene)
    assert r is not None and r["code"] == "stop_line"
    assert 0.0 <= r["confidence"] <= 1.0


def test_stop_line_negative_green():
    # Crossed line but signal is green -> no stop-line offence.
    scene = {"signal_state": "GREEN", "stop_line_y": 100.0}
    track = {"track_id": 2, "class_name": "car", "stop_line_rel": "AFTER"}
    assert V.detect_stop_line(track, scene) is None


def test_stop_line_negative_before():
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    track = {"track_id": 2, "class_name": "car", "stop_line_rel": "BEFORE"}
    assert V.detect_stop_line(track, scene) is None


def test_stop_line_no_signal_uses_geometry():
    # No signal configured: any crossing of the registered line fires.
    scene = {"stop_line_y": 100.0}
    track = {"track_id": 2, "class_name": "car", "bbox": [0, 95, 10, 20]}  # 115 > 100
    r = V.detect_stop_line(track, scene)
    assert r is not None and r["code"] == "stop_line"


# --------------------------- red-light ------------------------------------

def test_red_light_positive():
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    track = {"track_id": 3, "class_name": "car", "stop_line_rel": "AFTER",
             "bbox": [0, 90, 10, 30]}
    r = V.detect_red_light(track, scene)
    assert r is not None and r["code"] == "red_light"
    assert 0.0 <= r["confidence"] <= 1.0


def test_red_light_negative_yellow():
    # Crossed during yellow -> not a red-light violation (legal-attribution).
    scene = {"signal_state": "YELLOW", "stop_line_y": 100.0}
    track = {"track_id": 3, "class_name": "car", "stop_line_rel": "AFTER"}
    assert V.detect_red_light(track, scene) is None


def test_red_light_negative_not_crossed():
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    track = {"track_id": 3, "class_name": "car", "stop_line_rel": "ON"}
    assert V.detect_red_light(track, scene) is None


# --------------------------- illegal parking ------------------------------

def test_illegal_parking_positive():
    scene = {"no_park_zone_ids": ["NP-1", "NP-2"]}
    track = {"track_id": 4, "class_name": "car", "is_stationary": True,
             "stationary_ms": 400_000, "zone_ids": ["NP-2"]}
    r = V.detect_illegal_parking(track, scene)
    assert r is not None and r["code"] == "illegal_parking"
    assert r["evidence"]["no_park_zone_ids"] == ["NP-2"]


def test_illegal_parking_negative_short_dwell():
    scene = {"no_park_zone_ids": ["NP-1"]}
    track = {"track_id": 4, "class_name": "car", "is_stationary": True,
             "stationary_ms": 10_000, "zone_ids": ["NP-1"]}
    assert V.detect_illegal_parking(track, scene) is None


def test_illegal_parking_negative_wrong_zone():
    scene = {"no_park_zone_ids": ["NP-1"]}
    track = {"track_id": 4, "class_name": "car", "is_stationary": True,
             "stationary_ms": 400_000, "zone_ids": ["LANE-3"]}
    assert V.detect_illegal_parking(track, scene) is None


# --------------------------- helmet ---------------------------------------

def test_helmet_positive():
    track = {"track_id": 5, "class_name": "two_wheeler",
             "riders": [{"has_helmet": True}, {"has_helmet": False}]}
    r = V.detect_helmet(track, {})
    assert r is not None and r["code"] == "helmet_absence"
    assert r["evidence"]["rider_count"] == 1


def test_helmet_negative_all_helmeted():
    track = {"track_id": 5, "class_name": "two_wheeler",
             "riders": [{"has_helmet": True}, {"has_helmet": True}]}
    assert V.detect_helmet(track, {}) is None


def test_helmet_negative_not_two_wheeler():
    track = {"track_id": 5, "class_name": "car",
             "riders": [{"has_helmet": False}]}
    assert V.detect_helmet(track, {}) is None


# --------------------------- triple riding --------------------------------

def test_triple_riding_positive():
    track = {"track_id": 6, "class_name": "two_wheeler",
             "riders": [{"has_helmet": True}] * 3}
    r = V.detect_triple_riding(track, {})
    assert r is not None and r["code"] == "triple_riding"
    assert r["evidence"]["rider_count"] == 3


def test_triple_riding_negative_two():
    track = {"track_id": 6, "class_name": "two_wheeler", "rider_count": 2}
    assert V.detect_triple_riding(track, {}) is None


# --------------------------- seatbelt -------------------------------------

def test_seatbelt_driver_positive():
    track = {"track_id": 7, "class_name": "car", "zone_ids": ["WINDSHIELD-A"],
             "driver_belted": False}
    res = V.detect_seatbelt(track, {})
    assert res is not None
    codes = [f["code"] for f in res]
    assert "seatbelt_driver" in codes


def test_seatbelt_passenger_positive():
    track = {"track_id": 7, "class_name": "car", "zone_ids": ["anpr-zone-2"],
             "driver_belted": True, "passenger_present": True,
             "passenger_belted": False}
    res = V.detect_seatbelt(track, {})
    codes = [f["code"] for f in res]
    assert codes == ["seatbelt_passenger"], codes


def test_seatbelt_negative_belted():
    track = {"track_id": 7, "class_name": "car", "zone_ids": ["WINDSHIELD-A"],
             "driver_belted": True}
    assert V.detect_seatbelt(track, {}) is None


def test_seatbelt_negative_not_in_windshield_zone():
    # Fail-safe: unbelted but not in a windshield/ANPR zone -> no read.
    track = {"track_id": 7, "class_name": "car", "zone_ids": ["LANE-1"],
             "driver_belted": False}
    assert V.detect_seatbelt(track, {}) is None


# --------------------------- phone use ------------------------------------

def test_phone_use_positive():
    track = {"track_id": 8, "class_name": "car", "phone_in_use": True}
    r = V.detect_phone_use(track, {})
    assert r is not None and r["code"] == "phone_use"


def test_phone_use_negative():
    track = {"track_id": 8, "class_name": "car", "phone_in_use": False}
    assert V.detect_phone_use(track, {}) is None


# --------------------------- run_modules gating ---------------------------

def test_run_modules_runs_active():
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    track = {"track_id": 9, "class_name": "car", "stop_line_rel": "AFTER",
             "bbox": [0, 90, 10, 30]}
    fired = V.run_modules(track, scene, ["red_light", "stop_line"])
    codes = sorted(f["code"] for f in fired)
    assert codes == ["red_light", "stop_line"], codes


def test_run_modules_gates_inactive():
    # phone_in_use is True but phone_use is NOT in active_modules -> no fire.
    track = {"track_id": 9, "class_name": "car", "phone_in_use": True,
             "stop_line_rel": "AFTER"}
    scene = {"signal_state": "RED", "stop_line_y": 100.0}
    fired = V.run_modules(track, scene, ["red_light"])
    codes = [f["code"] for f in fired]
    assert "phone_use" not in codes
    assert codes == ["red_light"], codes


def test_run_modules_seatbelt_family_alias():
    # Passing the concrete code "seatbelt_driver" routes to the seatbelt family.
    track = {"track_id": 10, "class_name": "car", "zone_ids": ["WINDSHIELD-A"],
             "driver_belted": False}
    fired = V.run_modules(track, {}, ["seatbelt_driver"])
    assert [f["code"] for f in fired] == ["seatbelt_driver"]


def test_run_modules_empty_active():
    track = {"track_id": 10, "class_name": "two_wheeler",
             "riders": [{"has_helmet": False}]}
    assert V.run_modules(track, {}, []) == []


# --------------------------- build_violation_record -----------------------

def test_build_violation_record_fields():
    scene = {"no_park_zone_ids": ["NP-9"]}
    track = {"track_id": 42, "class_name": "truck", "is_stationary": True,
             "stationary_ms": 600_000, "zone_ids": ["NP-9"], "speed_kmh": 0.0}
    fired = V.detect_illegal_parking(track, scene)
    rec = V.build_violation_record(
        track, scene, fired,
        camera_id="CAM-07", event_id="EVT-1", timestamp_utc="2026-06-23T10:00:00Z",
        plate="TS09AB1234", plate_conf=0.91,
    )
    assert rec["violation_type"] == "illegal_parking"
    assert rec["violation_type"] in SCHEDULE
    assert rec["camera_id"] == "CAM-07"
    assert rec["event_id"] == "EVT-1"
    assert rec["track_id"] == 42
    assert rec["vehicle_class"] == "truck"
    assert rec["plate_string"] == "TS09AB1234"
    assert rec["plate_confidence"] == 0.91
    assert rec["zone_id"] == "NP-9"
    assert 0.0 <= rec["confidence"] <= 1.0


def test_build_violation_record_code_maps_to_penalty():
    # Every detector code must exist in the penalty SCHEDULE.
    track = {"track_id": 1, "class_name": "two_wheeler",
             "riders": [{"has_helmet": False}, {"has_helmet": False}]}
    fired = V.detect_helmet(track, {})
    rec = V.build_violation_record(
        track, {}, fired, camera_id="C", event_id="E",
        timestamp_utc="2026-06-23T10:00:00Z",
    )
    assert rec["violation_type"] == "helmet_absence"
    assert rec["violation_type"] in SCHEDULE
    assert rec["plate_string"] is None


# --------------------------- runner ---------------------------------------

def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
    print(f"ran {passed} tests, all passed")


if __name__ == "__main__":
    _run_all()
