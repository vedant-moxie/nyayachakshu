"""Violation state machines (report §"Violation-Specific Detection Modules").

Layer 5 of the NyayaChakshu pipeline. Each detector is a lightweight,
zero-inference geometric/temporal rule that consumes the perception layer's
tracked-object output plus pre-cached scene geometry and emits a violation
*event* (not evidence). This mirrors the report's design where violation
modules are pure post-processing on the shared BoT-SORT track stream
(violation_detection_modules.tex) and the attribute checks
(triple_rider_and_helment.tex, seatbelt.tex).

STDLIB ONLY. A ``track`` is a plain dict shaped like the perception module's
TrackedObject (see app/schemas.py):

    {
      "track_id": int,
      "class_id": int,
      "class_name": str,            # "two_wheeler" | "car" | "truck" | ...
      "bbox": [x, y, w, h],
      "speed_kmh": float,
      "heading_deg": float,         # direction of travel, degrees [0,360)
      "is_stationary": bool,
      "stationary_ms": int,
      "zone_ids": [str, ...],
      "stop_line_rel": "BEFORE" | "ON" | "AFTER" | None,
      "centroid": [x, y],           # optional
      # attribute-check fields (upstream classifiers):
      "riders": [{"has_helmet": bool}, ...],   # two-wheeler riders
      "rider_count": int,                       # optional override
      "driver_belted": bool,
      "passenger_belted": bool,
      "passenger_present": bool,
      "phone_in_use": bool,
    }

``scene`` carries the per-camera geometry/state (never hardcoded per camera):

    {
      "lane_dir_deg": float,        # permitted lane-flow direction (degrees)
      # or a vector form:
      "lane_dir_vec": [dx, dy],
      "stop_line_y": float,         # registered stop-line image-y
      "signal_state": "RED" | "YELLOW" | "GREEN" | "DARK" | None,
      "no_park_zone_ids": [str, ...],
      "park_dwell_ms": int,         # optional override of default dwell
      "traffic_side": "right" | "left",  # for driver/passenger seat prior
    }

Every detector returns either ``None`` or a dict::

    {"code": <penalty code str>, "confidence": float, "evidence": {...}}

where ``code`` is one of the strings in app/penalties.SCHEDULE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- Tunables (report-sourced defaults) -----------------------------------

# Wrong-side: flag when heading opposes permitted flow by > this angle.
WRONG_SIDE_ANGLE_DEG = 120.0
# Below this speed direction is undetermined (Gate 2, ~5 km/h world space).
WRONG_SIDE_MIN_SPEED_KMH = 5.0

# Illegal parking dwell threshold (loading/unloading grace = 5 min).
PARKING_DWELL_MS_DEFAULT = 300_000

# Two-wheeler triple-riding threshold.
TRIPLE_RIDING_MIN_RIDERS = 3

# Vehicle classes that count as two-wheelers / cars.
TWO_WHEELER_CLASSES = {"two_wheeler", "motorcycle", "motorbike", "scooter"}
CAR_CLASSES = {"car", "truck", "bus", "van", "suv"}

# Zones in which an interior (windshield/ANPR) seatbelt read is possible.
WINDSHIELD_ZONE_HINTS = ("windshield", "anpr")


def _clamp01(x: float) -> float:
    """Clamp to the closed unit interval."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _class_name(track: dict) -> str:
    return str(track.get("class_name", "")).lower()


def _is_two_wheeler(track: dict) -> bool:
    return _class_name(track) in TWO_WHEELER_CLASSES


def _is_car(track: dict) -> bool:
    return _class_name(track) in CAR_CLASSES


def _rider_count(track: dict) -> int:
    riders = track.get("riders")
    if isinstance(riders, list):
        return len(riders)
    return int(track.get("rider_count", 0) or 0)


def _lane_dir_deg(scene: dict):
    """Permitted lane-flow direction in degrees, or None if unspecified.

    Accepts either ``lane_dir_deg`` (scalar degrees) or ``lane_dir_vec``
    ([dx, dy] in image coords). Returns None when neither is usable.
    """
    if scene.get("lane_dir_deg") is not None:
        return float(scene["lane_dir_deg"]) % 360.0
    vec = scene.get("lane_dir_vec")
    if vec and len(vec) == 2 and (vec[0] or vec[1]):
        return math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 360.0
    return None


def _angle_between(a_deg: float, b_deg: float) -> float:
    """Smallest absolute angle between two headings, in [0, 180]."""
    d = abs((a_deg - b_deg) % 360.0)
    return d if d <= 180.0 else 360.0 - d


# ====================================================================
#  Geometric / temporal violations (violation_detection_modules.tex)
# ====================================================================

def detect_wrong_side(track: dict, scene: dict):
    """Wrong-side (contraflow) driving — §subsec:wrongside.

    Trajectory-based rule: compare the track's motion heading against the
    lane's permitted-direction vector. Flag when the angle between them
    exceeds WRONG_SIDE_ANGLE_DEG (~120 deg, i.e. travelling against flow),
    after the minimum-speed gate (Gate 2) that suppresses stationary/near-
    stationary tracks whose heading is dominated by localisation noise.
    """
    perm = _lane_dir_deg(scene)
    if perm is None:
        return None

    speed = float(track.get("speed_kmh", -1.0))
    # Gate 2: undetermined direction below v_min (negative speed = unknown).
    if speed >= 0.0 and speed < WRONG_SIDE_MIN_SPEED_KMH:
        return None

    heading = float(track.get("heading_deg", 0.0)) % 360.0
    angle = _angle_between(heading, perm)
    if angle <= WRONG_SIDE_ANGLE_DEG:
        return None

    # Confidence scales with how far past the 120 deg threshold we are.
    # 120 deg -> 0.0; 180 deg (full opposition) -> 1.0.
    span = 180.0 - WRONG_SIDE_ANGLE_DEG
    margin = (angle - WRONG_SIDE_ANGLE_DEG) / span if span > 0 else 1.0
    confidence = _clamp01(0.6 + 0.4 * margin)
    return {
        "code": "wrong_side",
        "confidence": confidence,
        "evidence": {
            "heading_deg": heading,
            "permitted_deg": perm,
            "angle_deg": round(angle, 2),
            "speed_kmh": speed,
        },
    }


def _crossed_stop_line(track: dict, scene: dict):
    """True if the track's ground-contact point is past the registered line.

    Prefers the upstream ``stop_line_rel`` flag (BEFORE/ON/AFTER from the
    perception layer); falls back to comparing the bbox bottom-centre y
    against ``stop_line_y`` (image-plane mode, positive side = junction box,
    larger y = further into the junction in image coords).
    """
    rel = track.get("stop_line_rel")
    rel = rel.value if hasattr(rel, "value") else rel
    if rel is not None:
        return rel == "AFTER"
    y_stop = scene.get("stop_line_y")
    if y_stop is None:
        return None
    bbox = track.get("bbox") or [0, 0, 0, 0]
    front_y = float(bbox[1]) + float(bbox[3])  # bottom edge of bbox
    return front_y > float(y_stop)


def detect_stop_line(track: dict, scene: dict):
    """Stop-line crossing — §subsec:stopline.

    A stop-line violation fires when the vehicle's front proxy is past the
    registered stop line. When a signal is present the check is gated to the
    RED phase (Gate 1); when no signal is configured, any crossing of the
    registered line is a stop-line offence.
    """
    crossed = _crossed_stop_line(track, scene)
    if not crossed:  # None or False
        return None

    signal = scene.get("signal_state")
    signal = signal.upper() if isinstance(signal, str) else signal
    if signal is not None and signal != "RED":
        # Green/yellow: module suppressed (report Gate 1).
        return None

    # Confidence: how far past the line (image-y margin) when geometry known,
    # else a solid default from the discrete AFTER flag.
    confidence = 0.85
    y_stop = scene.get("stop_line_y")
    bbox = track.get("bbox")
    if y_stop is not None and bbox:
        front_y = float(bbox[1]) + float(bbox[3])
        margin_px = front_y - float(y_stop)
        # 0 px past -> 0.7; >=40 px past -> ~0.98.
        confidence = _clamp01(0.7 + 0.28 * min(margin_px / 40.0, 1.0))
    return {
        "code": "stop_line",
        "confidence": confidence,
        "evidence": {
            "stop_line_rel": track.get("stop_line_rel"),
            "signal_state": signal,
            "stop_line_y": y_stop,
        },
    }


def detect_red_light(track: dict, scene: dict):
    """Red-light running — §subsec:redlight.

    Formal event: signal confirmed RED AND the vehicle crosses the stop line
    into the junction box (stop_line_rel == AFTER during RED). This is a
    strict superset of the stop-line crossing geometry plus the signal-state
    pre-condition.
    """
    signal = scene.get("signal_state")
    signal = signal.upper() if isinstance(signal, str) else signal
    if signal != "RED":
        return None

    crossed = _crossed_stop_line(track, scene)
    if not crossed:  # None or False
        return None

    confidence = 0.9
    y_stop = scene.get("stop_line_y")
    bbox = track.get("bbox")
    if y_stop is not None and bbox:
        front_y = float(bbox[1]) + float(bbox[3])
        margin_px = front_y - float(y_stop)
        confidence = _clamp01(0.78 + 0.2 * min(margin_px / 40.0, 1.0))
    return {
        "code": "red_light",
        "confidence": confidence,
        "evidence": {
            "signal_state": signal,
            "stop_line_rel": track.get("stop_line_rel"),
            "stop_line_y": y_stop,
        },
    }


def detect_illegal_parking(track: dict, scene: dict):
    """Illegal parking — §subsec:parking.

    Fires when the vehicle is confirmed stationary, has dwelled past the
    zone-type dwell threshold (default 5 min loading/unloading grace), AND
    its ground-contact point lies inside a registered no-park zone.
    """
    if not track.get("is_stationary"):
        return None

    dwell_ms = int(track.get("stationary_ms", 0) or 0)
    threshold = int(scene.get("park_dwell_ms", PARKING_DWELL_MS_DEFAULT))
    if dwell_ms < threshold:
        return None

    no_park = set(scene.get("no_park_zone_ids") or [])
    track_zones = set(track.get("zone_ids") or [])
    matched = track_zones & no_park
    if not matched:
        return None

    # Confidence scales with dwell time past threshold (capped at 2x).
    over = (dwell_ms - threshold) / threshold if threshold > 0 else 1.0
    confidence = _clamp01(0.7 + 0.3 * min(over, 1.0))
    return {
        "code": "illegal_parking",
        "confidence": confidence,
        "evidence": {
            "stationary_ms": dwell_ms,
            "dwell_threshold_ms": threshold,
            "no_park_zone_ids": sorted(matched),
        },
    }


# ====================================================================
#  Attribute-based violations (triple_rider_and_helment.tex, seatbelt.tex)
# ====================================================================

def detect_helmet(track: dict, scene: dict):
    """Helmet absence on a two-wheeler — triple_rider_and_helment.tex.

    Per-rider helmet flags arrive in track['riders'] as
    [{"has_helmet": bool}, ...]. Emit helmet_absence when one or more riders
    lack a helmet; rider_count carries the number lacking one (the per-rider
    fine scales on this in the penalty engine).
    """
    if not _is_two_wheeler(track):
        return None
    riders = track.get("riders")
    if not isinstance(riders, list) or not riders:
        return None

    without = [r for r in riders if not r.get("has_helmet", False)]
    if not without:
        return None

    n_without = len(without)
    # Confidence: the more riders confirmed bare-headed, the more certain.
    confidence = _clamp01(0.75 + 0.1 * n_without)
    return {
        "code": "helmet_absence",
        "confidence": confidence,
        "evidence": {
            "rider_count": n_without,
            "total_riders": len(riders),
        },
    }


def detect_triple_riding(track: dict, scene: dict):
    """Triple-riding — triple_rider_and_helment.tex.

    Two-wheeler carrying 3+ occupants (track-level majority count).
    """
    if not _is_two_wheeler(track):
        return None
    n = _rider_count(track)
    if n < TRIPLE_RIDING_MIN_RIDERS:
        return None
    # One extra occupant past the legal two -> 0.8; more -> higher.
    confidence = _clamp01(0.7 + 0.1 * (n - TRIPLE_RIDING_MIN_RIDERS + 1))
    return {
        "code": "triple_riding",
        "confidence": confidence,
        "evidence": {"rider_count": n},
    }


def _in_windshield_zone(track: dict) -> bool:
    """True if any of the track's zone ids names a windshield/ANPR zone."""
    for z in track.get("zone_ids") or []:
        zl = str(z).lower()
        if any(h in zl for h in WINDSHIELD_ZONE_HINTS):
            return True
    return False


def detect_seatbelt(track: dict, scene: dict):
    """Seatbelt non-compliance — seatbelt.tex.

    For a car observed in a windshield/ANPR zone: driver_belted == False ->
    seatbelt_driver; an present-and-unbelted passenger -> seatbelt_passenger.
    Returns a list of fired dicts (a car can book both at once), or None.

    Fail-safe over fail-open: only an explicit False (belt confidently
    absent) fires; missing/None is treated as ambiguous -> no violation.
    """
    if not _is_car(track):
        return None
    if not _in_windshield_zone(track):
        return None

    fired = []
    if track.get("driver_belted") is False:
        fired.append({
            "code": "seatbelt_driver",
            "confidence": 0.85,
            "evidence": {"occupant": "driver"},
        })
    if track.get("passenger_present") and track.get("passenger_belted") is False:
        fired.append({
            "code": "seatbelt_passenger",
            "confidence": 0.8,
            "evidence": {"occupant": "passenger"},
        })
    return fired or None


def detect_phone_use(track: dict, scene: dict):
    """Handheld phone use — driver distraction (MV Act §184).

    Upstream classifier sets track['phone_in_use'] True when a handheld
    device is detected in use by the driver.
    """
    if track.get("phone_in_use") is not True:
        return None
    return {
        "code": "phone_use",
        "confidence": 0.85,
        "evidence": {"phone_in_use": True},
    }


# ====================================================================
#  Orchestration (Layer-5 routing)
# ====================================================================

# Maps an active-module name to its detector. Each detector may return None,
# a single fired dict, or a list of fired dicts (seatbelt).
_DETECTORS = {
    "wrong_side": detect_wrong_side,
    "stop_line": detect_stop_line,
    "red_light": detect_red_light,
    "illegal_parking": detect_illegal_parking,
    "helmet_absence": detect_helmet,
    "triple_riding": detect_triple_riding,
    "seatbelt": detect_seatbelt,
    "phone_use": detect_phone_use,
}

# Module "family" aliases accepted in active_modules. Both the family name
# (e.g. "seatbelt") and the concrete codes route to the right detector.
_FAMILY_ALIASES = {
    "seatbelt_driver": "seatbelt",
    "seatbelt_passenger": "seatbelt",
    "helmet": "helmet_absence",
}


def run_modules(track: dict, scene: dict, active_modules) -> list[dict]:
    """Run only the detectors whose module/family is in ``active_modules``.

    ``active_modules`` is the Layer-5 routing output: a list of module names
    or violation codes. A detector whose family is absent never fires.
    Returns the flattened list of all fired violation dicts.
    """
    active = set(active_modules or [])
    # Expand aliases so callers may pass either a family or a concrete code.
    resolved = set()
    for name in active:
        resolved.add(_FAMILY_ALIASES.get(name, name))

    fired: list[dict] = []
    for name, detector in _DETECTORS.items():
        if name not in resolved:
            continue
        result = detector(track, scene)
        if result is None:
            continue
        if isinstance(result, list):
            fired.extend(result)
        else:
            fired.append(result)
    return fired


# Codes whose fine scales with the number of (offending) riders.
_PER_RIDER_CODES = {"helmet_absence"}


def build_violation_record(
    track: dict,
    scene: dict,
    fired: dict,
    camera_id: str,
    event_id: str,
    timestamp_utc: str,
    plate: str = None,
    plate_conf: float = 0.0,
) -> dict:
    """Assemble a ViolationRecord-shaped dict from one fired violation.

    Mirrors app/schemas.ViolationRecord. ``fired`` is a single detector
    output dict ({code, confidence, evidence}). The emitted ``violation_type``
    is the fired ``code`` so it maps straight onto the penalty schedule.
    """
    code = fired["code"]
    evidence = fired.get("evidence") or {}

    # zone_id: prefer the no-park match (parking), else first track zone.
    zone_id = None
    npz = evidence.get("no_park_zone_ids")
    if npz:
        zone_id = npz[0]
    elif track.get("zone_ids"):
        zone_id = track["zone_ids"][0]

    return {
        "event_id": event_id,
        "camera_id": camera_id,
        "timestamp_utc": timestamp_utc,
        "violation_type": code,
        "vehicle_class": track.get("class_name", "two_wheeler"),
        "track_id": int(track.get("track_id", -1)),
        "confidence": round(float(fired.get("confidence", 0.0)), 4),
        "plate_string": plate,
        "plate_confidence": float(plate_conf),
        "speed_kmph": float(track.get("speed_kmh", -1.0)),
        "zone_id": zone_id,
    }
