"""NyayaChakshu Core Perception Pipeline (stdlib-only, runnable simulation).

This module implements the downstream logic of the five-layer perception
pipeline described in ``perception.tex`` ("Proposed Solution: The
NyayaChakshu Core Perception Pipeline"). The deep-learning stages (Layer 1
YOLOv11 object detection) are *simulated* deterministically -- the caller
supplies detections directly -- but every downstream stage is implemented
for real, in pure Python with no third-party dependencies:

  * Layer 2 -- Multi-Object Tracking. A greedy IoU-based tracker (a
    light stand-in for BoT-SORT's IoU association stage) that assigns
    persistent ``track_id`` values across ``process_frame`` calls and
    drives the TENT -> CONF -> LOST state machine.
  * Layer 3 -- Kinematic and Spatial Estimation. Pixel velocity from
    bbox-centroid deltas across frames, world speed via a pixels->metres
    scale, heading in degrees, ``is_stationary`` below ``v_stat``, and
    ``stationary_ms`` accumulation.
  * Layer 4 -- Tracked Object Metadata assembly into a plain dict whose
    keys match ``app.schemas.TrackedObject``.
  * Layer 5 -- Routing Engine. A deterministic rule evaluator that maps a
    CONFIRMED tracked object to the violation modules it should activate.

Public API (stable -- the orchestrator depends on these names/signatures):

    class PerceptionPipeline:
        def __init__(self, fps: float = 25.0, px_per_metre: float = 8.0,
                     v_stat_kmh: float = 2.0) -> None
        def process_frame(self, frame_index: int,
                          detections: list[dict]) -> list[dict]
        def route(self, tracked: dict) -> list[str]

``detections`` items are dicts shaped as::

    {"class_id": int, "class_name": str, "bbox": [x, y, w, h],
     "score": float, "zone_ids": [str, ...]}

``process_frame`` returns a list of TrackedObject-shaped dicts with keys::

    track_id, class_id, bbox, track_state, velocity_px, speed_kmh,
    heading_deg, is_stationary, stationary_ms, zone_ids, stop_line_rel

The values for ``track_state`` ("TENT"|"CONF"|"LOST") and ``stop_line_rel``
("BEFORE"|"ON"|"AFTER" or None) match the string enums in app.schemas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- tracker / state machine tuning -----------------------------------------
IOU_MATCH_THRESHOLD = 0.3      # greedy match accepted when IoU > this
CONFIRM_AGE = 3                # frames seen before TENT -> CONF
MAX_MISSED_FRAMES = 5          # consecutive misses before CONF/TENT -> LOST

# --- routing thresholds (from Algorithm "Routing Engine") --------------------
SEATBELT_MIN_SPEED_KMH = 5.0   # car seatbelt/phone routing speed gate
WRONG_SIDE_MIN_AGE = 10        # min track age before wrong-side dispatch
WRONG_SIDE_MIN_SPEED_KMH = 10.0
ILLEGAL_PARK_MIN_MS = 60_000   # T_park: stationary duration for parking

# Class-name aliases -> canonical vehicle/category labels. The simulated
# Layer 1 may emit any of these; routing keys off the canonical form.
_TWO_WHEELER_NAMES = {"motorcycle", "motorbike", "two_wheeler", "scooter", "bike"}
_CAR_NAMES = {"car", "sedan", "hatchback", "suv"}
_TRUCK_BUS_NAMES = {"truck", "bus", "lorry"}
_AUTO_NAMES = {"auto_rickshaw", "auto", "rickshaw", "three_wheeler"}

# Zone-id substrings that carry semantic meaning to the routing engine.
_NO_PARK_TOKENS = ("no_park", "noparking", "no_parking")
_WRONG_SIDE_TOKENS = ("wrong_side", "wrongside", "one_way", "oneway")
_WINDSHIELD_TOKENS = ("windshield", "windscreen", "front_window")


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes (top-left origin)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _centroid(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _canonical_class(class_name: str | None) -> str:
    name = (class_name or "").strip().lower()
    if name in _TWO_WHEELER_NAMES:
        return "two_wheeler"
    if name in _CAR_NAMES:
        return "car"
    if name in _TRUCK_BUS_NAMES:
        return "heavy_vehicle"
    if name in _AUTO_NAMES:
        return "auto_rickshaw"
    return name or "unknown"


@dataclass
class _Track:
    """Internal per-object state. Not part of the public API."""
    track_id: int
    class_id: int
    class_name: str
    bbox: list[float]
    zone_ids: list[str]
    stop_line_rel: str | None = None

    # state machine
    state: str = "TENT"           # TENT | CONF | LOST
    hits: int = 1                 # frames matched (track age proxy)
    misses: int = 0               # consecutive frames unmatched
    last_frame: int = 0

    # kinematics
    centroid: tuple[float, float] = (0.0, 0.0)
    velocity_px: list[float] = field(default_factory=lambda: [0.0, 0.0])
    speed_kmh: float = -1.0
    heading_deg: float = 0.0
    is_stationary: bool = False
    stationary_ms: int = 0


class PerceptionPipeline:
    """Stateful perception pipeline driving Layers 2-5 across frames."""

    def __init__(self, fps: float = 25.0, px_per_metre: float = 8.0,
                 v_stat_kmh: float = 2.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if px_per_metre <= 0:
            raise ValueError("px_per_metre must be positive")
        self.fps = float(fps)
        self.px_per_metre = float(px_per_metre)
        self.v_stat_kmh = float(v_stat_kmh)
        self._frame_ms = 1000.0 / self.fps

        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ Layer 2/3
    def process_frame(self, frame_index: int,
                      detections: list[dict]) -> list[dict]:
        """Run Layers 2-4 for one frame; return TrackedObject-shaped dicts.

        Greedily matches detections to existing tracks by IoU, spawns new
        tracks for unmatched detections, advances the state machine, and
        recomputes kinematics. Returns one dict per *live* (non-LOST) track
        that was matched or created this frame, plus tracks that just
        transitioned to LOST (so the orchestrator can observe the change).
        """
        tracks = list(self._tracks.values())

        # --- greedy IoU association (Layer 2) ---------------------------
        pairs: list[tuple[float, int, int]] = []  # (iou, det_idx, track_idx)
        for di, det in enumerate(detections):
            dbox = list(det["bbox"])
            for ti, trk in enumerate(tracks):
                score = iou(dbox, trk.bbox)
                if score > IOU_MATCH_THRESHOLD:
                    pairs.append((score, di, ti))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        det_to_track: dict[int, _Track] = {}
        for score, di, ti in pairs:
            if di in matched_dets or ti in matched_tracks:
                continue
            matched_dets.add(di)
            matched_tracks.add(ti)
            det_to_track[di] = tracks[ti]

        emitted_ids: set[int] = set()
        results: list[dict] = []

        # --- update matched tracks --------------------------------------
        for di, det in enumerate(detections):
            if di not in matched_dets:
                continue
            trk = det_to_track[di]
            self._update_track(trk, det, frame_index)
            emitted_ids.add(trk.track_id)
            results.append(self._to_tracked_object(trk))

        # --- spawn new tracks for unmatched detections ------------------
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            trk = self._spawn_track(det, frame_index)
            emitted_ids.add(trk.track_id)
            results.append(self._to_tracked_object(trk))

        # --- age unmatched existing tracks ------------------------------
        for trk in tracks:
            if trk.track_id in emitted_ids:
                continue
            trk.misses += 1
            # stationary objects keep accumulating dwell time while occluded
            if trk.is_stationary and trk.state != "LOST":
                trk.stationary_ms += int(round(self._frame_ms))
            if trk.misses > MAX_MISSED_FRAMES:
                if trk.state != "LOST":
                    trk.state = "LOST"
                    results.append(self._to_tracked_object(trk))

        return results

    # ----------------------------------------------------------------- helpers
    def _spawn_track(self, det: dict, frame_index: int) -> _Track:
        trk = _Track(
            track_id=self._next_id,
            class_id=int(det.get("class_id", -1)),
            class_name=str(det.get("class_name", "")),
            bbox=list(det["bbox"]),
            zone_ids=list(det.get("zone_ids", [])),
            stop_line_rel=det.get("stop_line_rel"),
        )
        self._next_id += 1
        trk.centroid = _centroid(trk.bbox)
        trk.last_frame = frame_index
        # speed unknown until we have a second observation
        trk.speed_kmh = -1.0
        trk.is_stationary = False
        self._tracks[trk.track_id] = trk
        return trk

    def _update_track(self, trk: _Track, det: dict, frame_index: int) -> None:
        new_bbox = list(det["bbox"])
        new_centroid = _centroid(new_bbox)

        dframes = frame_index - trk.last_frame
        if dframes <= 0:
            dframes = 1
        dt_s = dframes / self.fps

        dx = new_centroid[0] - trk.centroid[0]
        dy = new_centroid[1] - trk.centroid[1]

        # per-frame pixel velocity
        vx = dx / dframes
        vy = dy / dframes
        trk.velocity_px = [vx, vy]

        # world speed: pixel displacement -> metres -> km/h
        dist_px = math.hypot(dx, dy)
        dist_m = dist_px / self.px_per_metre
        speed_mps = dist_m / dt_s if dt_s > 0 else 0.0
        trk.speed_kmh = speed_mps * 3.6

        # heading in degrees (image coords: +x right, +y down).
        # Convert to a conventional compass-free heading measured CCW from +x,
        # with +y treated as "up" so motion to the right is 0 deg.
        if dist_px > 1e-9:
            trk.heading_deg = math.degrees(math.atan2(-dy, dx)) % 360.0

        # stationary classification + dwell accumulation
        was_stationary = trk.is_stationary
        trk.is_stationary = trk.speed_kmh < self.v_stat_kmh
        if trk.is_stationary:
            trk.stationary_ms += int(round(dt_s * 1000.0))
        else:
            trk.stationary_ms = 0
        # (was_stationary retained for clarity; reset handled above)
        del was_stationary

        # commit detection-derived fields
        trk.bbox = new_bbox
        trk.centroid = new_centroid
        trk.zone_ids = list(det.get("zone_ids", trk.zone_ids))
        if "stop_line_rel" in det:
            trk.stop_line_rel = det.get("stop_line_rel")
        if "class_id" in det:
            trk.class_id = int(det["class_id"])
        if "class_name" in det:
            trk.class_name = str(det["class_name"])

        # state machine
        trk.hits += 1
        trk.misses = 0
        trk.last_frame = frame_index
        if trk.state == "LOST":
            # re-acquired: treat as confirmed again
            trk.state = "CONF"
        elif trk.state == "TENT" and trk.hits >= CONFIRM_AGE:
            trk.state = "CONF"

    def _to_tracked_object(self, trk: _Track) -> dict:
        """Layer 4: assemble the TrackedObject metadata dict."""
        return {
            "track_id": trk.track_id,
            "class_id": trk.class_id,
            "bbox": [float(v) for v in trk.bbox],
            "track_state": trk.state,
            "velocity_px": [float(trk.velocity_px[0]), float(trk.velocity_px[1])],
            "speed_kmh": float(trk.speed_kmh),
            "heading_deg": float(trk.heading_deg),
            "is_stationary": bool(trk.is_stationary),
            "stationary_ms": int(trk.stationary_ms),
            "zone_ids": list(trk.zone_ids),
            "stop_line_rel": trk.stop_line_rel,
            # carried for the routing engine (not part of schemas.TrackedObject)
            "class_name": _canonical_class(trk.class_name),
            "track_age": trk.hits,
        }

    # --------------------------------------------------------------- Layer 5
    def route(self, tracked: dict) -> list[str]:
        """Routing Engine: violation modules to activate for one track.

        Mirrors Algorithm "Routing Engine - per-frame dispatch". Routing only
        fires for CONFIRMED tracks; any other state yields no modules.
        """
        if tracked.get("track_state") != "CONF":
            return []

        modules: list[str] = []
        cls = _canonical_class(tracked.get("class_name"))
        zones = [str(z).lower() for z in tracked.get("zone_ids", [])]
        speed = float(tracked.get("speed_kmh", -1.0))
        age = int(tracked.get("track_age", 0))

        def in_any(tokens: tuple[str, ...]) -> bool:
            return any(any(tok in z for tok in tokens) for z in zones)

        # Two-wheeler: helmet (always) + triple riding.
        if cls == "two_wheeler":
            modules.append("helmet")
            modules.append("triple_riding")

        # Car: seatbelt + phone use when moving and viewed through windshield.
        if cls == "car" and speed > SEATBELT_MIN_SPEED_KMH:
            modules.append("seatbelt_driver")
            if in_any(_WINDSHIELD_TOKENS):
                modules.append("phone_use")

        # Wrong-side: aged track moving the wrong way in a wrong-side zone.
        if (age >= WRONG_SIDE_MIN_AGE
                and speed > WRONG_SIDE_MIN_SPEED_KMH
                and in_any(_WRONG_SIDE_TOKENS)):
            modules.append("wrong_side")

        # Stop line / red light: track crossed past the stop line.
        if tracked.get("stop_line_rel") == "AFTER":
            modules.append("stop_line")
            if any("red_light" in z or "red" in z or "signal_red" in z
                   for z in zones):
                modules.append("red_light")

        # Illegal parking: stationary long enough inside a no-park zone.
        if (tracked.get("is_stationary")
                and in_any(_NO_PARK_TOKENS)
                and int(tracked.get("stationary_ms", 0)) > ILLEGAL_PARK_MIN_MS):
            modules.append("illegal_parking")

        # de-duplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for m in modules:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return ordered
