"""Real image-based violation detection (local YOLO).

This runs an actual YOLOv8 model (ultralytics, COCO-pretrained) on an uploaded
image — genuine inference, not simulation. It returns the real detections and
derives the violations that COCO classes can honestly support.

HONESTY NOTE (read this):
  COCO has classes for person, bicycle, car, motorcycle, bus, truck, traffic
  light — so we can REALLY detect vehicles/persons and DERIVE:
    * triple_riding   — >=3 persons riding one two-wheeler (rider association)
    * two-wheeler / car scene composition, vehicle + person counts
    * presence of a traffic light in frame (state classification is a separate step)
  COCO has NO helmet or seatbelt class. Detecting helmet-absence / seatbelt
  non-compliance reliably needs the fine-tuned multi-task head described in the
  report (perception.tex / seatbelt.tex), which requires labelled training data
  we do not bundle. So this module does NOT claim those — it reports them as
  "requires specialised model" rather than faking a result.

The model file (yolov8n.pt, ~6 MB) is auto-downloaded by ultralytics on first
use and cached. CPU inference is fine for single images.
"""
from __future__ import annotations

import base64
import io
from functools import lru_cache

# COCO class ids we care about.
PERSON = 0
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7
TRAFFIC_LIGHT = 9
VEHICLE_CLASSES = {BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK}
TWO_WHEELER = {MOTORCYCLE, BICYCLE}

_BOX_COLOUR = {"red": (255, 77, 109), "amber": (246, 166, 9),
               "cyan": (47, 211, 195), "green": (53, 209, 153)}


class DetectorUnavailable(RuntimeError):
    """Raised when ultralytics/torch isn't installed (e.g. on the 3.14 deploy)."""


_MODEL_NAME = "yolov8m.pt"  # balance of accuracy vs CPU speed; auto-downloaded


@lru_cache(maxsize=1)
def _model():
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover - env dependent
        raise DetectorUnavailable(
            "ultralytics/torch not installed. Run the detection service in the "
            "Python 3.11 venv: pip install -r requirements-detect.txt"
        ) from e
    return YOLO(_MODEL_NAME)


def _overlap_ratio(p, tw) -> float:
    """Fraction of the person box's horizontal span that overlaps the vehicle."""
    px1, px2 = p[0], p[0] + p[2]
    tx1, tx2 = tw[0], tw[0] + tw[2]
    inter = max(0.0, min(px2, tx2) - max(px1, tx1))
    return inter / max(1e-6, p[2])


def _rider_association(persons, two_wheelers):
    """Greedily assign each detected person to one two-wheeler they're riding.

    A person rides a two-wheeler when (a) their horizontal span substantially
    overlaps the vehicle's and (b) they sit vertically on/above it (their lower
    body is within the vehicle's vertical band). Each person is assigned to at
    most one vehicle — the best-overlapping one — so riders are never double
    counted across adjacent bikes. Returns a rider count per two-wheeler.
    """
    counts = [0] * len(two_wheelers)
    for p in persons:
        px, py, pw, ph = p["xywh"]
        p_bottom = py + ph
        best, best_score = -1, 0.0
        for i, tw in enumerate(two_wheelers):
            tx, ty, tww, twh = tw["xywh"]
            horiz = _overlap_ratio([px, py, pw, ph], [tx, ty, tww, twh])
            # rider's feet/seat fall within the bike's vertical band (+margin)
            vertical_ok = (p_bottom >= ty - 0.15 * twh) and (py <= ty + twh)
            if horiz > 0.35 and vertical_ok and horiz > best_score:
                best, best_score = i, horiz
        if best >= 0:
            counts[best] += 1
    return counts


def detect(image_bytes: bytes, conf: float = 0.30) -> dict:
    """Run YOLO on an image and derive supportable violations.

    Returns: { image_w, image_h, objects:[...], violations:[...],
               supported, unsupported, annotated_image_b64 }
    Coordinates in `objects` are percentages (0-100) of width/height so they
    render directly in the console's SVG overlay convention.
    """
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = img.size
    res = _model().predict(img, conf=conf, verbose=False)[0]

    objects = []
    for b in res.boxes:
        cid = int(b.cls[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        objects.append({
            "class_id": cid,
            "class_name": res.names[cid],
            "confidence": round(float(b.conf[0]), 3),
            # percentage xywh for the overlay
            "xywh": [round(x1 / W * 100, 2), round(y1 / H * 100, 2),
                     round((x2 - x1) / W * 100, 2), round((y2 - y1) / H * 100, 2)],
            "xyxy_px": [round(x1), round(y1), round(x2), round(y2)],
        })

    persons = [o for o in objects if o["class_id"] == PERSON]
    two_wheelers = [o for o in objects if o["class_id"] in TWO_WHEELER]
    cars = [o for o in objects if o["class_id"] in {CAR, BUS, TRUCK}]
    lights = [o for o in objects if o["class_id"] == TRAFFIC_LIGHT]

    violations = []
    # REAL derivation: triple-riding from rider association.
    rider_counts = _rider_association(persons, two_wheelers)
    for tw, n in zip(two_wheelers, rider_counts):
        if n >= 3:
            violations.append({
                "code": "triple_riding",
                "confidence": round(min(0.99, tw["confidence"] * 0.9 + 0.05 * n), 3),
                "evidence": {"riders_detected": n, "vehicle": tw["class_name"],
                             "vehicle_box": tw["xywh"]},
                "basis": f"{n} persons associated with one two-wheeler",
            })

    supported = {
        "objects_detected": len(objects),
        "persons": len(persons),
        "two_wheelers": len(two_wheelers),
        "cars_buses_trucks": len(cars),
        "traffic_lights": len(lights),
        "max_riders_on_a_two_wheeler": max(rider_counts) if rider_counts else 0,
    }
    # Honest about what this model cannot decide.
    unsupported = []
    if two_wheelers:
        unsupported.append({
            "violation": "helmet_absence",
            "reason": "COCO has no helmet class; needs the fine-tuned helmet head "
                      "(perception.tex). Riders are detected, helmet state is not."})
    if cars:
        unsupported.append({
            "violation": "seatbelt_driver / phone_use",
            "reason": "Needs windshield ROI + occupant attribute head (seatbelt.tex); "
                      "not a COCO class."})

    annotated = _annotate(img.copy(), objects, two_wheelers, rider_counts)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image_w": W, "image_h": H,
        "objects": objects,
        "violations": violations,
        "supported": supported,
        "unsupported": unsupported,
        "annotated_image_b64": f"data:image/jpeg;base64,{b64}",
        "model": f"{_MODEL_NAME} (COCO)",
    }


def _annotate(img, objects, two_wheelers, rider_counts):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    for o in objects:
        x, y, w, h = o["xyxy_px"]
        cid = o["class_id"]
        col = _BOX_COLOUR["red"] if cid == PERSON else (
            _BOX_COLOUR["cyan"] if cid in VEHICLE_CLASSES else _BOX_COLOUR["amber"])
        d.rectangle([x, y, w, h], outline=col, width=3)
        label = f"{o['class_name']} {o['confidence']:.2f}"
        d.rectangle([x, y - 16, x + 8 * len(label), y], fill=col + (220,))
        d.text((x + 2, y - 14), label, fill=(10, 11, 30))
    # Flag triple-riding vehicles.
    for tw, n in zip(two_wheelers, rider_counts):
        if n >= 3:
            x, y, ww, hh = tw["xyxy_px"]
            d.rectangle([x - 2, y - 2, ww + 2, hh + 2],
                        outline=_BOX_COLOUR["red"], width=5)
            tag = f"TRIPLE-RIDING · {n} riders"
            d.rectangle([x, hh, x + 9 * len(tag), hh + 18], fill=_BOX_COLOUR["red"] + (235,))
            d.text((x + 3, hh + 3), tag, fill=(255, 255, 255))
    return img
