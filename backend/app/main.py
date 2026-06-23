"""NyayaChakshu enforcement backend — FastAPI application.

Endpoints mirror exactly what the command-centre console needs, plus expose
the report's tools as callable services:

  GET  /api/health
  GET  /api/console/bootstrap     -> everything the static console renders
  GET  /api/dashboard             -> overview tiles, gauges, junctions, ticker
  GET  /api/cases                 -> detection queue
  GET  /api/cases/{event_id}      -> single case detail
  GET  /api/analytics             -> per-class precision/recall/F1 table
  POST /api/alpr/resolve          -> multi-frame Vahan/HSRP plate resolution
  POST /api/challan/{event_id}    -> build a challan from booked offences
  POST /api/echallan/dispatch/{event_id} -> simulate eChallan/Vahan dispatch
  GET  /api/ledger                -> the evidence hash chain
  GET  /api/ledger/verify         -> tamper-evidence integrity proof
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, proof, seed
from .pipelines import run_scene
from .schemas import AlprRequest
from .store import store

app = FastAPI(
    title="NyayaChakshu Enforcement API",
    version=__version__,
    description="Backend for the NyayaChakshu traffic-violation command centre.",
)

# The console is a static site (separate origin); allow it to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    integ = store.ledger.verify()
    return {
        "status": "ok",
        "version": __version__,
        "cameras_online": "12/12",
        "ledger_intact": integ["intact"],
        "ledger_records": integ["records_checked"],
    }


@app.get("/api/console/bootstrap")
def console_bootstrap() -> dict:
    """One call returns every dataset the static console hardcodes today."""
    return {
        "cases": seed.CASES,
        "metrics": seed.METRICS,
        "class_today": seed.CLASS_TODAY,
        "junctions": seed.JUNCTIONS,
        "gauges": seed.GAUGES,
        "review_donut": seed.REVIEW_DONUT,
    }


@app.get("/api/dashboard")
def dashboard() -> dict:
    today_total = sum(c[1] for c in seed.CLASS_TODAY)
    return {
        "date": "2026-06-23",
        "gauges": seed.GAUGES,
        "class_today": seed.CLASS_TODAY,
        "junctions": seed.JUNCTIONS,
        "review_donut": seed.REVIEW_DONUT,
        "totals": {
            "events_today": today_total,
            "auto_cleared": 612,
            "pending_review": 7,
            "sent_to_senior": 142,
            "dismissed_fp": 35,
        },
    }


@app.get("/api/cases")
def list_cases() -> dict:
    return {"count": len(store.cases), "cases": store.list_cases()}


@app.get("/api/cases/{event_id}")
def get_case(event_id: str) -> dict:
    c = store.get_case(event_id)
    if c is None:
        raise HTTPException(404, f"unknown event {event_id}")
    return c


@app.get("/api/analytics")
def analytics() -> dict:
    rows = [
        {"violation": m[0], "precision": m[1], "recall": m[2],
         "f1": m[3], "false_positive_rate": m[4]}
        for m in seed.METRICS
    ]
    mean_f1 = round(sum(m[3] for m in seed.METRICS) / len(seed.METRICS), 3)
    return {"per_class": rows, "mean_f1": mean_f1}


@app.post("/api/alpr/resolve")
def alpr_resolve(req: AlprRequest) -> dict:
    if not req.candidates:
        raise HTTPException(400, "no candidates supplied")
    return store.run_alpr(req.candidates)


@app.post("/api/challan/{event_id}")
def build_challan(event_id: str) -> dict:
    challan = store.build_challan(event_id)
    if challan is None:
        raise HTTPException(404, f"unknown event {event_id}")
    return challan


@app.post("/api/echallan/dispatch/{event_id}")
def echallan_dispatch(event_id: str) -> dict:
    """Simulate transmission to the MoRTH eChallan / Vahan system.

    Mirrors the report's Integration Interfaces: registration number, MV Act
    code, datetime + GPS + camera, and a signed evidence-bundle URL. SMS to the
    registered owner is triggered by eChallan via Vahan.
    """
    challan = store.build_challan(event_id)
    if challan is None:
        raise HTTPException(404, f"unknown event {event_id}")
    dispatched = bool(challan["plate_confidence"])
    return {
        "challan_no": challan["challan_no"],
        "registration_vahan": challan["registration"],
        "violation_codes": [o["code"] for o in challan["offences"]],
        "sections": [o["section"] for o in challan["offences"]],
        "total_payable_inr": challan["total_payable_inr"],
        "evidence_url": f"https://evidence.tspolice.gov.in/bundle/{challan['challan_no']}",
        "evidence_frame_sha256": challan["evidence_frame_sha256"],
        "sms_to_owner": dispatched,
        "status": "DISPATCHED" if dispatched else "HELD_PLATE_UNVERIFIED",
    }


@app.get("/api/ledger")
def ledger(camera_id: str | None = None) -> dict:
    entries = store.ledger.entries(camera_id)
    return {
        "count": len(entries),
        "entries": [
            {"seq": e.seq, "camera_id": e.camera_id, "event_id": e.event_id,
             "recorded_at": e.recorded_at, "raw_frame_sha256": e.raw_frame_sha256,
             "prev_hash": e.prev_hash, "record_hash": e.record_hash,
             "violation_type": e.payload.get("violation_type")}
            for e in entries
        ],
    }


@app.get("/api/ledger/verify")
def ledger_verify(camera_id: str | None = None) -> dict:
    return store.ledger.verify(camera_id)


@app.post("/api/detect")
async def detect_violations(file: UploadFile = File(...)) -> dict:
    """Run REAL YOLO inference on an uploaded image and derive violations.

    Genuine object detection (ultralytics YOLOv8, COCO). Returns the detected
    objects, the violations the model can honestly support (e.g. triple-riding
    via rider association), an annotated image, and — when a violation fires —
    a challan preview from the penalty engine. Available only where the
    detection stack is installed (the local Python 3.11 service).
    """
    from . import detector
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        result = detector.detect(data)
    except detector.DetectorUnavailable as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(422, f"could not process image: {e}")

    # If a real violation fired, build a challan preview off the penalty engine.
    from . import penalties
    offences = []
    for v in result["violations"]:
        riders = v.get("evidence", {}).get("riders_detected", 1)
        offences.append({
            "offence": v["code"], "section": penalties.section_for(v["code"]),
            "code": v["code"], "ai_confidence": v["confidence"],
            "fine": penalties.fine_for(v["code"], riders)})
    result["challan_preview"] = {
        "offences": offences,
        "total_payable_inr": penalties.challan_total(offences),
    } if offences else None
    return result


@app.get("/api/detect/status")
def detect_status() -> dict:
    """Report whether the real-detection model is available in this runtime."""
    from . import detector
    try:
        detector._model()
        return {"available": True, "model": f"{detector._MODEL_NAME} (COCO)"}
    except detector.DetectorUnavailable as e:
        return {"available": False, "reason": str(e)}


@app.get("/api/proof")
def run_proof() -> dict:
    """Live self-verification — runs real checks and returns checkable evidence.

    Each check shows expected vs actual; the hash-chain check prints the exact
    bytes so a sceptic can recompute the SHA-256 independently.
    """
    return proof.run_all()


@app.get("/api/pipeline/run")
def pipeline_run() -> dict:
    """Run the full perception → routing → violation stack on a synthetic scene.

    Demonstrates the end-to-end pipeline with no model weights: a two-wheeler
    carrying three un-helmeted riders moves across an NH-65 service-road camera.
    The stack tracks it, routes it to the helmet + triple-riding modules, and
    emits the corresponding ViolationRecords.
    """
    # Two-wheeler moving right across three frames (bbox drifts in x).
    frames = []
    for k in range(4):
        frames.append([{
            "class_id": 3, "class_name": "two_wheeler",
            "bbox": [10 + k * 9, 30, 18, 44], "score": 0.93,
            "zone_ids": ["NH65_NB"],
            "riders": [{"has_helmet": False}, {"has_helmet": False},
                       {"has_helmet": False}],
        }])
    scene = {
        "lane_dir_deg": 0.0, "stop_line_y": None, "signal_state": None,
        "no_park_zone_ids": [], "plate": "AP 37 BK 6798", "plate_conf": 0.94,
    }
    return run_scene(frames, scene, camera_id="CAM-07A")


# --- Local convenience: serve the static console + pages from the same origin.
# A single `uvicorn` command then gives you the full app (console at /,
# /detect.html, /proof.html) AND the API at one URL. On Vercel the static files
# are served by the platform (rewrites) and this directory isn't bundled, so the
# mount is simply skipped there.
import os as _os  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402

_FRONTEND = _os.path.join(_os.path.dirname(__file__), "..", "..", "frontend")
if _os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
