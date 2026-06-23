"""API schemas.

Two families:
  * Report schemas: TrackedObject (perception Layer 4), ViolationRecord and
    EvidencePackage (evidence packaging §). proto3 in the report; Pydantic here.
  * Console schemas: the exact shapes the NyayaChakshu console renders, so the
    static frontend can be driven by this backend with zero markup changes.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ----------------------- report: perception Layer 4 -----------------------
class TrackState(str, Enum):
    TENTATIVE = "TENT"
    CONFIRMED = "CONF"
    LOST = "LOST"


class StopLineRel(str, Enum):
    BEFORE = "BEFORE"
    ON = "ON"
    AFTER = "AFTER"


class TrackedObject(BaseModel):
    track_id: int
    class_id: int
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    track_state: TrackState = TrackState.CONFIRMED
    velocity_px: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    speed_kmh: float = -1.0
    heading_deg: float = 0.0
    is_stationary: bool = False
    stationary_ms: int = 0
    zone_ids: list[str] = Field(default_factory=list)
    stop_line_rel: Optional[StopLineRel] = None


# ----------------------- report: evidence packaging -----------------------
class ViolationType(str, Enum):
    helmet_absence = "helmet_absence"
    triple_riding = "triple_riding"
    seatbelt_driver = "seatbelt_driver"
    seatbelt_passenger = "seatbelt_passenger"
    phone_use = "phone_use"
    wrong_side = "wrong_side"
    red_light = "red_light"
    stop_line = "stop_line"
    illegal_parking = "illegal_parking"


class ViolationRecord(BaseModel):
    event_id: str
    camera_id: str
    timestamp_utc: str
    violation_type: ViolationType
    vehicle_class: str = "two_wheeler"
    track_id: int
    confidence: float
    plate_string: Optional[str] = None
    plate_confidence: float = 0.0
    speed_kmph: float = -1.0
    zone_id: Optional[str] = None


class EvidencePackage(BaseModel):
    primary_frame_uri: str
    temporal_strip_uri: Optional[str] = None
    raw_frames_uri: Optional[str] = None
    metadata_json_uri: Optional[str] = None
    sha256_primary_frame: str


# ----------------------------- console shapes -----------------------------
class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float
    c: str
    l: str


class Offence(BaseModel):
    nm: str
    c: str
    sc: str
    sec: str
    fine: int


class Case(BaseModel):
    id: str
    n: str
    img: str
    loc: str
    cam: str
    plate: str
    pconf: float
    frame: str
    tracks: str
    inf: str
    hash: str
    sig: str
    gps: str
    boxes: list[Box] = Field(default_factory=list)
    viols: list[Offence] = Field(default_factory=list)


# ----------------------------- tool requests ------------------------------
class AlprRequest(BaseModel):
    candidates: list[tuple[str, float]] = Field(
        ..., description="(raw_ocr_string, ocr_confidence) per frame")


class ChallanRequest(BaseModel):
    case_id: str


class EchallanDispatch(BaseModel):
    challan_no: str
    plate: str
    sent_sms: bool
    sent_email: bool
    evidence_url: str
    status: str
