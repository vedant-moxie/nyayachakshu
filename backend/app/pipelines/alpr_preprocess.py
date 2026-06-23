"""ALPR Preprocessing Subsystem — image-conditioning front-half (stages P1-P8).

This module implements the *preprocessing* half of the NyayaChakshu ALPR pipeline
described in ``license_plate.tex`` ("ALPR Preprocessing Subsystem"). It is the front
end that conditions a raw plate crop and decides — at the P8 quality gate — whether a
frame is good enough to forward to the consensus/validation back-half in
``app.alpr`` (which does the multi-frame majority vote + Vahan-grammar validation).

Because the runtime here has STDLIB ONLY (no cv2/numpy/torch), the actual pixel
operations are *simulated deterministically*. A plate crop is represented as a small
metadata bundle (:class:`PlateCrop`) and each stage is a pure function that transforms
that metadata the way the real image op would move the underlying measurement:

    P1  Coarse Plate Localisation        — establish the working crop / pad border.
    P2  Four-Corner Keypoint Detection   — locate the quad corners (skew becomes known).
    P3  Perspective Warp + Skew Correction — drive skew_deg -> ~0, small sharpness gain.
    P4  Condition Classification         — NORMAL / LOW_LIGHT / MOTION_BLUR / GLARE / SOILED.
    P5  Condition-Specific Enhancement   — condition-specific quality bump.
    P6  Universal Normalisation          — fixed contrast/greyscale normalise.
    P7  Resolution Check + Conditional SR — 4x upscale + SR gain when width < MIN_SR_WIDTH.
    P8  Quality Gate                     — composite pass/fail from resolution+sharpness+skew.

Public API (stable):

    PlateCondition          — Enum: NORMAL, LOW_LIGHT, MOTION_BLUR, GLARE, SOILED
    PlateCrop               — dataclass holding the per-crop metadata
    StageResult             — dataclass {stage, ok, notes, crop}
    ALPRPreprocessor        — .run(crop) -> list[StageResult] (P1..P8 in order)
                              .passed_quality_gate(results) -> bool
    resolve_plate(crops)    — run preprocessing on each crop, keep the P8-passing
                              ones, feed their (ocr_text, ocr_conf) into the EXISTING
                              app.alpr.resolve(...), return a consensus dict.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum

from app import alpr

# --- Tunables (mirroring the report's thresholds where the simulation allows) -------

#: Below this original width (px) P7 applies a 4x super-resolution upscale.
#: Mirrors the console's "4x super-resolved" tag shown on low-res evidence frames.
MIN_SR_WIDTH = 120

#: P8 composite-score acceptance threshold (Q > GATE_PASS proceeds to OCR).
GATE_PASS = 0.55

#: P8 hard sharpness floor. Even with good resolution/skew, a crop whose final
#: sharpness stays below this is unreadable and is rejected outright — this is
#: what keeps a 4x-upscaled-but-still-blurry frame from sneaking past the gate.
MIN_SHARPNESS = 0.50

#: Composite weights for the P8 quality score: resolution, sharpness, skew.
_W_RES, _W_SHARP, _W_SKEW = 0.35, 0.45, 0.20

#: Skew magnitude (deg) beyond which the skew sub-score is fully penalised.
_SKEW_FULL_PENALTY_DEG = 15.0


class PlateCondition(Enum):
    """Coarse capture-condition classes assigned by Stage P4."""

    NORMAL = "NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    MOTION_BLUR = "MOTION_BLUR"
    GLARE = "GLARE"
    SOILED = "SOILED"


@dataclass
class PlateCrop:
    """Metadata bundle standing in for a real plate-crop image.

    Scores are normalised to ``[0, 1]`` unless noted:
      * ``blur_score``  — sharpness proxy (1.0 = razor sharp, 0.0 = unreadable).
      * ``brightness``  — mean luminance (0.0 = black, 1.0 = blown highlights).
      * ``skew_deg``    — in-plane rotation in degrees (signed).
    """

    width: int = 400
    height: int = 100
    skew_deg: float = 0.0
    blur_score: float = 0.9
    brightness: float = 0.5
    condition: PlateCondition | None = None
    ocr_text: str = ""
    ocr_conf: float = 0.0


@dataclass
class StageResult:
    """Outcome of a single pipeline stage."""

    stage: str
    ok: bool
    notes: list[str] = field(default_factory=list)
    crop: PlateCrop = None  # type: ignore[assignment]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ALPRPreprocessor:
    """Runs the P1-P8 conditioning stages over a single plate crop."""

    def __init__(self, min_sr_width: int = MIN_SR_WIDTH, gate_pass: float = GATE_PASS,
                 min_sharpness: float = MIN_SHARPNESS):
        self.min_sr_width = min_sr_width
        self.gate_pass = gate_pass
        self.min_sharpness = min_sharpness

    # -- individual stages ----------------------------------------------------------

    def _p1_localise(self, crop: PlateCrop) -> StageResult:
        """P1 Coarse Plate Localisation: establish the working crop + 15% border pad."""
        notes = [f"localised {crop.width}x{crop.height}px"]
        ok = crop.width > 0 and crop.height > 0
        if not ok:
            notes.append("empty_crop")
        return StageResult("P1", ok, notes, replace(crop))

    def _p2_corners(self, crop: PlateCrop) -> StageResult:
        """P2 Four-Corner Keypoint Detection: corner quad localises the in-plane skew."""
        notes = [f"four_corners_detected skew={crop.skew_deg:+.1f}deg"]
        return StageResult("P2", True, notes, replace(crop))

    def _p3_warp(self, crop: PlateCrop) -> StageResult:
        """P3 Perspective Warp + Skew Correction.

        Rectifies the quad: drives ``skew_deg`` toward ~0 and yields a small sharpness
        gain (the warp resamples onto a clean rectilinear grid).
        """
        out = replace(crop)
        residual = round(crop.skew_deg * 0.05, 3)  # ~95% of skew removed
        out.skew_deg = residual
        out.blur_score = _clamp(crop.blur_score + 0.03)
        notes = [f"warped->400x100, skew {crop.skew_deg:+.1f}->{residual:+.2f}deg"]
        return StageResult("P3", True, notes, out)

    def _p4_classify(self, crop: PlateCrop) -> StageResult:
        """P4 Condition Classification from brightness + blur.

        Honours an explicit ``crop.condition`` hint if one was supplied, otherwise
        derives the dominant condition from the measurements.
        """
        out = replace(crop)
        if crop.condition is not None:
            cond = crop.condition
            src = "hint"
        else:
            src = "derived"
            if crop.brightness < 0.30:
                cond = PlateCondition.LOW_LIGHT
            elif crop.brightness > 0.85:
                cond = PlateCondition.GLARE
            elif crop.blur_score < 0.40:
                cond = PlateCondition.MOTION_BLUR
            elif crop.blur_score < 0.60:
                cond = PlateCondition.SOILED
            else:
                cond = PlateCondition.NORMAL
        out.condition = cond
        return StageResult("P4", True, [f"condition={cond.value} ({src})"], out)

    def _p5_enhance(self, crop: PlateCrop) -> StageResult:
        """P5 Condition-Specific Enhancement: a condition-targeted quality bump."""
        out = replace(crop)
        cond = crop.condition or PlateCondition.NORMAL
        notes: list[str] = []
        if cond is PlateCondition.LOW_LIGHT:
            out.brightness = _clamp(crop.brightness + 0.30)
            out.blur_score = _clamp(crop.blur_score + 0.08)
            notes.append("zero_dce_lowlight + clahe")
        elif cond is PlateCondition.GLARE:
            out.brightness = _clamp(crop.brightness - 0.25)
            out.blur_score = _clamp(crop.blur_score + 0.05)
            notes.append("highlight_rolloff + tonemap")
        elif cond is PlateCondition.MOTION_BLUR:
            out.blur_score = _clamp(crop.blur_score + 0.22)
            notes.append("wiener_deconv (tracker velocity)")
        elif cond is PlateCondition.SOILED:
            out.blur_score = _clamp(crop.blur_score + 0.15)
            notes.append("morphological_declutter + unsharp")
        else:  # NORMAL
            out.blur_score = _clamp(crop.blur_score + 0.04)
            notes.append("clean_day clahe(L) + unsharp")
        return StageResult("P5", True, notes, out)

    def _p6_normalise(self, crop: PlateCrop) -> StageResult:
        """P6 Universal Normalisation: greyscale + 2/98 percentile contrast stretch."""
        out = replace(crop)
        # Contrast stretch pulls mean luminance toward mid-grey and modestly
        # sharpens the character/background separation.
        out.brightness = round(0.5 * crop.brightness + 0.5 * 0.5, 3)
        out.blur_score = _clamp(crop.blur_score + 0.03)
        return StageResult("P6", True, ["greyscale + contrast_stretch(2,98)"], out)

    def _p7_super_resolve(self, crop: PlateCrop) -> StageResult:
        """P7 Resolution Check + Conditional Super-Resolution.

        When the (original) width is below ``min_sr_width`` the crop is judged too small
        for reliable OCR: apply a Real-ESRGAN-style 4x upscale and an SR sharpness gain.
        """
        out = replace(crop)
        if crop.width < self.min_sr_width:
            out.width = crop.width * 4
            out.height = crop.height * 4
            out.blur_score = _clamp(crop.blur_score + 0.20)
            notes = [f"4x super-resolved {crop.width}->{out.width}px"]
        else:
            notes = ["resolution_ok, SR skipped"]
        return StageResult("P7", True, notes, out)

    def _quality_score(self, crop: PlateCrop) -> float:
        """Composite P8 quality score Q in [0, 1] from resolution, sharpness, skew."""
        res = _clamp(crop.width / 400.0)
        sharp = _clamp(crop.blur_score)
        skew = _clamp(1.0 - abs(crop.skew_deg) / _SKEW_FULL_PENALTY_DEG)
        return _W_RES * res + _W_SHARP * sharp + _W_SKEW * skew

    def _p8_gate(self, crop: PlateCrop) -> StageResult:
        """P8 Quality Gate: pass/fail decision for forwarding to OCR consensus."""
        q = self._quality_score(crop)
        sharp_ok = crop.blur_score >= self.min_sharpness
        ok = q > self.gate_pass and sharp_ok
        notes = [f"Q={q:.3f} threshold={self.gate_pass:.2f}"]
        if not sharp_ok:
            notes.append(
                f"reject (sharpness {crop.blur_score:.2f}<{self.min_sharpness:.2f})")
        elif ok:
            notes.append("pass->OCR")
        else:
            notes.append("reject (insufficient quality)")
        return StageResult("P8", ok, notes, replace(crop))

    # -- orchestration --------------------------------------------------------------

    def run(self, crop: PlateCrop) -> list[StageResult]:
        """Run P1..P8 in order; returns one :class:`StageResult` per stage.

        Each stage receives the (possibly transformed) crop from the previous stage.
        The last entry is the P8 quality gate.
        """
        results: list[StageResult] = []
        current = copy.deepcopy(crop)
        stages = (
            self._p1_localise,
            self._p2_corners,
            self._p3_warp,
            self._p4_classify,
            self._p5_enhance,
            self._p6_normalise,
            self._p7_super_resolve,
            self._p8_gate,
        )
        for stage_fn in stages:
            result = stage_fn(current)
            results.append(result)
            current = result.crop
        return results

    def passed_quality_gate(self, results: list[StageResult]) -> bool:
        """True iff the pipeline ran to P8 and the P8 gate passed."""
        if not results:
            return False
        last = results[-1]
        return last.stage == "P8" and last.ok


def resolve_plate(crops: list[PlateCrop]) -> dict:
    """End-to-end: preprocess each crop, gate it, then run consensus on survivors.

    Each crop is conditioned through P1-P8. Crops that fail the P8 quality gate are
    rejected. The surviving crops' ``(ocr_text, ocr_conf)`` candidates are handed to
    the EXISTING :func:`app.alpr.resolve` multi-frame consensus resolver.

    Returns a dict::

        {
          plate, confidence, valid, method,
          frames_used, frames_rejected, notes
        }
    """
    pre = ALPRPreprocessor()
    candidates: list[tuple[str, float]] = []
    frames_used = 0
    frames_rejected = 0
    notes: list[str] = []

    for idx, crop in enumerate(crops):
        results = pre.run(crop)
        gate = results[-1]
        if pre.passed_quality_gate(results):
            frames_used += 1
            candidates.append((crop.ocr_text, crop.ocr_conf))
            notes.append(f"frame{idx}: accepted ({gate.notes[0]})")
        else:
            frames_rejected += 1
            notes.append(f"frame{idx}: rejected ({gate.notes[0]})")

    if not candidates:
        return {
            "plate": "",
            "confidence": 0.0,
            "valid": False,
            "method": "no_frames_passed_gate",
            "frames_used": 0,
            "frames_rejected": frames_rejected,
            "notes": notes + ["all frames failed P8 quality gate"],
        }

    result = alpr.resolve(candidates)
    notes.extend(result.notes)
    return {
        "plate": result.plate,
        "confidence": result.confidence,
        "valid": result.valid,
        "method": result.method,
        "frames_used": frames_used,
        "frames_rejected": frames_rejected,
        "notes": notes,
    }
