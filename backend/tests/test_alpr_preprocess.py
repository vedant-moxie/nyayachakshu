"""Tests for the ALPR preprocessing front-half (stages P1-P8).

Runnable two ways:
    python3 tests/test_alpr_preprocess.py        # plain assert harness
    pytest tests/test_alpr_preprocess.py         # if pytest is present

We insert the backend root on sys.path so the existing back-half importer
``from app import alpr`` (used inside alpr_preprocess) resolves cleanly. The
``pipelines`` directory has no __init__.py, so the module under test is loaded
by file path rather than as a package.
"""
import importlib.util
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_ROOT)


def _load_module():
    path = os.path.join(_BACKEND_ROOT, "app", "pipelines", "alpr_preprocess.py")
    spec = importlib.util.spec_from_file_location("alpr_preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution can find the module
    # via sys.modules (required on Python 3.14+ when loading by file path).
    sys.modules["alpr_preprocess"] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load_module()
PlateCrop = pp.PlateCrop
PlateCondition = pp.PlateCondition
StageResult = pp.StageResult
ALPRPreprocessor = pp.ALPRPreprocessor
resolve_plate = pp.resolve_plate


def test_clean_highres_crop_passes_gate():
    """A clean, sharp, well-lit high-res crop sails through P8."""
    crop = PlateCrop(width=400, height=100, skew_deg=2.0, blur_score=0.9,
                     brightness=0.55, ocr_text="AP37BK6798", ocr_conf=0.95)
    pre = ALPRPreprocessor()
    results = pre.run(crop)

    assert [r.stage for r in results] == ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    assert all(isinstance(r, StageResult) for r in results)
    # P3 should have rectified the skew toward ~0.
    p3 = next(r for r in results if r.stage == "P3")
    assert abs(p3.crop.skew_deg) < 1.0
    # P4 classifies a good crop as NORMAL.
    p4 = next(r for r in results if r.stage == "P4")
    assert p4.crop.condition is PlateCondition.NORMAL
    assert pre.passed_quality_gate(results) is True


def test_tiny_lowres_crop_triggers_super_resolution():
    """A tiny crop (< MIN_SR_WIDTH) must get 4x super-resolved at P7 and still pass."""
    crop = PlateCrop(width=80, height=20, skew_deg=1.0, blur_score=0.7,
                     brightness=0.5, ocr_text="TS09EM4521", ocr_conf=0.9)
    pre = ALPRPreprocessor()
    results = pre.run(crop)

    p7 = next(r for r in results if r.stage == "P7")
    assert p7.crop.width == 80 * 4  # 4x upscale applied
    assert any("4x super-resolved" in n for n in p7.notes)
    # 4x upscale lifts both resolution + sharpness, so the gate should pass.
    assert pre.passed_quality_gate(results) is True


def test_no_super_resolution_when_wide_enough():
    """A crop at/above MIN_SR_WIDTH keeps its width — SR is skipped."""
    pre = ALPRPreprocessor()
    crop = PlateCrop(width=pp.MIN_SR_WIDTH, height=40, blur_score=0.9, brightness=0.5)
    p7 = next(r for r in pre.run(crop) if r.stage == "P7")
    assert p7.crop.width == pp.MIN_SR_WIDTH
    assert any("SR skipped" in n for n in p7.notes)


def test_blurred_dark_crop_fails_gate():
    """A heavily blurred, dark crop is below the composite quality threshold -> reject."""
    crop = PlateCrop(width=90, height=22, skew_deg=12.0, blur_score=0.05,
                     brightness=0.08, ocr_text="??????????", ocr_conf=0.2)
    pre = ALPRPreprocessor()
    results = pre.run(crop)

    p4 = next(r for r in results if r.stage == "P4")
    assert p4.crop.condition is PlateCondition.LOW_LIGHT  # dark -> low light
    assert pre.passed_quality_gate(results) is False
    assert results[-1].stage == "P8" and results[-1].ok is False


def test_resolve_plate_consensus_over_noisy_frames():
    """3 frames (one noisy OCR variant) resolve to a valid consensus plate."""
    crops = [
        PlateCrop(width=400, height=100, blur_score=0.9, brightness=0.55,
                  ocr_text="AP37BK6798", ocr_conf=0.96),
        PlateCrop(width=400, height=100, blur_score=0.85, brightness=0.5,
                  ocr_text="4P37BK6798", ocr_conf=0.80),  # noisy: '4' for 'A'
        PlateCrop(width=400, height=100, blur_score=0.88, brightness=0.6,
                  ocr_text="AP37BK6798", ocr_conf=0.92),
    ]
    out = resolve_plate(crops)

    assert out["valid"] is True
    assert out["plate"] == "AP 37 BK 6798"
    assert out["frames_used"] == 3
    assert out["frames_rejected"] == 0
    assert "majority vote" in out["method"]
    assert out["confidence"] > 0.0


def test_resolve_plate_drops_rejected_frames():
    """Garbage low-quality frames are dropped before consensus; good ones still win."""
    crops = [
        PlateCrop(width=400, height=100, blur_score=0.9, brightness=0.55,
                  ocr_text="TS09EM4521", ocr_conf=0.95),
        # junk frame: tiny, dark, blurred, skewed -> must fail P8 and be excluded.
        PlateCrop(width=60, height=15, skew_deg=14.0, blur_score=0.02,
                  brightness=0.05, ocr_text="ZZZZZZZZZZ", ocr_conf=0.1),
        PlateCrop(width=400, height=100, blur_score=0.88, brightness=0.5,
                  ocr_text="TS09EM4521", ocr_conf=0.9),
    ]
    out = resolve_plate(crops)

    assert out["frames_rejected"] == 1
    assert out["frames_used"] == 2
    assert out["plate"] == "TS 09 EM 4521"
    assert out["valid"] is True


def test_resolve_plate_all_rejected():
    """If every frame fails the gate, return a no-consensus result (no crash)."""
    crops = [
        PlateCrop(width=50, height=12, skew_deg=14.0, blur_score=0.02,
                  brightness=0.04, ocr_text="garbage", ocr_conf=0.1),
    ]
    out = resolve_plate(crops)
    assert out["valid"] is False
    assert out["frames_used"] == 0
    assert out["frames_rejected"] == 1
    assert out["plate"] == ""


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"ok  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _main()
