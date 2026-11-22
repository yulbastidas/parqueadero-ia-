import numpy as np
import pytest

from src.detector import BackgroundDetector, Detection, DetectorConfig
from src.visualization import draw_detections


@pytest.mark.parametrize("overrides", [
    {"pixel_threshold": 0}, {"pixel_threshold": True}, {"pixel_threshold": 256},
    {"min_area": 0}, {"kernel_size": 4},
])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        DetectorConfig(**overrides)


def test_reference_detects_static_change_and_rejects_uniform_lighting():
    reference = np.full((120, 160, 3), 110, dtype=np.uint8)
    detector = BackgroundDetector(reference, DetectorConfig())
    assert detector.predict(reference).detections == ()
    assert detector.predict(reference + 35).detections == ()
    changed = reference.copy()
    changed[20:70, 30:85] = (170, 25, 25)
    detection, = detector.predict(changed).detections
    assert detection.confidence is None
    assert detection.xyxy == pytest.approx((30, 20, 85, 70), abs=2)
    with pytest.raises(ValueError, match="misma resolución"):
        detector.predict(changed[:80])


def test_annotation_clips_and_preserves_input():
    frame = np.zeros((64, 95, 3), dtype=np.uint8)
    rendered = draw_detections(frame, (Detection(2, "automóvil", 0.9, (-5, -5, 90, 90)),))
    assert rendered.shape == frame.shape
    assert not frame.any()
    assert rendered.any()
