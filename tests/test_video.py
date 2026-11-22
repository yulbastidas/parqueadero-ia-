import cv2
import numpy as np
import pytest

from src.detector import Detection, Prediction
from src.video_processor import analyze_frames, process_video


class FixedDetector:
    def predict(self, frame):
        return Prediction((Detection(2, "automóvil", 0.9, (4, 4, 40, 40)),), 2, 3)


@pytest.fixture
def source_video(tmp_path):
    source = tmp_path / "entrada con espacio.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (96, 64))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    finally:
        writer.release()
    return source


def test_sequential_indices_and_timestamps(source_video):
    capture = cv2.VideoCapture(str(source_video))
    try:
        frames = list(analyze_frames(capture, FixedDetector(), 20))
    finally:
        capture.release()
    assert [item.index for item in frames] == [0, 1, 2]
    assert [item.time_s for item in frames] == [0, 0.05, 0.1]


def test_export_limited_video(source_video, tmp_path):
    output = tmp_path / "resultado.mp4"
    stats = process_video(source_video, output, FixedDetector(), max_frames=2)
    assert (stats.frames, stats.vehicle_detections, stats.source_fps) == (2, 2, 20)
    assert (stats.mean_inference_ms, stats.mean_prediction_ms) == (2, 3)
    assert stats.processing_fps > 0
    capture = cv2.VideoCapture(str(output))
    try:
        ok, frame = capture.read()
        assert ok and frame.any()
        assert capture.read()[0]
        assert not capture.read()[0]
    finally:
        capture.release()
    with pytest.raises(FileExistsError):
        process_video(source_video, output, FixedDetector())


def test_invalid_input_cleans_partial_output(tmp_path):
    source = tmp_path / "corrupto.mp4"
    source.write_bytes(b"invalid video")
    output = tmp_path / "resultado.mp4"
    with pytest.raises(ValueError, match="abrir"):
        process_video(source, output, FixedDetector())
    assert list(tmp_path.iterdir()) == [source]


def test_detector_failure_cleans_partial_output(source_video, tmp_path):
    class FailingDetector(FixedDetector):
        calls = 0

        def predict(self, frame):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("Fallo de inferencia")
            return super().predict(frame)

    with pytest.raises(RuntimeError, match="Fallo de inferencia"):
        process_video(source_video, tmp_path / "resultado.mp4", FailingDetector())
    assert list(tmp_path.iterdir()) == [source_video]
