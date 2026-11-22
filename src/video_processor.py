"""Lectura secuencial y exportación de video anotado con OpenCV."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event
from time import perf_counter

import cv2

from src.detector import Frame, Prediction, VehicleDetector
from src.occupancy import Occupancy, OccupancyMonitor
from src.visualization import draw_detections


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    index: int
    time_s: float
    image: Frame
    prediction: Prediction


@dataclass(frozen=True, slots=True)
class VideoSummary:
    frames: int
    vehicle_detections: int
    source_fps: float
    processing_fps: float
    mean_inference_ms: float
    mean_prediction_ms: float
    occupancy: Occupancy | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class VideoProgress:
    frame: ProcessedFrame
    occupancy: Occupancy | None
    total_frames: int
    processing_fps: float


def analyze_frames(
    capture: cv2.VideoCapture,
    detector: VehicleDetector,
    fps: float,
    max_frames: int | None = None,
    stop: Event | None = None,
) -> Iterator[ProcessedFrame]:
    index = 0
    while max_frames is None or index < max_frames:
        if stop is not None and stop.is_set():
            break
        success, frame = capture.read()
        if not success:
            break
        prediction = detector.predict(frame)
        yield ProcessedFrame(index, index / fps, draw_detections(frame, prediction.detections), prediction)
        index += 1


def process_video(
    source: Path,
    output: Path,
    detector: VehicleDetector,
    max_frames: int | None = None,
    monitor: OccupancyMonitor | None = None,
    *,
    on_progress: Callable[[VideoProgress], None] | None = None,
    stop: Event | None = None,
) -> VideoSummary:
    if not source.is_file():
        raise FileNotFoundError(f"No existe el video de entrada: {source}")
    if output.suffix.lower() != ".mp4" or source.resolve() == output.resolve():
        raise ValueError("La salida debe ser un MP4 distinto del archivo de entrada.")
    if output.exists():
        raise FileExistsError(f"La salida ya existe: {output}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("El límite de fotogramas debe ser mayor que cero.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temp:
        temporary_path = Path(temp.name)
    capture = cv2.VideoCapture()
    writer = cv2.VideoWriter()
    start = perf_counter()
    frames = detections = 0
    inference_ms = prediction_ms = 0.0
    occupancy = None
    try:
        try:
            if not capture.open(str(source)):
                raise ValueError(f"OpenCV no pudo abrir el video: {source}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not isfinite(fps) or fps <= 0:
                raise ValueError("El video no informa una frecuencia de fotogramas válida.")
            total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            if max_frames is not None:
                total_frames = min(total_frames, max_frames) if total_frames else max_frames
            expected_size = None
            for item in analyze_frames(capture, detector, fps, max_frames, stop):
                height, width = item.image.shape[:2]
                if expected_size is None:
                    # MPEG-4 necesita dimensiones pares. Se agrega un borde si hace falta.
                    expected_size = (width, height)
                    size = (width + width % 2, height + height % 2)
                    if not writer.open(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size):
                        raise RuntimeError("OpenCV no pudo iniciar el codificador MP4.")
                if (width, height) != expected_size:
                    raise ValueError("El video cambia de dimensiones durante la lectura.")
                image = item.image
                if monitor is not None:
                    states = item.prediction.occupied_spaces
                    occupancy = (monitor.update(image.shape, item.prediction.detections) if states is None
                                 else monitor.update_states(states))
                    monitor.draw(image, occupancy)
                if (width, height) != size:
                    image = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(image)
                frames += 1
                detections += len(item.prediction.detections)
                inference_ms += item.prediction.inference_ms
                prediction_ms += item.prediction.prediction_ms
                if on_progress is not None:
                    on_progress(VideoProgress(item, occupancy, total_frames, frames / (perf_counter() - start)))
            if not frames:
                if stop is not None and stop.is_set():
                    return VideoSummary(0, 0, fps, 0, 0, 0, cancelled=True)
                raise ValueError("El video no contiene fotogramas decodificables.")
        finally:
            writer.release()
            capture.release()
        elapsed = perf_counter() - start
        verification = cv2.VideoCapture(str(temporary_path))
        try:
            if not verification.read()[0] or round(verification.get(cv2.CAP_PROP_FRAME_COUNT)) != frames:
                raise RuntimeError("El MP4 exportado es ilegible o tiene un recuento de fotogramas incorrecto.")
        finally:
            verification.release()
        temporary_path.rename(output)
        return VideoSummary(
            frames, detections, fps, frames / elapsed, inference_ms / frames, prediction_ms / frames,
            occupancy, stop is not None and stop.is_set(),
        )
    finally:
        temporary_path.unlink(missing_ok=True)
