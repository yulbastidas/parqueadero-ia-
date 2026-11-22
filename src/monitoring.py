"""Ejecución de un análisis en segundo plano y registro por fotograma."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread

from src.detector import VehicleDetector
from src.occupancy import OccupancyMonitor
from src.video_processor import VideoProgress, VideoSummary, process_video


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame: int
    seconds: float
    available: int
    occupied: int
    pending: int
    occupation: float | None
    detections: int
    inference_ms: float
    processing_fps: float


class AnalysisJob:
    def __init__(self, source: Path, output: Path, detector: VehicleDetector,
                 monitor: OccupancyMonitor, limit: int | None) -> None:
        self.source = source
        self.output = output
        self.started = datetime.now().astimezone()
        self.stop = Event()
        self._lock = Lock()
        self._latest = None
        self._records: list[FrameRecord] = []
        self.summary: VideoSummary | None = None
        self.error: str | None = None
        self._worker = Thread(target=self._run, args=(detector, monitor, limit), daemon=True)
        self._worker.start()

    @property
    def running(self) -> bool:
        return self._worker.is_alive()

    def snapshot(self) -> tuple[VideoProgress | None, tuple[FrameRecord, ...]]:
        with self._lock:
            return self._latest, tuple(self._records)

    def _record(self, progress: VideoProgress) -> None:
        frame = progress.frame
        state = progress.occupancy
        if state is None:
            raise RuntimeError("El análisis web requiere una configuración de plazas.")
        row = FrameRecord(frame.index + 1, frame.time_s, len(state.available), len(state.occupied),
                          len(state.pending), state.occupancy_percent, len(frame.prediction.detections),
                          frame.prediction.inference_ms, progress.processing_fps)
        with self._lock:
            self._latest = progress
            self._records.append(row)

    def _run(self, detector: VehicleDetector, monitor: OccupancyMonitor, limit: int | None) -> None:
        try:
            self.summary = process_video(self.source, self.output, detector, limit,
                                         monitor, on_progress=self._record, stop=self.stop)
        except Exception as error:
            self.error = str(error)
