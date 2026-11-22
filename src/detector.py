"""Visión clásica mediante comparación con una referencia vacía."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    pixel_threshold: int = 30
    min_area: int = 300
    kernel_size: int = 5

    def __post_init__(self) -> None:
        if type(self.pixel_threshold) is not int or not 1 <= self.pixel_threshold <= 255:
            raise ValueError("El umbral de píxel debe estar entre 1 y 255.")
        if type(self.min_area) is not int or self.min_area < 1:
            raise ValueError("El área mínima debe ser un entero positivo.")
        if type(self.kernel_size) is not int or not 1 <= self.kernel_size <= 31 or self.kernel_size % 2 == 0:
            raise ValueError("El tamaño del filtro debe ser impar, entre 1 y 31.")


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float | None
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Prediction:
    detections: tuple[Detection, ...]
    inference_ms: float
    prediction_ms: float
    occupied_spaces: frozenset[str] | None = None


class VehicleDetector(Protocol):
    def predict(self, frame: Frame) -> Prediction: ...


class BackgroundDetector:
    def __init__(self, reference: Frame, config: DetectorConfig) -> None:
        if not reference.size or reference.dtype != np.uint8 or reference.ndim != 3 or reference.shape[2] != 3:
            raise ValueError("Se requiere una imagen BGR de referencia.")
        self.config = config
        self.shape = reference.shape
        self.reference = self._prepare(reference)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size))

    @staticmethod
    def _prepare(frame: Frame) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.int16)

    def predict(self, frame: Frame) -> Prediction:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError("La referencia debe tener la misma resolución y encuadre que el video.")
        start = perf_counter()
        difference = self._prepare(frame) - self.reference
        luminance = difference[:, :, 0]
        # Compensa un cambio global de brillo utilizando una muestra uniforme.
        luminance -= int(np.median(luminance[::8, ::8]))
        maximum = np.abs(difference).max(axis=2)
        mask = (maximum >= self.config.pixel_threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        _, _, regions, _ = cv2.connectedComponentsWithStats(mask)
        detections = tuple(
            Detection(0, "Cambio visual", None, (float(x), float(y), float(x + width), float(y + height)))
            for x, y, width, height, area in regions[1:]
            if area >= self.config.min_area
        )
        elapsed = (perf_counter() - start) * 1000
        return Prediction(detections, elapsed, elapsed)
