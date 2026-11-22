"""Asociación de cajas con plazas convexas y confirmación temporal."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detector import Detection, Frame


def load_spaces(path: Path) -> dict[str, np.ndarray]:
    return validate_spaces(json.loads(path.read_text(encoding="utf-8-sig")))


def validate_spaces(document: dict) -> dict[str, np.ndarray]:
    spaces = {}
    for item in document["spaces"]:
        identifier = item["id"]
        polygon = np.asarray(item["points"], dtype=np.float32)
        if not isinstance(identifier, str) or not identifier.strip() or identifier in spaces:
            raise ValueError("Cada plaza debe tener un identificador único.")
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.all((polygon >= 0) & (polygon <= 1))
                or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0):
            raise ValueError(f"Polígono convexo normalizado inválido: {identifier}")
        if any(cv2.intersectConvexConvex(polygon, other)[0] > 1e-7 for other in spaces.values()):
            raise ValueError(f"La plaza {identifier} se superpone con otra.")
        spaces[identifier] = polygon
    return spaces


@dataclass(frozen=True)
class Occupancy:
    occupied: tuple[str, ...]
    available: tuple[str, ...]
    pending: tuple[str, ...]
    total: int
    occupancy_percent: float | None


class OccupancyMonitor:
    def __init__(self, spaces: dict[str, np.ndarray], method: str = "overlap",
                 threshold: float = 0.20, confirmation: int = 3) -> None:
        if method not in {"point", "overlap"} or not 0 < threshold <= 1 or confirmation < 1:
            raise ValueError("Configuración de ocupación inválida.")
        self.spaces = spaces
        self.method = method
        self.threshold = threshold
        self.confirmation = confirmation
        self.states: dict[str, bool] = {}
        self.previous: dict[str, tuple[bool, int]] = {}

    def update(self, shape: tuple[int, ...], detections: tuple[Detection, ...]) -> Occupancy:
        height, width = shape[:2]
        candidates = []
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.xyxy
            box = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float32)
            box /= (width, height)
            center = tuple(map(float, box.mean(axis=0)))
            for identifier, polygon in self.spaces.items():
                overlap = cv2.intersectConvexConvex(polygon, box)[0] / cv2.contourArea(polygon)
                inside = cv2.pointPolygonTest(polygon, center, False) >= 0
                if (self.method == "point" and inside) or (self.method == "overlap" and overlap >= self.threshold):
                    candidates.append((overlap, detection.confidence or 0.0, identifier, index))

        assigned_spaces = set()
        assigned_detections = set()
        for _, _, identifier, index in sorted(candidates, reverse=True):
            if identifier not in assigned_spaces and index not in assigned_detections:
                assigned_spaces.add(identifier)
                assigned_detections.add(index)

        return self.update_states(assigned_spaces)

    def update_states(self, assigned_spaces: set[str] | frozenset[str]) -> Occupancy:
        for identifier in self.spaces:
            occupied = identifier in assigned_spaces
            prior, count = self.previous.get(identifier, (occupied, 0))
            count = min(count + 1, self.confirmation) if prior == occupied else 1
            self.previous[identifier] = occupied, count
            if count >= self.confirmation:
                self.states[identifier] = occupied

        occupied = tuple(key for key in self.spaces if self.states.get(key) is True)
        available = tuple(key for key in self.spaces if self.states.get(key) is False)
        pending = tuple(key for key in self.spaces if key not in self.states)
        total = len(self.spaces)
        return Occupancy(occupied, available, pending, total, 100 * len(occupied) / total if total and not pending else None)

    def draw(self, frame: Frame, occupancy: Occupancy) -> None:
        height, width = frame.shape[:2]
        for identifier, polygon in self.spaces.items():
            points = np.rint(polygon * (width - 1, height - 1)).astype(np.int32)
            state = self.states.get(identifier)
            color = {True: (40, 40, 240), False: (40, 210, 60), None: (160, 160, 160)}[state]
            cv2.polylines(frame, [points], True, color, 3, cv2.LINE_AA)
            x, y = points.mean(axis=0).astype(int)
            cv2.putText(frame, identifier, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        percentage = "pendiente" if occupancy.occupancy_percent is None else f"{occupancy.occupancy_percent:.1f}%"
        text = (f"Libres: {len(occupancy.available)}   Ocupadas: {len(occupancy.occupied)}   "
                f"En confirmacion: {len(occupancy.pending)}   Ocupacion: {percentage}")
        scale = min(0.85, width / 1500)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(frame, (0, 0), (text_width + 24, text_height + baseline + 16), (15, 20, 25), cv2.FILLED)
        cv2.putText(frame, text, (12, text_height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
