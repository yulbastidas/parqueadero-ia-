"""Localización de vehículos separada de la clasificación de plazas."""

from pathlib import Path
from time import perf_counter

import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from src.detector import Detection, Frame, Prediction
from src.space_models import SpaceClassifier


def build_detector():
    return ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=2, score_thresh=.05,
    )


def image_tensor(frame: Frame) -> torch.Tensor:
    return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255)


class VehicleModel:
    def __init__(self, checkpoint: Path, occupancy: SpaceClassifier):
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if artifact["format"] != "vehicles-ssdlite-v1":
            raise ValueError("Checkpoint de vehículos incompatible.")
        self.model = build_detector()
        self.model.load_state_dict(artifact["state"])
        self.model.eval()
        self.occupancy = occupancy
        self.threshold = artifact["threshold"]
        if not .05 <= self.threshold <= 1:
            raise ValueError("El umbral de cajas debe estar entre 0.05 y 1.")

    @torch.inference_mode()
    def predict(self, frame: Frame) -> Prediction:
        start = perf_counter()
        image = image_tensor(frame)
        inference = perf_counter()
        result = self.model([image])[0]
        elapsed = (perf_counter() - inference) * 1000
        detections = tuple(
            Detection(0, "Vehículo", score, tuple(box))
            for box, score in zip(result["boxes"].tolist(), result["scores"].tolist(), strict=True)
            if score >= self.threshold
        )
        occupation = self.occupancy.predict(frame)
        return Prediction(detections, elapsed + occupation.inference_ms,
                          (perf_counter() - start) * 1000, occupation.occupied_spaces)
