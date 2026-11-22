"""Dos CNN binarias inicializadas desde cero con PyTorch."""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn

from src.detector import Frame, Prediction

ARCHITECTURES = ("cnn", "cnn-separable")
SIZE = 64


def build_model(architecture: str) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Arquitectura desconocida: {architecture}")
    layers: list[nn.Module] = []
    incoming = 3
    for outgoing in (16, 32, 64):
        if architecture == "cnn-separable" and incoming != 3:
            layers.extend((nn.Conv2d(incoming, incoming, 3, padding=1, groups=incoming, bias=False),
                           nn.Conv2d(incoming, outgoing, 1, bias=False)))
        else:
            layers.append(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False))
        layers.extend((nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)))
        incoming = outgoing
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))


def prepare_patch(image: Frame) -> torch.Tensor:
    resized = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


class SpaceClassifier:
    def __init__(self, checkpoint: Path, spaces: dict[str, np.ndarray], threshold: float = .5) -> None:
        if not 0 < threshold < 1 or not spaces:
            raise ValueError("Se requieren plazas y un umbral entre cero y uno.")
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if artifact["format"] != 1:
            raise ValueError("Formato de checkpoint incompatible.")
        self.model = build_model(artifact["architecture"])
        self.model.load_state_dict(artifact["state"])
        self.model.eval()
        self.spaces = spaces
        self.threshold = threshold

    def predict(self, frame: Frame) -> Prediction:
        start = perf_counter()
        height, width = frame.shape[:2]
        patches = []
        for polygon in self.spaces.values():
            left, top = np.floor(polygon.min(axis=0) * (width, height)).astype(int)
            right, bottom = np.ceil(polygon.max(axis=0) * (width, height)).astype(int)
            patches.append(prepare_patch(frame[top:bottom, left:right]))
        batch = torch.stack(patches)
        inference_start = perf_counter()
        with torch.inference_mode():
            scores = self.model(batch).softmax(1)[:, 1].tolist()
        now = perf_counter()
        occupied = frozenset(key for key, score in zip(self.spaces, scores, strict=True) if score >= self.threshold)
        return Prediction((), (now - inference_start) * 1000, (now - start) * 1000, occupied)
