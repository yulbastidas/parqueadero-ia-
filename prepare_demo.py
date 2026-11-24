"""Crea un MP4 de prueba con capturas separadas de una cámara fija de MetaPKLot."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.occupancy import validate_spaces


def prepare(root, output):
    metadata = json.loads((root / "metadatos.json").read_text(encoding="utf-8"))
    test = [item for item in metadata["frames"] if item["split"] == "test"]
    day = Path(test[0]["image"]).parent
    frames = [item for item in test if Path(item["image"]).parent == day][:4]
    if len(frames) < 2:
        raise ValueError("Se necesitan al menos dos capturas del mismo día y encuadre.")
    image = cv2.imread(str(root / frames[0]["image"]))
    if image is None:
        raise ValueError("Falta la primera captura del MP4 de prueba.")
    height, width = image.shape[:2]
    scale = np.array([width, height], dtype=np.float32)
    spaces = []
    expected = []
    for index, spot in enumerate(sorted(frames[0]["spaces"],
                                        key=lambda item: -abs(cv2.contourArea(np.asarray(
                                            item["segmentation"][0], dtype=np.float32).reshape(-1, 2))))):
        if len(spaces) >= 12:
            break
        points = np.array(spot["segmentation"][0], dtype=np.float32).reshape(-1, 2)
        hull = cv2.convexHull(np.clip(points / scale, 0, 1)).reshape(-1, 2)
        candidate = {"id": f"P{index + 1:03d}", "points": hull.tolist()}
        try:
            validate_spaces({"spaces": [*spaces, candidate]})
        except ValueError:
            continue
        spaces.append(candidate)
        expected.append((candidate["id"], hull))
    if not spaces:
        raise ValueError("La captura no contiene polígonos válidos.")
    output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output / "entrada.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 3, (width, height))
    if not writer.isOpened():
        raise RuntimeError("No se pudo crear el MP4 de prueba.")
    reference = []
    try:
        for index, frame in enumerate(frames):
            image = cv2.imread(str(root / frame["image"]))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Captura inválida: {frame['image']}")
            for _ in range(3):
                writer.write(image)
            labels = {}
            for name, hull in expected:
                matches = []
                for spot in frame["spaces"]:
                    contour = np.asarray(spot["segmentation"][0], dtype=np.float32).reshape(-1, 2)
                    other = cv2.convexHull(np.clip(contour / scale, 0, 1))
                    intersection = cv2.intersectConvexConvex(hull, other)[0]
                    union = cv2.contourArea(hull) + cv2.contourArea(other) - intersection
                    if union and intersection / union > .8:
                        matches.append(spot["label"])
                labels[name] = matches[0] if len(matches) == 1 else None
            reference.append({"frame": index * 3 + 3, "image": frame["image"], "labels": labels})
    finally:
        writer.release()
    (output / "plazas.json").write_text(json.dumps({"spaces": spaces}), encoding="utf-8")
    (output / "referencia.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"MP4 de prueba con {len(frames)} capturas de {day} y {len(spaces)} plazas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/meta"))
    parser.add_argument("--output", type=Path, default=Path("data/demo-meta"))
    args = parser.parse_args()
    prepare(args.root, args.output)
