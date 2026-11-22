"""Editor local de polígonos sobre una imagen o el primer fotograma."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.occupancy import load_spaces, validate_spaces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    frame = cv2.imread(str(args.source)) if args.source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} else None
    if frame is None:
        capture = cv2.VideoCapture(str(args.source))
        try:
            _, frame = capture.read()
        finally:
            capture.release()
    if frame is None:
        parser.exit(1, "No se pudo abrir la imagen o el video.\n")
    spaces = {key: value.tolist() for key, value in load_spaces(args.configuration).items()} if args.configuration.exists() else {}
    window = "SMARTPARK | Editor de plazas"
    height, width = frame.shape[:2]
    scale = min(1200 / width, 720 / height, 1)
    base = cv2.resize(frame, (round(width * scale), round(height * scale)))
    height, width = base.shape[:2]
    points: list[list[float]] = []
    selected = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal selected
        point = (x / width, y / height)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(list(point))
        elif event == cv2.EVENT_RBUTTONDOWN:
            selected = next((key for key, vertices in spaces.items()
                             if cv2.pointPolygonTest(np.asarray(vertices, dtype=np.float32), point, False) >= 0), None)
            points.clear()

    def payload(items: dict) -> dict:
        return {"spaces": [{"id": key, "points": vertices} for key, vertices in items.items()]}

    print("Clic izquierdo: vértice | Derecho: seleccionar | Enter: confirmar | D: eliminar | S: guardar | Esc: salir")
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 0:
            display = base.copy()
            for key, vertices in spaces.items():
                polygon = np.rint(np.asarray(vertices) * (width, height)).astype(np.int32)
                color = (40, 210, 250) if key == selected else (40, 220, 100)
                cv2.polylines(display, [polygon], True, color, 2)
                cv2.putText(display, key, tuple(polygon.mean(axis=0).astype(int)), 0, 0.5, color, 2)
            if points:
                draft = np.rint(np.asarray(points) * (width, height)).astype(np.int32)
                cv2.polylines(display, [draft], False, (40, 210, 250), 2)
                for point in draft:
                    cv2.circle(display, tuple(point), 4, (40, 210, 250), -1)
            cv2.imshow(window, display)
            key = cv2.waitKey(25) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13) and len(points) >= 3:
                identifier = selected or next(f"P{i:03}" for i in range(1, len(spaces) + 2) if f"P{i:03}" not in spaces)
                updated = spaces | {identifier: points.copy()}
                try:
                    validate_spaces(payload(updated))
                except ValueError as error:
                    print(error)
                    continue
                spaces = updated
                selected = None
                points.clear()
            if key == 8 and points:
                points.pop()
            if key == ord("d") and selected is not None:
                del spaces[selected]
                selected = None
                points.clear()
            if key == ord("s"):
                args.configuration.parent.mkdir(parents=True, exist_ok=True)
                args.configuration.write_text(json.dumps(payload(spaces), indent=2) + "\n", encoding="utf-8")
                print(f"Configuración guardada: {args.configuration}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
