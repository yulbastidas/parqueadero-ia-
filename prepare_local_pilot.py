"""Prepara una prueba piloto con plazas etiquetadas del video local de Sergey Rudakov."""

import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "data" / "pexels-5957716.mp4"
DESTINATION = ROOT / "data" / "piloto-local"
# Rectángulos revisados en el primer segundo, sobre una vista de 960 × 540.
# Cada plaza física pertenece a una sola partición.
SPACES = [
    ("derecha-blanco", "train", 1, (564, 116, 651, 160)),
    ("derecha-negro", "train", 1, (587, 183, 659, 228)),
    ("derecha-libre-1", "train", 0, (544, 90, 634, 121)),
    ("derecha-libre-2", "train", 0, (569, 156, 651, 188)),
    ("centro-rojo", "val", 1, (404, 263, 490, 306)),
    ("centro-libre", "val", 0, (392, 230, 474, 265)),
    ("izquierda-rojo", "test", 1, (77, 208, 170, 261)),
    ("izquierda-libre", "test", 0, (99, 297, 185, 338)),
]
FRAMES = range(0, 16, 2)


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {VIDEO}.")
    samples = []
    try:
        for index in range(FRAMES[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("El video terminó antes de obtener los fotogramas del piloto.")
            if index in FRAMES:
                samples.append((index, cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)))
    finally:
        capture.release()

    crops = DESTINATION / "recortes"
    crops.mkdir(parents=True, exist_ok=True)
    manifest = DESTINATION / "manifest.csv"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        for index, frame in samples:
            for name, split, label, (left, top, right, bottom) in SPACES:
                relative = Path("recortes") / f"{name}-{index:02d}.png"
                if not cv2.imwrite(str(DESTINATION / relative), frame[top:bottom, left:right]):
                    raise OSError(f"No se pudo guardar {relative}.")
                writer.writerow({"path": relative.as_posix(), "label": label, "split": split, "group": name})
    temporary.replace(manifest)
    print(f"{len(samples) * len(SPACES)} recortes, {len(SPACES)} plazas y un solo video. CSV: {manifest}")


if __name__ == "__main__":
    main()
