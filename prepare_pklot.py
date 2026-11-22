"""Convierte los XML originales de PKLot en recortes y un manifiesto CSV."""

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def prepare(source: Path, output: Path, frames_per_group: int, seed: int) -> None:
    if not source.is_dir():
        raise ValueError(
            f"No existe la carpeta {source.resolve()}. Extrae allí el archivo original de PKLot "
            "o pasa la ruta donde ya lo extrajiste."
        )
    if frames_per_group < 1:
        raise ValueError("--frames-per-group debe ser mayor que cero.")
    groups = defaultdict(list)
    for path in sorted(source.rglob("*.xml")):
        if len(path.relative_to(source).parts) < 4:
            continue
        image = path.with_suffix(".jpg")
        if image.is_file():
            groups[path.parent.parent.parent.name, path.parent.name].append((image, path))
    group_keys = sorted(groups)
    if len(group_keys) < 5:
        raise ValueError(
            f"Encontré {len(group_keys)} grupos cámara/fecha en {source.resolve()}. "
            "Se necesitan al menos cinco con imágenes JPG y sus XML originales."
        )
    random.Random(seed).shuffle(group_keys)
    holdout = max(1, round(len(group_keys) * .15))
    output.mkdir(parents=True, exist_ok=False)
    crops = output / "recortes"
    crops.mkdir()
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        counter = 0
        for index, key in enumerate(group_keys):
            split = "test" if index < holdout else "val" if index < 2 * holdout else "train"
            samples = groups[key]
            indices = np.linspace(0, len(samples) - 1, min(len(samples), frames_per_group), dtype=int)
            for sample in indices:
                image_path, annotation = samples[sample]
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"No se pudo leer {image_path}")
                height, width = image.shape[:2]
                for space in ET.parse(annotation).getroot().findall("space"):
                    label = space.get("occupied")
                    points = space.findall("./contour/point")
                    if label not in {"0", "1"} or len(points) < 3:
                        continue
                    x, y, w, h = cv2.boundingRect(np.array(
                        [(int(point.attrib["x"]), int(point.attrib["y"])) for point in points], dtype=np.int32,
                    ))
                    patch = image[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
                    if not patch.size:
                        continue
                    relative = Path("recortes") / f"{counter:07d}.jpg"
                    patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(output / relative), patch):
                        raise OSError(f"No se pudo guardar {relative}")
                    writer.writerow({"path": relative.as_posix(), "label": label, "split": split,
                                     "group": "/".join(key)})
                    counter += 1
    print(f"{counter} recortes de {len(groups)} grupos. Manifiesto: {output / 'manifest.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/pklot-preparado"))
    parser.add_argument("--frames-per-group", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.frames_per_group, args.seed)
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
