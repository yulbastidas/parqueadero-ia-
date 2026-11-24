"""Prepara una muestra reproducible de plazas y cajas de MetaPKLot."""

import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil, floor
from pathlib import Path
from tarfile import open as open_tar
from urllib.request import urlopen

SOURCE = "https://huggingface.co/datasets/DSBD-Research/MetaPKLot-Dataset"
CAMERAS = ("ufpr04", "ufpr05")


def fetch(url, path):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=45) as response:
            data = response.read()
        path.write_bytes(data)


def read_annotations(root, camera, kind, base, download):
    name = f"{camera}_{kind}"
    path = root / f"{name}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    archive = root / f"{name}.tar.xz"
    if download:
        fetch(f"{base}/annotations/original/{kind}/PKLot/{archive.name}", archive)
    with open_tar(archive) as tar:
        with tar.extractfile(f"{name}.json") as stream:
            return json.load(stream)


def select_frames(cars, spots, camera, seed):
    by_id = {item["id"]: item for item in spots["images"]}
    days = defaultdict(list)
    for item in cars["images"]:
        if item["id"] in by_id and item["file_name"] == by_id[item["id"]]["file_name"] and (
            item["annotationsRectangle"] == [0, 0, item["width"], item["height"]]
        ):
            days[tuple(item["date"])].append(item)
    keys = sorted(day for day, items in days.items() if len(items) >= 4)
    random.Random(seed).shuffle(keys)
    if len(keys) < (22 if camera == "ufpr04" else 4):
        raise ValueError(f"Muestra insuficiente de fechas para {camera}.")
    subsets = {"val": keys[:4], "train": keys[4:22]} if camera == "ufpr04" else {"test": keys[:4]}
    for split, dates in subsets.items():
        for date in dates:
            ordered = sorted(days[date], key=lambda item: item["time"])
            for position in range(4):
                yield split, f"{camera}/" + "-".join(map(str, date)), ordered[position * (len(ordered) - 1) // 3]


def prepare(root, seed, revision, download):
    import cv2

    base = f"{SOURCE}/resolve/{revision}"
    crops = root / "recortes"
    crops.mkdir(parents=True, exist_ok=True)
    spaces, vehicles, refs = [], [], []
    for camera in CAMERAS:
        cars = read_annotations(root, camera, "cars", base, download)
        spots = read_annotations(root, camera, "spots", base, download)
        selected = list(select_frames(cars, spots, camera, seed))
        ids = {item["id"] for _, _, item in selected}
        labels = []
        for dataset in (cars, spots):
            annotations = defaultdict(list)
            for entry in dataset["annotations"]:
                if entry["image_id"] in ids and not entry.get("iscrowd", 0):
                    annotations[entry["image_id"]].append(entry)
            labels.append(annotations)
        if download:
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(lambda name: fetch(f"{base}/{name}", root / name),
                                  [item["file_name"] for _, _, item in selected]))
        for split, group, item in selected:
            image = cv2.imread(str(root / item["file_name"]))
            if image is None or image.shape[:2] != (item["height"], item["width"]):
                raise ValueError(f"Falta JPG válido: {item['file_name']}")
            height, width = image.shape[:2]
            boxes = []
            for label in labels[0][item["id"]]:
                x, y, w, h = label["bbox"]
                box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
            vehicles.append({"image": item["file_name"], "group": group, "split": split, "boxes": boxes})
            polygons = []
            for label in labels[1][item["id"]]:
                x, y, w, h = label["bbox"]
                patch = image[max(0, floor(y)):min(height, ceil(y + h)), max(0, floor(x)):min(width, ceil(x + w))]
                if not patch.size:
                    continue
                name = f"recortes/{label['id']}.jpg"
                if not cv2.imwrite(str(root / name), cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)):
                    raise OSError(f"No se pudo escribir {name}")
                spaces.append({"path": name, "label": label["category_id"], "split": split, "group": group})
                polygons.append({"label": label["category_id"], "segmentation": label["segmentation"]})
            refs.append({"image": item["file_name"], "split": split, "spaces": polygons})
    (root / "metadatos.json").write_text(json.dumps({"source": SOURCE, "revision": revision, "seed": seed,
                                                       "frames": refs}), encoding="utf-8")
    (root / "vehiculos.json").write_text(json.dumps(vehicles), encoding="utf-8")
    with (root / "plazas.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "split", "group"))
        writer.writeheader()
        writer.writerows(spaces)
    print(f"{len(vehicles)} JPG verificados, {sum(len(v['boxes']) for v in vehicles)} cajas y {len(spaces)} recortes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/meta"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.seed, args.revision, args.download)
