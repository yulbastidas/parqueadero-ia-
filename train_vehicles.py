"""Entrena SSDLite desde cero con un manifiesto JSON de cajas por imagen."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from src.vehicle_model import build_detector, image_tensor


class Images(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image"]))
        if image is None:
            raise ValueError(f"No se pudo leer {row['image']}")
        height, width = image.shape[:2]
        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in boxes.tolist():
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Caja inválida en {row['image']}")
        return image_tensor(image), {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.long)}


def collate(batch):
    return tuple(zip(*batch))


@torch.inference_mode()
def evaluate(model, loader, thresholds):
    model.eval()
    counts = {threshold: [0, 0, 0] for threshold in thresholds}
    for images, targets in loader:
        for result, target in zip(model(list(images)), targets, strict=True):
            for threshold, count in counts.items():
                boxes = result["boxes"][result["scores"] >= threshold]
                overlaps = box_iou(boxes, target["boxes"])
                matched = set()
                for row in overlaps:
                    choices = [i for i in range(len(row)) if i not in matched and row[i] >= .5]
                    if choices:
                        matched.add(max(choices, key=lambda i: float(row[i])))
                count[0] += len(matched)
                count[1] += len(boxes) - len(matched)
                count[2] += len(target["boxes"]) - len(matched)
    return {threshold: {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp),
                        "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn)}
            for threshold, (tp, fp, fn) in counts.items()}


def train(manifest, output, epochs, seed):
    rows = {key: [] for key in ("train", "val", "test")}
    groups, paths = {}, set()
    for row in json.loads(manifest.read_text(encoding="utf-8-sig")):
        split, group = row["split"], row["group"]
        path = (manifest.parent / row["image"]).resolve()
        if not group or groups.setdefault(group, split) != split or path in paths:
            raise ValueError("Se requieren grupos exclusivos y rutas únicas.")
        paths.add(path)
        rows[split].append(row)
    if epochs < 1 or len(rows["train"]) < 2 or any(
        not any(row["boxes"] for row in items) for items in rows.values()
    ):
        raise ValueError("Se requieren cajas en cada partición, dos imágenes de entrenamiento y una época.")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    loaders = {
        split: DataLoader(Images(manifest.parent, items), batch_size=2, shuffle=split == "train",
                          drop_last=split == "train", collate_fn=collate)
        for split, items in rows.items()
    }
    model = build_detector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    started = perf_counter()
    best, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = sum(model(list(images), list(targets)).values())
            loss.backward()
            optimizer.step()
        scores = evaluate(model, loaders["val"], (.15, .25, .4, .5, .65))
        threshold = max(scores, key=lambda value: scores[value]["f1"])
        score = scores[threshold]["f1"]
        history.append({"epoch": epoch, "threshold": threshold, "validation": scores[threshold]})
        if best is None or score > best[0]:
            best = score, threshold, epoch, {key: value.clone() for key, value in model.state_dict().items()}
        print(f"Época {epoch}/{epochs} | F1 validación {score:.3f}", flush=True)
    _, threshold, epoch, weights = best
    seconds = perf_counter() - started
    model.load_state_dict(weights)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "vehicles-ssdlite-v1", "state": weights, "threshold": threshold}, output / "vehicles.pt")
    report = {"manifest": str(manifest), "seed": seed, "seconds": seconds, "best_epoch": epoch,
              "threshold": threshold, "iou": .5, "history": history, "samples": {k: len(v) for k, v in rows.items()},
              "test": evaluate(model, loaders["test"], (threshold,))[threshold]}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/vehiculos"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.manifest, args.output, args.epochs, args.seed)
