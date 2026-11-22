"""Entrena las dos CNN desde cero y compara sus predicciones en datos de prueba."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.space_models import ARCHITECTURES, build_model, prepare_patch


def load_dataset(manifest: Path) -> dict[str, TensorDataset]:
    groups = {name: ([], []) for name in ("train", "val", "test")}
    seen = set()
    group_splits = {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split, label = row["split"], int(row["label"])
            path = (manifest.parent / row["path"]).resolve()
            if split not in groups or label not in (0, 1) or path in seen:
                raise ValueError(f"Partición, etiqueta o ruta duplicada inválida: {row}")
            group = row.get("group")
            if group and group_splits.setdefault(group, split) != split:
                raise ValueError(f"El grupo {group} aparece en más de una partición.")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"No se pudo leer {path}.")
            seen.add(path)
            groups[split][0].append((prepare_patch(image) * 255).round().byte())
            groups[split][1].append(label)
    for name, (_, labels) in groups.items():
        if set(labels) != {0, 1}:
            raise ValueError(f"La partición {name} debe incluir plazas libres y ocupadas.")
    return {name: TensorDataset(torch.stack(images), torch.tensor(labels))
            for name, (images, labels) in groups.items()}


@torch.inference_mode()
def evaluate(model: nn.Module, data: DataLoader) -> dict:
    model.eval()
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for images, labels in data:
        predicted = model(images.float().div_(255)).argmax(1)
        matrix += torch.bincount(2 * labels + predicted, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = matrix.flatten().tolist()
    return {
        "accuracy": (tn + tp) / int(matrix.sum()),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fn + fp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def train_models(manifest: Path, destination: Path, epochs: int, seed: int) -> dict:
    if epochs < 1:
        raise ValueError("Indica al menos una época.")
    data = load_dataset(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    validation = DataLoader(data["val"], batch_size=64)
    test = DataLoader(data["test"], batch_size=64)
    for architecture in ARCHITECTURES:
        torch.manual_seed(seed)
        loader = DataLoader(data["train"], batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
        model = build_model(architecture)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        criterion = nn.CrossEntropyLoss()
        history = []
        best = None
        started = perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for images, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images.float().div_(255)), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(images)
            score = evaluate(model, validation)
            history.append({"epoch": epoch, "loss": total_loss / len(data["train"]), **score})
            if best is None or score["f1"] > best[0]:
                best = score["f1"], epoch, {key: value.clone() for key, value in model.state_dict().items()}
            print(f"{architecture} | época {epoch}/{epochs} | F1 validación {score['f1']:.3f}", flush=True)
        _, best_epoch, weights = best
        model.load_state_dict(weights)
        torch.save({"format": 1, "architecture": architecture, "state": weights}, destination / f"{architecture}.pt")
        reports[architecture] = {"parameters": sum(p.numel() for p in model.parameters()),
                                 "best_epoch": best_epoch, "seconds": perf_counter() - started,
                                 "history": history, "test": evaluate(model, test)}
    report = {"manifest": str(manifest), "seed": seed, "epochs": epochs,
              "samples": {name: len(items) for name, items in data.items()}, "models": reports}
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/ocupacion"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.exit(1, f"Error: no existe {args.manifest.resolve()}. Ejecuta primero prepare_pklot.py.\n")
    torch.set_num_threads(4)
    train_models(args.manifest, args.output, args.epochs, args.seed)


if __name__ == "__main__":
    main()
