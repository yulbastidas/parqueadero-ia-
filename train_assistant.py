"""Entrena y evalúa un Transformer de intenciones en español."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.assistant import INTENTS, IntentClassifier, IntentModel, encode, tokens


def train(corpus, output, epochs, seed):
    if epochs < 1:
        raise ValueError("Se requiere al menos una época.")
    data = json.loads(corpus.read_text(encoding="utf-8"))
    splits = {}
    seen = set()
    for split in ("train", "val", "test"):
        splits[split] = [(question, intent) for intent in INTENTS for question in data[split][intent]]
        if any(not data[split][intent] for intent in INTENTS):
            raise ValueError(f"Faltan intenciones en {split}.")
        for question, _ in splits[split]:
            key = tuple(tokens(question))
            if not key or key in seen:
                raise ValueError(f"Pregunta vacía o repetida: {question}")
            seen.add(key)
    words = sorted({word for question, _ in splits["train"] for word in tokens(question)})
    vocab = {word: i for i, word in enumerate(words, 3)}
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = IntentModel(len(vocab) + 3)
    artifact = IntentClassifier.__new__(IntentClassifier)
    artifact.model, artifact.vocab = model, vocab
    examples = TensorDataset(torch.tensor([encode(q, vocab) for q, _ in splits["train"]]),
                             torch.tensor([INTENTS.index(i) for _, i in splits["train"]]))
    loader = DataLoader(examples, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    criterion = nn.CrossEntropyLoss()
    best = None
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(features), labels).backward()
            optimizer.step()
        model.eval()
        for threshold in (.45, .6, .75):
            artifact.threshold = threshold
            correct = sum(artifact.predict(q) == i for q, i in splits["val"])
            if best is None or correct > best[0]:
                best = correct, epoch, threshold, {k: v.clone() for k, v in model.state_dict().items()}
    correct, epoch, threshold, weights = best
    seconds = perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "intent.pt"
    torch.save({"format": "intent-v1", "vocab": vocab, "threshold": threshold, "state": weights}, checkpoint)
    classifier = IntentClassifier(checkpoint)
    matrix = [[0] * len(INTENTS) for _ in INTENTS]
    details = []
    for question, expected in splits["test"]:
        predicted = classifier.predict(question)
        matrix[INTENTS.index(expected)][INTENTS.index(predicted)] += 1
        details.append({"question": question, "expected": expected, "predicted": predicted})
    report = {"seed": seed, "seconds": seconds, "best_epoch": epoch, "threshold": threshold,
              "parameters": sum(p.numel() for p in model.parameters()),
              "sizes": {name: len(items) for name, items in splits.items()},
              "validation_accuracy": correct / len(splits["val"]),
              "accuracy": sum(matrix[i][i] for i in range(len(INTENTS))) / len(details),
              "intents": INTENTS, "confusion_matrix": matrix, "test": details}
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Accuracy de prueba: {report['accuracy']:.1%} en {len(details)} preguntas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("config/intenciones.json"))
    parser.add_argument("--output", type=Path, default=Path("models/asistente"))
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.corpus, args.output, args.epochs, args.seed)
