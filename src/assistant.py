"""Clasificación de intención y respuestas basadas en el estado del video."""

import re
import unicodedata
from pathlib import Path

import torch
from torch import nn

from src.occupancy import Occupancy

INTENTS = ("libres", "ubicacion", "porcentaje", "lleno", "fuera")
LENGTH = 24


def tokens(text):
    plain = unicodedata.normalize("NFKD", text.casefold())
    return re.findall(r"[a-z0-9]+|%", plain.encode("ascii", "ignore").decode())


def encode(text, vocab):
    ids = [2] + [vocab.get(word, 1) for word in tokens(text)[:LENGTH - 1]]
    return ids + [0] * (LENGTH - len(ids))


class IntentModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.words = nn.Embedding(vocabulary_size, 32, padding_idx=0)
        self.positions = nn.Embedding(LENGTH, 32)
        self.encoder = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
        self.head = nn.Linear(32, len(INTENTS))

    def forward(self, ids):
        vectors = self.words(ids) + self.positions(torch.arange(ids.size(1), device=ids.device))
        return self.head(self.encoder(vectors, src_key_padding_mask=ids.eq(0))[:, 0])


class IntentClassifier:
    def __init__(self, checkpoint: Path):
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if artifact["format"] != "intent-v1":
            raise ValueError("Checkpoint del asistente incompatible.")
        self.vocab = artifact["vocab"]
        self.model = IntentModel(len(self.vocab) + 3)
        self.model.load_state_dict(artifact["state"])
        self.model.eval()
        self.threshold = artifact["threshold"]

    @torch.inference_mode()
    def predict(self, question):
        words = tokens(question)
        if not any(word in self.vocab for word in words):
            return "fuera"
        scores = self.model(torch.tensor([encode(question, self.vocab)])).softmax(1)[0]
        index = scores.argmax().item()
        return INTENTS[index] if scores[index] >= self.threshold else "fuera"


def answer_status(intent: str, state: Occupancy | None) -> str:
    if intent == "fuera":
        return "Puedo responder por plazas libres, sus ubicaciones, el porcentaje ocupado y si está lleno."
    if state is None or state.total == 0:
        return "Todavía no hay resultados de ocupación."
    free, pending = len(state.available), len(state.pending)
    if intent == "ubicacion":
        answer = "Plazas libres: " + ", ".join(state.available) if free else "No hay plazas libres confirmadas."
    elif intent == "porcentaje":
        if state.occupancy_percent is None:
            answer = "El porcentaje está pendiente de confirmación."
        else:
            answer = f"La ocupación es del {state.occupancy_percent:.1f} %."
    elif intent == "lleno":
        if free:
            answer = f"Quedan {free} plazas libres."
        elif pending:
            answer = "Aún no se puede confirmar si está lleno."
        else:
            answer = "El parqueadero está lleno."
    else:
        answer = f"Hay {free} plazas libres confirmadas."
    return answer + (f" Plazas pendientes ({pending}): {', '.join(state.pending)}." if pending else "")
