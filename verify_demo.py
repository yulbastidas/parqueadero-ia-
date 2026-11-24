"""Compara la referencia de MetaPKLot y comprueba el Dashboard y el asistente."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import torch
from streamlit.testing.v1 import AppTest

from src.monitoring import AnalysisJob
from src.occupancy import Occupancy, OccupancyMonitor, load_spaces


def check_web(job, expected, app=None):
    if app is None:
        app = AppTest.from_file("dashboard.py", default_timeout=30).run()
    if job is not None:
        app.session_state["job"] = job
    app.sidebar.radio[0].set_value("Dashboard").run()
    if expected is not None:
        metrics = {item.label: item.value for item in app.metric}
        assert metrics["Libres"] == str(len(expected.available))
        assert metrics["Ocupadas"] == str(len(expected.occupied))
        assert metrics["Ocupación"] == f"{expected.occupancy_percent:.1f} %"
        assert any(", ".join(expected.available) in item.value for item in app.markdown)
    app.sidebar.radio[0].set_value("Disponibilidad").run()
    questions = {"¿Cuántas plazas hay libres?": "Todavía no hay resultados de ocupación."}
    if expected is not None:
        free = len(expected.available)
        questions = {
            "¿Cuántas plazas hay libres?": f"Hay {free} plazas libres confirmadas.",
            "¿Dónde puedo estacionar?": ("Plazas libres: " + ", ".join(expected.available)
                                         if free else "No hay plazas libres confirmadas."),
            "¿Qué porcentaje está ocupado?": f"La ocupación es del {expected.occupancy_percent:.1f} %.",
            "¿Está lleno?": f"Quedan {free} plazas libres." if free else "El parqueadero está lleno.",
        }
    for question, answer in questions.items():
        app.text_input[0].set_value(question)
        app.button[0].click().run()
        assert not app.exception
        assert any(item.value == answer for item in app.markdown), question
    return questions


class ControlledDetector:
    def predict(self, frame):
        from src.detector import Detection, Prediction

        # Caja de prueba en el pasillo, con ocupación independiente y conocida.
        box = Detection(0, "Vehículo", 1.0, (390, 40, 450, 70))
        return Prediction((box,), 0, 0, frozenset(("P001", "P002")))


def submit_video(data):
    app = AppTest.from_file("dashboard.py", default_timeout=30).run()
    uploads = {"Video MP4 de cámara fija": data / "entrada.mp4",
               "Configuración JSON de plazas": data / "plazas.json"}

    def uploaded_file(label, **kwargs):
        path = uploads.get(label)
        return BytesIO(path.read_bytes()) if path else None

    # AppTest carece de una API pública para cargar archivos. Se entregan sus bytes al formulario.
    with patch("streamlit.file_uploader", side_effect=uploaded_file):
        app.sidebar.radio[0].set_value("Monitoreo").run()
        app.selectbox[0].set_value("cnn").run()
        app.checkbox[0].check().run()
        submit = next(button for button in app.button if button.label == "Iniciar análisis")
        submit.click().run()
    assert not app.exception
    return app, app.session_state["job"]


def verify():
    from dataclasses import asdict
    from datetime import datetime
    from time import sleep

    torch.set_num_threads(4)
    data = Path("data/demo-meta")
    output = Path("results") / datetime.now().strftime("demo-%Y%m%d-%H%M%S-%f")
    spaces = load_spaces(data / "plazas.json")
    check_web(None, None)
    controlled = AnalysisJob(data / "entrada.mp4", output / "control.mp4", ControlledDetector(),
                             OccupancyMonitor(spaces, confirmation=1), 1)
    while controlled.running:
        sleep(.1)
    assert controlled.error is None, controlled.error
    expected = Occupancy(("P001", "P002"), tuple(key for key in spaces if key not in ("P001", "P002")),
                         (), len(spaces), 100 * 2 / len(spaces))
    assert controlled.summary.occupancy == expected
    controlled_answers = check_web(controlled, expected)
    print("Comprobados el estado sin datos y la prueba controlada en Streamlit.", flush=True)
    app, job = submit_video(data)
    while job.running:
        sleep(.1)
    assert job.error is None, job.error
    latest, records = job.snapshot()
    assert job.output.is_file() and latest is not None
    assert job.summary.vehicle_detections > 0, "El detector no dibujó ninguna caja."
    app.run()
    assert not app.error and not app.exception
    assert app.get("video"), "Monitoreo no mostró el reproductor."
    answers = check_web(job, latest.occupancy, app)
    references = json.loads((data / "referencia.json").read_text(encoding="utf-8"))
    comparison = []
    for row in references:
        labels = row["labels"]
        assert None not in labels.values()
        count = sum(labels.values())
        actual = records[row["frame"] - 1]
        comparison.append({"reference_occupied": count, "reference_percent": 100 * count / len(labels),
                           **asdict(actual)})
    report = {"controlled_questions": controlled_answers, "annotated_video": str(job.output),
              "real_frame_questions": answers, "final_state": asdict(latest.occupancy), "frames": comparison}
    (output / "verificacion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    verify()
