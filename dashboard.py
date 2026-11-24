"""Interfaz local de SMARTPARK AI."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.assistant import IntentClassifier, answer_status
from src.detector import ROOT, BackgroundDetector, DetectorConfig
from src.monitoring import AnalysisJob
from src.occupancy import OccupancyMonitor, validate_spaces
from src.space_models import SpaceClassifier

MODEL_DIR = ROOT / "models" / "ocupacion"
VEHICLE_CHECKPOINT = ROOT / "models" / "vehiculos" / "vehicles.pt"
INTENT_CHECKPOINT = ROOT / "models" / "asistente" / "intent.pt"


@st.cache_resource
def load_assistant(path: Path, modified_ns: int) -> IntentClassifier:
    return IntentClassifier(path)


@st.cache_data(show_spinner=False, max_entries=4)
def browser_video(source: str) -> bytes:
    import imageio_ffmpeg

    destination = Path(source).with_name("reproduccion.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", source, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return destination.read_bytes()


def indicators(state) -> None:
    total, free, occupied, ratio = st.columns(4)
    total.metric("Plazas configuradas", state.total)
    free.metric("Libres", len(state.available))
    occupied.metric("Ocupadas", len(state.occupied))
    ratio.metric("Ocupación", "Pendiente" if state.occupancy_percent is None else f"{state.occupancy_percent:.1f} %")
    if state.pending:
        st.caption(f"{len(state.pending)} plazas en confirmación.")


@st.fragment(run_every=1)
def monitor_view() -> None:
    job = st.session_state.get("job")
    if job is None:
        return
    latest, _ = job.snapshot()
    if latest is not None:
        if latest.occupancy is not None:
            indicators(latest.occupancy)
        done = latest.frame.index + 1
        if latest.total_frames:
            st.progress(min(done / latest.total_frames, 1.0), text=f"{done} / {latest.total_frames} fotogramas")
        st.caption(f"{latest.processing_fps:.2f} FPS de procesamiento")
    if job.running:
        if st.button("Detener análisis", disabled=job.stop.is_set()):
            job.stop.set()
        if latest is None:
            st.info("Preparando el detector y el video.")
        else:
            st.image(cv2.cvtColor(latest.frame.image, cv2.COLOR_BGR2RGB), width="stretch")
        return
    if job.error:
        st.error(job.error)
        return
    if job.summary and job.summary.cancelled:
        st.info("Análisis detenido. Se muestran los fotogramas procesados.")
    if job.output.exists():
        try:
            video = browser_video(str(job.output))
            st.video(video)
            st.download_button("Descargar video anotado", video, "smartpark.mp4", "video/mp4")
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            st.warning(f"La reproducción web no está disponible: {error}")
            st.download_button("Descargar MP4", job.output.read_bytes(), "smartpark.mp4", "video/mp4")


def monitoring_page() -> None:
    st.header("Monitoreo")
    job = st.session_state.get("job")
    running = job is not None and job.running
    checkpoints = {path.stem: path for path in sorted(MODEL_DIR.glob("*.pt"))}
    method = st.selectbox("Método", [*checkpoints, "Comparación de fondo"])
    use_boxes = st.checkbox("Detectar vehículos con cajas", disabled=method not in checkpoints
                            or not VEHICLE_CHECKPOINT.is_file())
    use_boxes = use_boxes and method in checkpoints and VEHICLE_CHECKPOINT.is_file()
    if method in checkpoints and not VEHICLE_CHECKPOINT.is_file():
        st.caption("Entrena el detector de vehículos para habilitar las cajas.")
    with st.form("video_configuration"):
        video = st.file_uploader("Video MP4 de cámara fija", type=["mp4"])
        reference = None
        if method == "Comparación de fondo":
            reference = st.file_uploader("Imagen de las plazas vacías", type=["png", "jpg", "jpeg"])
        configuration = st.file_uploader("Configuración JSON de plazas", type=["json"])
        left, middle, right = st.columns(3)
        limit = left.number_input("Máximo de fotogramas · 0 para todo", min_value=0, value=0, step=1)
        if method == "Comparación de fondo":
            threshold = middle.slider("Umbral de píxel", 1, 255, 30)
            area = right.number_input("Área mínima (píxeles)", min_value=1, value=300)
            st.caption("La referencia debe tener la misma resolución y el mismo encuadre que el video.")
        start = st.form_submit_button("Iniciar análisis", type="primary", disabled=running)
    if start:
        try:
            if video is None or configuration is None:
                raise ValueError("Carga el video y el JSON de plazas.")
            spaces = validate_spaces(json.loads(configuration.getvalue().decode("utf-8-sig")))
            if not spaces:
                raise ValueError("Configura al menos una plaza.")
            if method in checkpoints:
                detector = SpaceClassifier(checkpoints[method], spaces)
                if use_boxes:
                    from src.vehicle_model import VehicleModel

                    detector = VehicleModel(VEHICLE_CHECKPOINT, detector)
            else:
                if reference is None:
                    raise ValueError("Carga la imagen de referencia vacía.")
                image = cv2.imdecode(np.frombuffer(reference.getbuffer(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("No se pudo decodificar la imagen de referencia.")
                detector = BackgroundDetector(image, DetectorConfig(threshold, area))
            folder = ROOT / "results" / "web" / uuid4().hex
            folder.mkdir(parents=True)
            source = folder / "entrada.mp4"
            source.write_bytes(video.getbuffer())
            st.session_state.job = AnalysisJob(
                source, folder / "analisis.mp4", detector,
                OccupancyMonitor(spaces), limit or None,
            )
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            st.error(str(error))
    monitor_view()


@st.fragment(run_every=2)
def dashboard_page() -> None:
    st.header("Dashboard")
    job = st.session_state.get("job")
    if job is None:
        st.info("Inicia un análisis en Monitoreo para ver la ocupación y su historial.")
        return
    latest, records = job.snapshot()
    if latest is None or latest.occupancy is None:
        st.info(job.error or "Aún no hay fotogramas procesados.")
        return
    state = latest.occupancy
    indicators(state)
    st.write("Plazas libres · " + (", ".join(state.available) or "Ninguna confirmada"))
    st.caption(f"Análisis iniciado el {job.started:%d/%m/%Y a las %H:%M}. Datos del video procesado.")
    history = pd.DataFrame(asdict(row) for row in records)
    split = pd.DataFrame({"Estado": ["Libres", "Ocupadas", "En confirmación"],
                          "Plazas": [len(state.available), len(state.occupied), len(state.pending)]})
    left, right = st.columns([1, 2])
    left.plotly_chart(px.pie(split, names="Estado", values="Plazas", hole=.62,
                            color="Estado", color_discrete_map={"Libres": "#32ca91", "Ocupadas": "#f16b72",
                                                                "En confirmación": "#93a2b7"}),
                      width="stretch")
    line = px.line(history, x="seconds", y="occupation",
                   labels={"seconds": "Tiempo del video (s)", "occupation": "Ocupación (%)"}, markers=True)
    line.update_yaxes(range=[0, 100])
    right.plotly_chart(line, width="stretch")
    left, right = st.columns(2)
    left.metric("Inferencia media", f"{history['inference_ms'].mean():.1f} ms")
    right.metric("Procesamiento", f"{latest.processing_fps:.2f} FPS")
    st.dataframe(history.rename(columns={
        "frame": "Fotograma", "seconds": "Tiempo (s)", "available": "Libres", "occupied": "Ocupadas",
        "pending": "En confirmación", "occupation": "Ocupación (%)", "detections": "Detecciones",
        "inference_ms": "Inferencia (ms)", "processing_fps": "FPS",
    }), hide_index=True, width="stretch")
    st.download_button("Descargar historial CSV", history.to_csv(index=False).encode("utf-8-sig"),
                       "historial-smartpark.csv", "text/csv")


def configuration_page() -> None:
    st.header("Configuración")
    st.write("Cada cámara necesita un JSON de plazas delimitadas sobre su primer fotograma.")
    st.write("Abre el editor de polígonos desde PowerShell en la carpeta del proyecto.")
    st.code(r'& .\.venv\Scripts\python.exe edit_spaces.py "data/mi-video.mp4" "config/mis-plazas.json"', language="powershell")
    st.caption("Clic izquierdo agrega vértices, Enter confirma y S guarda. Carga el JSON en Monitoreo junto con tu video.")


def main() -> None:
    st.set_page_config(page_title="SMARTPARK AI", page_icon="🚘", layout="wide")
    st.sidebar.title("SMARTPARK AI")
    page = st.sidebar.radio("Navegación", ["Inicio", "Configuración", "Monitoreo", "Dashboard", "Disponibilidad", "Evaluación"])
    if page == "Inicio":
        st.title("SMARTPARK AI")
        st.write("Analiza un video de parqueadero y consulta qué plazas están libres u ocupadas.")
        st.caption("CNN entrenadas desde cero · Procesamiento local")
        st.info("En Monitoreo selecciona un modelo local y carga el video y el JSON de plazas.")
    elif page == "Configuración":
        configuration_page()
    elif page == "Monitoreo":
        monitoring_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Disponibilidad":
        st.header("Consulta de disponibilidad")
        st.caption("El asistente consulta el último fotograma procesado del mismo análisis que muestra el Dashboard.")
        if not INTENT_CHECKPOINT.is_file():
            st.info("Entrena el asistente con python train_assistant.py para habilitar las consultas.")
            return
        with st.form("pregunta_disponibilidad"):
            question = st.text_input("Pregunta", placeholder="¿Dónde puedo estacionar?", max_chars=200)
            submitted = st.form_submit_button("Consultar")
        if submitted and question.strip():
            job = st.session_state.get("job")
            latest = job.snapshot()[0] if job is not None else None
            try:
                classifier = load_assistant(INTENT_CHECKPOINT, INTENT_CHECKPOINT.stat().st_mtime_ns)
                st.write(answer_status(classifier.predict(question), latest.occupancy if latest else None))
                if latest:
                    st.caption(f"Fotograma consultado · {latest.frame.index + 1}")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                st.error(str(error))
    else:
        st.header("Evaluación")
        st.subheader("Ocupación")
        report = MODEL_DIR / "evaluation.json"
        if report.is_file():
            data = json.loads(report.read_text(encoding="utf-8"))
            if Path(data.get("manifest", "")).parent.name == "piloto-local":
                st.warning("Piloto de una sola toma. Las métricas no estiman el rendimiento en otros parqueaderos.")
            rows = [{"Modelo": name, "Parámetros": item["parameters"], "Tiempo (s)": item["seconds"],
                     "Predice ambas clases": item["test"].get("predicts_both_classes"),
                     **{key: item["test"][key] for key in ("accuracy", "precision", "recall", "f1")}}
                    for name, item in data["models"].items()]
            st.dataframe(rows, hide_index=True)
            st.download_button("Descargar evaluación", report.read_bytes(), "evaluacion.json", "application/json")
        else:
            st.info("Ejecuta el entrenamiento sobre imágenes etiquetadas para obtener los resultados de prueba.")
        for title, directory, filename in (
            ("Detección de vehículos", VEHICLE_CHECKPOINT.parent, "evaluacion-vehiculos.json"),
            ("Intenciones del asistente", INTENT_CHECKPOINT.parent, "evaluacion-asistente.json"),
        ):
            report = directory / "evaluation.json"
            if report.is_file():
                payload = report.read_bytes()
                data = json.loads(payload)
                with st.expander(title):
                    if "accuracy" in data:
                        st.metric("Exactitud en preguntas nuevas", f"{data['accuracy']:.1%}")
                    else:
                        st.json(data["test"])
                    st.download_button("Descargar informe", payload, filename, "application/json", key=filename)


if __name__ == "__main__":
    main()
