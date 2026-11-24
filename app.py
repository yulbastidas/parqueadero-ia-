"""Comando local de detección para las fases 1 y 2."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from src.detector import BackgroundDetector, DetectorConfig
from src.occupancy import OccupancyMonitor, load_spaces
from src.video_processor import process_video


def main() -> int:
    parser = argparse.ArgumentParser(description="SMARTPARK AI | Ocupación por comparación de imágenes")
    parser.add_argument("video", type=Path, help="Ruta del video de entrada")
    parser.add_argument("--output", required=True, type=Path, help="Ruta del MP4 anotado")
    approach = parser.add_mutually_exclusive_group(required=True)
    approach.add_argument("--reference", type=Path, help="Imagen de las plazas vacías para el método clásico")
    approach.add_argument("--checkpoint", type=Path, help="Modelo entrenado con train_models.py")
    parser.add_argument("--vehicle-checkpoint", type=Path, help="Checkpoint entrenado con train_vehicles.py")
    parser.add_argument("--pixel-threshold", type=int, default=30)
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--spaces", required=True, type=Path, help="JSON de plazas normalizadas")
    parser.add_argument("--method", choices=("point", "overlap"), default="overlap")
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--confirmation", type=int, default=3)
    args = parser.parse_args()
    if args.vehicle_checkpoint and not args.checkpoint:
        parser.error("--vehicle-checkpoint requiere --checkpoint.")

    try:
        config = DetectorConfig(args.pixel_threshold, args.min_area, args.kernel_size)
        spaces = load_spaces(args.spaces)
        monitor = OccupancyMonitor(spaces, args.method, args.overlap, args.confirmation)
        if args.checkpoint is not None:
            from src.space_models import SpaceClassifier

            detector = SpaceClassifier(args.checkpoint, spaces)
            if args.vehicle_checkpoint:
                from src.vehicle_model import VehicleModel

                detector = VehicleModel(args.vehicle_checkpoint, detector)
        else:
            reference = cv2.imread(str(args.reference))
            if reference is None:
                raise ValueError("No se pudo abrir la imagen de referencia.")
            detector = BackgroundDetector(reference, config)
        summary = process_video(args.video, args.output, detector, args.max_frames, monitor)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    report = asdict(summary)
    if not args.vehicle_checkpoint:
        report["visual_changes"] = report.pop("vehicle_detections")
    method = args.checkpoint.stem if args.checkpoint else "comparacion-de-fondo"
    print(json.dumps({"method": method, "device": "cpu", "config": None if args.checkpoint else asdict(config),
                      "output": str(args.output), **report}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
