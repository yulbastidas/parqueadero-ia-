"""Anotación de vehículos sobre fotogramas BGR."""

import cv2

from src.detector import Detection, Frame

COLORS = {3: (245, 182, 42), 4: (220, 150, 240), 6: (99, 219, 255), 8: (240, 130, 130)}


def draw_detections(frame: Frame, detections: tuple[Detection, ...]) -> Frame:
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    scale = max(0.45, min(width, height) / 1200)
    thickness = max(1, round(scale * 2))
    for detection in detections:
        left, top, right, bottom = map(round, detection.xyxy)
        left, right = (min(width - 1, max(0, x)) for x in (left, right))
        top, bottom = (min(height - 1, max(0, y)) for y in (top, bottom))
        color = COLORS.get(detection.class_id, (245, 182, 42))
        # Las fuentes Hershey de OpenCV solo representan caracteres ASCII.
        label = detection.label.translate(str.maketrans("áéíóú", "aeiou"))
        text = label if detection.confidence is None else f"{label} {detection.confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        origin_y = max(top - baseline - 3, text_height + 3)
        origin_x = max(0, min(left, width - text_width - 6))
        cv2.rectangle(rendered, (left, top), (right, bottom), color, thickness)
        cv2.rectangle(
            rendered, (origin_x, origin_y - text_height - 3),
            (origin_x + text_width + 6, origin_y + baseline + 2), color, cv2.FILLED,
        )
        cv2.putText(rendered, text, (origin_x + 3, origin_y), font, scale, (15, 20, 25), thickness)
    return rendered
