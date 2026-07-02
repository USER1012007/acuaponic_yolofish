"""Preprocessing and YOLOv8 postprocessing utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from inference.types import Detection, ResizeInfo

PADDING_VALUE = 114


@dataclass(frozen=True)
class PreprocessConfig:
    """Image preprocessing configuration."""

    input_width: int
    input_height: int
    input_format: str
    input_color: str


@dataclass(frozen=True)
class PostprocessConfig:
    """Detection postprocessing configuration."""

    mode: str
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    raw_has_objectness: bool
    nms_box_format: str


def preprocess_frame(frame_bgr: np.ndarray, config: PreprocessConfig) -> tuple[np.ndarray, ResizeInfo]:
    """Resize, letterbox and batch one frame for Hailo inference.

    Args:
        frame_bgr: Original OpenCV BGR frame.
        config: Preprocessing options.

    Returns:
        Tuple of batched input tensor and resize metadata.
    """
    original_height, original_width = frame_bgr.shape[:2]
    scale = min(config.input_width / original_width, config.input_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_x = (config.input_width - resized_width) // 2
    pad_y = (config.input_height - resized_height) // 2

    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(
        (config.input_height, config.input_width, 3),
        PADDING_VALUE,
        dtype=np.uint8,
    )
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    if config.input_color == "rgb":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    elif config.input_color != "bgr":
        raise ValueError(f"model.input_color debe ser 'rgb' o 'bgr', recibido: {config.input_color}")

    if config.input_format == "float32":
        input_tensor = canvas.astype(np.float32) / 255.0
    elif config.input_format == "uint8":
        input_tensor = canvas
    else:
        raise ValueError(f"model.input_format debe ser 'uint8' o 'float32', recibido: {config.input_format}")

    resize_info = ResizeInfo(
        original_width=original_width,
        original_height=original_height,
        input_width=config.input_width,
        input_height=config.input_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return np.expand_dims(input_tensor, axis=0), resize_info


def postprocess_outputs(
    outputs: dict[str, np.ndarray],
    resize_info: ResizeInfo,
    class_names: Sequence[str],
    config: PostprocessConfig,
) -> list[Detection]:
    """Decode Hailo outputs into frame-space detections."""
    if not outputs:
        return []

    output_values = list(outputs.values())
    if config.mode == "hailo_nms":
        return _postprocess_hailo_nms(output_values, resize_info, class_names, config)
    if config.mode == "raw_yolov8":
        return _postprocess_raw_yolov8(output_values[0], resize_info, class_names, config)
    if config.mode == "auto":
        try:
            return _postprocess_hailo_nms(output_values, resize_info, class_names, config)
        except (TypeError, ValueError, IndexError):
            return _postprocess_raw_yolov8(output_values[0], resize_info, class_names, config)

    raise ValueError("postprocess.mode debe ser 'auto', 'hailo_nms' o 'raw_yolov8'")


def draw_detections(frame_bgr: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels over a frame."""
    output = frame_bgr.copy()
    for detection in detections:
        color = _color_for_class(detection.class_id)
        cv2.rectangle(output, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        label_y1 = max(detection.y1 - text_height - baseline - 4, 0)
        label_y2 = label_y1 + text_height + baseline + 4
        cv2.rectangle(output, (detection.x1, label_y1), (detection.x1 + text_width + 6, label_y2), color, -1)
        cv2.putText(
            output,
            label,
            (detection.x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def _postprocess_raw_yolov8(
    output: np.ndarray,
    resize_info: ResizeInfo,
    class_names: Sequence[str],
    config: PostprocessConfig,
) -> list[Detection]:
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        raise ValueError(f"Salida raw YOLOv8 invalida, shape={output.shape}")

    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] in {len(class_names) + 4, len(class_names) + 5}:
        predictions = predictions.T

    class_start = 5 if config.raw_has_objectness else 4
    if predictions.shape[1] < class_start + len(class_names):
        raise ValueError(
            "La salida raw no contiene suficientes columnas para bbox + clases: "
            f"shape={predictions.shape}, classes={len(class_names)}"
        )

    boxes_xywh = predictions[:, 0:4]
    class_scores = predictions[:, class_start : class_start + len(class_names)]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    if config.raw_has_objectness:
        confidences = confidences * predictions[:, 4]

    candidate_indices = np.where(confidences >= config.confidence_threshold)[0]
    boxes: list[list[int]] = []
    scores: list[float] = []
    kept_class_ids: list[int] = []

    for index in candidate_indices:
        x_center, y_center, width, height = boxes_xywh[index]
        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        mapped = _map_model_box_to_frame((x1, y1, x2, y2), resize_info)
        box_width = max(mapped[2] - mapped[0], 0)
        box_height = max(mapped[3] - mapped[1], 0)
        if box_width == 0 or box_height == 0:
            continue

        boxes.append([mapped[0], mapped[1], box_width, box_height])
        scores.append(float(confidences[index]))
        kept_class_ids.append(int(class_ids[index]))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        config.confidence_threshold,
        config.iou_threshold,
        top_k=config.max_detections,
    )
    return _detections_from_nms_indices(indices, boxes, scores, kept_class_ids, class_names)


def _postprocess_hailo_nms(
    outputs: Sequence[np.ndarray],
    resize_info: ResizeInfo,
    class_names: Sequence[str],
    config: PostprocessConfig,
) -> list[Detection]:
    rows: list[tuple[int, float, tuple[float, float, float, float]]] = []

    for output in outputs:
        parsed = _parse_hailo_nms_output(output, class_names, config.nms_box_format)
        rows.extend(parsed)

    detections: list[Detection] = []
    for class_id, confidence, model_box in rows:
        if confidence < config.confidence_threshold:
            continue
        x1, y1, x2, y2 = _map_nms_box_to_frame(model_box, resize_info)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(class_id, class_names),
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections[: config.max_detections]


def _parse_hailo_nms_output(
    output: np.ndarray,
    class_names: Sequence[str],
    box_format: str,
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    if isinstance(output, np.ndarray) and output.dtype == object:
        return _parse_object_array_nms(output, class_names)

    array = np.asarray(output)
    if array.ndim == 0:
        raise ValueError("Salida NMS vacia o escalar.")

    squeezed = np.squeeze(array)
    if squeezed.ndim == 1:
        if squeezed.size < 6:
            raise ValueError(f"Fila NMS invalida: shape={squeezed.shape}")
        squeezed = squeezed.reshape(1, -1)

    if squeezed.ndim != 2 or squeezed.shape[1] < 6:
        raise ValueError(f"Salida NMS no reconocida: shape={array.shape}")

    parsed: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for row in squeezed:
        class_id, confidence, box = _parse_numeric_nms_row(row, box_format)
        if class_id < 0 or class_id >= len(class_names):
            continue
        parsed.append((class_id, confidence, box))
    return parsed


def _parse_object_array_nms(
    output: np.ndarray,
    class_names: Sequence[str],
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    flattened = output.reshape(-1)
    parsed: list[tuple[int, float, tuple[float, float, float, float]]] = []

    if len(flattened) != len(class_names):
        raise ValueError(
            "La salida object-array NMS no coincide con el numero de clases: "
            f"outputs={len(flattened)}, classes={len(class_names)}"
        )

    for class_id, class_detections in enumerate(flattened):
        detections_array = np.asarray(class_detections)
        if detections_array.size == 0:
            continue
        detections_array = detections_array.reshape(-1, detections_array.shape[-1])
        if detections_array.shape[1] < 5:
            raise ValueError(f"Deteccion NMS invalida para clase {class_id}: {detections_array.shape}")
        for row in detections_array:
            ymin, xmin, ymax, xmax, score = row[:5]
            parsed.append((class_id, float(score), (float(xmin), float(ymin), float(xmax), float(ymax))))
    return parsed


def _parse_numeric_nms_row(row: np.ndarray, box_format: str) -> tuple[int, float, tuple[float, float, float, float]]:
    if box_format == "xyxy":
        x1, y1, x2, y2, score, class_id = row[:6]
    elif box_format == "yxyx":
        y1, x1, y2, x2, score, class_id = row[:6]
    else:
        raise ValueError(f"postprocess.nms_box_format debe ser 'xyxy' o 'yxyx', recibido: {box_format}")

    return int(class_id), float(score), (float(x1), float(y1), float(x2), float(y2))


def _map_nms_box_to_frame(
    box: tuple[float, float, float, float],
    resize_info: ResizeInfo,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= resize_info.input_width
        x2 *= resize_info.input_width
        y1 *= resize_info.input_height
        y2 *= resize_info.input_height
    return _map_model_box_to_frame((x1, y1, x2, y2), resize_info)


def _map_model_box_to_frame(
    box: tuple[float, float, float, float],
    resize_info: ResizeInfo,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    mapped_x1 = int(round((x1 - resize_info.pad_x) / resize_info.scale))
    mapped_y1 = int(round((y1 - resize_info.pad_y) / resize_info.scale))
    mapped_x2 = int(round((x2 - resize_info.pad_x) / resize_info.scale))
    mapped_y2 = int(round((y2 - resize_info.pad_y) / resize_info.scale))
    return (
        max(0, min(mapped_x1, resize_info.original_width - 1)),
        max(0, min(mapped_y1, resize_info.original_height - 1)),
        max(0, min(mapped_x2, resize_info.original_width - 1)),
        max(0, min(mapped_y2, resize_info.original_height - 1)),
    )


def _detections_from_nms_indices(
    indices: Any,
    boxes: Sequence[Sequence[int]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    class_names: Sequence[str],
) -> list[Detection]:
    detections: list[Detection] = []
    if len(indices) == 0:
        return detections

    for raw_index in np.asarray(indices).reshape(-1):
        index = int(raw_index)
        x, y, width, height = boxes[index]
        class_id = class_ids[index]
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(class_id, class_names),
                confidence=scores[index],
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
            )
        )
    return detections


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (46, 134, 193),
        (39, 174, 96),
        (230, 126, 34),
        (142, 68, 173),
        (192, 57, 43),
        (22, 160, 133),
    )
    return palette[class_id % len(palette)]
