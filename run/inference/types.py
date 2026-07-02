"""Shared data structures for inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Single object detection in original-frame coordinates.

    Attributes:
        class_id: Numeric class index emitted by the model.
        class_name: Human-readable class name.
        confidence: Detection confidence after postprocessing.
        x1: Left coordinate in pixels.
        y1: Top coordinate in pixels.
        x2: Right coordinate in pixels.
        y2: Bottom coordinate in pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ResizeInfo:
    """Metadata needed to map model-space boxes back to frame-space boxes."""

    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_x: int
    pad_y: int
