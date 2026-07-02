"""Albumentations transforms for fish detection training data."""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np


MIN_BBOX_VISIBILITY = 0.25
MIN_BBOX_AREA = 8.0
BBOX_EPSILON = 1e-9


def _image_compression() -> A.BasicTransform:
    """Create an ImageCompression transform across Albumentations versions."""
    try:
        return A.ImageCompression(quality_range=(55, 95), p=0.25)
    except TypeError:
        return A.ImageCompression(quality_lower=55, quality_upper=95, p=0.25)


def _pad_if_needed(image_size: int) -> A.BasicTransform:
    """Create a PadIfNeeded transform across Albumentations versions."""
    try:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=1.0,
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
            p=1.0,
        )


def _gauss_noise() -> A.BasicTransform:
    """Create a GaussNoise transform across Albumentations versions."""
    try:
        return A.GaussNoise(var_limit=(5.0, 35.0), p=1.0)
    except TypeError:
        return A.GaussNoise(std_range=(0.02, 0.12), p=1.0)


def _affine() -> A.BasicTransform:
    """Create an Affine transform across Albumentations versions."""
    try:
        return A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.06, 0.06),
            rotate=(-8, 8),
            shear=(-4, 4),
            mode=cv2.BORDER_CONSTANT,
            cval=(114, 114, 114),
            p=0.55,
        )
    except TypeError:
        return A.Affine(
            scale=(0.9, 1.12),
            translate_percent=(-0.06, 0.06),
            rotate=(-8, 8),
            shear=(-4, 4),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
            p=0.55,
        )


def build_train_augmentation(image_size: int = 640) -> A.Compose:
    """Build the offline augmentation pipeline for YOLO bboxes.

    Args:
        image_size: Final square image size.

    Returns:
        Albumentations Compose configured for YOLO-format bounding boxes.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            _pad_if_needed(image_size),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=1.0,
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=25,
                        val_shift_limit=20,
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    _gauss_noise(),
                    _image_compression(),
                ],
                p=0.35,
            ),
            _affine(),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=MIN_BBOX_VISIBILITY,
            min_area=MIN_BBOX_AREA,
        ),
    )


def apply_yolo_augmentation(
    image: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    image_size: int = 640,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Apply the training augmentation to one image and its YOLO labels.

    Args:
        image: BGR image loaded by OpenCV.
        labels: YOLO labels as ``(class_id, cx, cy, w, h)``.
        image_size: Final square image size.

    Returns:
        Augmented image and filtered labels in YOLO format.
    """
    transform = build_train_augmentation(image_size=image_size)
    sanitized_labels = sanitize_yolo_labels(labels)
    bboxes = [label[1:] for label in sanitized_labels]
    class_labels = [label[0] for label in sanitized_labels]
    augmented: dict[str, Any] = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )
    aug_labels: list[tuple[int, float, float, float, float]] = []
    for class_id, bbox in zip(augmented["class_labels"], augmented["bboxes"]):
        cx, cy, width, height = (float(value) for value in bbox)
        if width <= 0.0 or height <= 0.0:
            continue
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        width = min(max(width, 0.0), 1.0)
        height = min(max(height, 0.0), 1.0)
        aug_labels.append((int(class_id), cx, cy, width, height))
    return augmented["image"], aug_labels


def sanitize_yolo_labels(
    labels: list[tuple[int, float, float, float, float]],
) -> list[tuple[int, float, float, float, float]]:
    """Clamp YOLO labels so Albumentations receives strictly valid bboxes.

    Small rounding errors in normalized labels can make edge-touching boxes
    become values such as -0.0000005 after Albumentations converts YOLO cxcywh
    to xyxy. This function clamps in xyxy space and recomputes cxcywh.
    """
    sanitized: list[tuple[int, float, float, float, float]] = []
    for class_id, cx, cy, width, height in labels:
        x1 = max(0.0, cx - width / 2.0)
        y1 = max(0.0, cy - height / 2.0)
        x2 = min(1.0, cx + width / 2.0)
        y2 = min(1.0, cy + height / 2.0)
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= BBOX_EPSILON or clipped_height <= BBOX_EPSILON:
            continue
        sanitized.append(
            (
                int(class_id),
                x1 + clipped_width / 2.0,
                y1 + clipped_height / 2.0,
                clipped_width,
                clipped_height,
            )
        )
    return sanitized
