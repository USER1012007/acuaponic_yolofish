"""Training metrics and validation visual reports for YOLO fish models."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORT, ReportConfig

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_COLOR = (60, 220, 60)
PRED_COLOR = (60, 80, 255)


def import_cv2() -> Any:
    """Import OpenCV with a clear runtime error."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "Missing or broken OpenCV dependency. Activate the training environment "
            "or install system libraries required by opencv-python."
        ) from exc
    return cv2


def import_numpy() -> Any:
    """Import NumPy with a clear runtime error."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing dependency 'numpy'.") from exc
    return np


def import_pyplot() -> Any:
    """Import matplotlib pyplot using a non-interactive backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing dependency 'matplotlib'.") from exc
    return plt


def import_yolo() -> Any:
    """Import Ultralytics YOLO with a clear runtime error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Activate the training environment "
            "or install it with: pip install ultralytics"
        ) from exc
    return YOLO


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def resolve_dataset_path(data_yaml: Path, split: str = "val") -> Path:
    """Resolve the image directory for a split from a YOLO data YAML."""
    data = load_yaml(data_yaml)
    root = Path(str(data.get("path", ".")))
    if not root.is_absolute():
        root = root.resolve()
    split_value = Path(str(data.get(split, data.get("val", ""))))
    return split_value if split_value.is_absolute() else root / split_value


def list_images(image_dir: Path, max_images: int | None) -> list[Path]:
    """List validation images."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:max_images] if max_images is not None else images


def read_results_csv(csv_path: Path) -> list[dict[str, float]]:
    """Read Ultralytics results.csv."""
    if not csv_path.exists():
        LOGGER.warning("results.csv not found: %s", csv_path)
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None or value is None:
                    continue
                clean_key = key.strip()
                try:
                    parsed[clean_key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def plot_columns(
    rows: list[dict[str, float]],
    columns: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot selected metric columns."""
    available = [column for column in columns if rows and column in rows[0]]
    if not available:
        LOGGER.warning("No columns available for plot %s: %s", title, columns)
        return
    plt = import_pyplot()
    epochs = [row.get("epoch", index) for index, row in enumerate(rows)]
    plt.figure(figsize=(10, 5))
    for column in available:
        plt.plot(epochs, [row.get(column, math.nan) for row in rows], label=column)
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_metrics(run_dir: Path, output: Path) -> None:
    """Generate plots from Ultralytics results.csv."""
    rows = read_results_csv(run_dir / "results.csv")
    if not rows:
        return
    plot_columns(
        rows,
        ["train/cls_loss", "val/cls_loss"],
        "Classification loss (YOLO cls_loss)",
        output / "classification_loss.png",
    )
    plot_columns(
        rows,
        ["train/box_loss", "val/box_loss"],
        "Box regression loss",
        output / "box_loss.png",
    )
    plot_columns(
        rows,
        ["train/dfl_loss", "val/dfl_loss"],
        "Distribution focal loss",
        output / "dfl_loss.png",
    )
    plot_columns(
        rows,
        ["metrics/precision(B)", "metrics/recall(B)"],
        "Precision and recall",
        output / "precision_recall.png",
    )
    plot_columns(
        rows,
        ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        "mAP",
        output / "map.png",
    )


def label_path_for_image(image_path: Path) -> Path:
    """Resolve the YOLO label path from an image path."""
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path is not under an images directory: {image_path}") from exc
    parts[images_index] = "labels"
    label = Path(*parts).with_suffix(".txt")
    return label


def read_label_boxes(image_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """Read YOLO labels and convert them to xyxy pixel boxes."""
    label_path = label_path_for_image(image_path)
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, width, height = (float(value) for value in parts)
        if int(class_id) != 0:
            continue
        x1 = (cx - width / 2.0) * image_width
        y1 = (cy - height / 2.0) * image_height
        x2 = (cx + width / 2.0) * image_width
        y2 = (cy + height / 2.0) * image_height
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def greedy_match_ious(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    """Greedily match predictions to GT and return matched IoUs."""
    matched_predictions: set[int] = set()
    ious: list[float] = []
    for gt_box in gt_boxes:
        best_index = -1
        best_iou = 0.0
        for index, pred_box in enumerate(pred_boxes):
            if index in matched_predictions:
                continue
            iou = box_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            matched_predictions.add(best_index)
            ious.append(best_iou)
    return ious


def draw_boxes(
    image: Any,
    boxes: list[tuple[float, float, float, float]],
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw xyxy boxes on an image."""
    cv2 = import_cv2()
    for box in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def make_tile(
    image_path: Path,
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
    tile_width: int,
) -> Any:
    """Create one annotated tile."""
    cv2 = import_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read validation image: {image_path}")
    draw_boxes(image, gt_boxes, GT_COLOR, "GT fish")
    draw_boxes(image, pred_boxes, PRED_COLOR, "Pred fish")
    scale = tile_width / image.shape[1]
    tile_height = max(1, int(round(image.shape[0] * scale)))
    tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    cv2.putText(
        tile,
        image_path.name[:48],
        (8, tile.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def make_contact_sheet(tiles: list[Any], group_size: int, columns: int = 5) -> Any:
    """Create a contact sheet from tiles."""
    np = import_numpy()
    if not tiles:
        raise ValueError("No tiles supplied for contact sheet")
    tile_width = max(tile.shape[1] for tile in tiles)
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = int(np.ceil(group_size / columns))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        y1 = row * tile_height
        x1 = column * tile_width
        sheet[y1 : y1 + tile.shape[0], x1 : x1 + tile.shape[1]] = tile
    return sheet


def run_model_validation(
    model: Any,
    data_yaml: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    """Run Ultralytics validation and return its metrics dictionary."""
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou, split="val")
    results_dict = getattr(metrics, "results_dict", {})
    return {str(key): float(value) for key, value in results_dict.items()}


def prediction_boxes(result: Any) -> list[tuple[float, float, float, float]]:
    """Extract class-0 prediction boxes from one Ultralytics result."""
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    boxes: list[tuple[float, float, float, float]] = []
    for box, class_id in zip(xyxy, classes):
        if int(class_id) != 0:
            continue
        boxes.append(tuple(float(value) for value in box))
    return boxes


def generate_prediction_sheets(
    model: Any,
    image_paths: list[Path],
    output: Path,
    imgsz: int,
    conf: float,
    iou: float,
    group_size: int,
    tile_width: int,
) -> dict[str, Any]:
    """Generate grouped validation PNGs and IoU stats."""
    cv2 = import_cv2()
    np = import_numpy()
    sheets_dir = output / "prediction_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    all_ious: list[float] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for group_index in range(0, len(image_paths), group_size):
        group = image_paths[group_index : group_index + group_size]
        results = model.predict(
            source=[str(path) for path in group],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        tiles: list[Any] = []
        for image_path, result in zip(group, results):
            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", image_path)
                continue
            height, width = image.shape[:2]
            gt_boxes = read_label_boxes(image_path, width, height)
            pred_boxes = prediction_boxes(result)
            matched_ious = greedy_match_ious(gt_boxes, pred_boxes)
            all_ious.extend(matched_ious)
            image_true_positive = sum(1 for matched_iou in matched_ious if matched_iou >= iou)
            true_positive += image_true_positive
            false_negative += max(0, len(gt_boxes) - image_true_positive)
            false_positive += max(0, len(pred_boxes) - image_true_positive)
            tiles.append(make_tile(image_path, gt_boxes, pred_boxes, tile_width))

        if tiles:
            sheet = make_contact_sheet(tiles, group_size=group_size)
            output_path = sheets_dir / f"val_group_{group_index // group_size:05d}.png"
            cv2.imwrite(str(output_path), sheet)
            if group_index and group_index % (group_size * 50) == 0:
                LOGGER.info("Generated %s validation sheets", group_index // group_size)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    summary = {
        "matched_boxes": len(all_ious),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "precision_at_iou": precision,
        "recall_at_iou": recall,
        "iou_threshold": iou,
    }
    plot_iou_histogram(all_ious, output / "iou_histogram.png")
    return summary


def plot_iou_histogram(ious: list[float], output_path: Path) -> None:
    """Plot an IoU histogram."""
    if not ious:
        LOGGER.warning("No IoU values available for histogram")
        return
    plt = import_pyplot()
    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=30, range=(0.0, 1.0), color="#3274a1")
    plt.title("Matched GT/prediction IoU")
    plt.xlabel("IoU")
    plt.ylabel("matched boxes")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_report(config: ReportConfig) -> None:
    """Run the complete validation report."""
    if not config.weights.exists():
        raise FileNotFoundError(f"Weights not found: {config.weights}")
    config.output.mkdir(parents=True, exist_ok=True)
    plot_training_metrics(config.run_dir, config.output)

    YOLO = import_yolo()
    model = YOLO(str(config.weights))
    summary: dict[str, Any] = {}
    if not config.skip_model_val:
        LOGGER.info("Running Ultralytics validation for mAP metrics")
        summary["ultralytics_val"] = run_model_validation(
            model=model,
            data_yaml=config.data,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
        )
    if not config.skip_prediction_images:
        image_dir = resolve_dataset_path(config.data, split="val")
        image_paths = list_images(image_dir, config.max_images)
        LOGGER.info("Generating prediction sheets for %s validation images", len(image_paths))
        summary["prediction_sheets"] = generate_prediction_sheets(
            model=model,
            image_paths=image_paths,
            output=config.output,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            group_size=config.group_size,
            tile_width=config.tile_width,
        )

    (config.output / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

