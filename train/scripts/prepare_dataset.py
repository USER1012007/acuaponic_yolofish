"""Prepare the community fish dataset in YOLO format."""

from __future__ import annotations

import json
from json import JSONDecodeError
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PREPARE_DATASET, PrepareDatasetConfig

LOGGER = logging.getLogger(__name__)
FISH_CATEGORY_ID = 1
YOLO_CLASS_ID = 0
PILOT_TRAIN_IMAGES = 10_000
PILOT_VAL_IMAGES = 1_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ImageRecord:
    """Resolved image metadata used for YOLO conversion."""

    image_id: str
    source_path: str
    output_name: str
    split: str
    width: int
    height: int
    dataset: str | None
    original_data_source: str | None


@dataclass
class PrepareStats:
    """Counters collected while preparing the dataset."""

    selected_train: int = 0
    selected_val: int = 0
    missing_images: int = 0
    fish_annotations: int = 0
    empty_annotations: int = 0
    skipped_annotations: int = 0
    invalid_bboxes: int = 0
    augmented_images: int = 0
    calibration_images: int = 0


def import_ijson() -> Any | None:
    """Import ijson when available."""
    try:
        import ijson  # type: ignore
    except ImportError as exc:
        LOGGER.warning("ijson is not installed; using slower stdlib streaming parser: %s", exc)
        return None
    return ijson


def iter_json_items(
    json_path: Path,
    prefix: str,
    allow_stdlib_json: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield items from a large JSON array using ijson."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    ijson = import_ijson()
    if ijson is not None:
        with json_path.open("rb") as handle:
            yield from ijson.items(handle, f"{prefix}.item")
        return
    if not allow_stdlib_json:
        raise SystemExit(
            "Missing dependency 'ijson'. This JSON is large; install it with "
            "the training environment or run: pip install ijson. To use the "
            "slower fallback anyway, pass --allow-stdlib-json."
        )
    yield from iter_top_level_array_stdlib(json_path, prefix)


def iter_top_level_array_stdlib(json_path: Path, key: str) -> Iterable[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading the full file."""
    pattern = f'"{key}"'
    decoder = json.JSONDecoder()
    buffer = ""

    with json_path.open("r", encoding="utf-8") as handle:
        while pattern not in buffer:
            chunk = handle.read(READ_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Top-level array not found in JSON: {key}")
            buffer += chunk
            if pattern not in buffer and len(buffer) > len(pattern) + 1024:
                buffer = buffer[-(len(pattern) + 1024) :]

        buffer = buffer[buffer.index(pattern) + len(pattern) :]
        while "[" not in buffer:
            chunk = handle.read(READ_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"Array start not found for JSON key: {key}")
            buffer += chunk
        buffer = buffer[buffer.index("[") + 1 :]

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            while True:
                try:
                    item, index = decoder.raw_decode(buffer)
                except JSONDecodeError:
                    chunk = handle.read(READ_CHUNK_SIZE)
                    if not chunk:
                        raise ValueError(f"Unexpected end of JSON while reading {key}") from None
                    buffer += chunk
                    continue
                if isinstance(item, dict):
                    yield item
                buffer = buffer[index:]
                break


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing its current contents."""
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def basename_from(value: str | None) -> str | None:
    """Return a usable filename from a JSON id or file_name value."""
    if not value:
        return None
    name = Path(value).name
    return name if Path(name).suffix.lower() in IMAGE_EXTENSIONS else None


def candidate_paths(dataset_root: Path, item: dict[str, Any]) -> list[Path]:
    """Build likely local paths for a JSON image entry."""
    image_id = str(item.get("id", ""))
    file_name = str(item.get("file_name", ""))
    candidates: list[Path] = []

    for value in (image_id, file_name):
        if not value:
            continue
        candidates.append(dataset_root / value)
        if value.startswith("valid/"):
            candidates.append(dataset_root / "val" / Path(value).name)
        if value.startswith("val/"):
            candidates.append(dataset_root / "val" / Path(value).name)
        if value.startswith("train/"):
            candidates.append(dataset_root / "train" / Path(value).name)

    for name in {basename_from(image_id), basename_from(file_name)}:
        if not name:
            continue
        candidates.append(dataset_root / "train" / name)
        candidates.append(dataset_root / "val" / name)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def build_source_index(dataset_root: Path) -> dict[str, dict[str, Path]]:
    """Index local image files by basename for fast JSON resolution."""
    source_index: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for path in split_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                source_index.setdefault(path.name, {})[split] = path
    LOGGER.info("Indexed %s local image basenames", len(source_index))
    return source_index


def expected_splits(item: dict[str, Any]) -> list[str]:
    """Return preferred local splits for a JSON image entry."""
    image_id = str(item.get("id", ""))
    file_name = str(item.get("file_name", ""))
    values = (image_id, file_name)
    if any(value.startswith("train/") or "/train/" in value for value in values):
        return ["train", "val"]
    if any(value.startswith(("val/", "valid/")) or "/valid/" in value for value in values):
        return ["val", "train"]
    return ["train", "val"] if bool(item.get("is_train")) else ["val", "train"]


def resolve_image_path(
    dataset_root: Path,
    item: dict[str, Any],
    source_index: dict[str, dict[str, Path]],
) -> Path | None:
    """Resolve the local image path for one JSON image record."""
    image_id = str(item.get("id", ""))
    file_name = str(item.get("file_name", ""))
    for name in (basename_from(image_id), basename_from(file_name)):
        if not name or name not in source_index:
            continue
        paths_by_split = source_index[name]
        for split in expected_splits(item):
            if split in paths_by_split:
                return paths_by_split[split]
        return next(iter(paths_by_split.values()))
    for path in candidate_paths(dataset_root, item):
        if path.exists() and path.is_file():
            return path
    return None


def infer_split(item: dict[str, Any], source_path: Path) -> str:
    """Infer train or val split from local path and JSON metadata."""
    if source_path.parent.name == "train":
        return "train"
    if source_path.parent.name in {"val", "valid"}:
        return "val"
    return "train" if bool(item.get("is_train")) else "val"


def should_select(split: str, counts: dict[str, int], max_train: int | None, max_val: int | None) -> bool:
    """Return whether an image should be included in the prepared dataset."""
    if split == "train" and max_train is not None and counts["train"] >= max_train:
        return False
    if split == "val" and max_val is not None and counts["val"] >= max_val:
        return False
    return split in {"train", "val"}


def materialize_image(source: Path, destination: Path, link_mode: str) -> None:
    """Create a symlink, hardlink, or copy for an image."""
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        destination.symlink_to(source.resolve())
    elif link_mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def select_images(config: PrepareDatasetConfig, stats: PrepareStats) -> dict[str, ImageRecord]:
    """Select and materialize images that exist locally."""
    max_train = config.max_train
    max_val = config.max_val
    if config.pilot:
        max_train = max_train if max_train is not None else PILOT_TRAIN_IMAGES
        max_val = max_val if max_val is not None else PILOT_VAL_IMAGES

    selected: dict[str, ImageRecord] = {}
    counts = {"train": 0, "val": 0}
    images_dir = config.output / "images"
    labels_dir = config.output / "labels"
    for split in ("train", "val"):
        ensure_clean_dir(images_dir / split, overwrite=config.overwrite)
        ensure_clean_dir(labels_dir / split, overwrite=config.overwrite)

    LOGGER.info("Scanning JSON images from %s", config.json_path)
    source_index = build_source_index(config.dataset_root)
    for item in iter_json_items(config.json_path, "images", config.allow_stdlib_json):
        image_id = str(item.get("id", ""))
        if not image_id:
            stats.missing_images += 1
            continue
        source_path = resolve_image_path(config.dataset_root, item, source_index)
        if source_path is None:
            stats.missing_images += 1
            continue
        split = infer_split(item, source_path)
        if not should_select(split, counts, max_train, max_val):
            continue

        output_name = source_path.name
        record = ImageRecord(
            image_id=image_id,
            source_path=str(source_path),
            output_name=output_name,
            split=split,
            width=int(item["width"]),
            height=int(item["height"]),
            dataset=item.get("dataset"),
            original_data_source=item.get("original_data_source"),
        )
        selected[image_id] = record
        counts[split] += 1
        materialize_image(source_path, images_dir / split / output_name, config.link_mode)
        (labels_dir / split / f"{Path(output_name).stem}.txt").write_text("", encoding="utf-8")

        if counts["train"] and counts["train"] % 25_000 == 0:
            LOGGER.info("Selected %s train images", counts["train"])
        if counts["val"] and counts["val"] % 5_000 == 0:
            LOGGER.info("Selected %s val images", counts["val"])

        if max_train is not None and max_val is not None:
            if counts["train"] >= max_train and counts["val"] >= max_val:
                break

    stats.selected_train = counts["train"]
    stats.selected_val = counts["val"]
    LOGGER.info("Selected %s train and %s val images", counts["train"], counts["val"])
    return selected


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert a COCO xywh bbox in pixels to normalized YOLO cxcywh."""
    if len(bbox) != 4 or image_width <= 0 or image_height <= 0:
        return None
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0.0 or height <= 0.0:
        return None

    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 1.0 or clipped_height <= 1.0:
        return None

    cx = (x1 + clipped_width / 2.0) / image_width
    cy = (y1 + clipped_height / 2.0) / image_height
    return cx, cy, clipped_width / image_width, clipped_height / image_height


def append_annotations(
    config: PrepareDatasetConfig,
    selected: dict[str, ImageRecord],
    stats: PrepareStats,
) -> None:
    """Write YOLO labels for selected image records."""
    labels_dir = config.output / "labels"
    LOGGER.info("Streaming annotations")
    for annotation in iter_json_items(config.json_path, "annotations", config.allow_stdlib_json):
        image_id = str(annotation.get("image_id", ""))
        record = selected.get(image_id)
        if record is None:
            continue

        category_id = int(annotation.get("category_id", -1))
        if category_id == 0:
            stats.empty_annotations += 1
            continue
        if category_id != FISH_CATEGORY_ID:
            stats.skipped_annotations += 1
            continue

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list):
            stats.skipped_annotations += 1
            continue
        yolo_bbox = coco_bbox_to_yolo(bbox, record.width, record.height)
        if yolo_bbox is None:
            stats.invalid_bboxes += 1
            continue

        label_path = labels_dir / record.split / f"{Path(record.output_name).stem}.txt"
        with label_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{YOLO_CLASS_ID} "
                f"{yolo_bbox[0]:.10f} {yolo_bbox[1]:.10f} "
                f"{yolo_bbox[2]:.10f} {yolo_bbox[3]:.10f}\n"
            )
        stats.fish_annotations += 1
        if stats.fish_annotations % 100_000 == 0:
            LOGGER.info("Wrote %s fish annotations", stats.fish_annotations)


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO labels from a txt file."""
    labels: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        labels.append((int(parts[0]), *(float(value) for value in parts[1:])))
    return labels


def write_yolo_labels(label_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    """Write YOLO labels to a txt file."""
    lines = [
        f"{class_id} {cx:.10f} {cy:.10f} {width:.10f} {height:.10f}"
        for class_id, cx, cy, width, height in labels
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def create_augmented_training_samples(
    config: PrepareDatasetConfig,
    selected: dict[str, ImageRecord],
    stats: PrepareStats,
) -> None:
    """Create offline augmented images and labels for selected train images."""
    if not config.augment_train or config.aug_copies <= 0:
        return
    try:
        import cv2

        from scripts.augmentations import apply_yolo_augmentation
    except ImportError as exc:
        raise SystemExit(
            "Could not import augmentation dependencies. Check that this same Python "
            "environment can run: import cv2; import albumentations. "
            f"Original error: {exc}"
        ) from exc

    images_dir = config.output / "images" / "train"
    labels_dir = config.output / "labels" / "train"
    LOGGER.info("Creating %s Albumentations copies per train image", config.aug_copies)
    for record in selected.values():
        if record.split != "train":
            continue
        image_path = Path(record.source_path)
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Could not read image for augmentation: %s", image_path)
            continue
        label_path = labels_dir / f"{Path(record.output_name).stem}.txt"
        labels = read_yolo_labels(label_path)
        for copy_index in range(config.aug_copies):
            augmented_image, augmented_labels = apply_yolo_augmentation(
                image=image,
                labels=labels,
                image_size=config.imgsz,
            )
            stem = Path(record.output_name).stem
            suffix = Path(record.output_name).suffix
            output_name = f"{stem}_aug{copy_index + 1}{suffix}"
            cv2.imwrite(str(images_dir / output_name), augmented_image)
            write_yolo_labels(labels_dir / f"{Path(output_name).stem}.txt", augmented_labels)
            stats.augmented_images += 1
        if stats.augmented_images and stats.augmented_images % 5_000 == 0:
            LOGGER.info("Created %s augmented images", stats.augmented_images)


def create_calibration_subset(
    config: PrepareDatasetConfig,
    selected: dict[str, ImageRecord],
    stats: PrepareStats,
) -> None:
    """Create a small image subset for Hailo quantization calibration."""
    if config.calib_count <= 0:
        return
    ensure_clean_dir(config.calibration_output, overwrite=config.overwrite)
    train_records = [record for record in selected.values() if record.split == "train"]
    for record in train_records[: config.calib_count]:
        source = config.output / "images" / "train" / record.output_name
        destination = config.calibration_output / record.output_name
        materialize_image(source, destination, config.link_mode)
        stats.calibration_images += 1
    LOGGER.info("Created %s calibration images", stats.calibration_images)


def write_dataset_yaml(output: Path) -> None:
    """Write a YOLO dataset.yaml next to the prepared dataset."""
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "nc: 1",
                "names:",
                "  0: fish",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(output: Path, selected: dict[str, ImageRecord], stats: PrepareStats) -> None:
    """Write a JSON manifest for reproducibility."""
    manifest = {
        "stats": asdict(stats),
        "records": [asdict(record) for record in selected.values()],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_prepare_dataset(config: PrepareDatasetConfig = PREPARE_DATASET) -> PrepareStats:
    """Prepare images, labels, augmentations, and calibration subset."""
    config.output.mkdir(parents=True, exist_ok=True)

    stats = PrepareStats()
    selected = select_images(config, stats)
    if not selected:
        raise SystemExit("No images were selected. Check dataset paths and JSON mapping.")
    append_annotations(config, selected, stats)
    create_augmented_training_samples(config, selected, stats)
    create_calibration_subset(config, selected, stats)
    write_dataset_yaml(config.output)
    write_manifest(config.output, selected, stats)

    LOGGER.info("Done: %s", asdict(stats))
    return stats
