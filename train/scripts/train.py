"""Train YOLOv8n for fish detection."""

from __future__ import annotations

from dataclasses import asdict
import logging
import shutil
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAINING, TrainConfig

LOGGER = logging.getLogger(__name__)
TRAIN_KEYS = {
    "data",
    "imgsz",
    "epochs",
    "batch",
    "workers",
    "device",
    "optimizer",
    "lr0",
    "weight_decay",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "fliplr",
    "mosaic",
    "close_mosaic",
    "mixup",
    "copy_paste",
    "project",
    "name",
    "patience",
    "seed",
    "freeze",
}


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def validate_paths(config: TrainConfig) -> None:
    """Validate required input files."""
    if not config.model.exists():
        raise FileNotFoundError(f"YOLO base weights not found: {config.model}")
    if not config.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config.data}")


def build_train_args(config: TrainConfig) -> dict[str, Any]:
    """Build Ultralytics train kwargs from central config."""
    raw_config = asdict(config)
    train_args = {key: value for key, value in raw_config.items() if key in TRAIN_KEYS}
    train_args["data"] = str(config.data)
    train_args["project"] = str(config.project)
    train_args["freeze"] = 0
    train_args["plots"] = True
    train_args["val"] = True
    train_args["save"] = True
    train_args["exist_ok"] = True
    return train_args


def resume_checkpoint(config: TrainConfig) -> Path:
    """Return the expected Ultralytics checkpoint for resuming a run."""
    if config.resume_checkpoint is not None:
        return config.resume_checkpoint
    return config.project / config.name / "weights" / "last.pt"


def train_model(config: TrainConfig) -> Path:
    """Run Ultralytics training and return the best checkpoint path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Activate the training environment "
            "or install it with: pip install ultralytics"
        ) from exc

    train_args = build_train_args(config)

    model_path = config.model
    if config.resume:
        checkpoint = resume_checkpoint(config)
        if checkpoint.exists():
            model_path = checkpoint
            train_args["resume"] = True
            LOGGER.info("Resuming training from %s", checkpoint)
        else:
            LOGGER.warning("Resume requested but checkpoint was not found: %s", checkpoint)
            LOGGER.warning("Starting a new fine-tuning run from %s", config.model)
    else:
        LOGGER.info("Starting full fine-tuning from %s", config.model)

    model = YOLO(str(model_path))
    results = model.train(**train_args)

    save_dir = Path(getattr(results, "save_dir", train_args.get("project", "runs/train")))
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Ultralytics did not create best checkpoint: {best_path}")
    LOGGER.info("Best checkpoint: %s", best_path)
    return best_path


def copy_best_checkpoint(best_path: Path, output_dir: Path) -> Path:
    """Copy best.pt to the stable models directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best.pt"
    shutil.copy2(best_path, destination)
    LOGGER.info("Copied best checkpoint to %s", destination)
    return destination


def run_training(config: TrainConfig = TRAINING) -> Path:
    """Run training."""
    validate_paths(config)
    best_path = train_model(config)
    return copy_best_checkpoint(best_path, config.output_dir)
