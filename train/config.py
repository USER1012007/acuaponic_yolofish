"""Central project configuration for dataset prep, training, export, and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class PrepareDatasetConfig:
    """Configuration for scripts/prepare_dataset.py."""

    json_path: Path = Path("dataset/community_fish_detection_dataset.json")
    dataset_root: Path = Path("dataset")
    output: Path = Path("data/processed")
    calibration_output: Path = Path("data/calibration/images")
    pilot: bool = True
    max_train: int | None = 2_000
    max_val: int | None = 300
    calib_count: int = 1_024
    link_mode: str = "symlink"
    augment_train: bool = True
    aug_copies: int = 1
    imgsz: int = 640
    overwrite: bool = True
    allow_stdlib_json: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for scripts/train.py."""

    model: Path = Path("model/yolov8n.pt")
    data: Path = Path("configs/dataset.yaml")
    output_dir: Path = Path("models")
    resume: bool = False
    imgsz: int = 640
    epochs: int = 5
    batch: int = 16
    workers: int = 8
    device: int | str = 0
    optimizer: str = "AdamW"
    lr0: float = 0.001
    weight_decay: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 10
    mixup: float = 0.05
    copy_paste: float = 0.0
    project: Path = Path("runs/train")
    name: str = "fish_yolov8n_pilot"
    patience: int = 15
    seed: int = 42
    freeze: int = 0


@dataclass(frozen=True)
class ExportOnnxConfig:
    """Configuration for scripts/export_onnx.py."""

    weights: Path = Path("models/best.pt")
    output: Path = Path("models/model.onnx")
    imgsz: int = 640
    opset: int = 11


@dataclass(frozen=True)
class ExportHefConfig:
    """Configuration for scripts/export_hef.py."""

    onnx: Path = Path("models/model.onnx")
    output: Path = Path("models/model.hef")
    calib_path: Path = Path("data/calibration/images")
    work_dir: Path = Path("models/hailo_compile")
    hw_arch: str = "hailo8"
    classes: int = 1
    model_name: str = "fish_yolov8n"
    dry_run: bool = False
    command_template: str | None = None
    start_node_names: tuple[str, ...] = ()
    end_node_names: tuple[str, ...] = (
        "/model.22/cv2.0/cv2.0.2/Conv",
        "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv",
        "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
    )


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for validation reports."""

    weights: Path = Path("models/best.pt")
    data: Path = Path("configs/dataset.yaml")
    run_dir: Path = Path("runs/train/fish_yolov8n_pilot")
    output: Path = Path("reports/validation")
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    group_size: int = 10
    tile_width: int = 384
    max_images: int | None = 300
    skip_model_val: bool = False
    skip_prediction_images: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for main.py orchestration."""

    prepare_dataset: bool = False
    train: bool = False
    export_onnx: bool = False
    export_hef: bool = True
    report: bool = False


PREPARE_DATASET = PrepareDatasetConfig()
TRAINING = TrainConfig()
EXPORT_ONNX = ExportOnnxConfig()
EXPORT_HEF = ExportHefConfig()
REPORT = ReportConfig()
PIPELINE = PipelineConfig()
