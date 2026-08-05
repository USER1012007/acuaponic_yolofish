"""Central project configuration for dataset prep, training, export, and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LOG_LEVEL = "INFO"
TRAIN_EPOCHS = 100
RUN_DIR = Path("runs/detect/runs/train/fish_yolov8n_pilot_ouyeah")


@dataclass(frozen=True)
class PrepareDatasetConfig:

    json_path: Path = Path("dataset/community_fish_detection_dataset.json")
    dataset_root: Path = Path("dataset")
    output: Path = Path("data/processed")
    calibration_output: Path = Path("data/calibration/images")
    pilot: bool = True
    max_train: int | None = 100_000
    max_val: int | None = 10_000
    calib_count: int = 1_024
    link_mode: str = "symlink"
    augment_train: bool = False
    aug_copies: int = 1
    imgsz: int = 640
    overwrite: bool = True
    allow_stdlib_json: bool = False


@dataclass(frozen=True)
class TrainConfig:

    model: Path = Path("model/yolov8n.pt")
    data: Path = Path("configs/dataset.yaml")
    output_dir: Path = Path("models")
    resume: bool = True
    resume_checkpoint: Path | None = RUN_DIR / "weights/last.pt"
    imgsz: int = 640
    epochs: int = TRAIN_EPOCHS
    batch: int = 8
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
    close_mosaic: int = 15
    mixup: float = 0.05
    copy_paste: float = 0.0
    project: Path = Path("runs/train")
    name: str = "fish_yolov8n_pilot_ouyeah"
    patience: int = 30
    seed: int = 42
    freeze: int = 0


@dataclass(frozen=True)
class ExportOnnxConfig:
    """Configuration for scripts/export_onnx.py."""

    weights: Path = RUN_DIR / "weights/last.pt"
    output: Path = Path("models/model.onnx")
    imgsz: int = 640
    opset: int = 11


@dataclass(frozen=True)
class ExportHefConfig:
    """Configuration for scripts/export_hef.py."""

    onnx: Path = Path("models/model.onnx")
    output: Path | None = None
    output_dir: Path = Path("hef")
    output_name_template: str = "model_hailo8l_e{epoch}.hef"
    calib_path: Path = Path("data/calibration/images")
    work_dir: Path | None = None
    work_dir_name_template: str = "hailo_compile_hailo8l_e{epoch}"
    epoch_source: Path = RUN_DIR / "results.csv"
    epoch_override: int | None = None
    hw_arch: str = "hailo8l"
    classes: int = 1
    model_name: str = "fish_yolov8n_hailo8l"
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

    weights: Path = RUN_DIR / "weights/best.pt"
    data: Path = Path("configs/dataset.yaml")
    run_dir: Path = RUN_DIR
    output: Path = Path("reports/validation")
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    group_size: int = 10
    tile_width: int = 384
    max_images: int | None = 1_000
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
