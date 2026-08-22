from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUTPUT_DIR = Path("output")
MODELS_DIR = Path("models")
CONFIGS_DIR = Path("configs")
DATA_YAML = CONFIGS_DIR / "dataset.yaml"

MODELS = [
    "yolov8n.pt",
    "yolov9n.pt",
    "yolov10n.pt",
    "yolo11n.pt",
]

def get_model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name

def get_model_output_dir(model_name: str) -> Path:
    return OUTPUT_DIR / model_name.replace(".pt", "")

TRAIN_EPOCHS = 1
LOG_LEVEL = "INFO"


@dataclass(frozen=False)
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

@dataclass(frozen=False)
class TrainConfig:
    model: Path = Path("") 
    data: Path = Path("configs/dataset.yaml")
    output_dir: Path = OUTPUT_DIR / "models"
    data: Path = DATA_YAML
    project: Path = OUTPUT_DIR
    name: str = "fish_model"
    imgsz: int = 640
    epochs: int = TRAIN_EPOCHS
    batch: int = 8
    workers: int = 8
    device: int | str = 0
    optimizer: str = "AdamW"
    patience: int = 30
    seed: int = 42


@dataclass(frozen=False)
class ExportOnnxConfig:
    weights: Path = Path("models/model.pt")
    output: Path = Path("models/model.onnx")
    imgsz: int = 640
    opset: int = 11
    simplify: bool = False


@dataclass(frozen=False)
class ExportHefConfig:
    onnx: Path = Path("runs/detect/output/")
    output: Path | None = None
    output_dir: Path = OUTPUT_DIR / "hef"
    calib_path: Path = Path("data/calibration/images")
    hw_arch: str = "hailo8l"
    classes: int = 1
    dry_run: bool = False
    epoch_override: int | None = None
    epoch_source: Path = Path("results.csv")
    output_name_template: str = "model.hef"
    work_dir: Path | None = None
    work_dir_name_template: str = "hailo_compile"
    model_name: str = "yolov8n"
    command_template: str = (
        "hailomz compile {model_name} "
        "--ckpt {onnx} "
        "--hw-arch {hw_arch} "
        "--calib-path {calib_path} "
        "--classes {classes} "
        "--performance"
    )
    start_node_names: tuple[str, ...] = ()
    end_node_names: tuple[str, ...] = ()


@dataclass(frozen=False)
class ReportConfig:
    weights: Path = Path("models/model.pt")
    run_dir: Path = OUTPUT_DIR
    data: Path = Path("configs/dataset.yaml")
    output: Path = Path("reports/validation")
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.5
    skip_model_val: bool = False
    skip_prediction_images: bool = False
    max_images: int | None = None
    group_size: int = 25
    tile_width: int = 320


@dataclass(frozen=False)
class PipelineConfig:
    prepare_dataset: bool = False
    train: bool = False
    export_onnx: bool = True
    export_hef: bool = True
    report: bool = True


PREPARE_DATASET = PrepareDatasetConfig()
TRAINING = TrainConfig()
EXPORT_ONNX = ExportOnnxConfig()
EXPORT_HEF = ExportHefConfig()
REPORT = ReportConfig()
PIPELINE = PipelineConfig()
