"""Export the trained YOLO checkpoint to ONNX for Hailo compilation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPORT_ONNX, ExportOnnxConfig

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def export_onnx(weights: Path, output: Path, imgsz: int, opset: int) -> Path:
    """Export a YOLO checkpoint to ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ultralytics'. Activate the training/export environment "
            "or install it with: pip install ultralytics"
        ) from exc

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=False,
            simplify=True,
            nms=False,
            batch=1,
        )
    )
    if not exported.exists():
        raise FileNotFoundError(f"ONNX export did not produce a file: {exported}")
    if exported.resolve() != output.resolve():
        shutil.copy2(exported, output)
    LOGGER.info("ONNX exported to %s", output)
    return output


def run_export_onnx(config: ExportOnnxConfig = EXPORT_ONNX) -> Path:
    """Run ONNX export."""
    return export_onnx(config.weights, config.output, config.imgsz, config.opset)
