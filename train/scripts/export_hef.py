"""Compile an ONNX YOLO model to Hailo HEF."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPORT_HEF, ExportHefConfig

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure process logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def validate_inputs(onnx: Path, calib_path: Path) -> None:
    """Validate compiler inputs."""
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration image directory not found: {calib_path}")


def build_default_command(config: ExportHefConfig) -> list[str]:
    """Build the default Hailo Model Zoo compile command."""
    return [
        "hailomz",
        "compile",
        "yolov8n",
        "--ckpt",
        str(config.onnx),
        "--hw-arch",
        config.hw_arch,
        "--calib-path",
        str(config.calib_path),
        "--classes",
        str(config.classes),
        "--performance",
        "--output-dir",
        str(config.work_dir),
    ]


def build_custom_command(config: ExportHefConfig) -> list[str]:
    """Build a custom compiler command from a format string."""
    if config.command_template is None:
        return build_default_command(config)
    rendered = config.command_template.format(
        onnx=config.onnx,
        output=config.output,
        calib_path=config.calib_path,
        work_dir=config.work_dir,
        hw_arch=config.hw_arch,
        classes=config.classes,
        model_name=config.model_name,
    )
    return rendered.split()


def find_hef(work_dir: Path) -> Path:
    """Find the newest HEF created by the compiler."""
    hefs = sorted(work_dir.rglob("*.hef"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not hefs:
        raise FileNotFoundError(f"No .hef file was produced under {work_dir}")
    return hefs[0]


def compile_hef(config: ExportHefConfig) -> Path:
    """Compile ONNX to HEF and copy the artifact to the requested output."""
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)

    command = build_custom_command(config)
    LOGGER.info("Compiler command: %s", " ".join(command))
    if config.dry_run:
        return config.output

    validate_inputs(config.onnx, config.calib_path)

    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(
            f"Compiler executable not found: {command[0]}. "
            "Install Hailo Dataflow Compiler / Hailo Model Zoo in the export environment."
        )

    subprocess.run(command, check=True)
    hef_path = find_hef(config.work_dir)
    shutil.copy2(hef_path, config.output)
    LOGGER.info("HEF exported to %s", config.output)
    return config.output


def run_export_hef(config: ExportHefConfig = EXPORT_HEF) -> Path:
    """Run HEF compilation."""
    return compile_hef(config)
