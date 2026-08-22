"""Single entrypoint for the complete fish YOLO fine-tuning pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace

from config import (
    EXPORT_HEF, EXPORT_ONNX, LOG_LEVEL, PIPELINE, PREPARE_DATASET,
    REPORT, TRAINING, MODELS, OUTPUT_DIR, MODELS_DIR, get_model_path, get_model_output_dir
)
from scripts.export_hef import run_export_hef
from scripts.export_onnx import run_export_onnx
from scripts.metrics_report import run_report
from scripts.prepare_dataset import run_prepare_dataset
from scripts.train import run_training
import time
from ultralytics.utils.downloads import attempt_download_asset

def ensure_weights(model_name: str, models_dir, retries: int = 3, wait: float = 10.0):
    """Download base weights into models_dir if missing, with retries."""
    dest = models_dir / model_name
    if dest.exists():
        return dest
    models_dir.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info("Downloading %s (intento %s/%s)", model_name, attempt, retries)
            attempt_download_asset(str(dest))
            if dest.exists():
                return dest
        except Exception as exc:
            last_exc = exc
            LOGGER.warning("Fallo descarga %s intento %s: %s", model_name, attempt, exc)
            time.sleep(wait)
    raise RuntimeError(f"No se pudo descargar {model_name} tras {retries} intentos") from last_exc

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure process logging once for the whole pipeline."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    """Run the full pipeline according to config.py for multiple models."""
    configure_logging()
    LOGGER.info("Starting pipeline: %s", asdict(PIPELINE))

    if PIPELINE.prepare_dataset:
        LOGGER.info("Step 1: prepare dataset")
        run_prepare_dataset(PREPARE_DATASET)

    for model_name in MODELS:
        try:
            LOGGER.info(">>> Processing model: %s", model_name)
            model_path = ensure_weights(model_name, MODELS_DIR)
            model_out = get_model_output_dir(model_name)
            weights_dir = model_out / "weights"

            current_training = replace(
                TRAINING, model=model_path, project=OUTPUT_DIR,
                name=model_out.name, output_dir=model_out,
            )

            best_weights = None
            if PIPELINE.train:
                best_weights = run_training(current_training)
            else:
                best_weights = weights_dir / "best.pt"

            current_onnx = replace(EXPORT_ONNX, weights=weights_dir / "last.pt", output=model_out / "best.onnx")
            hef_model_name = model_name.replace(".pt", "")
            current_hef = replace(
                EXPORT_HEF, onnx=current_onnx.output, output_dir=model_out,
                epoch_source=model_out / "results.csv", model_name=hef_model_name,
            )
            current_report = replace(REPORT, weights=best_weights, run_dir=model_out, output=model_out / "stats")

            if PIPELINE.export_onnx:
                run_export_onnx(current_onnx)
            if PIPELINE.export_hef:
                run_export_hef(current_hef)
            if PIPELINE.report:
                run_report(current_report)

            LOGGER.info(">>> %s done", model_name)
        except Exception:
            LOGGER.exception(">>> %s FAILED, seguimos con el siguiente modelo", model_name)
            continue

    LOGGER.info("Pipeline finished for all models")


if __name__ == "__main__":
    main()
