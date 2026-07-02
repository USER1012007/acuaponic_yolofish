"""Single entrypoint for the complete fish YOLO fine-tuning pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict

from config import EXPORT_HEF, EXPORT_ONNX, LOG_LEVEL, PIPELINE, PREPARE_DATASET, REPORT, TRAINING
from scripts.export_hef import run_export_hef
from scripts.export_onnx import run_export_onnx
from scripts.metrics_report import run_report
from scripts.prepare_dataset import run_prepare_dataset
from scripts.train import run_training


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure process logging once for the whole pipeline."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    """Run the full pipeline according to config.py."""
    configure_logging()
    LOGGER.info("Starting pipeline: %s", asdict(PIPELINE))

    if PIPELINE.prepare_dataset:
        LOGGER.info("Step 1/5: prepare dataset")
        run_prepare_dataset(PREPARE_DATASET)

    if PIPELINE.train:
        LOGGER.info("Step 2/5: train model")
        run_training(TRAINING)

    if PIPELINE.export_onnx:
        LOGGER.info("Step 3/5: export ONNX")
        run_export_onnx(EXPORT_ONNX)

    if PIPELINE.export_hef:
        LOGGER.info("Step 4/5: export HEF")
        run_export_hef(EXPORT_HEF)

    if PIPELINE.report:
        LOGGER.info("Step 5/5: generate validation report")
        run_report(REPORT)

    LOGGER.info("Pipeline finished")


if __name__ == "__main__":
    main()
