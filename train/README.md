# Acuaponic YOLO Fish Pipeline

This project automates the fine-tuning, conversion, and validation pipeline for YOLO models trained on fish detection data. It supports batch processing of multiple model versions and automated export to formats compatible with Raspberry Pi (via Hailo-8L).

## Prerequisites

- **Environment**: This project uses a Nix shell for dependencies. Run `nix-shell` to enter the environment.
- **Python**: Requirements are listed in `requirements.txt`.
- **Hardware**: For model compilation, ensure your Hailo compiler environment is set up.

## Configuration

All pipeline settings are centralized in `config.py`.

1. **Models to process**: Edit the `MODELS` list to include the YOLO models you want to train (e.g., `["yolov8n.pt", "yolov8s.pt"]`).
2. **Pipeline Toggles**: Enable or disable steps (Training, ONNX Export, HEF Compilation, Validation) in the `PipelineConfig` class.
3. **Paths**: `OUTPUT_DIR` defines where all artifacts are stored.

## Running the Pipeline

The `main.py` script orchestrates the full pipeline. It iterates through the configured `MODELS_TO_PROCESS`, automatically fetching them from Hugging Face if not present, and executes the enabled steps sequentially.

```bash
conda install -c conda-forge glib -y
conda activate fishbowl-hailo
pip install /ruta/a/hailo_model_zoo-2.19.0-py3-none-any.whl
pip install /ruta/a/hailo_dataflow_compiler-X.X.X-py3-none-any.whl
pip install -r requirements.txt
conda install -c conda-forge glib -y

python main.py
```

## Project Structure

- `main.py`: Main entry point and orchestration loop.
- `config.py`: Centralized configuration.
- `scripts/`: Implementation details for training, exporting, and reporting.
- `output/`:
    - `onnx/`: Generated ONNX model files.
    - `hef/`: Compiled `.hef` files ready for deployment on Raspberry Pi.
- `runs/`: Training logs and model weights.
- `dataset/`: Contains dataset configuration and source files.

## Workflow Summary

1. **Dataset Preparation**: Prepares data and generates calibration images.
2. **Training**: Trains each model defined in `MODELS_TO_PROCESS`.
3. **ONNX Export**: Converts the best weights to ONNX.
4. **HEF Compilation**: Compiles the ONNX model to `.hef` format for Hailo-8L.
5. **Report Generation**: Computes metrics and generates visual validation reports.

All generated artifacts are organized by model name inside the `output/` directory for easy access.
