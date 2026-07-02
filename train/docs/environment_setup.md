# Environment Setup

Use two separate environments. Do not mix Hailo compiler dependencies with the
training environment.

## 1. Training / Dataset / ONNX

```bash
bash scripts/setup_train_env.sh
conda activate fishbowl-train
python main.py
```

This environment uses `requirements.txt` and is for:

- dataset conversion
- Albumentations augmentation
- YOLO training
- ONNX export
- validation plots/reports

## 2. Hailo HEF Compilation

```bash
bash scripts/setup_hailo_env.sh
conda activate fishbowl-hailo
```

This environment is only for:

```text
models/model.onnx -> models/model.hef
```

Current local Hailo wheels:

```text
hailo/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
hailo/hailort-4.24.0-cp311-cp311-linux_x86_64.whl
```

The HEF compile environment uses Python 3.10 and installs only the Dataflow
Compiler wheel. The local HailoRT wheel is `cp311`; it is not installed there
because HailoRT is not required for compilation.

Important: `hailo/hailo_model_zoo` must be on the actual `v2.19.0` release tag.
Do not use `update-hailo8-link-v2.19`; that branch updates docs/links but still
contains Model Zoo 5.3.0 setup metadata.

Use:

```bash
git -C hailo/hailo_model_zoo checkout v2.19.0
```

Then install and verify:

```bash
conda activate fishbowl-hailo
pip install --no-build-isolation -e hailo/hailo_model_zoo
pip check
hailo --help
hailomz --help
```

If `pip check` reports a conflict between `hailo-dataflow-compiler` and
`hailo_model_zoo`, the Model Zoo version is not compatible with the installed
DFC wheel.

Then run only the HEF step by editing `config.py`:

```python
PIPELINE = PipelineConfig(
    prepare_dataset=False,
    train=False,
    export_onnx=False,
    export_hef=True,
    report=False,
)
```

and run:

```bash
python main.py
```
