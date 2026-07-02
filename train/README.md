# fishbowl-yolo

Fine-tuning de YOLOv8n para detección de peces en peceras, con deployment en Raspberry Pi 5 + Hailo-8.

## Arquitectura del pipeline

```
JSON + imágenes
      ↓
main.py
      ↓
prepare_dataset.py         →  data/processed en formato YOLO txt
      ↓
train.py                   →  models/best.pt  (fine-tuning completo)
      ↓
export_onnx.py             →  models/model.onnx
      ↓
export_hef.py              →  models/model.hef  (requiere Hailo SDK en Linux x86)
      ↓
metrics_report.py          →  gráficas + PNGs comparativos de validación
```

## Estructura del repo

```
fishbowl-yolo/
├── main.py                 # único entrypoint del pipeline completo
├── config.py               # parámetros centrales de dataset, training, export y reportes
├── configs/
│   ├── dataset.yaml        # clases y rutas del dataset
│   └── hyperparams.yaml    # nota de compatibilidad; usar config.py
├── dataset/                # dataset original: JSON + train/ + val/ (gitignored)
├── data/
│   ├── calibration/        # subset para calibración Hailo (gitignored)
│   └── processed/          # splits train/val en formato YOLO (gitignored)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── docs/
│   ├── dataset_format.md   # estructura esperada del JSON de entrada
│   └── hailo_setup.md      # instalación del Hailo SDK y compilación .hef
├── inference/
│   ├── camera.py           # captura desde cámara RPi
│   └── infer.py            # pipeline de inferencia con HailoRT
├── models/                 # checkpoints .pt, .onnx, .hef (gitignored)
├── notebooks/
│   └── eda.ipynb           # exploración del dataset y métricas visuales
├── scripts/
│   ├── prepare_dataset.py  # JSON → YOLO txt + split train/val/test
│   ├── augmentations.py    # augmentations offline con Albumentations
│   ├── train.py            # fine-tuning YOLOv8n
│   ├── export_onnx.py      # .pt → .onnx
│   ├── export_hef.py       # .onnx → .hef (solo Linux x86 + Hailo SDK)
│   └── metrics_report.py   # gráficas y contact sheets de validación
└── requirements.txt        # dependencias Python del proyecto
```

## Setup

### Entornos limpios

```bash
bash scripts/setup_train_env.sh
conda activate fishbowl-train
```

Ese entorno es para dataset, entrenamiento, exportación ONNX y reportes.

Para Hailo, usa un entorno separado:

```bash
bash scripts/setup_hailo_env.sh
conda activate fishbowl-hailo
```

Ver [docs/environment_setup.md](docs/environment_setup.md) y [docs/hailo_setup.md](docs/hailo_setup.md).

## Uso

### Ejecutar pipeline completo

El JSON actual es COCO-like:
- `category_id: 1` = `fish`, con bbox `[x, y, w, h]` en píxeles.
- `category_id: 0` = `empty`, sin bbox. Se trata como background implícito.

Editar parámetros y pasos en `config.py`. `PipelineConfig` controla qué etapas se ejecutan.

```bash
python main.py
```

Para GPUs con menos VRAM, cambiar `TrainConfig.batch` en `config.py`: `16` para 5060 8GB o `8` para 3050 6GB.
Para verificar el comando HEF sin compilar, cambiar `ExportHefConfig.dry_run = True`.

Genera gráficas de `cls_loss`, `box_loss`, `dfl_loss`, precision, recall, mAP50,
mAP50-95, histograma de IoU y contact sheets de 10 frames de validación con
GT en verde y predicción en rojo.

## Notas

- El dataset (`dataset/`, `data/`) y los modelos (`models/`) están en `.gitignore` por su tamaño.
- El paso de compilación `.onnx → .hef` requiere el Hailo Dataflow Compiler instalado en Linux x86. Ver `docs/hailo_setup.md`.
- El entrenamiento no congela capas (`freeze: 0`) para hacer fine-tuning completo.
