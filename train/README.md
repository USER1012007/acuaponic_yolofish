# fishbowl-yolo

Fine-tuning de YOLOv8n para detección de peces en peceras, con deployment en Raspberry Pi 5 + Hailo-8.

## Arquitectura del pipeline

```
JSON + imágenes
      ↓
scripts/prepare_dataset.py  →  splits en formato YOLO txt
      ↓
scripts/train.py            →  best.pt  (GPU: 4070 PC o 5060 laptop)
      ↓
scripts/export_onnx.py      →  model.onnx
      ↓
scripts/export_hef.py       →  model.hef  (requiere Hailo SDK en Linux x86)
      ↓
inference/infer.py + camera.py  [Raspberry Pi 5 + Hailo-8]
```

## Estructura del repo

```
fishbowl-yolo/
├── configs/
│   ├── dataset.yaml        # clases y rutas del dataset
│   └── hyperparams.yaml    # hiperparámetros de entrenamiento
├── data/
│   ├── raw/                # dataset original (gitignored)
│   └── processed/          # splits train/val/test en formato YOLO (gitignored)
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
│   ├── train.py            # fine-tuning YOLOv8n
│   ├── export_onnx.py      # .pt → .onnx
│   ├── export_hef.py       # .onnx → .hef (solo Linux x86 + Hailo SDK)
│   └── validate.py         # evaluación de métricas sobre val/test
├── environment-train.yml   # entorno conda para entrenamiento (Windows/Linux GPU)
├── environment-export.yml  # entorno conda para exportación Hailo (Linux x86)
└── requirements-rpi.txt    # dependencias para Raspberry Pi
```

## Setup

### Entorno de entrenamiento (Windows con GPU o Linux)

```bash
conda env create -f environment-train.yml
conda activate fishbowl-train
```

### Entorno de exportación Hailo (Linux x86 — compañero)

```bash
conda env create -f environment-export.yml
conda activate fishbowl-export
# Instalar Hailo Dataflow Compiler manualmente — ver docs/hailo_setup.md
```

### Raspberry Pi 5

```bash
pip install -r requirements-rpi.txt
# HailoRT se instala a nivel sistema — ver docs/hailo_setup.md
```

## Uso

### 1. Preparar el dataset

```bash
python scripts/prepare_dataset.py \
  --json ruta/al/dataset.json \
  --images ruta/a/las/imagenes/ \
  --output data/processed/
```

### 2. Entrenar

```bash
python scripts/train.py --config configs/hyperparams.yaml
```

### 3. Exportar a ONNX

```bash
python scripts/export_onnx.py --weights models/best.pt
```

### 4. Compilar a HEF (Linux x86 + Hailo SDK)

```bash
python scripts/export_hef.py --onnx models/model.onnx
```

### 5. Inferencia en Raspberry Pi

```bash
python inference/infer.py --model models/model.hef --source 0
```

## Notas

- El dataset (`data/`) y los modelos (`models/`) están en `.gitignore` por su tamaño.
- El paso de compilación `.onnx → .hef` requiere el Hailo Dataflow Compiler instalado en Linux x86. Ver `docs/hailo_setup.md`.
- Confirmar si la clase `no-pez` debe entrenarse explícitamente o tratarse como background implícito.
