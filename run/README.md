# fishbowl-yolo

Base para deteccion con YOLOv8n en Raspberry Pi 5 + Hailo-8 usando webcam USB.

El objetivo inicial es correr un modelo YOLOv8n compatible con COCO en formato
`.hef`. Cuando el fine-tuning este listo, el pipeline de inferencia no cambia:
solo se reemplaza el `.hef` y se actualizan las clases en la configuracion.

## Estructura

```text
configs/
  inference.yaml        Configuracion del runtime de inferencia.
inference/
  camera.py             Captura de video desde webcam USB con OpenCV.
  hailo_detector.py     Wrapper de HailoRT para ejecutar modelos .hef.
  infer.py              CLI principal para correr inferencia manual.
  postprocess.py        Decodificacion YOLOv8/NMS y filtrado de detecciones.
  types.py              Tipos compartidos del pipeline.
models/
  .gitkeep              Carpeta para archivos .hef.
runs/
  .gitkeep              Carpeta para salidas locales de inferencia.
requirements-rpi.txt    Dependencias para Raspberry Pi 5 con Ubuntu.
```

## Deployment en Raspberry Pi

Este runtime asume Raspberry Pi 5 con Ubuntu, Hailo-8/Hailo-8L instalado y
HailoRT funcionando. El paquete `hailo_platform` debe venir de la instalacion
oficial de HailoRT para la arquitectura del dispositivo.

Instalacion base:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rpi.txt
```

Colocar el modelo `.hef` en:

```text
models/yolov8n_coco.hef
```

Ejecutar:

```bash
python -m inference.infer --config configs/inference.yaml
```

Controles:

- `q` o `Esc`: cerrar la ventana.

## Configuracion del modelo

Para COCO, `configs/inference.yaml` contiene las 80 clases estandar. Para el
modelo fine-tuned:

1. Convertir `models/best.pt` a ONNX con input fijo `(1, 3, 640, 640)`.
2. Compilar ONNX a `models/model.hef` en Linux x86 con Hailo SDK.
3. Cambiar `model.hef_path` en `configs/inference.yaml`.
4. Cambiar `model.class_names` para que coincida con el modelo exportado.

## Salida JSON

El script mantiene un archivo JSON con conteo acumulado por clase:

```json
{
  "frames": 120,
  "updated_at": "2026-07-01T12:00:00Z",
  "class_counts": {
    "person": 4,
    "cup": 2
  }
}
```

El conteo se actualiza por frame usando las detecciones que pasen los umbrales
configurados.
