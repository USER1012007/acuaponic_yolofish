# CODEX — fishbowl-yolo

Instrucciones para agentes de IA asistiendo en este proyecto.
Leer completo antes de escribir cualquier línea de código.

---

## 1. Contexto del proyecto

Sistema de detección de peces en peceras basado en YOLOv8n fine-tuneado,
con deployment en Raspberry Pi 5 (8 GB RAM) + acelerador Hailo-8.

El repositorio cubre exclusivamente:
- Preparación del dataset
- Fine-tuning del modelo
- Exportación del modelo (ONNX → HEF)
- Pipeline de inferencia en la Raspberry Pi

No cubre integración con otros sistemas, UI, ni base de datos.

### Stack

| Componente | Tecnología |
|---|---|
| Modelo base | YOLOv8n (Ultralytics) |
| Entrenamiento | PyTorch + Ultralytics, GPU local |
| Formato intermedio | ONNX |
| Formato de deployment | HEF (Hailo) |
| Runtime en Pi | HailoRT + hailo-python-api |
| Entornos | Conda (entrenamiento/exportación), pip (Pi) |

### Entornos por máquina

- **Entrenamiento** (`environment-train.yml`): Windows (Darío) o Linux Nix (compañero), GPU local
- **Exportación HEF** (`environment-export.yml`): solo Linux x86 — le toca al compañero
- **Inferencia** (`requirements-rpi.txt`): Raspberry Pi 5, sin conda

---

## 2. División de responsabilidades

Hay dos roles en este proyecto. Cada integrante debe identificar cuál le corresponde
y respetar los archivos del otro.

---

### Rol: `dataset-inference`

Responsable de la preparación del dataset y el pipeline de inferencia en la Pi.
Entorno de trabajo: Windows o Linux, GPU local.

**Archivos propios:**
```
scripts/prepare_dataset.py
configs/dataset.yaml
configs/hyperparams.yaml   ← define los valores iniciales
inference/infer.py
inference/camera.py
requirements-rpi.txt
docs/dataset_format.md
README.md                  ← secciones: Dataset y Deployment
```

**No tocar sin coordinación con `training-export`:**
```
scripts/train.py
scripts/export_onnx.py
scripts/export_hef.py
scripts/validate.py
environment-train.yml
environment-export.yml
docs/hailo_setup.md
```

---

### Rol: `training-export`

Responsable del entrenamiento del modelo y la exportación a HEF.
Entorno de trabajo: Linux x86 (único entorno compatible con el Hailo SDK).

**Archivos propios:**
```
scripts/train.py
scripts/export_onnx.py
scripts/export_hef.py
scripts/validate.py
environment-train.yml
environment-export.yml
docs/hailo_setup.md
notebooks/eda.ipynb
README.md                  ← secciones: Entrenamiento y Exportación
```

**No tocar sin coordinación con `dataset-inference`:**
```
scripts/prepare_dataset.py
configs/dataset.yaml
inference/infer.py
inference/camera.py
requirements-rpi.txt
docs/dataset_format.md
```

---

## 3. Flujo de trabajo obligatorio antes de escribir código

Seguir este orden en cada petición, sin saltarse pasos.

### Paso 1 — Leer antes de escribir

Antes de modificar cualquier archivo existente:
1. Leer el archivo completo
2. Identificar qué funciones/clases ya existen
3. Verificar si ya existe lógica que resuelve (parcialmente) la tarea

Nunca asumir que un archivo está vacío o que una función no existe.

### Paso 2 — Entender el contrato del módulo

Identificar:
- ¿Qué recibe este script como entrada? (argumentos CLI, archivos, variables)
- ¿Qué produce como salida? (archivos, return values, efectos)
- ¿De qué otros módulos depende?

### Paso 3 — Planear antes de implementar

Para cualquier cambio no trivial (más de ~20 líneas):
1. Escribir en comentarios el esqueleto de la solución
2. Identificar casos borde (archivos corruptos, directorios vacíos, clases desbalanceadas)
3. Solo entonces implementar

### Paso 4 — Convenciones de código

**Python:**
- Versión mínima: 3.10
- Type hints en todas las funciones
- Docstrings en funciones públicas (formato Google style)
- Constantes en UPPER_SNAKE_CASE al inicio del archivo
- Sin lógica en el nivel de módulo — todo dentro de funciones o `if __name__ == "__main__"`
- Argumentos CLI via `argparse`, no hardcodeados

**Archivos:**
- Un archivo = una responsabilidad clara
- No crear archivos nuevos sin que estén en la estructura definida en el README
- Si se necesita un archivo nuevo, documentarlo en el README antes de crearlo

**Imports:**
- Stdlib primero, luego third-party, luego locales
- Sin imports con wildcard (`from x import *`)

### Paso 5 — Manejo de errores

Todo I/O debe tener manejo explícito de errores:
- Verificar existencia de archivos antes de abrirlos
- Mensajes de error descriptivos con la ruta del archivo problemático
- En scripts de preparación de datos: loggear imágenes/anotaciones saltadas, no silenciarlas

### Paso 6 — No romper lo que ya funciona

Antes de modificar una función existente:
1. Entender por qué está escrita así
2. Si el cambio afecta la interfaz (argumentos, salida), coordinarlo con la otra persona
3. Nunca cambiar el formato de salida de `prepare_dataset.py` sin avisar — train.py depende de él

---

## 4. Contratos de interfaz entre módulos

Estos contratos son fijos. Cambiarlos requiere coordinación entre ambos.

### `prepare_dataset.py` → `train.py`

Salida esperada en `data/processed/`:
```
images/
  train/   *.jpg
  val/     *.jpg
  test/    *.jpg
labels/
  train/   *.txt  (una línea por objeto: "clase cx cy w h")
  val/     *.txt
  test/    *.txt
```

Formato de cada línea en los `.txt`:
```
<int:clase> <float:cx> <float:cy> <float:w> <float:h>
```
Todos los valores de bbox normalizados entre 0.0 y 1.0.

### `train.py` → `export_onnx.py`

- Checkpoint de salida: `models/best.pt`
- Debe ser el mejor checkpoint por mAP50 en validación, no el último epoch

### `export_onnx.py` → `export_hef.py`

- Salida: `models/model.onnx`
- opset_version: 11 (requerido por el Hailo Dataflow Compiler)
- Input shape fijo: `(1, 3, 640, 640)`

### `export_hef.py` → `infer.py`

- Salida: `models/model.hef`
- `infer.py` asume que el modelo tiene exactamente una clase de salida de interés: `pez` (índice 0)

---

## 5. Incógnitas abiertas (no asumir, preguntar)

Antes de implementar lógica que dependa de estos puntos, confirmar con el contacto:

1. **Estructura del JSON del dataset** — no se conoce hasta que llegue el dataset.
   Ver `docs/dataset_format.md` para las preguntas específicas.

2. **Clase `no-pez`** — no está claro si debe entrenarse como clase explícita
   o tratarse como background implícito. Afecta `prepare_dataset.py` y `dataset.yaml`.

3. **Versiones del Hailo SDK** — documentar en `docs/hailo_setup.md` una vez instalado.

4. **Requisitos de latencia en Pi** — define si YOLOv8n es suficiente o se necesita
   un modelo más grande/pequeño.

---

## 6. Lo que este agente NO debe hacer

- No entrenar el modelo (proceso largo, corre en GPU local)
- No modificar archivos fuera de la responsabilidad asignada a quien invoca el agente
- No hardcodear rutas absolutas
- No instalar dependencias fuera de los archivos de entorno definidos
- No crear archivos nuevos sin que estén documentados en el README
- No cambiar el formato de salida de ningún script sin actualizar este CODEX y el README
