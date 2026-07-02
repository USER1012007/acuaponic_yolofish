# Hailo SDK Setup

## Requisitos
- Linux x86_64
- Python 3.10 para compilación HEF con DFC 3.x / Model Zoo v2.x
- Cuenta en el portal de desarrolladores de Hailo: https://developer.hailo.ai

## Wheels locales detectados

```text
hailo/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
hailo/hailort-4.24.0-cp311-cp311-linux_x86_64.whl
```

`hailort-4.24.0-cp311...whl` exige Python 3.11, pero HailoRT no es necesario
para compilar `.hef`; se usa en runtime/inferencia.

## Instalación limpia

```bash
bash scripts/setup_hailo_env.sh
conda activate fishbowl-hailo
```

Si OpenCV dentro de `hailomz` falla con `libgthread-2.0.so.0`, instala GLib
en el entorno:

```bash
conda activate fishbowl-hailo
conda install -c conda-forge libglib glib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Hailo Model Zoo

Para usar `hailomz compile`, además de los wheels necesitas Hailo Model Zoo.
La carpeta local `hailo/hailo_model_zoo` parece ser Model Zoo 5.3.0, pero tus
wheels son Dataflow Compiler 3.34.0 / HailoRT 4.24.0. Esa combinación no debe
mezclarse para Hailo-8.

Para Hailo-8, usa el tag release real `v2.19.0`. No uses la branch
`update-hailo8-link-v2.19`, porque conserva metadata de Model Zoo 5.3.0.

```bash
git -C hailo/hailo_model_zoo checkout v2.19.0
conda activate fishbowl-hailo
pip install --no-build-isolation -e hailo/hailo_model_zoo
pip check
hailomz --help
```

## Instalación de HailoRT en Raspberry Pi 5

1. Descargar el `.deb` de HailoRT desde el portal de Hailo
2. En la RPi:

```bash
sudo dpkg -i hailort_*.deb
pip install hailort  # Python bindings
```

## Versiones confirmadas
- Hailo Dataflow Compiler: 3.34.0
- HailoRT wheel local x86_64: 4.24.0
- Hailo Model Zoo: falta versión v2.x compatible para Hailo-8/DFC 3.x
