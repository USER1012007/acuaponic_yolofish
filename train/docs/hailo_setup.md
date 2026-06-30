# Hailo SDK Setup

## Requisitos
- Linux x86_64
- Python 3.10
- Cuenta en el portal de desarrolladores de Hailo: https://developer.hailo.ai

## Instalación del Hailo Dataflow Compiler

1. Descargar el paquete desde https://developer.hailo.ai (requiere registro)
2. Instalar `hailo_dataflow_compiler` siguiendo las instrucciones del portal
3. Activar el entorno `fishbowl-export` e instalar el wheel:

```bash
conda activate fishbowl-export
pip install hailo_dataflow_compiler-*.whl
```

## Instalación de HailoRT en Raspberry Pi 5

1. Descargar el `.deb` de HailoRT desde el portal de Hailo
2. En la RPi:

```bash
sudo dpkg -i hailort_*.deb
pip install hailort  # Python bindings
```

## Versiones confirmadas
<!-- Actualizar cuando se confirmen las versiones usadas -->
- Hailo Dataflow Compiler: pendiente
- HailoRT: pendiente
