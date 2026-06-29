# Formato del dataset

## Pendiente de confirmar con el contacto

Este documento debe completarse una vez que llegue el dataset.

## Preguntas abiertas

1. ¿En qué formato vienen las bboxes en el JSON?
   - `[x, y, w, h]` en píxeles absolutos
   - `[x1, y1, x2, y2]` en píxeles absolutos
   - `[cx, cy, w, h]` normalizado

2. ¿Las coordenadas ya están normalizadas (0-1) o en píxeles absolutos?

3. ¿El JSON sigue algún estándar conocido (COCO, Pascal VOC, etc.)?

4. ¿La clase `no-pez` (1) debe entrenarse explícitamente o ignorarse?

## Formato de salida esperado (YOLO txt)

Cada imagen `data/processed/images/{split}/imagen.jpg` debe tener un archivo
`data/processed/labels/{split}/imagen.txt` con una línea por objeto:

```
clase cx cy w h
```

Donde:
- `clase`: entero (0 = pez, 1 = no-pez)
- `cx`, `cy`: centro de la bbox normalizado entre 0 y 1
- `w`, `h`: ancho y alto normalizados entre 0 y 1
