# Formato del dataset

## Formato detectado

El archivo actual es:

```text
dataset/community_fish_detection_dataset.json
```

Tiene estructura COCO-like:

```json
{
  "images": [
    {
      "id": "string",
      "file_name": "string",
      "width": 1920,
      "height": 1080,
      "is_train": true,
      "dataset": "source_name",
      "original_data_source": "source_video_or_image"
    }
  ],
  "annotations": [
    {
      "id": "string_or_int",
      "image_id": "string",
      "category_id": 1,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 1, "name": "fish"},
    {"id": 0, "name": "empty"}
  ]
}
```

## Bboxes

Las bboxes vienen como `[x, y, width, height]` en píxeles absolutos.
`scripts/prepare_dataset.py` las convierte a YOLO normalizado:

```text
class_id cx cy w h
```

## Clases

- `category_id: 1`, `fish`: se convierte a clase YOLO `0`.
- `category_id: 0`, `empty`: no tiene bbox y se trata como background implícito.
  Para estas imágenes se crea un `.txt` vacío.

## Formato de salida esperado (YOLO txt)

Cada imagen `data/processed/images/{split}/imagen.jpg` debe tener un archivo
`data/processed/labels/{split}/imagen.txt` con una línea por objeto:

```
clase cx cy w h
```

Donde:
- `clase`: entero (`0 = fish`)
- `cx`, `cy`: centro de la bbox normalizado entre 0 y 1
- `w`, `h`: ancho y alto normalizados entre 0 y 1
