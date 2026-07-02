from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/inference.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre inferencia YOLOv8 con Hailo HEF y webcam USB.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Ruta al YAML de configuracion. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--hef",
        type=Path,
        default=None,
        help="Override para model.hef_path.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override para camera.device_index.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Override para output.json_path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Falta PyYAML. Instala las dependencias con: pip install -r requirements-rpi.txt"
        ) from error

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as error:
        raise OSError(f"No se pudo leer el archivo de configuracion: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"YAML invalido en {path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"La configuracion debe ser un mapa YAML: {path}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.hef is not None:
        config.setdefault("model", {})["hef_path"] = str(args.hef)
    if args.camera_index is not None:
        config.setdefault("camera", {})["device_index"] = args.camera_index
    if args.json is not None:
        config.setdefault("output", {})["json_path"] = str(args.json)
    return config


def build_camera_config(config: dict[str, Any]) -> CameraConfig:
    from inference.camera import CameraConfig

    camera = _required_section(config, "camera")
    return CameraConfig(
        device_index=int(camera.get("device_index", 0)),
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        fps=int(camera.get("fps", 30)),
    )


def build_hailo_config(config: dict[str, Any]) -> HailoModelConfig:
    from inference.hailo_detector import HailoModelConfig

    model = _required_section(config, "model")
    hef_path = model.get("hef_path")
    if not hef_path:
        raise ValueError("Falta model.hef_path en la configuracion.")
    return HailoModelConfig(
        hef_path=Path(str(hef_path)),
        input_format=str(model.get("input_format", "uint8")),
    )


def build_preprocess_config(config: dict[str, Any]) -> PreprocessConfig:
    from inference.postprocess import PreprocessConfig

    model = _required_section(config, "model")
    return PreprocessConfig(
        input_width=int(model.get("input_width", 640)),
        input_height=int(model.get("input_height", 640)),
        input_format=str(model.get("input_format", "uint8")),
        input_color=str(model.get("input_color", "rgb")),
    )


def build_postprocess_config(config: dict[str, Any]) -> PostprocessConfig:
    from inference.postprocess import PostprocessConfig

    postprocess = _required_section(config, "postprocess")
    return PostprocessConfig(
        mode=str(postprocess.get("mode", "auto")),
        confidence_threshold=float(postprocess.get("confidence_threshold", 0.35)),
        iou_threshold=float(postprocess.get("iou_threshold", 0.45)),
        max_detections=int(postprocess.get("max_detections", 100)),
        raw_has_objectness=bool(postprocess.get("raw_has_objectness", False)),
        nms_box_format=str(postprocess.get("nms_box_format", "yxyx")),
    )


def get_class_names(config: dict[str, Any]) -> list[str]:
    model = _required_section(config, "model")
    class_names = model.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("model.class_names debe ser una lista no vacia.")
    return [str(class_name) for class_name in class_names]


def run_inference(config: dict[str, Any]) -> None:
    import cv2

    from inference.camera import USBCamera
    from inference.hailo_detector import HailoDetector
    from inference.postprocess import draw_detections, postprocess_outputs, preprocess_frame

    camera_config = build_camera_config(config)
    hailo_config = build_hailo_config(config)
    preprocess_config = build_preprocess_config(config)
    postprocess_config = build_postprocess_config(config)
    class_names = get_class_names(config)
    output_config = _required_section(config, "output")
    window_name = str(output_config.get("window_name", "YOLOv8 Hailo"))
    json_path = Path(str(output_config.get("json_path", "runs/class_counts.json")))
    write_json_every_n_frames = max(1, int(output_config.get("write_json_every_n_frames", 1)))

    total_counts: Counter[str] = Counter()
    frame_index = 0

    with HailoDetector(hailo_config) as detector, USBCamera(camera_config) as camera:
        while True:
            frame = camera.read()
            input_batch, resize_info = preprocess_frame(frame, preprocess_config)
            outputs = detector.infer(input_batch)
            detections = postprocess_outputs(outputs, resize_info, class_names, postprocess_config)
            frame_index += 1

            last_frame_counts = _count_detections(detections)
            total_counts.update(last_frame_counts)
            if frame_index % write_json_every_n_frames == 0:
                write_counts_json(json_path, frame_index, total_counts, last_frame_counts)

            annotated = draw_detections(frame, detections)
            _draw_status(annotated, frame_index, detections)
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()
    write_counts_json(json_path, frame_index, total_counts, Counter())


def write_counts_json(
    json_path: Path,
    frame_index: int,
    total_counts: Counter[str],
    last_frame_counts: Counter[str],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frames": frame_index,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "class_counts": dict(sorted(total_counts.items())),
        "last_frame_counts": dict(sorted(last_frame_counts.items())),
    }
    temp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(json_path)
    except OSError as error:
        raise OSError(f"No se pudo escribir el JSON de conteos: {json_path}") from error


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_inference(config)


def _required_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Falta la seccion '{section}' en la configuracion.")
    return value


def _count_detections(detections: list[Any]) -> Counter[str]:
    return Counter(detection.class_name for detection in detections)


def _draw_status(frame: Any, frame_index: int, detections: list[Any]) -> None:
    import cv2

    label = f"Frame {frame_index} | Detecciones {len(detections)} | q/Esc salir"
    cv2.rectangle(frame, (8, 8), (420, 36), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
