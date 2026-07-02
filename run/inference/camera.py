"""USB webcam capture utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    device_index: int
    width: int
    height: int
    fps: int


class USBCamera:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._config.device_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("La camara no esta abierta. Llama open() primero.")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo leer un frame desde la webcam USB.")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "USBCamera":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
