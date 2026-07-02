from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HailoModelConfig:

    hef_path: Path
    input_format: str


class HailoDetector:

    def __init__(self, config: HailoModelConfig) -> None:
        if not config.hef_path.exists():
            raise FileNotFoundError(f"No existe el modelo bro: {config.hef_path}")

        self._config = config
        self._hailo: dict[str, Any] = {}
        self._input_name: str | None = None

    def __enter__(self) -> "HailoDetector":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def open(self) -> None:
        try:
            from hailo_platform import ( 
                ConfigureParams,
                FormatType,
                HEF,
                HailoStreamInterface,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as error:
            raise RuntimeError(
                "No se pudo importar hailo_platform. Instala HailoRT y su Python API "
                "en la Raspberry Pi antes de correr inferencia."
            ) from error

        format_type = self._resolve_format_type(FormatType)
        hef = HEF(str(self._config.hef_path))
        vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = vdevice.configure(hef, configure_params)
        if not network_groups:
            raise RuntimeError(f"HailoRT no configuro ningun network group para {self._config.hef_path}")

        network_group = network_groups[0]
        network_group_params = network_group.create_params()
        input_params = InputVStreamParams.make(network_group, format_type=format_type)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_infos = hef.get_input_vstream_infos()
        if not input_infos:
            raise RuntimeError(f"El HEF no expone vstreams de entrada: {self._config.hef_path}")

        self._input_name = input_infos[0].name
        activation = network_group.activate(network_group_params)
        infer_pipeline = InferVStreams(network_group, input_params, output_params)
        activation.__enter__()
        infer_pipeline.__enter__()
        self._hailo = {
            "activation": activation,
            "infer_pipeline": infer_pipeline,
            "vdevice": vdevice,
        }

    def close(self) -> None:
        infer_pipeline = self._hailo.get("infer_pipeline")
        activation = self._hailo.get("activation")
        vdevice = self._hailo.get("vdevice")

        if infer_pipeline is not None:
            infer_pipeline.__exit__(None, None, None)
        if activation is not None:
            activation.__exit__(None, None, None)
        if vdevice is not None and hasattr(vdevice, "release"):
            vdevice.release()

        self._hailo = {}
        self._input_name = None

    def infer(self, input_batch: np.ndarray) -> dict[str, np.ndarray]:
        if not self._hailo or self._input_name is None:
            raise RuntimeError("HailoDetector no esta abierto. Llama open() primero.")

        infer_pipeline = self._hailo["infer_pipeline"]
        return infer_pipeline.infer({self._input_name: input_batch})

    def _resolve_format_type(self, format_type_enum: Any) -> Any:
        if self._config.input_format == "uint8":
            return format_type_enum.UINT8
        if self._config.input_format == "float32":
            return format_type_enum.FLOAT32
        raise ValueError(
            "model.input_format debe ser 'uint8' o 'float32', "
            f"recibido: {self._config.input_format}"
        )
