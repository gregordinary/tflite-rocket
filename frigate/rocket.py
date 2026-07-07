# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""
rocket.py — Frigate detector plugin for the RK3588 NPU via the tflite-rocket
external delegate.

Drop this file into Frigate's ``frigate/detectors/plugins/`` directory (the bundled
Dockerfile does this) and it auto-registers a ``rocket`` detector: Frigate's plugin
loader imports every module under that package and indexes ``DetectionApi``
subclasses by ``type_key``, and builds the config union from
``BaseDetectorConfig`` subclasses — so no other Frigate change is needed.

The plugin is a thin adapter. It loads an unmodified SSD ``.tflite`` (Frigate's
default SSDLite-MobileDet) with the tflite-rocket delegate
(``libtflite_rocket.so``), which runs the convolutional backbone on the NPU; the
SSD ``TFLite_Detection_PostProcess`` op is a custom op and stays on the CPU.
Detection output is byte-for-byte Frigate's standard SSD path
(``frigate.detectors.detector_utils.tflite_detect_raw``), so the detector behaves
exactly like the built-in ``cpu``/``edgetpu`` SSD detectors.

Multi-camera throughput: Frigate runs one OS process per configured detector, so
configuring several ``rocket`` detectors each with a distinct ``device`` (a big A76
core, ``"4"``..``"7"``) and ``num_threads: 1`` spreads that many independent NPU
contexts across the four A76 cores. A single detector with ``num_threads: 4`` fans
one inference across the NPU cores for the lowest single-stream latency.
"""

import logging
import os
from typing import Optional

from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.detectors.detector_utils import tflite_detect_raw, tflite_init

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate

    try:
        from tflite_runtime.interpreter import OpResolverType
    except ImportError:
        OpResolverType = None
except ModuleNotFoundError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate

    try:
        from ai_edge_litert.interpreter import OpResolverType
    except ImportError:
        OpResolverType = None

logger = logging.getLogger(__name__)

DETECTOR_KEY = "rocket"

# The delegate ships as a single self-contained .so: librocketnpu is statically
# linked in and it talks to /dev/accel via raw ioctls, so the only runtime deps are
# libstdc++/libc (already in the Frigate image). The bundled Dockerfile installs it
# here; override `delegate_path` to point elsewhere.
DEFAULT_DELEGATE = "/usr/local/lib/libtflite_rocket.so"


class RocketDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: Optional[str] = Field(
        default=None,
        title="A76 core",
        description=(
            "Big (A76) CPU core this detector's NPU worker pins to, e.g. "
            '"4".."7" — sets ROCKET_CPU_AFFINITY for this detector process. '
            "Leave unset for a single detector (the worker uses all big cores); set "
            "a distinct core per detector when running several for multi-camera "
            "throughput."
        ),
    )
    num_threads: int = Field(
        default=4,
        title="NPU worker threads",
        description=(
            "Delegate worker fan-out across the 3 NPU cores (clamped 1..8). Use 4 for "
            "a single detector (lowest latency); use 1 per detector when pooling "
            "several across distinct cores."
        ),
    )
    native_int8: bool = Field(
        default=True,
        title="Native int8/uint8 conv path",
        description=(
            "Run quantized convs on the NPU's native int8 datapath, validated at COCO "
            "mAP = CPU parity. Disable only to A/B against the fp16 dequant path."
        ),
    )
    delegate_path: str = Field(
        default=DEFAULT_DELEGATE,
        title="Delegate .so path",
        description="Path to libtflite_rocket.so inside the container.",
    )


class RocketDetector(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [ModelTypeEnum.ssd]

    def __init__(self, detector_config: RocketDetectorConfig):
        # Each Frigate detector runs in its own process, so the affinity env set here
        # is process-local: N detectors with distinct `device` land on N distinct
        # A76s. Set it BEFORE the delegate loads — the driver reads it when its
        # worker threads pin (at context creation, during the first Prepare/invoke).
        if detector_config.device is not None:
            os.environ["ROCKET_CPU_AFFINITY"] = str(detector_config.device)

        options = {
            "native_int8": "1" if detector_config.native_int8 else "0",
            "nthreads": str(detector_config.num_threads),
        }

        try:
            delegate = load_delegate(detector_config.delegate_path, options)
        except (ValueError, OSError) as e:
            logger.error(
                "Failed to load the rocket delegate from %s: %s. Check that the .so "
                "exists in the container and that /dev/accel/accel0 is passed through "
                "(devices:) with the host render group (group_add:).",
                detector_config.delegate_path,
                e,
            )
            raise

        logger.info(
            "rocket NPU detector: delegate=%s nthreads=%d native_int8=%s%s",
            detector_config.delegate_path,
            detector_config.num_threads,
            detector_config.native_int8,
            f" affinity={detector_config.device}" if detector_config.device else "",
        )

        # BUILTIN_WITHOUT_DEFAULT_DELEGATES stops TFLite from auto-applying XNNPACK
        # during construction; otherwise XNNPACK can claim a conv before the external
        # delegate gets a turn and the op silently runs on the CPU instead of the NPU.
        kw = {}
        if OpResolverType is not None:
            kw["experimental_op_resolver_type"] = (
                OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )

        interpreter = Interpreter(
            model_path=detector_config.model.path,
            experimental_delegates=[delegate],
            **kw,
        )

        tflite_init(self, interpreter)

    def detect_raw(self, tensor_input):
        return tflite_detect_raw(self, tensor_input)
