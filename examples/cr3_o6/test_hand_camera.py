"""Connection/read-only test for the right LinkerHand O6 and three D435 cameras.

Example with the RS485 setup from the LinkerHand SDK configuration::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    python3.12 examples/cr3_o6/test_hand_camera.py \
        --transport rs485 --modbus /dev/ttyUSB0

Example with CAN::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    python3.12 examples/cr3_o6/test_hand_camera.py \
        --transport can --can can0

This script only initializes the right O6, reads its current six joint
positions, connects the three cameras, and reads camera frames.  It does not
send a hand or robot motion command.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence

import numpy as np

try:
    from .camera_interface import CAMERA_NAMES, CR3O6CameraInterface
    from .linkerhand_compat import patch_pymodbus_slave_argument
except ImportError:
    # Also support ``python examples/cr3_o6/test_hand_camera.py``.
    from camera_interface import CAMERA_NAMES, CR3O6CameraInterface
    from linkerhand_compat import patch_pymodbus_slave_argument


LOGGER = logging.getLogger("cr3_o6_connection_test")


def _load_hand_api():
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi

        return LinkerHandApi
    except ImportError as exc:
        raise RuntimeError(
            "LinkerHand SDK import failed. Add the SDK root to PYTHONPATH, for example: "
            "export PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH"
        ) from exc


def _validate_hand_state(state: Sequence[float]) -> list[int]:
    values = [float(value) for value in state]
    if len(values) != 6:
        raise AssertionError(f"Expected 6 O6 joint values, got {len(values)}: {values!r}")
    if not all(math.isfinite(value) and 0 <= value <= 255 for value in values):
        raise AssertionError(f"O6 joint values must be finite and in [0, 255], got {values!r}")
    return [int(round(value)) for value in values]


def _close_hand(hand_api) -> None:
    """Close either the RS485 or CAN implementation behind LinkerHandApi."""

    hand = getattr(hand_api, "hand", None)
    if hand is None:
        return
    for method_name in ("close", "close_can_interface"):
        method = getattr(hand, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                LOGGER.exception("Failed to close LinkerHand through %s()", method_name)
            return
    LOGGER.warning("No close method found on LinkerHand backend %s", type(hand).__name__)


def _validate_frames(frames: dict[str, np.ndarray]) -> None:
    if set(frames) != set(CAMERA_NAMES):
        raise AssertionError(f"Expected camera keys {CAMERA_NAMES}, got {tuple(frames)}")
    for name in CAMERA_NAMES:
        image = np.asarray(frames[name])
        if image.shape != (224, 224, 3):
            raise AssertionError(f"Camera {name} returned shape {image.shape}, expected (224, 224, 3)")
        if image.dtype != np.uint8:
            raise AssertionError(f"Camera {name} returned dtype {image.dtype}, expected uint8")
        if not image.flags.c_contiguous:
            raise AssertionError(f"Camera {name} returned a non-contiguous image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("rs485", "can"), default="rs485")
    parser.add_argument("--modbus", default="/dev/ttyUSB0", help="O6 RS485 device path")
    parser.add_argument("--can", default="can0", help="O6 CAN interface")
    parser.add_argument("--frames", type=int, default=1, help="Number of camera frames to read")
    parser.add_argument("--camera-timeout-ms", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hand_api = None
    cameras = None
    try:
        LinkerHandApi = _load_hand_api()
        patch_pymodbus_slave_argument()
        modbus = args.modbus if args.transport == "rs485" else "None"
        LOGGER.info(
            "Connecting right LinkerHand O6 via %s",
            args.modbus if args.transport == "rs485" else args.can,
        )
        hand_api = LinkerHandApi(
            hand_type="right",
            hand_joint="O6",
            modbus=modbus,
            can=args.can,
        )
        if hand_api.hand_type != "right" or hand_api.hand_joint.upper() != "O6":
            raise AssertionError(
                f"Unexpected hand configuration: type={hand_api.hand_type!r}, "
                f"joint={hand_api.hand_joint!r}"
            )
        hand_state = _validate_hand_state(hand_api.get_state())
        LOGGER.info("Right O6 connected; current state [0..255]: %s", hand_state)

        cameras = CR3O6CameraInterface(read_timeout_ms=args.camera_timeout_ms)
        cameras.connect()
        if not cameras.connected:
            raise AssertionError("Camera interface reports disconnected after connect()")
        LOGGER.info("Three D435 cameras connected: %s", cameras.camera_serials)

        for index in range(args.frames):
            frames = cameras.read()
            _validate_frames(frames)
            LOGGER.info(
                "Camera frame %d/%d received: %s",
                index + 1,
                args.frames,
                {name: frames[name].shape for name in CAMERA_NAMES},
            )

        LOGGER.info("Right O6 + three-camera connection test passed")
        return 0
    except (AssertionError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Connection test failed: %s", exc)
        return 1
    finally:
        if cameras is not None:
            cameras.disconnect()
        if hand_api is not None:
            _close_hand(hand_api)


if __name__ == "__main__":
    sys.exit(main())
