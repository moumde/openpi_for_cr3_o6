"""Deploy the CR3/O6 policy through the OpenPI websocket policy server.

Start the server first, for example::

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_cr3_o6_lora \
        --policy.dir=checkpoints/pi05_cr3_o6_lora/pi05_cr3_o6_lora/99999

Preview inference without moving the robot::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    uv run python examples/cr3_o6/deploy_policy.py \
        --preview-only --prompt "Pick up the motor and place it on the right side with the protruding side facing left."

Real control requires the explicit ``--confirm-motion`` flag::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    uv run python examples/cr3_o6/deploy_policy.py \
        --confirm-motion

The client uses the dataset contract directly:

* observation state: six CR3 joint angles in radians followed by six O6
  positions in the native 0..255 range;
* action: six absolute CR3 joint targets in radians followed by six absolute
  O6 targets in the native 0..255 range;
* control frequency: 20 Hz; each policy response is a 30-row action chunk.

The default mode is intentionally not a motion mode.  ``--preview-only``
connects to all devices and requests policy actions, but discards them.
"""

from __future__ import annotations

import argparse
from collections import deque
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy

try:
    from .camera_interface import CAMERA_NAMES
    from .camera_interface import CR3O6CameraInterface
    from .linkerhand_compat import patch_pymodbus_slave_argument
except ImportError:  # Support ``python examples/cr3_o6/deploy_policy.py``.
    from camera_interface import CAMERA_NAMES
    from camera_interface import CR3O6CameraInterface
    from linkerhand_compat import patch_pymodbus_slave_argument


LOGGER = logging.getLogger("cr3_o6_policy_deploy")

ACTION_DIM = 12
ACTION_HORIZON = 30
CONTROL_HZ = 20.0
DEFAULT_POLICY_HOST = "127.0.0.1"
DEFAULT_POLICY_PORT = 8000
DEFAULT_ROBOT_IP = "192.168.2.14"
DEFAULT_HAND_CONFIG = Path("/home/je/code/linkerhand-python-sdk/LinkerHand/config/setting.yaml")
DEFAULT_PROMPT = "Pick up the motor and place it on the right side with the protruding side facing left."
INITIAL_ROBOT_POSITION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
INITIAL_HAND_POSITION = (157, 95, 175, 0, 0, 0)
DEFAULT_INITIAL_MOVE_DURATION_S = 3.0
INITIAL_MOVE_HZ = CONTROL_HZ
# These are deliberately conservative limits.  CR3's NRC ServoJ limits use
# degrees-based SDK units; the action slew limit below uses the public OpenPI
# radians-based units.
CR3_SERVO_VMAX_DEG_S = 50.1
CR3_SERVO_AMAX_DEG_S2 = 100.0
CR3_SERVO_JMAX_DEG_S3 = 200.0
MAX_JOINT_TARGET_SPEED_RAD_S = 1.0
DEFAULT_O6_SPEED = 100


def _load_hand_config(path: Path) -> tuple[str, str]:
    """Read the right-hand O6/RS485 selection from the SDK YAML file."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 LinkerHand setting.yaml 需要 PyYAML") from exc

    try:
        config = yaml.safe_load(path.read_text()) or {}
        right = config["LINKER_HAND"]["RIGHT_HAND"]
        joint = str(right.get("JOINT", "")).upper()
        modbus = right.get("MODBUS")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"无法读取 O6 配置 {path}: {exc}") from exc

    if joint != "O6":
        raise RuntimeError(f"右手配置必须是 O6, 当前是 {joint!r}: {path}")
    if not modbus or str(modbus).lower() == "none":
        raise RuntimeError(
            f"当前右手不是 RS485 配置, 请检查 {path} 的 RIGHT_HAND.MODBUS"
        )
    return joint, str(modbus)


def _load_hand_api():
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
    except ImportError as exc:
        raise RuntimeError(
            "LinkerHand SDK import failed. Set PYTHONPATH, for example: "
            "export PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH"
        ) from exc
    return LinkerHandApi


def _close_hand(hand_api: Any) -> None:
    """Close the RS485 backend without using the SDK's broken close_can()."""

    hand = getattr(hand_api, "hand", None)
    close = getattr(hand, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            LOGGER.exception("Failed to close LinkerHand RS485 backend")


def _finite_vector(values: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} must be a finite vector with shape ({length},), got {array.shape}")
    return array


def _validate_action_chunk(result: dict[str, Any]) -> np.ndarray:
    if not isinstance(result, dict) or "actions" not in result:
        raise RuntimeError(f"Policy response must contain 'actions', got {type(result).__name__}")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError(
            f"Expected action chunk with shape ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Policy action chunk contains non-finite values")
    return np.ascontiguousarray(actions)


def _prepare_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = _finite_vector(action, ACTION_DIM, "action")
    arm_target = action[:6]
    hand_target = np.clip(action[6:], 0.0, 255.0)
    return arm_target, hand_target


def _move_to_initial_position(
    robot: Any,
    hand: Any,
    *,
    duration_s: float = DEFAULT_INITIAL_MOVE_DURATION_S,
    control_hz: float = INITIAL_MOVE_HZ,
    max_joint_speed_rad_s: float = MAX_JOINT_TARGET_SPEED_RAD_S,
) -> None:
    """Move the CR3 and O6 to the configured safe/start position.

    CR3 positions are public API values in radians.  The seventh CR3 value is
    the external axis and is passed through unchanged.  The O6 values are the
    native integer command range used by the LinkerHand SDK.
    """

    if duration_s <= 0 or not math.isfinite(duration_s):
        raise ValueError("initial move duration must be finite and positive")
    if control_hz <= 0 or not math.isfinite(control_hz):
        raise ValueError("initial move frequency must be finite and positive")
    if max_joint_speed_rad_s <= 0 or not math.isfinite(max_joint_speed_rad_s):
        raise ValueError("initial move joint speed must be finite and positive")

    target = np.asarray(INITIAL_ROBOT_POSITION[:6], dtype=np.float64)
    external_axis = float(INITIAL_ROBOT_POSITION[6])
    hand_target = [int(value) for value in INITIAL_HAND_POSITION]
    current = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise RuntimeError(f"CR3 current joint position must have shape (6,), got {current.shape}")

    LOGGER.info(
        "Returning CR3/O6 to initial position: robot=%s, hand=%s, duration=%.2fs",
        list(INITIAL_ROBOT_POSITION),
        hand_target,
        duration_s,
    )
    hand.finger_move(hand_target)

    requested_steps = max(1, int(math.ceil(duration_s * control_hz)))
    speed_limited_steps = int(
        math.ceil(np.max(np.abs(target - current)) / max_joint_speed_rad_s * control_hz)
    )
    steps = max(requested_steps, speed_limited_steps, 1)
    period_s = 1.0 / control_hz
    started = time.monotonic()
    for step in range(1, steps + 1):
        alpha = step / steps
        command = current + alpha * (target - current)
        robot.send_joint_positions(command.tolist(), external_axis=external_axis)
        deadline = started + step * period_s
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    LOGGER.info("CR3/O6 reached initial position")


class CR3O6ObservationSource:
    """Build observations using the exact structure expected by CR3O6Inputs."""

    def __init__(self, robot: Any, hand: Any, cameras: CR3O6CameraInterface):
        self.robot = robot
        self.hand = hand
        self.cameras = cameras

    def read(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        hand_state = _finite_vector(self.hand.get_state(), 6, "O6 state")
        if np.any((hand_state < 0) | (hand_state > 255)):
            raise RuntimeError(f"O6 state outside [0, 255]: {hand_state.tolist()}")
        state = np.asarray(self.robot.get_openpi_state(hand_state.tolist()), dtype=np.float32)
        state = _finite_vector(state, ACTION_DIM, "OpenPI state")
        images = self.cameras.read()

        observation: dict[str, Any] = {
            "images": {},
            "state": state,
            "prompt": prompt,
        }
        for name in CAMERA_NAMES:
            image = np.asarray(images[name])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise RuntimeError(
                    f"Camera {name} returned {image.shape}/{image.dtype}; expected (224, 224, 3)/uint8"
                )
            observation["images"][name] = image
        return observation


class ActionChunkController:
    """Run 30-step policy chunks at the dataset's 20 Hz control rate."""

    def __init__(
        self,
        policy: websocket_client_policy.WebsocketClientPolicy,
        observation_source: CR3O6ObservationSource,
        robot: Any,
        hand: Any,
        prompt: str,
        *,
        control_hz: float = CONTROL_HZ,
        request_lead_steps: int = 6,
        max_joint_target_speed_rad_s: float = MAX_JOINT_TARGET_SPEED_RAD_S,
        initial_actions: np.ndarray | None = None,
    ) -> None:
        if control_hz <= 0 or request_lead_steps < 1 or max_joint_target_speed_rad_s <= 0:
            raise ValueError(
                "control_hz and max_joint_target_speed_rad_s must be positive, "
                "and request_lead_steps must be >= 1"
            )
        self.policy = policy
        self.observation_source = observation_source
        self.robot = robot
        self.hand = hand
        self.prompt = prompt
        self.control_hz = float(control_hz)
        self.request_lead_steps = int(request_lead_steps)
        self.max_joint_target_speed_rad_s = float(max_joint_target_speed_rad_s)
        self._actions: deque[np.ndarray] = deque()
        self._stop = threading.Event()
        self._request_event = threading.Event()
        self._request_in_flight = False
        self._worker_error: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._result: np.ndarray | None = None
        self._lock = threading.RLock()
        self._last_arm_command = _finite_vector(
            self.robot.get_joint_positions(), 6, "CR3 current joint position"
        ).astype(np.float64)
        if initial_actions is not None:
            self._actions.extend(_validate_action_chunk({"actions": initial_actions}))

    def _request_worker(self) -> None:
        while not self._stop.is_set():
            self._request_event.wait(0.1)
            if self._stop.is_set():
                return
            if not self._request_event.is_set():
                continue
            self._request_event.clear()
            try:
                observation = self.observation_source.read(self.prompt)
                result = _validate_action_chunk(self.policy.infer(observation))
                with self._lock:
                    self._result = result
                    self._request_in_flight = False
            except BaseException as exc:
                with self._lock:
                    self._worker_error = exc
                    self._request_in_flight = False
                self._stop.set()
                return

    def _raise_worker_error(self) -> None:
        with self._lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError(f"Policy worker failed: {type(error).__name__}: {error}") from error

    def _schedule_request(self) -> None:
        with self._lock:
            if self._request_in_flight or self._result is not None or self._stop.is_set():
                return
            self._request_in_flight = True
        self._request_event.set()

    def _accept_result(self) -> None:
        with self._lock:
            result = self._result
            self._result = None
        if result is not None:
            self._actions.extend(result)
            LOGGER.info("Received policy action chunk: shape=%s, queue=%d", result.shape, len(self._actions))

    def _apply_action(self, action: np.ndarray) -> None:
        arm_target, hand_target = _prepare_action(action)
        # Limit both the lead over fresh feedback and the change from the
        # previous accepted command.  This prevents a single policy action
        # jump from producing a fast arm motion when the model is uncertain.
        actual = _finite_vector(
            self.robot.get_joint_positions(), 6, "CR3 current joint position"
        ).astype(np.float64)
        max_step = self.max_joint_target_speed_rad_s / self.control_hz
        feedback_target = actual + np.clip(arm_target - actual, -max_step, max_step)
        commanded = self._last_arm_command + np.clip(
            feedback_target - self._last_arm_command, -max_step, max_step
        )
        self._last_arm_command = commanded
        self.robot.send_joint_positions(commanded.tolist())
        self.hand.finger_move([round(float(value)) for value in hand_target])

    def run(self, duration_s: float | None = None) -> None:
        self._worker = threading.Thread(
            target=self._request_worker,
            name="cr3-o6-policy-worker",
            daemon=True,
        )
        self._worker.start()
        self._schedule_request()

        period_s = 1.0 / self.control_hz
        next_tick = time.monotonic()
        started = next_tick
        try:
            while not self._stop.is_set():
                self._raise_worker_error()
                self._accept_result()
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    return
                if len(self._actions) <= self.request_lead_steps:
                    self._schedule_request()
                if self._actions:
                    self._apply_action(self._actions.popleft())
                else:
                    LOGGER.warning("Policy action buffer empty; holding current CR3 position")
                    self.robot.send_joint_positions(self.robot.get_joint_positions())

                next_tick += period_s
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_tick = time.monotonic()
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._request_event.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=5.0)
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("Policy worker did not stop cleanly")
        self._worker = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_POLICY_HOST, help="OpenPI policy server host/IP")
    parser.add_argument("--port", type=int, default=DEFAULT_POLICY_PORT, help="OpenPI policy server port")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="CR3 controller IP")
    parser.add_argument("--hand-config", type=Path, default=DEFAULT_HAND_CONFIG)
    parser.add_argument("--o6-modbus", default=None, help="Override RIGHT_HAND.MODBUS from setting.yaml")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--preview-only", action="store_true", help="Infer and discard actions; never move hardware")
    parser.add_argument("--confirm-motion", action="store_true", help="Required for real CR3/O6 control")
    parser.add_argument(
        "--power-on",
        action="store_true",
        help="Deprecated compatibility flag; CR3 power-on is automatic in real-control mode",
    )
    parser.add_argument(
        "--no-auto-power-on",
        action="store_true",
        help="Do not power on CR3 automatically; fail if it is not already running",
    )
    parser.add_argument("--request-lead-steps", type=int, default=6)
    parser.add_argument(
        "--initial-move-duration-s",
        type=float,
        default=DEFAULT_INITIAL_MOVE_DURATION_S,
        help="Seconds used to move CR3 to the initial position at start/end",
    )
    parser.add_argument(
        "--max-joint-speed-rad-s",
        type=float,
        default=MAX_JOINT_TARGET_SPEED_RAD_S,
        help="Maximum accepted CR3 target change per second during policy control",
    )
    parser.add_argument(
        "--o6-speed",
        type=int,
        default=DEFAULT_O6_SPEED,
        help="O6 motion speed for all six joints, native range 10..255",
    )
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 runs until Ctrl-C")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_preview(
    policy: websocket_client_policy.WebsocketClientPolicy,
    source: CR3O6ObservationSource,
    prompt: str,
    duration_s: float | None = None,
) -> None:
    LOGGER.info("Preview-only mode: actions will be discarded")
    started_at = time.monotonic()
    while True:
        if duration_s is not None and time.monotonic() - started_at >= duration_s:
            return
        started = time.monotonic()
        result = _validate_action_chunk(policy.infer(source.read(prompt)))
        LOGGER.info("Preview action chunk: shape=%s first_action=%s", result.shape, np.round(result[0], 4).tolist())
        time.sleep(max(0.0, 0.5 - (time.monotonic() - started)))


def main() -> int:
    args = _parser().parse_args()
    if args.preview_only and args.confirm_motion:
        raise SystemExit("--preview-only and --confirm-motion cannot be combined")
    if args.preview_only and (args.power_on or args.no_auto_power_on):
        raise SystemExit("power options cannot be used with --preview-only")
    if not args.preview_only and not args.confirm_motion:
        raise SystemExit("Real control requires --confirm-motion; use --preview-only for a no-motion test")
    if args.port < 1 or args.port > 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.request_lead_steps < 1:
        raise SystemExit("--request-lead-steps must be >= 1")
    if args.initial_move_duration_s <= 0 or not math.isfinite(args.initial_move_duration_s):
        raise SystemExit("--initial-move-duration-s must be finite and > 0")
    if args.max_joint_speed_rad_s <= 0 or not math.isfinite(args.max_joint_speed_rad_s):
        raise SystemExit("--max-joint-speed-rad-s must be finite and > 0")
    if not 10 <= args.o6_speed <= 255:
        raise SystemExit("--o6-speed must be between 10 and 255")
    if args.duration_s < 0 or not math.isfinite(args.duration_s):
        raise SystemExit("--duration-s must be >= 0")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # Import the vendor NRC binding only after argument validation.  This keeps
    # ``--help`` and configuration-only checks usable on machines without the
    # controller's Python-version-specific shared library.
    try:
        from .interface import SERVO_RUNNING
        from .interface import CR3Interface
        from .interface import NRCError
    except ImportError:
        from interface import SERVO_RUNNING
        from interface import CR3Interface
        from interface import NRCError

    _, modbus_from_config = _load_hand_config(args.hand_config)
    modbus = args.o6_modbus or modbus_from_config
    patch_pymodbus_slave_argument()
    linker_hand_api = _load_hand_api()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    LOGGER.info("Connected to policy server %s:%d; metadata=%s", args.host, args.port, policy.get_server_metadata())

    robot = CR3Interface(
        args.robot_ip,
        servo_vmax=[CR3_SERVO_VMAX_DEG_S] * 7,
        servo_amax=[CR3_SERVO_AMAX_DEG_S2] * 7,
        servo_jmax=[CR3_SERVO_JMAX_DEG_S3] * 7,
    )
    hand_api = None
    cameras = CR3O6CameraInterface()
    servo_opened = False
    powered_by_client = False
    controller: ActionChunkController | None = None
    try:
        LOGGER.info("Connecting CR3 at %s", args.robot_ip)
        robot.connect()
        LOGGER.info("Connecting right O6 over RS485 at %s", modbus)
        hand_api = linker_hand_api(hand_type="right", hand_joint="O6", modbus=modbus, can="None")
        if hand_api.hand_type != "right" or str(hand_api.hand_joint).upper() != "O6":
            raise RuntimeError(f"Unexpected hand configuration: {hand_api.hand_type}/{hand_api.hand_joint}")
        cameras.connect()
        source = CR3O6ObservationSource(robot, hand_api, cameras)

        if args.preview_only:
            LOGGER.info("Preview-only mode: initial-position motion is disabled")
            warmup = _validate_action_chunk(policy.infer(source.read(args.prompt)))
            LOGGER.info("Policy warmup succeeded: action chunk shape=%s", warmup.shape)
            _run_preview(policy, source, args.prompt, args.duration_s or None)
            return 0

        hand_api.set_speed([args.o6_speed] * 6)
        LOGGER.info("Configured O6 speed to %d for all six joints", args.o6_speed)
        if robot.get_servo_state() != SERVO_RUNNING:
            if args.no_auto_power_on:
                raise NRCError(
                    "CR3 is not in running state and --no-auto-power-on was specified"
                )
            LOGGER.info("CR3 is not running; powering it on automatically")
            robot.power_on()
            powered_by_client = True
        if robot.get_servo_state() != SERVO_RUNNING:
            raise NRCError("CR3 is not in running state; add --power-on or power it on manually")
        robot.open_servoj()
        servo_opened = True
        _move_to_initial_position(
            robot,
            hand_api,
            duration_s=args.initial_move_duration_s,
            max_joint_speed_rad_s=args.max_joint_speed_rad_s,
        )

        # Read the warmup observation after homing so the first policy request
        # matches the actual robot/hand state.
        warmup = _validate_action_chunk(policy.infer(source.read(args.prompt)))
        LOGGER.info("Policy warmup succeeded: action chunk shape=%s", warmup.shape)
        controller = ActionChunkController(
            policy,
            source,
            robot,
            hand_api,
            args.prompt,
            request_lead_steps=args.request_lead_steps,
            max_joint_target_speed_rad_s=args.max_joint_speed_rad_s,
            initial_actions=warmup,
        )
        controller.run(duration_s=args.duration_s or None)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Deployment stopped by user")
        return 0
    except (NRCError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Deployment failed: %s", exc)
        return 1
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                LOGGER.exception("Failed to stop policy controller")
        if servo_opened:
            try:
                _move_to_initial_position(
                    robot,
                    hand_api,
                    duration_s=args.initial_move_duration_s,
                    max_joint_speed_rad_s=args.max_joint_speed_rad_s,
                )
            except Exception:
                LOGGER.exception("Failed to return CR3/O6 to initial position")
            try:
                robot.close_servoj()
            except Exception:
                LOGGER.exception("Failed to close CR3 ServoJ")
        cameras.disconnect()
        if hand_api is not None:
            _close_hand(hand_api)
        if powered_by_client:
            try:
                robot.power_off()
            except Exception:
                LOGGER.exception("Failed to power off CR3")
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
