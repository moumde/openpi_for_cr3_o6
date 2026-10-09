"""Manual hardware test for :mod:`examples.cr3_o6.interface`.

The default run is read-only.  It connects to the teach and tracking ports,
reads servo state/joints/TCP, and checks Cartesian-to-joint conversion.

Examples::

    # Read-only validation
    python3.12 examples/cr3_o6/test_interface.py --ip 192.168.2.14

    # Explicitly test the control path.  The arm target is its current
    # position, so this should not intentionally move the robot.
    python3.12 examples/cr3_o6/test_interface.py \
        --ip 192.168.2.14 --control-test

The control test powers the robot on only when necessary, opens ServoJ, sends
the current joint position once, closes ServoJ, and powers the robot off again
if this script powered it on.  Keep the robot clear and supervised anyway.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence

try:
    from .interface import (
        CR3Interface,
        NRCError,
        SERVO_ERROR,
        SERVO_READY,
        SERVO_RUNNING,
        SERVO_STOPPED,
    )
except ImportError:  # Support ``python examples/cr3_o6/test_interface.py``.
    from interface import (
        CR3Interface,
        NRCError,
        SERVO_ERROR,
        SERVO_READY,
        SERVO_RUNNING,
        SERVO_STOPPED,
    )


LOGGER = logging.getLogger("cr3_interface_test")
SERVO_STATE_NAMES = {
    SERVO_STOPPED: "stopped",
    SERVO_READY: "ready",
    SERVO_ERROR: "error",
    SERVO_RUNNING: "running",
}


def _assert_vector(name: str, values: Sequence[float], expected_length: int) -> None:
    if len(values) != expected_length:
        raise AssertionError(f"{name} has length {len(values)}, expected {expected_length}: {values!r}")
    if not all(math.isfinite(float(value)) for value in values):
        raise AssertionError(f"{name} contains non-finite values: {values!r}")


def run_read_only_test(robot: CR3Interface, *, check_cartesian: bool = True) -> None:
    """Validate connections and all non-motion interface calls."""

    state = robot.get_servo_state()
    LOGGER.info("Servo state: %d (%s)", state, SERVO_STATE_NAMES.get(state, "unknown"))
    if state not in SERVO_STATE_NAMES:
        raise AssertionError(f"Unexpected servo state: {state}")

    joints = robot.get_joint_positions()
    _assert_vector("joint positions [rad]", joints, 6)
    LOGGER.info("Joint positions [rad]: %s", [round(value, 6) for value in joints])

    tcp = robot.get_tcp_pose()
    _assert_vector("TCP pose [mm, rad, ext]", tcp, 7)
    LOGGER.info("TCP pose [mm, rad, ext]: %s", [round(value, 6) for value in tcp])

    if check_cartesian:
        converted = robot.convert_pose_to_joints(tcp)
        _assert_vector("Cartesian-converted joints [rad]", converted, 6)
        LOGGER.info(
            "Cartesian conversion result [rad]: %s",
            [round(value, 6) for value in converted],
        )


def run_control_test(robot: CR3Interface) -> None:
    """Exercise power, ServoJ, and command paths with a no-motion target."""

    initial_state = robot.get_servo_state()
    powered_by_test = False
    servo_opened = False
    try:
        if initial_state != SERVO_RUNNING:
            LOGGER.info("Servo is not running; executing power_on()")
            robot.power_on()
            powered_by_test = True
        if robot.get_servo_state() != SERVO_RUNNING:
            raise AssertionError("Servo did not reach running state after power_on()")

        robot.open_servoj()
        servo_opened = True
        current_joints = robot.get_joint_positions()
        robot.send_joint_positions(current_joints)
        LOGGER.info("ServoJ accepted the current joint position without an intentional move")

    finally:
        if servo_opened:
            robot.close_servoj()
            LOGGER.info("ServoJ closed")
        if powered_by_test:
            robot.power_off()
            LOGGER.info("Robot powered off because this test powered it on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.2.14", help="CR3 controller IP address")
    parser.add_argument(
        "--control-test",
        action="store_true",
        help="Also power on if needed, open ServoJ, send the current position, and power off",
    )
    parser.add_argument(
        "--skip-cartesian",
        action="store_true",
        help="Skip the Cartesian pose to joint conversion check",
    )
    parser.add_argument(
        "--connect-control",
        action="store_true",
        help="Also connect the optional 6000 control port",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    robot = CR3Interface(
        args.ip,
        connect_control=args.connect_control,
        connection_timeout=args.timeout,
    )
    try:
        LOGGER.info("Connecting to CR3 at %s", args.ip)
        robot.connect()
        run_read_only_test(robot, check_cartesian=not args.skip_cartesian)
        if args.control_test:
            run_control_test(robot)
        LOGGER.info("CR3 interface test passed")
        return 0
    except (NRCError, AssertionError, ValueError) as exc:
        LOGGER.error("CR3 interface test failed: %s", exc)
        return 1
    finally:
        robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
