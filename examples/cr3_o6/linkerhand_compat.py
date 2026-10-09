"""Compatibility helpers for the LinkerHand SDK's pymodbus calls."""

from __future__ import annotations

from functools import wraps
import inspect
from typing import Any


def patch_pymodbus_slave_argument() -> None:
    """Make SDK ``slave=`` calls work with newer pymodbus releases.

    The LinkerHand SDK pins pymodbus 3.5.1, whose client methods accept the
    ``slave`` keyword.  Newer pymodbus releases renamed it to ``device_id``.
    Patch only the two methods used by the O6 RS485 implementation, and only
    when their installed signature no longer accepts ``slave``.
    """

    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError as exc:
        raise RuntimeError("LinkerHand RS485 requires pymodbus") from exc

    for method_name in ("read_input_registers", "write_registers"):
        method = getattr(ModbusSerialClient, method_name)
        parameters = inspect.signature(method).parameters
        if "slave" in parameters or "device_id" not in parameters:
            continue
        if getattr(method, "_openpi_slave_compat", False):
            continue

        original = method

        @wraps(original)
        def compatible_method(self: Any, *args: Any, _original=original, **kwargs: Any):
            if "slave" in kwargs:
                if "device_id" in kwargs:
                    raise TypeError("Specify only one of slave or device_id")
                kwargs["device_id"] = kwargs.pop("slave")
            return _original(self, *args, **kwargs)

        compatible_method._openpi_slave_compat = True
        setattr(ModbusSerialClient, method_name, compatible_method)

