# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The tflite-rocket authors
"""Read the delegate's process-wide counts, so a harness can assert that the delegate ran.

The delegate counts the nodes it claimed and, per invoke, the claimed ops dispatched to an NPU
route with a device open (`npu`) and those run on the host (`host`). Without a device the
RK3588 path computes every conv on the driver's CPU oracle, close enough to TFLite to pass a
comparison, so a harness reading only outputs cannot tell a run on the NPU from none.

ctypes.CDLL on the path the delegate was loaded from returns the same library instance, so
the counts are the ones the interpreter's delegate updated. Take a snapshot before the arm and
check the difference after it.
"""
import ctypes
import os


class Counts:
    def __init__(self, claimed, npu, host):
        self.claimed, self.npu, self.host = claimed, npu, host

    def __sub__(self, o):
        return Counts(self.claimed - o.claimed, self.npu - o.npu, self.host - o.host)

    def __str__(self):
        return f"{self.claimed} node(s) claimed, {self.npu} op(s) on the NPU, {self.host} on the host"


def read(delegate_so):
    """The counts now, or None when the library predates the counters."""
    lib = ctypes.CDLL(os.path.abspath(delegate_so))
    fn = getattr(lib, "rocket_delegate_counts", None)
    if fn is None:
        return None
    c, n, h = ctypes.c_long(), ctypes.c_long(), ctypes.c_long()
    fn(ctypes.byref(c), ctypes.byref(n), ctypes.byref(h))
    return Counts(c.value, n.value, h.value)


def check(before, after, what="the delegate arm"):
    """True when the arm claimed nodes and ran at least one op on the NPU. Prints either way."""
    if before is None or after is None:
        print(f"   FAIL: {what}: the delegate exports no rocket_delegate_counts, so whether it "
              f"ran cannot be checked (rebuild it)")
        return False
    d = after - before
    ok = d.claimed > 0 and d.npu > 0
    print(f"   {what}: {d} -> {'ok' if ok else 'FAIL'}")
    if not ok:
        print("   (0 claimed: nothing was delegated. 0 on the NPU: no device opened, so the "
              "CPU oracle computed the claimed convs)")
    return ok
