"""Use a pinned mirror without changing the frozen publisher acquisition code."""

from __future__ import annotations

from types import FunctionType

import fetch_c5_broad_how2sign as original
from how2sign_hf_mirror import How2SignMirrorZip, provenance


def bind(function, **overrides):
    namespace = {**function.__globals__, **overrides}
    result = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def specification(reservation_path, data):
    plan = original.specification(reservation_path, data)
    plan["transport"] = provenance()
    for name in ("fetch_c5_broad_how2sign_mirror.py", "how2sign_hf_mirror.py"):
        relative = "tools/" + name
        plan["source_sha256"][relative] = original.sha256(original.ROOT / relative)
    return plan


freeze = bind(original.freeze, specification=specification)
checked = bind(original.checked, specification=specification)
run = bind(original.run, checked=checked, How2SignRemoteZip=How2SignMirrorZip)
main = bind(original.main, freeze=freeze, run=run)


if __name__ == "__main__":
    main()
