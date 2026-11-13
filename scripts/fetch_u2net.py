#!/usr/bin/env python3
"""Download the optional U-2-Net-lite person-matte weights (Apache-2.0).

Convenience wrapper around ``skin_smoothing.download_u2net``; saves the model to
``models/u2netp.onnx`` (source checkout) or to the user cache
(``~/.cache/bilateral-skin-smoothing/u2netp.onnx``).

Installed packages can use the ``bss-fetch-model`` console script instead.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import skin_smoothing as P  # noqa: E402


def main() -> int:
    dest = os.path.join(ROOT, "models", "u2netp.onnx")
    try:
        path = P.download_u2net(dest)
    except Exception as exc:  # fall back to the user cache on a read-only checkout
        print(f"could not write {dest} ({exc}); using cache instead", file=sys.stderr)
        path = P.download_u2net()
    print(f"ok: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
