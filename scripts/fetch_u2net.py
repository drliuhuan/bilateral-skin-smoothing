#!/usr/bin/env python3
"""Download the optional U-2-Net-lite person-matte weights (Apache-2.0).

Writes models/u2netp.onnx and verifies its SHA-256.
"""
import hashlib
import os
import sys
import urllib.request

URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"

def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "models")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "u2netp.onnx")
    if os.path.exists(out):
        print(f"already exists: {out}")
    else:
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, out)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"sha256 {h}")
    if h != SHA256:
        print("SHA-256 mismatch — delete the file and retry", file=sys.stderr)
        return 1
    print("ok:", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
