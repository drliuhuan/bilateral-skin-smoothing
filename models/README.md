# Model directory

The optional automatic mask uses **U-2-Net-lite** (`u2netp.onnx`), which is
released under the **Apache-2.0** licence.

The weights are not bundled in this repository. Fetch them with:

```bash
python3 scripts/fetch_u2net.py
```

Upstream / provenance:

- U-2-Net (Xuebin Qin et al.), <https://github.com/xuebinqin/U-2-Net> — Apache-2.0
- weights mirror used by the downloader: rembg release asset `u2netp.onnx`
  (SHA-256 `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`)

If you redistribute the weights, keep the upstream licence/notice.
