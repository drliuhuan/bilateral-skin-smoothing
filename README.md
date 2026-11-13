# bilateral-skin-smoothing · 皮肤磨皮参考实现

A self-contained NumPy implementation of a multi-scale, edge-preserving skin
smoothing pipeline, with an optional **U-2-Net (Apache-2.0)** automatic mask.

一个自包含的皮肤磨皮参考实现：多尺度双边网格（bilateral grid）边缘保持平滑 +
U-2-Net（Apache-2.0）自动蒙版 + 细节层混合 + 锐化/柔化 + 色调调整。

> 这是本项目的独立实现，与任何商业产品无关；代码以 Apache-2.0 发布。

---

## Features

- **8/16-bit RGB** in and out (custom 16-bit PNG codec: Pillow truncates 16-bit RGB).
- **Bilateral-grid smoothing** — O(N) approximation of an edge-preserving filter:
  splat → separable (1,2,1)/4 blur → normalise by accumulated weight →
  trilinear slice back to full resolution.
- **Multi-scale pyramid** — coarse-to-fine levels, two grid passes per level with
  a per-level strength schedule; 12-bit fixed-point level mixing.
- **Detail layers** — fine / medium / large detail decomposition and per-level
  weights; soft threshold for feature edges.
- **Automatic mask** — U-2-Net-lite person matte (Apache-2.0) × classical skin
  term (HSV and/or a soft HSL spline), with HSV-only fallback when OpenCV/model
  is unavailable.
- **Sharpness / softness** — soft-clipped unsharp (`x^(2/3)` knee) + 1/2 re-blend.
- **Tone** — warmth / brightness / contrast.
- Pure NumPy filters (no SciPy); OpenCV is used **only** for the optional person matte.

## Install

```bash
python3 -m pip install -r requirements.txt
# optional: the automatic U-2-Net mask
python3 scripts/fetch_u2net.py           # downloads models/u2netp.onnx (Apache-2.0)
```

`requirements.txt`:

```
numpy>=1.24
Pillow>=9
# optional, for the U-2-Net mask:
opencv-python-headless>=4.8
```

## Usage

```python
import skin_smoothing as P

img, meta = P.load_image("your.jpg")          # → float64 [0,1], 8 or 16 bit
res = P.smooth_skin(
    img,
    smoothing=75,        # 0..100  overall strength
    threshold=50,        # 0..100  detail/edge threshold
    fine=50, medium=50, large=50,
    sharpness=0.0, softness=0.0,
    warmth=0.0, brightness=0.0, contrast=0.0,
    portrait_size="medium",
)
P.save_image("out.png", res["output"], bit_depth=meta["bit_depth"])

res["mask"]        # the mask that was used
res["base"]        # smoothed base layer
res["details"]     # fine/medium/large detail layers
```

Force the classical mask instead of U-2-Net:

```python
res = P.smooth_skin(img, mask_method="hsv")
```

Build the mask alone:

```python
m = P.build_mask(img)               # U-2-Net × skin term, HSV fallback
m = P.build_mask(img, skin_term="hsv")   # person × HSV only
pm = P.u2net_person_matte(img)      # None if OpenCV/model unavailable
```

Custom model path:

```bash
export U2NET_MODEL_PATH=/path/to/u2netp.onnx
```

## Pipeline

```
load image (8/16-bit RGB)
  → [automatic skin mask]  person matte × skin term, feathered
  → multi-scale bilateral grid:
       splat (spatial × luminance bins, 12-bit)
       → separable (1,2,1)/4 blur, two axes
       → normalise by ΣW
       → trilinear slice back to full resolution
  → coarse-to-fine level combine (12-bit lerp)
  → detail-layer mix + edge threshold
  → masked blend (opacity)
  → sharpen / soften (soft-clipped unsharp + 1/2 re-blend)
  → tone adjustments
  → save image (8/16-bit RGB)
```

See [`docs/algorithm.md`](docs/algorithm.md) for the exact math (packed 16-bit
YCC format, 12-bit fixed point, element-wise formulas, parameter mapping).

## Demo

```bash
python3 demo.py                 # synthetic portrait, writes out/*.png + out/metrics.json
```

Outputs (before/after, mask, detail layers, slider sweep, 16-bit comparison,
filter variants) go to `out/`.

## Tests

```bash
python3 tests/test_primitives.py
# 27/27 tests passed
```

The suite covers filter primitives, 8/16-bit round-trips, pyramid/detail
invariants, mask behaviour, and the full pipeline. The U-2-Net test is skipped
automatically when OpenCV or the model file is absent (the HSV fallback is still
exercised).

## Model licence

The optional person matte uses **U-2-Net** (`u2netp.onnx`, 4.57 MB), which is
**Apache-2.0**. It is not bundled here; `scripts/fetch_u2net.py` downloads it
from the upstream release and checks its SHA-256. Keep the upstream licence/notice
when redistributing the weights.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE).
