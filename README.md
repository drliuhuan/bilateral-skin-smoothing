# bilateral-skin-smoothing

[![CI](https://github.com/drliuhuan/bilateral-skin-smoothing/actions/workflows/ci.yml/badge.svg)](https://github.com/drliuhuan/bilateral-skin-smoothing/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

English · [中文](README.zh-CN.md)

A self-contained PyTorch-free, NumPy implementation of a multi-scale,
edge-preserving **skin smoothing** pipeline, with an optional **U-2-Net
(Apache-2.0)** automatic mask.

It is a reference implementation of a classic retouching pipeline:

* multi-scale **bilateral grid** smoothing (splat → separable blur → normalise →
  slice) instead of a per-pixel bilateral filter — O(N), fast on large images;
* fine / medium / large **detail layers** with per-scale weights and a soft
  edge threshold;
* **sharpening / softening** with a soft-clipped unsharp mask;
* **tone** adjustments (warmth / brightness / contrast);
* **automatic skin mask** from a licence-safe person matte (U-2-Net) times a
  classical skin-colour term, with an HSV-only fallback.

Only NumPy and Pillow are required; OpenCV is optional and used *only* to run the
U-2-Net matte.

> This is an independent implementation. It is not affiliated with, endorsed by,
> or derived from any commercial product's code or model weights.

## Installation

### From PyPI

```bash
pip install bilateral-skin-smoothing
pip install "bilateral-skin-smoothing[mask]"   # optional U-2-Net mask dependencies
```

> The package is release-ready; publishing to PyPI is automated in
> `.github/workflows/publish.yml` (PyPI trusted publishing) once configured in
> the PyPI project settings. Until then use one of the options below.

### From GitHub

```bash
pip install git+https://github.com/drliuhuan/bilateral-skin-smoothing.git
```

### From a release wheel

Download the `.whl` from [Releases](https://github.com/drliuhuan/bilateral-skin-smoothing/releases)
and `pip install` it, or build it yourself:

```bash
python -m pip install build
python -m build            # dist/bilateral_skin_smoothing-*.whl + .tar.gz
```

### From source

```bash
git clone https://github.com/drliuhuan/bilateral-skin-smoothing.git
cd bilateral-skin-smoothing
pip install -r requirements.txt
```

## Quick start

```python
import skin_smoothing as P

img, meta = P.load_image("portrait.jpg")        # float64 [0,1], 8 or 16-bit RGB
res = P.smooth_skin(
    img,
    smoothing=75,          # 0..100 overall strength
    threshold=50,          # 0..100 detail / edge threshold
    fine=50, medium=50, large=50,
    sharpness=0.0, softness=0.0,
    warmth=0.0, brightness=0.0, contrast=0.0,
    portrait_size="medium",
)
P.save_image("portrait_smoothed.png", res["output"], bit_depth=meta["bit_depth"])

res["mask"]        # the mask that was used
res["base"]        # smoothed base layer
res["details"]     # fine / medium / large detail layers
```

### Automatic mask (U-2-Net)

```bash
python -m pip install "bilateral-skin-smoothing[mask]"   # or: pip install opencv-python-headless
bss-fetch-model                                          # downloads to ~/.cache/bilateral-skin-smoothing/
```

In a source checkout, `python scripts/fetch_u2net.py` stores the weights in
`models/` instead. The resolver checks (in order): `u2net_path=`, the
`U2NET_MODEL_PATH` env var, `models/` next to the module / in the current
directory, and the user cache.

`skin_smoothing` then uses **U-2-Net-lite person matte × classical skin term**
by default. If OpenCV or the model file is missing it silently falls back to the
HSV matte, so the pipeline always runs.

```python
res = P.smooth_skin(img, mask_method="hsv")   # force the classical fallback
mask = P.build_mask(img)                      # build the mask alone
pm = P.u2net_person_matte(img)                # None if unavailable
```

Custom model path: `export U2NET_MODEL_PATH=/path/to/u2netp.onnx`
(or pass `u2net_path=`).

## Pipeline

```
load image (8/16-bit RGB)
  → automatic skin mask        person matte × skin term, feathered
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

The exact math is documented in [`docs/algorithm.md`](docs/algorithm.md).

## Demo

```bash
python demo.py          # synthetic portrait → out/*.png, out/metrics.json
```

Produces before/after, the mask, detail layers, a smoothing-slider sweep, a
16-bit comparison and filter variants under `out/`.

## Tests

```bash
python tests/test_primitives.py     # 27/27
```

The suite covers the filter primitives, 8/16-bit round-trips, pyramid and detail
invariants, mask behaviour, and the full pipeline. Without OpenCV or the model
file it automatically exercises the HSV fallback path.

## Project layout

```
skin_smoothing.py         core implementation (importable module)
synthetic.py              deterministic synthetic test images
demo.py                   runnable demo
docs/algorithm.md         the math
models/README.md          optional U-2-Net weights (not bundled)
scripts/fetch_u2net.py    model downloader with SHA-256 check
tests/test_primitives.py  27 unit tests
.github/workflows/ci.yml      CI: tests (py3.9–3.13) + build + wheel smoke test
.github/workflows/publish.yml PyPI trusted publishing on release
pyproject.toml            package metadata (setuptools)
```

## Versioning & releasing

1. bump `version` in `pyproject.toml` and `__version__` in `skin_smoothing.py`;
2. commit, create a Git tag and a GitHub Release (`vX.Y.Z`);
3. the release workflow builds the sdist/wheel and uploads them to PyPI via
   trusted publishing (configure the PyPI project environment first).

Local build check:

```bash
python -m pip install build twine
python -m build
twine check dist/*
python -m pip install dist/*.whl
```

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE).

The optional U-2-Net weights are **Apache-2.0** as well; they are not bundled.
Keep the upstream licence/notice if you redistribute them.
