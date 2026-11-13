#!/usr/bin/env python3
"""demo.py -- runnable demonstration of the skin-smoothing pipeline.

Generates a synthetic portrait-like image, runs the full reference smoothing
pipeline on it, and writes before/after images plus diagnostic contact sheets
into ``out/``:

    01_input.png                 synthetic portrait (8-bit)
    02_skin_mask.png             skin-colour matte produced by the fallback mask
    03_smoothed_base.png         most-smoothed pyramid base (before detail mix)
    04_detail_fine_x4.png        fine detail layer (x4 amplified around 0.5)
    05_detail_medium_x4.png      medium detail layer
    06_detail_large_x4.png       large detail layer
    07_result.png                final smoothed result (8-bit)
    08_side_by_side.png          input | result
    09_input_16bit.png           16-bit version of the input
    10_result_16bit.png          pipeline result saved at 16 bit/channel
    11_slider_sweep.png          Smoothing x Threshold 3x3 contact sheet
    12_variants.png              filter-variant contact sheet
    metrics.json                 numeric validation summary

Run::

    cd bilateral-skin-smoothing
    python3 demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import skin_smoothing as P          # noqa: E402
import synthetic                 # noqa: E402

OUT_DIR = os.path.join(_HERE, "out")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_out() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def _u8(img: np.ndarray) -> np.ndarray:
    return P.to_uint(img, 8)


def _u8_pil(img: np.ndarray):
    from PIL import Image
    return Image.fromarray(_u8(img))


def _save(path: str, img: np.ndarray, bit_depth: int = 8) -> None:
    P.save_image(path, img, bit_depth=bit_depth)
    print(f"  wrote {os.path.relpath(path)}  ({img.shape[1]}x{img.shape[0]}, "
          f"{bit_depth}-bit)")


def _hf_noise(img: np.ndarray, sigma: float = 3.0) -> float:
    """High-frequency (texture/noise) std of a region, gradient removed."""
    a = np.asarray(img, np.float64)
    return float(np.std(a - P.gaussian_blur(a, sigma)))


def _grad_energy(img: np.ndarray, quantile: float = 99.5) -> float:
    """Edge strength: high quantile of the gradient magnitude in a face region."""
    a = np.asarray(img, np.float64)
    gx = np.diff(a, axis=1)[:-1, :]          # (H-1, W-1, C)
    gy = np.diff(a, axis=0)[:, :-1]          # (H-1, W-1, C)
    g = np.sqrt((gx ** 2 + gy ** 2).sum(axis=-1))
    return float(np.percentile(g, quantile))


def contact_sheet(items, cols: int, path: str, cell=(200, 200),
                  title: str = "") -> None:
    """Write a labelled contact sheet of ``(label, float image)`` items."""
    from PIL import Image, ImageDraw
    rows = int(np.ceil(len(items) / cols))
    cw, ch = cell
    pad, header = 4, 16
    canvas = Image.new("RGB",
                       (cols * (cw + pad) + pad,
                        (rows * (ch + pad + header) + pad
                         + (18 if title else 0))),
                       (28, 28, 30))
    draw = ImageDraw.Draw(canvas)
    y0 = pad
    if title:
        draw.text((pad, 3), title, fill=(240, 240, 240))
        y0 += 18
    for i, (label, img) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = y0 + r * (ch + pad + header)
        im = _u8_pil(np.clip(img, 0.0, 1.0)).resize(cell, Image.LANCZOS)
        canvas.paste(im, (x, y + header))
        draw.text((x + 2, y + 2), label, fill=(230, 230, 230))
    canvas.save(path)
    print(f"  wrote {os.path.relpath(path)}")


def _timed(fn, *args, **kwargs):
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t


# ---------------------------------------------------------------------------
# demo stages
# ---------------------------------------------------------------------------

def demo_main_pipeline() -> dict:
    print("\n[1] synthetic portrait + full pipeline (8 bit)")
    img, face_gt = synthetic.synthetic_portrait(height=720, width=540, seed=7)
    _save(os.path.join(OUT_DIR, "01_input.png"), img)

    res, t_total = _timed(P.smooth_skin, img)
    out = res["output"]
    der = res["derived"]
    if der["method"] == "grid":
        desc = (f"method=grid levels={der['levels']} strength={der['strength']} "
                f"sigma_s_px={der['sigma_s_px']:.2f} sigma_r={der['sigma_r']:.3f} "
                f"level_weights12={der['level_weights12']}")
    else:
        desc = (f"method=recursive levels={der['levels']} "
                f"sigma_s={der['sigma_s']:.3f} sigma_r={der['sigma_r']:.3f} "
                f"weights={der['detail_weights']}")
    print(f"  pipeline time: {t_total:.3f} s   {desc}")

    _save(os.path.join(OUT_DIR, "02_skin_mask.png"), res["mask"])
    _save(os.path.join(OUT_DIR, "03_smoothed_base.png"), res["base"])
    for name, idx, amp in (("fine", 0, 5.0), ("medium", 1, 8.0), ("large", 2, 12.0)):
        detail = res["details"][idx]
        vis = np.clip(0.5 + amp * detail, 0.0, 1.0)
        _save(os.path.join(OUT_DIR, f"04_detail_{name}_x{int(amp)}.png"), vis)
    _save(os.path.join(OUT_DIR, "07_result.png"), out)

    # side-by-side (input | result), separated by a thin line
    sep = np.zeros((img.shape[0], 4, 3))
    side = np.concatenate([img, sep, out], axis=1)
    _save(os.path.join(OUT_DIR, "08_side_by_side.png"), side)

    # ---- metrics ---------------------------------------------------------
    # noise metric on a flat cheek patch, edge metric on the eye/brow region
    # (see synthetic.synthetic_portrait for the geometry).
    h, w = img.shape[:2]
    skin = (slice(int(0.52 * h), int(0.60 * h)),
            slice(int(0.33 * w), int(0.43 * w)))
    feature = (slice(int(0.34 * h), int(0.48 * h)),
               slice(int(0.26 * w), int(0.47 * w)))
    noise_before = _hf_noise(img[skin])
    noise_after = _hf_noise(out[skin])
    edge_before = _grad_energy(img[feature])
    edge_after = _grad_energy(out[feature])
    face_region = face_gt > 0.5
    metrics = {
        "image_shape": list(img.shape),
        "pipeline_seconds": t_total,
        "derived_params": {k: (v if not isinstance(v, list) else v)
                           for k, v in der.items()},
        "mask_mean_face": float(res["mask"][face_region].mean()),
        "mask_mean_outside": float(res["mask"][~face_region].mean()),
        "skin_noise_hf": {
            "region": "cheek",
            "before": noise_before,
            "after": noise_after,
            "ratio": noise_after / noise_before,
        },
        "feature_edge_strength_p99_5": {
            "region": "eye/brow",
            "before": edge_before,
            "after": edge_after,
            "ratio": edge_after / edge_before,
        },
        "mean_luminance": {
            "before": float(img.mean()),
            "after": float(out.mean()),
            "delta": float(out.mean() - img.mean()),
        },
        "detail_energy": {
            name: float(np.std(res["details"][i]))
            for i, name in enumerate(("fine", "medium", "large"))
        },
    }
    print(f"  skin high-freq noise (cheek): {noise_before:.4f} -> "
          f"{noise_after:.4f} (x{noise_after / noise_before:.2f} retained)")
    print(f"  feature edge strength (eye/brow, p99.5 |grad|): "
          f"{edge_before:.4f} -> {edge_after:.4f} "
          f"(x{edge_after / edge_before:.2f})")
    print(f"  mean luminance: {img.mean():.4f} -> {out.mean():.4f}")
    return {"image": img, "result": res, "metrics": metrics}


def demo_16bit(img: np.ndarray) -> dict:
    print("\n[2] 16-bit-per-channel path")
    # Build a genuine 16-bit source: quantise the float image to 16 bits with a
    # small amount of deterministic dither so the extra precision is exercised.
    rng = np.random.default_rng(123)
    dither = (rng.random(img.shape) - 0.5) / 65535.0
    img16 = np.clip(np.round(img * 65535.0) / 65535.0 + dither, 0.0, 1.0)
    in16 = os.path.join(OUT_DIR, "09_input_16bit.png")
    _save(in16, img16, bit_depth=16)
    loaded, meta = P.load_image(in16)
    assert meta["bit_depth"] == 16, meta

    res = P.smooth_skin(loaded)
    out16_path = os.path.join(OUT_DIR, "10_result_16bit.png")
    _save(out16_path, res["output"], bit_depth=16)

    # 8-bit vs 16-bit quantisation error of the *same* result
    out8_path = os.path.join(OUT_DIR, "_result_8bit_for_compare.png")
    P.save_image(out8_path, res["output"], bit_depth=8)
    out8, _ = P.load_image(out8_path)
    out16, meta16 = P.load_image(out16_path)
    err8 = float(np.abs(out8 - res["output"]).max())
    err16 = float(np.abs(out16 - res["output"]).max())
    patch = res["output"][100:300, 100:300]
    n_levels = lambda a: int(len(np.unique(np.round(a * 65535).astype(np.int32))))
    os.remove(out8_path)

    metrics = {
        "input_bit_depth": meta["bit_depth"],
        "output_bit_depth": meta16["bit_depth"],
        "max_quantisation_error_8bit": err8,
        "max_quantisation_error_16bit": err16,
        "error_ratio_8bit_over_16bit": err8 / max(err16, 1e-12),
        "distinct_levels_patch_8bit": n_levels(np.round(patch * 255) / 255),
        "distinct_levels_patch_16bit": n_levels(patch),
        "note": ("the implementation runs float64 [0,1] internally for both depths; "
                 "the 8-bit and 16-bit paths share the same core math"),
    }
    print(f"  16-bit PNG round-trip exact to {err16:.2e}; 8-bit error "
          f"{err8:.2e} ({err8 / max(err16, 1e-12):.0f}x larger)")
    print(f"  distinct levels in skin patch: 8-bit {metrics['distinct_levels_patch_8bit']}"
          f", 16-bit {metrics['distinct_levels_patch_16bit']}")
    return metrics


def _crop_metrics(orig: np.ndarray, out: np.ndarray) -> dict:
    """Noise/edge metrics on the fixed cheek and brow sub-regions of a crop."""
    h, w = orig.shape[:2]
    # the sweep crop is img[180:500, 108:428]: cheek patch and brow band inside
    skin_l = (slice(int(0.62 * h), int(0.80 * h)),
              slice(int(0.03 * w), int(0.25 * w)))
    feat_l = (slice(int(0.12 * h), int(0.45 * h)),
              slice(int(0.10 * w), int(0.85 * w)))
    return {
        "noise_ratio": _hf_noise(out[skin_l]) / max(_hf_noise(orig[skin_l]), 1e-9),
        "edge_ratio": _grad_energy(out[feat_l]) / max(_grad_energy(orig[feat_l]), 1e-9),
    }


def demo_slider_sweep(img: np.ndarray) -> dict:
    print("\n[3] Detail-Smoothing x Threshold sweep (unmasked, 320px face crop)")
    crop = img[180:500, 108:428]
    items, table = [], {}
    for detail in (0.0, 50.0, 100.0):
        for threshold in (20.0, 50.0, 80.0):
            out = P.smooth_skin(crop, use_mask=False, fine=detail, medium=detail,
                                large=detail, threshold=threshold)["output"]
            m = _crop_metrics(crop, out)
            label = f"D={int(detail)} T={int(threshold)}"
            items.append((label, out))
            table[label] = m
            print(f"  {label}: noise x{m['noise_ratio']:.2f}, "
                  f"edges x{m['edge_ratio']:.2f}")
    contact_sheet(items, cols=3, path=os.path.join(OUT_DIR, "11_slider_sweep.png"),
                  title="Fine/Medium/Large detail sliders x Threshold (rows:"
                        " detail=0/50/100, cols: T=20/50/80); unmasked")
    return {"crop_shape": list(crop.shape), "table": table}


def demo_variants(img: np.ndarray) -> dict:
    print("\n[4] filter variants (evaluated on the full portrait)")
    h, w = img.shape[:2]
    skin = (slice(int(0.52 * h), int(0.60 * h)),
            slice(int(0.33 * w), int(0.43 * w)))
    feature = (slice(int(0.34 * h), int(0.48 * h)),
               slice(int(0.26 * w), int(0.47 * w)))
    crop = img[220:380, 150:390]          # visual crop: eyes / brows / nose

    def crop_of(full: np.ndarray) -> np.ndarray:
        return full[220:380, 150:390]

    sigma_s, sigma_r = 0.115, 0.128
    variants = {
        "rbf gauss sr=0.128": dict(range_mode="gauss", sigma_r=0.128),
        "rbf exp sr=0.128": dict(range_mode="exp", sigma_r=0.128),
        "rbf gauss sr=0.400": dict(range_mode="gauss", sigma_r=0.400),
        "rbf per-channel": dict(range_mode="gauss", sigma_r=0.128,
                                per_channel=True),
        "pyramid base (3 lv)": dict(pyramid=True),
        "full smooth_skin": dict(full=True),
    }
    items = [("input", crop)]
    table = {}
    for name, cfg in variants.items():
        t0 = time.perf_counter()
        if cfg.get("full"):
            full = P.smooth_skin(img)["output"]
        elif cfg.get("pyramid"):
            full = P.bilateral_pyramid(img, levels=3, iters=2,
                                       sigma_s=sigma_s, sigma_r=sigma_r)[-1]
        else:
            full = P.recursive_bilateral(img, sigma_s, cfg["sigma_r"],
                                         range_mode=cfg["range_mode"],
                                         per_channel=cfg.get("per_channel", False))
        dt = time.perf_counter() - t0
        items.append((name, crop_of(full)))
        table[name] = {
            "seconds": dt,
            "noise_ratio": _hf_noise(full[skin]) / max(_hf_noise(img[skin]), 1e-9),
            "edge_ratio": _grad_energy(full[feature]) / max(_grad_energy(img[feature]), 1e-9),
        }
    contact_sheet(items, cols=4, path=os.path.join(OUT_DIR, "12_variants.png"),
                  title="recursive bilateral variants / stages (eye/brow crop)")
    for k, v in table.items():
        print(f"  {k:<22s} {v['seconds'] * 1e3:7.1f} ms  "
              f"noise x{v['noise_ratio']:.2f}  edges x{v['edge_ratio']:.2f}")
    return table


def main() -> int:
    _ensure_out()
    t_start = time.perf_counter()
    print("Skin smoothing -- self-contained NumPy demo")
    print(f"output directory: {os.path.relpath(OUT_DIR)}")
    main = demo_main_pipeline()
    metrics = {"main": main["metrics"]}
    metrics["sixteen_bit"] = demo_16bit(main["image"])
    metrics["slider_sweep"] = demo_slider_sweep(main["image"])
    metrics["variants"] = demo_variants(main["image"])
    metrics["total_seconds"] = time.perf_counter() - t_start

    path = os.path.join(OUT_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=float)
    print(f"\nwrote {os.path.relpath(path)}")
    print(f"total demo time: {metrics['total_seconds']:.2f} s")
    print("\nDone. Open out/08_side_by_side.png for the before/after.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
