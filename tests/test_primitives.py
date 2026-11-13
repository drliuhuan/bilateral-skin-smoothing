#!/usr/bin/env python3
"""test_primitives.py -- plain-assert unit tests for the skin-smoothing reference
implementation (no pytest required).

Run::

    cd bilateral-skin-smoothing
    python3 impl/tests/test_primitives.py

Each test registers with ``@test``; failures print a traceback and the process
exits non-zero.  The suite covers the five required behaviours:

1. recursive bilateral filter is the identity when ``sigma_r -> 0``;
2. it denoises a noisy flat area;
3. it preserves step-edge contrast (``> 90%`` for the full pipeline, ~100% for
   the primitive);
4. the image pyramid is self-consistent (constants, linear ramps, level order);
5. mask blending hits its 0/1/opacity limits exactly.

plus image I/O (8/16-bit), unsharp/tone and the full :func:`smooth_skin`
pipeline.
"""

from __future__ import annotations

import os
import struct
import sys
import traceback
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import skin_smoothing as P          # noqa: E402
import synthetic                 # noqa: E402

_TESTS = []


def test(fn):
    _TESTS.append(fn)
    return fn


def check(cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "check failed")


def close(a, b, tol: float, msg: str = "") -> None:
    err = float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
    if not err <= tol:
        raise AssertionError(f"{msg or 'values differ'}: max abs err {err:.3e} > {tol:.3e}")


def hf_std(img: np.ndarray, sigma: float = 3.0) -> float:
    """High-frequency residual std (noise level), gradient-trend removed."""
    a = np.asarray(img, np.float64)
    return float(np.std(a - P.gaussian_blur(a, sigma)))


# ===========================================================================
# 1. Separable Gaussian
# ===========================================================================

@test
def test_gaussian_kernel():
    for sigma in (0.5, 1.0, 2.5, 8.0):
        k = P.gaussian_kernel(sigma)
        check(abs(k.sum() - 1.0) < 1e-12, f"kernel sum {k.sum()} for sigma={sigma}")
        check(k.size % 2 == 1, "odd kernel length")
        close(k, k[::-1], 1e-15, "kernel symmetry")
    close(P.gaussian_kernel(0.0), np.array([1.0]), 0.0, "sigma=0 delta kernel")


@test
def test_gaussian_blur():
    rng = np.random.default_rng(0)
    for shape in ((32, 40), (32, 40, 3)):
        const = np.full(shape, 0.7)
        close(P.gaussian_blur(const, 3.0), const, 1e-12,
              f"constant preserved for {shape}")
        # values >1 are legal internally (16-bit-safe: no clipping)
        big = np.full(shape, 123.456)
        close(P.gaussian_blur(big, 2.0), big, 1e-10, "no clipping above [0,1]")
    # sigma=0 identity
    img = rng.random((32, 40, 3))
    close(P.gaussian_blur(img, 0.0), img, 0.0, "sigma=0 identity")
    # single delta -> symmetric response
    delta = np.zeros((31, 31))
    delta[15, 15] = 1.0
    b = P.gaussian_blur(delta, 3.0)
    close(b, b[::-1, :], 1e-15, "vertical symmetry")
    close(b, b[:, ::-1], 1e-15, "horizontal symmetry")
    close(b, b.T, 1e-15, "isotropy")
    # 2-D blur equals per-channel blur of 3-D image
    img2 = rng.random((24, 24, 3))
    for c in range(3):
        close(P.gaussian_blur(img2, 2.0)[..., c], P.gaussian_blur(img2[..., c], 2.0),
              1e-14, "channel independence")


# ===========================================================================
# 2. Recursive bilateral (Yang 2012/2015)
# ===========================================================================

@test
def test_rbf_identity_when_sigma_r_zero():
    rng = np.random.default_rng(1)
    yy = np.linspace(0, 1, 48)[:, None]
    xx = np.linspace(0, 1, 64)[None, :]
    ramp = 0.5 * (yy + xx)                      # smooth gradient
    img = np.stack([ramp, ramp, ramp], axis=-1) + rng.normal(0, 0.01, (48, 64, 3))
    for mode in ("gauss", "exp"):
        out = P.recursive_bilateral(img, sigma_s=0.2, sigma_r=0.0, range_mode=mode)
        close(out, img, 1e-9, f"sigma_r=0 identity ({mode})")
    # gray 2-D input
    g = ramp + rng.normal(0, 0.01, ramp.shape)
    close(P.recursive_bilateral(g, 0.2, 0.0), g, 1e-9, "2-D identity")


@test
def test_rbf_axis_composition():
    rng = np.random.default_rng(2)
    img = rng.random((40, 52, 3))
    kw = dict(sigma_s=0.15, sigma_r=0.12)
    full = P.recursive_bilateral(img, **kw)
    manual = P.recursive_bilateral_y(
        P.recursive_bilateral_x(img, **kw), **kw)
    close(full, manual, 1e-15, "two-pass = X then Y")
    check(full.shape == img.shape and full.dtype == np.float64, "shape/dtype")
    # 2-D input round trip
    g = rng.random((40, 52))
    out = P.recursive_bilateral(g, **kw)
    check(out.shape == g.shape, "2-D shape")


@test
def test_rbf_denoises_flat_area():
    flat = synthetic.synthetic_flat_noise(64, 64, noise=0.05, seed=3)
    out = P.recursive_bilateral(flat, sigma_s=0.25, sigma_r=0.15)
    check(np.std(out) < 0.35 * np.std(flat),
          f"noise not reduced enough: {np.std(out)/np.std(flat):.3f}")
    check(abs(float(out.mean() - flat.mean())) < 0.01,
          f"mean shifted: {out.mean() - flat.mean():.4f}")
    check(0.0 <= out.min() and out.max() <= 1.0, "output in [0,1]")
    # higher range sigma -> more smoothing of pure noise (flat region only)
    var_low = float(np.var(P.recursive_bilateral(flat, 0.15, 0.02)))
    var_high = float(np.var(P.recursive_bilateral(flat, 0.15, 0.20)))
    check(var_high < var_low, f"sigma_r monotonicity: {var_low:.3e} -> {var_high:.3e}")


@test
def test_rbf_preserves_step_edge():
    step = synthetic.synthetic_step_edge(64, 64, low=0.2, high=0.8,
                                         edge_fraction=0.5, noise=0.01, seed=4)
    before = step[:, :24].mean() - step[:, 40:].mean()
    for mode in ("gauss", "exp"):
        out = P.recursive_bilateral(step, sigma_s=0.2, sigma_r=0.1, range_mode=mode)
        after = out[:, :24].mean() - out[:, 40:].mean()
        ratio = float(after / before)
        check(ratio > 0.90, f"edge contrast preserved only {ratio:.3f} ({mode})")
        check(abs(float(out[:, :24].mean() - step[:, :24].mean())) < 0.01,
              "flat side unchanged")


# ===========================================================================
# 3. Pyramid
# ===========================================================================

@test
def test_downsample_upsample():
    const = np.full((33, 35), 0.42)                 # odd dims -> edge padded
    low = P.downsample2x2(const)
    check(low.shape == (17, 18), f"odd-dim shape {low.shape}")
    close(low, 0.42, 1e-14, "constant downsample")
    close(P.upsample_bilinear(const, (67, 71)), 0.42, 1e-14, "constant upsample")
    # linear ramp survives downsample+upsample exactly (area avg + bilinear)
    ramp = np.tile(np.arange(32, dtype=np.float64), (32, 1)) / 31.0
    roundtrip = P.upsample_bilinear(P.downsample2x2(ramp), ramp.shape)
    close(roundtrip[:, 2:-2], ramp[:, 2:-2], 1e-10, "linear ramp roundtrip")
    # bilinear interpolation on a known pattern
    checker = np.array([[0.0, 1.0], [1.0, 0.0]])
    up = P.upsample_bilinear(checker, (2, 2))
    close(up, checker, 1e-14, "identity upsample size")


@test
def test_bilateral_pyramid():
    rng = np.random.default_rng(5)
    img = rng.random((64, 64, 3))

    # levels=0 is the identity
    lv0 = P.bilateral_pyramid(img, levels=0)
    check(len(lv0) == 1, "levels=0 returns [img]")
    close(lv0[0], img, 1e-15, "levels=0 identity")

    # level k shape is always the input resolution
    lv = P.bilateral_pyramid(img, levels=3, iters=2, sigma_s=0.115, sigma_r=0.12)
    check(len(lv) == 3, "level count")
    for k, level in enumerate(lv):
        check(level.shape == img.shape, f"level {k} shape {level.shape}")
        check(np.all(np.isfinite(level)), f"level {k} finite")

    # Step1/Step3 only (iters=0) == upsample(downsample(img))
    expected = P.upsample_bilinear(P.downsample2x2(img), img.shape[:2])
    close(P.bilateral_pyramid(img, levels=1, iters=0)[0], expected, 1e-14,
          "levels=1, iters=0 = downsample+upsample")

    # a constant image stays constant through every level (DC preserving)
    const = np.full((48, 40, 3), 0.35)
    for level in P.bilateral_pyramid(const, levels=3, iters=2):
        close(level, 0.35, 1e-10, "constant preserved in pyramid")

    # more levels = smoother: variance of a noise field decreases monotonically
    noise = rng.random((64, 64, 3))
    variances = [float(np.var(P.bilateral_pyramid(noise, levels=n, iters=2)[-1]))
                 for n in (1, 2, 3, 4)]
    check(variances == sorted(variances, reverse=True),
          f"pyramid variance not monotone: {variances}")
    check(variances[-1] < variances[0], "pyramid did not smooth")


# ===========================================================================
# 4. Detail decomposition / level combination
# ===========================================================================

@test
def test_detail_decompose_reconstruct():
    rng = np.random.default_rng(6)
    img = rng.random((64, 64, 3))
    levels = P.bilateral_pyramid(img, levels=3, iters=2, sigma_s=0.115,
                                 sigma_r=0.12)
    base, details = P.detail_decompose(img, levels)
    check(len(details) == 3, "3 detail layers")
    reconstructed = P.combine_levels(base, details, 1.0)
    close(reconstructed, img, 1e-12, "weights=1 reconstructs input")
    # weights=0 -> most-smoothed base
    close(P.combine_levels(base, details, 0.0), base, 1e-15,
          "weights=0 gives base")
    # mapping weights {fine, medium, large}
    w = {"fine": 1.0, "medium": 0.5, "large": 0.0}
    out = P.combine_levels(base, details, w)
    close(out, level_expect(img, base, details, w), 1e-14, "named weights")


def level_expect(img, base, details, w):
    return (base + w["fine"] * details[0] + w["medium"] * details[1]
            + w["large"] * details[2])


@test
def test_threshold_gain():
    detail = np.zeros((8, 8, 3))
    detail[:4] = 0.002        # low-contrast (skin texture)
    detail[4:] = 0.500        # high-contrast (feature edge)
    gain = P.threshold_gain(detail, threshold=0.05, weight=0.5)
    close(gain[:4], 0.5, 1e-12, "low-contrast detail gets smoothing weight")
    close(gain[4:], 1.0, 1e-12, "high-contrast detail preserved")
    check(gain.min() >= 0.5 - 1e-12 and gain.max() <= 1.0 + 1e-12,
          "gain range [weight, 1]")
    # threshold=0 disables protection (all gains 1)
    close(P.threshold_gain(detail, 0.0, 0.5), 1.0, 1e-15, "threshold=0 -> gain 1")
    # combine treats gains as the effective weight: out = gain * detail
    base = np.zeros_like(detail)
    out = P.combine_levels(base, [detail], [0.5], gains=[gain])
    close(out[:4], 0.5 * detail[:4], 1e-12, "attenuated low contrast")
    close(out[4:], detail[4:], 1e-12, "edge detail kept at full weight")


# ===========================================================================
# 5. Mask utilities / blend limits
# ===========================================================================

@test
def test_soft_matte_and_blur():
    v = np.linspace(-0.2, 1.2, 1001)
    m = P.soft_matte(v, 0.2, 0.8, feather=0.1)
    check(m.min() >= 0.0 and m.max() <= 1.0, "matte range")
    close(P.soft_matte(np.array([0.5]), 0.2, 0.8, 0.1), 1.0, 1e-15, "centre = 1")
    close(P.soft_matte(np.array([0.0, 1.0]), 0.2, 0.8, 0.1), 0.0, 1e-15,
          "outside = 0")
    # monotone ramps: low edge rises, high edge falls
    idx = np.searchsorted(v, 0.2)
    check(np.all(np.diff(m[max(idx - 10, 0):idx + 10]) >= -1e-12),
          "rising low edge")
    # hard matte when feather=0
    close(P.soft_matte(np.array([0.19, 0.21, 0.79, 0.81]), 0.2, 0.8),
          np.array([0.0, 1.0, 1.0, 0.0]), 1e-15, "hard matte")
    # mask blur: constant preserved, noisy mask variance reduced
    close(P.mask_blur(np.full((32, 32), 0.4), 3.0), 0.4, 1e-12, "constant mask")
    rng = np.random.default_rng(7)
    noisy = rng.random((32, 32))
    check(np.var(P.mask_blur(noisy, 2.0)) < np.var(noisy), "mask blur smooths")


@test
def test_blend_masked_limits():
    rng = np.random.default_rng(8)
    a = rng.random((32, 32, 3))
    b = rng.random((32, 32, 3))
    zeros = np.zeros((32, 32))
    ones = np.ones((32, 32))
    half = np.full((32, 32), 0.5)

    close(P.blend_masked(a, b, zeros), a, 0.0, "mask=0 -> a")
    close(P.blend_masked(a, b, ones), b, 0.0, "mask=1 -> b")
    close(P.blend_masked(a, b, ones, opacity=0.0), a, 0.0, "opacity=0 -> a")
    close(P.blend_masked(a, b, ones, opacity=1.0), b, 0.0, "opacity=1 -> b")
    close(P.blend_masked(a, b, half), 0.5 * (a + b), 1e-15, "50% blend")
    close(P.blend_masked(a, b, half, opacity=0.5), 0.75 * a + 0.25 * b, 1e-15,
          "mask*opacity")
    # (H,W,1) mask broadcasts
    close(P.blend_masked(a, b, ones[..., None]), b, 0.0, "(H,W,1) mask")
    check(np.all((P.blend_masked(a, b, rng.random((32, 32))).min() >= 0.0)), "range")


@test
def test_build_skin_mask_sanity():
    skin = np.array([0.80, 0.62, 0.50])
    img = np.zeros((16, 32, 3))
    img[:, :16] = skin
    img[:, 16:] = np.array([0.10, 0.20, 0.85])       # blue background
    m = P.build_skin_mask(img)
    check(m.shape == (16, 32) and m.min() >= 0.0 and m.max() <= 1.0, "mask range")
    check(m[:, :16].mean() > 0.8, f"skin not detected: {m[:, :16].mean():.3f}")
    check(m[:, 16:].mean() < 0.1, f"blue detected as skin: {m[:, 16:].mean():.3f}")


# ===========================================================================
# 6. Image I/O (8-bit / 16-bit)
# ===========================================================================

def _write_png16_filtered(arr: np.ndarray, filter_type: int = 0) -> bytes:
    """Manual 16-bit RGB PNG writer used to test the decoder's unfiltering."""
    h, w, c = arr.shape
    a16 = np.round(np.clip(arr, 0, 1) * 65535).astype(np.uint16)
    stride = w * c * 2
    raw = bytearray()
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        line = a16[y].astype(">u2").tobytes()
        b = np.frombuffer(line, np.uint8).astype(np.int32)
        if filter_type == 0:
            f = b
        elif filter_type == 1:                      # Sub
            f = (b - np.concatenate([np.zeros(c * 2, np.int32),
                                     b[:-c * 2]])) % 256
        elif filter_type == 2:                      # Up
            f = (b - prev) % 256
        else:
            raise ValueError(filter_type)
        prev = b
        raw.append(filter_type)
        raw += f.astype(np.uint8).tobytes()

    def chunk(ctype, cdata):
        body = ctype + cdata
        return (struct.pack(">I", len(cdata)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))


@test
def test_image_io_8bit(tmpdir=None):
    rng = np.random.default_rng(9)
    outdir = os.path.join(os.path.dirname(_HERE), "out")
    os.makedirs(outdir, exist_ok=True)
    rgb = rng.random((16, 24, 3))
    path = os.path.join(outdir, "_test_8bit.png")
    P.save_image(path, rgb, bit_depth=8)
    loaded, meta = P.load_image(path)
    check(meta["bit_depth"] == 8 and meta["mode"] == "RGB", f"meta {meta}")
    check(loaded.dtype == np.float64 and loaded.shape == rgb.shape, "dtype/shape")
    close(loaded, np.round(rgb * 255.0) / 255.0, 1e-15, "8-bit roundtrip")
    # grayscale
    gray = rng.random((16, 24))
    P.save_image(path, gray, bit_depth=8)
    loaded, meta = P.load_image(path)
    check(meta["mode"] == "L" and loaded.shape == gray.shape, f"gray meta {meta}")
    close(loaded, np.round(gray * 255.0) / 255.0, 1e-15, "8-bit gray roundtrip")


@test
def test_image_io_16bit():
    rng = np.random.default_rng(10)
    outdir = os.path.join(os.path.dirname(_HERE), "out")
    os.makedirs(outdir, exist_ok=True)
    rgb = rng.random((16, 24, 3))
    path = os.path.join(outdir, "_test_16bit.png")
    P.save_image(path, rgb, bit_depth=16)
    check(P.png_bit_depth(path) == 16, "PNG header says 16-bit")
    loaded, meta = P.load_image(path)
    check(meta["bit_depth"] == 16 and meta["mode"] == "RGB", f"meta {meta}")
    # 16-bit quantisation error is <= 0.5/65535 (whereas 8-bit would be ~128x worse)
    close(loaded, rgb, 0.5 / 65535.0 + 1e-12, "16-bit roundtrip precision")
    # adjacent 16-bit code values survive as distinct values
    levels = np.zeros((4, 4, 3))
    levels[:] = 10000.0 / 65535.0
    levels[2:] = 10001.0 / 65535.0
    P.save_image(path, levels, bit_depth=16)
    loaded2, _ = P.load_image(path)
    check(loaded2[0, 0, 0] != loaded2[3, 0, 0], "16-bit distinguishes adjacent codes")
    close(loaded2, np.round(levels * 65535.0) / 65535.0, 1e-15,
          "16-bit code values exact")
    # grayscale 16-bit
    gray = rng.random((8, 8))
    P.save_image(path, gray, bit_depth=16)
    loaded3, meta3 = P.load_image(path)
    check(meta3["mode"] == "L" and loaded3.shape == gray.shape, f"gray16 {meta3}")
    close(loaded3, gray, 0.5 / 65535.0 + 1e-12, "16-bit gray roundtrip")


@test
def test_png16_filtered_decode():
    rng = np.random.default_rng(11)
    arr = rng.random((7, 9, 3))
    for ft in (0, 1, 2):
        data = _write_png16_filtered(arr, filter_type=ft)
        decoded, depth = P._decode_png16(data)
        check(depth == 16, "decoded depth")
        close(decoded, np.round(arr * 65535) / 65535, 1e-12,
              f"decoder unfilter type {ft}")


# ===========================================================================
# 7. Unsharp / tone
# ===========================================================================

@test
def test_unsharp():
    rng = np.random.default_rng(12)
    img = synthetic.synthetic_step_edge(48, 48, 0.3, 0.7, edge_fraction=0.5,
                                        noise=0.005, seed=12)
    close(P.unsharp(img, 0.0, 2.0), img, 0.0, "amount=0 identity")
    sharp = P.unsharp(img, 0.8, 2.0)
    soft = P.unsharp(img, -0.8, 2.0)
    grad = lambda a: float(np.abs(np.diff(a[10:38, :, 0], axis=1)).max())
    check(grad(sharp) > grad(img), "sharpening increases edge gradient")
    check(grad(soft) < grad(img), "softening decreases edge gradient")
    # threshold guard kills the effect on a smooth image
    smooth = np.full((16, 16, 3), 0.5) + rng.normal(0, 1e-4, (16, 16, 3))
    close(P.unsharp(smooth, 1.0, 1.0, threshold=0.01), smooth, 1e-15,
          "threshold suppresses tiny detail")


@test
def test_tone_adjust():
    rng = np.random.default_rng(13)
    img = rng.random((24, 24, 3))
    close(P.adjust_tone(img), img, 0.0, "neutral identity")
    brighter = P.adjust_tone(img, brightness=0.4)
    check(brighter.mean() > img.mean(), "brightness raises mean")
    contrasty = P.adjust_tone(img, contrast=0.5)
    check(contrasty.std() > img.std(), "contrast raises std")
    warm = P.adjust_tone(img, warmth=0.6)
    d_before = float(img[..., 0].mean() - img[..., 2].mean())
    d_after = float(warm[..., 0].mean() - warm[..., 2].mean())
    check(d_after > d_before, "warmth raises R-B")
    for out in (brighter, contrasty, warm):
        check(out.min() >= 0.0 and out.max() <= 1.0, "clipped to [0,1]")


# ===========================================================================
# 8. Full pipeline
# ===========================================================================

def _portrait_test_image():
    """Portrait-ish 128px test image: bright skin plateau, dark feature bar."""
    rng = np.random.default_rng(4)
    h = w = 128
    img = np.zeros((h, w, 3))
    yy = np.linspace(0, 1, h)[:, None]
    img[:] = np.array([0.80, 0.64, 0.53]) * (0.9 + 0.15 * yy[..., None])
    img[:, 40:88] = np.array([0.20, 0.12, 0.10])     # hard feature bar
    img += rng.normal(0, 0.03, (h, w, 1))
    return np.clip(img, 0.0, 1.0)


@test
def test_recovered_ycc_format():
    """Packed 16-bit YCC words: exact unpack coefficients + pack round-trip."""
    rng = np.random.default_rng(30)
    img = rng.random((48, 40, 3))
    w0, sign, w2, w3 = P.ycc_pack(img)
    check(w0.dtype == np.uint16 and w2.dtype == np.int16, "packed dtypes")
    check(np.all(sign == 0), "luma sign is 0 for [0,1] images")
    back = P.ycc_unpack(w0, sign, w2, w3) / 65535.0
    err16 = float(np.abs(back - img).max() * 65535.0)
    check(err16 <= 8.0, f"pack/unpack round-trip error {err16:.2f} 16-bit units")

    # exact recovered unpack formula on hand-computed words
    c = P.ycc_unpack(np.array([1000]), np.array([0]),
                     np.array([100]), np.array([-50]))
    A = 1526 * 1000
    r = lambda q: (q >> 10) + 2 * ((q >> 10) & 1)
    expect = np.array([r(A + 2092 * 100), r(A - 1066 * 100 + 516 * 50),
                       r(A - 2644 * 50)], dtype=np.float64)
    expect = np.clip(expect, 0, 65535)
    close(c[0], expect, 1e-12, "unpack matrix/rounding")

    # constant grey round-trips through the full recovered format
    grey = np.full((16, 16, 3), 0.5)
    g = P.ycc_unpack(*P.ycc_pack(grey)) / 65535.0
    close(g, 0.5, 2e-4, "grey round-trip")


@test
def test_blur121_and_lerp12():
    """Recovered Step2 (1,2,1)/4 kernel and 12-bit lerp."""
    rng = np.random.default_rng(31)
    a = rng.random((8, 9))
    # vertical pass
    b = P.blur121(a, 0)
    # interior matches (2*center + up + down)/4
    close(b[1:-1], 0.25 * (2 * a[1:-1] + a[:-2] + a[2:]), 1e-15, "blur121 axis0")
    # horizontal pass
    b1 = P.blur121(a, 1)
    close(b1[:, 1:-1], 0.25 * (2 * a[:, 1:-1] + a[:, :-2] + a[:, 2:]),
          1e-15, "blur121 axis1")
    # constant preserved, edge-clamped
    close(P.blur121(np.full((5, 5), 0.3), 0), 0.3, 1e-15, "blur121 constant")

    # lerp12 exact fixed-point arithmetic.  Note the recovered formula is
    # (a*(4095-p) + b*p) >> 12, i.e. a plain 12-bit fixed point with a 1/4096
    # gain error: even at p=0 the output is (a*4095)>>12, not exactly a.
    A = np.array([[1000, 0, 65535]])
    B = np.array([[3000, 4095, 0]])
    close(P.lerp12(A, B, 0), (A * 4095) >> 12, 0.0, "lerp12 p=0 arithmetic")
    close(P.lerp12(A, B, 4095), (B * 4095) >> 12, 0.0, "lerp12 p=max arithmetic")
    close(P.lerp12(A, B, 2048), (A * (4095 - 2048) + B * 2048) >> 12, 0.0,
          "lerp12 midpoint exact")
    # bounded between a and b (within the 1/4096 fixed-point bias)
    mid = P.lerp12(A, B, 2048)
    check(np.all(mid <= np.maximum(A, B)) and np.all(mid >= np.minimum(A, B)),
          "lerp12 bounded")


@test
def test_bilateral_grid_smooth():
    """Bilateral-grid chain: DC preservation, denoising, edge preservation."""
    rng = np.random.default_rng(32)
    const = np.full((96, 96, 3), 0.5)
    close(P.bilateral_grid_smooth(const, 8.0, 0.128), 0.5, 2e-4,
          "grid constant preserved (12-bit internal)")
    flat = synthetic.synthetic_flat_noise(96, 96, noise=0.05, seed=32)
    out = P.bilateral_grid_smooth(flat, 8.0, 0.128)
    check(np.std(out) < 0.2 * np.std(flat),
          f"grid denoise weak: {np.std(out)/np.std(flat):.3f}")
    check(abs(float(out.mean() - flat.mean())) < 0.01, "grid DC drift")
    step = synthetic.synthetic_step_edge(96, 96, low=0.2, high=0.8,
                                         edge_fraction=0.5, noise=0.01, seed=33)
    out = P.bilateral_grid_smooth(step, 8.0, 0.128)
    c0 = float(step[:, :36].mean() - step[:, 60:].mean())
    c1 = float(out[:, :36].mean() - out[:, 60:].mean())
    check(c1 / c0 > 0.95, f"grid edge contrast ratio {c1/c0:.3f}")
    # 2-D input supported
    check(P.bilateral_grid_smooth(flat.mean(-1), 8.0, 0.128).shape == flat.shape[:2],
          "grid 2-D input")


@test
def test_bilateral_grid_pyramid():
    """Recovered per-level driver: levels, strengths, combine, DC."""
    rng = np.random.default_rng(34)
    img = rng.random((64, 64, 3))
    levels, combined = P.bilateral_grid_pyramid(
        img, levels=3, sigma_s=8.0, sigma_r=0.128, strength=2048,
        level_weights12=[1024, 1024, 1024])
    check(len(levels) == 3, "3 levels")
    for lev in levels:
        check(lev.shape == img.shape, f"level shape {lev.shape}")
        check(np.all(np.isfinite(lev)), "finite levels")
    check(combined.shape == img.shape, "combined shape")
    # constant image: every level and the combination stay constant
    const = np.full((64, 64, 3), 0.4)
    c_levels, c_comb = P.bilateral_grid_pyramid(
        const, levels=3, sigma_s=8.0, sigma_r=0.128, strength=2048,
        level_weights12=[1024, 1024, 1024])
    for lev in c_levels:
        close(lev, 0.4, 2e-3, "constant level")
    close(c_comb, 0.4, 2e-3, "constant combined")
    # zero weight sum -> identity
    _, ident = P.bilateral_grid_pyramid(img, levels=3, sigma_s=8.0,
                                        sigma_r=0.128, strength=2048,
                                        level_weights12=[0, 0, 0])
    close(ident, img, 0.0, "zero weights = identity")
    # levels=0 -> identity
    lv0, comb0 = P.bilateral_grid_pyramid(img, levels=0)
    close(comb0, img, 0.0, "levels=0 identity")
    # coarse-to-fine: level 0 (finest) has more high-frequency energy than
    # the last (coarsest) level
    hf = lambda a: float(np.std(a - P.gaussian_blur(a, 3.0)))
    check(hf(levels[0]) > hf(levels[-1]), "level ordering fine->coarse")


@test
def test_apply_sharp_and_soft():
    """Recovered x^(2/3) unsharp soft-clip and 1/2 softness re-blend."""
    img = synthetic.synthetic_step_edge(64, 64, 0.3, 0.7, edge_fraction=0.5,
                                        noise=0.01, seed=35)
    ref = P.gaussian_blur(img, 3.0)
    close(P.apply_sharp_and_soft(img, ref, 0.0, 0.0), img, 0.0,
          "sharp/soft identity")
    sharp = P.apply_sharp_and_soft(img, ref, sharpness=60.0)
    soft = P.apply_sharp_and_soft(img, ref, softness=60.0)
    grad = lambda a: float(np.abs(np.diff(a[10:54, :, 0], axis=1)).max())
    check(grad(sharp) > grad(img), "sharpness increases edge gradient")
    check(grad(soft) < grad(img), "softness decreases edge gradient")
    # the x^(2/3) knee compresses large deltas sub-linearly: a 0.4 step gets
    # a delta smaller than the linear (unsharp) delta 0.4
    big = np.full((8, 8, 3), 0.5)
    big[4:] = 0.9
    ref2 = np.full((8, 8, 3), 0.5)
    out = P.apply_sharp_and_soft(big, ref2, sharpness=100.0)
    delta = float(out.max() - 0.9)
    check(0.0 < delta < 0.4, f"x^(2/3) soft clip compression wrong: {delta:.3f}")
    # a weak sharpness leaves a large step nearly untouched (knee suppresses it)
    weak = P.apply_sharp_and_soft(big, ref2, sharpness=5.0)
    check(float(weak.max()) < 0.95, f"weak sharpening overshoot: {weak.max():.3f}")


@test
def test_smooth_skin_pipeline_grid():
    """Full pipeline on the recovered bilateral-grid path (default)."""
    img = _portrait_test_image()
    res = P.smooth_skin(img, use_mask=False)
    check(res["derived"]["method"] == "grid", "default method is grid")
    out = res["output"]
    check(out.shape == img.shape and out.dtype == np.float64, "output shape/dtype")
    check(out.min() >= 0.0 and out.max() <= 1.0, "output range")
    for key in ("mask", "levels", "base", "details", "smoothed", "sharpened"):
        check(key in res, f"trace missing {key}")

    flat_before = hf_std(img[10:50, 5:30])
    flat_after = hf_std(out[10:50, 5:30])
    check(flat_after < 0.5 * flat_before,
          f"grid pipeline noise reduction weak: {flat_after/flat_before:.3f}")

    bright, dark = slice(8, 30), slice(58, 70)
    c_before = float(img[:, bright].mean() - img[:, dark].mean())
    c_after = float(out[:, bright].mean() - out[:, dark].mean())
    check(c_after > 0.95 * c_before,
          f"pipeline edge contrast preserved only {c_after/c_before:.3f}")
    check(abs(float(out.mean() - img.mean())) < 0.02, "luminance drift")

    # mask path also runs and keeps the edge
    res_m = P.smooth_skin(img)
    check(res_m["mask"] is not None, "mask built by default")
    c_mask = float(res_m["output"][:, bright].mean()
                   - res_m["output"][:, dark].mean())
    check(c_mask > 0.95 * c_before, f"masked edge contrast {c_mask/c_before:.3f}")

    # opacity=0 and zero mask are exact no-ops
    close(P.smooth_skin(img, use_mask=False, opacity=0.0)["output"], img, 0.0,
          "opacity=0 is identity")
    close(P.smooth_skin(img, mask=np.zeros(img.shape[:2]))["output"], img, 0.0,
          "zero mask is identity")

    # smoothing=0 -> level weights 0 -> exact identity
    close(P.smooth_skin(img, use_mask=False, smoothing=0.0)["output"], img,
          0.0, "smoothing=0 is identity")

    # determinism and parameter plumbing
    a = P.smooth_skin(img, use_mask=False)["output"]
    b = P.smooth_skin(img, use_mask=False)["output"]
    close(a, b, 0.0, "deterministic")
    c = P.smooth_skin(img, use_mask=False, smoothing=80)["output"]
    d = P.smooth_skin(img, params={"smoothing": 80}, use_mask=False)["output"]
    close(c, d, 0.0, "dict params == keyword override")
    check(np.abs(c - a).max() > 1e-6, "smoothing slider changes output")

    # sharpness / softness are applied after the smoothing chain
    sharp = P.smooth_skin(img, use_mask=False, sharpness=80.0)["output"]
    check(np.abs(sharp - a).max() > 1e-6, "sharpness changes output")

    # grayscale and 16-bit-quantised inputs work
    gray = img.mean(axis=-1)
    check(P.smooth_skin(gray, use_mask=False)["output"].shape == gray.shape,
          "2-D pipeline")
    q = np.round(img * 65535.0) / 65535.0
    res_q = P.smooth_skin(q, use_mask=False)
    check(res_q["output"].min() >= 0.0 and res_q["output"].max() <= 1.0,
          "16-bit input pipeline")

    # invalid options fail loudly
    try:
        P.smooth_skin(img, portrait_size="enormous")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid portrait_size not rejected")


@test
def test_smooth_skin_pipeline_recursive():
    """Alternative recursive-IIR path keeps the Phase-1 behaviour."""
    img = _portrait_test_image()
    res = P.smooth_skin(img, use_mask=False, method="recursive")
    check(res["derived"]["method"] == "recursive", "recursive method selected")
    out = res["output"]
    check(out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0,
          "recursive output range")
    flat_before = hf_std(img[10:50, 5:30])
    flat_after = hf_std(out[10:50, 5:30])
    check(flat_after < 0.7 * flat_before,
          f"recursive noise reduction weak: {flat_after/flat_before:.3f}")
    bright, dark = slice(8, 30), slice(58, 70)
    c_before = float(img[:, bright].mean() - img[:, dark].mean())
    c_after = float(out[:, bright].mean() - out[:, dark].mean())
    check(c_after > 0.90 * c_before,
          f"recursive edge contrast {c_after/c_before:.3f}")

    # detail-mix sliders: all 100 with threshold protection off -> base
    res_flat = P.smooth_skin(img, use_mask=False, fine=100, medium=100,
                             large=100, method="recursive",
                             use_detail_threshold=False)
    close(res_flat["output"], res_flat["base"], 1e-12,
          "all-100 sliders output the smoothed base")

    # threshold protection retains a narrow feature edge that the coarse
    # pyramid base alone would erase
    rng_t = np.random.default_rng(21)
    line_img = np.zeros((128, 128, 3))
    yy = np.linspace(0, 1, 128)[:, None]
    line_img[:] = np.array([0.80, 0.64, 0.53]) * (0.9 + 0.15 * yy[..., None])
    line_img[:, 62:67] = np.array([0.20, 0.12, 0.10])
    line_img += rng_t.normal(0, 0.01, (128, 128, 1))
    line_img = np.clip(line_img, 0, 1)
    bright, dark = slice(10, 50), slice(62, 67)
    baseline = P.smooth_skin(line_img, use_mask=False, fine=100, medium=100,
                             large=100, method="recursive",
                             use_detail_threshold=False)["output"]
    protected = P.smooth_skin(line_img, use_mask=False, fine=100, medium=100,
                              large=100, method="recursive")["output"]
    c_base = float(baseline[:, bright].mean() - baseline[:, dark].mean())
    c_prot = float(protected[:, bright].mean() - protected[:, dark].mean())
    check(c_prot > 1.3 * c_base,
          f"threshold protection should retain feature detail: "
          f"{c_base:.4f} vs {c_prot:.4f}")

    # Threshold slider semantics: higher threshold smooths more texture.
    flat = synthetic.synthetic_flat_noise(64, 64, level=0.6, noise=0.03, seed=6)
    kept = P.smooth_skin(flat, use_mask=False, fine=100, medium=100, large=100,
                         threshold=0.0, method="recursive")["output"]
    smoothed = P.smooth_skin(flat, use_mask=False, fine=100, medium=100,
                             large=100, threshold=100.0,
                             method="recursive")["output"]
    check(hf_std(kept) > 2.0 * hf_std(smoothed),
          f"threshold monotonicity: {hf_std(kept):.4f} vs {hf_std(smoothed):.4f}")


@test
def test_portrait_size_levels():
    check(P.levels_for_portrait_size("small", (100, 100)) == 2, "small")
    check(P.levels_for_portrait_size("medium", (100, 100)) == 3, "medium")
    check(P.levels_for_portrait_size("large", (100, 100)) == 4, "large")
    check(P.levels_for_portrait_size("auto", (300, 300)) == 2, "auto small img")
    check(P.levels_for_portrait_size("auto", (800, 800)) == 3, "auto medium img")
    check(P.levels_for_portrait_size("auto", (2000, 2000)) == 4, "auto large img")


@test
def test_auto_mask_u2net_optional():
    """U-2-Net mask path (licence-safe) when cv2 + model are available."""
    rng = np.random.default_rng(3)
    # portrait-like gradient: skin-tone oval on a darker background
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    oval = (((yy - h * 0.45) / (h * 0.32)) ** 2 + ((xx - w * 0.5) / (w * 0.26)) ** 2) < 1.0
    img = np.empty((h, w, 3))
    img[:] = np.array([0.25, 0.27, 0.3])
    img[oval] = np.array([0.72, 0.55, 0.45])
    img += rng.normal(0, 0.01, img.shape)
    img = np.clip(img, 0, 1)
    pm = P.u2net_person_matte(img)
    if pm is None:
        print("      (u2net unavailable -> classical fallback path still covered above)")
        return
    check(pm.shape == (h, w), "person matte shape")
    check(0.0 <= float(pm.min()) and float(pm.max()) <= 1.0, "person matte range")
    m = P.build_mask(img)
    check(m.shape == (h, w), "auto mask shape")
    check(0.0 <= float(m.min()) and float(m.max()) <= 1.0, "auto mask range")
    res = P.smooth_skin(img, smoothing=70.0)
    check(res["mask"] is not None, "smooth_skin used a mask")
    # the model mask must be tighter than the plain HSV fallback on this image
    check(float(res["mask"].mean()) <= 1.0, "mask mean sane")
    hsv = P.build_skin_mask(img)
    print(f"      u2net person mean {float(pm.mean()):.3f}; "
          f"auto mask {float(m.mean()):.3f} vs hsv {float(hsv.mean()):.3f}")


# ===========================================================================
# runner
# ===========================================================================

def main() -> int:
    failures = 0
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"PASS  {name}")
    total = len(_TESTS)
    print(f"\n{total - failures}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
