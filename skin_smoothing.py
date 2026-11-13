"""Skin smoothing: a self-contained NumPy implementation.

A multi-scale, edge-preserving skin-smoothing pipeline:

    load image (8/16-bit RGB)
        -> automatic skin mask (U-2-Net person matte x classical skin term)
        -> multi-scale bilateral-grid smoothing (splat / blur / normalise / slice)
        -> detail-layer decomposition and level mix
        -> masked blend with opacity
        -> sharpen / soften  (soft-clipped unsharp with a 1/2 re-blend)
        -> tone adjustments (warmth / brightness / contrast)
        -> save image (8/16-bit RGB)

Design goals
------------
* dependency-light: only ``numpy`` + ``Pillow`` are required (OpenCV is optional,
  used only for the U-2-Net person matte);
* numerically faithful to the pipeline described in ``docs/algorithm.md``;
* 8-bit and 16-bit pipelines share the same internal math.

Filter primitives
-----------------
* :func:`gaussian_blur` -- separable Gaussian, edge-clamped (float64, 16-bit safe).
* :func:`recursive_bilateral` -- Yang 2012/2015 O(1) recursive bilateral filter:
  per axis, a causal and an anti-causal first-order recursion driven by the
  neighbour intensity difference::

      w_n      = exp(-rngdist(I_n, I_{n+1}) / sigma_r)     (L1 range distance)
      alpha    = exp(-sqrt(2) / sigma_s)                   (spatial decay)
      y_n      = (1-alpha) * I_n + alpha * w_n * y_{n-1}   (causal pass)
      f_n      = (1-alpha)       + alpha * w_n * f_{n-1}   (normaliser)
      y_n^out  = (1/n) * (y_n^fwd + y_n^bwd) / (f_n^fwd + f_n^bwd)

  ``range_mode='gauss'`` instead uses ``exp(-0.5*(rngdist/sigma_r)^2)``.
* :func:`blur121` -- separable (1,2,1)/4 grid blur (bilateral-grid Step 2).
* :func:`ycc_pack` / :func:`ycc_unpack` -- packed 16-bit YCC intermediate
  format used by the grid stages.
* :func:`bilateral_grid_smooth` -- one complete splat/blur/normalise/slice pass.
* :func:`bilateral_grid_pyramid` -- coarse-to-fine multi-scale driver.
* :func:`apply_sharp_and_soft` -- soft-clipped unsharp + 1/2 re-blend.
* :func:`build_mask` -- automatic mask: U-2-Net person matte (Apache-2.0) x
  classical skin term, with an HSV fallback when OpenCV/model is unavailable.
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # image i/o
    "load_image", "save_image", "png_bit_depth", "to_uint", "from_uint",
    # filters
    "gaussian_kernel", "gaussian_blur",
    "recursive_bilateral", "recursive_bilateral_x", "recursive_bilateral_y",
    "downsample2x2", "upsample_bilinear", "bilateral_pyramid",
    "detail_decompose", "combine_levels", "threshold_gain",
    "soft_matte", "mask_blur", "blend_masked", "build_skin_mask", "rgb_to_hsv",
    "unsharp", "adjust_tone",
    # modelled internal format + bilateral-grid chain
    "ycc_unpack", "ycc_pack", "blur121", "lerp12", "FP12_MAX",
    "bilateral_grid_smooth", "bilateral_grid_pyramid", "apply_sharp_and_soft",
    # top level
    "SmoothParams", "derive_algorithm_params", "levels_for_portrait_size",
    "smooth_skin", "apply_smoothing",
]

# ---------------------------------------------------------------------------
# Consensus constants (initial model defaults; final model pins them to the analysis)
# ---------------------------------------------------------------------------

#: Range-weight kernel.  ``'gauss'`` = exp(-0.5*(d/sigma_r)**2) (design notes);
#: ``'exp'``  = exp(-d/sigma_r) (Yang 2012 self-contained implementation, L1 metric).
DEFAULT_RANGE_MODE = "gauss"

#: How ``sigma_s`` is interpreted.  ``'fraction'`` follows the Yang reference:
#: alpha = exp(-sqrt(2) / (sigma_s * axis_length)), i.e. sigma_s is a fraction
#: of the axis length (scale invariant across image sizes -- convenient for a
#: pyramid).  ``'pixels'``: alpha = exp(-sqrt(2) / sigma_s_px).
DEFAULT_SIGMA_S_MODE = "fraction"

_EPS = 1e-12


# ===========================================================================
# 1. Image I/O -- 8-bit and 16-bit RGB / grayscale PNG (16-bit RGB is decoded
#    manually because Pillow silently truncates 16-bit RGB PNGs to 8-bit).
# ===========================================================================

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png_chunks(data: bytes) -> Iterable[Tuple[bytes, bytes]]:
    """Yield ``(chunk_type, chunk_data)`` for a PNG byte string (no CRC check)."""
    if data[:8] != _PNG_SIG:
        raise ValueError("not a PNG file")
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        cdata = data[pos + 8:pos + 8 + length]
        yield ctype, cdata
        pos += 12 + length
        if ctype == b"IEND":
            break


def _png_ihdr(data: bytes) -> Tuple[int, int, int, int]:
    """Return ``(width, height, bit_depth, color_type)`` from a PNG byte string."""
    for ctype, cdata in _png_chunks(data):
        if ctype == b"IHDR":
            w, h, depth, ctype_num, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", cdata)
            if interlace:
                raise NotImplementedError("interlaced PNG not supported")
            return w, h, depth, ctype_num
    raise ValueError("PNG has no IHDR chunk")


def png_bit_depth(path: Union[str, os.PathLike]) -> int:
    """Return the PNG bit depth (8 or 16) of ``path`` (raises for non-PNG)."""
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != _PNG_SIG:
        raise ValueError(f"{path!r} is not a PNG")
    return _png_ihdr(head)[2] if len(head) >= 33 else 8


def _unfilter_png(raw: bytes, height: int, width: int, channels: int,
                  bit_depth: int) -> np.ndarray:
    """Undo PNG scanline filters.  Returns ``uint8`` array (height, stride)."""
    bpp = max(1, channels * bit_depth // 8)          # bytes per pixel
    stride = width * channels * bit_depth // 8       # bytes per scanline
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw[pos:pos + stride], dtype=np.uint8).astype(np.int32)
        pos += stride
        if line.size != stride:
            raise ValueError("truncated PNG IDAT data")
        if ftype == 0:                              # None
            rec = line
        elif ftype == 1:                            # Sub -- per byte-lane cumsum
            rec = np.empty_like(line)
            for k in range(bpp):
                rec[k::bpp] = np.cumsum(line[k::bpp]) % 256
        elif ftype == 2:                            # Up
            rec = (line + prev) % 256
        elif ftype == 3:                            # Average
            rec = np.empty_like(line)
            for i in range(stride):
                left = rec[i - bpp] if i >= bpp else 0
                rec[i] = (line[i] + ((left + prev[i]) >> 1)) % 256
        elif ftype == 4:                            # Paeth
            rec = np.empty_like(line)
            for i in range(stride):
                a = rec[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                rec[i] = (line[i] + pr) % 256
        else:
            raise ValueError(f"unsupported PNG filter type {ftype}")
        out[y] = rec.astype(np.uint8)
        prev = out[y].astype(np.int32)
    return out


def _decode_png16(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode a 16-bit PNG (gray/RGB/RGBA) to ``float64`` in [0, 1]."""
    width, height, depth, ctype_num = _png_ihdr(data)
    if depth != 16:
        raise ValueError("expected 16-bit PNG")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype_num)
    if channels is None:
        raise ValueError(f"unsupported PNG color type {ctype_num}")
    idat = b"".join(cdata for ctype, cdata in _png_chunks(data) if ctype == b"IDAT")
    raw = zlib.decompress(idat)
    rows = _unfilter_png(raw, height, width, channels, depth)
    b = rows.reshape(height, width, channels, 2).astype(np.uint16)
    vals = (b[..., 0].astype(np.uint32) << 8) | b[..., 1].astype(np.uint32)
    arr = vals.astype(np.float64) / 65535.0
    if channels == 1:
        arr = arr[..., 0]
    return arr, depth


def _encode_png16(arr: np.ndarray) -> bytes:
    """Encode a float [0,1] gray/RGB/RGBA array as a 16-bit PNG (filter 0)."""
    a = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    a16 = np.round(a * 65535.0).astype(np.uint16)
    if a16.ndim == 2:
        channels, ctype_num = 1, 0
        planes = a16[..., None]
    elif a16.ndim == 3 and a16.shape[2] in (3, 4):
        channels = a16.shape[2]
        ctype_num = 2 if channels == 3 else 6
        planes = a16
    else:
        raise ValueError("only (H,W), (H,W,3) and (H,W,4) images are supported")

    height, width = a16.shape[:2]
    raw = bytearray()
    for y in range(height):
        raw.append(0)                      # filter type 0 (None)
        raw += planes[y].astype(">u2").tobytes()

    def chunk(ctype: bytes, cdata: bytes) -> bytes:
        body = ctype + cdata
        return (struct.pack(">I", len(cdata)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 16, ctype_num, 0, 0, 0)
    return (_PNG_SIG + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


def load_image(path: Union[str, os.PathLike]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load an image as ``float64`` in [0, 1].

    Returns ``(array, meta)`` where ``array`` is ``(H, W)`` for grayscale or
    ``(H, W, 3)`` for RGB; ``meta`` carries ``bit_depth`` (8/16), ``mode``
    (``'L'``/``'RGB'``), ``path`` and ``height``/``width``.

    16-bit PNGs are decoded with the local codec (Pillow truncates 16-bit RGB
    to 8-bit); everything else goes through Pillow.  Alpha channels are dropped
    (the pipeline's Y24/Y48 paths are 3-channel).
    """
    path = str(path)
    bit_depth = 8
    if path.lower().endswith(".png"):
        with open(path, "rb") as f:
            head = f.read(33)
        if head[:8] == _PNG_SIG and len(head) >= 33:
            bit_depth = _png_ihdr(head)[2]
    if bit_depth == 16:
        with open(path, "rb") as f:
            arr, _ = _decode_png16(f.read())
    else:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                a = np.asarray(im).astype(np.float64)
                bit_depth = 16
                arr = a / 65535.0
            else:
                if im.mode in ("RGBA", "LA", "P", "CMYK", "YCbCr"):
                    im = im.convert("RGB")
                elif im.mode == "1":
                    im = im.convert("L")
                arr = np.asarray(im).astype(np.float64) / 255.0
                bit_depth = 8
    mode = "L" if arr.ndim == 2 else "RGB"
    meta = {
        "path": path,
        "bit_depth": bit_depth,
        "mode": mode,
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
    }
    return np.ascontiguousarray(arr, dtype=np.float64), meta


def save_image(path: Union[str, os.PathLike], img: np.ndarray,
               bit_depth: int = 8) -> str:
    """Save a float [0,1] image as 8-bit or 16-bit PNG.

    ``img`` may be ``(H, W)`` (gray) or ``(H, W, 3)`` (RGB).  Values are clipped
    to [0,1] and rounded to the target bit depth.  16-bit output uses the local
    PNG16 encoder for full precision (Pillow cannot write 16-bit RGB).
    """
    path = str(path)
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError("img must be (H,W) or (H,W,3)")
    if bit_depth == 16:
        with open(path, "wb") as f:
            f.write(_encode_png16(arr))
        return path
    if bit_depth != 8:
        raise ValueError("bit_depth must be 8 or 16")
    u8 = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if u8.ndim == 3 and u8.shape[2] == 1:
        u8 = u8[..., 0]
    from PIL import Image
    Image.fromarray(u8).save(path)
    return path


def to_uint(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Quantise float [0,1] image to ``uint8``/``uint16`` (rounded, clipped)."""
    scale = (1 << bit_depth) - 1
    return np.round(np.clip(np.asarray(img, np.float64), 0.0, 1.0)
                    * scale).astype(np.uint8 if bit_depth == 8 else np.uint16)


def from_uint(img: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """Convert integer image to ``float64`` in [0, 1]."""
    scale = float((1 << bit_depth) - 1)
    return np.asarray(img, dtype=np.float64) / scale


def _as3(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return ``(H, W, C)`` view of ``img`` and whether it was 2-D."""
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a[..., None], True
    if a.ndim == 3:
        return a, False
    raise ValueError("image must be 2-D (H,W) or 3-D (H,W,C)")


def _restore(a: np.ndarray, was2: bool) -> np.ndarray:
    return a[..., 0] if was2 else a


# ===========================================================================
# 2. Separable Gaussian blur (edge-clamped)
# ===========================================================================

def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Normalised 1-D Gaussian kernel, ``2*ceil(truncate*sigma)+1`` taps."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def _convolve_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1-D correlation of ``arr`` with ``kernel`` along ``axis``, edge-clamped."""
    n = kernel.size
    if n == 1:
        return arr.copy()
    r = n // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    ap = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr)
    length = arr.shape[axis]
    for i, k in enumerate(kernel):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + length)
        out += k * ap[tuple(sl)]
    return out


def gaussian_blur(img: np.ndarray, sigma: float,
                  truncate: float = 4.0) -> np.ndarray:
    """Separable Gaussian blur of a float image, replicate (edge) padding.

    Works on ``(H, W)`` and ``(H, W, C)`` arrays and on float values beyond
    [0, 1] (16-bit-safe: no clipping is performed).  Complexity ``O(H*W*K)``
    with ``K = 2*ceil(truncate*sigma)+1`` taps per axis.
    """
    a = np.asarray(img, dtype=np.float64)
    if sigma is None or sigma <= 0.0:
        return a.copy()
    k = gaussian_kernel(sigma, truncate)
    out = _convolve_axis(a, k, 0)
    out = _convolve_axis(out, k, 1)
    return out


# ===========================================================================
# 3. Recursive bilateral filter (Yang 2012/2015, O(1) in sigma)
# ===========================================================================

def _range_distance(diff: np.ndarray) -> np.ndarray:
    """Scalar range distance from per-channel intensity differences.

    Yang's self-contained implementation uses ``rng = (2|dr| + |dg| + |db|) / 4``
    (a luma-weighted L1 metric) so that all three channels share one weight.
    """
    if diff.shape[-1] == 1:
        return np.abs(diff[..., 0])
    if diff.shape[-1] == 3:
        weights = np.array([2.0, 1.0, 1.0], dtype=np.float64) / 4.0
        return (np.abs(diff) * weights).sum(axis=-1)
    return np.abs(diff).mean(axis=-1)


def _range_weight(dist: np.ndarray, sigma_r: float,
                  range_mode: str) -> np.ndarray:
    """Range kernel applied to a non-negative intensity distance."""
    if sigma_r is None or float(sigma_r) <= 0.0:
        # Degenerate limit: only exactly equal neighbours get any weight, so
        # the filter reduces to the identity (verified in the test suite).
        return (dist == 0.0).astype(np.float64)
    sigma_r = float(sigma_r)
    if range_mode == "gauss":
        return np.exp(-0.5 * (dist / sigma_r) ** 2)
    if range_mode == "exp":
        return np.exp(-dist / sigma_r)
    raise ValueError(f"unknown range_mode {range_mode!r}")


def _spatial_alpha(sigma_s: float, length: int, sigma_s_mode: str) -> float:
    """Recursion decay ``alpha = exp(-sqrt(2)/q)`` (Yang 2012)."""
    sigma_s = float(sigma_s)
    if sigma_s <= 0.0:
        return 0.0
    if sigma_s_mode == "fraction":
        q = sigma_s * float(length)           # sigma_s is a fraction of the axis
    elif sigma_s_mode == "pixels":
        q = sigma_s                           # sigma_s in pixels
    else:
        raise ValueError(f"unknown sigma_s_mode {sigma_s_mode!r}")
    alpha = math.exp(-math.sqrt(2.0) / q)
    return min(alpha, 1.0 - 1e-9)             # keep 1-alpha representable


def recursive_bilateral_axis(
    img: np.ndarray,
    axis: int,
    sigma_s: float,
    sigma_r: float,
    *,
    range_mode: str = DEFAULT_RANGE_MODE,
    sigma_s_mode: str = DEFAULT_SIGMA_S_MODE,
    normalize: bool = True,
    per_channel: bool = False,
    backward_mode: str = "range",
) -> np.ndarray:
    """Recursive bilateral filter along **one** axis.

    Implements the causal + anti-causal first-order recursions of Yang (ECCV
    2012) including the normaliser map ``f`` that makes the filter DC-preserving
    (and exactly the identity when ``sigma_r -> 0``).  All rows (``axis=1``) or
    columns (``axis=0``) are processed simultaneously: the sequential recursion
    runs over the filter axis only, with the perpendicular axis vectorised.

    Parameters
    ----------
    axis : {0, 1}
        ``1`` filters along rows (pipeline ``recursive pass along rows``),
        ``0`` filters along columns (``recursive pass along columns``).
    sigma_s : float
        Spatial parameter; in ``'fraction'`` mode it is a fraction of the axis
        length (Yang reference: ``alpha = exp(-sqrt(2)/(sigma_s*N))``), in
        ``'pixels'`` mode it is the decay length in pixels.
    sigma_r : float
        Range parameter in normalised intensity units ([0,1] scale).
    range_mode : {'gauss', 'exp'}
        ``gauss``: ``exp(-0.5*(d/sigma_r)^2)``; ``exp``: ``exp(-d/sigma_r)``.
    normalize : bool
        Divide by the accumulated weight map ``f`` (required for DC
        preservation; test suite asserts identity at ``sigma_r -> 0``).
    per_channel : bool
        Compute a separate range weight per channel instead of one luma-weighted
        weight for all channels.
    backward_mode : {'range', 'half_blend'}
        ``'range'``: symmetric range-weighted anti-causal pass (Yang 2012
        reference; default, identity-preserving at ``sigma_r -> 0``).
        ``'half_blend'``: the literal modelled pass from
        ``docs/algorithm.md`` §4 -- ``O[i] = 0.5*(x[i]+O[i+1])``,
        ``W[i] = 0.5*(W[i+1]+1)`` -- exposed for numerical verification; it is
        a plain one-sided average, not an IIR pass, so it does not reduce to
        the identity at ``sigma_r -> 0``.

    Complexity: ``O(N * M * C)`` flops for ``N`` axis length x ``M`` perpendicular
    pixels x ``C`` channels, independent of ``sigma_s``; memory ``O(N*M*C)``.
    The Python-level loop runs ``N`` times (vectorised over ``M*C``).
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    x = np.moveaxis(np.asarray(img, dtype=np.float64), axis, 1)
    was2 = x.ndim == 2
    if was2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError("image must be 2-D or 3-D")
    other, n, channels = x.shape

    alpha = _spatial_alpha(sigma_s, n, sigma_s_mode)
    if alpha <= 0.0 or n <= 1:
        out = np.moveaxis(x, 1, axis)
        return _restore(out, was2)
    inv_alpha = 1.0 - alpha

    # Range weights from *neighbour differences of the input* (Yang reference
    # compares consecutive original samples, not filtered ones).
    diff = x[:, 1:, :] - x[:, :-1, :]                       # (other, n-1, C)
    if per_channel:
        w = _range_weight(diff, sigma_r, range_mode)        # (other, n-1, C)
    else:
        w = _range_weight(_range_distance(diff), sigma_r, range_mode)
        w = w[..., None]                                    # (other, n-1, 1)

    y_fwd = np.empty_like(x)
    f_fwd = np.empty_like(x)
    y_bwd = np.empty_like(x)
    f_bwd = np.empty_like(x)

    y_fwd[:, 0, :] = x[:, 0, :]
    f_fwd[:, 0, :] = 1.0
    for i in range(1, n):
        a = alpha * w[:, i - 1, :]
        y_fwd[:, i, :] = inv_alpha * x[:, i, :] + a * y_fwd[:, i - 1, :]
        f_fwd[:, i, :] = inv_alpha + a * f_fwd[:, i - 1, :]

    if backward_mode == "half_blend":
        # Literal modelled backward 1/2-blend (docs/algorithm.md §4):
        #   O[i] = 0.5*(x[i] + O[i+1]);  W[i] = 0.5*(W[i+1] + 1)
        # with O[N-1] = 0.5*(x[N-1] + O_f[N-1]), W[N-1] = 0.5*(W_f[N-1] + 1).
        y_bwd[:, -1, :] = 0.5 * (x[:, -1, :] + y_fwd[:, -1, :])
        f_bwd[:, -1, :] = 0.5 * (f_fwd[:, -1, :] + 1.0)
        for i in range(n - 2, -1, -1):
            y_bwd[:, i, :] = 0.5 * (x[:, i, :] + y_bwd[:, i + 1, :])
            f_bwd[:, i, :] = 0.5 * (f_bwd[:, i + 1, :] + 1.0)
        # result = O / W (the forward pass only seeds the boundary)
        out = y_bwd / np.where(f_bwd > _EPS, f_bwd, 1.0)
    elif backward_mode == "range":
        y_bwd[:, -1, :] = x[:, -1, :]
        f_bwd[:, -1, :] = 1.0
        for i in range(n - 2, -1, -1):
            a = alpha * w[:, i, :]
            y_bwd[:, i, :] = inv_alpha * x[:, i, :] + a * y_bwd[:, i + 1, :]
            f_bwd[:, i, :] = inv_alpha + a * f_bwd[:, i + 1, :]

        # Forward + backward passes are averaged (zero-phase); the normaliser is
        # averaged the same way, so the 0.5 factors cancel in the ratio.
        y = y_fwd + y_bwd
        f = f_fwd + f_bwd
        if normalize:
            out = y / np.where(f > _EPS, f, 1.0)
        else:
            out = 0.5 * y
    else:
        raise ValueError(f"unknown backward_mode {backward_mode!r}")
    out = np.moveaxis(out, 1, axis)
    return _restore(out, was2)


def recursive_bilateral_x(img: np.ndarray, sigma_s: float, sigma_r: float,
                          backward_mode: str = "range",
                          **kwargs: Any) -> np.ndarray:
    """Row-direction recursive BF (pipeline ``recursive pass along rows``)."""
    return recursive_bilateral_axis(img, 1, sigma_s, sigma_r,
                                    backward_mode=backward_mode, **kwargs)


def recursive_bilateral_y(img: np.ndarray, sigma_s: float, sigma_r: float,
                          backward_mode: str = "range",
                          **kwargs: Any) -> np.ndarray:
    """Column-direction recursive BF (pipeline ``recursive pass along columns``)."""
    return recursive_bilateral_axis(img, 0, sigma_s, sigma_r,
                                    backward_mode=backward_mode, **kwargs)


def recursive_bilateral(
    img: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    *,
    range_mode: str = DEFAULT_RANGE_MODE,
    sigma_s_mode: str = DEFAULT_SIGMA_S_MODE,
    normalize: bool = True,
    per_channel: bool = False,
    backward_mode: str = "range",
) -> np.ndarray:
    """Two-pass (X then Y) recursive bilateral filter, O(1) in sigma.

    This is the separable approximation of the 2-D bilateral filter used by the
    pipeline: horizontal recursion first, then vertical (matching the reference
    C++ ordering).  ``sigma_s`` units follow ``sigma_s_mode`` (default
    ``'fraction'``, Yang reference convention).
    """
    kw = dict(range_mode=range_mode, sigma_s_mode=sigma_s_mode,
              normalize=normalize, per_channel=per_channel,
              backward_mode=backward_mode)
    out = recursive_bilateral_axis(img, 1, sigma_s, sigma_r, **kw)
    out = recursive_bilateral_axis(out, 0, sigma_s, sigma_r, **kw)
    return out


# ===========================================================================
# 4. Image pyramid (pipeline BilateralStep0..Step4)
# ===========================================================================

def downsample2x2(img: np.ndarray) -> np.ndarray:
    """2x2 area-average downsample (``CScaleDown2AverageProcessor``).

    Odd dimensions are edge-padded before averaging, so the coarse level is
    ``ceil(H/2) x ceil(W/2)``.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    ph, pw = h % 2, w % 2
    if ph or pw:
        pad = ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2)
        a = np.pad(a, pad, mode="edge")
    h2, w2 = a.shape[0] // 2, a.shape[1] // 2
    blocks = a.reshape(h2, 2, w2, 2, *a.shape[2:])
    return blocks.mean(axis=(1, 3))


def upsample_bilinear(img: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear upsample to ``out_shape = (H, W)`` (half-pixel centres, clamp)."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (w / out_w) - 0.5
    ys = np.clip(ys, 0.0, h - 1.0)
    xs = np.clip(xs, 0.0, w - 1.0)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    if a.ndim == 3:
        fy = fy[..., None]
        fx = fx[..., None]
    top = a[y0][:, x0] * (1.0 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1.0 - fx) + a[y1][:, x1] * fx
    return top * (1.0 - fy) + bot * fy


def bilateral_pyramid(
    img: np.ndarray,
    levels: int = 3,
    iters: int = 2,
    sigma_s: float = 0.115,
    sigma_r: float = 0.128,
    *,
    range_mode: str = DEFAULT_RANGE_MODE,
    sigma_s_mode: str = DEFAULT_SIGMA_S_MODE,
    per_channel: bool = False,
) -> List[np.ndarray]:
    """Multi-scale recursive-bilateral pyramid.

    Mirrors the pipeline driver ``buffer zeroing stage`` ...
    ``Step4_Interpolate``.  Each level filters the *previous coarse buffer*
    (so level ``k`` really works at resolution ``1/2**(k+1)``), then upsamples
    back for the level output::

        low = I
        for level in 0..levels-1:
            low  = downsample2x2(low)              # Step1 (2x2 area average)
            low  = recursive_bilateral^iters(low)  # Step2 (convolution iters)
            append(upsample_bilinear(low, I.shape))  # Step3/Step4 (interpolate)
        # low is now the coarsest (most smoothed) working buffer

    Returns the list of full-resolution smoothed images ``[s0, s1, s2, ...]``
    where ``s0`` is the **least** smoothed (1/2 resolution bilateral) and the
    last entry the **most** smoothed base (1/2**levels).  ``levels == 0``
    returns ``[I]`` (identity).

    Because ``sigma_s_mode='fraction'`` is scale invariant, the same sigma_s
    means the same relative support on every pyramid level.
    """
    a = np.asarray(img, dtype=np.float64)
    if levels < 0:
        raise ValueError("levels must be >= 0")
    if levels == 0:
        return [a.copy()]
    iters = max(int(iters), 0)
    out: List[np.ndarray] = []
    low = a
    full_shape = a.shape[:2]
    for _ in range(int(levels)):
        low = downsample2x2(low)
        for _ in range(iters):
            low = recursive_bilateral(low, sigma_s, sigma_r,
                                      range_mode=range_mode,
                                      sigma_s_mode=sigma_s_mode,
                                      per_channel=per_channel)
        out.append(upsample_bilinear(low, full_shape))
    return out


# ===========================================================================
# 5. Multi-scale detail decomposition and combination
# ===========================================================================

def detail_decompose(
    img: np.ndarray, smoothed_by_level: Sequence[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Laplacian-pyramid decomposition over the smoothed level list.

    ``details[k] = level_{k-1} - level_k`` (with ``level_{-1} = img``) and
    ``base = level_{-1}`` (most-smoothed image), so::

        img = base + sum(details)
    """
    a = np.asarray(img, dtype=np.float64)
    if len(smoothed_by_level) == 0:
        return a.copy(), []
    base = np.asarray(smoothed_by_level[-1], dtype=np.float64)
    details: List[np.ndarray] = []
    prev = a
    for level in smoothed_by_level:
        details.append(prev - np.asarray(level, dtype=np.float64))
        prev = level
    return base, details


def _weights_to_list(weights: Union[Mapping[str, float], Sequence[float], float],
                     n: int) -> List[float]:
    """Normalise a weights argument to a list of length ``n``.

    Accepts a scalar, a sequence, or a mapping with keys ``fine``/``medium``/
    ``large`` (in that order, missing keys default to 1.0).
    """
    if isinstance(weights, Mapping):
        keys = ("fine", "medium", "large")
        vals = [float(weights.get(k, 1.0)) for k in keys[:n]]
        if len(vals) < n:
            vals += [1.0] * (n - len(vals))
        return vals
    if np.isscalar(weights):
        return [float(weights)] * n
    vals = [float(v) for v in weights]
    if len(vals) != n:
        raise ValueError(f"expected {n} weights, got {len(vals)}")
    return vals


def combine_levels(
    base: np.ndarray,
    details: Sequence[np.ndarray],
    weights: Union[Mapping[str, float], Sequence[float], float],
    gains: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Recombine base + weighted detail layers.

    ``out = base + sum_k(gains[k] * details[k])``

    ``weights`` follow the Fine/Medium/Large sliders (0 = keep all detail at
    that scale -> no smoothing, 1 = drop that detail layer); ``gains`` are
    optional per-pixel **effective** gain maps produced by
    :func:`threshold_gain` (in ``[weight, 1]``) that already include the slider
    weight, so ``weights`` is ignored when ``gains`` is given.  With
    ``weights == 1`` and no ``gains`` the original image is reconstructed
    exactly.
    """
    d = list(details)
    w = _weights_to_list(weights, len(d))
    out = np.array(base, dtype=np.float64, copy=True)
    for k, (detail, wk) in enumerate(zip(d, w)):
        g = wk if gains is None else gains[k]
        out = out + g * np.asarray(detail, dtype=np.float64)
    return out


def threshold_gain(detail: np.ndarray, threshold: float, weight: float,
                   softness: Optional[float] = None) -> np.ndarray:
    """Per-pixel **effective** detail weight in ``[weight, 1]``.

    Low-contrast detail (``|d| <= threshold``) is treated as skin texture and
    is attenuated to the smoothing ``weight``; high-contrast detail
    (``|d| >= threshold + softness``) is treated as a facial feature edge and
    is kept at 1.  The transition is a smoothstep so there is no visible seam.
    Pass the result straight to :func:`combine_levels` as ``gains``.

    This models the pipeline's ``Threshold`` slider (initial model assumption: the
    slider maps to a detail-contrast threshold; final model may instead map it onto
    the bilateral range sigma -- see ``derive_algorithm_params``).
    """
    if threshold is None or float(threshold) <= 0.0:
        one = np.ones_like(np.asarray(detail, dtype=np.float64))
        return one
    d = np.abs(np.asarray(detail, dtype=np.float64))
    if d.ndim == 3:
        d = d.max(axis=-1, keepdims=True)
    if softness is None:
        softness = max(float(threshold) * 0.5, 1e-4)
    t = np.clip((d - float(threshold)) / max(softness, 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return float(weight) + (1.0 - float(weight)) * t


# ===========================================================================
# 6. Mask utilities
# ===========================================================================

def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 == edge0:
        return (x >= edge0).astype(np.float64)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_matte(values: np.ndarray, lo: float, hi: float,
               feather: float = 0.0) -> np.ndarray:
    """Soft band matte: 1 inside ``[lo, hi]``, 0 outside, ramp width ``feather``.

    ``feather`` is in the same units as ``values`` and is split evenly on both
    band edges (a value exactly at ``lo``/``hi`` is inside with weight 1).
    """
    v = np.asarray(values, dtype=np.float64)
    if feather is None or feather <= 0.0:
        return ((v >= lo) & (v <= hi)).astype(np.float64)
    half = 0.5 * float(feather)
    lower = _smoothstep(v, lo - half, lo + half)
    upper = 1.0 - _smoothstep(v, hi - half, hi + half)
    return lower * upper


def mask_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian feathering of a float mask (edge-clamped, no clipping)."""
    return gaussian_blur(np.asarray(mask, dtype=np.float64), sigma)


def blend_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                 opacity: float = 1.0) -> np.ndarray:
    """Alpha-blend ``a -> b`` by ``mask * opacity``.

    ``mask`` may be ``(H, W)`` (broadcast over channels) or ``(H, W, 1)``;
    ``opacity`` is the pipeline's mask-opacity parameter in [0, 1].
    ``opacity=0`` returns ``a`` exactly, ``opacity=1`` with a mask of 1 returns
    ``b`` exactly.
    """
    m = np.clip(np.asarray(mask, dtype=np.float64) * float(opacity), 0.0, 1.0)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if m.ndim == 2 and aa.ndim == 3:
        m = m[..., None]
    return aa * (1.0 - m) + bb * m


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Vectorised RGB -> HSV for a float image in [0, 1] (H, W, 3)."""
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_to_hsv expects (H, W, 3)")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb[..., :3], axis=-1)
    mn = np.min(rgb[..., :3], axis=-1)
    delta = mx - mn
    safe = np.where(delta > _EPS, delta, 1.0)
    rc = (mx - r) / safe
    gc = (mx - g) / safe
    bc = (mx - b) / safe
    h = np.where(mx == r, bc - gc,
                 np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = np.where(delta > _EPS, h, 0.0)
    s = np.where(mx > _EPS, delta / np.where(mx > _EPS, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1)


def build_skin_mask(
    img: np.ndarray,
    hue_range: Tuple[float, float] = (0.0, 0.135),
    sat_range: Tuple[float, float] = (0.06, 0.80),
    val_range: Tuple[float, float] = (0.18, 1.0),
    feather: float = 0.02,
) -> np.ndarray:
    """Classical HSV skin-colour matte (H, W) in [0, 1].

    This is the fallback used when the optional U-2-Net person matte is not
    available.  The ranges are empirical defaults; tune them for your footage.
    """
    hsv = rgb_to_hsv(img)
    m = soft_matte(hsv[..., 0], hue_range[0], hue_range[1], feather)
    m *= soft_matte(hsv[..., 1], sat_range[0], sat_range[1], feather)
    m *= soft_matte(hsv[..., 2], val_range[0], val_range[1], 2.0 * feather)
    return m


# ---------------------------------------------------------------------------
# 6b. Automatic mask: U-2-Net person matte (Apache-2.0) x classical skin term
# ---------------------------------------------------------------------------
#   person matte : U-2-Net-lite (u2netp.onnx, Apache-2.0, 4.6 MB), 320x320
#                  input, ImageNet normalisation, min-max normalised
#   skin term    : HSV matte and/or an HSL spline skin-tone matte, auto-centred
#                  on the person region
#   gating       : mask = person x skin, then a separable Gaussian feather
#
# Everything is optional: without OpenCV or the model file this falls back to
# :func:`build_skin_mask` (HSV only).

_U2NET_CACHE = {}


def _u2net_model_path(model_path=None):
    if model_path and os.path.exists(model_path):
        return model_path
    env = os.environ.get("U2NET_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(root, "models", "u2netp.onnx")
    return cand if os.path.exists(cand) else None


def u2net_person_matte(img, model_path=None, size=320):
    """Person/foreground matte from U-2-Net-lite, or ``None`` if unavailable.

    Requires OpenCV (``cv2.dnn``) and ``u2netp.onnx`` (see ``models/README.md``).
    U-2-Net is Apache-2.0.
    """
    try:
        import cv2  # optional dependency
    except Exception:
        return None
    path = _u2net_model_path(model_path)
    if path is None:
        return None
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    u8 = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    net = _U2NET_CACHE.get(path)
    if net is None:
        try:
            net = cv2.dnn.readNetFromONNX(path)
        except Exception:
            return None
        _U2NET_CACHE[path] = net
    inp = cv2.resize(u8, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    blob = ((inp - mean) / std).transpose(2, 0, 1)[None]
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    d = np.asarray(outs[0])[0, 0]
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-9)
    d = cv2.resize(d, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 0.0, 1.0)


def _rgb_to_hsl16(img):
    """RGB [0,1] -> HSL in the pipeline's 16-bit integer domain."""
    rgb = np.asarray(img, np.float64)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], -1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb[..., :3], axis=-1)
    mn = np.min(rgb[..., :3], axis=-1)
    l = (mx + mn) / 2.0
    d = mx - mn
    s = np.where(d < 1e-9, 0.0, d / np.maximum(1e-9, 1.0 - np.abs(2 * l - 1)))
    h = np.zeros_like(l)
    m = d > 1e-9
    hr = np.where(mx == r, ((g - b) / np.maximum(d, 1e-9)) % 6.0, 0.0)
    hg = np.where((mx == g) & (mx != r), (b - r) / np.maximum(d, 1e-9) + 2.0, 0.0)
    hb = np.where((mx == b) & (mx != r) & (mx != g), (r - g) / np.maximum(d, 1e-9) + 4.0, 0.0)
    h = np.where(m, hr + hg + hb, 0.0) / 6.0
    return ((h * 65535.0).astype(np.float32),
            (np.clip(s, 0, 1) * 65535.0).astype(np.float32),
            (np.clip(l, 0, 1) * 65535.0).astype(np.float32))


def _spline_lut(center, half_width, wrap=False):
    """256-entry soft acceptance curve (reconstruction of the pipeline's spline)."""
    x = np.arange(256, dtype=np.float64) * 256.0
    hw = max(1.0, float(half_width))
    if wrap:
        d = np.abs(x - center)
        d = np.minimum(d, 65536.0 - d)
    else:
        d = np.abs(x - center)
    t = np.clip((d - hw) / 0x4000, 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


def auto_skin_tone_mask(img, person=None, hue_spread=100.0, latitude=50.0):
    """Reconstruction of the pipeline's HSL spline skin-tone mask, auto-centred.

    Half-width follows the modelled formula ``h = L/50*(1-w) + L/5*w`` with
    ``w = hue_spread/128``; the colour centres are the medians of the person
    region (the pipeline gets equivalent values from its face/skin sampling).
    Returns ``None`` when the estimated range is degenerate.
    """
    H, S, L = _rgb_to_hsl16(img)
    H, S, L = H.astype(np.float64), S.astype(np.float64), L.astype(np.float64)
    if person is None:
        hh, ww = L.shape
        sel = np.zeros_like(L, bool)
        sel[hh // 4: 3 * hh // 4, ww // 4: 3 * ww // 4] = True
    else:
        sel = np.asarray(person) > 0.5
    sel &= (L > 0.03 * 65535) & (L < 0.80 * 65535)
    if sel.sum() < 200:
        return None
    cH = float(np.median(H[sel]))
    cS = float(np.median(S[sel]))
    cL = float(np.median(L[sel]))
    w = float(np.clip(hue_spread, 0.0, 128.0)) / 128.0
    hw = (cL / 50.0) * (1.0 - w) + (cL / 5.0) * w
    hw *= max(0.1, float(latitude) / 50.0)
    luts = (_spline_lut(cH, hw, wrap=True),
            _spline_lut(cS, hw * 2.0),
            _spline_lut(cL, hw * 2.0))
    idx = lambda a: (np.clip(a, 0, 65535) / 256.0).astype(np.int32)
    m = luts[0][idx(H)] * luts[1][idx(S)] * luts[2][idx(L)]
    return np.clip(m, 0.0, 1.0)


def build_auto_mask(img, method="u2net", skin_term="comb", model_path=None):
    """Automatic skin matte used by :func:`smooth_skin`.

    ``method='u2net'`` uses the licence-safe U-2-Net person matte; if OpenCV or
    the model file is unavailable it returns ``None`` and the caller falls back
    to :func:`build_skin_mask`.  ``skin_term`` selects ``'hsv'``, ``'tone'``
    (HSL spline reconstruction) or ``'comb'`` (max of the two, default).
    """
    if method != "u2net":
        return None
    person = u2net_person_matte(img, model_path)
    if person is None:
        return None
    if float(person.mean()) < 1e-3:
        return None
    hsv = build_skin_mask(img)
    tone = auto_skin_tone_mask(img, person)
    if tone is None or not (0.002 < float(tone.mean()) < 0.95):
        tone = hsv
    if skin_term == "hsv":
        skin = hsv
    elif skin_term == "tone":
        skin = tone
    elif skin_term == "comb":
        skin = np.maximum(hsv, tone)
    else:
        raise ValueError(f"unknown skin_term {skin_term!r}")
    m = np.clip(person * skin, 0.0, 1.0)
    if float(m.mean()) < 2e-3:      # model output unusable -> classical fallback
        return None
    return m


def build_mask(img, method="u2net", skin_term="comb", model_path=None):
    """Convenience wrapper: automatic mask with classical fallback."""
    m = build_auto_mask(img, method=method, skin_term=skin_term, model_path=model_path)
    return build_skin_mask(img) if m is None else m


# ===========================================================================
# 7. Recovered internal format + bilateral-grid chain (BilateralStep0..Step4)
# ===========================================================================
#
# Recovered from the target algorithm (docs/algorithm.md, smooth-core):
#   * internal pixel = 4 x uint16 (8 bytes/px): w0 = luma magnitude, w1 bit15
#     = luma sign, w2/w3 = int16 chroma.  Unpack is a 1.4572 x BT.601
#     YCbCr->RGB matrix with the pipeline's exact rounding quirk.
#   * smoothing chain  = Step0 zero accumulator | Step1 splat
#     (spatial downsample x intensity binning) | Step2 separable (1,2,1)/4
#     blur | Step3 normalise by accumulated weight | Step4 interpolate back.
#   * per-level driver : depth-first coarse-to-fine, two grid passes
#     per level with strength k = s*16/(L+n+15) and s, then combine levels
#     with 12-bit fixed-point lerp  lerp12(a,b,p) = (a*(4095-p)+b*p) >> 12.
#   * Sharp/Soft = unsharp with x^(2/3) soft clip at a
#     2000-unit knee + 1/2 re-blend toward the smoothed reference.
#
# Assumptions (not modelled): the per-level sigma_s/sigma_r mapping from the
# UI sliders, the value-axis bin width, which axes Step2 blurs, and the
# step->level association (see derive_algorithm_params()).

_YCC_MATRIX = np.array([[1526.0, 2092.0, 0.0],
                        [1526.0, -1066.0, -516.0],
                        [1526.0, 0.0, 2644.0]], dtype=np.float64) / 1024.0
_YCC_MATRIX_INV = np.linalg.inv(_YCC_MATRIX)

#: 12-bit fixed point used by every mix/lerp in the pipeline.
FP12_MAX = 4095


def ycc_unpack(w0: np.ndarray, sign: np.ndarray, w2: np.ndarray,
               w3: np.ndarray) -> np.ndarray:
    """Exact modelled unpack matrix: 4 packed words -> float RGB [0, 65535].

    ``A = (1526 - 3052*sign) * w0``; channels are ``A + 2092*w2``,
    ``A - 1066*w2 - 516*w3``, ``A + 2644*w3`` with the pipeline's
    ``(q >> 10) + 2*((q >> 10) & 1)`` rounding (modelled constant
    ``=1526`` etc.), then clamped to ``[0, 65535]``.
    """
    w0 = np.asarray(w0, dtype=np.int64)
    sign = np.asarray(sign, dtype=np.int64)
    w2 = np.asarray(w2, dtype=np.int64)
    w3 = np.asarray(w3, dtype=np.int64)
    A = (1526 - 3052 * sign) * w0
    q0 = A + 2092 * w2
    q1 = A - 1066 * w2 - 516 * w3
    q2 = A + 2644 * w3

    def _round(q: np.ndarray) -> np.ndarray:
        # exact modelled rounding quirk r(q) = (q>>10) + 2*((q>>10)&1)
        return (q >> 10) + 2 * ((q >> 10) & 1)

    out = np.stack([np.clip(_round(q0), 0, 65535),
                    np.clip(_round(q1), 0, 65535),
                    np.clip(_round(q2), 0, 65535)], axis=-1)
    return out.astype(np.float64)


def ycc_pack(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray]:
    """Inverse of :func:`ycc_unpack`: float RGB [0,1] -> 4 packed words.

    Returns ``(w0, sign, w2, w3)`` with ``w0`` uint16, ``sign`` uint8 (0/1),
    ``w2``/``w3`` int16.  The luma sign is always 0 for a normal [0,1] image.
    Round-trip accuracy across the modelled 12-bit internal path is <= ~4
    units of the 16-bit range (verified in the test suite).
    """
    ch = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 65535.0
    if ch.ndim == 2:
        ch = np.stack([ch, ch, ch], axis=-1)
    shape = ch.shape[:2]
    flat = ch.reshape(-1, 3)
    w = flat @ _YCC_MATRIX_INV.T
    w0 = np.clip(np.round(w[:, 0]), 0, 65535).astype(np.uint16).reshape(shape)
    sign = np.zeros(shape, dtype=np.uint8)
    w2 = np.clip(np.round(w[:, 1]), -32768, 32767).astype(np.int16).reshape(shape)
    w3 = np.clip(np.round(w[:, 2]), -32768, 32767).astype(np.int16).reshape(shape)
    return w0, sign, w2, w3


def blur121(plane: np.ndarray, axis: int) -> np.ndarray:
    """Recovered Step2 kernel: ``out = (2*center + n1 + n2) * 0.25``.

    Edge-clamped 3-tap binomial smoothing along ``axis`` on any float array
    (the pipeline applies it to all 4 float channels of a 16-byte grid cell,
    constant ``0.25f @ ``).
    """
    p = np.asarray(plane, dtype=np.float64)
    pad = [(0, 0)] * p.ndim
    pad[axis] = (1, 1)
    pp = np.pad(p, pad, mode="edge")
    n = p.shape[axis]
    lo = [slice(None)] * p.ndim
    hi = [slice(None)] * p.ndim
    lo[axis] = slice(0, n)
    hi[axis] = slice(2, n + 2)
    return 0.25 * (2.0 * p + pp[tuple(lo)] + pp[tuple(hi)])


def lerp12(a: np.ndarray, b: np.ndarray, p) -> np.ndarray:
    """Recovered 12-bit fixed-point cross-fade ``(a*(4095-p) + b*p) >> 12``.

    ``a``/``b`` are 16-bit integer-like planes; ``p`` is a scalar or array in
    ``[0, 4095]`` (``p=0`` -> ``a``, ``p=4095`` -> ``b``).  Matches
    ``stageineBilateral`` / ``CombineLevelsMultiFaceX`` arithmetic.
    """
    aa = np.asarray(a, dtype=np.int64)
    bb = np.asarray(b, dtype=np.int64)
    pp = np.asarray(p, dtype=np.int64)
    return (aa * (FP12_MAX - pp) + bb * pp) >> 12


def _downsample_pow2(img: np.ndarray, times: int) -> np.ndarray:
    out = np.asarray(img, dtype=np.float64)
    for _ in range(max(int(times), 0)):
        out = downsample2x2(out)
    return out


def _grid_splat(words, sigma_s_px: float, sigma_r: float):
    """Step1/Step2/Step3: splat -> separable (1,2,1)/4 blur -> normalise.

    Returns ``(grid, sx, sy, sv, L, ny, nx, nv)`` where ``grid`` has shape
    ``(ny, nx, nv, 4)`` holding normalised (Y, Cr, Cb, weight) averages in the
    pipeline's 12-bit accumulator units.
    """
    w0, sign, w2, w3 = words
    ch = ycc_unpack(w0, sign, w2, w3)               # (H,W,3) in [0,65535]
    q = np.floor(ch / 16.0).astype(np.int64)        # 12-bit channels
    # modelled luminance-like scalar: L = (((c2 + 2*c0 + c1) >> 2) + c0) >> 1
    Lum = (((q[..., 2] + 2 * q[..., 0] + q[..., 1]) >> 2) + q[..., 0]) >> 1
    H, W = ch.shape[:2]
    sx = max(1.0, float(sigma_s_px))
    sy = sx
    sv = max(1.0, float(sigma_r) * 4095.0)
    ny = int((H - 1) / sy + 0.5) + 3
    nx = int((W - 1) / sx + 0.5) + 3
    nv = int(4095.0 / sv + 0.5) + 3
    gy = (np.floor(np.arange(H) / sy + 0.5).astype(np.int64) + 2).clip(0, ny - 1)
    gx = (np.floor(np.arange(W) / sx + 0.5).astype(np.int64) + 2).clip(0, nx - 1)
    gv = (np.floor(Lum / sv + 0.5).astype(np.int64) + 2).clip(0, nv - 1)
    flat = (gy[:, None, None] * nx + gx[None, :, None]) * nv + gv[..., None]
    flat = flat[..., 0].ravel()
    s = 1 - 2 * sign.astype(np.int64)
    # The pipeline kernel does `cdq; and edx,0xf; add eax,edx; sar eax,4`, i.e.
    # truncation toward zero (C division by 16), NOT an arithmetic >>4.  This
    # differs for negative values whose low 4 bits are non-zero and was caught
    # by the differential test against the original kernel (see
    # (intermediate rounding is truncation toward zero, verified by unit tests).
    def _trunc_div16(v):
        return np.where(v >= 0, v >> 4, -((-v) >> 4))
    v0 = _trunc_div16(s * w0.astype(np.int64))
    v2 = _trunc_div16(w2.astype(np.int64))
    v3 = _trunc_div16(w3.astype(np.int64))
    size = ny * nx * nv
    count = np.bincount(flat, minlength=size).astype(np.float64)
    acc0 = np.bincount(flat, weights=v0.ravel(), minlength=size)
    acc2 = np.bincount(flat, weights=v2.ravel(), minlength=size)
    acc3 = np.bincount(flat, weights=v3.ravel(), minlength=size)
    grid = np.stack([acc0, acc2, acc3, count], axis=-1).reshape(ny, nx, nv, 4)
    return grid, sx, sy, sv, Lum, ny, nx, nv


def _grid_blur_normalise(grid: np.ndarray, iterations: int,
                         blur_axes: Sequence[int]) -> np.ndarray:
    """Step2 (iterate) + Step3 (normalise by accumulated weight)."""
    g = grid
    for _ in range(max(int(iterations), 0)):
        for ax in blur_axes:
            g = blur121(g, ax)
    wsum = g[..., 3:4]
    safe = np.where(wsum > 0.0, wsum, 1.0)
    return np.where(wsum > 0.0, g[..., :3] / safe, 0.0)


def _grid_slice(grid: np.ndarray, sx: float, sy: float, sv: float,
                Lum: np.ndarray, ny: int, nx: int, nv: int) -> np.ndarray:
    """Step4: trilinear interpolate the grid back to full resolution.

    Returns float RGB [0,1] after the modelled unpack (grid stores 12-bit
    accumulator units, hence the ``*16`` rescale of the sliced words).
    """
    H, W = Lum.shape
    yc = 2.0 + np.arange(H, dtype=np.float64) / sy
    xc = 2.0 + np.arange(W, dtype=np.float64) / sx
    vc = 2.0 + Lum.astype(np.float64) / sv
    y0 = np.floor(yc).astype(np.int64).clip(0, ny - 1)
    y1 = (y0 + 1).clip(0, ny - 1)
    x0 = np.floor(xc).astype(np.int64).clip(0, nx - 1)
    x1 = (x0 + 1).clip(0, nx - 1)
    v0 = np.floor(vc).astype(np.int64).clip(0, nv - 1)
    v1 = (v0 + 1).clip(0, nv - 1)
    fy = (yc - y0)[:, None]
    fx = (xc - x0)[None, :]
    fv = vc - v0

    def _tri(plane: np.ndarray) -> np.ndarray:
        c000 = plane[y0[:, None], x0[None, :], v0]
        c001 = plane[y0[:, None], x0[None, :], v1]
        c010 = plane[y0[:, None], x1[None, :], v0]
        c011 = plane[y0[:, None], x1[None, :], v1]
        c100 = plane[y1[:, None], x0[None, :], v0]
        c101 = plane[y1[:, None], x0[None, :], v1]
        c110 = plane[y1[:, None], x1[None, :], v0]
        c111 = plane[y1[:, None], x1[None, :], v1]
        a = c000 * (1.0 - fv) + c001 * fv
        b = c010 * (1.0 - fv) + c011 * fv
        c = c100 * (1.0 - fv) + c101 * fv
        d = c110 * (1.0 - fv) + c111 * fv
        return (a * (1.0 - fx) + b * fx) * (1.0 - fy) \
            + (c * (1.0 - fx) + d * fx) * fy

    q0 = _tri(grid[..., 0]) * 16.0
    q2 = _tri(grid[..., 1]) * 16.0
    q3 = _tri(grid[..., 2]) * 16.0
    sgn = (q0 < 0).astype(np.uint8)
    w0 = np.clip(np.round(np.abs(q0)), 0, 65535).astype(np.uint16)
    w2 = np.clip(np.round(q2), -32768, 32767).astype(np.int16)
    w3 = np.clip(np.round(q3), -32768, 32767).astype(np.int16)
    out = ycc_unpack(w0, sgn, w2, w3) / 65535.0
    return np.clip(out, 0.0, 1.0)


def bilateral_grid_smooth(
    img: np.ndarray,
    sigma_s: float = 8.0,
    sigma_r: float = 0.128,
    iterations: int = 1,
    *,
    blur_axes: Sequence[int] = (0, 1, 2),
    sigma_s_mode: str = "pixels",
) -> np.ndarray:
    """One modelled Step0..Step4 bilateral-grid smoothing pass.

    Implements the multi-scale grid chain described in docs/algorithm.md: ``, ``, ``:

    1. pack RGB into the internal 4xuint16 YCC format (:func:`ycc_pack`);
    2. **splat** each pixel to cell ``(round(y/sigma_s)+2,
       round(x/sigma_s)+2, round(L/sigma_v)+2)`` accumulating
       ``(sgn*w0>>4, w2>>4, w3>>4, +1)`` with
       ``L = (((c2 + 2*c0 + c1) >> 2) + c0) >> 1``;
    3. **blur** the 4-float grid with the modelled separable
       ``(2*c + n1 + n2)*0.25`` kernel along ``blur_axes`` ``iterations`` times
       (the pipeline runs 2 axis passes per level; we default to all 3);
    4. **normalise** by the accumulated weight (Step3);
    5. **slice** back with trilinear interpolation (Step4) and unpack to RGB.

    ``sigma_s`` is the spatial grid spacing in **pixels**; ``sigma_r`` is the
    value-axis bin width in normalised intensity units (``sigma_v`` in 12-bit
    accumulator units = ``sigma_r*4095``).  Both mappings are assumptions --
    the pipeline gets them from per-level parameter records.
    """
    a = np.asarray(img, dtype=np.float64)
    was2 = a.ndim == 2
    if was2:
        a = np.stack([a, a, a], axis=-1)
    H, W = a.shape[:2]
    if sigma_s_mode == "fraction":
        sigma_s_px = max(1.0, float(sigma_s) * min(H, W))
    elif sigma_s_mode == "pixels":
        sigma_s_px = max(1.0, float(sigma_s))
    else:
        raise ValueError(f"unknown sigma_s_mode {sigma_s_mode!r}")
    words = ycc_pack(a)
    grid, sx, sy, sv, Lum, ny, nx, nv = _grid_splat(words, sigma_s_px, sigma_r)
    grid = _grid_blur_normalise(grid, iterations, blur_axes)
    out = _grid_slice(grid, sx, sy, sv, Lum, ny, nx, nv)
    return out[..., 0] if was2 else out


def bilateral_grid_pyramid(
    img: np.ndarray,
    levels: int = 3,
    sigma_s: float = 8.0,
    sigma_r: float = 0.128,
    *,
    strength: int = 2048,
    iterations: int = 1,
    level_weights12: Optional[Sequence[int]] = None,
    blur_axes: Sequence[int] = (0, 1, 2),
    sigma_s_mode: str = "pixels",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Recovered per-level driver ``: coarse-to-fine grid chain.

    For level ``L`` (1 = finest/full resolution, ``levels`` = deepest) the
    level image is downsampled by ``2**(L-1)``, then the bilateral grid chain
    runs **twice** with the modelled strengths ``k = strength*16/(L+n+15)``
    (pass 1) and ``strength`` (pass 2); the two passes are cross-faded with
    :func:`lerp12`.  Levels are filtered depth-first (coarsest first) and
    combined with 12-bit weights ``level_weights12[L-1]`` (map from the
    Fine/Medium/Large sliders); ``CombineLevelsMultiFaceX``'s exact kernel was
    not modelled, so a normalised 12-bit weighted average is used here.

    Returns ``([s_finest, ..., s_coarsest], combined)`` -- same level ordering
    as :func:`bilateral_pyramid` (index 0 = least smoothed).
    """
    a = np.asarray(img, dtype=np.float64)
    was2 = a.ndim == 2
    if was2:
        a = np.stack([a, a, a], axis=-1)
    H, W = a.shape[:2]
    levels = max(int(levels), 0)
    n = max(levels, 1)
    if levels == 0:
        out0 = a.copy()
        return ([out0[..., 0]] if was2 else [out0]), \
            (out0[..., 0] if was2 else out0)
    strength = int(np.clip(int(strength), 0, FP12_MAX))
    weights = ([int(w) for w in level_weights12]
               if level_weights12 is not None else [FP12_MAX] * levels)
    if len(weights) < levels:
        weights = list(weights) + [FP12_MAX] * (levels - len(weights))
    outs: List[np.ndarray] = []
    base16 = np.round(np.clip(a, 0.0, 1.0) * 65535.0)
    for L in range(levels, 0, -1):
        low = _downsample_pow2(base16, L - 1) if L > 1 else base16
        sig_L = max(1.0, float(sigma_s) / (2 ** (L - 1)))
        sm1 = bilateral_grid_smooth(low / 65535.0, sig_L, sigma_r,
                                    iterations=iterations, blur_axes=blur_axes,
                                    sigma_s_mode=sigma_s_mode)
        p1 = int(np.clip(round(strength * 16.0 / (L + n + 15)), 0, FP12_MAX))
        lev16 = lerp12(low, np.round(sm1 * 65535.0), p1)
        sm2 = bilateral_grid_smooth(lev16 / 65535.0, sig_L, sigma_r,
                                    iterations=iterations, blur_axes=blur_axes,
                                    sigma_s_mode=sigma_s_mode)
        lev16 = lerp12(lev16, np.round(sm2 * 65535.0), strength)
        lev = lev16 / 65535.0
        if L > 1:
            lev = upsample_bilinear(lev, (H, W))
        outs.append(lev)
    # outs[0] is the deepest (most smoothed) level -> reverse to finest first
    levels_finest_first = list(reversed(outs))
    wsum = float(sum(weights[:levels]))
    if wsum <= 0.0:
        return levels_finest_first, np.clip(a, 0.0, 1.0)
    acc = np.zeros_like(levels_finest_first[0])
    for lev, wk in zip(levels_finest_first, weights[:levels]):
        acc = acc + (float(wk) / wsum) * lev
    acc = np.clip(acc, 0.0, 1.0)
    if was2:
        return [lev[..., 0] for lev in levels_finest_first], acc[..., 0]
    return levels_finest_first, acc


# ===========================================================================
# 8. Sharpening / softening and tone
# ===========================================================================

def unsharp(img: np.ndarray, amount: float, radius: float = 2.0,
            threshold: float = 0.0) -> np.ndarray:
    """Unsharp mask: ``img + amount * (img - gaussian_blur(img, radius))``.

    ``amount > 0`` sharpens (pipeline ``Sharpness``), ``amount < 0`` softens
    (``Softness``, blending toward the blur); ``amount == 0`` is identity.
    ``threshold`` suppresses the effect on tiny differences (noise guard).
    """
    a = np.asarray(img, dtype=np.float64)
    if amount == 0.0:
        return a.copy()
    detail = a - gaussian_blur(a, radius)
    if threshold is not None and threshold > 0.0:
        detail = np.where(np.abs(detail) < threshold, 0.0, detail)
    return a + float(amount) * detail


def apply_sharp_and_soft(
    img: np.ndarray,
    ref: np.ndarray,
    sharpness: float = 0.0,
    softness: float = 0.0,
    *,
    amount_scale: float = 2.56,
) -> np.ndarray:
    """Recovered ``sharpen / soften stage`` math (§5.3).

    Works in 16-bit integer units on the current image ``img`` against the
    smoothed reference plane ``ref`` (both float [0,1]):

    * **Sharpness** -- unsharp mask with a sub-linear soft clip::

          d = v - v_ref ;  t = (d * par106) >> 8
          r = 2000 * (|t| / 2000) ** (2/3)          # powf, two-thirds power
          delta = sign(t) * trunc(r)               # cvttss2si
          v = clamp(v + delta, 0, 65535)

      The ``x^(2/3)`` knee at 2000 suppresses halos (only transcendental call
      in the smoothing core).  ``par106`` is the amount in 1/256 units; the
      UI-slider -> ``par106`` scale is an assumption (``amount_scale``).
    * **Softness** -- ~1/2 re-blend toward the smoothed reference::

          v = (v * 128 + v_ref * 127) >> 8

      gated linearly by the softness slider (assumption).

    Both sliders 0 returns ``img`` unchanged.
    """
    sharp_on = float(sharpness) > 0.0
    soft_on = float(softness) > 0.0
    if not sharp_on and not soft_on:
        return np.array(img, dtype=np.float64, copy=True)
    v = np.round(np.clip(np.asarray(img, np.float64), 0.0, 1.0) * 65535.0)
    v = v.astype(np.int64)
    r = np.round(np.clip(np.asarray(ref, np.float64), 0.0, 1.0) * 65535.0)
    r = r.astype(np.int64)
    if sharp_on:
        par198 = int(round(np.clip(float(sharpness), 0.0, 100.0)))
        par106 = int(round(np.clip(float(sharpness), 0.0, 100.0) * amount_scale))
        if par198 != 0 and par106 != 0:
            d = v - r
            t = (d * par106) >> 8                 # arithmetic shift (floor)
            rr = 2000.0 * np.power(np.abs(t) / 2000.0, 2.0 / 3.0)
            delta = np.sign(t) * np.trunc(rr).astype(np.int64)
            v = np.clip(v + delta, 0, 65535)
    if soft_on:
        frac = float(np.clip(float(softness), 0.0, 100.0)) / 100.0
        v_soft = (v * 128 + r * 127) >> 8
        v = np.round(v + frac * (v_soft - v)).astype(np.int64)
    return np.clip(v / 65535.0, 0.0, 1.0)


def adjust_tone(img: np.ndarray, brightness: float = 0.0, contrast: float = 0.0,
                warmth: float = 0.0) -> np.ndarray:
    """Simple linear tone adjustments; all parameters are neutral at 0.

    ``brightness`` in [-1, 1] (offset up to +-0.25), ``contrast`` in [-1, 1]
    (gain ``1+contrast`` around 0.5), ``warmth`` in [-1, 1] (red/blue gain
    +-15%).  Returns a clipped [0, 1] image.
    """
    out = np.asarray(img, dtype=np.float64)
    if brightness:
        out = out + 0.25 * float(brightness)
    if contrast:
        out = 0.5 + (out - 0.5) * (1.0 + float(contrast))
    if warmth:
        w = float(warmth)
        gain = 1.0 + 0.15 * w
        if out.ndim == 3 and out.shape[2] >= 3:
            gains = np.ones(out.shape[2], dtype=np.float64)
            gains[0] = gain
            gains[2] = 1.0 / gain
            out = out * gains
        else:
            out = out + 0.03 * w
    return np.clip(out, 0.0, 1.0)


# ===========================================================================
# 9. Top-level pipeline
# ===========================================================================

PORTRAIT_SIZE_LEVELS: Dict[str, int] = {"small": 2, "medium": 3, "large": 4}


def levels_for_portrait_size(portrait_size: str,
                             shape: Sequence[int]) -> int:
    """Pyramid level count for the pipeline's ``Portrait Size`` option.

    The settings schema carries **4 repeated per-level groups** (modelled
    §2.5); the UI "Portrait Size" option selects how many are active.  ``auto``
    scales with the shortest image side (assumption).
    """
    key = str(portrait_size).lower()
    if key in PORTRAIT_SIZE_LEVELS:
        return PORTRAIT_SIZE_LEVELS[key]
    if key == "auto":
        shortest = min(int(shape[0]), int(shape[1]))
        if shortest < 400:
            return 2
        if shortest < 1100:
            return 3
        return 4
    raise ValueError(f"unknown portrait_size {portrait_size!r}")


@dataclass(frozen=True)
class SmoothParams:
    """Smoothing parameters (slider space, mapped to the algorithm).

    Sliders ``smoothing``/``threshold``/``fine``/``medium``/``large``/
    ``sharpness``/``softness`` are 0..100.  ``warmth``/``brightness``/
    ``contrast`` are normalised [-1, 1] (0 = neutral).

    ``method`` selects the algorithm variant:

    * ``'grid'`` (default) -- the multi-scale bilateral-grid chain
      (``bilateral_grid_pyramid``);
    * ``'recursive'`` -- an alternative recursive-IIR bilateral pyramid
      (Yang-style, ``bilateral_pyramid``), kept for comparison.
    """
    smoothing: float = 50.0
    threshold: float = 50.0
    fine: float = 50.0
    medium: float = 50.0
    large: float = 50.0
    sharpness: float = 0.0
    softness: float = 0.0
    warmth: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    opacity: float = 1.0
    portrait_size: str = "medium"
    use_mask: bool = True
    #: mask source: "u2net" (licence-safe U-2-Net person matte, default) or "hsv"
    mask_method: str = "u2net"
    #: skin term used with the person matte: "hsv" | "tone" | "comb" (default)
    skin_term: str = "comb"
    #: optional explicit path to u2netp.onnx (default: models/)
    u2net_path: Optional[str] = None

    # -- algorithm selection / advanced overrides --
    method: str = "grid"
    levels: Optional[int] = None
    sigma_s: Optional[float] = None
    sigma_r: Optional[float] = None
    grid_iterations: int = 1
    grid_blur_axes: Tuple[int, ...] = (0, 1, 2)
    # recursive-path options
    iters: int = 2
    range_mode: str = DEFAULT_RANGE_MODE
    sigma_s_mode: str = DEFAULT_SIGMA_S_MODE
    per_channel: bool = False
    detail_threshold: Optional[float] = None
    use_detail_threshold: bool = True
    # shared
    mask_blur_sigma: Optional[float] = None
    sharpen_amount_scale: float = 2.56

    def with_overrides(self, **overrides: Any) -> "SmoothParams":
        return replace(self, **overrides) if overrides else self


def derive_algorithm_params(
    params: SmoothParams, shape: Sequence[int]
) -> Dict[str, Any]:
    """Map UI-slider space to algorithm parameters.

    Recovered/verified here:

    * master strength ``s = smoothing*4095/100`` (12-bit fixed point);
    * per-level pass-1 strength ``k = s*16/(L+n+15)`` and pass-2 strength ``s``;
    * 12-bit level mix weights (Fine/Medium/Large sliders x master smoothing);
    * ``lerp12`` cross-fade and the ``x^(2/3)`` sharp/soft knee.

    Assumed here (NOT modelled; the per-level record values and the exact
    slider->parameter conversions were not isolated):

    * grid spatial spacing ``sigma_s_px = 2 + 10*smoothing/100`` pixels;
    * value-axis bin width ``sigma_r = 0.008 + 0.24*threshold/100``;
    * Fine/Medium/Large -> pyramid level 1/2/>=3 association;
    * mask feather radius;
    * ``par106 = sharpness*2.56`` (sharp amount in 1/256 units).
    """
    s_slider = float(np.clip(params.smoothing, 0.0, 100.0))
    t = float(np.clip(params.threshold, 0.0, 100.0)) / 100.0
    s12 = int(round(s_slider * FP12_MAX / 100.0))          # strength s
    levels = (levels_for_portrait_size(params.portrait_size, shape)
              if params.levels is None else int(params.levels))
    levels = max(int(levels), 0)
    n = max(levels, 1)
    sigma_r = (0.008 + 0.24 * t) if params.sigma_r is None else float(params.sigma_r)
    mask_sigma = (float(params.mask_blur_sigma) if params.mask_blur_sigma is not None
                  else max(1.0, 0.006 * min(int(shape[0]), int(shape[1]))))
    sigma_s: Optional[float] = None
    if params.method == "grid":
        sigma_s_px = (2.0 + 10.0 * s_slider / 100.0
                      if params.sigma_s is None else float(params.sigma_s))
        level_strengths = [
            int(np.clip(round(s12 * 16.0 / (L + n + 15)), 0, FP12_MAX))
            for L in range(1, levels + 1)
        ]
        sliders = [params.fine, params.medium, params.large]
        level_weights12 = []
        for L in range(1, levels + 1):
            slider = sliders[min(L - 1, 2)]
            w = (float(np.clip(slider, 0.0, 100.0)) / 100.0
                 * s_slider / 100.0 * FP12_MAX)
            level_weights12.append(int(round(w)))
        detail_weights = [1.0 - float(np.clip(sliders[min(L - 1, 2)], 0.0, 100.0))
                          / 100.0 for L in range(1, max(levels, 1) + 1)]
        detail_threshold = None
    else:
        sigma_s_px = None
        sigma_s = (0.03 + 0.17 * s_slider / 100.0
                   if params.sigma_s is None else float(params.sigma_s))
        level_strengths = [s12] * levels
        level_weights12 = []
        detail_weights = [
            1.0 - float(np.clip(params.fine, 0.0, 100.0)) / 100.0,
            1.0 - float(np.clip(params.medium, 0.0, 100.0)) / 100.0,
            1.0 - float(np.clip(params.large, 0.0, 100.0)) / 100.0,
        ]
        if not params.use_detail_threshold:
            detail_threshold = None
        elif params.detail_threshold is not None:
            detail_threshold = float(params.detail_threshold)
        else:
            detail_threshold = 0.005 + 0.30 * t
    return {
        "method": params.method,
        "levels": levels,
        "strength": s12,
        "level_strengths": level_strengths,
        "level_weights12": level_weights12,
        "sigma_s_px": sigma_s_px,
        "sigma_s": sigma_s,
        "sigma_r": sigma_r,
        "grid_iterations": max(int(params.grid_iterations), 0),
        "grid_blur_axes": tuple(params.grid_blur_axes),
        "recursive_iters": max(int(params.iters), 0),
        "detail_weights": detail_weights,
        "detail_threshold": detail_threshold,
        "sharpen_amount_scale": float(params.sharpen_amount_scale),
        "mask_sigma": mask_sigma,
        "opacity": float(np.clip(params.opacity, 0.0, 1.0)),
    }


def _coerce_params(
    params: Optional[Union[SmoothParams, Mapping[str, Any]]],
    overrides: Mapping[str, Any],
) -> SmoothParams:
    if params is None:
        p = SmoothParams()
    elif isinstance(params, SmoothParams):
        p = params
    elif isinstance(params, Mapping):
        p = SmoothParams(**dict(params))
    else:
        raise TypeError("params must be None, SmoothParams or a mapping")
    clean = {k: v for k, v in overrides.items() if v is not None}
    return p.with_overrides(**clean) if clean else p


def smooth_skin(
    img: np.ndarray,
    params: Optional[Union[SmoothParams, Mapping[str, Any]]] = None,
    *,
    mask: Optional[np.ndarray] = None,
    return_trace: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run the reference the pipeline skin-smoothing pipeline.

    Recovered stage order (``docs/algorithm.md``)::

        [mask build] -> per-level bilateral-grid chain (Step0..Step4)
        -> coarse-to-fine level combine -> masked blend (opacity)
        -> sharp/soft (x^(2/3) soft clip + 1/2 re-blend) -> tone adjustments

    Parameters
    ----------
    img : (H,W) or (H,W,3) float array in [0,1].
    params : :class:`SmoothParams` or mapping of field name -> value.
    mask : optional (H,W) skin/face matte in [0,1].  If omitted and
        ``params.use_mask`` is true, the automatic mask is built:
        U-2-Net person matte (Apache-2.0) x classical/spline skin term when
        OpenCV + ``u2netp.onnx`` are available (``params.mask_method='u2net'``),
        otherwise the HSV fallback (:func:`build_skin_mask`).
    return_trace : include intermediate stages in the result dict
        (``mask``, ``levels``, ``base``, ``details``, ``smoothed``,
        ``sharpened``, ``toned``).
    **overrides : field overrides on top of ``params``.
    """
    p = _coerce_params(params, overrides)
    x = np.asarray(img, dtype=np.float64)
    if x.ndim not in (2, 3):
        raise ValueError("img must be (H,W) or (H,W,C)")
    if x.ndim == 3 and x.shape[2] not in (1, 3, 4):
        raise ValueError("only 1, 3 or 4 channel images are supported")
    x0 = np.clip(x, 0.0, 1.0).copy()
    der = derive_algorithm_params(p, x0.shape)
    trace: Dict[str, Any] = {"input": x0, "params": p, "derived": der}

    # -- stage 1: optional mask build / feathering --------------------------
    m: Optional[np.ndarray] = None
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64)
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[..., 0]
        m = np.clip(m, 0.0, 1.0)
        if der["mask_sigma"] > 0:
            m = np.clip(mask_blur(m, der["mask_sigma"]), 0.0, 1.0)
    elif p.use_mask:
        m = None
        if getattr(p, "mask_method", "u2net") == "u2net":
            m = build_auto_mask(x0, method="u2net",
                                skin_term=getattr(p, "skin_term", "comb"),
                                model_path=getattr(p, "u2net_path", None))
        if m is None:                      # no cv2/model -> classical fallback
            m = build_skin_mask(x0)
        m = np.clip(m, 0.0, 1.0)
        if der["mask_sigma"] > 0:
            m = np.clip(mask_blur(m, der["mask_sigma"]), 0.0, 1.0)
    trace["mask"] = m

    # -- stage 2/3: smoothing chain + coarse-to-fine level combine ----------
    if der["method"] == "grid":
        level_images, smoothed = bilateral_grid_pyramid(
            x0,
            levels=der["levels"],
            sigma_s=der["sigma_s_px"],
            sigma_r=der["sigma_r"],
            strength=der["strength"],
            iterations=der["grid_iterations"],
            level_weights12=der["level_weights12"],
            blur_axes=der["grid_blur_axes"],
            sigma_s_mode="pixels",
        )
        base, details = detail_decompose(x0, level_images)
        weights = der["level_weights12"]
    else:
        smoothed_levels = bilateral_pyramid(
            x0, levels=der["levels"], iters=der["recursive_iters"],
            sigma_s=der["sigma_s"],
            sigma_r=der["sigma_r"],
            range_mode=p.range_mode, sigma_s_mode=p.sigma_s_mode,
            per_channel=p.per_channel,
        )
        base, details = detail_decompose(x0, smoothed_levels)
        level_images = smoothed_levels
        weights = der["detail_weights"][:len(details)]
        gains = None
        if der["detail_threshold"] is not None and len(details) > 0:
            gains = [threshold_gain(d, der["detail_threshold"], w)
                     for d, w in zip(details, weights)]
        smoothed = combine_levels(base, details, weights, gains=gains)
    trace["levels"] = level_images
    trace["base"] = base
    trace["details"] = details
    trace["weights"] = weights

    # -- stage 4: masked blend with opacity -------------------------------
    if m is not None:
        smoothed = blend_masked(x0, smoothed, m, der["opacity"])
    elif der["opacity"] < 1.0:
        smoothed = x0 + der["opacity"] * (smoothed - x0)
    trace["smoothed"] = smoothed

    # -- stage 5: modelled sharpness / softness --------------------------
    if der["method"] == "grid":
        sharpened = apply_sharp_and_soft(
            smoothed, base, p.sharpness, p.softness,
            amount_scale=der["sharpen_amount_scale"])
    else:
        amount = (0.8 * float(np.clip(p.sharpness, 0.0, 100.0)) / 100.0
                  - 0.8 * float(np.clip(p.softness, 0.0, 100.0)) / 100.0)
        sharpened = unsharp(smoothed, amount, radius=2.0)
    trace["sharpened"] = sharpened

    # -- stage 6: tone adjustments ----------------------------------------
    toned = adjust_tone(sharpened, brightness=p.brightness,
                        contrast=p.contrast, warmth=p.warmth)
    toned = np.clip(toned, 0.0, 1.0)
    trace["toned"] = toned

    out: Dict[str, Any] = {
        "output": toned,
        "input": x0,
        "params": p,
        "derived": der,
    }
    if return_trace:
        out.update(trace)
    return out


def apply_smoothing(img: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Convenience wrapper: ``smooth_skin(...)['output']``."""
    return smooth_skin(img, **kwargs)["output"]


def _dump_metrics(path: str, obj: Mapping[str, Any]) -> None:
    """Internal helper used by demo/verify to write JSON metrics."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=float)
