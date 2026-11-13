"""synthetic.py -- deterministic synthetic images for the skin-smoothing
implementation demo and unit tests.

Everything here is generated procedurally, so the repository needs no external
image assets: ``synthetic_portrait`` builds a portrait-like RGB image with a
skin-tone gradient, facial-feature edges (eyes/iris boundary, eyebrows, lips),
hair, fine skin texture and sensor noise; ``synthetic_step_edge`` and
``synthetic_flat_noise`` provide the two canonical filter test patterns.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "ellipse_mask",
    "synthetic_portrait",
    "synthetic_step_edge",
    "synthetic_flat_noise",
]


def ellipse_mask(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float,
                 ry: float, rx: float, feather: float = 1.5,
                 angle: float = 0.0) -> np.ndarray:
    """Soft ellipse matte: 1 inside, 0 outside, ``feather`` pixels transition."""
    dy = yy - cy
    dx = xx - cx
    if angle:
        ca, sa = np.cos(angle), np.sin(angle)
        dy, dx = ca * dy + sa * dx, -sa * dy + ca * dx
    r = np.sqrt((dy / max(ry, 1e-6)) ** 2 + (dx / max(rx, 1e-6)) ** 2)
    # convert normalised radius to ~pixel distance from the boundary
    scale = min(abs(ry), abs(rx))
    t = np.clip((1.0 + feather / max(scale, 1e-6) - r)
                / (2.0 * feather / max(scale, 1e-6)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fractal_noise(rng: np.random.Generator, h: int, w: int,
                   octaves: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
                   weights: Tuple[float, ...] = (1.0, 0.6, 0.35, 0.2)
                   ) -> np.ndarray:
    """Zero-mean multi-octave smooth noise in [-1, 1] (bilinear-upsampled)."""
    out = np.zeros((h, w), dtype=np.float64)
    total = 0.0
    for octave, weight in zip(octaves, weights):
        gh = max(2, int(round(h / octave)))
        gw = max(2, int(round(w / octave)))
        grid = rng.normal(0.0, 1.0, (gh, gw))
        ys = (np.arange(h) + 0.5) * gh / h - 0.5
        xs = (np.arange(w) + 0.5) * gw / w - 0.5
        ys = np.clip(ys, 0.0, gh - 1.0)
        xs = np.clip(xs, 0.0, gw - 1.0)
        y0 = np.floor(ys).astype(np.intp)
        x0 = np.floor(xs).astype(np.intp)
        y1 = np.minimum(y0 + 1, gh - 1)
        x1 = np.minimum(x0 + 1, gw - 1)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
        bot = grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx
        layer = top * (1 - fy) + bot * fy
        out += weight * layer
        total += weight
    out /= max(total, 1e-9)
    out -= out.mean()
    return out / max(np.abs(out).max(), 1e-9)


def synthetic_portrait(height: int = 720, width: int = 540, seed: int = 7,
                       noise: float = 0.012,
                       texture: float = 0.010) -> Tuple[np.ndarray, np.ndarray]:
    """Portrait-like synthetic RGB image + analytic face matte.

    Returns ``(img, face_mask)`` with ``img`` float64 ``(H, W, 3)`` in [0, 1] and
    ``face_mask`` float64 ``(H, W)`` in [0, 1].  Skin tone is a warm mid-tone
    (~190, 150, 122 in 8-bit); the image contains hard feature edges (iris
    boundary, eyebrows, lips) as well as fine skin texture and sensor noise.
    """
    rng = np.random.default_rng(seed)
    h, w = int(height), int(width)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # --- background: cool gray-blue gradient + vignette ---------------------
    bg = np.zeros((h, w, 3))
    grad = 0.28 + 0.10 * (yy / max(h - 1, 1))
    bg[..., 0] = 0.30 * grad / 0.38
    bg[..., 1] = 0.34 * grad / 0.38
    bg[..., 2] = 0.38 * grad / 0.38
    fg = ellipse_mask(yy, xx, h * 0.46, w * 0.50, h * 0.52, w * 0.44, feather=2.0)
    img = bg

    # --- hair: dark brown large ellipse behind the face ---------------------
    hair = ellipse_mask(yy, xx, h * 0.38, w * 0.50, h * 0.50, w * 0.44, feather=2.5)
    hair_col = np.array([0.10, 0.065, 0.045])
    img = img * (1 - hair[..., None]) + hair_col * hair[..., None]

    # --- face: skin tone with directional lighting --------------------------
    face = ellipse_mask(yy, xx, h * 0.47, w * 0.50, h * 0.40, w * 0.31, feather=2.0)
    skin = np.array([0.790, 0.615, 0.498])
    light = 1.0 + 0.10 * (1.0 - np.clip(((xx - w * 0.36) ** 2
                                        + (yy - h * 0.34) ** 2) ** 0.5
                                       / (0.7 * w), 0.0, 1.0)) - 0.05 * (yy / h)
    face_col = skin[None, None, :] * light[..., None]
    # cheek blush (subtle red)
    for cx, cy in ((w * 0.35, h * 0.56), (w * 0.65, h * 0.56)):
        blush = ellipse_mask(yy, xx, cy, cx, h * 0.055, w * 0.06, feather=8.0)
        face_col = face_col + np.array([0.045, -0.005, -0.005]) * blush[..., None]
    img = img * (1 - face[..., None]) + face_col * face[..., None]

    def paint(col: np.ndarray, mask: np.ndarray) -> None:
        nonlocal img
        img = img * (1 - mask[..., None]) + np.asarray(col)[None, None, :] * mask[..., None]

    # --- eyes: sclera / iris / pupil / specular highlight -------------------
    for ex in (w * 0.375, w * 0.625):
        ey = h * 0.415
        sclera = ellipse_mask(yy, xx, ey, ex, h * 0.030, w * 0.047, feather=1.2)
        paint((0.88, 0.86, 0.84), sclera)
        iris = ellipse_mask(yy, xx, ey + h * 0.002, ex, h * 0.021, w * 0.030, feather=1.0)
        paint((0.20, 0.30, 0.36), iris)
        pupil = ellipse_mask(yy, xx, ey + h * 0.002, ex, h * 0.010, w * 0.014, feather=1.0)
        paint((0.03, 0.03, 0.035), pupil)
        spec = ellipse_mask(yy, xx, ey - h * 0.008, ex - w * 0.009,
                            h * 0.006, w * 0.009, feather=1.5)
        paint((0.98, 0.98, 0.98), spec)
        # upper eyelid shadow + lash line (hard-ish edge above the eye)
        lid = ellipse_mask(yy, xx, ey - h * 0.030, ex, h * 0.022, w * 0.050, feather=1.0)
        paint((0.32, 0.24, 0.19), 0.55 * lid)
        # eyebrow: thick dark arc
        brow = ellipse_mask(yy, xx, ey - h * 0.075, ex, h * 0.012, w * 0.055,
                            feather=1.0, angle=0.10 * np.sign(ex - w * 0.5))
        paint((0.16, 0.11, 0.085), brow)

    # --- nose shadow --------------------------------------------------------
    nose = ellipse_mask(yy, xx, h * 0.535, w * 0.50, h * 0.085, w * 0.045, feather=6.0)
    img = img * (1 - 0.10 * nose[..., None])

    # --- lips --------------------------------------------------------------
    lips = ellipse_mask(yy, xx, h * 0.635, w * 0.50, h * 0.030, w * 0.095, feather=1.5)
    paint((0.62, 0.30, 0.28), lips)
    mouth_line = ellipse_mask(yy, xx, h * 0.635, w * 0.50, h * 0.004, w * 0.085, feather=0.8)
    paint((0.25, 0.12, 0.11), mouth_line)

    # --- fine skin texture and sensor noise --------------------------------
    tex = _fractal_noise(rng, h, w)
    img = img + texture * tex[..., None]
    img = img + rng.normal(0.0, noise, (h, w, 1))
    img = np.clip(img, 0.0, 1.0)
    face_mask = np.clip(face, 0.0, 1.0)
    return img, face_mask


def synthetic_step_edge(height: int = 128, width: int = 128, low: float = 0.20,
                        high: float = 0.80, edge_fraction: float = 0.25,
                        noise: float = 0.0, seed: int = 0,
                        channels: int = 3) -> np.ndarray:
    """Vertical step edge at ``edge_fraction * width`` with optional noise."""
    rng = np.random.default_rng(seed)
    h, w = int(height), int(width)
    img = np.full((h, w, channels), low, dtype=np.float64)
    img[:, int(round(edge_fraction * w)):] = high
    if noise > 0:
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0.0, 1.0)


def synthetic_flat_noise(height: int = 64, width: int = 64, level: float = 0.5,
                         noise: float = 0.05, seed: int = 0,
                         channels: int = 3) -> np.ndarray:
    """Flat field + zero-mean Gaussian noise."""
    rng = np.random.default_rng(seed)
    return np.clip(level + rng.normal(0.0, noise, (height, width, channels)),
                   0.0, 1.0)
