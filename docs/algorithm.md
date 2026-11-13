# Algorithm notes

This document describes the math implemented in `skin_smoothing.py`.

## 1. Internal pixel format

The smoothing stages operate on a packed 16-bit YCC representation, 8 bytes per
pixel = four `uint16` words:

```
w0 : unsigned 16-bit   luma magnitude
w1 : bit 15 only       sign of the luma
w2 : int16             chroma 1
w3 : int16             chroma 2
```

Unpack to three floating-point channels with the fixed-point matrix below
(`K = 1024`, rounding `r(q) = (q >> 10) + 2*((q >> 10) & 1)`):

```
s  = 1 - 2*(w1 >> 15)                  # ±1
A  = 1526 * s * w0
c0 = clamp(r(A + 2092*w2),        0, 65535)
c1 = clamp(r(A - 1066*w2 - 516*w3), 0, 65535)
c2 = clamp(r(A + 2644*w3),        0, 65535)
```

The coefficients correspond to a BT.601-style YCbCr→RGB matrix
(`1.4902, 2.0430, 1.0410, 2.5820, 0.5039` after `/1024`). The pack side uses the
inverse matrix.

The grid accumulator is 12-bit: channel values are shifted right by 4
(`-(v >> 4)` truncating toward zero, not an arithmetic shift) and the weight
count is accumulated as a float.

## 2. Bilateral grid

Per level, six sub-passes:

**Step 0 — zero.** Two 4-float planes (accumulated value + weight) are zeroed.

**Step 1 — splat.** For each pixel `i` in a row and each position index:

```
row = (int)(i / sigma_s + 0.5) + 2
L   = (((c2 + 2*c0 + c1) >> 2) + c0) >> 1
col = (int)(L / sigma_v + 0.5) + 2
cell[row][col] += ( c0 >> 4, c1 >> 4, c2 >> 4, 1.0 )
```

i.e. spatial down-sampling by `sigma_s` and luminance binning by `sigma_v`.

**Step 2 — separable blur.** Two axis passes of

```
out = (2*center + n1 + n2) * 0.25
```

applied to all four float channels.

**Step 3 — normalise.**

```
if (cell[3] != 0) cell[0..2] /= cell[3]
```

**Step 4 — slice.** Trilinear interpolation back to full resolution with a
12-bit value axis (`t = clamp(coord, 0, 4095)`, `a = 4095 - t`), then saturating
int16 packing with the luma sign re-inserted into bit 15 of `w1`.

## 3. Multi-scale driver

Levels are processed coarse-to-fine. For level `L` with total levels `n`:

```
s = strength                      # 0..4095, from the smoothing slider
k = s * 16 / (L + n + 15)         # first pass (weaker)
pass 1: grid chain with strength k
pass 2: grid chain with strength s
levels are combined with a 12-bit lerp:
    lerp12(a, b, p) = (a*(4095 - p) + b*p) >> 12
```

## 4. Detail layers and threshold

The base is the smoothed image; detail layers are `image - smoothed` per scale.
Per-scale weights mix the detail back. A soft threshold suppresses detail
differences below the threshold and ramps up above it, which protects feature
edges while keeping skin texture.

## 5. Sharpness / softness

Unsharp with a soft-clipped delta:

```
d = v - v_ref
t = (d * scale) >> 8
r = 2000 * (|t| / 2000)^(2/3)
v' = clamp(v + sign(t) * r, 0, 65535)
```

and for softness a 1/2 re-blend toward the smoothed reference:

```
v' = (v * 128 + v_ref * 127) >> 8
```

## 6. Mask

Automatic mask:

```
mask = person_matte × skin_term
person_matte : U-2-Net-lite (Apache-2.0), 320×320 input, ImageNet normalisation,
               min-max normalised, bilinearly upsampled
skin_term    : max(HSV skin matte, HSL spline skin-tone matte)
               HSL spline: per-channel 256-entry soft acceptance curve around the
               median colour of the person region;
               half-width h = L/50 * (1 - w) + L/5 * w,  w = hue_spread / 128
mask is feathered with a separable Gaussian.
```

Fallback: HSV-only matte when OpenCV or the model file is unavailable.

## 7. Parameter mapping (slider → algorithm)

| parameter | range | effect |
|---|---|---|
| `smoothing` | 0..100 | overall strength `s` |
| `threshold` | 0..100 | detail/edge threshold |
| `fine`, `medium`, `large` | −20..20 | per-scale detail trim |
| `sharpness`, `softness` | 0..40 | unsharp amount / re-blend |
| `warmth`, `brightness`, `contrast` | −20..20 | tone adjustments |
| `portrait_size` | auto/small/medium/large | pyramid level count |
