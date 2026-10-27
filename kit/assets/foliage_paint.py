"""Foliage and bark textures, painted with numpy alone (no Blender, so they can be previewed and
tuned quickly): leaf sprays per species in a 2x2 atlas (each quarter one spray on its twig, the
twig entering at the bottom centre), conifer branchlets, grass, reed and weed tufts, and tileable
bark (albedo, normal, roughness).

Leaves are drawn from botanical outlines (lobed oak, serrated birch and bramble, lanceolate
willow, small ovate privet) and shaded the way real sprays read: a fold along the midrib, one half
catching more light, pale midrib and veins, darker margins, paler undersides on a few, and soft
contact shadows cast on the leaves behind. Everything is painted at twice the size and filtered
down; transparent pixels carry the nearby leaf colour so mipmaps do not grow dark fringes.

Arrays are float RGBA in linear colour, row 0 = the bottom of the image (Blender's convention).
"""
import math

import numpy as np

SS = 2   # supersampling factor


def srgb_lin(hexs):
    c = np.array([int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# -- tileable value noise -------------------------------------------------------------------------

def _lattice(shape, cells, rng):
    """Smooth periodic value noise, (H, W) array in 0-1, `cells` = (cx, cy) lattice cells."""
    H, W = shape
    cx, cy = cells
    lat = rng.random((cy, cx))
    x = (np.arange(W) + 0.5) / W * cx
    y = (np.arange(H) + 0.5) / H * cy
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    x0 %= cx
    y0 %= cy
    x1 = (x0 + 1) % cx
    y1 = (y0 + 1) % cy
    a = lat[np.ix_(y0, x0)]
    b = lat[np.ix_(y0, x1)]
    c = lat[np.ix_(y1, x0)]
    d = lat[np.ix_(y1, x1)]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(shape, cells, rng, octaves=4, gain=0.5):
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * _lattice(shape, (cells[0] * 2 ** o, cells[1] * 2 ** o), rng)
        tot += amp
        amp *= gain
    return out / tot


def worley(shape, cells, rng, warp=None):
    """Tileable cellular noise: (F1, F2 - F1) in cell units for jittered points on a cx x cy grid.
    F2 - F1 is small along the borders between cells (cracks, furrows)."""
    H, W = shape
    cx, cy = cells
    pts = rng.random((cy, cx, 2))
    x = (np.arange(W) + 0.5) / W * cx
    y = (np.arange(H) + 0.5) / H * cy
    X, Y = np.meshgrid(x, y)
    if warp is not None:
        X = X + warp[0]
        Y = Y + warp[1]
    ix, iy = np.floor(X).astype(int), np.floor(Y).astype(int)
    f1 = np.full(shape, 9.0)
    f2 = np.full(shape, 9.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            jx, jy = ix + dx, iy + dy
            p = pts[jy % cy, jx % cx]
            d = np.hypot(jx + p[..., 0] - X, jy + p[..., 1] - Y)
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
    return f1, f2 - f1


# -- canvas ---------------------------------------------------------------------------------------

class Canvas:
    """An RGBA painting surface at SS x the output size; paint calls take output-pixel units."""

    def __init__(self, px, py=None):
        self.px = px
        self.py = py or px
        self.rgb = np.zeros((self.py * SS, px * SS, 3))
        self.a = np.zeros((self.py * SS, px * SS))
        self.reset_clip()

    def reset_clip(self):
        self.clip = (0, 0, self.px * SS, self.py * SS)

    def set_clip(self, x0, y0, x1, y1):
        self.clip = (int(x0 * SS), int(y0 * SS), int(x1 * SS), int(y1 * SS))

    def window(self, cx, cy, r):
        """Local pixel grid (canvas units) round (cx, cy) in output px, radius r output px."""
        cx, cy, r = cx * SS, cy * SS, r * SS
        X0, Y0, X1, Y1 = self.clip
        x0, x1 = max(X0, int(cx - r) - 1), min(X1, int(cx + r) + 2)
        y0, y1 = max(Y0, int(cy - r) - 1), min(Y1, int(cy + r) + 2)
        if x1 <= x0 or y1 <= y0:
            return None
        X, Y = np.meshgrid((np.arange(x0, x1) + 0.5) / SS, (np.arange(y0, y1) + 0.5) / SS)
        return (slice(y0, y1), slice(x0, x1)), X, Y

    def blend(self, sl, rgb, alpha):
        sub = self.rgb[sl]
        sub[...] = sub * (1 - alpha[..., None]) + rgb * alpha[..., None]
        self.a[sl] = np.maximum(self.a[sl], alpha)

    def darken(self, sl, amount):
        self.rgb[sl] *= (1 - amount)[..., None]

    def cut(self, cx, cy, r):
        """Punch a soft-edged hole (insect damage) through whatever is painted."""
        win = self.window(cx, cy, r + 2)
        if win is None:
            return
        sl, X, Y = win
        d = np.hypot(X - cx, Y - cy)
        self.a[sl] *= np.clip((d - r) * SS + 0.5, 0, 1)

    def result(self):
        """Filter down to the output size and bleed colour into transparent pixels."""
        w, h = self.px, self.py
        a = self.a.reshape(h, SS, w, SS).mean((1, 3))
        pre = (self.rgb * self.a[..., None]).reshape(h, SS, w, SS, 3).sum((1, 3))
        wsum = self.a.reshape(h, SS, w, SS).sum((1, 3))
        rgb = np.where(wsum[..., None] > 1e-6, pre / np.maximum(wsum, 1e-6)[..., None], 0.0)
        rgb = bleed(rgb, a)
        return np.concatenate([rgb, a[..., None]], -1)


def _box(img, r):
    """Box blur (wrapping) of an (H, W, ...) array by radius r via cumulative sums."""
    for ax in (0, 1):
        c = np.cumsum(np.concatenate([np.take(img, range(-r - 1, 0), axis=ax), img, np.take(img, range(0, r), axis=ax)], ax), ax)
        n = img.shape[ax]
        hi = np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=ax)
        lo = np.take(c, range(0, n), axis=ax)
        img = (hi - lo) / (2 * r + 1)
    return img


def bleed(rgb, a, radii=(2, 6, 16, 48)):
    """Fill transparent pixels with the colour of the nearest covered ones (coarser and coarser)."""
    out = rgb.copy()
    have = a > 0.02
    for r in radii:
        w = _box(have.astype(float), r)
        c = _box(np.where(have[..., None], out, 0.0), r)
        fill = (~have) & (w > 1e-4)
        out[fill] = c[fill] / w[fill][..., None]
        have = have | fill
    return out


# -- primitives -----------------------------------------------------------------------------------

def _seg_dist(X, Y, p0, p1):
    d = np.array(p1, float) - np.array(p0, float)
    L2 = max(float(d @ d), 1e-9)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / L2, 0, 1)
    return np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1])), t


def stroke(cv, pts, w0, w1, col0, col1=None, shadow=0.0):
    """A tapered stroke along a polyline (width w0 at the start to w1 at the end), output px."""
    pts = [np.asarray(p, float) for p in pts]
    seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
    total = max(sum(seg), 1e-9)
    acc = 0.0
    col1 = col0 if col1 is None else col1
    for a, b, ln in zip(pts[:-1], pts[1:], seg):
        t0, t1 = acc / total, (acc + ln) / total
        acc += ln
        r = max(w0, w1) / 2 + 2
        win = cv.window((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, ln / 2 + r + 2)
        if win is None:
            continue
        sl, X, Y = win
        if shadow:
            ds, _ = _seg_dist(X, Y, a + (1.2, -1.6), b + (1.2, -1.6))
            cv.darken(sl, shadow * np.clip(1 - ds / (w0 * 1.5 + 1.5), 0, 1))
        d, t = _seg_dist(X, Y, a, b)
        tt = t0 + (t1 - t0) * t
        w = (w0 + (w1 - w0) * tt) / 2
        alpha = np.clip((w - d) * SS + 0.5, 0, 1)
        if not alpha.any():
            continue
        across = np.clip(d / np.maximum(w, 1e-3), 0, 1)
        rgb = (col0[None, None, :] * (1 - tt[..., None]) + col1[None, None, :] * tt[..., None]) * (1.08 - 0.3 * across ** 2)[..., None]
        cv.blend(sl, rgb, alpha)


def curve(p0, d0, length, bend, n=10, droop=0.0):
    """Points of a gently curving line from p0 heading d0 (radians); bend radians over its length."""
    pts = [np.array(p0, float)]
    a = d0
    for k in range(n):
        a += bend / n
        step = np.array([math.cos(a), math.sin(a)]) * length / n
        step[1] -= droop * length / n * (k / n)
        pts.append(pts[-1] + step)
    return pts


# -- leaves ---------------------------------------------------------------------------------------

# width profile u^a (1-u)^b scaled to a peak half-width W (as a fraction of length); lobes and
# serration modulate the margin
LEAF = {
    "oak": dict(W=0.25, a=1.3, b=0.5, lobes=0, lobe=0.0, serr=0.0, bend=0.06, petiole=0.04, notches=4.2, notch=0.72),
    "birch": dict(W=0.36, a=0.45, b=1.25, lobes=0, lobe=0.0, serr=0.08, serr_n=26, bend=0.05, petiole=0.22),
    "willow": dict(W=0.1, a=0.55, b=1.0, lobes=0, lobe=0.0, serr=0.03, serr_n=40, bend=0.18, petiole=0.05),
    "privet": dict(W=0.30, a=0.6, b=0.85, lobes=0, lobe=0.0, serr=0.0, bend=0.04, petiole=0.08),
    "beech": dict(W=0.33, a=0.6, b=0.9, lobes=0, lobe=0.0, serr=0.03, serr_n=9, bend=0.05, petiole=0.1),
    "bramble": dict(W=0.36, a=0.5, b=1.05, lobes=0, lobe=0.0, serr=0.13, serr_n=22, bend=0.05, petiole=0.06),
    "dock": dict(W=0.22, a=0.5, b=0.9, lobes=0, lobe=0.0, serr=0.02, serr_n=8, bend=0.1, petiole=0.15),
    "nettle": dict(W=0.36, a=0.35, b=1.3, lobes=0, lobe=0.0, serr=0.14, serr_n=14, bend=0.05, petiole=0.15),
    "dandelion": dict(W=0.17, a=0.75, b=0.55, lobes=5.0, lobe=0.72, serr=0.0, bend=0.1, petiole=0.0),
    "plantain": dict(W=0.13, a=0.75, b=0.8, lobes=0, lobe=0.0, serr=0.0, bend=0.05, petiole=0.0, parallel=5),
    "leaflet": dict(W=0.29, a=0.6, b=1.1, lobes=0, lobe=0.0, serr=0.11, serr_n=24, bend=0.04, petiole=0.0),
    "needle": dict(W=0.05, a=0.5, b=0.5, lobes=0, lobe=0.0, serr=0.0, bend=0.05, petiole=0.0),
}


def _width(sp, u):
    a, b = sp["a"], sp["b"]
    peak = (a ** a) * (b ** b) / ((a + b) ** (a + b))
    w = sp["W"] * np.clip(u, 0, 1) ** a * np.clip(1 - u, 0, 1) ** b / peak
    if sp["lobes"]:
        # rounded lobes with sinuses between them, none at the base
        k = 0.5 + 0.5 * np.cos(2 * math.pi * sp["lobes"] * u + 0.6)
        w = w * (1 - sp["lobe"] * k ** 3 * np.clip(u * 4, 0, 1))
    if sp["serr"]:
        tri = np.abs(((u * sp["serr_n"]) % 1.0) - 0.5) * 2
        w = w * (1 - sp["serr"] * tri)
    if sp.get("notches"):
        # oak: rounded lobes separated by narrow sinuses, auricles at the base, a round tip lobe
        f = (u * sp["notches"] + 0.15) % 1.0
        sinus = np.exp(-((f - 0.5) / 0.13) ** 2) * np.clip((u - 0.08) * 6, 0, 1) * np.clip((0.9 - u) * 8, 0, 1)
        w = w * (1 - sp["notch"] * sinus)
    return w


def leaf(cv, base, ang, length, kind, col, rng, under=False, shade=1.0, shadow=0.32, rib=None, age=0.0):
    """One leaf from its base (output px) pointing at `ang` (radians), `length` px long. `rib`
    colours the midrib (dock's red one); `age` browns the margin inwards (older, lower leaves)."""
    sp = LEAF[kind]
    ax = np.array([math.cos(ang), math.sin(ang)])
    pp = np.array([-ax[1], ax[0]])
    bend = sp["bend"] * rng.uniform(-1, 1)
    centre = np.asarray(base, float) + ax * length * 0.5
    off = np.array([1.0, -1.5]) * max(1.5, length * 0.05)
    win = cv.window(centre[0], centre[1], length * 0.62 + abs(off).max() + 3)
    if win is None:
        return
    sl, X, Y = win

    def field(ox, oy):
        dx, dy = X - base[0] - ox, Y - base[1] - oy
        u = (dx * ax[0] + dy * ax[1]) / length
        s = (dx * pp[0] + dy * pp[1]) / length
        s = s - bend * u * u
        w = _width(sp, u)
        d = np.minimum((w - np.abs(s)), np.minimum(u, 1 - u) * 0.5) * length
        return d, u, s, w

    if shadow:
        ds, _, _, _ = field(*off)
        soft = max(2.0, length * 0.06)
        cv.darken(sl, shadow * np.clip(ds / soft + 0.5, 0, 1))
    d, u, s, w = field(0.0, 0.0)
    alpha = np.clip(d * SS + 0.5, 0, 1)
    if not alpha.any():
        return
    sw = np.maximum(w, 1e-4)
    rel = np.clip(np.abs(s) / sw, 0, 1)
    fold = rng.choice([-1.0, 1.0])
    tone = 1.0 + (0.08 if under else 0.13) * np.tanh(s / 0.015) * fold
    tone *= 0.93 + 0.12 * (1 - rel ** 2)                       # leaf blade cups slightly
    tone *= np.where(d < 1.2, 0.82, 1.0)                        # thin dark margin
    c = col.copy()
    if under:
        lum = c @ np.array([0.3, 0.6, 0.1])
        c = (c * 0.55 + np.array([lum * 0.9, lum * 1.05, lum * 0.8]) * 0.45) * 1.35
    rgb = c[None, None, :] * tone[..., None] * shade
    # yellower towards the base
    rgb = rgb * (1 + 0.12 * np.clip(0.25 - u, 0, 0.25)[..., None] * np.array([1.2, 0.6, -1.5]))
    # midrib and lateral veins: paler, finer towards the tip
    uc = np.clip(u, 0, 1)
    rib_m = np.exp(-(s / (0.010 + 0.006 * (1 - uc))) ** 2) * (1 - uc) ** 0.4
    nv = 7 if kind not in ("willow", "dandelion") else 14
    if sp.get("parallel"):
        vp = (rel * sp["parallel"]) % 1.0
        vein = np.exp(-((vp - 0.5) / 0.09) ** 2) * (u > 0.05) * 1.6
    else:
        vp = (u * nv - np.abs(s) / sw * 0.9) % 1.0
        vein = np.exp(-((vp - 0.5) / 0.045) ** 2) * (rel < 0.85) * (u > 0.08)
    pale = c * 1.55 + np.array([0.02, 0.025, 0.0])
    rgb = rgb + (pale - rgb) * (vein * (0.18 if under else 0.1))[..., None]
    rc = pale if rib is None else np.asarray(rib, float)
    rgb = rgb + (rc - rgb) * (rib_m * (0.6 if under else 0.45 if rib is None else 0.8))[..., None]
    if age > 0:
        # browning from the margin and the tip inwards, blotchy
        edge = np.clip(1 - d / (length * 0.09 * age + 1e-3), 0, 1) + np.clip((u - 1 + 0.35 * age) * 3, 0, 1)
        blot = 0.5 + 0.5 * np.sin(X * 0.37 + Y * 0.23) * np.sin(X * 0.11 - Y * 0.29)
        k = np.clip(edge * (0.6 + 0.6 * blot), 0, 1) * min(age * 1.4, 1)
        brown = np.array([0.30, 0.17, 0.05]) * (0.7 + 0.5 * blot[..., None])
        yel = np.array([0.42, 0.36, 0.06])
        rgb = rgb + (yel - rgb) * (age * 0.45 * (1 - k))[..., None]
        rgb = rgb + (brown - rgb) * k[..., None]
    cv.blend(sl, rgb, alpha)


def _jitter(col, rng, v=0.12, h=0.08):
    """A leaf colour varied in value and a little in hue (towards yellow or blue)."""
    hh = rng.normal(0, h)
    return col * rng.uniform(1 - v, 1 + v) * np.array([1 + hh, 1.0, 1 - 1.4 * hh])


# -- sprays ---------------------------------------------------------------------------------------

SPRAY = {
    # leaf kind, palette (sRGB), leaf length / quarter, leaves per quarter, side shoots,
    # underside share, twig colour, layout ("spray", "hang", "dense")
    "oak": ("oak", ["#37551f", "#3f5c24", "#4a6629", "#2f4a1b", "#52692e"], 0.21, 40, 5, 0.15, "#5a4a38", "spray"),
    "birch": ("birch", ["#4a6e26", "#557a2c", "#3f6122", "#62873a", "#6f8f3c"], 0.062, 46, 3, 0.18, "#3a2c26", "strand"),
    "willow": ("willow", ["#6a8245", "#77904f", "#5c743c", "#839a58"], 0.2, 70, 3, 0.35, "#7a6a3a", "strand"),
    "privet": ("privet", ["#2c4520", "#355226", "#26401c", "#3f5c2a"], 0.11, 120, 6, 0.08, "#4a3a2a", "dense"),
    "beech": ("beech", ["#3f5f22", "#4b6d28", "#365420", "#58782f"], 0.14, 75, 6, 0.1, "#5a4a3a", "spray"),
    "bramble": ("leaflet", ["#2d4b23", "#355629", "#28441f", "#3d5e2b", "#304d22"], 0.15, 7, 0, 0.2, "#5e2a34", "bramble"),
    "shrub": ("beech", ["#2f4a22", "#3a5a28", "#2a4420", "#476a2e", "#344f24"], 0.2, 30, 3, 0.15, "#4a3a2a", "cluster"),
}


def _spray_quarter(cv, ox, oy, q, spec, rng):
    kind, pal, lsz, nleaf, nshoot, under, twig, layout = spec
    pal = [srgb_lin(c) for c in pal]
    twig_c = srgb_lin(twig)
    cv.set_clip(ox + 1, oy + 1, ox + q - 1, oy + q - 1)
    base = np.array([ox + q * 0.5, oy + 1.0])
    shoots = []
    if layout == "bramble":
        _bramble_quarter(cv, ox, oy, q, spec, rng)
        cv.reset_clip()
        return
    if layout == "cluster":
        # an evergreen shrub's shoot (Viburnum, Photinia): a twig forking into 2-3 tips, each
        # ending in a whorl of glossy ovate leaves, older leaves lower down the stems
        main = curve(base, math.pi / 2 + rng.uniform(-0.15, 0.15), q * 0.35, rng.uniform(-0.3, 0.3), n=6)
        stroke(cv, main, 4.0, 3.0, twig_c, twig_c * 1.2)
        tips = []
        for k in range(nshoot):
            a = math.pi / 2 + (k - (nshoot - 1) / 2) * 0.55 + rng.normal(0, 0.12)
            sh = curve(main[-1], a, q * rng.uniform(0.2, 0.3), rng.normal(0, 0.3), n=5)
            stroke(cv, sh, 3.0, 2.0, twig_c, twig_c * 1.3)
            tips.append(sh)
        leaves = []
        for sh in tips:
            da = math.atan2(*(sh[-1] - sh[-2])[::-1])
            n = int(rng.integers(6, 10))
            for j in range(n):
                ang = da + (j / (n - 1) - 0.5) * rng.uniform(2.6, 3.4) + rng.normal(0, 0.12)
                leaves.append((sh[-1] + rng.normal(0, 2, 2), ang, q * lsz * rng.uniform(0.75, 1.1), rng.uniform()))
            for j in range(2):                                         # older leaves lower down
                p = sh[1 + j]
                leaves.append((p, da + (1 if j else -1) * rng.uniform(0.9, 1.4), q * lsz * rng.uniform(0.85, 1.05), rng.uniform() * 0.5))
        leaves.sort(key=lambda L: L[3])
        for p, ang, ln, depth in leaves:
            col = _jitter(pal[rng.integers(len(pal))], rng)
            pet = np.array([math.cos(ang), math.sin(ang)]) * ln * 0.12
            stroke(cv, [p, p + pet], 1.6, 1.2, col * 0.8)
            leaf(cv, p + pet, ang, ln, kind, col, rng, under=rng.uniform() < under, shade=0.62 + 0.38 * depth)
        cv.reset_clip()
        return
    if layout == "strand":
        # weeping strands hanging from the top centre (willow, birch): the card hangs from its top edge
        top = np.array([ox + q * 0.5, oy + q - 1.0])
        spread = 0.25 if kind == "willow" else 0.38
        for k in range(nshoot):
            d0 = -math.pi / 2 + rng.uniform(-spread, spread)
            pts = curve(top, d0, q * rng.uniform(0.8, 0.95), rng.uniform(-0.3, 0.3))
            if kind == "birch":
                # fine pendulous twigs wander a little
                pts = [p + np.array([math.sin(j * 0.9 + k) * q * 0.012, 0.0]) for j, p in enumerate(pts)]
            shoots.append((pts, 1.6 if kind == "willow" else 1.2))
    else:
        main = curve(base, math.pi / 2 + rng.uniform(-0.2, 0.2), q * rng.uniform(0.55, 0.66), rng.uniform(-0.5, 0.5))
        shoots.append((main, 3.0))
        for k in range(nshoot):
            t = rng.uniform(0.15, 0.8)
            i = max(1, int(t * (len(main) - 1)))
            side = 1 if k % 2 else -1
            d0 = math.atan2(*(main[i] - main[i - 1])[::-1]) + side * rng.uniform(0.45, 0.95)
            ln = q * rng.uniform(0.24, 0.38) * (1.1 - 0.4 * t)
            shoots.append((curve(main[i], d0, ln, -side * rng.uniform(0.1, 0.6)), 1.8))
    leaves = []
    total_len = sum(sum(np.linalg.norm(b - a) for a, b in zip(p[:-1], p[1:])) for p, _ in shoots)
    for pts, w in shoots:
        L = sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))
        n = max(2, int(round(nleaf * L / total_len)))
        for j in range(n):
            t = (j + rng.uniform(0.2, 0.8)) / n
            if layout == "spray" and kind == "oak":
                t = 1 - (1 - t) ** 2.2                       # oak leaves bunch at the shoot tips
            f = t * (len(pts) - 1)
            i = min(int(f), len(pts) - 2)
            p = pts[i] + (pts[i + 1] - pts[i]) * (f - i)
            da = math.atan2(*(pts[i + 1] - pts[i])[::-1])
            side = 1 if j % 2 else -1
            if layout == "strand":
                ang = da + side * (rng.uniform(0.15, 0.5) if kind == "willow" else rng.uniform(0.35, 0.95))
            elif layout == "dense":
                ang = da + side * rng.uniform(0.3, 1.4)
            else:
                ang = da + side * rng.uniform(0.35, 1.05)
            ln = q * lsz * rng.uniform(0.75, 1.15) * (0.75 + 0.3 * t)
            leaves.append((p, ang, ln, rng.uniform()))
        tip_a = math.atan2(*(pts[-1] - pts[-2])[::-1])
        leaves.append((pts[-1], tip_a + rng.normal(0, 0.15), q * lsz * 1.05, 1.0))
    if layout == "dense":
        # clipped hedges and shrubs: leaves everywhere, overlapping
        for j in range(nleaf):
            p = np.array([ox, oy]) + rng.uniform([0.2, 0.12], [0.8, 0.8]) * q
            leaves.append((p, rng.uniform(0, 2 * math.pi), q * lsz * rng.uniform(0.75, 1.1), rng.uniform()))
    # twigs first, then leaves back to front (the back ones darker, in the spray's own shade)
    for pts, w in shoots:
        stroke(cv, pts, w, w * 0.45, twig_c, twig_c * 1.3)
    leaves.sort(key=lambda L: L[3])
    for p, ang, ln, depth in leaves:
        sp = LEAF[kind]
        pet = np.array([math.cos(ang), math.sin(ang)]) * ln * sp["petiole"]
        col = _jitter(pal[rng.integers(len(pal))], rng)
        if rng.uniform() < 0.025:
            col = col * np.array([1.5, 1.15, 0.6])           # the odd yellowing leaf
        if sp["petiole"] > 0.02:
            stroke(cv, [p, p + pet], 1.3, 1.0, col * 0.8)
        leaf(cv, p + pet, ang, ln, kind, col, rng, under=rng.uniform() < under, shade=0.62 + 0.38 * depth)
    cv.reset_clip()


def dot(cv, c, r, col):
    """A small shaded disc (berries, drupelets, flower centres)."""
    c = np.asarray(c, float)
    stroke(cv, [c, c + (0.01, 0.0)], 2 * r, 2 * r, np.asarray(col, float))


def compound_leaf(cv, base, ang, L, col, rng, n=5, under=False, shade=1.0, pet_col=None):
    """A bramble leaf: a petiole and 3 or 5 serrated leaflets fanning from its end (palmate)."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    end = np.asarray(base, float) + ax * L * 0.55
    stroke(cv, [base, end], 2.6, 1.8, pet_col if pet_col is not None else col * 0.8, shadow=0.15)
    # each leaflet on its own short stalk, spread so the leaf reads as compound, not palmate
    parts = [(-1.85, 0.6, 0.05), (1.85, 0.6, 0.05)] if n == 5 else []
    parts += [(-0.95, 0.84, 0.08), (0.95, 0.84, 0.08), (0.0, 1.0, 0.2)]
    for off, ls, stalk in parts:
        a = ang + off + rng.normal(0, 0.08)
        st = end + np.array([math.cos(a), math.sin(a)]) * L * stalk
        stroke(cv, [end, st], 1.6, 1.2, pet_col if pet_col is not None else col * 0.8)
        leaf(cv, st, a, L * ls * rng.uniform(0.9, 1.08), "leaflet", _jitter(col, rng, 0.06, 0.03), rng,
             under=under, shade=shade, shadow=0.3)


def _bramble_quarter(cv, ox, oy, q, spec, rng):
    """An arching bramble cane with thorns and alternate compound leaves; one quarter carries
    flowers and unripe fruit, another ripe and ripening blackberries."""
    kind, pal, lsz, nleaf, nshoot, under, twig, layout = spec
    pal = [srgb_lin(c) for c in pal]
    k = int(ox // q + 2 * (oy // q))
    cane0, cane1 = srgb_lin("#5e2a34"), srgb_lin("#58602c")
    base = np.array([ox + q * 0.5, oy + 1.0])
    main = curve(base, math.pi / 2 + rng.uniform(-0.25, 0.25), q * 0.9, rng.uniform(-0.7, 0.7), n=14)
    stroke(cv, main, 5.0, 3.2, cane0, cane1, shadow=0.25)
    # thorns: hooked back towards the base, on both sides
    for j in range(1, len(main) - 1):
        for side in (-1, 1):
            if rng.uniform() < 0.25:
                continue
            d = main[j + 1] - main[j - 1]
            d /= np.linalg.norm(d)
            nrm = np.array([-d[1], d[0]]) * side
            p0 = main[j] + nrm * 2.0 + d * rng.uniform(-3, 3)
            tip = p0 + nrm * 5.5 - d * 3.0
            stroke(cv, [p0, tip], 2.6, 0.4, srgb_lin("#7a3a3a"), srgb_lin("#c9a890"))
    leaves = []
    for j in range(nleaf):
        t = 0.12 + 0.78 * (j + rng.uniform(0.0, 0.6)) / nleaf
        f = t * (len(main) - 1)
        i = min(int(f), len(main) - 2)
        p = main[i] + (main[i + 1] - main[i]) * (f - i)
        da = math.atan2(*(main[i + 1] - main[i])[::-1])
        side = 1 if j % 2 else -1
        leaves.append((p, da + side * rng.uniform(0.55, 1.1), q * lsz * rng.uniform(0.85, 1.12) * (1.12 - 0.35 * t),
                       5 if t < 0.55 else 3, rng.uniform()))
    tip_a = math.atan2(*(main[-1] - main[-2])[::-1])
    leaves.append((main[-1], tip_a, q * lsz * 0.55, 3, 1.0))
    leaves.sort(key=lambda L: L[4])
    for p, a, L, n, depth in leaves:
        col = pal[rng.integers(len(pal))]
        r = rng.uniform()
        if r < 0.05:
            col = srgb_lin("#5a3424") * rng.uniform(0.8, 1.2)     # an old leaf turning red-brown
        elif r < 0.16:
            col = col * np.array([1.25, 0.9, 1.1])                # purple-tinged in the sun
        compound_leaf(cv, p, a, L, col, rng, n, under=rng.uniform() < under, shade=0.65 + 0.35 * depth,
                      pet_col=cane0 * 1.2)
    if k in (2, 3):
        # a truss at the cane tip and one on a side shoot
        for c0 in (main[-1], main[len(main) // 2], main[len(main) // 3], main[2 * len(main) // 3]):
            for b in range(rng.integers(3, 6)):
                a = tip_a + rng.uniform(-1.3, 1.3)
                end = c0 + np.array([math.cos(a), math.sin(a)]) * rng.uniform(12, 26)
                stroke(cv, [c0, end], 1.6, 1.2, srgb_lin("#556a2c"))
                if k == 2 and rng.uniform() < 0.6:
                    for pe in range(5):                                # five white-pink petals
                        pa = 2 * math.pi * pe / 5 + rng.uniform(-0.2, 0.2)
                        leaf(cv, end, pa, rng.uniform(9, 12), "privet", srgb_lin("#efe2e6") * rng.uniform(0.9, 1.05),
                             rng, shadow=0.1)
                    dot(cv, end, 2.6, srgb_lin("#c2c060"))
                else:
                    ripe = rng.uniform()
                    col = (srgb_lin("#1a0e16") if ripe < (0.55 if k == 3 else 0.1) else
                           srgb_lin("#8c1c22") if ripe < 0.8 else srgb_lin("#7c8a3a"))
                    for dpl in range(20):                              # drupelets
                        o = rng.normal(0, 3.4, 2) * np.array([0.85, 1.1])
                        dot(cv, end + o, 2.7, col * rng.uniform(0.8, 1.3))
                        dot(cv, end + o + (-0.8, 0.9), 0.9, col * 2.8 + 0.05)   # highlight


def mottle(cv, rng, amount=0.12):
    """Fine blotchy variation over everything painted (leaf texture, sun flecks)."""
    h, w = cv.rgb.shape[:2]
    m = fbm((h, w), (max(1, 32 * w // h), max(1, 32 * h // w)) if w != h else (32, 32), rng, 4)
    cv.rgb *= (1 - amount + 2 * amount * m)[..., None]


def spray_atlas(species, px=1024, seed=0):
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    q = px // 2
    for k in range(4):
        _spray_quarter(cv, (k % 2) * q, (k // 2) * q, q, SPRAY[species], rng)
    mottle(cv, rng)
    return cv.result()


def leaf_wall(species, px=512, seed=0, n=900):
    """A tileable face of densely packed leaves over deep shade (a clipped hedge, the body of a
    shrub): opaque, so it carries the mass while cut-out sprays make the silhouette."""
    rng = np.random.default_rng(seed)
    kind, pal, lsz, *_ = SPRAY[species]
    pal = [srgb_lin(c) for c in pal]
    cv = Canvas(px)
    cv.rgb[...] = pal[0] * 0.62
    cv.a[...] = 1.0
    size = px * lsz * 0.9
    for k in range(n):
        p = rng.uniform(0, px, 2)
        ang = rng.uniform(0, 2 * math.pi)
        ln = size * rng.uniform(0.75, 1.15)
        depth = k / n
        col = _jitter(pal[rng.integers(len(pal))], rng)
        for dx in (-px, 0, px):
            for dy in (-px, 0, px):
                q = p + (dx, dy)
                if -ln < q[0] < px + ln and -ln < q[1] < px + ln:
                    leaf(cv, q, ang, ln, kind, col, np.random.default_rng(k), shade=0.78 + 0.32 * depth, shadow=0.18)
    mottle(cv, rng, 0.15)
    return cv.result()


def strand_strip(seed=0, w=256, h=1024):
    """Weeping willow whips as a strip that tiles along v: three thin hanging stems side by side
    (out of step), long lanceolate leaves pointing down them, silvery undersides on some. The
    strip is 0.42 m across and repeats every 1.68 m."""
    rng = np.random.default_rng(seed)
    cv = Canvas(w, h)
    pal = [srgb_lin(c) for c in ("#6f8a45", "#7c964f", "#62793c", "#8aa25a", "#71883f", "#58703a")]
    leaves = []
    for st, x0 in enumerate((0.2, 0.5, 0.8)):
        amp, ph = rng.uniform(3, 7), rng.uniform(0, 6.3)
        xs = lambda y, x0=x0, amp=amp, ph=ph: w * x0 + amp * math.sin(2 * math.pi * y / h * 2 + ph)
        for dy in (-h, 0, h):
            stroke(cv, [np.array([xs(y), y + dy]) for y in np.linspace(0, h, 41)], 2.4, 2.4, srgb_lin("#7c6a3a"))
        n = int(rng.integers(52, 62))
        for i in range(n):
            y = (i + rng.uniform(0, 0.7)) * h / n
            side = 1 if i % 2 else -1
            leaves.append((xs(y), y, -math.pi / 2 + side * rng.uniform(0.15, 0.6), rng.uniform(60, 88),
                           pal[rng.integers(len(pal))] * rng.uniform(0.85, 1.15), rng.uniform() < 0.3, rng.uniform(), st * 100 + i))
    leaves.sort(key=lambda L: L[6])
    for x, y, ang, ln, col, und, depth, i in leaves:
        for dy in (-h, 0, h):
            leaf(cv, (x, y + dy), ang, ln, "willow", col, np.random.default_rng(i), under=und,
                 shade=0.65 + 0.35 * depth, shadow=0.25)
    mottle(cv, rng, 0.1)
    return cv.result()


def blade_texture(w=64, h=256, seed=0):
    """A neutral grass/reed blade (u across, v along, tiling): pale midrib, darker margins, fine
    parallel veins. Near white, so the vertex colour carries each blade's own colour."""
    rng = np.random.default_rng(seed)
    U, V = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    across = np.abs(U - 0.5) * 2
    val = 0.93 - 0.16 * across ** 2 + 0.07 * np.exp(-((U - 0.5) / 0.05) ** 2)
    val += 0.035 * np.cos(2 * math.pi * U * 9)
    val += (fbm((h, w), (4, 16), rng, 3) - 0.5) * 0.14
    rgb = np.clip(val, 0, 1)[..., None] * np.array([1.0, 1.0, 0.95])
    return np.concatenate([rgb, np.ones((h, w, 1))], -1)


def seedhead_atlas(seed=0, px=1024):
    """Four grass culms with their panicles, each a 1/4 x 1 cell, culm entering bottom centre:
    a bleached cock's-foot, a fresh green-purple one, an airy hair-grass, a slender bleached spike."""
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    cw = px // 4
    for k in range(4):
        ox = k * cw
        cv.set_clip(ox + 1, 1, ox + cw - 1, px - 1)
        cx = ox + cw / 2
        if k == 1:
            pal = [srgb_lin(c) for c in ("#6f8040", "#6d5060", "#7a8a48", "#5e4a58")]
            culm = srgb_lin("#5e7034")
        elif k == 2:
            pal = [srgb_lin(c) for c in ("#b5a7a8", "#9e8f95", "#c9c0a6", "#a89a8a")]
            culm = srgb_lin("#8a8a5a")
        else:
            pal = [srgb_lin(c) for c in ("#bfae7a", "#a8966a", "#cbbd8f", "#9a8a60")]
            culm = srgb_lin("#a39467")
        lean = rng.normal(0, 8)
        axis = curve((cx, 2), math.pi / 2 + lean / 400, px * 0.95, rng.normal(0, 0.12), n=16)
        if k == 3:   # a culm bent over part way up
            axis = curve((cx - 30, 2), math.pi / 2 + 0.05, px * 0.5, 0.0, n=8)
            axis += curve(axis[-1], math.pi / 2 - 0.45, px * 0.42, -0.1, n=8)[1:]
        stroke(cv, axis, 4.0, 2.4, culm * 0.8, culm)
        top = len(axis) - 1
        if k in (0, 1):
            for j in range(6):
                t = 0.62 + 0.36 * j / 6
                i = int(t * top)
                side = 1 if j % 2 else -1
                a = math.pi / 2 + side * rng.uniform(0.4, 0.9)
                ln = rng.uniform(25, 60) * (1.2 - 0.5 * (t - 0.6) / 0.4)
                end = axis[i] + np.array([math.cos(a), math.sin(a)]) * ln
                stroke(cv, [axis[i], end], 1.8, 1.2, culm)
                for sp in range(46):
                    c = end + rng.normal(0, 7, 2)
                    leaf(cv, c, math.pi / 2 + rng.normal(0, 0.6), rng.uniform(9, 15), "willow",
                         pal[rng.integers(len(pal))] * rng.uniform(0.8, 1.2), rng, shadow=0.12)
            for sp in range(30):
                c = axis[-1] + rng.normal(0, 6, 2) - (0, rng.uniform(0, 30))
                leaf(cv, c, math.pi / 2 + rng.normal(0, 0.5), rng.uniform(9, 15), "willow",
                     pal[rng.integers(len(pal))] * rng.uniform(0.8, 1.2), rng, shadow=0.12)
        elif k == 2:
            for j in range(40):
                t = rng.uniform(0.55, 0.98)
                i = int(t * top)
                a = math.pi / 2 + rng.normal(0, 0.6)
                br = curve(axis[i], a, rng.uniform(40, 110) * (1.2 - t), rng.normal(0, 0.5), n=5)
                stroke(cv, br, 1.2, 0.8, culm * 1.1)
                for sp in range(rng.integers(2, 5)):
                    c = br[-1] + rng.normal(0, 4, 2)
                    leaf(cv, c, a + rng.normal(0, 0.4), rng.uniform(6, 9), "willow",
                         pal[rng.integers(len(pal))] * rng.uniform(0.85, 1.2), rng, shadow=0.05)
        else:
            for sp in range(260):
                t = rng.uniform(0.72, 1.0)
                f = t * top
                i = min(int(f), top - 1)
                c = axis[i] + (axis[i + 1] - axis[i]) * (f - i) + rng.normal(0, 3.5, 2)
                d = axis[i + 1] - axis[i]
                a = math.atan2(d[1], d[0]) + rng.normal(0, 0.5)
                leaf(cv, c, a, rng.uniform(6, 9), "willow", pal[rng.integers(len(pal))] * rng.uniform(0.8, 1.15),
                     rng, shadow=0.1)
    cv.reset_clip()
    return cv.result()


def weed_atlas(seed=0, px=1024):
    """Weed parts on a 4 x 4 grid (256 px cells, each leaf's base at the bottom centre):
    0-1 dock leaves (red midrib; old, yellowing, holed), 2-3 dandelion leaves, 4-5 nettle leaves,
    6-7 ribwort plantain leaves, 8 dandelion head (face on), 9 seed clock, 10 dock seed spike,
    11 nettle tip, 12 dandelion head (side on), 13 plantain flower spike, 14 dead leaf,
    15 yellowing dandelion leaf."""
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    c = px // 4
    green = lambda h: srgb_lin(h) * rng.uniform(0.8, 0.95) * np.array([1.02, 0.95, 0.9])
    for k in range(16):
        ox, oy = (k % 4) * c, (k // 4) * c
        cv.set_clip(ox + 1, oy + 1, ox + c - 1, oy + c - 1)
        b = (ox + c / 2, oy + 3)
        cx, cy = ox + c / 2, oy + c / 2
        up = math.pi / 2
        if k in (0, 1):
            leaf(cv, b, up + rng.normal(0, 0.03), c * 0.95, "dock", green("#3d5c26" if k == 0 else "#4f6a2a"), rng,
                 rib=srgb_lin("#8a2a22"), age=0.0 if k == 0 else 0.5, shadow=0.0)
            if k == 1:
                for hh in range(5):
                    cv.cut(cx + rng.normal(0, 14), oy + rng.uniform(60, 210), rng.uniform(3, 8))
        elif k in (2, 3, 15):
            leaf(cv, b, up, c * 0.95, "dandelion", green("#4b6e2b"), rng, rib=srgb_lin("#8a6a6a") if k == 3 else None,
                 age=0.4 if k == 15 else 0.0, shadow=0.0)
        elif k in (4, 5):
            stroke(cv, [b, (b[0], b[1] + c * 0.15)], 4, 3, green("#4a6a2c"))
            leaf(cv, (b[0], b[1] + c * 0.15), up, c * (0.8 if k == 4 else 0.62), "nettle",
                 green("#2e4a1e" if k == 4 else "#46682a"), rng, shadow=0.0)
        elif k in (6, 7):
            leaf(cv, b, up, c * 0.95, "plantain", green("#3e5e28" if k == 6 else "#4a6a2e"), rng, shadow=0.0)
        elif k == 8:
            for ring, (r0, r1, n, col, w) in enumerate(((20, 104, 70, "#f4c81c", 8), (12, 70, 50, "#f0b814", 6),
                                                         (4, 38, 30, "#e5a412", 5))):
                for j in range(n):
                    a = 2 * math.pi * (j + rng.uniform(0, 0.5)) / n
                    d = np.array([math.cos(a), math.sin(a)])
                    stroke(cv, [(cx, cy) + d * r0, (cx, cy) + d * r1 * rng.uniform(0.9, 1.0)], w, w * 0.7,
                           srgb_lin(col) * rng.uniform(0.9, 1.1), shadow=0.08)
            dot(cv, (cx, cy), 14, srgb_lin("#d08a0c"))
        elif k == 9:
            for j in range(260):
                v = rng.normal(0, 1, 3)
                v /= np.linalg.norm(v)
                if v[2] < -0.2:
                    continue
                e = np.array([cx, cy]) + v[:2] * 108 * rng.uniform(0.9, 1.0)
                stroke(cv, [(cx, cy) + v[:2] * 8, e], 1.1, 0.7, srgb_lin("#e8e6dc") * rng.uniform(0.85, 1.05))
                dot(cv, e, 2.0, srgb_lin("#f6f6f0"))
            dot(cv, (cx, cy), 9, srgb_lin("#6a5a40"))
        elif k == 10:
            # a dock seed spike: a slender axis with whorls of three-winged red-brown seeds
            axis = curve(b, up, c * 0.97, rng.normal(0, 0.06), n=12)
            stroke(cv, axis, 3.2, 1.6, srgb_lin("#5a3a1e"))
            for j in range(2, len(axis)):
                for sd in range(int(rng.integers(5, 9))):
                    a = rng.uniform(0, 2 * math.pi)
                    off = np.array([math.cos(a) * rng.uniform(5, 13), rng.uniform(-6, 6)])
                    stroke(cv, [axis[j], axis[j] + off * 0.6], 1.0, 0.8, srgb_lin("#6a4a28"))
                    col = srgb_lin(["#8a4626", "#7a3a1e", "#9a5a30", "#6a3218", "#a0683a"][rng.integers(5)])
                    leaf(cv, axis[j] + off, a, rng.uniform(6, 9), "privet", col * rng.uniform(0.85, 1.15), rng, shadow=0.15)
        elif k == 11:
            for j in range(5):
                a = up + rng.normal(0, 0.45)
                if j < 3:
                    p0 = np.array([cx, oy + 90 + 20 * j])
                    hang = curve(p0, -up + rng.normal(0, 0.4), rng.uniform(50, 80), 0.2, n=6)
                    stroke(cv, hang, 1.4, 1.0, srgb_lin("#8a9a4a"))
                    for q in hang[1:]:
                        dot(cv, q, 2.2, srgb_lin("#9aa252"))
                leaf(cv, (cx, oy + 40 + 25 * j), a, rng.uniform(70, 110), "nettle", green("#466a2a"), rng, shadow=0.2)
        elif k == 12:
            for j in range(9):
                a = -up + rng.uniform(-0.9, 0.9)
                stroke(cv, [(cx, oy + 40), (cx, oy + 40) + np.array([math.cos(a), math.sin(a)]) * 30], 5, 1, srgb_lin("#5a7a30"))
            for j in range(16):
                a = up + rng.uniform(-0.6, 0.6)
                stroke(cv, [(cx, oy + 8), (cx, oy + 8) + np.array([math.cos(a), math.sin(a)]) * 40], 9, 6, srgb_lin("#5f7f32"))
            for j in range(60):
                a = up + rng.uniform(-1.25, 1.25)
                stroke(cv, [(cx, oy + 40), (cx, oy + 40) + np.array([math.cos(a), math.sin(a)]) * rng.uniform(50, 80)],
                       6, 4, srgb_lin("#f2c41c") * rng.uniform(0.9, 1.1), shadow=0.06)
        elif k == 13:
            stroke(cv, [b, (cx, oy + 210)], 3, 3, srgb_lin("#5d6a35"))
            for j in range(420):
                y = rng.uniform(oy + 120, oy + 210)
                dot(cv, (cx + rng.normal(0, 7), y), 2.4, srgb_lin(["#3a2a1a", "#4a3a22", "#5a4a2a"][rng.integers(3)]))
            for j in range(40):
                y = rng.uniform(oy + 140, oy + 168)
                side = rng.choice([-1, 1])
                stroke(cv, [(cx + side * 8, y), (cx + side * rng.uniform(16, 24), y + rng.normal(0, 4))], 1.0, 1.0,
                       srgb_lin("#e8e4d0"))
                dot(cv, (cx + side * 22, y), 1.8, srgb_lin("#efe8c8"))
        elif k == 14:
            leaf(cv, b, up, c * 0.9, "dock", srgb_lin("#5a3a22"), rng, age=1.0, shadow=0.0)
    cv.reset_clip()
    mottle(cv, rng, 0.08)
    return cv.result()


def litter_atlas(seed=0, px=1024):
    """Ground litter patches (2 x 2, each a ragged disc on a quarter): broadleaf leaf litter and
    twigs, spruce needles and cones, wet mud with dead reed straps, dry soil with dead grass."""
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    q = px // 2
    H = W = q * SS
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / SS, (np.arange(W) + 0.5) / SS, indexing="ij")
    for k in range(4):
        ox, oy = (k % 2) * q, (k // 2) * q
        cv.set_clip(ox + 1, oy + 1, ox + q - 1, oy + q - 1)
        dx, dy = xx - q / 2, yy - q / 2
        ang = np.arctan2(dy, dx)
        n = fbm((H, W), (6, 6), rng, 4)
        R = q * 0.43 * (1 + 0.07 * np.sin(3 * ang + k) + 0.05 * np.sin(5 * ang + 2 * k))
        d = np.hypot(dx, dy) / R + (n - 0.5) * 0.4
        fine = fbm((H, W), (40, 40), rng, 3)
        # soil thins out towards the edge into islands, so the patch has no outline
        a = np.clip((1 - d - (1 - fine) * 0.55 * np.clip(d * 1.6 - 0.45, 0, 1)) * 5 + 0.5, 0, 1)
        soil = {0: "#2e241a", 1: "#33271d", 2: "#4a4235", 3: "#3e3428"}[k]
        col = srgb_lin(soil) * (0.7 + 0.6 * fbm((H, W), (24, 24), rng, 4))[..., None]
        if k == 2:   # wet: darker hollows and a green film
            col = col * (1 - 0.3 * np.clip((n - 0.55) * 4, 0, 1))[..., None]
            col = col + (srgb_lin("#3a4a22") - col) * (0.35 * np.clip((fbm((H, W), (5, 5), rng, 3) - 0.6) * 4, 0, 1))[..., None]
        sl = (slice(oy * SS, oy * SS + H), slice(ox * SS, ox * SS + W))
        cv.blend(sl, col, a)
        def spot(spread=0.46):
            r = q * spread * math.sqrt(rng.uniform()) 
            t = rng.uniform(0, 2 * math.pi)
            return np.array([ox + q / 2 + r * math.cos(t), oy + q / 2 + r * math.sin(t)])
        if k == 0:
            pal = [srgb_lin(h) for h in ("#4e3a26", "#5e4a32", "#45331f", "#5a4430", "#3a2a1c", "#6a5638", "#4a4428")]
            for j in range(40):
                p = spot()
                stroke(cv, curve(p, rng.uniform(0, 6.3), rng.uniform(20, 60), rng.normal(0, 0.5), n=4),
                       rng.uniform(1.5, 3.2), 1.0, srgb_lin("#5a4a3a") * rng.uniform(0.8, 1.3), shadow=0.3)
            for j in range(900):
                leaf(cv, spot(0.4), rng.uniform(0, 6.3), rng.uniform(12, 20), "oak", pal[rng.integers(len(pal))] * rng.uniform(0.75, 1.2),
                     rng, shadow=0.35)
        elif k == 1:
            pal = [srgb_lin(h) for h in ("#6a4a32", "#7a5a3c", "#53392a", "#5e4a36", "#86684a")]
            for j in range(5000):
                p = spot(0.5)
                a = rng.uniform(0, 6.3)
                stroke(cv, [p, p + np.array([math.cos(a), math.sin(a)]) * rng.uniform(3, 6)], 1.3, 1.0,
                       pal[rng.integers(5)] * rng.uniform(0.75, 1.2))
            for j in range(7):
                p, a = spot(0.4), rng.uniform(0, 6.3)
                ax = np.array([math.cos(a), math.sin(a)])
                for t in np.linspace(-1, 1, 9):
                    w = 7 * math.sqrt(1 - t * t * 0.8)
                    dot(cv, p + ax * t * 16, w, srgb_lin("#5a3a22") * (0.8 + 0.3 * (j % 2)))
                    dot(cv, p + ax * t * 16 + (1, 1), w * 0.4, srgb_lin("#7a5232"))
        elif k == 2:
            for j in range(45):
                p = spot()
                stroke(cv, curve(p, rng.uniform(0, 6.3), rng.uniform(40, 130), rng.normal(0, 0.3), n=5),
                       rng.uniform(4, 7), 2.0, srgb_lin(["#8a7a50", "#6a5e40", "#9a8a5a", "#4a4430"][rng.integers(4)]),
                       shadow=0.3)
        else:
            for j in range(450):
                p = spot(0.5)
                stroke(cv, curve(p, rng.uniform(0, 6.3), rng.uniform(15, 50), rng.normal(0, 0.6), n=4), 1.6, 0.8,
                       srgb_lin(["#9a8a60", "#8a7a50", "#b0a070", "#6a5a3a"][rng.integers(4)]) * rng.uniform(0.8, 1.15),
                       shadow=0.2)
            for j in range(50):
                leaf(cv, spot(), rng.uniform(0, 6.3), rng.uniform(10, 16), "leaflet", srgb_lin("#5a3a24") * rng.uniform(0.8, 1.2),
                     rng, shadow=0.3)
    cv.reset_clip()
    return cv.result()


# -- conifer --------------------------------------------------------------------------------------

def conifer_atlas(px=1024, seed=0):
    """Spruce branchlets: each quarter a shoot entering at the left edge, side shoots alternating,
    all bristling with short needles; this year's growth paler at the tips."""
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    q = px // 2
    dark = [srgb_lin(c) for c in ("#1f3320", "#243a24", "#1a2c1b", "#2a4229")]
    fresh = srgb_lin("#55753a")
    twig = srgb_lin("#5a3f2a")
    for k in range(4):
        ox, oy = (k % 2) * q, (k // 2) * q
        cv.set_clip(ox + 1, oy + 1, ox + q - 1, oy + q - 1)
        main = curve((ox + 2, oy + q * 0.62), rng.uniform(-0.12, 0.05), q * 0.95, rng.uniform(-0.35, -0.1), n=12)
        axes = [(main, 1.0)]
        for j in range(9):
            t = 0.08 + 0.85 * j / 9 + rng.uniform(-0.02, 0.02)
            i = max(1, int(t * (len(main) - 1)))
            side = 1 if j % 2 else -1
            da = math.atan2(*(main[i] - main[i - 1])[::-1])
            ln = q * (0.5 - 0.32 * t) * rng.uniform(0.85, 1.1)
            sub = curve(main[i], da + side * rng.uniform(0.6, 0.9), ln, -side * 0.3, n=6, droop=0.15)
            axes.append((sub, 0.7))
            if ln > q * 0.2:
                for jj in range(3):
                    tt = 0.3 + 0.25 * jj
                    ii = max(1, int(tt * (len(sub) - 1)))
                    s2 = 1 if jj % 2 else -1
                    da2 = math.atan2(*(sub[ii] - sub[ii - 1])[::-1])
                    axes.append((curve(sub[ii], da2 + s2 * 0.7, ln * 0.35, 0.0, n=4), 0.5))
        for pts, w in axes:
            stroke(cv, pts, 3.2 * w, 1.4 * w, twig)
        needles = []
        for pts, w in axes:
            L = sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:]))
            n = int(L / 0.9)
            for j in range(n):
                t = (j + rng.uniform()) / n
                f = t * (len(pts) - 1)
                i = min(int(f), len(pts) - 2)
                p = pts[i] + (pts[i + 1] - pts[i]) * (f - i)
                da = math.atan2(*(pts[i + 1] - pts[i])[::-1])
                for side in (-1, 1):
                    a = da + side * rng.uniform(0.35, 1.25)
                    ln = rng.uniform(12, 19) * (0.75 + 0.35 * w)
                    young = t > 0.82
                    col = (fresh if young else dark[rng.integers(4)]) * rng.uniform(0.8, 1.2)
                    needles.append((p, a, ln, col, rng.uniform()))
                if rng.uniform() < 0.6:   # needles on top of the shoot hide the twig
                    a = da + rng.normal(0, 0.25)
                    needles.append((p, a, rng.uniform(8, 13), dark[rng.integers(4)] * rng.uniform(0.9, 1.25), rng.uniform() * 0.5 + 0.5))
        needles.sort(key=lambda n: n[4])
        for p, a, ln, col, depth in needles:
            tip = p + np.array([math.cos(a), math.sin(a)]) * ln
            stroke(cv, [p, tip], 2.6, 0.6, col * (0.6 + 0.4 * depth), col * (0.85 + 0.5 * depth))
    cv.reset_clip()
    return cv.result()


# -- grass, reeds, weeds --------------------------------------------------------------------------

def _blade(cv, x, y0, h, lean, w, c0, c1, rng, shadow=0.15):
    pts = [np.array([x, y0])]
    for k in range(1, 7):
        t = k / 6
        pts.append(np.array([x + lean * h * t * t, y0 + h * (t - 0.18 * abs(lean) * t * t)]))
    stroke(cv, pts, w, 0.6, c0, c1, shadow=shadow)


def tuft_atlas(kind, px=1024, seed=0):
    rng = np.random.default_rng(seed)
    cv = Canvas(px)
    q = px // 2
    for k in range(4):
        ox, oy = (k % 2) * q, (k // 2) * q
        cv.set_clip(ox + 1, oy + 1, ox + q - 1, oy + q - 1)
        if kind == "grass":
            base = [srgb_lin(c) for c in ("#2f4519", "#3a521f", "#344a1b")]
            tips = [srgb_lin(c) for c in ("#6b8436", "#7b8c43", "#8f9150", "#5d7a30")]
            straw = srgb_lin("#a39467")
            n = 140
            for j in range(n):
                x = ox + q * (0.5 + rng.normal(0, 0.16))
                h = q * rng.uniform(0.35, 0.92) * (1.0 - 1.2 * abs(x - ox - q / 2) / q)
                lean = rng.normal(0, 0.35)
                dry = rng.uniform() < 0.12
                c1 = straw * rng.uniform(0.8, 1.15) if dry else tips[rng.integers(4)] * rng.uniform(0.8, 1.2)
                _blade(cv, x, oy + 1, h, lean, rng.uniform(4, 7), base[rng.integers(3)], c1, rng)
            if k >= 2:   # seed heads on two of the four tufts (rough, uncut grass)
                for j in range(14):
                    x = ox + q * (0.5 + rng.normal(0, 0.14))
                    h = q * rng.uniform(0.75, 0.97)
                    lean = rng.normal(0, 0.15)
                    stem = curve((x, oy + 1), math.pi / 2 + lean, h, -lean * 0.8, n=8)
                    stroke(cv, stem, 2.2, 1.4, srgb_lin("#6d7a3c"), srgb_lin("#9a9460"))
                    tip = stem[-1]
                    for s in range(16):
                        a = math.pi / 2 + rng.normal(0, 0.5)
                        p = tip - np.array([0, rng.uniform(0, q * 0.12)])
                        leaf(cv, p, a, rng.uniform(8, 14), "willow", srgb_lin("#a09060") * rng.uniform(0.8, 1.2), rng, shadow=0.1)
        elif kind == "reed":
            base = srgb_lin("#3e4a22")
            tips = [srgb_lin(c) for c in ("#6f7d3c", "#7d8646", "#8c8a52")]
            for j in range(60):
                x = ox + q * (0.5 + rng.normal(0, 0.15))
                _blade(cv, x, oy + 1, q * rng.uniform(0.55, 0.97), rng.normal(0, 0.3), rng.uniform(6, 10), base, tips[rng.integers(3)], rng)
            for j in range(5):
                x = ox + q * (0.5 + rng.normal(0, 0.12))
                h = q * rng.uniform(0.75, 0.9)
                stroke(cv, [(x, oy + 1), (x + rng.normal(0, 4), oy + h)], 3.0, 2.0, srgb_lin("#5b6634"))
                leaf(cv, (x, oy + h - q * 0.14), math.pi / 2, q * 0.13, "dock", srgb_lin("#3d3024"), rng, shadow=0.2)
        elif kind == "weeds":
            green = [srgb_lin(c) for c in ("#3c6225", "#476d2c", "#355a21", "#52752f")]
            lk = ["dock", "dandelion", "nettle", "dock"][k]
            nl = [9, 14, 0, 7][k]
            for j in range(nl):
                a = math.pi / 2 + rng.normal(0, 0.55)
                p = (ox + q * 0.5 + rng.normal(0, q * 0.06), oy + 1)
                leaf(cv, p, a, q * rng.uniform(0.35, 0.62) * (0.8 if lk == "dandelion" else 1), lk, _jitter(green[rng.integers(4)], rng), rng)
            if lk == "nettle":
                for s in range(3):
                    x = ox + q * (0.35 + 0.15 * s)
                    h = q * rng.uniform(0.7, 0.92)
                    stroke(cv, [(x, oy + 1), (x + rng.normal(0, 6), oy + h)], 3.0, 1.5, srgb_lin("#4a5a2a"))
                    for j in range(7):
                        y = oy + h * (0.15 + 0.12 * j)
                        for side in (-1, 1):
                            leaf(cv, (x, y), math.pi / 2 + side * 1.0, q * 0.17 * (1.1 - 0.08 * j), "nettle", _jitter(green[0], rng), rng)
            if lk == "dandelion":
                for j in range(3):
                    x = ox + q * (0.5 + rng.normal(0, 0.12))
                    h = q * rng.uniform(0.45, 0.75)
                    stroke(cv, [(x, oy + 1), (x + rng.normal(0, 8), oy + h)], 2.5, 2.0, srgb_lin("#5d7a35"))
                    for s in range(24):
                        a = rng.uniform(0, 2 * math.pi)
                        stroke(cv, [(x, oy + h), (x + math.cos(a) * 11, oy + h + math.sin(a) * 6)], 3.0, 1.5, srgb_lin("#f0c419"), srgb_lin("#e8a811"))
            if k == 3:   # ragwort: yellow flower heads on a tall stem
                x = ox + q * 0.55
                h = q * 0.9
                stroke(cv, [(x, oy + 1), (x + 5, oy + h)], 3.0, 2.0, srgb_lin("#4e6630"))
                for s in range(22):
                    c = np.array([x + rng.normal(0, q * 0.1), oy + h + rng.normal(0, q * 0.04)])
                    stroke(cv, [(x + 5, oy + h - q * 0.1), c], 1.2, 1.0, srgb_lin("#4e6630"))
                    for r in range(8):
                        a = 2 * math.pi * r / 8
                        stroke(cv, [c, c + np.array([math.cos(a), math.sin(a)]) * 6], 2.4, 1.6, srgb_lin("#f2c21b"))
    cv.reset_clip()
    return cv.result()


# -- bark -----------------------------------------------------------------------------------------

def _normal_from_height(h, strength):
    """Tangent-space normal map (OpenGL convention: +y = up the image) from a height field."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def bark(kind, px=1024, seed=0):
    """Tileable bark covering 0.6 x 0.6 m: u runs round the trunk, v up it. Returns
    (albedo RGBA, normal RGB, roughness) arrays."""
    rng = np.random.default_rng(seed)
    S = (px, px)
    u = (np.arange(px) + 0.5) / px
    U, V = np.meshgrid(u, u)
    if kind in ("oak", "dead", "willow"):
        # corky plates split by dark furrows that wander, fork and rejoin up the trunk: cellular
        # noise on a grid stretched along the trunk (diagonally interlaced for willow)
        wx = (fbm(S, (3, 2), rng, 4) - 0.5) * 1.2
        wy = (fbm(S, (3, 2), rng, 4) - 0.5) * 0.8
        if kind == "willow":
            wx = wx + V * 2.0
        f1, edge = worley(S, (15, 4) if kind != "willow" else (10, 3), rng, (wx, wy))
        wide = 0.22 + 0.25 * fbm(S, (5, 3), rng, 3)                    # furrow width varies
        plate = np.clip((edge - 0.02) / wide, 0, 1)
        plate = plate * plate * (3 - 2 * plate)
        fibre = fbm(S, (90, 7), rng, 4)                                 # vertical fibres in the cork
        cross = np.clip((fbm(S, (10, 30), rng, 3) - 0.62) * 5, 0, 1)    # cross checks in the plates
        fine = fbm(S, (64, 32), rng, 4)
        h = plate * (0.7 + 0.3 * fibre) * (1 - 0.45 * cross) * (0.8 + 0.3 * fbm(S, (12, 6), rng, 3)) + 0.1 * fine
        h = np.clip(h, 0, 1)
        furrow, top = srgb_lin("#241d18"), srgb_lin("#6b6358")
        if kind == "dead":
            furrow, top = srgb_lin("#2e2924"), srgb_lin("#857b6e")
        alb = furrow + (top - furrow) * np.clip(h, 0, 1)[..., None] ** 0.8
        alb *= (0.8 + 0.4 * fbm(S, (4, 4), rng, 3))[..., None] * (0.85 + 0.3 * fibre)[..., None]
        lichen = np.clip((fbm(S, (8, 8), rng, 5) - (0.6 if kind == "dead" else 0.66)) * 6, 0, 1) * np.clip(h - 0.45, 0, 1) * 2
        alb = alb + (srgb_lin("#8c9472") - alb) * np.clip(lichen, 0, 0.8)[..., None]
        green = np.clip((fbm(S, (3, 2), rng, 3) - 0.6) * 4, 0, 1) * (1 - h)
        alb = alb + (srgb_lin("#3d4a26") - alb) * np.clip(green * 0.4, 0, 0.4)[..., None]
        rough = 0.82 + 0.14 * (1 - h)
        nrm = _normal_from_height(h, 18.0)
    elif kind in ("birch", "birchbase"):
        # chalky white with dark horizontal lenticels and black fissured patches; "birchbase" covers
        # 0.6 m round by 1.5 m up the butt: rough black plates at the ground giving way to white
        sv = 2.5 if kind == "birchbase" else 1.0
        base = srgb_lin("#d9d4c8")
        alb = np.broadcast_to(base, (px, px, 3)) * (0.9 + 0.15 * fbm(S, (3, int(8 * sv)), rng, 4))[..., None]
        h = 0.5 + 0.1 * fbm(S, (6, int(24 * sv)), rng, 3)
        dark = srgb_lin("#26221f")
        for i in range(int(140 * sv)):
            cx, cy = rng.uniform(0, px), rng.uniform(0, px)
            w, hh = rng.uniform(10, 70), rng.uniform(1.0, 3.5) / sv
            dx = (U * px - cx + px / 2) % px - px / 2
            dy = (V * px - cy + px / 2) % px - px / 2
            m = np.clip(1.2 - np.hypot(dx / w, dy / max(hh, 0.7)), 0, 1)
            alb = alb * (1 - 0.75 * m[..., None]) + dark * 0.75 * m[..., None]
            h = h - 0.25 * m
        patches = np.clip((fbm(S, (2, int(3 * sv)), rng, 5) - 0.6) * 5, 0, 1)
        fiss = np.abs(np.sin(math.pi * (U * 6 + (fbm(S, (2, 1), rng, 3) - 0.5) * 2)))
        blk = patches * (fiss < 0.5)
        alb = alb * (1 - 0.85 * blk[..., None]) + dark * 0.85 * blk[..., None]
        h = h - 0.4 * blk
        peel = np.clip((fbm(S, (8, int(30 * sv)), rng, 3) - 0.66) * 6, 0, 1)
        alb = alb * (1 - 0.15 * peel[..., None]) + srgb_lin("#c29a7a") * 0.15 * peel[..., None]
        rough = 0.7 + 0.2 * blk
        if kind == "birchbase":
            z = V * 1.5
            wx = (fbm(S, (3, 4), rng, 4) - 0.5) * 1.4
            f1, edge = worley(S, (16, 5), rng, (wx, wx * 0.3))
            plate = np.clip((edge - 0.02) / 0.18, 0, 1) * (0.75 + 0.25 * fbm(S, (30, 10), rng, 3))
            m = np.clip((0.95 - z + 0.9 * (fbm(S, (4, 3), rng, 4) - 0.5) + 0.35 * (fbm(S, (16, 12), rng, 3) - 0.5)) / 0.3, 0, 1)
            m = m * m * (3 - 2 * m)
            pl_col = srgb_lin("#3a3632") * (0.75 + 0.5 * fbm(S, (12, 20), rng, 3))[..., None]
            pl_col = pl_col * (0.35 + 0.65 * plate)[..., None]
            # in the transition the deep cracks still show white bark
            alb = alb * (1 - m[..., None]) + pl_col * m[..., None]
            h = h * (1 - m) + (0.25 + 0.7 * plate) * m
            rough = rough * (1 - m) + 0.92 * m
            nrm = _normal_from_height(h, 5.0 + 14.0 * m)
        else:
            nrm = _normal_from_height(h, 5.0)
    elif kind == "deadwood":
        # bare, weathered silver-grey wood: long checks along the grain, streaks, rot pockets
        grain = fbm(S, (40, 3), rng, 4)
        wx = (fbm(S, (3, 2), rng, 3) - 0.5) * 0.6
        f1, edge = worley(S, (22, 2), rng, (wx, wx * 0.2))
        crack = np.clip(1 - edge / 0.025, 0, 1) * np.clip((fbm(S, (6, 3), rng, 3) - 0.35) * 3, 0, 1)
        rot = np.clip((fbm(S, (3, 3), rng, 4) - 0.66) * 5, 0, 1)
        h = 0.6 + 0.3 * grain - 0.6 * crack
        alb = srgb_lin("#5e584f") + (srgb_lin("#8f887c") - srgb_lin("#5e584f")) * grain[..., None]
        alb = alb * (1 - 0.75 * crack)[..., None]
        alb = alb + (srgb_lin("#4a3a2c") - alb) * (0.7 * rot)[..., None]
        lichen = np.clip((fbm(S, (9, 9), rng, 5) - 0.68) * 6, 0, 1)
        alb = alb + (srgb_lin("#9aa27e") - alb) * (0.6 * lichen)[..., None]
        rough = 0.62 + 0.3 * crack + 0.15 * rot
        nrm = _normal_from_height(h, 9.0)
    else:   # spruce / pine: reddish brown flaking scales
        wx = (fbm(S, (3, 3), rng, 3) - 0.5) * 0.8
        f1, edge = worley(S, (26, 16), rng, (wx, wx * 0.5))
        plate = np.clip(edge / 0.12, 0, 1)
        lift = np.clip(1 - f1, 0, 1)
        h = plate * (0.5 + 0.5 * lift) * (0.8 + 0.2 * fbm(S, (60, 20), rng, 3)) + 0.12 * fbm(S, (40, 40), rng, 3)
        a, b = srgb_lin("#2f2620"), srgb_lin("#76685b")
        alb = a + (b - a) * np.clip(h, 0, 1)[..., None]
        alb *= (0.85 + 0.25 * fbm(S, (5, 5), rng, 3))[..., None]
        rough = 0.85 - 0.1 * h
        nrm = _normal_from_height(h, 6.0)
    alb = np.concatenate([np.clip(alb, 0, 1), np.ones((px, px, 1))], -1)
    return alb, nrm, np.clip(rough, 0, 1)
