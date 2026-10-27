"""Painted textures for the road vehicles (kit/assets/vehicles.py): a small anti-aliased vector
canvas in metric coordinates, a stroke font (round-ended strokes, close to the UK number plate
face) and image materials with packed roughness / metallic maps.

Canvas coordinates are metres on the surface the image is mapped to, y up; row 0 of the stored
image is the bottom (Blender's convention).
"""
import math

import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material


# -- geometry helpers -----------------------------------------------------------------------------

def arc(cx, cy, rx, ry, a0, a1, n=None):
    """Points on an elliptical arc from angle a0 to a1 (degrees, either direction)."""
    n = n or max(4, int(abs(a1 - a0) / 9))
    t = np.radians(np.linspace(a0, a1, n + 1))
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]


def rrect(x0, y0, x1, y1, r, n=6):
    """Rounded rectangle outline (CCW)."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        pts += arc(cx, cy, r, r, a, a + 90, n)
    return pts


def round_poly(pts, r, n=5):
    """Round every corner of a polygon by radius r (clamped to a third of each edge)."""
    P = np.asarray(pts, float)
    out = []
    k = len(P)
    for i in range(k):
        a, b, c = P[i - 1], P[i], P[(i + 1) % k]
        u, v = a - b, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-9 or lv < 1e-9:
            continue
        u, v = u / lu, v / lv
        ang = math.acos(float(np.clip(np.dot(u, v), -1, 1)))
        if ang > math.pi - 1e-3 or ang < 1e-3:
            out.append(tuple(b))
            continue
        d = min(r / math.tan(ang / 2), lu / 3, lv / 3)
        p0, p1 = b + u * d, b + v * d
        for t in np.linspace(0, 1, n):
            q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t * t * p1
            out.append(tuple(q))
    return out


def mirror_x(pts):
    return [(-x, y) for x, y in pts][::-1]


def _pip(X, Y, P):
    inside = np.zeros(X.shape, bool)
    n = len(P)
    for i in range(n):
        xi, yi = P[i]
        xj, yj = P[i - 1]
        if yi == yj:
            continue
        c = ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi) + xi)
        inside ^= c
    return inside


def _seg_dist(X, Y, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-18:
        return np.hypot(X - ax, Y - ay)
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1)
    return np.hypot(X - (ax + t * dx), Y - (ay + t * dy))


# -- stroke font ----------------------------------------------------------------------------------
# glyphs in a unit box (stroke centre lines; x 0..1 is scaled to the character width)

def _glyphs():
    A = arc
    g = {
        "A": [[(0, 0), (0.5, 1), (1, 0)], [(0.2, 0.36), (0.8, 0.36)]],
        "B": [[(0, 0), (0, 1), (0.58, 1)] + A(0.58, 0.76, 0.36, 0.24, 90, -90) + [(0, 0.52)],
              [(0.6, 0.52)] + A(0.6, 0.26, 0.4, 0.26, 90, -90) + [(0, 0)]],
        "C": [A(0.5, 0.5, 0.5, 0.5, 42, 318, 30)],
        "D": [[(0.42, 0), (0, 0), (0, 1), (0.42, 1)] + A(0.42, 0.5, 0.58, 0.5, 90, -90, 18)],
        "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, 0.52), (0.78, 0.52)]],
        "F": [[(1, 1), (0, 1), (0, 0)], [(0, 0.52), (0.78, 0.52)]],
        "G": [A(0.5, 0.5, 0.5, 0.5, 45, 350, 30) + [(1, 0.44), (0.56, 0.44)]],
        "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.52), (1, 0.52)]],
        "I": [[(0.5, 0), (0.5, 1)]],
        "J": [[(1, 1), (1, 0.32)] + A(0.5, 0.32, 0.5, 0.32, 0, -180, 14)],
        "K": [[(0, 0), (0, 1)], [(1, 1), (0, 0.38)], [(0.32, 0.6), (1, 0)]],
        "L": [[(0, 1), (0, 0), (1, 0)]],
        "M": [[(0, 0), (0, 1), (0.5, 0.32), (1, 1), (1, 0)]],
        "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
        "O": [A(0.5, 0.5, 0.5, 0.5, 0, 360, 32)],
        "P": [[(0, 0), (0, 1), (0.6, 1)] + A(0.6, 0.74, 0.4, 0.26, 90, -90) + [(0, 0.48)]],
        "Q": [A(0.5, 0.5, 0.5, 0.5, 0, 360, 32), [(0.6, 0.26), (1.02, -0.06)]],
        "R": [[(0, 0), (0, 1), (0.6, 1)] + A(0.6, 0.74, 0.4, 0.26, 90, -90) + [(0, 0.48)], [(0.48, 0.48), (1, 0)]],
        "S": [A(0.5, 0.75, 0.48, 0.25, 15, 270, 16) + A(0.5, 0.25, 0.5, 0.25, 90, -165, 18)],
        "T": [[(0, 1), (1, 1)], [(0.5, 1), (0.5, 0)]],
        "U": [[(0, 1), (0, 0.34)] + A(0.5, 0.34, 0.5, 0.34, 180, 360, 14) + [(1, 1)]],
        "V": [[(0, 1), (0.5, 0), (1, 1)]],
        "W": [[(0, 1), (0.2, 0), (0.5, 0.72), (0.8, 0), (1, 1)]],
        "X": [[(0, 1), (1, 0)], [(0, 0), (1, 1)]],
        "Y": [[(0, 1), (0.5, 0.48), (1, 1)], [(0.5, 0.48), (0.5, 0)]],
        "Z": [[(0, 1), (1, 1), (0, 0), (1, 0)]],
        "0": [A(0.5, 0.5, 0.5, 0.5, 0, 360, 32)],
        "1": [[(0.12, 0.78), (0.56, 1), (0.56, 0)]],
        "2": [A(0.5, 0.72, 0.48, 0.28, 158, -32, 16) + [(0, 0), (1, 0)]],
        "3": [A(0.5, 0.76, 0.46, 0.24, 150, -90, 16) + A(0.5, 0.27, 0.5, 0.27, 90, -150, 18)],
        "4": [[(0.72, 0), (0.72, 1), (0, 0.3), (1, 0.3)]],
        "5": [[(0.92, 1), (0.1, 1), (0.04, 0.56)] + A(0.5, 0.33, 0.5, 0.33, 138, -150, 20)],
        "6": [A(0.5, 0.31, 0.5, 0.31, 0, 360, 24), A(0.78, 0.31, 0.78, 0.69, 180, 86, 12)],
        "7": [[(0, 1), (1, 1), (0.34, 0)]],
        "8": [A(0.5, 0.76, 0.42, 0.24, 0, 360, 22), A(0.5, 0.27, 0.5, 0.27, 0, 360, 24)],
        "-": [[(0.2, 0.5), (0.8, 0.5)]],
        "/": [[(0.1, -0.05), (0.9, 1.05)]],
        ".": [[(0.5, 0.0), (0.5, 0.01)]],
    }
    g["9"] = [[(1 - x, 1 - y) for x, y in s] for s in g["6"]]
    return g


GLYPHS = _glyphs()


# -- canvas ---------------------------------------------------------------------------------------

class Canvas:
    def __init__(self, W, H, x0, x1, y0, y1, ss=3, rough=0.5, metal=0.0):
        self.W, self.H = W, H
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.sx = (x1 - x0) / W
        self.sy = (y1 - y0) / H
        self.P = np.zeros((H, W, 3), np.float32)      # premultiplied colour
        self.A = np.zeros((H, W), np.float32)
        self.R = np.full((H, W), rough, np.float32)
        self.M = np.full((H, W), metal, np.float32)
        self.ss = ss

    def _grid(self, bb):
        xa, xb, ya, yb = bb
        c0 = max(0, int(math.floor((xa - self.x0) / self.sx)) - 1)
        c1 = min(self.W, int(math.ceil((xb - self.x0) / self.sx)) + 1)
        r0 = max(0, int(math.floor((ya - self.y0) / self.sy)) - 1)
        r1 = min(self.H, int(math.ceil((yb - self.y0) / self.sy)) + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        off = (np.arange(self.ss) + 0.5) / self.ss
        ox, oy = np.meshgrid(off, off)
        cols = np.arange(c0, c1)[None, :, None] + ox.ravel()[None, None, :]
        rows = np.arange(r0, r1)[:, None, None] + oy.ravel()[None, None, :]
        X = self.x0 + cols * self.sx
        Y = self.y0 + rows * self.sy
        X, Y = np.broadcast_arrays(X, Y)
        Xc = self.x0 + (np.arange(c0, c1)[None, :] + 0.5) * self.sx
        Yc = self.y0 + (np.arange(r0, r1)[:, None] + 0.5) * self.sy
        Xc, Yc = np.broadcast_arrays(Xc, Yc)
        return (slice(r0, r1), slice(c0, c1)), X, Y, Xc, Yc

    def fill(self, inside, bb, color, alpha=1.0, rough=None, metal=None, cut=False):
        g = self._grid(bb)
        if g is None:
            return
        sl, X, Y, Xc, Yc = g
        cov = inside(X, Y).mean(-1).astype(np.float32)
        if callable(alpha):
            cov = cov * alpha(Xc, Yc)
        else:
            cov = cov * alpha
        if cut:      # erase (make transparent)
            self.P[sl] *= (1 - cov)[..., None]
            self.A[sl] *= (1 - cov)
            return
        c = color(Xc, Yc) if callable(color) else np.broadcast_to(np.asarray(color, np.float32), Xc.shape + (3,))
        self.P[sl] = self.P[sl] * (1 - cov[..., None]) + c * cov[..., None]
        self.A[sl] = self.A[sl] * (1 - cov) + cov
        if rough is not None:
            r = rough(Xc, Yc) if callable(rough) else rough
            self.R[sl] = self.R[sl] * (1 - cov) + r * cov
        if metal is not None:
            self.M[sl] = self.M[sl] * (1 - cov) + metal * cov

    def poly(self, pts, color, **kw):
        P = np.asarray(pts, float)
        bb = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max())
        self.fill(lambda X, Y: _pip(X, Y, P), bb, color, **kw)

    def lines(self, polylines, width, color, **kw):
        segs = []
        for pl in polylines:
            pl = np.asarray(pl, float)
            if len(pl) == 1:
                segs.append((pl[0], pl[0]))
            segs += [(pl[i], pl[i + 1]) for i in range(len(pl) - 1)]
        if not segs:
            return
        allp = np.array([p for s in segs for p in s])
        h = width / 2
        bb = (allp[:, 0].min() - h, allp[:, 0].max() + h, allp[:, 1].min() - h, allp[:, 1].max() + h)

        def inside(X, Y):
            d = np.full(X.shape, 1e9)
            for a, b in segs:
                d = np.minimum(d, _seg_dist(X, Y, a, b))
            return d < h
        self.fill(inside, bb, color, **kw)

    def ellipse(self, cx, cy, rx, ry, color, **kw):
        self.fill(lambda X, Y: ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 < 1, (cx - rx, cx + rx, cy - ry, cy + ry), color, **kw)

    def ring(self, cx, cy, r0, r1, color, **kw):
        def inside(X, Y):
            d = np.hypot(X - cx, Y - cy)
            return (d < r1) & (d > r0)
        self.fill(inside, (cx - r1, cx + r1, cy - r1, cy + r1), color, **kw)

    def text(self, s, x, y, h, color, cw=0.63, gap=0.14, stroke=0.18, anchor="c", flip_y=False, **kw):
        """Text with cap height h (stroke included); x is the left / centre / right edge."""
        sw = stroke * h
        gh = h - sw                     # centre-line box height
        adv = []
        for ch in s:
            adv.append(0.5 * cw * h if ch == " " else (cw * h + gap * h))
        total = sum(adv) - gap * h
        x0 = x - total / 2 if anchor == "c" else (x - total if anchor == "r" else x)
        polys = []
        cx = x0
        for ch, a in zip(s, adv):
            if ch != " ":
                gw = cw * h - sw
                for st in GLYPHS.get(ch.upper(), []):
                    polys.append([(cx + sw / 2 + px * gw, y + sw / 2 + py * gh) for px, py in st])
            cx += a
        if flip_y:
            polys = [[(px, 2 * y + h - py) for px, py in pl] for pl in polys]
        self.lines(polys, sw, color, **kw)
        return total

    # -- output --
    def rgba(self):
        a = np.clip(self.A, 0, 1)
        col = self.P / np.maximum(a, 1e-6)[..., None]
        return np.concatenate([np.clip(col, 0, 1), a[..., None]], -1)


def noise2(H, W, scale, seed=0, octaves=3):
    """Smooth value noise in [0, 1] of shape (H, W); scale = cells across the width."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = int(scale * 2 ** o) + 2
        m = int(scale * 2 ** o * H / W) + 2
        g = rng.random((m + 1, n + 1)).astype(np.float32)
        ys = np.linspace(0, m - 1, H)
        xs = np.linspace(0, n - 1, W)
        y0 = ys.astype(int)
        x0 = xs.astype(int)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
        b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
        out += amp * (a * (1 - fy) + b * fy)
        tot += amp
        amp *= 0.5
    return out / tot


# -- materials ------------------------------------------------------------------------------------

def _raw_image(name, arr):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(arr.astype(np.float32).ravel())
    img.pack()
    return img


def image_material(name, cv, mask=True, rough_map=True, emit=None, extension="EXTEND"):
    """A material from a canvas: base colour (+ alpha cut-out), and roughness / metallic packed
    in the green / blue channels of a second image (the glTF layout)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    rgba = cv.rgba()
    if not mask:
        rgba[..., 3] = 1.0
    img = MAT.image_from_array(name + "_img", rgba)
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", image=img, extension=extension)
    nb.connect(uv, tex.inputs["Vector"])
    kw = dict(Base_Color=tex.outputs["Color"])
    if rough_map:
        rm = np.zeros((cv.H, cv.W, 4), np.float32)
        rm[..., 1] = cv.R
        rm[..., 2] = cv.M
        rm[..., 3] = 1.0
        t2 = nb.node("ShaderNodeTexImage", image=_raw_image(name + "_rm", rm), extension=extension)
        nb.connect(uv, t2.inputs["Vector"])
        sep = nb.node("ShaderNodeSeparateColor")
        nb.connect(t2.outputs["Color"], sep.inputs["Color"])
        kw.update(Roughness=sep.outputs["Green"], Metallic=sep.outputs["Blue"])
    else:
        kw.update(Roughness=float(cv.R.mean()), Metallic=float(cv.M.mean()))
    if mask:
        kw["Alpha"] = nb.math("ROUND", tex.outputs["Alpha"])
    nb.output(nb.principled(**kw))
    m["image_res"] = (cv.W, cv.H)
    return m


def alpha_at(cv, u, v):
    """Canvas alpha (max over a small footprint) at metric coordinates (arrays)."""
    c = np.clip(((np.asarray(u) - cv.x0) / cv.sx).astype(int), 0, cv.W - 1)
    r = np.clip(((np.asarray(v) - cv.y0) / cv.sy).astype(int), 0, cv.H - 1)
    return cv.A[r, c]
