"""Painted texture atlases for props, with crisp lettering.

A stroke font (capitals, lower case, figures and the usual sign punctuation) drawn as anti-aliased
distance fields, and a small painting canvas: rounded rectangles, circles, polylines, polygons and
text, painted in metric coordinates into regions of an atlas with colour, alpha, height (turned into
a normal map), roughness, metalness and emission layers. The result is an image material, which the
game export keeps as it is (procedural materials on small props bake to one flat colour).

    cv = Canvas("Prop_Ticket", 1024, 1024)
    front = cv.alloc(0.44, 1.2, ppm=600)            # a region 0.44 x 1.2 m at 600 px/m
    front.rect(0.02, 0.6, 0.42, 0.9, r=0.01, col="#ffffff")
    front.text("PAY HERE", 0.22, 1.0, 0.04, col="#101010", align="c", weight="bold")
    mat = cv.material("Prop_TicketAtlas")
    decal(p, centre, normal, up, 0.44, 1.2, mat, front)

Geometry helpers map PM faces onto regions: decal (a quad), project (planar), wrap_uv
(cylindrical, for lathes), box_regions + box_uv (the faces of a box-like body).
"""
import math

import numpy as np


# -- the stroke font --------------------------------------------------------------------------------

def _arc(cx, cy, rx, ry, a0, a1, step=10.0):
    n = max(3, int(abs(a1 - a0) / step) + 1)
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]


def _glyphs():
    """char -> (advance width, [strokes]); stroke centrelines in cap-height units (baseline 0,
    cap 1, x-height 0.72, descender -0.28). A one-point stroke is a dot."""
    A = _arc
    G = {}
    G["A"] = (0.68, [[(0, 0), (0.34, 1), (0.68, 0)], [(0.13, 0.32), (0.55, 0.32)]])
    G["B"] = (0.6, [[(0, 0), (0, 1)], [(0, 1), (0.34, 1)] + A(0.34, 0.755, 0.245, 0.245, 90, -90) + [(0, 0.51)],
                    [(0, 0.51), (0.36, 0.51)] + A(0.36, 0.255, 0.255, 0.255, 90, -90) + [(0, 0)]])
    G["C"] = (0.68, [A(0.37, 0.5, 0.37, 0.5, 42, 318)])
    G["D"] = (0.68, [[(0, 0), (0, 1), (0.22, 1)] + A(0.22, 0.5, 0.46, 0.5, 90, -90) + [(0, 0)]])
    G["E"] = (0.56, [[(0.56, 1), (0, 1), (0, 0), (0.56, 0)], [(0, 0.52), (0.46, 0.52)]])
    G["F"] = (0.54, [[(0.54, 1), (0, 1), (0, 0)], [(0, 0.52), (0.44, 0.52)]])
    G["G"] = (0.72, [A(0.37, 0.5, 0.37, 0.5, 45, 360) + [(0.74, 0.0)], [(0.42, 0.46), (0.74, 0.46)]])
    G["H"] = (0.62, [[(0, 0), (0, 1)], [(0.62, 0), (0.62, 1)], [(0, 0.52), (0.62, 0.52)]])
    G["I"] = (0.0, [[(0, 0), (0, 1)]])
    G["J"] = (0.48, [[(0.48, 1), (0.48, 0.26)] + A(0.24, 0.26, 0.24, 0.26, 0, -180) + [(0, 0.3)]])
    G["K"] = (0.6, [[(0, 0), (0, 1)], [(0.58, 1), (0, 0.36)], [(0.2, 0.58), (0.6, 0)]])
    G["L"] = (0.52, [[(0, 1), (0, 0), (0.52, 0)]])
    G["M"] = (0.78, [[(0, 0), (0, 1), (0.39, 0.22), (0.78, 1), (0.78, 0)]])
    G["N"] = (0.64, [[(0, 0), (0, 1), (0.64, 0), (0.64, 1)]])
    G["O"] = (0.78, [A(0.39, 0.5, 0.39, 0.5, 0, 360)])
    G["P"] = (0.6, [[(0, 0), (0, 1), (0.33, 1)] + A(0.33, 0.73, 0.27, 0.27, 90, -90) + [(0, 0.46)]])
    G["Q"] = (0.78, [A(0.39, 0.5, 0.39, 0.5, 0, 360), [(0.46, 0.24), (0.8, -0.06)]])
    G["R"] = (0.62, [[(0, 0), (0, 1), (0.33, 1)] + A(0.33, 0.73, 0.27, 0.27, 90, -90) + [(0, 0.46)],
                     [(0.3, 0.46), (0.62, 0)]])
    G["S"] = (0.6, [A(0.3, 0.745, 0.29, 0.255, 20, 270) + A(0.3, 0.255, 0.3, 0.255, 90, -160)])
    G["T"] = (0.62, [[(0, 1), (0.62, 1)], [(0.31, 1), (0.31, 0)]])
    G["U"] = (0.62, [[(0, 1), (0, 0.32)] + A(0.31, 0.32, 0.31, 0.32, 180, 360) + [(0.62, 1)]])
    G["V"] = (0.68, [[(0, 1), (0.34, 0), (0.68, 1)]])
    G["W"] = (0.96, [[(0, 1), (0.23, 0), (0.48, 0.78), (0.73, 0), (0.96, 1)]])
    G["X"] = (0.62, [[(0, 1), (0.62, 0)], [(0, 0), (0.62, 1)]])
    G["Y"] = (0.64, [[(0, 1), (0.32, 0.48), (0.64, 1)], [(0.32, 0.48), (0.32, 0)]])
    G["Z"] = (0.58, [[(0, 1), (0.58, 1), (0, 0), (0.58, 0)]])
    G["0"] = (0.58, [A(0.29, 0.5, 0.29, 0.5, 0, 360)])
    G["1"] = (0.36, [[(0.0, 0.78), (0.3, 1), (0.3, 0)]])
    G["2"] = (0.58, [A(0.29, 0.71, 0.28, 0.29, 165, -35) + [(0, 0), (0.58, 0)]])
    G["3"] = (0.58, [A(0.28, 0.75, 0.26, 0.25, 155, -90) + A(0.28, 0.26, 0.3, 0.26, 90, -155)])
    G["4"] = (0.62, [[(0.46, 0), (0.46, 1), (0, 0.3), (0.62, 0.3)]])
    G["5"] = (0.58, [[(0.54, 1), (0.08, 1), (0.04, 0.56)] + A(0.29, 0.33, 0.29, 0.33, 135, -150)])
    G["6"] = (0.58, [A(0.29, 0.31, 0.29, 0.31, 0, 360), [(0.02, 0.36), (0.12, 0.7), (0.4, 1.0)]])
    G["7"] = (0.56, [[(0, 1), (0.56, 1), (0.18, 0)]])
    G["8"] = (0.58, [A(0.29, 0.765, 0.24, 0.235, 0, 360), A(0.29, 0.265, 0.29, 0.265, 0, 360)])
    G["9"] = (0.58, [A(0.29, 0.69, 0.29, 0.31, 0, 360), [(0.56, 0.64), (0.46, 0.3), (0.18, 0.0)]])
    G["a"] = (0.54, [A(0.27, 0.36, 0.27, 0.36, 0, 360), [(0.54, 0.72), (0.54, 0)]])
    G["b"] = (0.54, [[(0, 1), (0, 0)], A(0.27, 0.36, 0.27, 0.36, 0, 360)])
    G["c"] = (0.52, [A(0.28, 0.36, 0.28, 0.36, 40, 320)])
    G["d"] = (0.54, [[(0.54, 1), (0.54, 0)], A(0.27, 0.36, 0.27, 0.36, 0, 360)])
    G["e"] = (0.54, [[(0.01, 0.37), (0.54, 0.37)] + A(0.27, 0.36, 0.27, 0.36, 0, 310)])
    G["f"] = (0.38, [[(0.4, 0.99), (0.3, 1.0)] + A(0.3, 0.86, 0.15, 0.14, 90, 180) + [(0.15, 0)], [(0, 0.7), (0.36, 0.7)]])
    G["g"] = (0.54, [A(0.27, 0.38, 0.27, 0.34, 0, 360), [(0.54, 0.72), (0.54, -0.06)] + A(0.27, -0.06, 0.27, 0.2, 0, -165)])
    G["h"] = (0.52, [[(0, 1), (0, 0)], [(0, 0.44)] + A(0.26, 0.45, 0.26, 0.27, 180, 0) + [(0.52, 0)]])
    G["i"] = (0.0, [[(0, 0.72), (0, 0)], [(0, 0.96)]])
    G["j"] = (0.16, [[(0.16, 0.72), (0.16, -0.12)] + A(0.0, -0.12, 0.16, 0.16, 0, -110), [(0.16, 0.96)]])
    G["k"] = (0.48, [[(0, 1), (0, 0)], [(0.46, 0.72), (0, 0.28)], [(0.16, 0.42), (0.48, 0)]])
    G["l"] = (0.0, [[(0, 1), (0, 0)]])
    G["m"] = (0.8, [[(0, 0), (0, 0.72)], [(0, 0.44)] + A(0.2, 0.47, 0.2, 0.25, 180, 0) + [(0.4, 0)],
                    [(0.4, 0.44)] + A(0.6, 0.47, 0.2, 0.25, 180, 0) + [(0.8, 0)]])
    G["n"] = (0.52, [[(0, 0), (0, 0.72)], [(0, 0.44)] + A(0.26, 0.45, 0.26, 0.27, 180, 0) + [(0.52, 0)]])
    G["o"] = (0.56, [A(0.28, 0.36, 0.28, 0.36, 0, 360)])
    G["p"] = (0.54, [[(0, 0.72), (0, -0.28)], A(0.27, 0.36, 0.27, 0.36, 0, 360)])
    G["q"] = (0.54, [[(0.54, 0.72), (0.54, -0.28)], A(0.27, 0.36, 0.27, 0.36, 0, 360)])
    G["r"] = (0.34, [[(0, 0), (0, 0.72)], [(0, 0.42)] + A(0.3, 0.42, 0.3, 0.3, 180, 80)])
    G["s"] = (0.48, [A(0.24, 0.535, 0.23, 0.185, 25, 270) + A(0.24, 0.185, 0.24, 0.185, 90, -155)])
    G["t"] = (0.36, [[(0.13, 0.98), (0.13, 0.14)] + A(0.27, 0.14, 0.14, 0.14, 180, 290), [(0, 0.72), (0.34, 0.72)]])
    G["u"] = (0.52, [[(0, 0.72), (0, 0.28)] + A(0.26, 0.28, 0.26, 0.28, 180, 360), [(0.52, 0.72), (0.52, 0)]])
    G["v"] = (0.54, [[(0, 0.72), (0.27, 0), (0.54, 0.72)]])
    G["w"] = (0.78, [[(0, 0.72), (0.19, 0), (0.39, 0.6), (0.59, 0), (0.78, 0.72)]])
    G["x"] = (0.5, [[(0, 0.72), (0.5, 0)], [(0, 0), (0.5, 0.72)]])
    G["y"] = (0.54, [[(0, 0.72), (0.27, 0)], [(0.54, 0.72), (0.18, -0.28)]])
    G["z"] = (0.5, [[(0, 0.72), (0.5, 0.72), (0, 0), (0.5, 0)]])
    G["."] = (0.0, [[(0, 0)]])
    G[","] = (0.06, [[(0.06, 0.02), (-0.02, -0.16)]])
    G[":"] = (0.0, [[(0, 0)], [(0, 0.56)]])
    G["-"] = (0.32, [[(0, 0.42), (0.32, 0.42)]])
    G["/"] = (0.42, [[(0, -0.06), (0.42, 1.06)]])
    G["·"] = (0.0, [[(0, 0.45)]])
    G["'"] = (0.0, [[(0, 1), (0, 0.72)]])
    G["("] = (0.2, [A(0.36, 0.45, 0.36, 0.62, 118, 242)])
    G[")"] = (0.2, [A(-0.16, 0.45, 0.36, 0.62, 62, -62)])
    G["+"] = (0.52, [[(0, 0.42), (0.52, 0.42)], [(0.26, 0.16), (0.26, 0.68)]])
    G["!"] = (0.0, [[(0, 1), (0, 0.3)], [(0, 0)]])
    G["?"] = (0.48, [A(0.24, 0.74, 0.24, 0.26, 165, -60) + [(0.24, 0.42), (0.24, 0.3)], [(0.24, 0)]])
    G["&"] = (0.66, [[(0.66, 0), (0.12, 0.62)] + A(0.27, 0.8, 0.17, 0.2, 220, -40) + [(0.14, 0.42)] + A(0.27, 0.25, 0.27, 0.25, 140, 320) + [(0.6, 0.38)]])
    G["%"] = (0.72, [A(0.15, 0.78, 0.15, 0.2, 0, 360), A(0.57, 0.2, 0.15, 0.2, 0, 360), [(0.62, 1), (0.1, 0)]])
    G["£"] = (0.56, [A(0.38, 0.76, 0.2, 0.24, 15, 180) + [(0.18, 0.76), (0.18, 0.2), (0.04, 0), (0.56, 0)], [(0, 0.48), (0.42, 0.48)]])
    G[">"] = (0.42, [[(0, 0.8), (0.42, 0.42), (0, 0.04)]])
    G["<"] = (0.42, [[(0.42, 0.8), (0, 0.42), (0.42, 0.04)]])
    G["°"] = (0.24, [A(0.12, 0.86, 0.12, 0.14, 0, 360)])
    G["→"] = (0.8, [[(0, 0.42), (0.78, 0.42)], [(0.46, 0.78), (0.8, 0.42), (0.46, 0.06)]])
    G["←"] = (0.8, [[(0.02, 0.42), (0.8, 0.42)], [(0.34, 0.78), (0, 0.42), (0.34, 0.06)]])
    G["↑"] = (0.72, [[(0.36, 0), (0.36, 0.98)], [(0, 0.66), (0.36, 1.0), (0.72, 0.66)]])
    G["↗"] = (0.72, [[(0.02, 0.06), (0.7, 0.92)], [(0.2, 0.96), (0.72, 0.96), (0.72, 0.44)]])
    G["↓"] = (0.72, [[(0.36, 1.0), (0.36, 0.02)], [(0, 0.34), (0.36, 0.0), (0.72, 0.34)]])
    return G


_GLYPHS = _glyphs()
WEIGHTS = {"light": 0.085, "regular": 0.115, "medium": 0.14, "bold": 0.17, "black": 0.21}


def _layout(s, h, weight="regular", cond=1.0, track=0.0):
    """Segments ([(x0, y0, x1, y1)], stroke width) and advance of `s` at cap height h (metres),
    origin at the left end of the baseline."""
    sw = WEIGHTS.get(weight, 0.115) * h if isinstance(weight, str) else weight * h
    k = h - sw
    gap = h * (0.17 + track) + sw
    segs, x = [], 0.0
    for ch in s:
        if ch == " ":
            x += h * 0.36 * cond + gap * 0.3
            continue
        g = _GLYPHS.get(ch) or _GLYPHS.get(ch.upper())
        if g is None:
            x += h * 0.5
            continue
        w, strokes = g
        for st in strokes:
            P = [(x + sw / 2 + px * k * cond, sw / 2 + py * k) for px, py in st]
            if len(P) == 1:
                segs.append((P[0][0], P[0][1], P[0][0], P[0][1]))
            for a, b in zip(P, P[1:]):
                segs.append((a[0], a[1], b[0], b[1]))
        x += w * k * cond + gap
    return segs, sw, max(0.0, x - gap + sw)


def text_width(s, h, weight="regular", cond=1.0, track=0.0):
    return _layout(s, h, weight, cond, track)[2]


# -- painting ---------------------------------------------------------------------------------------

def lin(c):
    """sRGB hex / 0-1 tuple to a linear rgb array."""
    if isinstance(c, str):
        h = c.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = np.asarray(c, float)[:3]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def noise(shape, cells, seed=0, octaves=4, gain=0.5, wrap=True):
    """Value noise in [0, 1] on an (h, w) grid (tileable if wrap): `cells` (cy, cx) lattice cells
    across the first octave, doubled each octave."""
    rng = np.random.default_rng(seed)
    h, w = shape
    cy, cx = (cells, cells) if np.isscalar(cells) else cells
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        ny, nx = max(1, int(round(cy * 2 ** o))), max(1, int(round(cx * 2 ** o)))
        g = rng.random((ny + 1, nx + 1))
        if wrap:
            g[-1, :], g[:, -1] = g[0, :], g[:, 0]
        ys = (np.arange(h) + 0.5) * ny / h
        xs = (np.arange(w) + 0.5) * nx / w
        y0 = np.minimum(np.floor(ys).astype(int), ny - 1)
        x0 = np.minimum(np.floor(xs).astype(int), nx - 1)
        fy = ys - y0
        fx = xs - x0
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
        b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
        out += amp * (a * (1 - fy[:, None]) + b * fy[:, None])
        tot += amp
        amp *= gain
    return out / tot


class Canvas:
    """An atlas image being painted: colour (linear), alpha, height (metres), roughness,
    metalness and emission. Row 0 is the bottom (v = 0)."""

    def __init__(self, name, W=1024, H=1024, col="#808080", rough=0.5, metal=0.0, alpha=1.0):
        self.name, self.W, self.H = name, W, H
        self.col = np.broadcast_to(lin(col), (H, W, 3)).copy()
        self.a = np.full((H, W), float(alpha))
        self.ht = np.zeros((H, W))
        self.rough = np.full((H, W), float(rough))
        self.metal = np.full((H, W), float(metal))
        self.emit = np.zeros((H, W, 3))
        self.mpp = np.full((H, W), 0.001)          # metres per pixel (for the normal map)
        self._sky = np.zeros(W, int)                # packer: skyline (used height per column)

    def region(self, px0, py0, pw, ph, w, h):
        return Region(self, px0, py0, pw, ph, w, h)

    def alloc(self, w, h, ppm=500.0, pad=3):
        """A region for a w x h metre surface at `ppm` pixels per metre (skyline-packed: the
        lowest place it fits)."""
        pw, ph = max(2, int(round(w * ppm))), max(2, int(round(h * ppm)))
        best = None
        for x in range(0, self.W - pw - pad + 1, 4):
            y = int(self._sky[x:x + pw + pad].max())
            if best is None or y < best[1]:
                best = (x, y)
        if best is None or best[1] + ph + pad > self.H:
            raise ValueError(f"{self.name}: atlas full placing {w}x{h} m at {ppm} px/m")
        x, y = best
        self._sky[x:x + pw + pad] = y + ph + pad
        return Region(self, x, y, pw, ph, w, h)

    def image(self, name, rgba, colorspace="sRGB"):
        import bpy
        from ..core import materials as MAT
        if colorspace == "sRGB":
            return MAT.image_from_array(name, rgba)
        h, w = rgba.shape[:2]
        img = bpy.data.images.new(name, w, h, alpha=True)
        img.colorspace_settings.name = "Non-Color"
        img.pixels.foreach_set(rgba.astype(np.float32).ravel())
        img.pack()
        return img

    def normal_map(self):
        h = self.ht
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * self.mpp)
        gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * self.mpp)
        n = np.stack([-gx, -gy, np.ones_like(h)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return np.concatenate([n * 0.5 + 0.5, np.ones_like(h)[..., None]], -1)

    def material(self, name, rough=None, metal=None, normal=1.0, alpha=False, emission=0.0,
                 spec=0.5, repeat=False, blend=False):
        """An image material from the canvas. rough/metal None: use the painted layers (as maps
        when they vary), numbers: constants. normal: strength of the height map's normal (0 = none).
        alpha: cut-out by the alpha layer (blend: soft alpha instead). emission: strength of the
        emission layer. Several materials of one canvas share its images."""
        import bpy
        from ..core.nodes import material
        if bpy.data.materials.get(name):
            return bpy.data.materials[name]
        imgs = getattr(self, "_imgs", None)
        if imgs is None:
            rgba = np.concatenate([np.clip(self.col, 0, 1), self.a[..., None]], -1)
            imgs = {"col": self.image(self.name + "_col", rgba)}
            if np.ptp(self.ht) > 1e-6:
                imgs["nrm"] = self.image(self.name + "_nrm", self.normal_map(), "Non-Color")
            if np.ptp(self.rough) > 1e-3 or np.ptp(self.metal) > 1e-3:
                rm = np.stack([np.zeros_like(self.rough), np.clip(self.rough, 0, 1), np.clip(self.metal, 0, 1), np.ones_like(self.rough)], -1)
                imgs["rm"] = self.image(self.name + "_rm", rm, "Non-Color")
            if self.emit.max() > 1e-4:
                imgs["emit"] = self.image(self.name + "_emit", np.concatenate([np.clip(self.emit, 0, 1), np.ones_like(self.a)[..., None]], -1))
            self._imgs = imgs
        m, nb = material(name)
        ext = "REPEAT" if repeat else "EXTEND"
        uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
        tex = nb.node("ShaderNodeTexImage", image=imgs["col"], extension=ext)
        nb.connect(uv, tex.inputs["Vector"])
        kw = {"Base_Color": tex.outputs["Color"], "Specular_IOR_Level": spec}
        if "rm" in imgs and (rough is None or metal is None):
            t = nb.node("ShaderNodeTexImage", image=imgs["rm"], extension=ext)
            nb.connect(uv, t.inputs["Vector"])
            sep = nb.node("ShaderNodeSeparateColor")
            nb.connect(t.outputs["Color"], sep.inputs["Color"])
            kw["Roughness"] = sep.outputs["Green"] if rough is None else rough
            kw["Metallic"] = sep.outputs["Blue"] if metal is None else metal
        else:
            kw["Roughness"] = float(self.rough.mean()) if rough is None else rough
            kw["Metallic"] = float(self.metal.mean()) if metal is None else metal
        if "nrm" in imgs and normal:
            t = nb.node("ShaderNodeTexImage", image=imgs["nrm"], extension=ext)
            nb.connect(uv, t.inputs["Vector"])
            nm = nb.node("ShaderNodeNormalMap", uv_map="UVMap")
            nm.inputs["Strength"].default_value = normal
            nb.connect(t.outputs["Color"], nm.inputs["Color"])
            kw["Normal"] = nm.outputs[0]
        if "emit" in imgs and emission:
            t = nb.node("ShaderNodeTexImage", image=imgs["emit"], extension=ext)
            nb.connect(uv, t.inputs["Vector"])
            kw["Emission_Color"] = t.outputs["Color"]
            kw["Emission_Strength"] = emission
        if alpha:
            kw["Alpha"] = tex.outputs["Alpha"] if blend else nb.math("ROUND", tex.outputs["Alpha"])
        nb.output(nb.principled(**kw))
        return m


    def tint_material(self, name, base_hex, rough=0.5, normal=1.0, spec=0.5, repeat=False):
        """A recolourable TINT_<slot> material from a canvas painted in greys (white = the slot's
        colour, darker = dirt, seams; mottle and fades either way): a greyscale detail texture
        times a colour factor, with the gain in the material's extras (tint_gain), as the game
        bake makes TINT_ materials, so a placement's colour (e.g. the club's) replaces the factor.
        Name it TINT_<slot>_<suffix> so the slot matches it."""
        import bpy
        from ..core.nodes import material
        if bpy.data.materials.get(name):
            return bpy.data.materials[name]
        lum = np.clip(self.col, 0, None) @ np.array([0.2126, 0.7152, 0.0722])
        gain = float(max(lum.max(), 1e-3))
        det = np.clip(lum / gain, 0, 1)
        rgba = np.stack([det, det, det, np.ones_like(det)], -1)
        img = self.image(name + "_detail", rgba)
        m, nb = material(name)
        ext = "REPEAT" if repeat else "EXTEND"
        uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
        tex = nb.node("ShaderNodeTexImage", image=img, extension=ext)
        nb.connect(uv, tex.inputs["Vector"])
        mix = nb.node("ShaderNodeMix", data_type="RGBA")
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0
        nb.connect(tex.outputs["Color"], mix.inputs[6])
        base = np.clip(lin(base_hex) * gain, 0, 1)
        mix.inputs[7].default_value = (*base.tolist(), 1.0)
        kw = {"Base_Color": mix.outputs[2], "Roughness": rough, "Specular_IOR_Level": spec}
        if np.ptp(self.ht) > 1e-6 and normal:
            nimg = self.image(name + "_nrm", self.normal_map(), "Non-Color")
            t = nb.node("ShaderNodeTexImage", image=nimg, extension=ext)
            nb.connect(uv, t.inputs["Vector"])
            nm = nb.node("ShaderNodeNormalMap", uv_map="UVMap")
            nm.inputs["Strength"].default_value = normal
            nb.connect(t.outputs["Color"], nm.inputs["Color"])
            kw["Normal"] = nm.outputs[0]
        nb.output(nb.principled(**kw))
        m["tint_gain"] = round(gain, 4)
        m["base_hex"] = base_hex
        m["game_material"] = True
        return m


class Region:
    """A rectangle of a canvas addressed in metres: x right, y up from its bottom-left corner."""

    def __init__(self, cv, px0, py0, pw, ph, w, h):
        self.cv, self.px0, self.py0, self.pw, self.ph, self.w, self.h = cv, px0, py0, pw, ph, w, h
        self.sx, self.sy = pw / w, ph / h               # px per metre
        cv.mpp[py0:py0 + ph, px0:px0 + pw] = 1.0 / min(self.sx, self.sy)

    def uv(self, x, y):
        return ((self.px0 + x * self.sx) / self.cv.W, (self.py0 + y * self.sy) / self.cv.H)

    def grid(self, x0=None, y0=None, x1=None, y1=None):
        """(rows slice, cols slice, X, Y) over the pixels of [x0, x1] x [y0, y1] (whole region by
        default), X and Y the pixel centres in metres."""
        x0 = 0.0 if x0 is None else x0
        y0 = 0.0 if y0 is None else y0
        x1 = self.w if x1 is None else x1
        y1 = self.h if y1 is None else y1
        c0 = max(0, int(math.floor(x0 * self.sx)) - 2)
        c1 = min(self.pw, int(math.ceil(x1 * self.sx)) + 2)
        r0 = max(0, int(math.floor(y0 * self.sy)) - 2)
        r1 = min(self.ph, int(math.ceil(y1 * self.sy)) + 2)
        if c1 <= c0 or r1 <= r0:
            return None
        X = (np.arange(c0, c1) + 0.5) / self.sx
        Y = (np.arange(r0, r1) + 0.5) / self.sy
        X, Y = np.meshgrid(X, Y)
        return (slice(self.py0 + r0, self.py0 + r1), slice(self.px0 + c0, self.px0 + c1), X, Y)

    def cover(self, d):
        """Coverage of a signed distance field (metres, negative inside), anti-aliased."""
        return np.clip(0.5 - d * min(self.sx, self.sy), 0.0, 1.0)

    def apply(self, rs, cs, c, col=None, a=None, h=None, dh=None, rough=None, metal=None, emit=None,
              mul=None, k=1.0):
        """Blend paint into the canvas with coverage c (array over rows rs, cols cs)."""
        cv = self.cv
        c = np.clip(c * k, 0, 1)
        if col is not None:
            cc = lin(col) if isinstance(col, str) or np.ndim(col) == 1 else np.asarray(col)
            cv.col[rs, cs] = cv.col[rs, cs] * (1 - c[..., None]) + cc * c[..., None]
        if mul is not None:
            mm = lin(mul) if isinstance(mul, str) else np.asarray(mul, float)
            cv.col[rs, cs] *= 1 - c[..., None] * (1 - mm)
        if a is not None:
            cv.a[rs, cs] = cv.a[rs, cs] * (1 - c) + a * c
        if h is not None:
            cv.ht[rs, cs] = cv.ht[rs, cs] * (1 - c) + h * c
        if dh is not None:
            cv.ht[rs, cs] += dh * c
        if rough is not None:
            cv.rough[rs, cs] = cv.rough[rs, cs] * (1 - c) + rough * c
        if metal is not None:
            cv.metal[rs, cs] = cv.metal[rs, cs] * (1 - c) + metal * c
        if emit is not None:
            ee = lin(emit) if isinstance(emit, str) else np.asarray(emit)
            cv.emit[rs, cs] = cv.emit[rs, cs] * (1 - c[..., None]) + ee * c[..., None]

    def field(self, fn, x0=None, y0=None, x1=None, y1=None, **paint):
        """Paint with coverage fn(X, Y) (0-1) over a box (whole region by default)."""
        g = self.grid(x0, y0, x1, y1)
        if g is None:
            return
        rs, cs, X, Y = g
        self.apply(rs, cs, fn(X, Y), **paint)

    def _box_sd(self, X, Y, x0, y0, x1, y1, r):
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        qx, qy = np.abs(X - cx) - hx + r, np.abs(Y - cy) - hy + r
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r

    def _paint_sd(self, rs, cs, d, bevel, paint):
        if bevel > 0 and (paint.get("dh") is not None or paint.get("h") is not None):
            hp = {k: paint.pop(k) for k in ("dh", "h") if k in paint}
            self.apply(rs, cs, self.cover(d), **paint)
            self.apply(rs, cs, np.clip(-d / bevel, 0, 1), **hp)
            return
        self.apply(rs, cs, self.cover(d), **paint)

    def rect(self, x0, y0, x1, y1, r=0.0, bevel=0.0, **paint):
        """Rounded rectangle. bevel > 0 with dh/h: the height ramps in over `bevel` metres."""
        g = self.grid(x0, y0, x1, y1)
        if g is None:
            return
        rs, cs, X, Y = g
        self._paint_sd(rs, cs, self._box_sd(X, Y, x0, y0, x1, y1, r), bevel, paint)

    def frame(self, x0, y0, x1, y1, t, r=0.0, **paint):
        """A rectangular outline t thick (inside the box)."""
        g = self.grid(x0, y0, x1, y1)
        if g is None:
            return
        rs, cs, X, Y = g
        d = self._box_sd(X, Y, x0, y0, x1, y1, r)
        self.apply(rs, cs, self.cover(np.abs(d + t / 2) - t / 2), **paint)

    def circle(self, cx, cy, rad, bevel=0.0, **paint):
        g = self.grid(cx - rad, cy - rad, cx + rad, cy + rad)
        if g is None:
            return
        rs, cs, X, Y = g
        self._paint_sd(rs, cs, np.hypot(X - cx, Y - cy) - rad, bevel, paint)

    def ring(self, cx, cy, r0, r1, **paint):
        g = self.grid(cx - r1, cy - r1, cx + r1, cy + r1)
        if g is None:
            return
        rs, cs, X, Y = g
        d = np.abs(np.hypot(X - cx, Y - cy) - (r0 + r1) / 2) - (r1 - r0) / 2
        self.apply(rs, cs, self.cover(d), **paint)

    def _segs(self, segs, hw, **paint):
        S = np.asarray(segs, float).reshape(-1, 4)
        if not len(S):
            return
        g = self.grid(S[:, [0, 2]].min() - hw, S[:, [1, 3]].min() - hw, S[:, [0, 2]].max() + hw, S[:, [1, 3]].max() + hw)
        if g is None:
            return
        rs, cs, X, Y = g
        d = np.full(X.shape, 1e9)
        for x0, y0, x1, y1 in S:
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            if L2 < 1e-14:
                dd = np.hypot(X - x0, Y - y0)
            else:
                t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
                dd = np.hypot(X - x0 - t * dx, Y - y0 - t * dy)
            np.minimum(d, dd, out=d)
        self.apply(rs, cs, self.cover(d - hw), **paint)

    def line(self, pts, width, **paint):
        P = list(pts)
        segs = [(a[0], a[1], b[0], b[1]) for a, b in zip(P, P[1:])] or [(P[0][0], P[0][1], P[0][0], P[0][1])]
        self._segs(segs, width / 2, **paint)

    def poly(self, pts, **paint):
        """Filled polygon (any winding), anti-aliased edges."""
        P = np.asarray(pts, float)
        g = self.grid(P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())
        if g is None:
            return
        rs, cs, X, Y = g
        inside = np.zeros(X.shape, bool)
        d = np.full(X.shape, 1e9)
        n = len(P)
        for i in range(n):
            (x0, y0), (x1, y1) = P[i], P[(i + 1) % n]
            inside ^= ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / (y1 - y0 + 1e-12) + x0)
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy + 1e-14), 0, 1)
            np.minimum(d, np.hypot(X - x0 - t * dx, Y - y0 - t * dy), out=d)
        self.apply(rs, cs, self.cover(np.where(inside, -d, d)), **paint)

    def text(self, s, x, y, h, weight="regular", align="l", cond=1.0, track=0.0, max_w=None, **paint):
        """Text with its baseline at y; align l/c/r about x. max_w: shrink to fit."""
        segs, sw, wd = _layout(s, h, weight, cond, track)
        if max_w and wd > max_w:
            f = max_w / wd
            segs, sw, wd = _layout(s, h * f, weight, cond, track)
            y += h * (1 - f) / 2
        ox = x - (wd / 2 if align == "c" else wd if align == "r" else 0.0)
        self._segs([(a + ox, b + y, c + ox, d + y) for a, b, c, d in segs], sw / 2, **paint)
        return wd

    def fill(self, **paint):
        rs, cs, X, Y = self.grid()
        self.apply(rs, cs, np.ones(X.shape), **paint)

    def mottle(self, amount=0.06, cells=6, seed=1, rough=0.0):
        """Low-frequency tonal variation (and roughness variation) over the region."""
        rs, cs, X, Y = self.grid()
        m = max(self.w, self.h)
        n = noise(X.shape, (max(1, cells * self.h / m), max(1, cells * self.w / m)), seed, 4)
        self.cv.col[rs, cs] *= (1 + (n - 0.5) * 2 * amount)[..., None]
        if rough:
            self.cv.rough[rs, cs] = np.clip(self.cv.rough[rs, cs] + (n - 0.5) * 2 * rough, 0.02, 1)

    def grime(self, height=0.25, color="#4a4232", strength=0.6, seed=2, y_ground=0.0):
        """Soil splash and dirt rising from the ground (region y = y_ground), broken by noise."""
        rs, cs, X, Y = self.grid()
        n = noise(X.shape, (max(2, int(self.h * 20)), max(2, int(self.w * 20))), seed, 4)
        z = (Y - y_ground) / height
        self.apply(rs, cs, np.clip(1 - z + (n - 0.5) * 0.9, 0, 1) ** 1.5 * strength, col=color)

    def streaks(self, top=None, length=0.5, color="#3a3630", strength=0.25, seed=3, density=40.0):
        """Rain streaks running down from a top edge (region y = top)."""
        rs, cs, X, Y = self.grid()
        top = self.h if top is None else top
        cols = noise((1, X.shape[1]), (1, max(4, int(self.w * density))), seed, 3, wrap=False)[0]
        cols = np.clip((cols - 0.45) * 3, 0, 1)
        lens = length * (0.4 + 0.6 * noise((1, X.shape[1]), (1, max(3, int(self.w * 8))), seed + 1, 2, wrap=False)[0])
        d = (top - Y) / np.maximum(lens, 1e-3)
        self.apply(rs, cs, np.clip(1 - d, 0, 1) * (Y <= top) * cols * strength, col=color)

    def edge_wear(self, x0, y0, x1, y1, width=0.01, color="#9a9890", strength=0.7, seed=4, r=0.0):
        """Chipped paint along the edges of a box."""
        rs, cs, X, Y = self.grid(x0, y0, x1, y1)
        d = -self._box_sd(X, Y, x0, y0, x1, y1, r)
        n = noise(X.shape, (max(2, int(self.h / width / 3)), max(2, int(self.w / width / 3))), seed, 3)
        m = np.clip((1 - d / width) * 1.5 + (n - 0.6) * 3, 0, 1) * (d > -0.001) * strength
        self.apply(rs, cs, m, col=color)


# -- geometry helpers -------------------------------------------------------------------------------

def decal(p, c, n, v, w, h, mat, region=None, lift=0.002, uv=None):
    """Like PM.decal, but mapped onto an atlas region (the whole region, or uv=(x0, y0, x1, y1) of
    it in metres)."""
    c, n, v = (np.asarray(x, float) for x in (c, n, v))
    u = np.cross(v, n)
    c = c + n * lift
    P = [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
    if region is None:
        UV = [(0, 0), (1, 0), (1, 1), (0, 1)]
    else:
        x0, y0, x1, y1 = uv if uv is not None else (0, 0, region.w, region.h)
        UV = [region.uv(x0, y0), region.uv(x1, y0), region.uv(x1, y1), region.uv(x0, y1)]
    r, p.r = p.r, 0.0
    p.face(P, mat, UV)
    p.r = r


def _mats(p, mats):
    return None if mats is None else [p.m(m) for m in (mats if isinstance(mats, (list, tuple)) else [mats])]


def project(p, i0, region, origin, ex, ey, mats=None, i1=None):
    """Planar-project faces i0.. (that have no UVs yet, of `mats` if given) onto a region: local
    x = (P - origin) . ex, y = (P - origin) . ey, in metres."""
    o, ex, ey = (np.asarray(a, float) for a in (origin, ex, ey))
    mats = _mats(p, mats)
    for fi in range(i0, len(p.F) if i1 is None else i1):
        if p.uv[fi] is not None or (mats is not None and p.mi[fi] not in mats):
            continue
        P = np.array([p.V[i] for i in p.F[fi]]) - o
        p.uv[fi] = [region.uv(min(region.w, max(0.0, float(q @ ex))), min(region.h, max(0.0, float(q @ ey)))) for q in P]


def _newell(P):
    n = np.zeros(3)
    for a, b in zip(P, np.roll(P, -1, 0)):
        n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


def wrap_uv(p, i0, region, z0=0.0, centre=(0.0, 0.0), mats=None, i1=None, flat=None, ys=1.0):
    """Cylindrical projection of faces i0.. onto a region: one turn round the vertical axis through
    `centre` spans the region's width (anticlockwise from -x, so +y, the front, is at 3/4 of the
    width; the seam is on -x: keep a lathe vertex there, i.e. an even number of sides),
    y = (z - z0) * ys. Faces facing up or down (|nz| > 0.75) map flat: into region `flat` (centred
    on the axis) if given, else onto a strip at their height."""
    mats = _mats(p, mats)
    cx, cy = centre
    for fi in range(i0, len(p.F) if i1 is None else i1):
        if p.uv[fi] is not None or (mats is not None and p.mi[fi] not in mats):
            continue
        P = np.array([p.V[i] for i in p.F[fi]])
        nrm = _newell(P)
        if abs(nrm[2]) > 0.75:
            if flat is not None:
                p.uv[fi] = [flat.uv(flat.w / 2 + q[0] - cx, flat.h / 2 + q[1] - cy) for q in P]
            else:
                zm = min(region.h, max(0.0, (P[:, 2].mean() - z0) * ys))
                p.uv[fi] = [region.uv(0.5 * region.w + 0.4 * region.w * math.atan2(q[1] - cy, q[0] - cx) / math.pi, zm) for q in P]
            continue
        ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
        ang = np.where(ang - ang[0] > math.pi, ang - 2 * math.pi, np.where(ang - ang[0] < -math.pi, ang + 2 * math.pi, ang))
        xs = (ang + math.pi) / (2 * math.pi) * region.w
        base = math.floor(xs.mean() / region.w) * region.w
        p.uv[fi] = [region.uv(min(region.w, max(0.0, x - base)), min(region.h, max(0.0, (q[2] - z0) * ys))) for x, q in zip(xs, P)]


def box_regions(cv, lo, hi, ppm, faces=("+y", "-y", "+x", "-x", "+z")):
    """Atlas regions sized to the faces of the box lo..hi (keys +x -x +y -y +z -z)."""
    dx, dy, dz = (h - l for l, h in zip(lo, hi))
    dims = {"+y": (dx, dz), "-y": (dx, dz), "+x": (dy, dz), "-x": (dy, dz), "+z": (dx, dy), "-z": (dx, dy)}
    return {f: cv.alloc(*dims[f], ppm) for f in faces}


def box_uv(p, i0, regs, lo, hi, mats=None, i1=None):
    """Map faces i0.. onto the regions of a box (box_regions) by their dominant axis, each read the
    right way round from outside: on +y (the front) x runs from the viewer's left, i.e. hi.x - x;
    y is z - lo.z on the sides; on the top x - lo.x, y - lo.y."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ctr = (lo + hi) / 2
    mats = _mats(p, mats)
    for fi in range(i0, len(p.F) if i1 is None else i1):
        if p.uv[fi] is not None or (mats is not None and p.mi[fi] not in mats):
            continue
        P = np.array([p.V[i] for i in p.F[fi]])
        ax = int(np.argmax(np.abs(_newell(P))))
        key = ("+" if P.mean(0)[ax] >= ctr[ax] else "-") + "xyz"[ax]
        reg = regs.get(key)
        if reg is None:
            continue
        if key == "+y":
            L = [(hi[0] - q[0], q[2] - lo[2]) for q in P]
        elif key == "-y":
            L = [(q[0] - lo[0], q[2] - lo[2]) for q in P]
        elif key == "+x":
            L = [(q[1] - lo[1], q[2] - lo[2]) for q in P]
        elif key == "-x":
            L = [(hi[1] - q[1], q[2] - lo[2]) for q in P]
        elif key == "+z":
            L = [(q[0] - lo[0], q[1] - lo[1]) for q in P]
        else:
            L = [(q[0] - lo[0], hi[1] - q[1]) for q in P]
        p.uv[fi] = [reg.uv(min(reg.w, max(0.0, x)), min(reg.h, max(0.0, y))) for x, y in L]


def weather(reg, z0=0.0, seed=1, mottle=0.05, grime_h=0.25, grime="#4a4232", grime_k=0.55, streak_len=0.4,
            streak="#3a3630", streak_k=0.18, top=None):
    """The usual outdoor wear on a vertical face region whose bottom is at height z0: tonal mottle,
    soil splash rising from the ground, rain streaks from the top edge."""
    reg.mottle(mottle, 5, seed, rough=0.05)
    if grime_k and z0 < grime_h:
        reg.grime(grime_h, grime, grime_k, seed + 1, y_ground=-z0)
    if streak_k:
        reg.streaks(top, streak_len, streak, streak_k, seed + 2)


# -- signs ------------------------------------------------------------------------------------------

def label(name, lines, bg, fg, px=(512, 256), weight="bold", margin=0.08):
    """Drop-in for props.label in the stroke font: lines of text in fg centred on bg (bg None: the
    letters alone on a transparent card)."""
    import bpy
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    W, H = px
    cv = Canvas(name, W, H, col=bg or fg, alpha=0.0 if bg is None else 1.0, rough=0.4)
    r = cv.region(0, 0, W, H, W / H, 1.0)
    n = len(lines)
    h = min((1 - 2 * margin) / (n + 0.45 * (n - 1)), 0.5)
    y = 0.5 + (n * h + (n - 1) * 0.45 * h) / 2 - h
    for t in lines:
        r.text(t, r.w / 2, y, h, weight=weight, align="c", max_w=r.w * (1 - 2 * margin), col=fg, a=1.0)
        y -= h * 1.45
    return cv.material(name, rough=0.4, alpha=bg is None, normal=0)
