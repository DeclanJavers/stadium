"""Surface detail for the grounds and construction machines (props_machines): worn paint, bare
steel, track and tyre textures painted in numpy and mapped in metres; a decal atlas per machine
(hazard chevrons, louvre grilles, warning labels and crisp lettering rasterised from Blender's
built-in font); wheels with real tread bars and wheel nuts; mud and grime laid into the vertex
colours on top of the baked occlusion (it rises from the ground and gathers on the running gear).

Image-textured materials survive the game bake unchanged (game.prepare keeps any material with an
image node), so every bit of wear seen in the viewer is in these textures, not in node noise.
"""
import math

import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material, srgb
from .foliage_paint import fbm, srgb_lin, worley


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _lin(c):
    return srgb_lin(c) if isinstance(c, str) else np.asarray(c, float)


# -- tileable surface textures --------------------------------------------------------------------

def _segments(img, rng, n, length, width, col, alpha, px, wrap=True):
    """Thin scratches: n short straight strokes blended into img (H, W, 3), wrapping."""
    H, W = img.shape[:2]
    for _ in range(n):
        x0, y0 = rng.random() * W, rng.random() * H
        a = rng.normal(0.0, 0.5) if rng.random() < 0.6 else rng.random() * math.pi
        L = length * (0.3 + rng.random())
        x1, y1 = x0 + math.cos(a) * L, y0 + math.sin(a) * L
        r = width + 2
        xa, xb = int(min(x0, x1) - r), int(max(x0, x1) + r) + 1
        ya, yb = int(min(y0, y1) - r), int(max(y0, y1) + r) + 1
        X, Y = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(ya, yb) + 0.5)
        d = np.array([x1 - x0, y1 - y0])
        t = np.clip(((X - x0) * d[0] + (Y - y0) * d[1]) / max(d @ d, 1e-9), 0, 1)
        dist = np.hypot(X - (x0 + t * d[0]), Y - (y0 + t * d[1]))
        w = width * (0.5 + rng.random())
        al = np.clip(w - dist + 0.5, 0, 1) * alpha * (0.4 + 0.6 * np.sin(t * math.pi))
        iy, ix = np.arange(ya, yb) % H, np.arange(xa, xb) % W
        sub = img[np.ix_(iy, ix)]
        img[np.ix_(iy, ix)] = sub * (1 - al[..., None]) + np.asarray(col) * al[..., None]


def paint_image(base, px=1024, seed=0, wear=1.0, dust=0.35, streak=0.35, chip="#55534e",
                dust_col="#8c8170", mottle=0.6):
    """Linear RGB (px, px, 3), tileable: a paint colour with orange-peel grain, fading in patches,
    settled dust, rain streaks (running along v, which is height on vertical faces), stone chips
    down to primer and scratches."""
    rng = np.random.default_rng(seed)
    S = (px, px)
    b = _lin(base)
    lo = fbm(S, (2, 2), rng, 5)
    md = fbm(S, (9, 9), rng, 3)
    grain = fbm(S, (160, 160), rng, 2)
    shade = 1 + mottle * (0.16 * (lo - 0.5) + 0.07 * (md - 0.5) + 0.05 * (grain - 0.5))
    img = b[None, None, :] * shade[..., None]
    # sun fade: lighter, slightly desaturated patches
    fade = _smooth(0.5, 0.8, fbm(S, (3, 3), rng, 4)) * 0.12 * mottle
    grey = img.mean(-1, keepdims=True)
    img = img * (1 - fade[..., None]) + (grey * 0.5 + img * 0.5) * 1.15 * fade[..., None]
    # dust
    dm = _smooth(0.45, 0.8, fbm(S, (5, 5), rng, 5)) * dust
    img = img * (1 - dm[..., None] * 0.55) + _lin(dust_col) * (dm[..., None] * 0.55)
    # streaks: noise stretched along v
    st = _smooth(0.55, 0.85, fbm(S, (56, 3), rng, 3)) * streak
    img = img * (1 - 0.22 * st[..., None])
    if wear > 0:
        cl = _smooth(0.55, 0.9, fbm(S, (5, 5), rng, 4)) * wear
        hf = fbm(S, (96, 96), rng, 3)
        t = 0.83 - 0.1 * cl
        chips = _smooth(t, t + 0.015, hf) * (cl > 0.02)
        rim = _smooth(t - 0.03, t - 0.015, hf) * (1 - chips) * (cl > 0.02)
        img = img * (1 - rim[..., None] * 0.3) + _lin("#c9c4b8") * rim[..., None] * 0.3
        img = img * (1 - chips[..., None]) + _lin(chip) * chips[..., None]
        _segments(img, rng, int(90 * wear), px * 0.06, 0.7, _lin("#a9a59c"), 0.35, px)
        _segments(img, rng, int(60 * wear), px * 0.03, 0.9, _lin("#2c2a27"), 0.3, px)
    return np.clip(img, 0, 1)


def image_material(name, rgb, rough=0.45, metallic=0.0, tile=1.0, alpha=None, emissive=None):
    """A material on an (H, W, 3|4) linear image, mapped in metres (one repeat per `tile`)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    H, W = rgb.shape[:2]
    rgba = np.concatenate([rgb[..., :3], (rgb[..., 3:4] if rgb.shape[2] == 4 else np.ones((H, W, 1)))], -1)
    img = MAT.image_from_array(name + "_img", rgba)
    m, nb = material(name)
    tex = nb.node("ShaderNodeTexImage", image=img, extension="REPEAT")
    nb.connect(nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0], tex.inputs["Vector"])
    kw = dict(Base_Color=tex.outputs["Color"], Roughness=rough, Metallic=metallic)
    if alpha:
        kw["Alpha"] = nb.math("ROUND", tex.outputs["Alpha"])
    if emissive:
        kw["Emission_Color"] = tex.outputs["Color"]
        kw["Emission_Strength"] = emissive
    nb.output(nb.principled(**kw))
    m["uv_cell"] = (tile, tile)
    return m


def worn_paint(name, base, rough=0.45, tile=1.2, seed=0, px=1024, metallic=0.0, **kw):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    return image_material(name, paint_image(base, px, seed, **kw), rough, metallic, tile)


def bare_steel(name, base="#5d5a55", rough=0.5, tile=1.0, seed=3, rust=0.6, scrape=1.0, splat=0.0, metallic=0.55, splat_col="#0e0d0c"):
    """Abraded steel (a skip floor, a bucket, a roller drum): scrape lines along u, rust blooms."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    px = 1024
    rng = np.random.default_rng(seed)
    S = (px, px)
    b = _lin(base)
    lo = fbm(S, (3, 3), rng, 5)
    img = b[None, None, :] * (1 + 0.25 * (lo[..., None] - 0.5))
    # scrape marks running along u
    sc = fbm(S, (2, 220), rng, 2)
    img *= (1 + 0.28 * scrape * (sc[..., None] - 0.5))
    _segments(img, rng, int(70 * scrape), px * 0.05, 0.6, _lin("#8a867e"), 0.18, px)
    r = _smooth(0.55, 0.85, fbm(S, (6, 6), rng, 5)) * rust
    img = img * (1 - r[..., None]) + _lin("#5b3a22") * r[..., None] * (0.8 + 0.4 * fbm(S, (40, 40), rng, 2)[..., None])
    if splat:
        # bitumen: black flecks and smears picked up from fresh tarmac
        f1, _ = worley(S, (40, 40), rng)
        cl = _smooth(0.5, 0.8, fbm(S, (4, 4), rng, 4)) * splat
        dots = _smooth(0.0, -0.02, f1 - 0.12 * cl - 0.02 * fbm(S, (60, 60), rng, 2)) * (cl > 0.05)
        smear = _smooth(0.62, 0.8, fbm(S, (8, 30), rng, 4)) * splat * 0.7
        m = np.clip(dots + smear, 0, 1)
        img = img * (1 - m[..., None]) + _lin(splat_col) * m[..., None]
    return image_material(name, np.clip(img, 0, 1), rough, metallic, tile)


def dirty_rubber(name="Mach_Tyre", seed=5):
    """Tyre rubber, dusty and part mud-caked."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    img = paint_image("#1c1b1a", 512, seed, wear=0.0, dust=0.0, streak=0.0, mottle=0.8)
    rng = np.random.default_rng(seed + 1)
    S = img.shape[:2]
    mud = _smooth(0.5, 0.75, fbm(S, (6, 6), rng, 5))
    img = img * (1 - 0.6 * mud[..., None]) + _lin("#4a3f33") * 0.6 * mud[..., None]
    dust = _smooth(0.3, 0.9, fbm(S, (3, 3), rng, 3)) * 0.35
    img = img * (1 - dust[..., None]) + _lin("#55524c") * dust[..., None]
    return image_material(name, img, 0.88, 0.0, 0.6)


def track_steel(name="Mach_Track", seed=7):
    """Crawler track shoes: dark steel, worn bright on the grousers, mud and rust in the pockets."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    img = paint_image("#2b2925", 512, seed, wear=0.0, dust=0.6, dust_col="#5c4c3a", streak=0.0)
    rng = np.random.default_rng(seed)
    S = img.shape[:2]
    r = _smooth(0.5, 0.8, fbm(S, (5, 5), rng, 4)) * 0.55
    img = img * (1 - r[..., None]) + _lin("#4b3020") * r[..., None]
    return image_material(name, img, 0.7, 0.3, 0.5)


# -- lettering ------------------------------------------------------------------------------------

def text_mask(text, w, h, bold=0.0, ss=3, align="center", italic=0.0):
    """(h, w) coverage of `text` in Blender's built-in font, fitted inside w x h px (row 0 = top)."""
    cu = bpy.data.curves.new("_mach_txt", "FONT")
    cu.body = text
    cu.offset = bold
    cu.shear = italic
    ob = bpy.data.objects.new("_mach_txt", cu)
    me = ob.to_mesh()
    me.calc_loop_triangles()
    V = np.array([v.co[:2] for v in me.vertices], float)
    T = np.array([t.vertices[:] for t in me.loop_triangles], int)
    ob.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    out = np.zeros((h * ss, w * ss))
    if not len(T):
        return out[::ss, ::ss]
    x0, y0 = V.min(0)
    x1, y1 = V.max(0)
    s = min(w * ss / (x1 - x0), h * ss / (y1 - y0))
    ox = {"center": (w * ss - (x1 - x0) * s) / 2, "left": 0.0, "right": w * ss - (x1 - x0) * s}[align]
    P = np.empty_like(V)
    P[:, 0] = (V[:, 0] - x0) * s + ox
    P[:, 1] = (y1 - V[:, 1]) * s + (h * ss - (y1 - y0) * s) / 2
    for a, b, c in T:
        A, B, C = P[a], P[b], P[c]
        xa, xb = int(max(min(A[0], B[0], C[0]), 0)), int(min(max(A[0], B[0], C[0]) + 1, w * ss))
        ya, yb = int(max(min(A[1], B[1], C[1]), 0)), int(min(max(A[1], B[1], C[1]) + 1, h * ss))
        if xb <= xa or yb <= ya:
            continue
        X, Y = np.meshgrid(np.arange(xa, xb) + 0.5, np.arange(ya, yb) + 0.5)

        def edge(p, q):
            return (q[0] - p[0]) * (Y - p[1]) - (q[1] - p[1]) * (X - p[0])
        e0, e1, e2 = edge(A, B), edge(B, C), edge(C, A)
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        out[ya:yb, xa:xb] = np.maximum(out[ya:yb, xa:xb], inside)
    return out.reshape(h, ss, w, ss).mean((1, 3))


# -- decal atlas ----------------------------------------------------------------------------------

def _rgba(h, w, col, a=1.0):
    out = np.zeros((h, w, 4))
    out[..., :3] = _lin(col)
    out[..., 3] = a
    return out


def over(dst, mask, col):
    dst[..., :3] = dst[..., :3] * (1 - mask[..., None]) + _lin(col) * mask[..., None]
    dst[..., 3] = np.maximum(dst[..., 3], mask)
    return dst


def plate(text, w, h, fg, bg=None, border=None, bold=0.02, pad=0.14, italic=0.0, align="center"):
    """A sticker or stencil: text on a background (bg None: cut-out letters)."""
    img = _rgba(h, w, bg or fg, 1.0 if bg else 0.0)
    b = int(round(min(w, h) * 0.06)) if border else 0
    if border:
        m = np.zeros((h, w))
        m[:b], m[-b:], m[:, :b], m[:, -b:] = 1, 1, 1, 1
        over(img, m, border)
    p = int(h * pad)
    tm = text_mask(text, w - 2 * p - 2 * b, h - 2 * p - 2 * b, bold, align=align, italic=italic)
    m = np.zeros((h, w))
    m[p + b:h - p - b, p + b:w - p - b] = tm
    return over(img, m, fg)


def chevrons(w, h, c1="#e7b416", c2="#151515", period=None, angle=45):
    """Hazard stripes (diagonal bands)."""
    period = period or h * 0.9
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    t = (X + Y * math.tan(math.radians(angle))) / period
    f = np.clip((np.abs((t % 1.0) - 0.5) - 0.25) * period + 0.5, 0, 1)
    img = _rgba(h, w, c1)
    return over(img, f, c2)


def chevron_v(w, h, c1="#c8102e", c2="#f2f2f0", period=None):
    """Rear-marking chevrons (inverted Vs pointing up)."""
    period = period or w / 5
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    t = (np.abs(X - w / 2) + (h - Y)) / period
    f = np.clip((np.abs((t % 1.0) - 0.5) - 0.25) * period + 0.5, 0, 1)
    img = _rgba(h, w, c1)
    return over(img, f, c2)


def louvre(w, h, slots, frame="#2a2a2a", slot="#060606", lip="#5a5a58", vertical=False, margin=0.08):
    """A pressed louvre or slotted grille: dark slots with a lit lower lip and a shadowed upper
    edge (reads as depth from a few metres away). frame None: only the slots and lips (the
    panel's own paint shows round them)."""
    img = _rgba(h, w, frame or "#000000", 1.0 if frame else 0.0)
    if vertical:
        img = img.transpose(1, 0, 2).copy()
        w, h = h, w
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    mx, my = w * margin, h * margin
    inside = (X > mx) & (X < w - mx) & (Y > my) & (Y < h - my)
    pitch = (h - 2 * my) / slots
    t = ((Y - my) / pitch) % 1.0
    slotm = inside & (t > 0.18) & (t < 0.78)
    sh = np.clip((t - 0.18) / 0.6, 0, 1)
    col = _lin(slot) * (0.4 + 1.2 * sh[..., None] ** 2)
    img[..., :3] = np.where(slotm[..., None], col, img[..., :3])
    lipm = inside & (t >= 0.78) & (t < 0.9)
    img[..., :3] = np.where(lipm[..., None], _lin(lip), img[..., :3])
    img[..., 3] = np.where(slotm | lipm, 1.0, img[..., 3])
    if vertical:
        img = img.transpose(1, 0, 2).copy()
    return img


def mesh_grille(w, h, cell, wire=0.28, frame="#1a1a1a", hole="#050505", border=0.06):
    """A diamond-mesh grille insert with a frame."""
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    u = (X + Y) / cell
    v = (X - Y) / cell
    d = np.minimum(np.abs(u - np.round(u)), np.abs(v - np.round(v)))
    wiremask = np.clip((wire / 2 - d) * cell + 0.5, 0, 1)
    img = _rgba(h, w, hole)
    over(img, wiremask, "#3a3a3a")
    b = int(min(w, h) * border)
    m = np.zeros((h, w))
    m[:b], m[-b:], m[:, :b], m[:, -b:] = 1, 1, 1, 1
    return over(img, m, frame)


def warning(s, glyph="!", col="#f2c200"):
    """A yellow warning triangle sticker with a black border and a glyph."""
    img = _rgba(s, s, "#000000", 0.0)
    Y, X = np.mgrid[0:s, 0:s] + 0.5
    P = [np.array([s / 2, s * 0.06]), np.array([s * 0.96, s * 0.88]), np.array([s * 0.04, s * 0.88])]
    d = np.full((s, s), 1e9)
    for i in range(3):
        a, b = P[i], P[(i + 1) % 3]
        t = b - a
        nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)     # inward for this winding
        d = np.minimum(d, (X - a[0]) * nrm[0] + (Y - a[1]) * nrm[1])
    over(img, np.clip(d + 0.5, 0, 1), "#111111")
    over(img, np.clip(d - s * 0.07 + 0.5, 0, 1), col)
    tm = text_mask(glyph, int(s * 0.3), int(s * 0.45), 0.03)
    m = np.zeros((s, s))
    m[int(s * 0.38):int(s * 0.38) + tm.shape[0], int(s * 0.35):int(s * 0.35) + tm.shape[1]] = tm
    return over(img, m, "#111111")


class Atlas:
    """One decal image per machine: items shelf-packed into a square, placed with `put`."""

    def __init__(self, name, px=1024):
        self.name, self.px = name, px
        self.img = np.zeros((px, px, 4))
        self.rects = {}
        self.x = self.y = self.row = 0
        self._mat = None

    def add(self, key, rgba):
        h, w = rgba.shape[:2]
        if self.x + w > self.px:
            self.x, self.y, self.row = 0, self.y + self.row + 2, 0
        assert self.y + h <= self.px, f"atlas {self.name} full at {key}"
        self.img[self.y:self.y + h, self.x:self.x + w] = rgba
        # uv rect, inset half a pixel; image rows run top-down here, v bottom-up
        P = self.px
        self.rects[key] = ((self.x + 0.5) / P, 1 - (self.y + h - 0.5) / P, (self.x + w - 0.5) / P, 1 - (self.y + 0.5) / P)
        self.x += w + 2
        self.row = max(self.row, h)

    def material(self):
        if self._mat is None:
            if bpy.data.materials.get(self.name):
                self._mat = bpy.data.materials[self.name]
            else:
                img = self.img[::-1].copy()
                self._mat = image_material(self.name, img, 0.45, 0.0, 1.0, alpha=True)
        return self._mat

    def put(self, p, key, c, n, v, w, h, lift=0.003, flip=False):
        """Decal `key` w x h centred on c on a face with outward normal n, its top towards v."""
        c, n, v = (np.asarray(x, float) for x in (c, n, v))
        n = n / np.linalg.norm(n)
        v = v / np.linalg.norm(v)
        u = np.cross(v, n)
        c = c + n * lift
        P = [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
        u0, v0, u1, v1 = self.rects[key]
        if flip:
            u0, u1 = u1, u0
        r, p.r = p.r, 0.0
        p.face(P, self.material(), [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
        p.r = r


# -- geometry helpers -----------------------------------------------------------------------------

def metric_uvs(p):
    """Give every face without UVs a box projection in metres (so materials with a `uv_cell` tile
    at their own size)."""
    for i, (f, uv) in enumerate(zip(p.F, p.uv)):
        if uv is not None:
            continue
        P = np.array([p.V[k] for k in f])
        n = np.zeros(3)
        for k in range(len(P)):
            n += np.cross(P[k], P[(k + 1) % len(P)])
        ax = int(np.argmax(np.abs(n)))
        keep = [j for j in range(3) if j != ax]
        p.uv[i] = [tuple(q[keep]) for q in P]


def finish(p, name, col, smooth=30, mud=0.55, mud_h=0.7, seed=0, sharp=(), tint=(0.66, 0.55, 0.42)):
    """Make the object, bake occlusion, then lay mud and grime into the vertex colours: browner and
    darker towards the ground, patchy. Faces in `sharp` materials (small hardware: nuts, pins,
    rods, lamps) skip the edge rounding, which would cost thousands of triangles unseen."""
    from ..core import ao
    metric_uvs(p)
    for i, mi in enumerate(p.mi):
        if p.mats[mi] in sharp:
            p.rad[i] = 0.0
    o = p.obj(name, col, smooth)
    ao.bake(o)
    me = o.data
    attr = me.color_attributes["Col"]
    n = len(me.vertices)
    co = np.zeros(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    c = np.zeros(len(attr.data) * 4, np.float32)
    attr.data.foreach_get("color", c)
    c = c.reshape(-1, 4)
    x, y, z = co.T
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(6, 3)) * np.array([2.2, 2.2, 3.0])
    ph = rng.random(6) * 6.28
    noise = sum(np.sin(co @ k[i] + ph[i]) for i in range(6)) / 6 * 0.5 + 0.5
    h = mud_h * (0.75 + 0.5 * noise)
    f = np.clip((h - z) / h, 0, 1) ** 1.3 * mud * (0.55 + 0.45 * noise)
    tint = np.asarray(tint, float)
    c[:, :3] *= 1 - f[:, None] * (1 - tint)
    attr.data.foreach_set("color", c.ravel())
    return o


def ring_pts(c, ax, e1, e2, r, a):
    return c + (e1 * math.cos(a) + e2 * math.sin(a)) * r


def wheel(p, c, R, w, rim_r, rubber, rim, nuts_mat, out=1, lugs=24, lug_h=0.04, lug_w=0.07, kind="bar",
          sides=40, dish=0.35, nuts=8, hub_mat=None, rim_detail=True):
    """A wheel turning about x at c, its outer face towards +x*out: a tyre with rounded shoulders
    and sidewall bulge, tread bars (kind "bar": R-1 tractor chevrons running onto the shoulder;
    "block": industrial blocks; "turf": shallow chevrons), a pressed steel rim with a flange, a
    dished centre with a hub and wheel nuts."""
    c = np.asarray(c, float)
    ax = np.array([1.0, 0, 0]) * out
    e1 = np.array([0, 1.0, 0])
    e2 = np.array([0, 0, 1.0])
    hw = w / 2
    sw = R - rim_r
    r0 = p.r

    def ring(profile, mat, n=sides):
        G = [[c + ax * a + (e1 * math.cos(2 * math.pi * k / n) + e2 * math.sin(2 * math.pi * k / n)) * rr for rr, a in profile] for k in range(n)]
        m = len(profile)
        for k in range(n):
            k2 = (k + 1) % n
            for j in range(m - 1):
                q = [G[k][j], G[k2][j], G[k2][j + 1], G[k][j + 1]]
                p.face(q if out > 0 else q[::-1], mat)
    p.r = 0.0
    tread_r = R - (lug_h if kind != "turf" else 0.0)
    prof = [(rim_r * 1.04, -hw * 0.86), (rim_r + 0.3 * sw, -hw * 0.98), (rim_r + 0.62 * sw, -hw),
            (tread_r - 0.12 * sw, -hw * 0.95), (tread_r - 0.02 * sw, -hw * 0.82), (tread_r, -hw * 0.6),
            (tread_r, hw * 0.6), (tread_r - 0.02 * sw, hw * 0.82), (tread_r - 0.12 * sw, hw * 0.95),
            (rim_r + 0.62 * sw, hw), (rim_r + 0.3 * sw, hw * 0.98), (rim_r * 1.04, hw * 0.86)]
    ring(prof, rubber)
    # rim: flange, bead seat, well, then the centre disc dished in and the hub
    fl = rim_r * 1.06
    d0 = hw * (0.5 - dish)                      # where the centre disc meets the barrel
    hz = d0 + 0.09 * rim_r                      # the hub plate, standing proud of the disc
    ring([(rim_r * 1.04, hw * 0.86), (fl, hw * 0.84), (fl, hw * 0.76), (rim_r * 0.97, hw * 0.7),
          (rim_r * 0.93, hw * 0.55), (rim_r * 0.9, d0 + 0.02), (rim_r * 0.85, d0), (rim_r * 0.62, d0 + 0.05 * rim_r),
          (rim_r * 0.44, hz - 0.01 * rim_r), (rim_r * 0.42, hz)], rim)
    ring([(rim_r * 0.42, hz), (rim_r * 0.2, hz), (0.0, hz)], hub_mat or rim, n=24)
    ring([(0.0, -hw * 0.6), (rim_r * 0.9, -hw * 0.6), (rim_r * 0.97, -hw * 0.7), (fl, -hw * 0.76), (fl, -hw * 0.84), (rim_r * 1.04, -hw * 0.86)], rim)
    # hub boss with its cap, wheel nuts, rim-to-disc bolts, valve
    p.r = 0.0
    zf = hz
    p.cyl(c + ax * zf, c + ax * (zf + rim_r * 0.14), rim_r * 0.19, hub_mat or rim, 18, r2=rim_r * 0.16)
    p.cyl(c + ax * (zf + rim_r * 0.14), c + ax * (zf + rim_r * 0.2), rim_r * 0.1, nuts_mat, 14, r2=rim_r * 0.07)
    for k in range(nuts):
        a = 2 * math.pi * (k + 0.5) / nuts
        q = c + ax * zf + (e1 * math.cos(a) + e2 * math.sin(a)) * rim_r * 0.3
        p.cyl(q, q + ax * rim_r * 0.06, rim_r * 0.035, nuts_mat, 6)
    if rim_detail and rim_r > 0.2:
        for k in range(12):
            a = 2 * math.pi * (k + 0.25) / 12
            q = c + ax * (d0 + 0.012 * rim_r) + (e1 * math.cos(a) + e2 * math.sin(a)) * rim_r * 0.8
            p.cyl(q, q + ax * rim_r * 0.035, rim_r * 0.025, nuts_mat, 6)
    a = 0.3
    q = c + ax * (hw * 0.55) + (e1 * math.cos(a) + e2 * math.sin(a)) * rim_r * 0.9
    p.cyl(q, q + ax * 0.04 + (e1 * math.cos(a) + e2 * math.sin(a)) * -0.01, 0.006, nuts_mat, 6)
    if rim_detail and rim_r > 0.2:
        # hand holes in the disc (dark ovals) read as a pressed wheel
        for k in range(6):
            a = 2 * math.pi * k / 6
            rq = e1 * math.cos(a) + e2 * math.sin(a)
            q = c + ax * (d0 + 0.028 * rim_r + 0.004) + rq * rim_r * 0.7
            tq = e1 * -math.sin(a) + e2 * math.cos(a)
            pts = [q + tq * rim_r * 0.12 * math.cos(2 * math.pi * j / 10) + rq * rim_r * 0.07 * math.sin(2 * math.pi * j / 10) for j in range(10)]
            p.face(pts if out > 0 else pts[::-1], nuts_mat)
    if not lugs:
        p.r = r0
        return
    # tread bars
    p.r = 0.0
    for k in range(lugs):
        a0 = 2 * math.pi * k / lugs
        for s in (-1, 1):
            a = a0 + (math.pi / lugs if s > 0 else 0.0)
            if kind == "bar":
                xs = [0.04, 0.2, 0.4, 0.58, 0.78, 0.95]
                sweep = 0.55 * (hw / R)
            elif kind == "block":
                xs = [0.08, 0.3, 0.55, 0.78, 0.95]
                sweep = 0.25 * (hw / R)
            else:
                xs = [0.05, 0.3, 0.55, 0.8]
                sweep = 0.4 * (hw / R)
            rings = []
            for i, xf in enumerate(xs):
                xx = s * hw * xf
                ang = a + sweep * xf
                # drop over the shoulder
                drop = 0.0 if xf < 0.6 else (xf - 0.6) / 0.4 * 0.1 * sw
                rr = tread_r - drop
                rad = e1 * math.cos(ang) + e2 * math.sin(ang)
                tan = -e1 * math.sin(ang) + e2 * math.cos(ang)
                P0 = c + ax * xx + rad * (rr - 0.01)
                d_along = np.array(ax * s * hw * 0.2 + tan * R * sweep * 0.2)
                d_along /= np.linalg.norm(d_along)
                side = np.cross(rad, d_along)
                side /= np.linalg.norm(side)
                lw = lug_w * (1.0 if kind != "block" else 1.6) * (1.0 - 0.15 * xf)
                hh = lug_h + 0.01
                rings.append([P0 - side * lw / 2, P0 + side * lw / 2, P0 + side * lw * 0.38 + rad * hh, P0 - side * lw * 0.38 + rad * hh])
            p.loft(rings, rubber)
    p.r = r0


def checker_plate(name="Mach_Checker", base="#6d6c68", seed=9):
    """Diamond tread plate (steps, platforms), 0.25 m per repeat: raised lentil bosses in two
    directions, lit on one flank, worn shiny on the tops, dirt in the field."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    px = 512
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:px, 0:px] + 0.5
    cell = px / 8
    img = np.broadcast_to(_lin(base), (px, px, 3)).copy() * 0.7
    for off, ang in ((0.0, 45.0), (0.5, -45.0)):
        cx = ((X / cell + off) % 1.0) - 0.5
        cy = ((Y / cell + off) % 1.0) - 0.5
        a = math.radians(ang)
        u = cx * math.cos(a) + cy * math.sin(a)
        v = -cx * math.sin(a) + cy * math.cos(a)
        d = (u / 0.36) ** 2 + (v / 0.07) ** 2
        boss = np.clip((1 - d) * 4, 0, 1)
        light = 1.0 + 0.6 * np.clip(-v / 0.07, -1, 1)
        img = img * (1 - boss[..., None]) + (_lin(base) * 1.25 * light[..., None]) * boss[..., None]
    dirt = _smooth(0.45, 0.8, fbm((px, px), (4, 4), rng, 4)) * 0.5
    img = img * (1 - dirt[..., None]) + _lin("#4d4237") * dirt[..., None]
    return image_material(name, np.clip(img, 0, 1), 0.55, 0.6, 0.25)


def outline(w, h, line=3, col="#141414", radius=0.06, handle=None):
    """A panel-gap outline (service doors, hatches): a dark rounded-rectangle line on clear, with
    an optional handle slot ((u, v) centre as fractions)."""
    img = _rgba(h, w, col, 0.0)
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    r = min(w, h) * radius
    qx = np.maximum(np.abs(X - w / 2) - (w / 2 - r - line), 0)
    qy = np.maximum(np.abs(Y - h / 2) - (h / 2 - r - line), 0)
    d = np.hypot(qx, qy) - r
    m = np.clip(line / 2 - np.abs(d) + 0.5, 0, 1)
    over(img, m * 0.85, col)
    if handle:
        hx, hy = handle[0] * w, handle[1] * h
        hm = ((np.abs(X - hx) < w * 0.06) & (np.abs(Y - hy) < h * 0.025)).astype(float)
        over(img, hm, "#0a0a0a")
    return img


def gauges(w, h, n=2, bg="#161616"):
    """A dash pod face: n round dials (white ticks, an orange needle) and a row of tell-tales."""
    img = _rgba(h, w, bg)
    Y, X = np.mgrid[0:h, 0:w] + 0.5
    r = min(h * 0.36, w / (2.6 * n))
    for k in range(n):
        cx, cy = w * (k + 0.5) / n, h * 0.45
        d = np.hypot(X - cx, Y - cy)
        over(img, np.clip(r - d + 0.5, 0, 1), "#2b2d30")
        over(img, np.clip(1.2 - np.abs(d - r * 0.92) + 0.5, 0, 1) * 0.9, "#d8d8d8")
        ang = np.arctan2(Y - cy, X - cx)
        for t in np.linspace(math.radians(135), math.radians(405), 9):
            a = (t + math.pi) % (2 * math.pi) - math.pi
            m = (np.abs(np.angle(np.exp(1j * (ang - a)))) < 0.04) & (d > r * 0.7) & (d < r * 0.85)
            over(img, m.astype(float), "#e8e8e8")
        a = math.radians(200 + 70 * k)
        dd = np.array([math.cos(a), math.sin(a)])
        t = np.clip((X - cx) * dd[0] + (Y - cy) * dd[1], 0, r * 0.78)
        dist = np.hypot(X - cx - t * dd[0], Y - cy - t * dd[1])
        over(img, np.clip(1.5 - dist + 0.5, 0, 1), "#ff7a1a")
    for k, c in enumerate(("#3fbf3f", "#e0b020", "#d03030", "#3070e0")):
        cx = w * (0.3 + 0.13 * k)
        d = np.hypot(X - cx, Y - h * 0.88)
        over(img, np.clip(h * 0.045 - d + 0.5, 0, 1), c)
    return img


def perforated(name="Mach_Perforated", pitch_px=16, tile=0.08):
    """Perforated black steel (exhaust heat guards): round holes on a staggered grid."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    px = 128
    Y, X = np.mgrid[0:px, 0:px] + 0.5
    img = np.broadcast_to(_lin("#1c1c1c"), (px, px, 3)).copy()
    for off in (0.0, 0.5):
        cx = ((X / pitch_px + off) % 1.0) - 0.5
        cy = ((Y / pitch_px + off) % 1.0) - 0.5
        d = np.hypot(cx, cy) * pitch_px
        hole = np.clip(pitch_px * 0.28 - d + 0.5, 0, 1)
        img = img * (1 - hole[..., None]) + _lin("#030303") * hole[..., None]
    return image_material(name, img, 0.5, 0.4, tile)


def soot(s, seed=4):
    """A soft, broken black stain (exhaust soot, oil weep) on clear."""
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:s, 0:s] + 0.5
    d = np.hypot(X - s / 2, (Y - s / 2) * 1.3) / (s / 2)
    n = fbm((s, s), (6, 6), rng, 4)
    a = np.clip((1 - d) * 1.6 * (0.4 + 0.9 * n), 0, 0.85)
    img = _rgba(s, s, "#0d0c0b", 0.0)
    img[..., 3] = a
    return img
