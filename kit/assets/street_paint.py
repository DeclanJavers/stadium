"""Painted textures and UV-aware mesh helpers for the street furniture (props_street).

Textures are painted in numpy (linear RGB, row 0 = the bottom of the image) and kept as image
materials, so they survive the game bake as they are:

- skins: a surface finish laid out by HEIGHT. One repeat across (`tile` metres of arc or of x/y)
  and the prop's full height up the image, so a vertical face's metric UV (along, z) lands grime on
  the bottom 150 mm, splash, scuffs at bumper height, rain streaks under a rim, UV fading high up.
- timber: an atlas of eight boards, each mapped whole along its length (end staining at both ends)
  and round its section (weathered grey on the top face, greener and darker underneath).
- labels: vector type (Blender's font outlines, rasterised anti-aliased) and pictograms on cards.

Mesh helpers give explicit UVs in metres (u along, v up): a lathe whose u is arc length (no seams
from box projection), boards, a panel with a recessed opening, curved decals that hug a cylinder.
"""
import glob
import math
import zlib

import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material, srgb
from .foliage_paint import fbm

SS = 4   # supersampling for vector shapes


# -- basics ---------------------------------------------------------------------------------------

def lin(hexs):
    return np.array(srgb(hexs)[:3])


def rng(name):
    return np.random.default_rng(zlib.crc32(name.encode()))


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def noise(shape, cells, r, octaves=4, gain=0.5):
    """Periodic fbm in 0-1, contrast-stretched (fbm sits around 0.5)."""
    n = fbm(shape, cells, r, octaves, gain)
    return np.clip((n - n.mean()) / (n.std() * 3.2 + 1e-6) + 0.5, 0, 1)


def blur(a, r):
    """Wrapping box blur, twice (close to a gaussian)."""
    if r < 1:
        return a
    for _ in range(2):
        for ax in (0, 1):
            c = np.cumsum(np.concatenate([np.take(a, range(-r - 1, 0), axis=ax), a, np.take(a, range(0, r), axis=ax)], ax), ax)
            n = a.shape[ax]
            a = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=ax) - np.take(c, range(0, n), axis=ax)) / (2 * r + 1)
    return a


def mixc(base, col, m):
    """Blend an (H, W, 3) image towards colour col (or image) by mask m."""
    col = np.asarray(col, float)
    return base + (col - base) * m[..., None]


def normal_map(h, strength):
    """Tangent-space normal map (OpenGL, +y up the image) from a height field in pixels."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = np.stack([-dx, -dy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


# -- vector shapes --------------------------------------------------------------------------------

def _raster_tris(H, W, tris):
    """Coverage (H, W) of triangles in pixel units (x right, y up), SS x supersampled."""
    m = np.zeros((H * SS, W * SS), bool)
    t = np.asarray(tris, float) * SS
    for a, b, c in t:
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(W * SS, math.ceil(max(a[0], b[0], c[0])) + 1))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(H * SS, math.ceil(max(a[1], b[1], c[1])) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        e0 = (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])
        e1 = (c[0] - b[0]) * (Y - b[1]) - (c[1] - b[1]) * (X - b[0])
        e2 = (a[0] - c[0]) * (Y - c[1]) - (a[1] - c[1]) * (X - c[0])
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        m[y0:y1, x0:x1] |= inside
    return m.reshape(H, SS, W, SS).mean((1, 3))


def poly_mask(H, W, polys):
    """Coverage of polygons [(x, y), ...] (even-odd across all of them), pixel units."""
    P = [np.asarray(p, float) * SS for p in polys]
    allp = np.concatenate(P)
    x0, y0 = np.maximum(np.floor(allp.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(allp.max(0)).astype(int) + 1, (W * SS, H * SS))
    m = np.zeros((H * SS, W * SS), bool)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W))
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.zeros(X.shape, bool)
    for p in P:
        q = np.roll(p, -1, 0)
        for (ax, ay), (bx, by) in zip(p, q):
            if ay == by:
                continue
            cross = ((ay > Y) != (by > Y)) & (X < (bx - ax) * (Y - ay) / (by - ay) + ax)
            inside ^= cross
    m[y0:y1, x0:x1] = inside
    return m.reshape(H, SS, W, SS).mean((1, 3))


def circle(cx, cy, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def rect(x0, y0, x1, y1, r=0.0, n=6):
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    out = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


_FONTS = {}


def _font(bold=True):
    key = "b" if bold else "r"
    if key in _FONTS:
        return _FONTS[key]
    cands = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/arial.ttf"])
    cands += glob.glob(str(bpy.utils.resource_path("LOCAL")) + "/datafiles/fonts/Inter.woff2")
    f = None
    for c in cands:
        try:
            f = bpy.data.fonts.load(c, check_existing=True)
            break
        except Exception:
            continue
    _FONTS[key] = f
    return f


_TEXT = {}


def text_tris(s, bold=True):
    """Triangles of a line of text in font units (baseline y = 0, cap height ~0.72), and its width."""
    key = (s, bold)
    if key in _TEXT:
        return _TEXT[key]
    cu = bpy.data.curves.new("_txt", "FONT")
    f = _font(bold)
    if f:
        cu.font = f
    cu.body = s
    cu.fill_mode = "FRONT"
    ob = bpy.data.objects.new("_txt", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.array([v.co[:2] for v in me.vertices])
    tris = np.array([[co[i] for i in t.vertices] for t in me.loop_triangles]) if len(me.loop_triangles) else np.zeros((0, 3, 2))
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    w = float(co[:, 0].max()) if len(co) else 0.0
    _TEXT[key] = (tris, w)
    return tris, w


def text_mask(H, W, s, x, y, cap, align="c", bold=True, max_w=None, track=0.0):
    """Coverage of text with its cap height `cap` px, baseline at y, anchored at x (l/c/r)."""
    tris, w = text_tris(s, bold)
    if not len(tris):
        return np.zeros((H, W))
    k = cap / 0.72
    if max_w and w * k > max_w:
        k = max_w / w
    t = tris * k
    ww = w * k
    dx = {"l": 0.0, "c": -ww / 2, "r": -ww}[align]
    t = t + np.array([x + dx, y])
    return _raster_tris(H, W, t)


# -- image materials ------------------------------------------------------------------------------

def _data_image(name, arr):
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([arr, np.ones((h, w, 1))], -1).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def image_material(name, albedo, rough=0.5, normal=None, metal=0.0, cell=None, alpha=None, emission=None,
                   extension="REPEAT", base_hex=None, spec=0.5):
    """A material from painted maps. albedo (H, W, 3) linear; rough a number or (H, W) map;
    normal an encoded (H, W, 3) map; cell: metres per repeat (faces give metric UVs);
    alpha (H, W) clipped at 0.5; emission (colour hex, strength) masked by the albedo."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    a = np.concatenate([np.clip(albedo, 0, 1), (alpha if alpha is not None else np.ones(albedo.shape[:2]))[..., None]], -1)
    t_alb = nb.node("ShaderNodeTexImage", image=MAT.image_from_array(f"{name}_albedo", a), extension=extension)
    nb.connect(uv, t_alb.inputs["Vector"])
    kw = dict(Base_Color=t_alb.outputs["Color"], Metallic=metal, Specular_IOR_Level=spec)
    if isinstance(rough, np.ndarray):
        t_r = nb.node("ShaderNodeTexImage", image=_data_image(f"{name}_rough", np.clip(rough, 0.02, 1)), extension=extension)
        nb.connect(uv, t_r.inputs["Vector"])
        kw["Roughness"] = t_r.outputs["Color"]
    else:
        kw["Roughness"] = rough
    if normal is not None:
        t_n = nb.node("ShaderNodeTexImage", image=_data_image(f"{name}_normal", normal), extension=extension)
        nb.connect(uv, t_n.inputs["Vector"])
        nm = nb.node("ShaderNodeNormalMap")
        nb.connect(t_n.outputs["Color"], nm.inputs["Color"])
        kw["Normal"] = nm.outputs[0]
    if alpha is not None:
        kw["Alpha"] = nb.math("ROUND", t_alb.outputs["Alpha"])
    if emission:
        kw["Emission_Color"] = t_alb.outputs["Color"]
        kw["Emission_Strength"] = emission
    nb.output(nb.principled(**kw))
    if cell:
        m["uv_cell"] = tuple(cell)
    if base_hex:
        m["base_hex"] = base_hex
    return m


# -- skins: finishes laid out by height -----------------------------------------------------------

def skin(name, base, H, tile=1.0, px=(512, 1024), rough=0.5, metal=0.0, seed=None,
         mottle=0.06, grime=0.16, grime_col="#6b604f", grime_amt=0.55, splash=0.4,
         scuff=None, streaks=(), chalk=0.0, peel=0.08, spangle=0.0, rust=0.0, rust_col="#6a3a1e",
         wear_bands=(), paint=None, normal_strength=1.0, base_hex=None, rubber=None, primer=None, gum=0,
         stickers=0, rust_runs=(), dents=0, top_fade=None):
    """A height-mapped surface finish: u across one repeat of `tile` m, v from z = 0 to H.

    grime: height (m) of the soil/dust band at the foot; splash: speckle above it; scuff: (z0, z1,
    amount) knocks and rubs (lighter on dark finishes, darker on pale ones); streaks: [(z_top,
    amount, length)] rain/dirt runs down from a rim; chalk: UV fading towards the top; spangle:
    galvanised crystal pattern; rust: rust bleed at the foot; wear_bands: [(z0, z1, amount, colour)]
    paint worn through to metal or primer; paint(alb, rough, hgt, X, Z, r): a hook for decals."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    W, Hp = px
    r = rng(seed or name)
    X, Z = np.meshgrid((np.arange(W) + 0.5) / W * tile, (np.arange(Hp) + 0.5) / Hp * H)
    cu = max(1, int(round(tile / 0.25)))       # noise cells across (periodic)
    cv = max(1, int(round(H / 0.25)))
    b = lin(base)
    lum_b = float(b @ (0.2126, 0.7152, 0.0722))
    alb = np.broadcast_to(b, (Hp, W, 3)).copy()
    hgt = np.zeros((Hp, W))
    rgh = np.full((Hp, W), float(rough))
    # large blotches and fine speckle in the moulding/paint
    blot = noise((Hp, W), (cu, cv), r, 4)
    alb *= (1 + mottle * (blot - 0.5) * 2)[..., None]
    fine = noise((Hp, W), (cu * 16, cv * 16), r, 2)
    alb *= (1 + 0.04 * (fine - 0.5))[..., None]
    hgt += peel * noise((Hp, W), (cu * 40, cv * 40), r, 1)
    rgh += 0.08 * (blot - 0.5)
    if spangle > 0:
        # zinc spangle: crystals 5-15 mm, each its own grey and sheen, plus dull white-rust blooms
        from .foliage_paint import worley
        f1, edge = worley((Hp, W), (int(tile / 0.012), int(H / 0.012)), r)
        cell_v = noise((Hp, W), (int(tile / 0.006), int(H / 0.006)), r, 1)
        cell_v = np.round(cell_v * 5) / 5                      # flat-toned crystals
        tone = (cell_v - 0.5) * 0.14 * spangle
        alb *= (1 + tone)[..., None]
        rgh += 0.12 * spangle * (0.5 - cell_v)
        bloom = sstep(0.45, 0.95, noise((Hp, W), (cu * 3, cv * 3), r, 5))
        alb = mixc(alb, alb * 0.85 + lin("#8e908c") * 0.15, bloom * 0.35 * spangle)
        rgh += bloom * 0.25 * spangle
        hgt += sstep(0.0, 0.08, edge) * 0.3 * spangle
    if chalk > 0:
        up = sstep(0.2 * H, H, Z)
        fade = up * (0.6 + 0.4 * noise((Hp, W), (cu * 2, cv), r, 3))
        alb = mixc(alb, alb * 0.75 + lin("#8a8c8c") * 0.25, fade * chalk)
        rgh += 0.15 * fade * chalk
    for z0, z1, amt, col in wear_bands:
        band = sstep(z0 - 0.02, z0 + 0.02, Z) * (1 - sstep(z1 - 0.02, z1 + 0.02, Z))
        wm = band * sstep(0.7, 0.86, noise((Hp, W), (cu * 14, cv * 14), r, 5)) * amt
        alb = mixc(alb, lin(col), wm)
        rgh -= 0.15 * wm
        hgt -= 0.6 * wm
    if streaks:
        sx = noise((Hp, W), (cu * 18, 1), r, 3)                 # columns
        for z_top, amt, length in streaks:
            below = (1 - sstep(z_top - 0.01, z_top + 0.005, Z)) * np.exp(-np.clip(z_top - Z, 0, None) / max(length, 1e-3))
            runs = sstep(0.62, 0.88, sx) * (0.5 + 0.5 * noise((Hp, W), (cu * 18, cv * 3), r, 2))
            m = np.clip(below * runs * amt * 0.6, 0, 1)
            dark = lum_b > 0.08
            alb = mixc(alb, alb * 0.7 if dark else alb * 1.25 + lin("#5f5a4c") * 0.1, m)
            rgh += 0.1 * m
    if scuff:
        z0, z1, amt = scuff
        band = sstep(z0 - 0.05, z0 + 0.05, Z) * (1 - sstep(z1 - 0.05, z1 + 0.05, Z))
        # short horizontal-ish marks: noise stretched across
        marks = sstep(0.8, 0.92, noise((Hp, W), (cu * 8, cv * 30), r, 3)) * sstep(0.6, 0.75, noise((Hp, W), (cu * 12, cv * 6), r, 2))
        m = np.clip(marks * band * amt * 0.45, 0, 1)
        col = b * 1.7 + lin("#6d6a64") * 0.06 if lum_b < 0.15 else b * 0.6
        alb = mixc(alb, col, m)
        rgh += 0.12 * m
        hgt -= 0.3 * m
    if rust > 0:
        foot = 1 - sstep(0.0, 0.12, Z)
        rm = foot * sstep(0.5, 0.8, noise((Hp, W), (cu * 8, cv * 8), r, 5)) * rust * 0.7
        bleed = (1 - sstep(0.0, 0.3, Z)) * sstep(0.6, 0.8, noise((Hp, W), (cu * 12, 2), r, 3)) * rust * 0.6
        rm = np.clip(rm + bleed, 0, 1)
        alb = mixc(alb, lin(rust_col) * (0.7 + 0.6 * fine[..., None]), rm)
        rgh += 0.25 * rm
        hgt += 0.5 * rm * fine
    if grime > 0:
        edge = grime * (0.7 + 0.6 * noise((Hp, W), (cu * 3, 1), r, 3))
        g = (1 - sstep(edge * 0.2, edge, Z)) * (0.75 + 0.25 * noise((Hp, W), (cu * 8, cv * 8), r, 4))
        g = np.clip(g * grime_amt, 0, 1)
        alb = mixc(alb, lin(grime_col), g)
        rgh += 0.25 * g
        if splash > 0:
            sp = noise((Hp, W), (cu * 90, cv * 90), r, 1)
            dots = sstep(0.86, 0.94, sp) * (1 - sstep(0.0, grime * 2.5, Z)) * splash * 0.6
            alb = mixc(alb, lin(grime_col) * 0.9, dots)
            rgh += 0.2 * dots
    def wrapd(x0):                                         # periodic distance across u
        d = np.abs(X - x0)
        return np.minimum(d, tile - d)
    if top_fade:
        z0, amt = top_fade
        f = sstep(z0, H, Z) * (0.7 + 0.3 * noise((Hp, W), (cu * 3, cv * 3), r, 3))
        alb = mixc(alb, alb * 0.6 + lin("#8f887c") * 0.4, f * amt)
        rgh += 0.12 * f * amt
    if primer:
        z0, z1, amt = primer
        band = sstep(z0 - 0.04, z0 + 0.04, Z) * (1 - sstep(z1 - 0.04, z1 + 0.04, Z))
        chips = sstep(0.86, 0.885, noise((Hp, W), (cu * 10, cv * 14), r, 4)) * band * amt
        alb = mixc(alb, lin("#5f615e") * (0.85 + 0.3 * fine[..., None]), chips)
        rgh -= 0.1 * chips
        hgt -= 1.2 * chips
    if rubber:
        z0, z1, amt = rubber
        band = sstep(z0 - 0.03, z0 + 0.03, Z) * (1 - sstep(z1 - 0.03, z1 + 0.03, Z))
        smear = sstep(0.62, 0.8, noise((Hp, W), (cu * 3, cv * 40), r, 3)) * sstep(0.45, 0.7, noise((Hp, W), (cu * 6, cv * 3), r, 2))
        m = np.clip(smear * band * amt, 0, 1)
        alb = mixc(alb, lin("#121212"), m * 0.85)
        rgh -= 0.12 * m
    for z_top, amt in rust_runs:
        for _ in range(4):
            x0, w = r.uniform(0, tile), r.uniform(0.003, 0.007)
            ln = r.uniform(0.04, 0.16)
            m = (1 - sstep(w * 0.5, w, wrapd(x0))) * (Z < z_top) * np.exp(-np.clip(z_top - Z, 0, None) / ln)
            alb = mixc(alb, lin("#7a4423"), np.clip(m * amt, 0, 1))
            chip = (np.hypot(wrapd(x0) / 0.006, (Z - z_top) / 0.004) < 1.0) * amt
            alb = mixc(alb, lin("#5b3a22"), chip)
    for _ in range(gum):
        x0, z0, rad = r.uniform(0, tile), r.uniform(0.35, min(1.0, H * 0.85)), r.uniform(0.006, 0.013)
        d = np.hypot(wrapd(x0), Z - z0) / rad * (0.85 + 0.3 * fine)
        m = 1 - sstep(0.8, 1.0, d)
        alb = mixc(alb, lin("#66645e") * (0.85 + 0.3 * fine[..., None]), m)
        rgh += 0.2 * m
        hgt += 2.5 * m * (1 - d * 0.5)
    for _ in range(stickers):
        x0, z0 = r.uniform(0, tile), r.uniform(0.45, min(1.0, H * 0.85))
        w, h = r.uniform(0.03, 0.07), r.uniform(0.02, 0.05)
        inside = (wrapd(x0) < w / 2) * (np.abs(Z - z0) < h / 2)
        torn = sstep(0.4, 0.6, noise((Hp, W), (cu * 30, cv * 30), r, 3))
        m = inside * torn
        alb = mixc(alb, lin("#c9c3b2") * (0.8 + 0.3 * fine[..., None]), m * 0.85)
        rgh += 0.25 * m
        hgt += 0.4 * m
    for _ in range(dents):
        x0, z0, rad = r.uniform(0, tile), r.uniform(0.3, H * 0.8), r.uniform(0.05, 0.12)
        d2 = (wrapd(x0) ** 2 + (Z - z0) ** 2) / rad ** 2
        hgt -= np.exp(-d2 * 2.5) * 6.0
    if paint:
        paint(alb, rgh, hgt, X, Z, r)
    nrm = normal_map(hgt, normal_strength)
    return image_material(name, alb, np.clip(rgh, 0.05, 1.0), nrm, metal=metal, cell=(tile, H), base_hex=base_hex or base)


# -- timber ---------------------------------------------------------------------------------------

BOARD_STRIPS = 8


def timber(name, base="#8a6a48", grey="#9a958a", weather=0.5, green=0.0, px=1024, seed=None, even=False, ground_strips=(),
           vary=0.1, knots=(1, 3)):
    """An atlas of BOARD_STRIPS boards, one per horizontal strip: u along the board end to end,
    v round its section (see `board`): top face, front edge, underside, back edge. Flat-sawn grain
    (cathedral arches of growth rings running along u), knots, drying checks; each strip its own
    tone; ends stain darker; the top face weathers grey (every face alike if `even`).
    ground_strips: strips for posts/legs, whose u = 1 end stands in the grass (algae, wet dark)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    r = rng(seed or name)
    W = H = px
    sh = H // BOARD_STRIPS
    alb = np.zeros((H, W, 3))
    rgh = np.zeros((H, W))
    hgt = np.zeros((H, W))
    U = (np.arange(W) + 0.5) / W
    for s in range(BOARD_STRIPS):
        V = (np.arange(sh) + 0.5) / sh
        Uu, Vv = np.meshgrid(U, V)
        # flat-sawn rings: elongated ellipses round a pith that lies off the face
        vc = r.uniform(-0.6, 1.6)
        uc = r.uniform(0.2, 0.8)
        sq = r.uniform(0.08, 0.18)
        warp = noise((sh, W), (5, 1), r, 3) * 0.06 + noise((sh, W), (40, 3), r, 2) * 0.012
        dist = np.sqrt(((Vv - vc) * 0.35) ** 2 + ((Uu - uc) * sq) ** 2) + warp * 0.3
        freq = r.uniform(26, 40)
        ring = 0.5 + 0.5 * np.sin(dist * freq * 2 * math.pi)
        late = sstep(0.6, 0.95, ring)
        fibre = noise((sh, W), (180, 30), r, 2)
        tone = r.uniform(-vary, vary)
        hue = r.uniform(-1, 1)
        b = lin(base) * (1 + tone)
        b = b * (1 + 0.05 * hue * np.array([-1.0, 0.6, -0.4]))       # greener or redder
        c = b * (1 - 0.4 * late[..., None]) * (0.9 + 0.2 * fibre[..., None])
        h = -0.6 * late + 0.25 * fibre
        for _ in range(r.integers(knots[0], knots[1] + 1)):
            ku, kv, kr = r.uniform(0.1, 0.9), r.uniform(0.04, 0.36), r.uniform(0.006, 0.016)
            d = np.hypot((Uu - ku) / (kr * 0.45), (Vv - kv) / (kr * 5))
            k = 1 - sstep(0.6, 1.1, d)
            halo = (1 - sstep(1.0, 3.5, d))
            c = mixc(c, b * 0.32, k * 0.92)
            c = c * (1 - halo[..., None] * 0.18)
            h -= k * 0.8 - halo * 0.3
        chk = sstep(0.93, 0.97, noise((sh, W), (3, 40), r, 2)) * sstep(0.6, 0.8, noise((sh, W), (8, 4), r, 2))
        c = mixc(c, b * 0.22, chk * 0.85)
        h -= chk * 1.5
        top = np.ones_like(Vv) if even else 1 - sstep(0.36, 0.46, Vv)
        under = sstep(0.5, 0.56, Vv) * (1 - sstep(0.86, 0.92, Vv))
        wv = weather * (0.75 + 0.5 * noise((sh, W), (5, 2), r, 4))
        c = mixc(c, lin(grey) * (0.85 + 0.3 * fibre[..., None]), np.clip(top * wv * (0.75 if even else 1.0), 0, 0.9))
        if not even:
            c = mixc(c, c * 0.85 + lin(grey) * 0.15, np.clip((1 - top) * wv * 0.4, 0, 1))
        if green > 0:
            alg = under * sstep(0.4, 0.75, noise((sh, W), (12, 3), r, 4)) * green
            c = mixc(c, lin("#4c5a2c"), alg)
        ends = np.maximum(1 - sstep(0.0, 0.06, Uu), sstep(0.94, 1.0, Uu))
        ends *= 0.6 + 0.4 * noise((sh, W), (40, 4), r, 3)
        c = mixc(c, c * 0.55, ends * 0.8)
        if s in ground_strips:
            foot = sstep(0.86, 0.97, Uu) * (0.7 + 0.3 * noise((sh, W), (30, 4), r, 3))
            c = mixc(c, lin("#3f4a26"), foot * 0.75)
            c = mixc(c, c * 0.5, sstep(0.95, 1.0, Uu))
        sl = slice(s * sh, (s + 1) * sh)
        alb[sl] = c
        rgh[sl] = 0.72 + 0.12 * top * weather - 0.05 * fibre + 0.1 * ends
        hgt[sl] = h
    nrm = normal_map(hgt, 1.5)
    return image_material(name, alb, rgh, nrm)


def end_grain(name, base="#7a5a3c", px=256):
    """Sawn end of a board: rings round an off-centre pith, radial checks, dark and rough."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    r = rng(name)
    X, Y = np.meshgrid((np.arange(px) + 0.5) / px, (np.arange(px) + 0.5) / px)
    d = np.hypot(X + 0.6, Y - 1.4) * 9
    ring = 0.5 + 0.5 * np.sin((d + noise((px, px), (6, 6), r, 3) * 0.6) * 2 * math.pi)
    c = lin(base) * 0.62 * (1 - 0.3 * sstep(0.5, 0.9, ring))[..., None]
    c *= (0.9 + 0.2 * noise((px, px), (40, 40), r, 2))[..., None]
    h = -ring * 0.6
    # radial checks from the pith side, a darker wet margin
    ang = np.arctan2(Y - 1.4, X + 0.6)
    for _ in range(3):
        a0 = r.uniform(-1.25, -0.35)
        crack = (np.abs(ang - a0) * d < 0.06) * (d > r.uniform(9.5, 11.5))
        c = mixc(c, c * 0.25, crack * 0.9)
        h -= crack * 2.0
    edge = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    c = mixc(c, c * 0.7, 1 - sstep(0.0, 0.12, edge))
    return image_material(name, c, 0.85, normal_map(h, 1.0))


# -- labels ---------------------------------------------------------------------------------------

def card(name, w_m, h_m, draw, ppm=1500, bg="#ffffff", rough=0.45, metal=0.0, max_px=1024, alpha=False, emission=None):
    """A painted card w x h metres: draw(c) paints into c.alb (H, W, 3) / c.alpha with helpers;
    1500 px per metre unless capped at max_px."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    k = min(ppm, max_px / max(w_m, h_m))
    W, H = max(8, int(round(w_m * k))), max(8, int(round(h_m * k)))
    c = Card(W, H, k, bg, name)
    draw(c)
    return image_material(name, c.alb, c.rough if c.rough is not None else rough, normal_map(c.hgt, 1.0) if c.hgt.any() else None,
                          metal=metal, alpha=c.alpha if alpha else None, extension="CLIP", emission=emission)


class Card:
    """A painting surface in metres: (0, 0) bottom-left, (w, h) top-right."""

    def __init__(self, W, H, ppm, bg, name):
        self.W, self.H, self.k = W, H, ppm
        self.alb = np.broadcast_to(lin(bg), (H, W, 3)).copy()
        self.alpha = np.zeros((H, W))
        self.hgt = np.zeros((H, W))
        self.rough = None
        self.r = rng(name)

    def px(self, v):
        return v * self.k

    def fill(self, mask, col, a=1.0):
        self.alb = mixc(self.alb, lin(col) if isinstance(col, str) else col, mask * a)
        self.alpha = np.maximum(self.alpha, mask * a)

    def poly(self, pts, col, a=1.0):
        """Polygon(s) in metres (a list of point lists, even-odd)."""
        if pts and not isinstance(pts[0][0], (list, tuple)):
            pts = [pts]
        m = poly_mask(self.H, self.W, [[(x * self.k, y * self.k) for x, y in p] for p in pts])
        self.fill(m, col, a)
        return m

    def text(self, s, x, y, cap, col, align="c", bold=True, max_w=None, a=1.0):
        m = text_mask(self.H, self.W, s, x * self.k, y * self.k, cap * self.k, align, bold, max_w * self.k if max_w else None)
        self.fill(m, col, a)
        return m

    def grime(self, amt=0.3, col="#5a5246", scale=6):
        n = noise((self.H, self.W), (scale, max(1, int(scale * self.H / self.W))), self.r, 4)
        self.alb = mixc(self.alb, lin(col), sstep(0.45, 0.85, n) * amt)

    def age(self, fade=0.15, dirt=0.15):
        """Sun-faded, dusty print: colours pulled towards a warm grey, dirt in the lower part."""
        n = noise((self.H, self.W), (4, 4), self.r, 4)
        self.alb = mixc(self.alb, lin("#b9b6ad"), fade * (0.5 + n))
        Y = (np.arange(self.H) + 0.5)[:, None] / self.H
        self.alb = mixc(self.alb, lin("#6a6355"), np.clip(dirt * (1 - Y) * (0.4 + n), 0, 1))


# -- pictograms (unit square, y up) ---------------------------------------------------------------

def picto_bottle(cx, cy, s):
    """A wine bottle in outline, height s, centred at (cx, cy)."""
    pts = [(-0.18, -0.5), (0.18, -0.5), (0.2, -0.47), (0.2, 0.08), (0.17, 0.16), (0.08, 0.24), (0.07, 0.42), (0.085, 0.44),
           (0.085, 0.5), (-0.085, 0.5), (-0.085, 0.44), (-0.07, 0.42), (-0.08, 0.24), (-0.17, 0.16), (-0.2, 0.08), (-0.2, -0.47)]
    return [(cx + x * s, cy + y * s) for x, y in pts]


def picto_can(cx, cy, s):
    w, h = 0.2 * s, 0.34 * s
    return rect(cx - w, cy - h, cx + w, cy + h, 0.04 * s)


def picto_paper(cx, cy, s):
    """Two overlapping sheets, the top one with text lines (as holes)."""
    back = [(cx - 0.25 * s, cy - 0.32 * s), (cx + 0.18 * s, cy - 0.38 * s), (cx + 0.28 * s, cy + 0.36 * s), (cx - 0.15 * s, cy + 0.42 * s)]
    front = rect(cx - 0.3 * s, cy - 0.42 * s, cx + 0.22 * s, cy + 0.34 * s)
    lines = [rect(cx - 0.22 * s, cy + (0.18 - 0.11 * i) * s, cx + 0.14 * s, cy + (0.22 - 0.11 * i) * s) for i in range(5)]
    return back, [front] + lines


def picto_bag(cx, cy, s):
    """A tied refuse sack."""
    pts = []
    for k in range(40):
        a = 2 * math.pi * k / 40
        rr = 0.3 * (1 + 0.08 * math.sin(5 * a))
        pts.append((cx + rr * s * math.cos(a), cy - 0.08 * s + rr * 1.05 * s * math.sin(a)))
    knot = [(cx - 0.06 * s, cy + 0.2 * s), (cx + 0.06 * s, cy + 0.2 * s), (cx + 0.16 * s, cy + 0.46 * s), (cx + 0.02 * s, cy + 0.32 * s),
            (cx - 0.02 * s, cy + 0.32 * s), (cx - 0.16 * s, cy + 0.46 * s)]
    return [pts, knot]


def picto_recycle(cx, cy, s):
    """The three chasing arrows (Mobius loop), as three bent arrow polygons."""
    out = []
    for k in range(3):
        a0 = math.radians(90 + 120 * k)
        pts_o, pts_i = [], []
        for j in range(9):
            a = a0 + math.radians(15 + 70 * j / 8)
            pts_o.append((cx + 0.42 * s * math.cos(a), cy + 0.42 * s * math.sin(a)))
            pts_i.append((cx + 0.27 * s * math.cos(a), cy + 0.27 * s * math.sin(a)))
        a = a0 + math.radians(85)
        tip = (cx + 0.345 * s * math.cos(a + 0.42), cy + 0.345 * s * math.sin(a + 0.42))
        wo = (cx + 0.49 * s * math.cos(a), cy + 0.49 * s * math.sin(a))
        wi = (cx + 0.2 * s * math.cos(a), cy + 0.2 * s * math.sin(a))
        out.append(pts_o + [wo, tip, wi] + pts_i[::-1])
    return out


# -- mesh helpers with UVs (metres) ---------------------------------------------------------------

def ring_pt(r, a, z, c=(0, 0, 0), sy=1.0, flutes=0, depth=0.0):
    f = r * (1 - depth * (0.5 - 0.5 * math.cos(flutes * a))) if flutes and depth else r
    return (c[0] + f * math.cos(a), c[1] + f * sy * math.sin(a), c[2] + z)


def lathe(p, prof, mat, sides=48, c=(0, 0, 0), flutes=0, depth=0.0, mats=None, skip=None, a0=0.0, uv_r=None, sy=1.0, uv_xy=False):
    """Surface of revolution of (r, z) with UVs (arc length, z): no box-projection seams.
    depth: flute depth, or a list with one per profile point; sy squashes it along y (ovals);
    skip(j, k) -> True leaves that face out (openings); mats(j, k) -> material."""
    rr = []
    for j, (r, z) in enumerate(prof):
        dj = depth[j] if isinstance(depth, (list, tuple)) else depth
        ring = []
        for k in range(sides + 1):
            a = a0 + 2 * math.pi * (k % sides) / sides
            ring.append(ring_pt(r, a, z, c, sy, flutes, dj))
        rr.append(ring)
    R = uv_r or max(r for r, _ in prof) or 0.1
    for j in range(len(prof) - 1):
        for k in range(sides):
            if skip and skip(j, k):
                continue
            u0, u1 = R * 2 * math.pi * k / sides, R * 2 * math.pi * (k + 1) / sides
            z0, z1 = prof[j][1], prof[j + 1][1]
            # flat discs (z0 == z1) spread v by radius so they are not a single texture row
            if abs(z1 - z0) < 1e-6:
                v0, v1 = z0 + prof[j][0] * 0.0, z1
            else:
                v0, v1 = z0, z1
            pts = [rr[j][k], rr[j][k + 1], rr[j + 1][k + 1], rr[j + 1][k]]
            uv = [(q[0], q[1]) for q in pts] if uv_xy else [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            p.face(pts, mats(j, k) if mats else mat, uv)


def board(p, a, b, up, w, t, mat, end, strip=0, u_off=0.0, flip=False, floor=None):
    """A sawn board from a to b (centre line), its broad face towards `up`, w wide, t thick.
    UVs: u 0-1 along the board (the whole atlas width); v round the section inside its strip:
    top 0-0.4, front edge 0.4-0.5, underside 0.5-0.9, back edge 0.9-1. end: material of the
    sawn ends (UV 0-1 over the end)."""
    a, b, up = (np.asarray(x, float) for x in (a, b, up))
    d = b - a
    L = np.linalg.norm(d)
    d /= L
    n = up - d * (up @ d)
    n /= np.linalg.norm(n)
    s = np.cross(d, n)                 # across the width
    hw, ht = w / 2, t / 2
    v0 = (strip % BOARD_STRIPS) / BOARD_STRIPS
    vs = 1.0 / BOARD_STRIPS

    def V(e, i, j):                    # e: 0 start / 1 end, i: -1/+1 across, j: -1/+1 thickness
        q = (a if e == 0 else b) + s * (i * hw) + n * (j * ht)
        if e == 1 and floor is not None and abs(d[2]) > 1e-3:      # sole cut level on the ground
            q = q + d * ((floor - q[2]) / d[2])
        return q
    u0, u1 = (0.998, 0.002) if flip else (0.002, 0.998)
    vv = lambda f: v0 + vs * (0.03 + 0.94 * f)
    # top (+n), front edge (+s), underside (-n), back edge (-s)
    p.face([V(0, -1, 1), V(0, 1, 1), V(1, 1, 1), V(1, -1, 1)], mat, [(u0, vv(0.0)), (u0, vv(0.4)), (u1, vv(0.4)), (u1, vv(0.0))][::1])
    p.face([V(0, 1, 1), V(0, 1, -1), V(1, 1, -1), V(1, 1, 1)], mat, [(u0, vv(0.4)), (u0, vv(0.5)), (u1, vv(0.5)), (u1, vv(0.4))])
    p.face([V(0, 1, -1), V(0, -1, -1), V(1, -1, -1), V(1, 1, -1)], mat, [(u0, vv(0.5)), (u0, vv(0.9)), (u1, vv(0.9)), (u1, vv(0.5))])
    p.face([V(0, -1, -1), V(0, -1, 1), V(1, -1, 1), V(1, -1, -1)], mat, [(u0, vv(0.9)), (u0, vv(1.0)), (u1, vv(1.0)), (u1, vv(0.9))])
    p.face([V(0, -1, -1), V(0, 1, -1), V(0, 1, 1), V(0, -1, 1)], end, [(0, 0), (1, 0), (1, 1), (0, 1)])
    p.face([V(1, -1, 1), V(1, 1, 1), V(1, 1, -1), V(1, -1, -1)], end, [(0, 1), (1, 1), (1, 0), (0, 0)])
    return s, n


def holed_panel(p, o, eu, ev, w, h, hole, depth, mat, jamb, back, uv_scale=1.0):
    """A flat rectangle (origin o its bottom-left, edges eu (w) and ev (h) unit vectors, facing
    eu x ev) with a rectangular opening hole = (u0, v0, u1, v1) going `depth` into the solid:
    8 faces round the opening, 4 jamb faces, a back face."""
    o, eu, ev = (np.asarray(x, float) for x in (o, eu, ev))
    nrm = np.cross(eu, ev)
    nrm /= np.linalg.norm(nrm)
    us = [0.0, hole[0], hole[2], w]
    vs = [0.0, hole[1], hole[3], h]
    P = lambda u, v, d=0.0: o + eu * u + ev * v - nrm * d
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            q = [P(us[i], vs[j]), P(us[i + 1], vs[j]), P(us[i + 1], vs[j + 1]), P(us[i], vs[j + 1])]
            p.face(q, mat, [(us[i] * uv_scale, vs[j] * uv_scale), (us[i + 1] * uv_scale, vs[j] * uv_scale),
                            (us[i + 1] * uv_scale, vs[j + 1] * uv_scale), (us[i] * uv_scale, vs[j + 1] * uv_scale)])
    u0, v0, u1, v1 = hole
    c = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    for k in range(4):
        (ua, va), (ub, vb) = c[k], c[(k + 1) % 4]
        p.face([P(ub, vb), P(ub, vb, depth), P(ua, va, depth), P(ua, va)], jamb)
    p.face([P(u0, v0, depth), P(u1, v0, depth), P(u1, v1, depth), P(u0, v1, depth)], back)


def wrap_decal(p, r, z0, z1, a_mid, width, mat, sides=12, c=(0, 0), lift=0.002, flutes=0, depth=0.0, sy=1.0):
    """A label hugging a vertical cylinder (radius r, flutes and y squash as in `lathe`): centred
    at angle a_mid (radians, 0 = +x, pi/2 = +y), `width` metres of arc, z0 to z1; UV 0-1."""
    half = width / (2 * r * (sy if abs(math.sin(a_mid)) > 0.7 else 1.0))
    saved, p.r = p.r, 0.0

    def P(a, z):
        f = r * (1 - depth * (0.5 - 0.5 * math.cos(flutes * a))) if flutes else r
        return (c[0] + (f + lift) * math.cos(a), c[1] + (f * sy + lift) * math.sin(a), z)
    for k in range(sides):
        t0, t1 = k / sides, (k + 1) / sides
        a0, a1 = a_mid - half * (1 - 2 * t0), a_mid - half * (1 - 2 * t1)
        p.face([P(a0, z0), P(a1, z0), P(a1, z1), P(a0, z1)], mat, [(t0, 0), (t1, 0), (t1, 1), (t0, 1)])
    p.r = saved


def bolt_head(p, c, n, r, mat, h=None, sides=10, hexa=False):
    """A domed (carriage) bolt head, or a hex nut with a washer, sitting on a face with normal n."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    n /= np.linalg.norm(n)
    h = h or r * 0.45
    saved, p.r = p.r, 0.0
    if hexa:
        p.cyl(tuple(c - n * 0.001), tuple(c + n * 0.0025), r * 1.45, mat, 8)
        p.cyl(tuple(c + n * 0.0025), tuple(c + n * (0.0025 + r * 0.8)), r, mat, 6)
        p.cyl(tuple(c + n * (0.0025 + r * 0.8)), tuple(c + n * (0.0025 + r * 1.2)), r * 0.55, mat, 6)
    else:
        e1 = np.cross(n, (0, 0, 1) if abs(n[2]) < 0.9 else (1, 0, 0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rows = 3
        ang = [0.0, 0.45, 0.9, 1.25]
        ring = lambda t: [c + n * (h * math.sin(t) - 0.001) + (e1 * math.cos(2 * math.pi * k / sides) + e2 * math.sin(2 * math.pi * k / sides)) * r * math.cos(t) for k in range(sides)]
        R = [ring(t) for t in ang]
        for j in range(rows):
            for k in range(sides):
                k2 = (k + 1) % sides
                p.face([R[j][k], R[j][k2], R[j + 1][k2], R[j + 1][k]], mat)
        p.face(R[-1], mat)
    p.r = saved


def spline(pts, n=6):
    """A Catmull-Rom curve through pts (end points kept), n samples per span."""
    P = [np.asarray(q, float) for q in pts]
    P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return out


def sweep(p, path, prof, mat, cap=True, normal=None, closed=False, mats=None):
    """A closed profile [(u, v)] swept along path with UVs (arc length + perimeter, z): finishes
    laid out by height read right on rails and frames. normal: a fixed axis for the profile's u
    (e.g. (1, 0, 0) for a frame drawn in the yz plane), else parallel transport from +z."""
    P = np.asarray(path, float)
    n = len(P)
    T = [P[(i + 1) % n] - P[i - 1] if closed else P[min(i + 1, n - 1)] - P[max(i - 1, 0)] for i in range(n)]
    T = [t / (np.linalg.norm(t) + 1e-12) for t in T]
    if normal is not None:
        N0 = np.asarray(normal, float)
    else:
        ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.95 else np.array([1.0, 0, 0])
        N0 = np.cross(ref, T[0])
    N = []
    prev = N0
    for i in range(n):
        v = prev - T[i] * (prev @ T[i])
        v /= np.linalg.norm(v) + 1e-12
        N.append(v)
        prev = v
    m = len(prof)
    per = np.concatenate([[0], np.cumsum([math.dist(prof[j], prof[(j + 1) % m]) for j in range(m)])])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    G = []
    for i in range(n):
        B = np.cross(T[i], N[i])
        G.append([P[i] + N[i] * u + B * v for u, v in prof])
    for i in range(n if closed else n - 1):
        i2 = (i + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            si, s2 = s[i], s[i2] if i2 else s[-1] + math.dist(P[-1], P[0])
            q = [G[i][j], G[i][j2], G[i2][j2], G[i2][j]]
            k0, k1 = 0.4 * per[j], 0.4 * per[j + 1]
            uv = [(si, q[0][2] + k0), (si, q[1][2] + k1), (s2, q[2][2] + k1), (s2, q[3][2] + k0)]
            p.face(q, mats(i, j) if mats else mat, uv)
    if cap and not closed:
        p.face([G[0][j] for j in range(m - 1, -1, -1)], mat)
        p.face([G[-1][j] for j in range(m)], mat)


def rrect2(w, h, r, n=3):
    """A rounded rectangle profile (u across w, v across h), counter-clockwise."""
    return _rr2(w, h, r, n)


def _rr2(w, h, r, n=3):
    r = min(r, w / 2 - 1e-4, h / 2 - 1e-4)
    out = []
    for qx, qy, a0 in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        cx, cy = qx * (w / 2 - r), qy * (h / 2 - r)
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def curved_panel(p, r, a_mid, width, z0, z1, t, mat, sides=8, c=(0, 0), sy=1.0):
    """A plate t thick standing proud of a cylinder of radius r: a door, a number plate."""
    half = width / (2 * r)
    A = [a_mid - half + 2 * half * k / sides for k in range(sides + 1)]
    O = lambda a, z: ring_pt(r + t, a, z, (c[0], c[1], 0), sy)
    I = lambda a, z: ring_pt(r - 0.002, a, z, (c[0], c[1], 0), sy)
    for k in range(sides):
        a, b = A[k], A[k + 1]
        p.face([O(a, z0), O(b, z0), O(b, z1), O(a, z1)], mat, [(r * a, z0), (r * b, z0), (r * b, z1), (r * a, z1)])
        p.face([I(a, z0), I(b, z0), O(b, z0), O(a, z0)], mat)
        p.face([O(a, z1), O(b, z1), I(b, z1), I(a, z1)], mat)
    p.face([I(A[0], z0), O(A[0], z0), O(A[0], z1), I(A[0], z1)], mat)
    p.face([O(A[-1], z0), I(A[-1], z0), I(A[-1], z1), O(A[-1], z1)], mat)


def top_uv(p):
    """Faces facing up or down without UVs read the skin row at their own height (u along x + y),
    so a lid or a ledge is not painted with whatever the box projection lands on."""
    for i, f in enumerate(p.F):
        if p.uv[i] is not None:
            continue
        P = np.array([p.V[k] for k in f])
        n = np.cross(P[1] - P[0], P[2] - P[0])
        ln = np.linalg.norm(n)
        if ln < 1e-12 or abs(n[2]) / ln < 0.7:
            continue
        p.uv[i] = [(q[0] + 0.37 * q[1], q[2] + 0.015 * q[1]) for q in P]


def block_paving(name="Street_BlockPaving", tile=2.0, px=1024):
    """Concrete block paving, 200 x 100 mm in stretcher bond (rows along x), brindle charcoal and
    grey blocks each its own tone, sanded joints with moss in places, chipped arrises, a few oil
    and dirt stains. One repeat per `tile` metres (faces give metric UVs)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    r = rng(name)
    k = px / tile
    X, Y = np.meshgrid((np.arange(px) + 0.5) / k, (np.arange(px) + 0.5) / k)
    bw, bh = 0.2, 0.1
    row = np.floor(Y / bh)
    xs = X + (row % 2) * bw / 2
    col = np.floor(xs / bw) % int(round(tile / bw))
    fx, fy = (xs % bw) / bw, (Y % bh) / bh
    edge = np.minimum(np.minimum(fx, 1 - fx) * bw, np.minimum(fy, 1 - fy) * bh)
    joint = 1 - sstep(0.0015, 0.0035, edge)
    nr, nc = int(round(tile / bh)), int(round(tile / bw))
    ri, ci = (row % nr).astype(int), col.astype(int) % nc
    t = r.uniform(-1, 1, (nr, nc))[ri, ci]
    redish = (r.uniform(0, 1, (nr, nc)) > 0.78)[ri, ci]
    base = np.where(redish[..., None], lin("#65534a"), lin("#555452"))
    alb = base * (1 + 0.16 * t[..., None])
    grit = noise((px, px), (int(tile / 0.004), int(tile / 0.004)), r, 1)
    alb *= (0.88 + 0.24 * grit[..., None])
    alb = mixc(alb, lin("#3c3a33"), joint * 0.85)
    moss = joint * sstep(0.55, 0.75, noise((px, px), (12, 12), r, 4))
    alb = mixc(alb, lin("#3f4a24"), moss * 0.8)
    stain = sstep(0.62, 0.85, noise((px, px), (4, 4), r, 5))
    alb = mixc(alb, alb * 0.6, stain * 0.5)
    hgt = -joint * 3.0 + grit * 0.3 - (1 - sstep(0.0, 0.006, edge)) * 0.8
    rgh = 0.82 + 0.08 * grit - 0.15 * stain
    return image_material(name, alb, rgh, normal_map(hgt, 1.0), cell=(tile, tile))


def pad(p, x0, y0, x1, y1, h, top, side, kerb=0.05, kerb_mat=None):
    """A paved pad standing h above the lawn: paving inside a concrete edging course."""
    kerb_mat = kerb_mat or side
    saved, p.r = p.r, 0.0
    ix0, iy0, ix1, iy1 = x0 + kerb, y0 + kerb, x1 - kerb, y1 - kerb
    p.face([(ix0, iy0, h), (ix1, iy0, h), (ix1, iy1, h), (ix0, iy1, h)], top, [(ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1)])
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    inner = [(ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1)]
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        ia, ib = inner[i], inner[(i + 1) % 4]
        p.face([(a[0], a[1], h), (b[0], b[1], h), (ib[0], ib[1], h), (ia[0], ia[1], h)], kerb_mat)
    p.r = 0.008
    for i in range(4):
        a, b = ring[i], ring[(i + 1) % 4]
        p.face([(a[0], a[1], -0.01), (b[0], b[1], -0.01), (b[0], b[1], h), (a[0], a[1], h)], side)
    p.r = saved


def concrete_top(name="Street_ConcreteTop", tile=1.0, px=512, base="#8f8b83"):
    """Cast concrete seen from above: aggregate speckle, trowel marks, a few hairline cracks, a
    darker weathered margin and grass-stain splashes (metric UVs, one repeat per `tile`)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    r = rng(name)
    alb = np.broadcast_to(lin(base), (px, px, 3)).copy()
    agg = noise((px, px), (int(tile / 0.006),) * 2, r, 1)
    blot = noise((px, px), (4, 4), r, 4)
    alb *= (0.86 + 0.22 * agg[..., None]) * (0.9 + 0.2 * blot[..., None])
    from .foliage_paint import worley
    f1, edge = worley((px, px), (3, 3), r)
    crack = (1 - sstep(0.0, 0.025, edge)) * sstep(0.55, 0.7, noise((px, px), (6, 6), r, 3))
    alb = mixc(alb, alb * 0.35, crack * 0.9)
    stain = sstep(0.6, 0.85, noise((px, px), (8, 8), r, 4))
    alb = mixc(alb, lin("#5d5a4a"), stain * 0.35)
    hgt = agg * 0.6 - crack * 3.0
    return image_material(name, alb, 0.88 - 0.1 * stain, normal_map(hgt, 1.0), cell=(tile, tile))
