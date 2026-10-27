"""Painted textures for the construction-site props (props_site): numpy canvases in metres, crisp
anti-aliased lettering from real outline fonts, simple vector shapes (signs, pictograms, logos),
weathering masks, and image materials (albedo, roughness/metal, normal; recolourable TINT_ detail).

Conventions
- A canvas is an (H, W, C) float array, row 0 = the TOP of the picture, colours linear.
- Atlas materials carry `uv_cell` (metres per texture repeat), so faces are given UVs in metres
  and atlas regions are laid out in "metre space": u to the right, v up from the bottom edge.
- TINT_ materials hold a greyscale detail (paint shading, dirt, streaks) times a colour factor and
  the gain the editor needs to recolour them (as the game bake does for procedural TINT_ surfaces);
  anything that must keep its own colour (rust, stencils, labels) lives in an overlay material.
"""
import math
import os

import numpy as np


# -- colour -----------------------------------------------------------------------------------------

def lin(h):
    """sRGB hex -> linear RGB array."""
    h = h.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lum(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# -- noise ------------------------------------------------------------------------------------------

def vnoise(H, W, cx, cy, seed):
    """Smooth periodic value noise in 0-1 over cx x cy lattice cells."""
    rng = np.random.default_rng(seed)
    cx, cy = max(1, int(cx)), max(1, int(cy))
    lat = rng.random((cy, cx))
    x = (np.arange(W) + 0.5) / W * cx
    y = (np.arange(H) + 0.5) / H * cy
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0 %= cx
    y0 %= cy
    x1, y1 = (x0 + 1) % cx, (y0 + 1) % cy
    a, b = lat[np.ix_(y0, x0)], lat[np.ix_(y0, x1)]
    c, d = lat[np.ix_(y1, x0)], lat[np.ix_(y1, x1)]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(H, W, cx, cy, seed, octaves=4, gain=0.5):
    out, amp, tot = np.zeros((H, W)), 1.0, 0.0
    for o in range(octaves):
        out += amp * vnoise(H, W, cx * 2 ** o, cy * 2 ** o, seed + 101 * o)
        tot += amp
        amp *= gain
    return out / tot


def blur(a, r):
    """Box blur (twice: close to a gaussian) with radius r pixels, wrapping."""
    r = int(r)
    if r < 1:
        return a
    for _ in range(2):
        for ax in (0, 1):
            c = np.cumsum(np.concatenate([np.take(a, range(-r - 1, 0), ax), a, np.take(a, range(0, r), ax)], ax), ax)
            n = a.shape[ax]
            a = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), ax) - np.take(c, range(0, n), ax)) / (2 * r + 1)
    return a


def streaks(H, W, seed, density=0.5, length=0.5, width_px=3):
    """Vertical run marks hanging from the top edge (rain, rust bleed): 0-1, row 0 = top."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W))
    n = max(1, int(W * density / max(1, width_px * 2)))
    ys = np.arange(H)[:, None]
    for _ in range(n):
        x = rng.uniform(0, W)
        L = rng.uniform(0.3, 1.0) * length * H
        y0 = rng.uniform(0, 0.15) * H
        w = width_px * rng.uniform(0.6, 1.6)
        xs = np.arange(W)[None, :]
        dx = np.minimum(np.abs(xs - x), W - np.abs(xs - x))
        prof = np.clip(1 - dx / w, 0, 1)
        fall = np.clip(1 - (ys - y0) / L, 0, 1) * (ys >= y0)
        out = np.maximum(out, prof * fall ** 1.5 * rng.uniform(0.4, 1.0))
    wob = vnoise(H, W, max(2, W // 24), max(2, H // 24), seed + 7)
    return out * (0.6 + 0.4 * wob)


# -- raster shapes (anti-aliased by supersampling) ---------------------------------------------------

def _down(M, ss):
    H, W = M.shape[0] // ss, M.shape[1] // ss
    return M[:H * ss, :W * ss].reshape(H, ss, W, ss).mean((1, 3))


def poly_mask(H, W, pts, ss=4):
    """Coverage of a polygon (x right, y down, pixels; even-odd fill) on an H x W grid."""
    P = np.asarray(pts, float) * ss
    x0, y0 = np.floor(P.min(0)).astype(int)
    x1, y1 = np.ceil(P.max(0)).astype(int)
    x0, y0 = max(0, x0 - ss), max(0, y0 - ss)
    x1, y1 = min(W * ss, x1 + ss), min(H * ss, y1 + ss)
    out = np.zeros((H, W))
    if x1 <= x0 or y1 <= y0:
        return out
    # snap the box to whole output pixels
    x0, y0 = x0 // ss * ss, y0 // ss * ss
    x1, y1 = -(-x1 // ss) * ss, -(-y1 // ss) * ss
    X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.zeros(X.shape, bool)
    n = len(P)
    for i in range(n):
        (ax, ay), (bx, by) = P[i], P[(i + 1) % n]
        if ay == by:
            continue
        cond = (ay > Y) != (by > Y)
        xc = ax + (Y - ay) * (bx - ax) / (by - ay)
        inside ^= cond & (X < xc)
    out[y0 // ss:y1 // ss, x0 // ss:x1 // ss] = _down(inside.astype(float), ss)
    return out


def circle_pts(cx, cy, r, n=48, ry=None, a0=0.0, a1=2 * math.pi):
    ry = r if ry is None else ry
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + ry * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + (0 if a1 - a0 >= 2 * math.pi - 1e-6 else 1))]


def rect_mask(H, W, x0, y0, x1, y1):
    """Coverage of an axis-aligned rectangle (pixels, fractional edges)."""
    xs = np.arange(W) + 0.0
    ys = np.arange(H) + 0.0
    cx = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    cy = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
    return cy[:, None] * cx[None, :]


def rrect_pts(x0, y0, x1, y1, r, n=6):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    out = []
    for cx, cy, a in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            t = math.radians(a + 90 * k / n)
            out.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return out


def put(img, mask, color, a=1.0):
    """Composite a colour through a coverage mask (in place). img (H,W,3) or (H,W,4): RGBA gets
    its alpha raised to the coverage."""
    m = np.clip(mask * a, 0, 1)[..., None]
    c = np.asarray(color, float)
    img[..., :3] = img[..., :3] * (1 - m) + c[:3] * m
    if img.shape[2] == 4:
        img[..., 3:] = np.maximum(img[..., 3:], m)
    return img


def put_at(img, mask, x, y, color, a=1.0):
    """Composite a small mask with its top-left at (x, y) pixels (clipped)."""
    H, W = img.shape[:2]
    h, w = mask.shape
    x, y = int(round(x)), int(round(y))
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return img
    sub = img[dy0:dy1, dx0:dx1]
    put(sub, mask[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0], color, a)
    return img


# -- lettering from outline fonts ---------------------------------------------------------------------

_FONTS = {
    "bold": ["Arial Bold.ttf", "Verdana Bold.ttf"],
    "regular": ["Arial.ttf", "Verdana.ttf"],
    "narrow": ["Arial Narrow Bold.ttf", "Arial Bold.ttf"],
    "din": ["DIN Condensed Bold.ttf", "Arial Narrow Bold.ttf"],
    "black": ["Arial Black.ttf", "Arial Bold.ttf"],
    "impact": ["Impact.ttf", "Arial Black.ttf"],
}
_FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts",
              "/usr/share/fonts/truetype/msttcorefonts", "/usr/share/fonts/truetype/dejavu"]
_TRIS = {}


def _font(kind):
    import bpy
    for f in _FONTS.get(kind, []):
        for d in _FONT_DIRS:
            path = os.path.join(d, f)
            if os.path.exists(path):
                return bpy.data.fonts.load(path, check_existing=True)
    return None          # Blender's built-in face (made bolder when drawn)


def _tris(text, kind):
    """Triangles of `text` set in the font at size 1 (x right, y up, baseline y = 0)."""
    key = (text, kind)
    if key in _TRIS:
        return _TRIS[key]
    import bmesh
    import bpy
    cu = bpy.data.curves.new("_sp_text", "FONT")
    f = _font(kind)
    if f:
        cu.font = f
    cu.body = text
    cu.size = 1.0
    cu.resolution_u = 5
    ob = bpy.data.objects.new("_sp_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    T = np.array([[tuple(v.co)[:2] for v in fc.verts] for fc in bm.faces], float).reshape(-1, 3, 2)
    bm.free()
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    _TRIS[key] = (T, f is None)
    return _TRIS[key]


def _cap(kind):
    T, _ = _tris("H", kind)
    return float(T[..., 1].max()) if len(T) else 0.7


def text_mask(text, cap_px, kind="bold", ss=4):
    """Coverage of a line of text with capitals cap_px tall. Returns (mask, cap_top_row): the mask
    spans the ink (descenders included); cap_top_row is the row of the capital tops in it."""
    T, builtin = _tris(text, kind)
    if not len(T):
        return np.zeros((1, 1)), 0
    s = cap_px * ss / _cap(kind)
    xmin, ymin = T[..., 0].min(), min(0.0, T[..., 1].min())
    ymax = max(_cap(kind), T[..., 1].max())
    pad = 2 * ss
    W = int(math.ceil((T[..., 0].max() - xmin) * s)) + 2 * pad
    H = int(math.ceil((ymax - ymin) * s)) + 2 * pad
    W, H = -(-W // ss) * ss, -(-H // ss) * ss
    P = np.empty_like(T)
    P[..., 0] = (T[..., 0] - xmin) * s + pad
    P[..., 1] = (ymax - T[..., 1]) * s + pad
    M = np.zeros((H, W), bool)
    for t in P:
        x0, y0 = np.floor(t.min(0)).astype(int)
        x1, y1 = np.ceil(t.max(0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        if x1 <= x0 or y1 <= y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = t
        e0 = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
        e1 = (cx - bx) * (Y - by) - (cy - by) * (X - bx)
        e2 = (ax - cx) * (Y - cy) - (ay - cy) * (X - cx)
        M[y0:y1, x0:x1] |= ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
    m = _down(M.astype(float), ss)
    if builtin:          # the built-in face is a light weight: embolden a touch
        m = np.maximum(m, np.maximum(np.roll(m, 1, 1), np.roll(m, -1, 1)) * 0.8)
    return m, int(round(((ymax - _cap(kind)) * s + pad) / ss))


def text(img, s, x, y, cap_px, color, kind="bold", align="l", max_w=None, a=1.0):
    """Draw text with its capitals' top-left (align l), top-centre (c) or top-right (r) at (x, y);
    shrunk to max_w pixels wide if longer. Returns the drawn width."""
    m, top = text_mask(s, cap_px, kind)
    w = m.shape[1]
    if max_w and w > max_w:
        m, top = text_mask(s, cap_px * max_w / w, kind)
        w = m.shape[1]
    ox = {"l": 0, "c": -w / 2, "r": -w}[align]
    put_at(img, m, x + ox - 2, y - top, color, a)
    return w


# -- pictograms and signs -------------------------------------------------------------------------------

def sign_mandatory(img, cx, cy, r, kind="hat"):
    """A blue mandatory-action disc (ISO 7010 M-series) with a white pictogram."""
    H, W = img.shape[:2]
    put(img, poly_mask(H, W, circle_pts(cx, cy, r)), lin("#ffffff"))
    put(img, poly_mask(H, W, circle_pts(cx, cy, r * 0.93)), lin("#1f5aa6"))
    wh = lin("#ffffff")
    if kind == "hat":
        put(img, poly_mask(H, W, circle_pts(cx, cy + r * 0.12, r * 0.42, 40, a0=math.pi, a1=2 * math.pi)), wh)
        put(img, poly_mask(H, W, [(cx - r * 0.6, cy + r * 0.1), (cx + r * 0.6, cy + r * 0.1), (cx + r * 0.6, cy + r * 0.2), (cx - r * 0.6, cy + r * 0.2)]), wh)
        put(img, poly_mask(H, W, circle_pts(cx, cy + r * 0.38, r * 0.3, 30, ry=r * 0.16, a0=0, a1=math.pi)), wh)
    elif kind == "boot":
        put(img, poly_mask(H, W, [(cx - r * 0.35, cy - r * 0.55), (cx + r * 0.05, cy - r * 0.55), (cx + r * 0.05, cy + r * 0.1),
                                   (cx + r * 0.55, cy + r * 0.2), (cx + r * 0.6, cy + r * 0.45), (cx - r * 0.4, cy + r * 0.45)]), wh)
    elif kind == "vest":
        put(img, poly_mask(H, W, [(cx - r * 0.45, cy - r * 0.5), (cx - r * 0.15, cy - r * 0.5), (cx, cy - r * 0.1), (cx + r * 0.15, cy - r * 0.5),
                                   (cx + r * 0.45, cy - r * 0.5), (cx + r * 0.5, cy + r * 0.55), (cx - r * 0.5, cy + r * 0.55)]), wh)
        put(img, poly_mask(H, W, [(cx - r * 0.5, cy + r * 0.12), (cx + r * 0.5, cy + r * 0.12), (cx + r * 0.5, cy + r * 0.24), (cx - r * 0.5, cy + r * 0.24)]),
            lin("#1f5aa6"))
    elif kind == "gloves":
        put(img, poly_mask(H, W, rrect_pts(cx - r * 0.3, cy - r * 0.1, cx + r * 0.3, cy + r * 0.55, r * 0.15)), wh)
        for k in range(4):
            x = cx - r * 0.28 + k * r * 0.15
            put(img, poly_mask(H, W, rrect_pts(x, cy - r * 0.55 + abs(k - 1.5) * r * 0.08, x + r * 0.12, cy, r * 0.06)), wh)
        put(img, poly_mask(H, W, rrect_pts(cx + r * 0.25, cy - r * 0.05, cx + r * 0.5, cy + r * 0.15, r * 0.08)), wh)
    elif kind == "eyes":
        for s in (-1, 1):
            put(img, poly_mask(H, W, circle_pts(cx + s * r * 0.3, cy, r * 0.24, 30)), wh)
            put(img, poly_mask(H, W, circle_pts(cx + s * r * 0.3, cy, r * 0.13, 30)), lin("#1f5aa6"))
        put(img, poly_mask(H, W, [(cx - r * 0.08, cy - r * 0.06), (cx + r * 0.08, cy - r * 0.06), (cx + r * 0.08, cy + r * 0.02), (cx - r * 0.08, cy + r * 0.02)]), wh)


def sign_warning(img, cx, cy, r):
    """A yellow warning triangle with a black border and an exclamation mark."""
    H, W = img.shape[:2]
    tri = lambda k: [(cx, cy - r * k), (cx + r * 0.866 * k * 1.15, cy + r * 0.5 * k * 1.15), (cx - r * 0.866 * k * 1.15, cy + r * 0.5 * k * 1.15)]
    put(img, poly_mask(H, W, tri(1.0)), lin("#111111"))
    put(img, poly_mask(H, W, tri(0.72)), lin("#f5c400"))
    put(img, poly_mask(H, W, [(cx - r * 0.07, cy - r * 0.38), (cx + r * 0.07, cy - r * 0.38), (cx + r * 0.045, cy + r * 0.2), (cx - r * 0.045, cy + r * 0.2)]), lin("#111111"))
    put(img, poly_mask(H, W, circle_pts(cx, cy + r * 0.36, r * 0.075, 20)), lin("#111111"))


def sign_prohibit(img, cx, cy, r):
    H, W = img.shape[:2]
    put(img, poly_mask(H, W, circle_pts(cx, cy, r)), lin("#c8102e"))
    put(img, poly_mask(H, W, circle_pts(cx, cy, r * 0.8)), lin("#ffffff"))
    put(img, poly_mask(H, W, [(cx - r * 0.62, cy - r * 0.5), (cx - r * 0.5, cy - r * 0.62), (cx + r * 0.62, cy + r * 0.5), (cx + r * 0.5, cy + r * 0.62)]), lin("#c8102e"))


def hazard_stripes(img, x0, y0, x1, y1, c1="#f5c400", c2="#111111", pitch=40, angle=45):
    H, W = img.shape[:2]
    Y, X = np.mgrid[0:H, 0:W]
    t = math.radians(angle)
    s = ((X * math.cos(t) + Y * math.sin(t)) / pitch) % 1.0
    band = smooth(0.48, 0.52, s) - smooth(0.98, 1.0, s)
    m = rect_mask(H, W, x0, y0, x1, y1)
    put(img, m, lin(c1))
    put(img, m * band, lin(c2))


# -- images and materials -------------------------------------------------------------------------------

def image(name, arr, data=False):
    """A packed Blender image from a top-row-first canvas (linear colour unless data)."""
    import bpy
    a = np.asarray(arr, np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones_like(a[..., :1])], -1)
    a = a[::-1]
    h, w = a.shape[:2]
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, w, h, alpha=True)
    rgb = np.clip(a[..., :3], 0, 1)
    if data:
        img.colorspace_settings.name = "Non-Color"
    else:
        img.colorspace_settings.name = "sRGB"
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    out = np.concatenate([rgb, np.clip(a[..., 3:4], 0, 1)], -1).astype(np.float32)
    img.pixels.foreach_set(out.ravel())
    img.pack()
    return img


def normal_from_height(h, metres_per_px, strength=1.0):
    """Tangent-space normal map (encoded 0-1, top-row-first) from a height canvas in metres."""
    b = h[::-1]                                        # bottom-up: rows follow +v
    du = (np.roll(b, -1, 1) - np.roll(b, 1, 1)) / (2 * metres_per_px)
    dv = (np.roll(b, -1, 0) - np.roll(b, 1, 0)) / (2 * metres_per_px)
    n = np.stack([-du * strength, -dv * strength, np.ones_like(b)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5)[::-1]


def pbr(name, albedo, rough=0.5, metal=0.0, cell=None, normal=None, nstrength=1.0, clip=False, tint=None,
        emission=None, blend=False):
    """An image material. albedo: (H,W,3|4) canvas (alpha cut out at 0.5 when clip). rough / metal:
    numbers or (H,W) canvases. cell: metres per repeat (faces give UVs in metres). tint: an sRGB hex
    making it recolourable: albedo is then a greyscale shade whose mean is about 1 (paint = 1,
    dirt < 1) and the colour comes from a factor the editor can change."""
    import bpy
    from ..core.nodes import material
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]

    def tex(img):
        t = nb.node("ShaderNodeTexImage", image=img, extension="REPEAT", interpolation="Linear")
        nb.connect(uv, t.inputs["Vector"])
        return t
    A = np.asarray(albedo, float)
    alpha = None
    if A.ndim == 3 and A.shape[2] == 4:
        alpha = A[..., 3]
        A = A[..., :3]
    gain = None
    if tint:
        S = lum(A) if A.ndim == 3 else A
        gain = float(max(S.max(), 1e-3))
        A = np.repeat((S / gain)[..., None], 3, -1)
    t_alb = tex(image(name + "_albedo", np.concatenate([A, (alpha if alpha is not None else np.ones(A.shape[:2]))[..., None]], -1)))
    base = t_alb.outputs["Color"]
    if tint:
        c = lin(tint) * gain
        mix = nb.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mix.inputs[0].default_value = 1.0
        nb.connect(base, mix.inputs[6])
        mix.inputs[7].default_value = (*c, 1.0)
        base = mix.outputs[2]
        m["tint_gain"] = round(gain, 4)
    kw = dict(Base_Color=base)
    if np.ndim(rough) == 0 and np.ndim(metal) == 0:
        kw.update(Roughness=float(rough), Metallic=float(metal))
    else:
        Hh, Ww = A.shape[:2]
        R = np.broadcast_to(np.asarray(rough, float), (Hh, Ww))
        Mt = np.broadcast_to(np.asarray(metal, float), (Hh, Ww))
        orm = np.stack([np.ones((Hh, Ww)), R, Mt], -1)
        t = tex(image(name + "_orm", orm, data=True))
        sep = nb.node("ShaderNodeSeparateColor")
        nb.connect(t.outputs["Color"], sep.inputs[0])
        kw.update(Roughness=sep.outputs[1], Metallic=sep.outputs[2])
    if normal is not None:
        t = tex(image(name + "_normal", normal, data=True))
        nm = nb.node("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = nstrength
        nb.connect(t.outputs["Color"], nm.inputs["Color"])
        kw["Normal"] = nm.outputs[0]
    if alpha is not None and clip:
        kw["Alpha"] = nb.math("ROUND", t_alb.outputs["Alpha"])
    elif alpha is not None and blend:
        kw["Alpha"] = t_alb.outputs["Alpha"]
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            m.blend_method = "BLEND"
    if emission is not None:
        kw.update(Emission_Color=base, Emission_Strength=emission)
    nb.output(nb.principled(**kw))
    if cell:
        m["uv_cell"] = tuple(cell)
    return m


class Atlas:
    """A canvas laid out in metres: ppm pixels per metre, size_m metres square (the material's
    uv_cell). region(u0, v0, w, h) -> a view of the pixels covering that rectangle (row 0 = top)."""

    def __init__(self, size_m, px=1024, ch=3, fill=None):
        self.size_m, self.px = size_m, px
        self.ppm = px / size_m
        self.img = np.zeros((px, px, ch))
        if fill is not None:
            self.img[...] = fill

    def box(self, u0, v0, w, h):
        p = self.ppm
        r0 = int(round(self.px - (v0 + h) * p))
        r1 = int(round(self.px - v0 * p))
        c0 = int(round(u0 * p))
        c1 = int(round((u0 + w) * p))
        return max(0, r0), min(self.px, r1), max(0, c0), min(self.px, c1)

    def region(self, u0, v0, w, h):
        r0, r1, c0, c1 = self.box(u0, v0, w, h)
        return self.img[r0:r1, c0:c1]
