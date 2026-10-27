"""Detail for the stadium kit's instanced props (seats, goals, corner flags, floodlight fixtures):
moulded plastic shells with rolled rims, textures painted in numpy (weathered timber, fabric weave,
LED optics, stroke lettering for seat numbers and row letters), image materials that the game
export keeps as they are, and the finishing pass (ambient occlusion baked into vertex colours,
which glTF carries as COLOR_0 and the engine multiplies into the base colour, so it survives the
per-instance seat colour).
"""
import math

import bmesh
import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import bake, mesh_object


# -- curves -----------------------------------------------------------------------------------

def spline(pts, n=48):
    """A Catmull-Rom curve through 2D/3D points, resampled to n points evenly by arc length."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        t = np.linspace(0, 1, n)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    Q = np.concatenate([[2 * P[0] - P[1]], P, [2 * P[-1] - P[-2]]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, 12, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    D = np.asarray(out)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))])
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, D[:, k]) for k in range(D.shape[1])], 1)


# -- moulded shells ---------------------------------------------------------------------------

class Shell:
    """A moulded seat shell: a side-view profile swept across x, dished and shaped, with a rolled
    rim turning into a rear flange, then given wall thickness.

    prof: side-view polyline [(y, z)] walked so the sitter is on its right-hand side, i.e. the
    surface normal towards the sitter is (tz, -ty): a seat back bottom -> top, a pan front -> back.
    half_width(s) and depth(xn, s) (metres towards the sitter) take s in [0, 1] along the profile
    and xn in [-1, 1] across."""

    def __init__(self, prof, half_width, depth):
        C = spline(prof, 80)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        self.length = s[-1]
        self.s = s / s[-1]
        T = np.gradient(C, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        self.C, self.N = C, np.stack([T[:, 1], -T[:, 0]], 1)
        self.half_width, self.depth = half_width, depth

    def point(self, xn, s, back=0.0):
        """Front surface point (or `back` metres behind it, along the profile normal)."""
        xn, s = np.broadcast_arrays(np.asarray(xn, float), np.asarray(s, float))
        cy = np.interp(s, self.s, self.C[:, 0])
        cz = np.interp(s, self.s, self.C[:, 1])
        ny = np.interp(s, self.s, self.N[:, 0])
        nz = np.interp(s, self.s, self.N[:, 1])
        d = self.depth(xn, s) - back
        return np.stack([xn * self.half_width(s), cy + ny * d, cz + nz * d], -1)

    def sitter(self, s):
        return np.array([0.0, np.interp(s, self.s, self.N[:, 0]), np.interp(s, self.s, self.N[:, 1])])

    def outline(self, n=64, corner=5.0, out=0.0, back=0.0):
        """A closed loop round the shell's outline (the superellipse edge), pushed `out` metres
        outwards in the surface and `back` metres away from the sitter (piping, edge trims)."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c, sn = np.cos(t), np.sin(t)
        r = (np.abs(c) ** corner + np.abs(sn) ** corner) ** (-1 / corner)
        X, Y = c * r, sn * r
        S = (Y + 1) / 2
        P = self.point(X, S, back=back)
        Q = self.point(X * 0.9, (Y * 0.9 + 1) / 2, back=back)
        o = P - Q
        o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
        return P + o * out

    def build(self, name, mat, col=None, nu=10, nv=10, corner=5.0, rim=0.010, flange=0.016, thick=0.005,
              smooth_deg=55.0):
        a = np.linspace(-1, 1, nu + 1)
        b = np.linspace(-1, 1, nv + 1)
        A, B = np.meshgrid(a, b, indexing="ij")
        # square grid -> superellipse outline (rounded corners), radially
        th = np.arctan2(B, A)
        c, sn = np.abs(np.cos(th)), np.abs(np.sin(th))
        k = (c ** corner + sn ** corner) ** (-1 / corner) * np.maximum(c, sn)
        X, Y = A * k, B * k
        S = (Y + 1) / 2
        P = self.point(X, S)                                   # (nu+1, nv+1, 3)
        dPi = np.gradient(P, axis=0)
        dPj = np.gradient(P, axis=1)
        Nn = np.cross(dPi, dPj)
        Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True)
        ref = np.stack([np.zeros_like(S), np.interp(S, self.s, self.N[:, 0]), np.interp(S, self.s, self.N[:, 1])], -1)
        Nn *= np.sign(np.sum(Nn * ref, -1, keepdims=True) + 1e-9)
        idx = lambda i, j: i * (nv + 1) + j
        V = list(P.reshape(-1, 3))
        UV = [(p[0], s * self.length) for p, s in zip(P.reshape(-1, 3), S.ravel())]
        F = [(idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)) for i in range(nu) for j in range(nv)]
        ring = [(i, 0) for i in range(nu + 1)] + [(nu, j) for j in range(1, nv + 1)] + \
               [(i, nv) for i in range(nu - 1, -1, -1)] + [(0, j) for j in range(nv - 1, 0, -1)]
        prev = [idx(i, j) for i, j in ring]
        steps = [(math.radians(50), 0.0), (math.radians(90), 0.0), (math.radians(90), flange)] if rim > 0 else []
        for phi, ext in steps:
            cur = []
            for (i, j) in ring:
                p, n = P[i, j], Nn[i, j]
                q = P[min(max(i, 1), nu - 1), min(max(j, 1), nv - 1)]
                o = p - q
                o -= n * (o @ n)
                o /= np.linalg.norm(o) + 1e-12
                cur.append(len(V))
                V.append(p + o * rim * math.sin(phi) - n * (rim * (1 - math.cos(phi)) + ext))
                UV.append((p[0] + o[0] * rim, S[i, j] * self.length + (rim + ext) * 0.5))
            m = len(ring)
            F += [(prev[t], prev[(t + 1) % m], cur[(t + 1) % m], cur[t]) for t in range(m)]
            prev = cur
        uvs = [UV[v] for f in F for v in f]
        o = mesh_object(name, np.array(V), F, col, materials=[mat], uvs=uvs)
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.faces.ensure_lookup_table()
        centre = bm.faces[(nu // 2) * nv + nv // 2]
        want = self.sitter(0.5)
        if centre.normal.dot(want) < 0:
            bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(o.data)
        bm.free()
        if thick > 0:
            so = o.modifiers.new("Solidify", "SOLIDIFY")
            so.thickness = thick
            so.offset = -1.0
            so.use_even_offset = True
            so.use_quality_normals = True
            bake(o)
        o.data.shade_smooth()
        o.data.set_sharp_from_angle(angle=math.radians(smooth_deg))
        return o


# -- finishing --------------------------------------------------------------------------------

def finish(obj, distance=0.35, strength=0.85, floor=0.25, ground=True):
    """Ambient occlusion into the vertex colours (COLOR_0)."""
    from ..core import ao
    ao.bake(obj, distance=distance, strength=strength, floor=floor, ground=ground)
    return obj


def tint_vertices(obj, fn, materials=None):
    """Multiply the vertex colours by fn(positions (N,3), normals (N,3)) -> (N,3); `materials`
    (names) limits it to the vertices of faces using those materials."""
    me = obj.data
    attr = me.color_attributes.get("Col")
    if attr is None:
        return
    mask = None
    if materials is not None:
        idx = [i for i, m in enumerate(me.materials) if m and m.name in materials]
        mask = np.zeros(len(me.vertices), bool)
        for poly in me.polygons:
            if poly.material_index in idx:
                mask[list(poly.vertices)] = True
    n = len(me.vertices)
    co = np.zeros(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    nr = np.zeros(n * 3, np.float32)
    me.vertices.foreach_get("normal", nr)
    c = np.zeros(n * 4, np.float32)
    attr.data.foreach_get("color", c)
    c = c.reshape(-1, 4)
    f = np.clip(fn(co.reshape(-1, 3), nr.reshape(-1, 3)), 0, 1.5)
    if mask is not None:
        f = np.where(mask[:, None], f, 1.0)
    c[:, :3] *= f
    attr.data.foreach_set("color", c.ravel())


# -- periodic noise and textures --------------------------------------------------------------

def pnoise(h, w, fy, fx, seed=0, beta=1.0):
    """Tileable noise in [0, 1]: random phases with a spectrum peaked at fy, fx cycles per tile
    (anisotropic when they differ), normalised."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    r = np.sqrt((KY / max(fy, 1e-3)) ** 2 + (KX / max(fx, 1e-3)) ** 2)
    amp = 1.0 / (1.0 + r ** (2 * beta + 1))
    amp[0, 0] = 0
    spec = amp * np.exp(2j * np.pi * rng.random((h, w)))
    n = np.real(np.fft.ifft2(spec))
    n = (n - n.mean()) / (n.std() + 1e-9)
    return np.clip(0.5 + 0.18 * n, 0, 1)


def to_lin(hexstr):
    return np.array(srgb(hexstr)[:3])


def image(name, rgba, non_color=False):
    """Packed image from (H, W, 3|4) values in [0, 1], row 0 the bottom. Colour data is linear and
    stored sRGB-encoded; non_color data is stored as it is (normal maps, masks)."""
    if rgba.shape[-1] == 3:
        rgba = np.concatenate([rgba, np.ones(rgba.shape[:2] + (1,))], -1)
    if not non_color:
        return MAT.image_from_array(name, rgba.astype(np.float32))
    h, w = rgba.shape[:2]
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.clip(rgba, 0, 1).astype(np.float32).ravel())
    img.pack()
    return img


def normal_from_height(h, strength=2.0):
    """Tangent-space normal map (as 0-1 RGB) from a tileable height field."""
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def image_material(name, albedo=None, normal=None, rough=0.5, metallic=0.0, base=None, normal_strength=1.0,
                   emission=None, alpha=False, extension="REPEAT", uv_scale=None):
    """A material the game export keeps as it is: image base colour (times `base`, a linear RGB
    factor), optional tangent normal map and emission map, constant roughness/metallic."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    if uv_scale:
        mp = nb.node("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (uv_scale, uv_scale, 1.0)
        nb.connect(uv, mp.inputs["Vector"])
        uv = mp.outputs[0]
    kw = dict(Roughness=rough, Metallic=metallic)
    if albedo is not None:
        tex = nb.node("ShaderNodeTexImage", image=albedo, extension=extension)
        nb.connect(uv, tex.inputs["Vector"])
        col = tex.outputs["Color"]
        if base is not None:
            mix = nb.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
            mix.inputs[0].default_value = 1.0
            nb.connect(col, mix.inputs[6])
            mix.inputs[7].default_value = (*base[:3], 1.0)
            col = mix.outputs[2]
        kw["Base_Color"] = col
        if alpha:
            kw["Alpha"] = nb.math("ROUND", tex.outputs["Alpha"])
    elif base is not None:
        kw["Base_Color"] = (*base[:3], 1.0)
    if normal is not None:
        nt = nb.node("ShaderNodeTexImage", image=normal, extension="REPEAT")
        nb.connect(uv, nt.inputs["Vector"])
        nm = nb.node("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nb.connect(nt.outputs["Color"], nm.inputs["Color"])
        kw["Normal"] = nm.outputs[0]
    if emission is not None:
        et = nb.node("ShaderNodeTexImage", image=emission[0], extension=extension)
        nb.connect(uv, et.inputs["Vector"])
        kw["Emission_Color"] = et.outputs["Color"]
        kw["Emission_Strength"] = emission[1]
    nb.output(nb.principled(**kw))
    return m


# -- stroke lettering -------------------------------------------------------------------------

def _arc(cx, cy, rx, ry, a0, a1, n=14):
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]


# glyphs on a 1 x 1.6 box (y up): lists of polylines
GLYPHS = {
    "0": [_arc(0.5, 0.8, 0.42, 0.78, 0, 360, 28)],
    "1": [[(0.22, 1.28), (0.56, 1.58), (0.56, 0.0)]],
    "2": [_arc(0.5, 1.14, 0.42, 0.44, 165, -30, 12) + [(0.08, 0.0), (0.94, 0.0)]],
    "3": [_arc(0.48, 1.2, 0.38, 0.38, 150, -90, 12), _arc(0.48, 0.42, 0.44, 0.42, 90, -155, 12)],
    "4": [[(0.72, 0.0), (0.72, 1.58), (0.05, 0.42), (0.98, 0.42)]],
    "5": [[(0.88, 1.58), (0.18, 1.58), (0.12, 0.86)] + _arc(0.5, 0.5, 0.44, 0.5, 125, -150, 14)],
    "6": [_arc(0.5, 0.48, 0.42, 0.48, 0, 360, 22), [(0.09, 0.55), (0.25, 1.15), (0.62, 1.58)]],
    "7": [[(0.06, 1.58), (0.94, 1.58), (0.38, 0.0)]],
    "8": [_arc(0.5, 1.2, 0.36, 0.38, 0, 360, 20), _arc(0.5, 0.44, 0.43, 0.44, 0, 360, 22)],
    "9": [_arc(0.5, 1.12, 0.42, 0.48, 0, 360, 22), [(0.91, 1.05), (0.75, 0.45), (0.38, 0.0)]],
    "A": [[(0.04, 0.0), (0.5, 1.58), (0.96, 0.0)], [(0.2, 0.52), (0.8, 0.52)]],
    "B": [[(0.1, 0.0), (0.1, 1.58), (0.55, 1.58)] + _arc(0.55, 1.2, 0.36, 0.38, 90, -90, 10) + [(0.1, 0.82)],
          [(0.1, 0.82), (0.58, 0.82)] + _arc(0.58, 0.41, 0.38, 0.41, 90, -90, 10) + [(0.1, 0.0)]],
    "C": [_arc(0.55, 0.8, 0.47, 0.8, 40, 320, 20)],
    "D": [[(0.1, 0.0), (0.1, 1.58), (0.4, 1.58)] + _arc(0.4, 0.79, 0.52, 0.79, 90, -90, 16) + [(0.1, 0.0)]],
    "E": [[(0.9, 1.58), (0.1, 1.58), (0.1, 0.0), (0.9, 0.0)], [(0.1, 0.82), (0.75, 0.82)]],
    "F": [[(0.9, 1.58), (0.1, 1.58), (0.1, 0.0)], [(0.1, 0.82), (0.75, 0.82)]],
    "G": [_arc(0.55, 0.8, 0.47, 0.8, 40, 340, 20) + [(0.98, 0.7), (0.6, 0.7)]],
    "H": [[(0.1, 0.0), (0.1, 1.58)], [(0.9, 0.0), (0.9, 1.58)], [(0.1, 0.82), (0.9, 0.82)]],
    "J": [[(0.75, 1.58)] + _arc(0.42, 0.42, 0.33, 0.42, 0, -180, 10)],
    "K": [[(0.1, 0.0), (0.1, 1.58)], [(0.9, 1.58), (0.1, 0.6)], [(0.35, 0.86), (0.92, 0.0)]],
    "L": [[(0.1, 1.58), (0.1, 0.0), (0.88, 0.0)]],
    "M": [[(0.06, 0.0), (0.12, 1.58), (0.5, 0.35), (0.88, 1.58), (0.94, 0.0)]],
    "N": [[(0.1, 0.0), (0.1, 1.58), (0.9, 0.0), (0.9, 1.58)]],
    "P": [[(0.1, 0.0), (0.1, 1.58), (0.55, 1.58)] + _arc(0.55, 1.16, 0.38, 0.42, 90, -90, 10) + [(0.1, 0.74)]],
    "R": [[(0.1, 0.0), (0.1, 1.58), (0.55, 1.58)] + _arc(0.55, 1.16, 0.38, 0.42, 90, -90, 10) + [(0.1, 0.74)],
          [(0.5, 0.74), (0.92, 0.0)]],
    "S": [_arc(0.5, 1.2, 0.4, 0.38, 20, 270, 12) + _arc(0.5, 0.42, 0.44, 0.42, 90, -160, 12)],
    "T": [[(0.04, 1.58), (0.96, 1.58)], [(0.5, 1.58), (0.5, 0.0)]],
    "-": [[(0.2, 0.75), (0.8, 0.75)]],
    "I": [[(0.5, 0.0), (0.5, 1.58)], [(0.25, 1.58), (0.75, 1.58)], [(0.25, 0.0), (0.75, 0.0)]],
    "O": [_arc(0.5, 0.8, 0.45, 0.78, 0, 360, 28)],
    "U": [[(0.1, 1.58)] + _arc(0.5, 0.42, 0.4, 0.42, 180, 360, 12) + [(0.9, 1.58)]],
    "V": [[(0.04, 1.58), (0.5, 0.0), (0.96, 1.58)]],
    "W": [[(0.0, 1.58), (0.22, 0.0), (0.5, 1.1), (0.78, 0.0), (1.0, 1.58)]],
    "X": [[(0.06, 0.0), (0.94, 1.58)], [(0.06, 1.58), (0.94, 0.0)]],
    "Y": [[(0.04, 1.58), (0.5, 0.8), (0.96, 1.58)], [(0.5, 0.8), (0.5, 0.0)]],
    "Z": [[(0.08, 1.58), (0.92, 1.58), (0.08, 0.0), (0.92, 0.0)]],
    ".": [[(0.45, 0.02), (0.55, 0.02)]],
    " ": [],
}


def lettering(text, px_h, weight=0.13, pad=0.25):
    """Anti-aliased stroke text as a coverage mask (H, W), row 0 the bottom. Glyph height px_h."""
    adv = 1.18
    width = (len(text) - 1) * adv + 1.0 + 2 * pad
    height = 1.6 + 2 * pad
    sc = px_h / 1.6
    W, H = int(math.ceil(width * sc)), int(math.ceil(height * sc))
    yy, xx = np.mgrid[0:H, 0:W]
    P = np.stack([(xx + 0.5) / sc - pad, (yy + 0.5) / sc - pad], -1)
    d = np.full((H, W), 1e9)
    for k, ch in enumerate(text):
        for line in GLYPHS.get(ch, []):
            L = np.asarray(line) + [k * adv, 0.0]
            for a, b in zip(L[:-1], L[1:]):
                ab = b - a
                t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                d = np.minimum(d, np.linalg.norm(P - (a + t[..., None] * ab), axis=-1))
    return np.clip((weight * 0.5 - d) * sc + 0.5, 0, 1)


def plate_image(name, text, fg="#f4f4f0", bg="#1b1c1e", px=(128, 64), border=None):
    """A small plate (seat number, row letter): text centred on a background, optional border."""
    W, H = px
    mask = lettering(text, H * 0.62)
    img = np.zeros((H, W))
    mh, mw = mask.shape
    s = min(1.0, W * 0.92 / mw, H * 0.95 / mh)
    if s < 1.0:
        mask = lettering(text, H * 0.62 * s)
        mh, mw = mask.shape
    y0, x0 = (H - mh) // 2, (W - mw) // 2
    img[max(y0, 0):max(y0, 0) + mh, max(x0, 0):max(x0, 0) + mw] = mask[:H, :W]
    if border:
        e = int(border)
        frame = np.zeros((H, W))
        frame[:e], frame[-e:], frame[:, :e], frame[:, -e:] = 1, 1, 1, 1
        img = np.maximum(img, frame)
    f, b = to_lin(fg), to_lin(bg)
    rgb = b + (f - b) * img[..., None]
    return image(name, rgb)
