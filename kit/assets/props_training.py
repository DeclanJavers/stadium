"""Training gear: free-kick mannequins, marker and pole cones, slalom poles, mini goals, a
rebounder, footballs and the ball cart, an agility ladder, a team shelter. Real sizes; origin on
the ground at the centre, +Y the front (the side a mannequin faces, a goal's mouth).

Surface detail is painted (training_paint) into image textures, so it survives the game bake:
recolourable plastics (TINT_) carry a greyscale wear map times their colour (glTF extras
tint_gain), fixed-colour parts a full albedo. Dirt that follows height (mud on a pole's foot, a
cone's grimy base) is multiplied into the baked occlusion in the vertex colours.
"""
import math

import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material, srgb
from . import stand
from . import training_paint as TP
from .props import PM, rrect, tyre


# -- materials ------------------------------------------------------------------------------------

def _image(name, arr, data=False):
    img = bpy.data.images.get(name)
    if img:
        return img
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,))], -1)
    if not data:
        return MAT.image_from_array(name, arr)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(arr.astype(np.float32).ravel())
    img.pack()
    return img


def _tex_mat(name, alb, rough, nrm=None, tile=None, tint=None, gain=None, metal=0.0, alpha=False, spec=0.5, blend=False):
    """An image-textured material. alb: (H, W, 3|4) linear, or a greyscale detail map when `tint`
    (a hex colour) is given: the colour becomes the base factor (times `gain`, the detail map's
    1 / mean) and runtime recolours scale it the same way. tile: metres per repeat on the
    mesh's metric UVs (None: the UVs are the image's)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    vec = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    if tile:
        mp = nb.node("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1 / tile[0], 1 / tile[1], 1.0)
        nb.connect(vec, mp.inputs["Vector"])
        vec = mp.outputs[0]

    def tex(img):
        t = nb.node("ShaderNodeTexImage", image=img, extension="REPEAT")
        nb.connect(vec, t.inputs["Vector"])
        return t
    ta = tex(_image(name + "_albedo", alb))
    col = ta.outputs["Color"]
    if tint:
        c = srgb(tint)
        g = gain or 1.0
        mix = nb.node("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
        mix.inputs[0].default_value = 1.0
        nb.connect(col, mix.inputs[6])
        mix.inputs[7].default_value = (min(1.0, c[0] * g), min(1.0, c[1] * g), min(1.0, c[2] * g), 1.0)
        col = mix.outputs[2]
        m["tint_gain"] = round(g, 4)
    kw = dict(Base_Color=col, Metallic=metal, Specular_IOR_Level=spec)
    if isinstance(rough, np.ndarray):
        kw["Roughness"] = tex(_image(name + "_rough", rough, data=True)).outputs["Color"]
    else:
        kw["Roughness"] = rough
    if nrm is not None:
        nm = nb.node("ShaderNodeNormalMap")
        nb.connect(tex(_image(name + "_normal", nrm, data=True)).outputs["Color"], nm.inputs["Color"])
        kw["Normal"] = nm.outputs[0]
    if alpha:
        kw["Alpha"] = ta.outputs["Alpha"] if blend else nb.math("ROUND", ta.outputs["Alpha"])
    nb.output(nb.principled(**kw))
    return m


def _flat(name, color, rough, metal=0.0):
    """A plain material (small parts: springs, pins, rubber caps)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=rough, Metallic=metal))
    return m


def _tint_plastic(name, color, seed, rough=0.4, tile=0.4, scuffs=10, scratches=60):
    """A recolourable moulded plastic with mottling, scratches and scuffs."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    w = TP.wear(512, seed, scuffs=scuffs, scratches=scratches, tile=tile)
    det, mean, r, n = TP.tint_detail(w, rough=rough)
    return _tex_mat(name, det, r, n, tile=(tile, tile), tint=color, gain=1 / mean)


def _powder(name, color, seed, chip="#8a8d90", rough=0.45, tile=0.5, chips=30, metal=0.0):
    """Powder-coated steel: mottled satin coat, handling scuffs and chips to bare steel."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    w = TP.wear(512, seed, chips=chips, scuffs=6, scratches=40, tile=tile)
    alb, r, n = TP.coloured(w, color, chip_color=chip, rough=rough, scuff=0.3, mottle=0.05)
    return _tex_mat(name, alb, r, n, tile=(tile, tile), metal=metal)


def _ball_mat():
    if bpy.data.materials.get("Prop_Ball"):
        return bpy.data.materials["Prop_Ball"]
    alb, nrm, r = TP.ball_atlas()
    return _tex_mat("Prop_Ball", alb, r, nrm, spec=0.55)


def _net_mat(name, color, cord, knot=1.7, twist=True, weld=False, metal=0.0, rough=0.8):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    rgba, nrm = TP.netting(256, cord=cord, knot=knot, color=color, twist=twist, weld=weld)
    return _tex_mat(name, rgba, rough, nrm, alpha=True, metal=metal)


def _mats():
    return {
        "zinc": _flat("Prop_Zinc", "#b8bcbf", 0.32, 1.0),
        "steel": _flat("Prop_SteelBare", "#7d8084", 0.45, 1.0),
        "rubber": _flat("Prop_RubberCap", "#151515", 0.75),
        "blackp": _flat("Prop_BlackPlastic", "#1a1b1c", 0.5),
        "mud": _flat("Prop_Mud", "#4a3b2b", 0.9),
    }


def _grime(o, fn):
    """Bake occlusion into the vertex colours, then multiply in fn(P) -> (N, 3) (dirt that follows
    height, per-part tint)."""
    from ..core import ao
    ao.bake(o)
    me = o.data
    P = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", P)
    P = P.reshape(-1, 3)
    attr = me.color_attributes["Col"]
    c = np.zeros(len(attr.data) * 4, np.float32)
    attr.data.foreach_get("color", c)
    c = c.reshape(-1, 4)
    c[:, :3] *= np.clip(fn(P), 0, 1)
    attr.data.foreach_set("color", c.ravel())
    return o


def _sm(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _mud(P, top=0.3, col=(0.62, 0.52, 0.38), bottom=0.0):
    """Mud and grass stain fading out from the ground to `top` metres."""
    f = _sm(bottom, top, P[:, 2])[:, None]
    return np.asarray(col)[None] * (1 - f) + f


# -- geometry helpers -----------------------------------------------------------------------------

def _face_out(p, pts, mat, out, uv=None):
    """A face wound so its normal points along `out`."""
    P = np.asarray(pts, float)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    if len(P) > 3:
        n = n + np.cross(P[2] - P[0], P[3] - P[0])
    if n @ np.asarray(out, float) < 0:
        P = P[::-1]
        uv = uv[::-1] if uv is not None else None
    p.face(P, mat, uv)


def _offset_profile(prof, t):
    """Offset an (r, z) polyline (running up towards the axis) by t to its inside."""
    P = np.asarray(prof, float)
    out = []
    for i in range(len(P)):
        a, b = P[max(i - 1, 0)], P[min(i + 1, len(P) - 1)]
        d = b - a
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])          # left of the direction of travel (the inside)
        out.append(P[i] + n * t)
    return out


def _shell_lathe(p, outer, t, mat, sides=32, open_cell=None, uvf=None):
    """A thin moulded shell of revolution: the outer (r, z) profile, its inner skin t inside, the
    ends closed; open_cell(j, k) cuts a window (segment j, sector k) with walls through the skin.
    uvf(u, z) gives UVs (u 0-1 round the axis) for both skins (alpha-cut holes line up)."""
    inner = _offset_profile(outer, t)
    nj = len(outer)
    ang = [2 * math.pi * k / sides for k in range(sides)]

    def pt(prof, j, k):
        r, z = prof[j]
        return np.array([r * math.cos(ang[k % sides]), r * math.sin(ang[k % sides]), z])
    O = [[pt(outer, j, k) for k in range(sides)] for j in range(nj)]
    I = [[pt(inner, j, k) for k in range(sides)] for j in range(nj)]
    op = (lambda j, k: open_cell(j, k % sides)) if open_cell else (lambda j, k: False)

    def uvs(q, ks):
        return [uvf(kk / sides, x[2]) for x, kk in zip(q, ks)] if uvf else None
    for j in range(nj - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            if op(j, k):
                cen = (O[j][k] + O[j + 1][k2] + I[j][k] + I[j + 1][k2]) / 4
                sidesq = (((j, k), (j, k2), (j - 1, k)), ((j + 1, k), (j + 1, k2), (j + 1, k)),
                          ((j, k), (j + 1, k), (j, k - 1)), ((j, k2), (j + 1, k2), (j, k + 1)))
                for a, b, nbr in sidesq:
                    if 0 <= nbr[0] < nj - 1 and op(*nbr):
                        continue
                    q = [O[a[0]][a[1]], O[b[0]][b[1]], I[b[0]][b[1]], I[a[0]][a[1]]]
                    fix = lambda x: sides if (x == 0 and k == sides - 1) else x
                    kk = [fix(a[1]), fix(b[1]), fix(b[1]), fix(a[1])]
                    _face_out(p, q, mat, np.mean(q, 0) - cen, uvs(q, kk))
                continue
            q = [O[j][k], O[j][k2], O[j + 1][k2], O[j + 1][k]]
            qi = [I[j][k], I[j][k2], I[j + 1][k2], I[j + 1][k]]
            ks = [k, k + 1, k + 1, k]
            out = np.mean(q, 0) - np.mean(qi, 0)
            _face_out(p, q, mat, out, uvs(q, ks))
            _face_out(p, qi, mat, -out, uvs(qi, ks))
    for j, sgn in ((0, -1), (nj - 1, 1)):
        for k in range(sides):
            k2 = (k + 1) % sides
            q = [O[j][k], O[j][k2], I[j][k2], I[j][k]]
            tang = np.asarray(outer[j]) - np.asarray(outer[j - sgn])
            d = np.array([tang[0] * math.cos(ang[k]), tang[0] * math.sin(ang[k]), tang[1]])
            _face_out(p, q, mat, d, uvs(q, [k, k + 1, k + 1, k]))


def _loft_uv(p, rings, mat, uvf, cap0=True, cap1=True):
    """Skin rings (closed, counter-clockwise from above) with UVs uvf(point)."""
    n, m = len(rings), len(rings[0])
    for i in range(n - 1):
        for j in range(m):
            j2 = (j + 1) % m
            q = [rings[i][j], rings[i + 1][j], rings[i + 1][j2], rings[i][j2]]
            p.face(q, mat, [uvf(x) for x in q])
    if cap0:
        q = rings[0][::-1]
        p.face(q, mat, [uvf(x) for x in q])
    if cap1:
        q = rings[-1]
        p.face(q, mat, [uvf(x) for x in q])


def _grid_patch(p, P, nu, nv, mat, uvf, out=None, jitter=0.0, seed=0):
    """A sheet P(u, v) (u, v in 0-1) as nu x nv quads, UVs uvf(u, v); jitter (texture units)
    nudges the inner UVs so a net's cords run a little unevenly, as tied netting does."""
    G = [[P(i / nu, j / nv) for j in range(nv + 1)] for i in range(nu + 1)]
    rng = np.random.default_rng(seed)
    UV = [[np.asarray(uvf(i / nu, j / nv), float) + (rng.normal(0, jitter, 2) if 0 < i < nu and 0 < j < nv else 0)
           for j in range(nv + 1)] for i in range(nu + 1)]
    for i in range(nu):
        for j in range(nv):
            q = [G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]]
            uv = [tuple(UV[i][j]), tuple(UV[i + 1][j]), tuple(UV[i + 1][j + 1]), tuple(UV[i][j + 1])]
            if out is None:
                p.face(q, mat, uv)
            else:
                _face_out(p, q, mat, out, uv)


def _coil(p, c, r, z0, z1, turns, wire, mat, n_per_turn=14):
    t = np.linspace(0, 2 * math.pi * turns, int(n_per_turn * turns) + 1)
    pts = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a), z0 + (z1 - z0) * a / t[-1]) for a in t]
    rr, p.r = p.r, 0.0
    p.tube(pts, wire, mat, 4, bend=0.0, cap=True)
    p.r = rr


def _ring_round(p, c, axis, r, wire, mat, n=6, sides=3):
    """A small closed ring (a clip, a bungee wrap) round an axis through c."""
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    u = np.cross(a, [0, 0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    pts = [np.asarray(c) + (u * math.cos(2 * math.pi * k / n) + v * math.sin(2 * math.pi * k / n)) * r for k in range(n)]
    rr, p.r = p.r, 0.0
    p.tube(pts, wire, mat, sides, bend=0.0, closed=True)
    p.r = rr


def _rot(rng):
    """A uniformly random rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# -- the free-kick mannequin ----------------------------------------------------------------------

# right-hand outline (half width, height) of the moulded body, metres: rounded feet, legs, hips with
# the hands moulded against them, waist, chest, sloping shoulders, a neck narrower than the head
_OUTLINE = [(0.13, 0.04), (0.148, 0.065), (0.152, 0.16), (0.15, 0.3), (0.135, 0.5), (0.15, 0.68), (0.168, 0.84),
            (0.19, 0.92), (0.2, 0.98), (0.19, 1.06), (0.205, 1.18), (0.228, 1.3), (0.24, 1.4), (0.236, 1.45),
            (0.215, 1.49), (0.17, 1.52), (0.11, 1.545), (0.072, 1.56), (0.064, 1.585), (0.072, 1.605), (0.092, 1.628),
            (0.102, 1.66), (0.104, 1.7), (0.098, 1.74), (0.084, 1.768), (0.062, 1.788), (0.035, 1.798)]
_CROTCH = 0.84
_GAP = 0.02            # half the gap between the legs


def _half_width(z):
    o = np.asarray(_OUTLINE)
    return np.interp(z, o[:, 1], o[:, 0])


def _half_thick(z):
    """Half the body's depth: 150 mm at the chest, fuller thighs and calves, a slim neck."""
    return np.interp(z, [0.04, 0.09, 0.3, 0.5, 0.7, 0.9, 1.05, 1.3, 1.45, 1.53, 1.575, 1.68, 1.78, 1.8],
                     [0.026, 0.038, 0.05, 0.042, 0.055, 0.06, 0.065, 0.075, 0.062, 0.045, 0.036, 0.052, 0.04, 0.024])


def _slab_ring(xc, hw, h, z, grooves=()):
    """A blow-moulded section: domed faces (2 h deep in the middle, 0.8 h at the edge), fully
    rounded edges (radius 0.4 h), centred at xc, half width hw; grooves: (u, depth, width) sunk into
    both faces (a negative depth raises a ridge)."""
    e = 0.4 * h
    wi = max(hw - e, 0.002)
    us = [1.0, 0.85, 0.72, 0.5, 0.25, 0.08, 0.0]
    us = us + [-u for u in reversed(us[:-1])]
    if grooves:
        us = sorted(set(us + [g[0] + d for g in grooves for d in (-g[2], 0.0, g[2])]), reverse=True)
        us = [u for u in us if -1 <= u <= 1]
    front = []
    for u in us:
        y = h * (0.4 + 0.6 * math.sqrt(max(0.0, 1 - u * u)))
        for gu, depth, gw in grooves:
            y -= depth * math.exp(-((u - gu) / gw) ** 2)
        front.append((xc + wi * u, y))
    ring = [(x, y, z) for x, y in front]
    for k in range(1, 8):
        t = math.radians(90 + 22.5 * k)
        ring.append((xc - wi + e * math.cos(t), e * math.sin(t), z))
    ring += [(x, -y, z) for x, y in front[::-1]]
    for k in range(1, 8):
        t = math.radians(270 + 22.5 * k)
        ring.append((xc + wi + e * math.cos(t), e * math.sin(t), z))
    return ring


def mannequin(col, M):
    """A blow-moulded free-kick mannequin (1.8 m): a hollow hi-vis body 150 mm deep at the chest,
    60 mm at its fully rounded edges; head on a narrower neck, sloping shoulders, arms standing
    proud of the body behind a groove, shaped thighs and calves, split legs with rounded feet 40 mm
    off the turf, a moulded number. Each foot sits on a steel ground spike (socket boss, washer on
    the turf, the spike driven in)."""
    p = PM(0.0)
    det, mean, r, n = TP.mannequin_map(outline=_half_width)
    body = _tex_mat("TINT_mannequin", det, r, n, tint="#ebc828", gain=1 / mean)
    X0, X1, Z0, Z1 = TP.MANNEQUIN_BOX
    uvf = lambda q: ((q[0] - X0) / (X1 - X0), (q[2] - Z0) / (Z1 - Z0))
    feet = []
    for s in (-1, 1):
        rings = []
        zs = [0.04, 0.046, 0.058, 0.075] + list(np.linspace(0.11, _CROTCH + 0.04, 9))
        for z in zs:
            w, h = float(_half_width(z)), float(_half_thick(z))
            foot = _sm(0.035, 0.09, z)
            hw = (w - _GAP) / 2 * (0.84 + 0.16 * foot)
            xc = s * (_GAP + (w - _GAP) / 2)
            rings.append(_slab_ring(xc, hw, h * (0.7 + 0.3 * foot), z))
        feet.append(s * (_GAP + (float(_half_width(0.05)) - _GAP) / 2))
        _loft_uv(p, rings, body, uvf, cap0=True, cap1=False)
    # body: hips to the crown; the arms stand proud of the sides behind a moulded groove
    zs = [_CROTCH - 0.03, _CROTCH - 0.012] + list(np.linspace(_CROTCH, 1.5, 13)) + list(np.linspace(1.52, 1.798, 11))
    rings = []
    for i, z in enumerate(zs):
        w, h = float(_half_width(z)), float(_half_thick(z))
        arms = _sm(0.92, 1.02, z) * _sm(1.48, 1.4, z)
        g = ((0.62, 0.011 * arms, 0.035), (-0.62, 0.011 * arms, 0.035),
             (0.83, -0.014 * arms, 0.07), (-0.83, -0.014 * arms, 0.07))
        if i == 0:
            w, h = w - 0.03, h * 0.55
        elif i == 1:
            w, h = w - 0.008, h * 0.88
        rings.append(_slab_ring(0.0, w, h, z, g))
    top = rings[-1]
    cz = top[0][2]
    rings.append([(x * 0.5, y * 0.55, cz + 0.003) for x, y, _ in top])
    _loft_uv(p, rings, body, uvf, cap0=True, cap1=True)
    # mount: a ground spike under each foot, through a moulded socket boss and a washer on the turf
    for x in feet:
        p.r = 0.002
        p.cyl((x, 0, 0.026), (x, 0, 0.05), 0.017, M["black"], 14)
        p.r = 0.0
        p.cyl((x, 0, 0.0), (x, 0, 0.004), 0.024, M["zinc"], 14)
        p.lathe([(0.0085, 0.03), (0.0085, 0.0), (0.007, -0.15), (0.002, -0.24), (0.0, -0.245)], M["steel"], c=(x, 0, 0), sides=10)
    o = p.obj("mannequin", col, 40)

    def mud(P):
        f = _sm(0.04, 0.4, P[:, 2])[:, None]
        brown = np.array([0.42, 0.3, 0.18])[None]
        return 1 - 0.6 * (1 - f) * (1 - brown)
    return _grime(o, mud)


def cone_tall(col, M):
    """A 450 mm sport cone: one moulded PE piece (3 mm wall) - a square 8 mm skirt with rounded
    corners running into the cone through a 15 mm fillet, stacking ridges and a rolled lip at the
    open top, and twelve 16 x 35 mm slots, four round at three heights in opposed pairs so a bar
    passes straight through; a worn maker's print."""
    p = PM(0.0015)
    c = _tint_plastic("TINT_cone", "#f05a14", 3, rough=0.42)
    p.loft([rrect(0.27, 0.27, 0.035, 0.0), rrect(0.27, 0.27, 0.035, 0.0065), rrect(0.262, 0.262, 0.031, 0.0085)], c)

    def rad(z):
        return 0.092 + (0.0255 - 0.092) * (z - 0.03) / (0.438 - 0.03)
    slot_z = [(0.11, 0.145), (0.19, 0.225), (0.27, 0.305)]
    zs = sorted({0.04, 0.07, 0.24, 0.33, 0.37} | {z for pr in slot_z for z in pr})
    outer = [(0.123, 0.0084), (0.11, 0.0105), (0.1015, 0.0155), (0.0965, 0.024), (rad(0.032), 0.032)]
    outer += [(rad(z), z) for z in zs]
    outer += [(rad(0.395), 0.395), (rad(0.4) + 0.0016, 0.4), (rad(0.405), 0.405), (rad(0.412) + 0.0016, 0.412),
              (rad(0.419), 0.419), (rad(0.438), 0.438), (0.0272, 0.4445), (0.0262, 0.4492), (0.0238, 0.4498)]
    zl = [z for _, z in outer]
    slot_j = {zl.index(z0) for z0, _ in slot_z}
    p.r = 0.0
    _shell_lathe(p, outer, 0.003, c, 48, open_cell=lambda j, k: j in slot_j and k in (47, 0, 11, 12, 23, 24, 35, 36))
    from . import lettering
    if not bpy.data.materials.get("Prop_ConePrint"):
        sg = lettering.sign("PRO TRAIN 45", 64, "#3b3532", None, pad=0.05, weight=1.0)
        _tex_mat("Prop_ConePrint", sg, 0.5, alpha=True)
    pr = bpy.data.materials["Prop_ConePrint"]
    zt = 0.075
    a = math.radians(45)
    rr = rad(zt) + 0.0012
    slope = math.atan2(0.092 - 0.0255, 0.438 - 0.03)
    n = np.array([math.cos(a) * math.cos(slope), math.sin(a) * math.cos(slope), math.sin(slope)])
    up = np.array([-math.cos(a) * math.sin(slope), -math.sin(a) * math.sin(slope), math.cos(slope)])
    p.decal((rr * math.cos(a), rr * math.sin(a), zt), n, up, 0.08, 0.012, pr, lift=0.0006)
    o = p.obj("cone_tall", col, 40)

    def wear(P):
        z = P[:, 2]
        stain = _mud(P, 0.07, (0.55, 0.5, 0.36))
        mid = 1 - 0.05 * np.exp(-((z - 0.25) / 0.06) ** 2) * (0.5 + 0.5 * np.sin(P[:, 0] * 90) * np.sin(P[:, 1] * 70))
        return stain * mid[:, None] * (0.95 + 0.05 * _sm(0.3, 0.45, z))[:, None]
    return _grime(o, wear)


def cone_disc(col, M):
    """A saucer marker cone: one soft-PE moulding 195 mm across and 60 mm tall - a rolled brim, a
    domed flank (about 32 degrees at the brim, steepening), two stacking ridges, four small
    rounded slots and a 45 mm top hole with a rolled 3 mm lip."""
    p = PM(0.0)
    c = _tint_plastic("TINT_cone", "#f05a14", 3, rough=0.42)
    outer = [(0.0975, 0.0022), (0.0968, 0.0045), (0.094, 0.0062), (0.088, 0.0075)]
    S = [0.0, 0.14, 0.27, 0.45, 0.6, 0.72, 0.84, 0.93, 1.0]
    for t in S[1:]:
        r = 0.088 - (0.088 - 0.0285) * t
        z = 0.0075 + (0.058 - 0.0075) * (1 - (1 - t) ** 1.55)
        if t in (0.6, 0.72):
            r += 0.0012                                   # stacking ridges
        outer.append((r, z))
    outer += [(0.0262, 0.0598), (0.0236, 0.0596), (0.0225, 0.0575), (0.0225, 0.0545)]
    j_slot = 4 + S[1:].index(0.27)                      # the segment from t = 0.27 to 0.45
    p.r = 0.0
    _shell_lathe(p, outer, 0.0021, c, 48, open_cell=lambda j, k: j == j_slot and k % 12 in (4, 5, 6, 7))
    p.V = [(x, y, z + 0.0012) for x, y, z in p.V]       # the inner skin's brim edge sits on the turf
    o = p.obj("cone_disc", col, 45)
    return _grime(o, lambda P: _mud(P, 0.016, (0.66, 0.6, 0.45)) * (0.96 + 0.04 * _sm(0.03, 0.06, P[:, 2]))[:, None])


def _lathe_uv(p, prof, mat, sides, uvf):
    """A surface of revolution with UVs uvf(u, z), u 0-1 round the axis (seam at +x)."""
    for j in range(len(prof) - 1):
        (r0, z0), (r1, z1) = prof[j], prof[j + 1]
        for k in range(sides):
            a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
            q = [(r0 * math.cos(a0), r0 * math.sin(a0), z0), (r0 * math.cos(a1), r0 * math.sin(a1), z0),
                 (r1 * math.cos(a1), r1 * math.sin(a1), z1), (r1 * math.cos(a0), r1 * math.sin(a0), z1)]
            p.face(q, mat, [uvf(k / sides, z0), uvf((k + 1) / sides, z0), uvf((k + 1) / sides, z1), uvf(k / sides, z1)])


def agility_pole(col, M):
    """A 1.7 m slalom pole: 25 mm gloss PVC tube (extrusion seam, a printed band, scuffs and grass
    stain at the foot) with a flared rubber cap, in a rubber socket on an open zinc-plated coil
    spring (40 mm across, 7 turns of 4 mm wire) over a steel ferrule and a tapered spike driven
    into the turf."""
    p = PM(0.0)
    if not bpy.data.materials.get("TINT_pole_train"):
        det, mean, r, n = TP.pole_map()
        _tex_mat("TINT_pole_train", det, r, n, tint="#ebc814", gain=1 / mean)
    c = bpy.data.materials["TINT_pole_train"]
    L = TP.POLE_LEN
    _lathe_uv(p, [(0.0125, z) for z in (0.215, 0.26, 0.32, 0.4, 0.5, 0.7, 1.0, 1.3, 1.45, 1.5, 1.6, 1.662)], c, 16,
              lambda u, z: (u, z / L))
    p.lathe([(0.0138, 1.652), (0.0143, 1.656), (0.0143, 1.684), (0.0158, 1.688), (0.016, 1.692), (0.0145, 1.695),
             (0.0125, 1.7), (0.0095, 1.705), (0.0055, 1.708), (0.0, 1.709)], M["rubber"], sides=16)
    p.lathe([(0.0128, 0.2295), (0.0168, 0.2275), (0.0205, 0.222), (0.0208, 0.16), (0.0198, 0.148), (0.012, 0.146)],
            M["rubber"], sides=16)
    _coil(p, (0, 0), 0.018, 0.06, 0.15, 7, 0.002, M["zinc"], 14)
    p.lathe([(0.008, 0.064), (0.0145, 0.062), (0.0155, 0.056), (0.0155, 0.034), (0.0135, 0.03), (0.007, 0.029)],
            M["zinc"], sides=12)
    p.lathe([(0.0068, 0.03), (0.0068, 0.0), (0.0055, -0.1), (0.0015, -0.17), (0.0, -0.175)], M["steel"], sides=10)
    o = p.obj("agility_pole", col, 40)
    return _grime(o, lambda P: _mud(P, 0.3, (0.7, 0.62, 0.48), 0.0) * _mud(P, 0.035, (0.55, 0.45, 0.33), 0.0))


# -- goals and rebounder --------------------------------------------------------------------------

def mini_goal(col, M):
    """A 1.5 x 1.0 m steel mini goal: 32 mm posts and bar, a bent-tube ground frame and back
    struts sleeved on, a soft 100 mm knotted net hanging slack on clips, pegged at the back."""
    p = PM(0.003)
    w, h, d = 1.5, 1.0, 0.75
    frame = _powder("Prop_GoalFrame", "#e9e9e5", 21, chip="#a9abab", rough=0.38, chips=10)
    net = _net_mat("Prop_GoalNetBlack", "#26282a", 0.045, knot=1.8, rough=0.85)
    clip = _flat("Prop_NetClip", "#c9cbc8", 0.5)
    R, r2 = 0.016, 0.0125
    p.tube([(-w / 2, 0, 0.0), (-w / 2, 0, h), (w / 2, 0, h), (w / 2, 0, 0.0)], R, frame, 14, bend=0.06)
    p.tube([(-w / 2, -0.03, r2), (-w / 2, -d, r2), (w / 2, -d, r2), (w / 2, -0.03, r2)], r2, frame, 12, bend=0.09)
    for s in (-1, 1):
        x = s * w / 2
        p.tube([(x, -0.02, h - 0.06), (x, -0.22, h - 0.1), (x, -d + 0.04, r2 + 0.03)], 0.011, frame, 10, bend=0.12)
        p.r = 0.002
        p.cyl((x, 0, h - 0.085), (x, 0, h - 0.035), R + 0.003, frame, 14)        # strut sleeves
        p.cyl((x - 0.02 * s, -d + 0.02, r2), (x + 0.02 * s, -d + 0.02, r2), r2 + 0.003, frame, 12)
        p.cyl((x, 0, -0.002), (x, 0, 0.02), R + 0.0025, M["rubber"], 14)          # foot caps
        p.cyl((x, -0.012, r2), (x, -0.045, r2), r2 + 0.0025, M["rubber"], 12)
        # U-peg over the side bar
        p.r = 0.0
        y = -d * 0.55
        p.tube([(x - 0.028, y, -0.06), (x - 0.028, y, 0.031), (x + 0.028, y, 0.031), (x + 0.028, y, -0.06)], 0.0035,
               M["zinc"], 6, bend=0.008)
        p.r = 0.003
    p.r = 0.0
    # the net: roof/back sheet from the bar to the back bar, sagging and bellying out
    cell = 0.1
    A0, A1 = np.array([0, -0.035, h - 0.012]), np.array([0, -d + 0.03, 0.03])
    L = np.linalg.norm(A1 - A0)

    def back(u, v):
        q = A0 + (A1 - A0) * v + np.array([(u - 0.5) * (w - 0.04), 0, 0])
        b = math.sin(math.pi * u) * math.sin(math.pi * v) ** 0.8
        scal = 0.02 * abs(math.sin(8 * math.pi * u)) * (1 - v) ** 4         # sag between the bar's clips
        return q + np.array([0, -0.03 * b, -0.035 * b - scal])
    _grid_patch(p, back, 24, 9, net, lambda u, v: (u * (w - 0.04) / cell, v * L / cell))
    for s in (-1, 1):
        x = s * (w / 2 - 0.012)
        P0, P1, P2 = np.array([x, -0.03, 0.025]), np.array([x, -d + 0.03, 0.025]), np.array([x, -0.03, h - 0.03])

        def side(u, v, P0=P0, P1=P1, P2=P2, s=s):
            q = P0 + (P1 - P0) * u * (1 - v) + (P2 - P0) * v
            b = math.sin(math.pi * u) * math.sin(math.pi * v) * (1 - v)
            post = 0.015 * abs(math.sin(5 * math.pi * v)) * (1 - u) ** 4        # sag between the post's clips
            return q + np.array([s * (0.03 * b - post), -post * 0.5, -0.01 * b])
        _grid_patch(p, side, 10, 15, net, lambda u, v, P0=P0, P1=P1, P2=P2: (u * (1 - v) * np.linalg.norm(P1 - P0) / cell,
                                                                         v * np.linalg.norm(P2 - P0) / cell))

    def ground(u, v):
        x = (u - 0.5) * (w - 0.04)
        y = -0.03 - v * (d - 0.06)
        ramp = _sm(0.6, 0.95, v)                  # a little slack lying against the back bar
        fold = 0.004 + ramp * 0.012 * max(0.0, math.sin(v * 46.0 + 1.5 * math.sin(u * 7.0)))
        return np.array([x, y, fold])
    _grid_patch(p, ground, 10, 20, net, lambda u, v: (u * (w - 0.04) / cell, v * (d - 0.06) / cell))
    # U-pegs over the rear ground bar
    for xg in (-0.45, 0.45):
        p.r = 0.0
        p.tube([(xg, -d - 0.028, -0.06), (xg, -d - 0.028, 0.031), (xg, -d + 0.028, 0.031), (xg, -d + 0.028, -0.06)], 0.0035,
               M["zinc"], 6, bend=0.008)
    # bolts through the elbow and strut sleeves
    p.r = 0.0
    for s in (-1, 1):
        x = s * w / 2
        for c_, ax in (((x, 0, h - 0.06), (0, 1, 0)), ((x, -d + 0.02, r2), (0, 0, 1))):
            c_, ax = np.array(c_), np.array(ax, float)
            rr = R + 0.003 if ax[1] else r2 + 0.003
            p.cyl(c_ + ax * rr, c_ + ax * (rr + 0.005), 0.0065, M["zinc"], 6)
            p.cyl(c_ - ax * rr, c_ - ax * (rr + 0.004), 0.005, M["zinc"], 8)
    # net clips on the bar, posts, struts and ground frame
    for k in range(1, 8):
        _ring_round(p, (-w / 2 + k * w / 8, 0, h), (1, 0, 0), R + 0.0035, 0.0022, clip)
    for s in (-1, 1):
        for k in range(1, 5):
            _ring_round(p, (s * w / 2, 0, k * h / 5), (0, 0, 1), R + 0.0035, 0.0022, clip)
            _ring_round(p, (s * w / 2, -k * d / 5, r2), (0, 1, 0), r2 + 0.0035, 0.0022, clip)
        for t in (0.3, 0.6):
            a, b = np.array([s * w / 2, -0.22, h - 0.1]), np.array([s * w / 2, -d + 0.04, r2 + 0.03])
            _ring_round(p, a + (b - a) * t, b - a, 0.0145, 0.0022, clip)
    for k in range(1, 6):
        _ring_round(p, (-w / 2 + k * w / 6, -d, r2), (1, 0, 0), r2 + 0.0035, 0.0022, clip)
    o = p.obj("mini_goal", col)
    return _grime(o, lambda P: _mud(P, 0.14, (0.68, 0.66, 0.52)))


def rebounder(col, M):
    """A 1.5 m football rebounder: 32 mm powder-coated frame tilted back 20 degrees on rubber pads,
    a U-shaped rear support hinged on pin brackets, a knotted shock-cord net laced to the frame
    with bungee loops."""
    p = PM(0.003)
    w = 1.5
    frame = _powder("Prop_Rebounder", "#1c1d1f", 22, chip="#4c4f52", rough=0.5, chips=14)
    net = _net_mat("Prop_ReboundNetTaut", "#d9d8d0", 0.055, knot=1.7, twist=True)
    cord = _flat("Prop_Bungee", "#d4531c", 0.55)
    tilt = 0.35
    c, s_ = math.cos(tilt), math.sin(tilt)
    R = 0.016
    zb = 0.033
    back = np.array([0, -c, -s_])                     # the net plane's normal, away from the kicker

    def at(x, u):
        return np.array([x, -u * s_, zb + u * c])
    p.tube([at(-w / 2, 0), at(w / 2, 0), at(w / 2, w), at(-w / 2, w)], R, frame, 12, bend=0.08, closed=True)
    for x in (-0.55, 0.55):                           # rubber pads under the bottom bar
        p.r = 0.004
        p.box((x - 0.035, -0.03, 0.0), (x + 0.035, 0.03, zb - R + 0.002), M["rubber"])
    # rear support: legs hinged outside the side bars, a ground bar between the feet
    xo = w / 2 + R + 0.0135
    hz = 0.62 * w
    H = at(xo, hz)
    foot_y = H[1] - 0.95
    p.r = 0.003
    p.tube([(-xo, H[1], H[2]), (-xo, foot_y, 0.0135), (xo, foot_y, 0.0135), (xo, H[1], H[2])], 0.0125, frame, 10, bend=0.07)
    for sx in (-1, 1):
        x = sx * (w / 2 + R + 0.002)
        # angle ratchet: a round toothed hub welded to the side bar, the leg's hub against it, a
        # knurled locking knob through both
        cen = at(x, hz)
        up = at(0, 1) - at(0, 0)
        up /= np.linalg.norm(up)
        ex = np.array([sx, 0, 0.0])
        p.r = 0.0
        p.cyl(cen - ex * 0.004, cen + ex * 0.008, 0.03, frame, 18)
        p.cyl(cen + ex * 0.008, cen + ex * 0.0135, 0.027, frame, 18)
        p.cyl(cen + ex * 0.0135, cen + ex * 0.0265, 0.029, frame, 18)
        p.r = 0.0
        # clamp plate and a 60 mm five-lobed star knob on a threaded boss
        side = np.cross(ex, up)
        p.r = 0.0015
        pl = [cen + up * a_ + side * b_ for a_, b_ in ((-0.045, -0.03), (0.045, -0.03), (0.045, 0.03), (-0.045, 0.03))]
        V8 = [q_ + ex * 0.0265 for q_ in pl] + [q_ + ex * 0.0305 for q_ in pl]
        pc = np.mean(V8, 0)
        for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            ff = [V8[i] for i in f]
            _face_out(p, ff, frame, np.mean(ff, 0) - pc)
        p.r = 0.0
        p.cyl(cen + ex * 0.0305, cen + ex * 0.042, 0.009, M["zinc"], 10)
        star = []
        for k in range(30):
            a_ = 2 * math.pi * k / 30
            rr_ = 0.022 + 0.008 * math.cos(5 * a_)
            star.append((rr_ * math.cos(a_), rr_ * math.sin(a_)))
        for zz0, zz1, sc in ((0.042, 0.06, 1.0),):
            bot = [cen + ex * zz0 + (side * x_ + up * y_) * sc for x_, y_ in star]
            topr = [cen + ex * zz1 + (side * x_ + up * y_) * sc * 0.92 for x_, y_ in star]
            kc = cen + ex * (zz0 + zz1) / 2
            for k in range(30):
                q_ = [bot[k], bot[(k + 1) % 30], topr[(k + 1) % 30], topr[k]]
                _face_out(p, q_, M["rubber"], np.mean(q_, 0) - kc)
            _face_out(p, topr, M["rubber"], ex)
            _face_out(p, bot, M["rubber"], -ex)
        p.r = 0.003
        p.cyl((sx * xo, foot_y + 0.0, 0.0135), (sx * (xo + 0.03), foot_y, 0.0135), 0.0145, M["rubber"], 10)
    # the net: sags back a little in the middle, laced at 7 points a side
    m = 0.075
    cell = 0.1
    span = w - 2 * m

    def sheet(u, v):
        q = at(-w / 2 + m + u * span, m + v * span)
        return q + back * (0.012 * math.sin(math.pi * u) * math.sin(math.pi * v))
    p.r = 0.0
    _grid_patch(p, sheet, 12, 12, net, lambda u, v: (u * span / cell + 0.5, v * span / cell + 0.5))
    edge = [sheet(u, 0) for u in np.linspace(0, 1, 9)] + [sheet(1, v) for v in np.linspace(0, 1, 9)[1:]] + \
           [sheet(u, 1) for u in np.linspace(1, 0, 9)[1:]] + [sheet(0, v) for v in np.linspace(1, 0, 9)[1:-1]]
    p.tube(edge, 0.003, _flat("Prop_NetRope", "#e4e4de", 0.7), 4, bend=0.0, closed=True)
    # bungee loops: from the net's edge out to the frame, stretched at about 30 degrees (alternate
    # ways), a couple hanging slack
    rng = np.random.default_rng(9)
    k_all = 0
    for k in range(7):
        t = (k + 0.5) / 7
        for (eu, ev), (fx, fu), ax in (((t, 0), (-w / 2 + m + t * span, 0), (1, 0, 0)), ((t, 1), (-w / 2 + m + t * span, w), (1, 0, 0)),
                                       ((0, t), (-w / 2, m + t * span), up), ((1, t), (w / 2, m + t * span), up)):
            k_all += 1
            E = sheet(eu, ev)
            a = np.asarray(ax, float)
            a /= np.linalg.norm(a)
            F = at(fx, fu) + a * (0.6 * m * (1 if k_all % 2 else -1))
            d = E - F
            d -= a * (d @ a)
            d /= np.linalg.norm(d)
            e2 = np.cross(a, d)
            rr = R + 0.004
            slack = 0.02 if rng.random() < 0.15 else 0.0
            mid = (E + F) / 2 - np.array([0, 0, slack])
            wrap = [F + (d * math.cos(th) + e2 * math.sin(th)) * rr for th in np.linspace(0.45, 2 * math.pi - 0.45, 9)]
            loop = [E, mid + e2 * 0.004] + wrap + [mid - e2 * 0.004]
            p.tube(loop, 0.003, cord, 5, bend=0.0, closed=True)
            p.sphere(E + back * 0.004, 0.009, M["black"], 6, 4)
    # a maker's label on the top rail, facing the player
    from . import lettering
    if not bpy.data.materials.get("Prop_ReboundLabel"):
        sg = lettering.sign("REBOUND PRO", 48, "#f4f4f0", "#c4161c", pad=0.25, weight=1.1)
        _tex_mat("Prop_ReboundLabel", sg, 0.35)
    lab = bpy.data.materials["Prop_ReboundLabel"]
    tc = at(0.0, w) - back * (R + 0.0006)
    p.decal(tc, -back, at(0, 1) - at(0, 0), 0.12, 0.016, lab, lift=0.0)
    o = p.obj("rebounder", col)

    def dirt(P):
        # mud on the feet and bottom rail, ball marks darkening the net's middle and lower third
        u_ = P[:, 2] - zb
        centre = np.exp(-((P[:, 0] / 0.35) ** 2 + ((u_ - 0.6) / 0.35) ** 2))
        low = 1 - _sm(0.1, 0.55, u_)
        k = np.clip(0.12 * centre + 0.1 * low, 0, 0.3)[:, None]
        return _mud(P, 0.12, (0.62, 0.56, 0.44)) * (1 - k * (1 - np.array([0.7, 0.66, 0.55])[None]))
    return _grime(o, dirt)


# -- footballs ------------------------------------------------------------------------------------

def _truncated_icosahedron():
    """The classic ball's panels: 12 pentagons, 20 hexagons (vertex rings, counter-clockwise from
    outside) on a truncated icosahedron round the origin."""
    phi = (1 + 5 ** 0.5) / 2
    ico = []
    for a in (-1, 1):
        for b in (-1, 1):
            ico += [(0, a, b * phi), (a, b * phi, 0), (b * phi, 0, a)]
    ico = np.array(ico, float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    d = np.linalg.norm(ico[:, None] - ico[None], axis=2)
    emin = d[d > 1e-6].min()
    nb = [[j for j in range(12) if abs(d[i, j] - emin) < 1e-6] for i in range(12)]
    pts = {(i, j): ico[i] * 2 / 3 + ico[j] / 3 for i in range(12) for j in nb[i]}

    def ccw(ring, c):
        u = ring[0] - c * (ring[0] @ c)
        u /= np.linalg.norm(u)
        v = np.cross(c, u)
        return sorted(ring, key=lambda q: math.atan2(q @ v, q @ u))
    pent = [ccw([pts[(i, j)] for j in nb[i]], ico[i]) for i in range(12)]
    hexes = []
    for i in range(12):
        for j in nb[i]:
            for k in nb[i]:
                if k in nb[j] and i < j < k:
                    c = (ico[i] + ico[j] + ico[k]) / 3
                    hexes.append(ccw([pts[(i, j)], pts[(j, i)], pts[(j, k)], pts[(k, j)], pts[(k, i)], pts[(i, k)]],
                                     c / np.linalg.norm(c)))
    return pent, hexes


_DESIGNS = {
    "classic": (lambda i: 1 if i in (3, 14) else 8 if i == 9 else 2 if i in (5, 17) else 0, lambda i: 6 if i in (2, 7) else 3),
    "hivis": (lambda i: 4, lambda i: 3),
    "winter": (lambda i: 5, lambda i: 7),
    "navy": (lambda i: 8 if i in (1, 12) else 0, lambda i: 7),
    "used": (lambda i: 2 if i % 3 else (1 if i == 3 else 0), lambda i: 6 if i % 2 else 3),
}


def _ball(p, mat, centre, R=0.11, sub=5, rot=None, design="classic", squash=0.0):
    """A 32-panel ball: each panel fanned and subdivided, pillowed (seams sunk 0.7 mm), its UVs
    into its atlas cell (training_paint.ball_atlas)."""
    hexc, pentc = _DESIGNS[design]
    pent, hexes = _truncated_icosahedron()
    if design == "classic":
        # the wordmark twice, on opposite panels
        cs = np.array([np.mean(h_, 0) for h_ in hexes])
        opp = int(np.argmin(cs @ cs[3]))
        base_hexc = hexc
        hexc = lambda i, b=base_hexc, o=opp: 1 if i == o else b(i)
    rot = np.eye(3) if rot is None else rot
    c0 = np.asarray(centre, float)
    a_hex = np.linalg.norm(hexes[0][0] - np.mean(hexes[0], 0))
    K = TP.BALL_RHO / a_hex
    pillow = 0.017
    for kind, rings in (("p", pent), ("h", hexes)):
        for idx, ring in enumerate(rings):
            n = len(ring)
            cen = np.mean(ring, 0)
            rho = np.linalg.norm(ring[0] - cen)
            e1 = (ring[0] - cen) / rho
            nrm = cen / np.linalg.norm(cen)
            e2 = np.cross(nrm, e1)
            apo = rho * math.cos(math.pi / n)
            mids = [(ring[k] + ring[(k + 1) % n]) / 2 - cen for k in range(n)]
            mids = [m_ / np.linalg.norm(m_) for m_ in mids]
            cu, cv = TP.ball_cell(pentc(idx) if kind == "p" else hexc(idx))

            def vert(q):
                l_ = q - cen
                dd = apo - max(l_ @ m_ for m_ in mids)
                t = min(max(dd, 0.0) / (0.5 * apo), 1.0)
                r = R * (1 - pillow + pillow * (t * t * (3 - 2 * t)) ** 0.7)
                d_ = rot @ (q / np.linalg.norm(q)) * r
                if squash:                              # sat under the weight of the pile
                    d_ = d_ * np.array([1 + squash / 2, 1 + squash / 2, 1 - squash])
                return c0 + d_ - np.array([0, 0, R * squash]), (cu + (l_ @ e1) * K, cv + (l_ @ e2) * K)
            for a, b in zip(ring, ring[1:] + ring[:1]):
                G = {(i, j): vert(cen + (a - cen) * (i / sub) + (b - cen) * (j / sub))
                     for i in range(sub + 1) for j in range(sub + 1 - i)}
                tris = []
                for i in range(sub):
                    for j in range(sub - i):
                        tris.append([(i, j), (i + 1, j), (i, j + 1)])
                        if i + j < sub - 1:
                            tris.append([(i + 1, j), (i + 1, j + 1), (i, j + 1)])
                P0 = np.array([G[k][0] for k in tris[0]])
                flip = np.cross(P0[1] - P0[0], P0[2] - P0[0]) @ (rot @ nrm) < 0
                for tr in tris:
                    tr = tr[::-1] if flip else tr
                    p.face([G[k][0] for k in tr], mat, [G[k][1] for k in tr])


def ball(col, M):
    """A size 5 football (Ø 220 mm), 32 machine-stitched panels: pillowed panels, sunk seams with
    stitching, pebble-grained PU with a printed graphic, light turf marks; resting 3 mm into turf."""
    p = PM(0.0)
    _ball(p, _ball_mat(), (0, 0, 0.107), sub=5, rot=_rot(np.random.default_rng(12)))
    o = p.obj("ball", col, 60, bevel=False)

    def stains(P):
        # training wear: grass and mud smudged over the side it is struck and rolled on most
        q = P - np.array([0, 0, 0.107])
        nq = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        side = np.clip(nq @ np.array([0.6, -0.3, -0.74]), 0, 1)
        g = 0.5 + 0.5 * np.sin(q[:, 0] * 41 + 1.3) * np.sin(q[:, 1] * 37 + 0.4) * np.sin(q[:, 2] * 29 + 2.1)
        s_ = np.clip((g - 0.45) * 2.0, 0, 0.6) * (0.3 + 0.9 * side)
        tone = np.where(g[:, None] > 0.75, np.array([[0.28, 0.2, 0.42]]), np.array([[0.22, 0.12, 0.3]]))
        return 1 - s_[:, None] * tone
    return _grime(o, stains)


def ball_cart(col, M):
    """A ball cart (0.8 x 0.6 m, 1.1 m to the handle): green powder-coated tube frame, welded
    50 mm galvanised mesh, a hinged mesh lid standing open, swivel castors (two braked), loaded
    with a settled pile of mixed training balls."""
    p = PM(0.003)
    frame = _powder("Prop_BallCartWorn", "#22382a", 23, chip="#8a8f8b", rough=0.55, chips=70)
    mesh = _net_mat("Prop_CartMesh", "#a7abad", 0.072, knot=1.3, twist=False, weld=True, metal=0.7, rough=0.42)
    bm = _ball_mat()
    W, D = 0.8, 0.6
    sx, sy = W / 2, D / 2
    z0, z1 = 0.2, 0.92
    r = 0.0125
    for z in (z0, (z0 + z1) / 2, z1):
        p.tube([(-sx, -sy, z), (sx, -sy, z), (sx, sy, z), (-sx, sy, z)], r, frame, 10, bend=0.04, closed=True)
    for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)):
        p.cyl((x, y, 0.128), (x, y, z1), r, frame, 10)
        # corner gusset at the base
        p.r = 0.0
        dx, dy = -math.copysign(1, x), -math.copysign(1, y)
        for ex, ey in ((dx, 0), (0, dy)):
            tri = [(x, y, z0 + 0.012), (x + ex * 0.07, y + ey * 0.07, z0 + 0.012), (x, y, z0 + 0.08)]
            nrm = np.array([ey, ex, 0.0]) * 0.002
            q0 = [np.asarray(t) - nrm for t in tri]
            q1 = [np.asarray(t) + nrm for t in tri]
            p.faces(q0 + q1, [(0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)], frame)
        # swivel castor: plate, kingpin, offset fork, wheel, axle (brakes on the handle end)
        p.r = 0.0
        p.box((x - 0.035, y - 0.035, 0.12), (x + 0.035, y + 0.035, 0.128), M["zinc"])
        p.cyl((x, y, 0.104), (x, y, 0.12), 0.014, M["zinc"], 12)
        trail = -0.022
        for fx in (-0.02, 0.02):
            p.box((x + fx - 0.002, y + trail - 0.03, 0.045), (x + fx + 0.002, y + 0.015, 0.106), M["zinc"])
        p.box((x - 0.022, y - 0.022, 0.1), (x + 0.022, y + 0.018, 0.106), M["zinc"])
        tyre(p, (x, y + trail, 0.048), 0.048, 0.03, 0.022, M["rubber"], M["grey"], sides=14)
        for hx in (-0.017, 0.017):                       # hub caps
            p.cyl((x + hx, y + trail, 0.048), (x + hx * 1.15, y + trail, 0.048), 0.012, M["zinc"], 10)
        for bx in (-0.025, 0.025):                       # plate bolts
            for by in (-0.025, 0.025):
                p.cyl((x + bx, y + by, 0.128), (x + bx, y + by, 0.132), 0.0045, M["zinc"], 6)
        p.cyl((x - 0.024, y + trail, 0.048), (x + 0.024, y + trail, 0.048), 0.005, M["zinc"], 8)
        if x < 0:
            p.obox((x, y + trail + 0.035, 0.085), (0.03, 0.04, 0.005), M["red"], pitch=-0.5)
    # push handle with a rubber grip on the -x end
    p.r = 0.003
    p.tube([(-sx, -0.22, z1), (-sx - 0.13, -0.22, 1.08), (-sx - 0.13, 0.22, 1.08), (-sx, 0.22, z1)], r, frame, 10, bend=0.05)
    p.cyl((-sx - 0.13, -0.17, 1.08), (-sx - 0.13, 0.17, 1.08), 0.0165, M["rubber"], 12)
    # mesh walls and floor
    p.r = 0.0
    cm = 0.05
    for a, b, c_, d in (((-sx, -sy, z0), (sx, -sy, z0), (sx, -sy, z1), (-sx, -sy, z1)), ((sx, sy, z0), (-sx, sy, z0), (-sx, sy, z1), (sx, sy, z1)),
                        ((sx, -sy, z0), (sx, sy, z0), (sx, sy, z1), (sx, -sy, z1)), ((-sx, sy, z0), (-sx, -sy, z0), (-sx, -sy, z1), (-sx, sy, z1))):
        L = np.linalg.norm(np.subtract(b, a))
        p.quad(a, b, c_, d, mesh, uv=[(0, 0), (L / cm, 0), (L / cm, (z1 - z0) / cm), (0, (z1 - z0) / cm)])
    p.quad((-sx, -sy, z0 + 0.004), (sx, -sy, z0 + 0.004), (sx, sy, z0 + 0.004), (-sx, sy, z0 + 0.004), mesh,
           uv=[(0, 0), (W / cm, 0), (W / cm, D / cm), (0, D / cm)])
    # clip tabs where the mesh is welded to the rails
    p.r = 0.0
    for z in (z0, z1):
        for k in range(1, 8):
            x = -sx + k * W / 8
            for y in (-sy, sy):
                p.box((x - 0.008, y - 0.004, z - 0.016), (x + 0.008, y + 0.004, z + 0.002), frame)
        for k in range(1, 6):
            y = -sy + k * D / 6
            for x in (-sx, sx):
                p.box((x - 0.004, y - 0.008, z - 0.016), (x + 0.004, y + 0.008, z + 0.002), frame)
    # the lid, hinged on the back rail, standing open past upright
    th = math.radians(104)
    hz = z1 + 0.012

    def lid(x, dy):
        return np.array([x, -sy + dy * math.cos(th), hz + dy * math.sin(th)])
    p.r = 0.003
    p.tube([lid(-sx + 0.01, 0.01), lid(sx - 0.01, 0.01), lid(sx - 0.01, D - 0.01), lid(-sx + 0.01, D - 0.01)], 0.01, frame, 8,
           bend=0.04, closed=True)
    p.r = 0.0
    p.quad(lid(-sx + 0.01, 0.01), lid(sx - 0.01, 0.01), lid(sx - 0.01, D - 0.01), lid(-sx + 0.01, D - 0.01), mesh,
           uv=[(0, 0), (W / cm, 0), (W / cm, D / cm), (0, D / cm)])
    p.r = 0.0
    for x in (-0.25, 0.25):
        p.cyl((x - 0.035, -sy, hz), (x + 0.035, -sy, hz), 0.011, M["zinc"], 10)
    hp = lid(0, D - 0.01)
    p.obox(hp + np.array([0, 0.008, 0]), (0.025, 0.004, 0.05), M["black"], pitch=th - math.pi / 2)
    p.r = 0.0
    # a brass padlock hanging from the staple
    brass = _flat("Prop_Brass", "#b8913e", 0.35, 1.0)
    p.tube([(0.0, sy + 0.018, z1 - 0.016), (0.0, sy + 0.03, z1 - 0.024), (0.0, sy + 0.03, z1 - 0.05)], 0.0028, M["zinc"], 6, bend=0.008)
    p.tube([(0.0, sy + 0.012, z1 - 0.05), (0.0, sy + 0.012, z1 - 0.022), (0.0, sy + 0.022, z1 - 0.012)], 0.0028, M["zinc"], 6, bend=0.008)
    p.r = 0.003
    p.box((-0.02, sy + 0.012, z1 - 0.09), (0.02, sy + 0.032, z1 - 0.05), brass)
    p.r = 0.0
    p.tube([(-0.012, sy, z1 - 0.02), (-0.012, sy + 0.02, z1 - 0.02), (0.012, sy + 0.02, z1 - 0.02), (0.012, sy, z1 - 0.02)],
           0.003, M["zinc"], 6, bend=0.006)
    # lid stay: a flat bar pinned to the lid's side frame and the top rail
    for sxs in (-1, 1):
        A = lid(sxs * (sx - 0.012), 0.28)
        B = np.array([sxs * (sx - 0.012), -sy + 0.2, z1 + 0.012])
        p.tube([A, B], 0.004, M["zinc"], 6)
        for q in (A, B):
            p.cyl(q - np.array([0.006 * sxs, 0, 0]), q + np.array([0.006 * sxs, 0, 0]), 0.008, M["zinc"], 8)
    # the balls: dropped in and settled under gravity (each comes to rest on the floor, the mesh
    # or the balls below; none overlap)
    rng = np.random.default_rng(8)
    R = 0.11
    lo = np.array([-sx + R + 0.009, -sy + R + 0.009, z0 + 0.006 + R])
    hi = np.array([sx - R - 0.009, sy - R - 0.009, 9.0])
    designs = ["classic"] * 4 + ["used"] * 5 + ["hivis"] * 4 + ["winter"] * 3 + ["navy"] * 2
    rng.shuffle(designs)
    n = len(designs)
    X = np.column_stack([rng.uniform(lo[0], hi[0], n), rng.uniform(lo[1], hi[1], n), lo[2] + 0.25 * np.arange(n) + 0.3])
    for it in range(1500):
        if it < 1200:
            X[:, 2] -= 0.004
        for _ in range(4):
            D = X[:, None] - X[None]
            dist = np.linalg.norm(D, axis=2) + np.eye(n)
            over = np.clip(2 * R + 0.0005 - dist, 0, None) * (1 - np.eye(n))
            push = (D / dist[..., None] * over[..., None] * 0.5).sum(1)
            X += push
            X = np.clip(X, lo, hi)
    for _ in range(600):                                 # let the pile relax until nothing overlaps
        D = X[:, None] - X[None]
        dist = np.linalg.norm(D, axis=2) + np.eye(n)
        over = np.clip(2 * R + 0.001 - dist, 0, None) * (1 - np.eye(n))
        if over.max() < 0.0015:
            break
        X += (D / dist[..., None] * over[..., None] * 0.5).sum(1)
        X[:, 2] += 0.0005 * (over.sum(1) > 0)              # an over-pressed ball rides up rather than squeezing in
        X = np.clip(X, lo, hi)
    low = np.argsort(X[:, 2])[:2]
    for i_, (c_, des) in enumerate(zip(X, designs)):
        _ball(p, bm, c_, R, sub=2 if c_[2] > z1 - 0.32 else 1, rot=_rot(rng), design=des, squash=0.06 if i_ in low else 0.0)
    # weld beads where the rails meet the corner posts
    for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)):
        for z in (z0, (z0 + z1) / 2, z1):
            _ring_round(p, (x, y, z), (0, 0, 1), r + 0.0015, 0.0022, frame, 8, 3)
    o = p.obj("ball_cart", col, 50)

    def dirt(P):
        rust = _mud(P, 0.3, (0.78, 0.62, 0.5), 0.1)
        return rust * _mud(P, 0.2, (0.68, 0.62, 0.48))
    return _grime(o, dirt)


# -- ladder and shelter ---------------------------------------------------------------------------

def _ribbon(p, P, Z, width, thick, mat, closed=False):
    """A flat webbing strap along the plan polyline P (N, 2) at heights Z, width x thickness, UVs in
    metres (u along, v across)."""
    P = np.asarray(P, float)
    n = len(P)
    acc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1))])
    L, R_ = [], []
    for i in range(n):
        if closed:
            d = P[(i + 1) % n] - P[i - 1]
        else:
            d = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        d /= np.linalg.norm(d)
        nr = np.array([-d[1], d[0]])
        L.append(P[i] + nr * width / 2)
        R_.append(P[i] - nr * width / 2)
    segs = range(n) if closed else range(n - 1)
    for i in segs:
        j = (i + 1) % n
        zt0, zt1, zb0, zb1 = Z[i] + thick, Z[j] + thick, Z[i], Z[j]
        s0, s1 = acc[i], acc[i + 1]
        uv = [(s0, 0), (s1, 0), (s1, width), (s0, width)]
        _face_out(p, [(*R_[i], zt0), (*R_[j], zt1), (*L[j], zt1), (*L[i], zt0)], mat, (0, 0, 1), uv)
        _face_out(p, [(*R_[i], zb0), (*R_[j], zb1), (*L[j], zb1), (*L[i], zb0)], mat, (0, 0, -1), uv)
        for A, sg in ((L, 1), (R_, -1)):
            q = [(*A[i], zb0), (*A[j], zb1), (*A[j], zt1), (*A[i], zt0)]
            _face_out(p, q, mat, (*(A[i] - P[i]), 0), [(s0, 0), (s1, 0), (s1, thick), (s0, thick)])
    if not closed:
        for i, sg in ((0, -1), (n - 1, 1)):
            d = P[i] - P[i - sg] if 0 <= i - sg < n else np.array([1.0, 0])
            q = [(*L[i], Z[i]), (*R_[i], Z[i]), (*R_[i], Z[i] + thick), (*L[i], Z[i] + thick)]
            _face_out(p, q, mat, (*d, 0))


def agility_ladder(col, M):
    """A 4 m speed ladder lying on the grass (the sward pressed to about 22 mm under it): nine
    moulded 430 x 40 x 4 mm slats with a cambered top, each end in a 50 x 50 x 12 mm clip block the
    straps thread through; two 25 x 1.5 mm nylon webbing straps that lift through the clips and
    sag into the grass between them, wandering a little, with a sewn carry loop at every end."""
    p = PM(0.0)
    rung = _powder("Prop_LadderSlat", "#e7ce3f", 24, chip="#b39a2e", rough=0.55, chips=0, tile=0.4)
    wa, wn, wr = TP.webbing()
    strap = _tex_mat("Prop_Webbing", wa, wr, wn, tile=(0.04, 0.04))
    ZB = 0.022                                   # the top of the pressed sward
    rng = np.random.default_rng(4)
    n, pitch, half = 9, 0.47, 0.215
    rungs = []
    for k in range(n):
        x = -(n - 1) * pitch / 2 + k * pitch + rng.normal(0, 0.012)
        rungs.append((x, rng.normal(0, 0.01), rng.normal(0, math.radians(2.0))))
    clips = {-1: [], 1: []}
    for k, (x, y, yaw) in enumerate(rungs):
        c, s = math.cos(yaw), math.sin(yaw)

        def xf(pts, x=x, y=y, c=c, s=s):
            return [(x + px * c - py * s, y + px * s + py * c, pz) for px, py, pz in pts]
        rings = [rrect(0.04, 0.43, 0.012, ZB, n=3), rrect(0.04, 0.43, 0.012, ZB + 0.0028, n=3),
                 rrect(0.036, 0.426, 0.011, ZB + 0.0036, n=3), rrect(0.024, 0.414, 0.009, ZB + 0.0046, n=3)]
        p.r = 0.0
        p.loft([xf(r_) for r_ in rings], rung)
        for sy in (-1, 1):
            cx, cy, _ = xf([(0, sy * (half - 0.005), 0)])[0]
            p.r = 0.002
            p.obox((cx, cy, ZB + 0.006), (0.05, 0.05, 0.012), rung, yaw=yaw)
            p.r = 0.0
            for sx in (-1, 1):                      # the slot the strap threads through
                fc = np.array([cx + sx * 0.0251 * c, cy + sx * 0.0251 * s, ZB + 0.0062])
                p.decal(fc, (sx * c, sx * s, 0), (0, 0, 1), 0.029, 0.0035, M["black"], lift=0.0004)
            clips[sy].append((cx, cy, c, s))
    tw, th = 0.025, 0.0015
    for sy in (-1, 1):
        P, Z = [], []
        cl = clips[sy]
        x0, y0, c0, s0 = cl[0]
        for dx in (-0.2, -0.13, -0.07):
            P.append((x0 + dx, y0 + rng.normal(0, 0.006)))
            Z.append(ZB - 0.003 if dx < -0.1 else ZB + 0.001)
        for k, (cx, cy, c, s) in enumerate(cl):
            for dx, dz in ((-0.032, ZB + 0.0045), (0.0, ZB + 0.0055), (0.032, ZB + 0.0045)):
                P.append((cx + dx * c, cy + dx * s))
                Z.append(dz)
            if k < n - 1:
                nx, ny = cl[k + 1][0], cl[k + 1][1]
                for f in (0.2, 0.4, 0.6, 0.8):
                    sag = math.sin(math.pi * f)
                    P.append((cx + (nx - cx) * f, cy + (ny - cy) * f + sag * rng.normal(0, 0.012)))
                    Z.append(ZB + 0.004 - sag * rng.uniform(0.006, 0.012))
        xl, yl = cl[-1][0], cl[-1][1]
        for dx in (0.07, 0.13, 0.2):
            P.append((xl + dx, yl + rng.normal(0, 0.006)))
            Z.append(ZB - 0.003 if dx > 0.1 else ZB + 0.001)
        _ribbon(p, P, Z, tw, th, strap)
        # sewn carry loops at both ends: the strap folded back into a flat teardrop
        for (ex, ey), dirn in ((P[0], -1), (P[-1], 1)):
            loop = [(ex + dirn * (0.06 * (1 - math.cos(t))), ey + 0.028 * math.sin(t) * sy) for t in np.linspace(0, 2 * math.pi, 15)[:-1]]
            _ribbon(p, loop, [ZB - 0.002] * len(loop), tw * 0.9, th, strap, closed=True)
            p.box((ex - 0.012, ey - tw / 2 - 0.001, ZB - 0.002), (ex + 0.012, ey + tw / 2 + 0.001, ZB + 0.0012), strap)
    o = p.obj("agility_ladder", col, 40)
    tints = {k: 0.92 + 0.08 * rng.random() for k in range(n)}

    def per_rung(P):
        out = np.ones((len(P), 3))
        for k, (x, y, yaw) in enumerate(rungs):
            m = (np.abs(P[:, 0] - x) < 0.045) & (np.abs(P[:, 1] - y) < 0.26) & (P[:, 2] > ZB - 0.0005)
            out[m] *= tints[k]
        under = 1 - _sm(ZB - 0.001, ZB + 0.004, P[:, 2])
        return out * (1 - under[:, None] * (1 - np.array([0.72, 0.7, 0.55])[None]))
    return _grime(o, per_rung)


def _seat_shell(p, x, y, mat):
    """A moulded bucket seat (polypropylene shell, 5 mm) centred at x, its pan front edge at y."""
    ns, nt = 6, 8
    th = 0.005

    def surf(s, t):
        # side profile: front lip, pan, the curve into a back reclined 14 degrees
        if t < 0.45:
            f = t / 0.45
            yy, zz = 0.0 - 0.36 * f, 0.445 - 0.03 * f - 0.012 * math.sin(math.pi * f)
            yy -= 0.012 * math.sin(math.pi * min(1, f * 4)) * (f < 0.25)
        else:
            f = (t - 0.45) / 0.55
            yy = -0.36 - 0.05 * math.sin(min(1.0, f * 3) * math.pi / 2) - 0.08 * f
            zz = 0.415 + 0.38 * f + 0.03 * math.sin(min(1.0, f * 3) * math.pi / 2)
        hw = 0.215 * (1 - 0.05 * t) - 0.04 * max(0.0, (t - 0.88) / 0.12) ** 2
        xx = s * hw
        # dished: the sides rise on the pan and wrap forward on the back
        if t < 0.45:
            zz += 0.03 * s * s
        else:
            yy += 0.045 * s * s
        return np.array([x + xx, y + yy, zz])

    def nrm(s, t):
        a = surf(min(s + 0.01, 1), t) - surf(max(s - 0.01, -1), t)
        b = surf(s, min(t + 0.01, 1)) - surf(s, max(t - 0.01, 0))
        n = np.cross(a, b)
        return n / np.linalg.norm(n)
    S = [-1 + 2 * i / ns for i in range(ns + 1)]
    T = [j / nt for j in range(nt + 1)]
    top = [[surf(s, t) for t in T] for s in S]
    bot = [[surf(s, t) - nrm(s, t) * th for t in T] for s in S]
    for i in range(ns):
        for j in range(nt):
            q = [top[i][j], top[i + 1][j], top[i + 1][j + 1], top[i][j + 1]]
            n = nrm((S[i] + S[i + 1]) / 2, (T[j] + T[j + 1]) / 2)
            _face_out(p, q, mat, n)
            _face_out(p, [bot[i][j], bot[i + 1][j], bot[i + 1][j + 1], bot[i][j + 1]], mat, -n)
    rim = [(i, 0) for i in range(ns)] + [(ns, j) for j in range(nt)] + [(i, nt) for i in range(ns, 0, -1)] + [(0, j) for j in range(nt, 0, -1)]
    cen = np.mean([top[i][j] for i in range(ns + 1) for j in range(nt + 1)], 0)
    for k in range(len(rim)):
        (i0, j0), (i1, j1) = rim[k], rim[(k + 1) % len(rim)]
        q = [top[i0][j0], top[i1][j1], bot[i1][j1], bot[i0][j0]]
        _face_out(p, q, mat, np.mean(q, 0) - cen)
    return surf, nrm


def team_shelter(col, M):
    """A 6 m team shelter facing +y: five 60 x 40 mm aluminium bows (full arches at the ends,
    J-bows between, each a glazing bar where the sheets join), a front fascia gutter, clear
    polycarbonate back, roof and ends (dust, water streaks, a grimy drip line), a riveted
    club-colour kick panel with joints and scuffs, an artificial-turf floor mat inside a 100 x 50 mm
    base channel bolted down, and nine numbered bucket seats on a twin-tube beam."""
    from . import lettering
    from .props import round_path
    p = PM(0.004)
    W = 6.0
    Y0, Y1 = -0.85, 0.9
    alu = _flat("Prop_ShelterAluSatin", "#c3c7ca", 0.32, 0.25)
    if not bpy.data.materials.get("Prop_ShelterPoly"):
        _tex_mat("Prop_ShelterPoly", TP.shelter_poly(), 0.06, alpha=True, blend=True, spec=0.8)
    poly = bpy.data.materials["Prop_ShelterPoly"]
    club = stand.painted("TINT_club", "#c8102e", 0.4)
    seat_red = _flat("Prop_ShelterSeat", "#af1e23", 0.45)
    base = _powder("Prop_ShelterBase", "#2a2c2f", 25, chip="#8d9093", rough=0.5, chips=20)
    if not bpy.data.materials.get("Prop_TurfMat"):
        ta, tn, tr = TP.turf_mat()
        _tex_mat("Prop_TurfMat", ta, tr, tn, tile=(0.5, 0.5))
    turf = bpy.data.materials["Prop_TurfMat"]
    if not bpy.data.materials.get("Prop_PanelGrime"):
        _tex_mat("Prop_PanelGrime", TP.grime_strip(), 0.8, alpha=True, blend=True)
    grime = bpy.data.materials["Prop_PanelGrime"]
    if not bpy.data.materials.get("Prop_Rivets"):
        _tex_mat("Prop_Rivets", TP.rivets(), 0.35, alpha=True, metal=1.0)
    rivet = bpy.data.materials["Prop_Rivets"]
    if not bpy.data.materials.get("Prop_SeatNumbers"):
        _tex_mat("Prop_SeatNumbers", TP.digits(), 0.5, alpha=True)
    nums = bpy.data.materials["Prop_SeatNumbers"]
    # bow centreline in (y, z): back leg, roof arc, a short flat run to the front edge
    A, B, zk = Y1 - 0.1 - Y0, 0.55, 1.62
    prof = [(Y0, 0.05)] + [(Y0 + A * (1 - math.cos(f)), zk + B * math.sin(f)) for f in np.linspace(0, math.pi / 2, 14)] + [(Y1, zk + B - 0.03)]
    front_leg = [(Y1, zk + B - 0.03 - 0.25), (Y1, 0.05)]
    box_prof = [(x_, y_) for x_, y_, _ in rrect(0.06, 0.04, 0.006, n=2)]
    xs = (-W / 2 + 0.03, -W / 4, 0.0, W / 4, W / 2 - 0.03)
    for x in xs:
        end = abs(x) > W / 3
        path = [np.array((x, y, z)) for y, z in prof]
        if end:
            corner = round_path([path[-2], path[-1], np.array((x, *front_leg[0])), np.array((x, *front_leg[1]))], 0.14, n=5)
            path = path[:-2] + corner
        p.r = 0.0
        p.sweep(path, box_prof, alu, cap=True, twist_ref=(1, 0, 0))
        p.r = 0.003
        for yy in ((Y0,) + ((Y1,) if end else ())):
            p.box((x - 0.07, yy - 0.07, 0.05), (x + 0.07, yy + 0.07, 0.058), alu)            # corner casting foot
            for bx in (-0.045, 0.045):
                for by in (-0.045, 0.045):
                    p.r = 0.0
                    p.cyl((x + bx, yy + by, 0.058), (x + bx, yy + by, 0.066), 0.008, M["zinc"], 6)
                    p.r = 0.003
    # base: a 100 x 50 mm channel round the back and ends, ground-anchor plates, the turf mat
    p.r = 0.004
    p.box((-W / 2 - 0.05, Y0 - 0.05, 0.0), (W / 2 + 0.05, Y0 + 0.05, 0.05), base)
    for s in (-1, 1):
        p.box((s * W / 2 - 0.05, Y0, 0.0), (s * W / 2 + 0.05, Y1 + 0.05, 0.05), base)
    for x, y in ((-W / 2, Y0), (W / 2, Y0), (-W / 2, Y1), (W / 2, Y1), (0.0, Y0)):
        p.box((x - 0.09, y - 0.09, 0.0), (x + 0.09, y + 0.09, 0.006), base)
    p.r = 0.006
    p.box((-W / 2 + 0.05, Y0 + 0.05, 0.0), (W / 2 - 0.05, Y1 - 0.02, 0.022), turf)
    # skins: polycarbonate outside the bows from the kick panel up over the roof; ends inside
    off = 0.034
    P2 = np.asarray(prof)
    sk = []
    for i in range(len(P2)):
        d = P2[min(i + 1, len(P2) - 1)] - P2[max(i - 1, 0)]
        d /= np.linalg.norm(d)
        sk.append(P2[i] + np.array([-d[1], d[0]]) * off)       # outward (behind / above)
    sk = np.asarray(sk)
    kick = 0.8
    sk[0] = (sk[0][0], kick)
    run = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(sk, axis=0), axis=1))])
    p.r = 0.0
    for i in range(len(sk) - 1):
        (y0, z0), (y1, z1) = sk[i], sk[i + 1]
        v0, v1 = run[i] / run[-1], run[i + 1] / run[-1]
        _face_out(p, [(W / 2, y0, z0), (-W / 2, y0, z0), (-W / 2, y1, z1), (W / 2, y1, z1)], poly, (0, -(z1 - z0), (y1 - y0)),
                  [(W / 2 / 2, v0), (-W / 2 / 2, v0), (-W / 2 / 2, v1), (W / 2 / 2, v1)])
    yk = Y0 - off
    for dy, out in ((0.0, -1), (0.004, 1)):
        _face_out(p, [(W / 2, yk + dy, 0.05), (-W / 2, yk + dy, 0.05), (-W / 2, yk + dy, kick), (W / 2, yk + dy, kick)], club, (0, out, 0))
    for s in (-1, 1):
        x = s * (W / 2 - 0.03)
        pts = [(Y0, kick)] + [(y, z) for y, z in prof[1:]] + [(Y1, zk + B - 0.3), (Y1, kick)]
        zmax = max(z for _, z in pts)
        _face_out(p, [(x, y, z) for y, z in pts], poly, (s, 0, 0), [((y - Y0) / 2, 0.3 * (z - kick) / (zmax - kick)) for y, z in pts])
        for dx, out in ((0.0, s), (-s * 0.004, -s)):
            _face_out(p, [(x + dx, Y0, 0.05), (x + dx, Y1, 0.05), (x + dx, Y1, kick), (x + dx, Y0, kick)], club, (out, 0, 0))
    # kick panel details: joints every 1.2 m, rivet rows, scuffs, the club vinyl
    for face_y, n_ in ((yk - 0.0012, (0, -1, 0)), (yk + 0.0052, (0, 1, 0))):
        for xj in np.arange(-W / 2 + 1.2, W / 2 - 0.1, 1.2):
            p.decal((xj, face_y, (0.05 + kick) / 2), n_, (0, 0, 1), 0.005, kick - 0.06, M["black"], lift=0.0)
        for zr in (0.08, kick - 0.03):
            p.decal((0, face_y, zr), n_, (0, 0, 1), W - 0.08, 0.012, rivet, lift=0.0004)
        p.decal((0, face_y + 0.0006 * n_[1], 0.05 + 0.175), n_, (0, 0, 1), W - 0.02, 0.35, grime, lift=0.0008)
    # fix rivet UVs: one head per 150 mm
    k_r = 0
    for i in range(len(p.F) - 1, -1, -1):
        if p.mats[p.mi[i]] is rivet and p.uv[i] is not None and k_r < 4:
            p.uv[i] = [(u * (W - 0.08) / 0.15, v) for u, v in p.uv[i]]
            k_r += 1
    sg = lettering.sign("HOME TEAM", 96, "#f4f4f0", None, pad=0.1, weight=1.05)
    vinyl = _tex_mat("Prop_ShelterVinylGloss", sg, 0.22, alpha=True, spec=0.6)
    if not bpy.data.materials.get("Prop_ShelterCrest"):
        _tex_mat("Prop_ShelterCrest", TP.crest(), 0.3, alpha=True)
    crest = bpy.data.materials["Prop_ShelterCrest"]
    for s in (-1, 1):
        p.decal((s * (W / 2 - 0.03), (Y0 + Y1) / 2 - 0.2, (0.05 + kick) / 2 + 0.02), (s, 0, 0), (0, 0, 1), 0.3, 0.36, crest, lift=0.0025)
    vh = 0.34
    p.decal((0, yk, 0.05 + (kick - 0.05) / 2 + 0.03), (0, -1, 0), (0, 0, 1), vh * sg.shape[1] / sg.shape[0], vh, vinyl, lift=0.0025)
    # trims: the front fascia gutter, a capping rail on the kick panels
    p.r = 0.003
    yf, zf = sk[-1]
    p.box((-W / 2 - 0.03, yf - 0.03, zf - 0.1), (W / 2 + 0.03, yf + 0.035, zf + 0.012), alu)
    p.box((-W / 2 - 0.03, yf + 0.02, zf - 0.012), (W / 2 + 0.03, yf + 0.045, zf + 0.012), alu)
    p.box((-W / 2, yk - 0.016, kick - 0.025), (W / 2, yk + 0.02, kick + 0.012), alu)
    for s in (-1, 1):
        p.box((s * W / 2 - 0.05, Y0 - 0.02, kick - 0.025), (s * W / 2 - 0.01, Y1 + 0.02, kick + 0.012), alu)
    # seat beam: twin 50 mm tubes on legs with foot plates and end plates; the seats on brackets
    p.r = 0.003
    yb = Y0 + 0.42
    for dy in (-0.09, 0.09):
        p.cyl((-W / 2 + 0.25, yb + dy, 0.36), (W / 2 - 0.25, yb + dy, 0.36), 0.025, alu, 14)
    for x in (-W / 2 + 0.24, W / 2 - 0.24):
        p.box((x - 0.006, yb - 0.13, 0.32), (x + 0.006, yb + 0.13, 0.4), alu)
    for x in (-2.55, -0.85, 0.85, 2.55):
        p.box((x - 0.03, yb - 0.11, 0.022), (x + 0.03, yb + 0.11, 0.33), base)
        p.box((x - 0.07, yb - 0.15, 0.022), (x + 0.07, yb + 0.15, 0.03), base)
    for k in range(9):
        x = -2.48 + k * 0.62
        p.r = 0.002
        p.box((x - 0.12, yb - 0.12, 0.385), (x + 0.12, yb + 0.12, 0.395), M["zinc"])
        p.r = 0.0
        surf, nrm = _seat_shell(p, x, yb + 0.22, seat_red)
        for hx in (-0.07, 0.07):                     # hinge knuckles under the pan
            p.cyl((x + hx - 0.018, yb + 0.03, 0.402), (x + hx + 0.018, yb + 0.03, 0.402), 0.008, M["zinc"], 8)
        p.cyl((x, yb + 0.06, 0.428), (x, yb + 0.06, 0.434), 0.007, M["black"], 8)       # drain hole
        # seat number on the back rest
        c_ = surf(0.0, 0.82)
        nn = nrm(0.0, 0.82)
        nn = nn if nn[1] > 0 else -nn
        upv = surf(0.0, 0.86) - surf(0.0, 0.78)
        upv /= np.linalg.norm(upv)
        p.decal(c_, nn, upv, 0.042, 0.056, nums, lift=0.0015)
        p.uv[-1] = [(k / 9, 0.08), ((k + 1) / 9, 0.08), ((k + 1) / 9, 0.92), (k / 9, 0.92)]
    # joint plates where the J-bows meet the fascia, 4 bolts each
    for x in xs[1:-1]:
        p.r = 0.002
        p.box((x - 0.05, yf - 0.034, zf - 0.09), (x + 0.05, yf - 0.03, zf - 0.01), alu)
        p.r = 0.0
        for bx in (-0.032, 0.032):
            for bz in (-0.075, -0.025):
                p.cyl((x + bx, yf - 0.034, zf + bz), (x + bx, yf - 0.041, zf + bz), 0.006, M["zinc"], 6)
    # everything stands on a 60 mm concrete pad
    PAD = 0.06
    p.V = [(x, y, z + PAD) for x, y, z in p.V]
    p.r = 0.008
    p.box((-W / 2 - 0.2, Y0 - 0.22, 0.0), (W / 2 + 0.2, Y1 + 0.28, PAD), M["concrete"])
    o = p.obj("team_shelter", col, 35)
    return _grime(o, lambda P: _mud(P, 0.36, (0.72, 0.68, 0.58), 0.06) * _mud(P, 0.06, (0.8, 0.78, 0.7), 0.0))


def _with_mats(fn):
    def build(col, M):
        return fn(col, dict(M, **_mats()))
    build.__name__ = fn.__name__
    build.__doc__ = fn.__doc__
    return build


BUILDERS = {k: _with_mats(v) for k, v in {
    "prop.mannequin": mannequin, "prop.cone_tall": cone_tall, "prop.cone_disc": cone_disc,
    "prop.agility_pole": agility_pole, "prop.mini_goal": mini_goal, "prop.rebounder": rebounder, "prop.ball": ball,
    "prop.ball_cart": ball_cart, "prop.agility_ladder": agility_ladder, "prop.team_shelter": team_shelter,
}.items()}
