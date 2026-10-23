"""Pitch: surface, Law 1 markings, grass shader (type, height, mowing, wear, wetness), sockets.

Field coordinates: centre mark at origin, length along X, width along Y (docs/ARCHITECTURE.md).
"""
import json
import math

import bpy
import numpy as np

from .. import catalog
from ..core import mesh as M
from ..core.nodes import NB, material, srgb
from ..core.scene import collection, empty, mesh_object, set_params
from ..core.sockets import socket

DEFAULTS = {
    "length": 105.0,
    "width": 68.0,
    "line_width": 0.12,
    "runoff_side": 6.0,
    "runoff_end": 7.5,
    "crown": 0.06,            # drainage camber along the spine, metres
    "grass": "perennial_ryegrass",
    "height_mm": None,        # None = grass type default
    "mow": {"pattern": "stripes", "width": 5.25, "angle": 0.0, "offset": 0.0},
    "wear": 0.25,             # 0 = opening day, 1 = end of a wet season
    "wetness": 0.0,
    "line_color": "#f5f5f0",
    "hand_painted": 0.0,      # 0 = machine-painted, 1 = wobbly volunteer lines
    "optional_marks": True,
    "seed": 1,
}

PENALTY_AREA = 16.5
GOAL_AREA = 5.5
GOAL_HALF = 7.32 / 2
CIRCLE_R = 9.15
PEN_MARK = 11.0
SPOT_R = 0.11
CORNER_R = 1.0


def resolve(params):
    p = catalog.merge(DEFAULTS, params)
    g = catalog.preset("grass", p["grass"])
    if p["height_mm"] is None:
        p["height_mm"] = g["height_mm"]["default"]
    p["_grass"] = g
    return p


def crown_fn(p):
    half = p["width"] / 2 + p["runoff_side"]
    c = p["crown"]
    return lambda x, y: c * (1 - np.clip(np.abs(y) / half, 0, 1) ** 2)


# -- markings ---------------------------------------------------------------------------------

def marking_paths(p):
    """Centre-lines of every marking. Returns (polylines[(pts, closed)], discs[(centre, r)])."""
    L, W, w = p["length"], p["width"], p["line_width"]
    hl, hw = L / 2, W / 2
    lines, discs = [], []
    # boundary (lines belong inside the field)
    lines.append((np.array([[hl - w / 2, hw - w / 2], [-hl + w / 2, hw - w / 2],
                            [-hl + w / 2, -hw + w / 2], [hl - w / 2, -hw + w / 2]]), True))
    lines.append((np.array([[0, -hw + w / 2], [0, hw - w / 2]]), False))
    lines.append((M.arc((0, 0), CIRCLE_R - w / 2, 0, 2 * np.pi, 0.08)[:-1, :2], True))
    discs.append(((0.0, 0.0), SPOT_R))
    for s in (1, -1):
        gx = s * (hl - w / 2)
        for depth, half in ((PENALTY_AREA, GOAL_HALF + PENALTY_AREA), (GOAL_AREA, GOAL_HALF + GOAL_AREA)):
            fx = s * (hl - depth + w / 2)
            yy = half - w / 2
            lines.append((np.array([[gx, yy], [fx, yy], [fx, -yy], [gx, -yy]]), False))
        mx = s * (hl - PEN_MARK)
        discs.append(((mx, 0.0), SPOT_R))
        r = CIRCLE_R - w / 2
        th = math.acos(min(1.0, (PENALTY_AREA - PEN_MARK) / r))
        phi0 = math.pi if s > 0 else 0.0
        lines.append((M.arc((mx, 0), r, phi0 - th, phi0 + th, 0.08)[:, :2], False))
        for t in (1, -1):
            a0 = math.atan2(0, -s)
            a1 = math.atan2(-t, 0)
            d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            lines.append((M.arc((s * hl, t * hw), CORNER_R - w / 2, a0, a0 + d, 0.05)[:, :2], False))
            if p["optional_marks"]:
                y = t * (hw - CORNER_R - CIRCLE_R)
                lines.append((np.array([[s * (hl + 0.05), y], [s * (hl + 0.55), y]]), False))
                x = s * (hl - CORNER_R - CIRCLE_R)
                lines.append((np.array([[x, t * (hw + 0.05)], [x, t * (hw + 0.55)]]), False))
    return lines, discs


def _wobble(pts, closed, amp, seed):
    """Perturb a polyline sideways with smooth noise (hand-painted lines)."""
    if amp <= 0:
        return pts
    q = np.concatenate([pts, pts[:1]]) if closed else pts
    q = M.resample(q, 0.4)
    d = np.gradient(q, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    off = amp * M.smooth_noise_1d(s, seed, octaves=3, base_freq=0.12)
    q = q + nrm * off[:, None]
    return q[:-1] if closed else q


def build_markings(p, col, parent, mat):
    zf = crown_fn(p)
    lines, discs = marking_paths(p)
    parts = []
    amp = 0.035 * p["hand_painted"]
    for i, (pts, closed) in enumerate(lines):
        pts = _wobble(pts, closed, amp, p["seed"] * 100 + i)
        # dense enough to follow the crowned surface instead of cutting under it
        pts = M.resample(np.concatenate([pts, pts[:1]]), 0.5)[:-1] if closed else M.resample(pts, 0.5)
        width = p["line_width"] * (1 + 0.15 * p["hand_painted"])
        V, F, UV = M.strip(pts, width, closed)
        parts.append((V, F, UV, None))
    for c, r in discs:
        V, F, UV = M.disc(c, r, 32)
        parts.append((V, F, UV, None))
    vs, fs, uvs, offs = [], [], [], 0
    for V, F, UV, _ in parts:
        vs.append(V)
        fs.append(F + offs)
        uvs.append(UV)
        offs += len(V)
    V = np.concatenate(vs)
    V[:, 2] = zf(V[:, 0], V[:, 1]) + 0.004
    obj = mesh_object("PitchLines", V, np.concatenate(fs), col, parent, materials=[mat], uvs=np.concatenate(uvs))
    return obj


# -- shading ----------------------------------------------------------------------------------

def _lean_field(nb, pos, mow):
    """Return (lean vector socket or None, parity scalar s) for the mowing pattern."""
    pattern = mow.get("pattern", "stripes")
    if pattern == "none":
        return None, 0.0
    w = float(mow.get("width", 5.25))
    ang = math.radians(float(mow.get("angle", 0.0)))
    if pattern in ("diagonal", "diamond"):
        ang += math.radians(45)
    off = float(mow.get("offset", 0.0))
    x, y, _ = nb.sep(pos)
    # mower lines are never perfectly straight
    wob = nb.mul(nb.sub(nb.noise(pos, 0.25, detail=1), 0.5), 0.08)
    ca, sa = math.cos(ang), math.sin(ang)
    xr = nb.add(nb.add(nb.mul(x, ca), nb.mul(y, sa)), wob)
    yr = nb.add(nb.sub(nb.mul(y, ca), nb.mul(x, sa)), wob)

    def parity(coord):
        idx = nb.math("FLOOR", nb.add(nb.div(coord, w), off))
        return nb.sub(1.0, nb.mul(nb.math("FLOORED_MODULO", idx, 2.0), 2.0))

    ex, ey = (ca, sa, 0.0), (-sa, ca, 0.0)
    if pattern in ("stripes", "diagonal"):
        s = parity(xr)
        return nb.vmath("SCALE", ey, scale=s), s
    if pattern == "stripes_long":
        s = parity(yr)
        return nb.vmath("SCALE", ex, scale=s), s
    if pattern in ("checkerboard", "diamond"):
        s1, s2 = parity(xr), parity(yr)
        v = nb.vmath("ADD", nb.vmath("SCALE", ey, scale=s1), nb.vmath("SCALE", ex, scale=s2))
        return nb.vmath("SCALE", v, scale=1 / math.sqrt(2)), nb.mul(nb.add(s1, nb.mul(s2, 0.5)), 1 / 1.5)
    if pattern == "circles":
        r = nb.vmath("LENGTH", nb.comb(x, y, 0.0))
        s = parity(r)
        t = nb.vmath("NORMALIZE", nb.comb(nb.mul(y, -1.0), x, 0.0))
        return nb.vmath("SCALE", t, scale=s), s
    raise ValueError(f"unknown mow pattern {pattern}")


def _gauss(nb, x, y, cx, cy, sx, sy):
    dx = nb.div(nb.sub(x, cx), sx)
    dy = nb.div(nb.sub(y, cy), sy)
    q = nb.mul(nb.add(nb.mul(dx, dx), nb.mul(dy, dy)), -0.5)
    return nb.math("EXPONENT", q)


def surface_group(p):
    """Shader node group: grass colour, roughness, soil mask and stripe factor at the shading point."""
    name = "PitchSurface"
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old)
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    for nm, kind in (("Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"),
                     ("Soil", "NodeSocketFloat"), ("Stripe", "NodeSocketFloat")):
        tree.interface.new_socket(nm, in_out="OUTPUT", socket_type=kind)
    nb = NB(tree)
    g = p["_grass"]
    natural = g["natural"]
    geo = nb.geometry()
    pos = geo["Position"]
    x, y, _ = nb.sep(pos)
    L, W = p["length"], p["width"]

    # base colour with variation at pitch, metre and clump scales
    base = srgb(g["color"])
    n_large = nb.noise(pos, 0.035, detail=3, rough=0.5)
    n_med = nb.noise(pos, 0.7, detail=3, rough=0.55)
    n_fine = nb.noise(pos, 45.0, detail=4, rough=0.6)
    var = nb.mul(nb.mul(nb.madd(nb.sub(n_large, 0.5), 0.45 if natural else 0.08, 1.0),
                        nb.madd(nb.sub(n_med, 0.5), 0.22 if natural else 0.06, 1.0)),
                 nb.madd(nb.sub(n_fine, 0.5), 0.35 if natural else 0.2, 1.0))
    col = nb.mix(1.0, base, nb.comb(var, var, var), blend="MULTIPLY")
    if natural:
        stress = nb.mul(nb.smooth(0.55, 0.8, nb.noise(pos, 0.08, detail=4)), 0.35)
        col = nb.mix(stress, col, srgb(g["color_dry"]))

    # mowing stripes
    k = g["stripe_strength"] * min(1.6, max(0.4, p["height_mm"] / 25.0))
    lean, s = _lean_field(nb, pos, p["mow"])
    if lean is None:
        stripe = 1.0
    elif natural:
        view = nb.vmath("SCALE", geo["Incoming"], scale=-1.0)
        d = nb.vmath("DOT_PRODUCT", nb.vmath("MULTIPLY", view, (1.0, 1.0, 0.0)), lean)
        stripe = nb.madd(d, k, 1.0)
    else:
        stripe = nb.madd(s, k, 1.0)
    col = nb.mix(1.0, col, nb.comb(stripe, stripe, stripe), blend="MULTIPLY")

    # wear: goalmouths, keeper's line, centre spot, assistant referees' lanes
    terms = []
    for sgn in (1, -1):
        terms.append(_gauss(nb, x, y, sgn * (L / 2 - 5.5), 0.0, 4.5, 6.0))
        terms.append(nb.mul(_gauss(nb, x, y, sgn * (L / 2 - 1.2), 0.0, 1.4, 3.6), 1.4))
        terms.append(nb.mul(_gauss(nb, x, y, sgn * (L / 2 - PEN_MARK), 0.0, 0.8, 0.8), 0.8))
        lane_y = sgn * (W / 2 + 1.0)
        half = nb.smooth(-2.0, 2.0, nb.mul(x, float(sgn)))
        lane = nb.mul(_gauss(nb, x, y, 0.0, lane_y, 1e4, 0.7), nb.smooth(0.45, 0.7, nb.noise(pos, 0.9, detail=3)))
        terms.append(nb.mul(nb.mul(lane, half), 0.4))
    terms.append(nb.mul(_gauss(nb, x, y, 0.0, 0.0, 2.2, 2.2), 0.7))
    wear_f = terms[0]
    for t in terms[1:]:
        wear_f = nb.add(wear_f, t)
    breakup = nb.noise(pos, 1.6, detail=6, rough=0.65)
    wear_amt = p["wear"] * (1.0 if natural else 0.25)
    wmask = nb.clamp01(nb.mul(nb.mul(wear_f, nb.madd(breakup, 1.4, -0.2)), wear_amt * 1.6))
    soil_col = nb.mix(nb.noise(pos, 6.0, detail=5), srgb("#4b3826"), srgb("#6b553a"))
    col = nb.mix(nb.mul(nb.smooth(0.12, 0.55, wmask), 0.85), col, srgb(g["color_dry"]))
    soil = nb.smooth(0.45, 0.9, wmask)
    col = nb.mix(soil, col, soil_col)

    if not natural:
        # black rubber crumb showing between fibres
        crumb = nb.smooth(0.62, 0.66, nb.noise(pos, 260.0, detail=1))
        col = nb.mix(nb.mul(crumb, 0.5), col, srgb("#1a1a1a"))

    wet = p["wetness"]
    col = nb.mix(1.0, col, (1 - 0.3 * wet,) * 3, blend="MULTIPLY")
    # wet turf stays fairly matte (droplets sparkle, blades don't turn into a mirror); mud goes shiny
    rough = nb.madd(nb.sub(n_fine, 0.5), 0.12, (0.74 if natural else 0.6) - 0.16 * wet)
    rough = nb.mixf(soil, rough, 0.85 - 0.6 * wet)

    out = nb.node("NodeGroupOutput")
    nb.connect(col, out.inputs["Color"])
    nb.connect(rough, out.inputs["Roughness"])
    nb.connect(soil, out.inputs["Soil"])
    nb.connect(stripe, out.inputs["Stripe"])
    return tree


def _group(nb, tree):
    n = nb.node("ShaderNodeGroup")
    n.node_tree = tree
    return n.outputs


def grass_material(p, group):
    m, nb = material("Pitch_Grass")
    s = _group(nb, group)
    pos = nb.geometry()["Position"]
    bump = nb.bump(nb.noise(pos, 180.0, detail=3), 0.35, 0.004)
    bsdf = nb.principled(Base_Color=s["Color"], Roughness=s["Roughness"], Specular_IOR_Level=0.18,
                         Sheen_Weight=0.12, Sheen_Roughness=0.5, Sheen_Tint=srgb("#b9d39a"), Normal=bump)
    nb.output(bsdf)
    return m


def paint_material(p, group):
    """Paint sprayed onto grass: the grass shader underneath, whitened by a coverage mask."""
    m, nb = material("TINT_Pitch_Paint")
    s = _group(nb, group)
    pos = nb.geometry()["Position"]
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    v = nb.sep(uv)[1]
    edge = nb.mul(nb.math("MINIMUM", v, nb.sub(1.0, v)), p["line_width"])
    hp = p["hand_painted"]
    ragged = nb.mul(nb.noise(pos, 30.0, detail=5), 0.02 + 0.03 * hp)
    cover = nb.smooth(0.0, 0.004, nb.sub(edge, ragged))
    patch = nb.smooth(0.25 + 0.2 * hp, 0.6, nb.noise(pos, 3.0 + 4 * hp, detail=4))
    cover = nb.mul(cover, nb.mixf(patch, 0.55 - 0.2 * hp, 0.95))
    paint = nb.mix(1.0, srgb(p["line_color"]), nb.comb(*(nb.madd(nb.sub(s["Stripe"], 1.0), 0.4, 1.0),) * 3),
                   blend="MULTIPLY")
    col = nb.mix(cover, s["Color"], paint)
    bump = nb.bump(nb.noise(pos, 180.0, detail=3), 0.3, 0.004)
    bsdf = nb.principled(Base_Color=col, Roughness=nb.mixf(cover, s["Roughness"], 0.6),
                         Specular_IOR_Level=0.2, Sheen_Weight=0.1, Normal=bump)
    nb.output(bsdf)
    m["tint_role"] = "line_paint"
    return m


# -- grass blades (close-up renders only; game engines use the shader) ------------------------

def _blade_shapes(p, col):
    g = p["_grass"]
    h = p["height_mm"] / 1000.0
    bw = g["blade_width_mm"] / 1000.0
    rng = np.random.default_rng(p["seed"])
    objs = []
    for i in range(8):
        hh = h * rng.uniform(0.75, 1.25)
        segs = 5
        t = np.linspace(0, 1, segs + 1)
        bend = rng.uniform(0.15, 0.5) * hh * (t ** 2)
        width = bw * (1 - 0.35 * t) if g["natural"] else np.full_like(t, bw)
        verts = []
        for ti, wi, bi in zip(t, width, bend):
            z = ti * hh
            verts += [(-wi / 2, bi, z), (wi / 2, bi, z)]
        faces = [(2 * k, 2 * k + 1, 2 * k + 3, 2 * k + 2) for k in range(segs)]
        o = mesh_object(f"Blade{i}", verts, faces, col, smooth=True)
        objs.append(o)
    return objs


def blade_material(p, group):
    m, nb = material("Pitch_Blade")
    s = _group(nb, group)
    paint = nb.attribute("paint", "INSTANCER")[2]  # outputs: Color, Vector, Fac, Alpha
    rnd = nb.attribute("rnd", "INSTANCER")[2]
    gen = nb.node("ShaderNodeTexCoord").outputs["Generated"]
    zt = nb.sep(gen)[2]
    shade = nb.madd(zt, 0.55, 0.6)                         # darker at the root
    shade = nb.madd(nb.sub(rnd, 0.5), 0.35, shade)
    col = nb.mix(1.0, s["Color"], nb.comb(shade, shade, shade), blend="MULTIPLY")
    paint_rgb = tuple(0.82 * c for c in srgb(p["line_color"])[:3])
    painted = nb.mix(0.22, paint_rgb, col)  # some green shows through
    col = nb.mix(nb.smooth(0.3, 0.7, paint), col, painted)
    natural = p["_grass"]["natural"]
    bsdf = nb.principled(Base_Color=col, Roughness=0.6 if natural else 0.45, Specular_IOR_Level=0.22,
                         Sheen_Weight=0.1)
    trans = nb.node("ShaderNodeBsdfTranslucent")
    nb.connect(col, trans.inputs["Color"])
    mixs = nb.node("ShaderNodeMixShader")
    mixs.inputs[0].default_value = 0.3 if natural else 0.1
    nb.connect(bsdf.outputs[0], mixs.inputs[1])
    nb.connect(trans.outputs[0], mixs.inputs[2])
    nb.output(mixs)
    return m


def add_blades(p, root, lines_obj, region, focus, density=20000.0, near=3.0, far=18.0):
    """Scatter instanced blades over region=(x0,x1,y0,y1); density falls off away from `focus`."""
    zf = crown_fn(p)
    col = collection("GrassBlades")
    shapes_col = collection("BladeShapes")
    shapes = _blade_shapes(p, shapes_col)
    group = bpy.data.node_groups["PitchSurface"]
    bm = blade_material(p, group)
    for o in shapes:
        o.data.materials.append(bm)
    bpy.context.view_layer.layer_collection.children["BladeShapes"].exclude = True

    x0, x1, y0, y1 = region
    nx, ny = max(1, int((x1 - x0) / 0.5)), max(1, int((y1 - y0) / 0.5))
    V, F, _ = M.grid(nx, ny, x1 - x0, y1 - y0)
    V[:, 0] += (x0 + x1) / 2
    V[:, 1] += (y0 + y1) / 2
    V[:, 2] = zf(V[:, 0], V[:, 1])
    patch = mesh_object("GrassPatch", V, F, col)
    patch["render_only"] = True

    tree = bpy.data.node_groups.new("GrassBlades", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nb = NB(tree)
    gin = nb.node("NodeGroupInput")
    gout = nb.node("NodeGroupOutput")
    posn = nb.node("GeometryNodeInputPosition").outputs[0]
    dist = nb.vmath("DISTANCE", posn, tuple(focus))
    fac = nb.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP")
    nb.connect(dist, fac.inputs["Value"])
    fac.inputs["From Min"].default_value = near
    fac.inputs["From Max"].default_value = far
    fac.inputs["To Min"].default_value = 1.0
    fac.inputs["To Max"].default_value = 0.015
    dp = nb.node("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM")
    nb.connect(gin.outputs[0], dp.inputs["Mesh"])
    nb.connect(nb.mul(fac.outputs["Result"], density), dp.inputs["Density"])
    dp.inputs["Seed"].default_value = p["seed"]

    lines_info = nb.node("GeometryNodeObjectInfo", transform_space="RELATIVE")
    lines_info.inputs["Object"].default_value = lines_obj
    prox = nb.node("GeometryNodeProximity", target_element="FACES")
    nb.connect(lines_info.outputs["Geometry"], prox.inputs["Geometry"])
    painted = nb.math("LESS_THAN", prox.outputs["Distance"], 0.0075)
    st1 = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st1.inputs["Name"].default_value = "paint"
    nb.connect(dp.outputs["Points"], st1.inputs["Geometry"])
    nb.connect(painted, st1.inputs["Value"])
    rv = nb.node("FunctionNodeRandomValue", data_type="FLOAT")
    st2 = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    st2.inputs["Name"].default_value = "rnd"
    nb.connect(st1.outputs[0], st2.inputs["Geometry"])
    nb.connect(rv.outputs[1], st2.inputs["Value"])

    ci = nb.node("GeometryNodeCollectionInfo", transform_space="RELATIVE")
    ci.inputs["Collection"].default_value = shapes_col
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    pick = nb.node("FunctionNodeRandomValue", data_type="INT")
    pick.inputs["Min"].default_value = 0
    pick.inputs["Max"].default_value = len(shapes) - 1
    pick.inputs["Seed"].default_value = 7
    tilt = 0.45 if p["_grass"]["natural"] else 0.2
    rot = nb.node("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
    rot.inputs["Min"].default_value = (-tilt, -tilt, 0.0)
    rot.inputs["Max"].default_value = (tilt, tilt, 2 * math.pi)
    rot.inputs["Seed"].default_value = 3
    scl = nb.node("FunctionNodeRandomValue", data_type="FLOAT")
    scl.inputs["Min"].default_value = 0.8
    scl.inputs["Max"].default_value = 1.2
    iop = nb.node("GeometryNodeInstanceOnPoints")
    nb.connect(st2.outputs[0], iop.inputs["Points"])
    nb.connect(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    nb.connect(pick.outputs[2], iop.inputs["Instance Index"])
    nb.connect(rot.outputs[0], iop.inputs["Rotation"])
    nb.connect(scl.outputs[1], iop.inputs["Scale"])
    nb.connect(iop.outputs[0], gout.inputs[0])
    mod = patch.modifiers.new("GrassBlades", "NODES")
    mod.node_group = tree
    return patch


# -- build ------------------------------------------------------------------------------------

def build(params=None, col=None):
    p = resolve(params or {})
    col = col or collection("Pitch")
    root = empty("Pitch", col, display="PLAIN_AXES", size=2.0)
    public = {k: v for k, v in p.items() if not k.startswith("_")}
    set_params(root, f"pitch.{p['grass']}", "pitch", "stadium_origin", public)
    root["gameplay"] = json.dumps(p["_grass"]["gameplay"])

    group = surface_group(p)
    gm = grass_material(p, group)
    pm = paint_material(p, group)

    L, W = p["length"], p["width"]
    sx, sy = L + 2 * p["runoff_end"], W + 2 * p["runoff_side"]
    V, F, UV = M.grid(int(round(sx)), int(round(sy)), sx, sy, crown_fn(p))
    surf = mesh_object("PitchSurface", V, F, col, root, materials=[gm], uvs=UV)
    surf["walkable"] = True
    lines = build_markings(p, col, root, pm)

    zf = crown_fn(p)
    w = p["line_width"]
    for s, ident in ((1, "east"), (-1, "west")):
        x = s * (L / 2 - w / 2)
        socket("goal", ident, (x, 0.0, float(zf(x, 0.0))), yaw_deg=90.0 * s, parent=root, col=col,
               line_width=w)
    for s, t, ident in ((1, 1, "NE"), (1, -1, "SE"), (-1, 1, "NW"), (-1, -1, "SW")):
        x, y = s * (L / 2 - w / 2), t * (W / 2 - w / 2)
        yaw = math.degrees(math.atan2(x, -y))  # local +Y towards the centre mark
        socket("corner_flag", ident, (x, y, float(zf(x, y))), yaw_deg=yaw, parent=root, col=col)
    # perimeter edges where stands, ad boards and dugouts will attach (stand system, phase 2)
    for ident, (cx, cy), yaw, length in (("north", (0, sy / 2), 180.0, sx), ("south", (0, -sy / 2), 0.0, sx),
                                          ("east", (sx / 2, 0), 90.0, sy), ("west", (-sx / 2, 0), -90.0, sy)):
        socket("stand_front", ident, (cx, cy, 0.0), yaw_deg=yaw, parent=root, col=col, edge_length=length)
    socket("center_mark", "centre", (0, 0, float(zf(0, 0))), parent=root, col=col)
    return root, p, lines
