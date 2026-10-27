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
    "markings": "full",       # full | outline (small-sided: touchlines, halfway, centre circle) | none
                              # | 5v5 | 7v7 | 9v9 (FA youth formats, see YOUTH)
    "seed": 1,
}

PENALTY_AREA = 16.5
GOAL_AREA = 5.5
GOAL_HALF = 7.32 / 2
CIRCLE_R = 9.15
PEN_MARK = 11.0
SPOT_R = 0.11
CORNER_R = 1.0

# FA youth formats (FA Rules: goalpost and pitch sizes; FA mini-soccer and 9v9 guidance), metres:
# penalty area (depth, half width), goal area (depth, half width), penalty mark, centre circle.
# 5v5 has a curved area round the goal instead; in 5v5 and 7v7 the halfway line is the retreat
# line, in 9v9 a dashed retreat line runs midway between the halfway line and each penalty area.
YOUTH = {
    "5v5": {"curved_area": 5.94, "circle": 2.75},
    "7v7": {"area": (9.14, 8.23), "mark": 7.32, "circle": 4.57},
    "9v9": {"area": (11.89, 14.63), "goal_area": (3.66, 6.10), "mark": 8.23, "circle": 6.4, "retreat": True},
}


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
    kind = p.get("markings", "full")
    if kind == "none":
        return lines, discs
    # boundary (lines belong inside the field)
    lines.append((np.array([[hl - w / 2, hw - w / 2], [-hl + w / 2, hw - w / 2],
                            [-hl + w / 2, -hw + w / 2], [hl - w / 2, -hw + w / 2]]), True))
    lines.append((np.array([[0, -hw + w / 2], [0, hw - w / 2]]), False))
    y = YOUTH.get(kind)
    r_c = y["circle"] if y else CIRCLE_R if kind == "full" else min(CIRCLE_R, W / 6)
    lines.append((M.arc((0, 0), r_c - w / 2, 0, 2 * np.pi, 0.08)[:-1, :2], True))
    discs.append(((0.0, 0.0), SPOT_R))
    if y:
        _youth_marks(y, hl, hw, w, lines, discs)
        return lines, discs
    if kind != "full":
        return lines, discs
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


def _youth_marks(y, hl, hw, w, lines, discs):
    """Penalty and goal areas, penalty marks, corner arcs and retreat lines of a youth format."""
    for s in (1, -1):
        gx = s * (hl - w / 2)
        if "curved_area" in y:
            a = math.pi if s > 0 else 0.0
            lines.append((M.arc((s * hl, 0), y["curved_area"] - w / 2, a - math.pi / 2, a + math.pi / 2, 0.08)[:, :2], False))
        for key in ("area", "goal_area"):
            if key in y:
                depth, half = y[key]
                fx = s * (hl - depth + w / 2)
                yy = half - w / 2
                lines.append((np.array([[gx, yy], [fx, yy], [fx, -yy], [gx, -yy]]), False))
        if "mark" in y:
            discs.append(((s * (hl - y["mark"]), 0.0), SPOT_R))
        for t in (1, -1):
            a0 = math.atan2(0, -s)
            a1 = math.atan2(-t, 0)
            d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            lines.append((M.arc((s * hl, t * hw), CORNER_R - w / 2, a0, a0 + d, 0.05)[:, :2], False))
        if y.get("retreat"):
            # dashed: a metre of paint, a metre gap, across the pitch
            x = s * (hl - y["area"][0]) / 2
            yy = -hw + w
            while yy + 1.0 <= hw - w + 1e-9:
                lines.append((np.array([[x, yy], [x, yy + 1.0]]), False))
                yy += 2.0


def _step(pts, base):
    """Resampling step for a marking: base, but at least 24 segments on short curves (the 1 m
    corner arcs would otherwise turn into a three-segment polyline)."""
    length = float(np.sum(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1)))
    return min(base, max(length / 24.0, 0.05))


def _wobble(pts, closed, amp, seed):
    """Perturb a polyline sideways with smooth noise (hand-painted lines)."""
    if amp <= 0:
        return pts
    q = np.concatenate([pts, pts[:1]]) if closed else pts
    q = M.resample(q, _step(q, 0.4))
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
        if closed:
            ring = np.concatenate([pts, pts[:1]])
            pts = M.resample(ring, _step(ring, 0.5))[:-1]
        else:
            pts = M.resample(pts, _step(pts, 0.5))
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


def _blade_noise(nb, pos, mow):
    """Blade-scale noise streaked along the mowing direction (laid-over blades)."""
    ang = math.radians(float(mow.get("angle", 0.0)))
    if mow.get("pattern") in ("diagonal", "diamond"):
        ang += math.radians(45)
    ca, sa = math.cos(ang), math.sin(ang)
    x, y, _ = nb.sep(pos)
    xr = nb.add(nb.mul(x, ca), nb.mul(y, sa))
    yr = nb.sub(nb.mul(y, ca), nb.mul(x, sa))
    return nb.noise(nb.comb(nb.mul(xr, 3.2), yr, 0.0), 40.0, detail=3, rough=0.6)


def _gauss(nb, x, y, cx, cy, sx, sy):
    dx = nb.div(nb.sub(x, cx), sx)
    dy = nb.div(nb.sub(y, cy), sy)
    q = nb.mul(nb.add(nb.mul(dx, dx), nb.mul(dy, dy)), -0.5)
    return nb.math("EXPONENT", q)


def surface_group(p):
    """Shader node group: grass colour, roughness, soil mask and stripe factor at the shading point."""
    # one group per surface: materials built earlier (other pitches, training-ground lawns) keep theirs
    tree = bpy.data.node_groups.new("PitchSurface", "ShaderNodeTree")
    p["_surface_group"] = tree.name
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
    var = nb.mul(nb.madd(nb.sub(n_large, 0.5), 0.16 if natural else 0.05, 1.0),
                 nb.madd(nb.sub(n_med, 0.5), 0.10 if natural else 0.04, 1.0))
    col = nb.mix(1.0, base, nb.comb(var, var, var), blend="MULTIPLY")
    if natural:
        stress = nb.mul(nb.smooth(0.62, 0.85, nb.noise(pos, 0.08, detail=4)), 0.14)
        col = nb.mix(stress, col, srgb(g["color_dry"]))
        # clumps lean yellow-green or blue-green
        hue = nb.noise(pos, 0.6, detail=2)
        col = nb.mix(nb.mul(nb.sub(hue, 0.5), 0.6), col, nb.mix(1.0, col, (1.0, 1.06, 1.25), blend="MULTIPLY"))
    # blade scale: streaks along the mowing direction, bright tips, a few dead blades, and dark
    # thatch between blades that shows looking down and hides at grazing angles
    blade = _blade_noise(nb, pos, p["mow"])
    col = nb.mix(1.0, col, nb.comb(*(nb.madd(blade, 0.42, 0.80),) * 3), blend="MULTIPLY")
    tips = nb.smooth(0.55, 0.85, nb.noise(pos, 140.0, detail=1))
    col = nb.mix(nb.mul(tips, 0.5), col, nb.mix(1.0, col, (1.12, 1.10, 0.80), blend="MULTIPLY"))
    if natural:
        dead = nb.smooth(0.86, 0.97, nb.noise(pos, 300.0, detail=1))
        col = nb.mix(nb.mul(dead, 0.35), col, srgb(g["color_dry"]))
    facing = nb.node("ShaderNodeLayerWeight").outputs["Facing"]
    gaps = nb.mul(nb.smooth(0.30, 0.0, blade), nb.madd(nb.sub(1.0, facing), 0.75, 0.25))
    col = nb.mix(nb.mul(gaps, 0.55 if natural else 0.3), col, srgb("#3a3420"))

    # mowing stripes
    k = g["stripe_strength"] * min(1.6, max(0.4, p["height_mm"] / 25.0))
    lean, s = _lean_field(nb, pos, p["mow"])
    if lean is None:
        stripe = 1.0
    elif natural:
        view = nb.vmath("SCALE", geo["Incoming"], scale=-1.0)
        d = nb.vmath("DOT_PRODUCT", nb.vmath("MULTIPLY", view, (1.0, 1.0, 0.0)), lean)
        if p["mow"].get("pattern") == "circles":
            # the lean swings round each ring; a little crushed-blade contrast keeps ring edges readable
            d = nb.add(nb.mul(d, 0.75), nb.mul(s, 0.3))
        stripe = nb.madd(d, k, 1.0)
    else:
        stripe = nb.madd(s, k, 1.0)
    col = nb.mix(1.0, col, nb.comb(stripe, stripe, stripe), blend="MULTIPLY")

    # wear where play happens (the viewer's shader.wgsl has the same field): an oval in each
    # goalmouth, the keeper's crease along the goal line, the penalty spots, the centre spot, a faint
    # channel down the middle, and the assistant referees' lanes (each on their own half)
    # (sized for a 105 m pitch, scaled down with smaller ones: youth formats have no assistant referees)
    ks = min(max(L / 105.0, 0.35), 1.0)
    lane_k = min(max((ks - 0.6) / 0.3, 0.0), 1.0)
    lane_k = lane_k * lane_k * (3 - 2 * lane_k)
    terms = []
    for sgn in (1, -1):
        terms.append(_gauss(nb, x, y, sgn * (L / 2 - 4.0 * ks), 0.0, 3.2 * ks, 5.0 * ks))
        terms.append(nb.mul(_gauss(nb, x, y, sgn * (L / 2 - 0.9), 0.0, 1.0, 3.4 * ks), 1.3))
        terms.append(nb.mul(_gauss(nb, x, y, sgn * (L / 2 - PEN_MARK * ks), 0.0, 0.9, 0.9), 0.9))
        if lane_k > 0:
            lane_y = sgn * (W / 2 + 1.0)
            half = nb.smooth(-2.0, 2.0, nb.mul(x, float(sgn)))
            lane = nb.mul(_gauss(nb, x, y, 0.0, lane_y, 1e4, 0.7), nb.smooth(0.45, 0.7, nb.noise(pos, 0.9, detail=3)))
            terms.append(nb.mul(nb.mul(lane, half), 0.45 * lane_k))
    terms.append(nb.mul(_gauss(nb, x, y, 0.0, 0.0, 2.2 * ks, 2.2 * ks), 0.7))
    terms.append(nb.mul(_gauss(nb, x, y, 0.0, 0.0, 25.0 * ks, 7.0 * ks), 0.25))
    wear_f = terms[0]
    for t in terms[1:]:
        wear_f = nb.add(wear_f, t)
    # solid cores, ragged edges: low-frequency breakup only where the wear thins out
    breakup = nb.add(nb.mul(nb.noise(pos, 0.5, detail=3), 0.6), nb.mul(nb.noise(pos, 2.0, detail=4), 0.4))
    wear_amt = p["wear"] * (1.0 if natural else 0.25)
    core = nb.mul(wear_f, wear_amt * 1.35)
    edge = nb.math("MINIMUM", nb.mul(core, 2.5), 1.0)
    wmask = nb.clamp01(nb.add(core, nb.mul(nb.mul(nb.sub(breakup, 0.5), 0.7), edge)))
    # bare soil is a muddy grey-brown, thin turf a washed-out straw
    soil_col = nb.mix(nb.noise(pos, 6.0, detail=5), srgb("#574a38"), srgb("#6f604a"))
    dry = srgb(g["color_dry"])
    lum = 0.3 * dry[0] + 0.55 * dry[1] + 0.15 * dry[2]
    thin = tuple(0.65 * c + 0.35 * lum for c in dry[:3]) + (1.0,)
    col = nb.mix(nb.mul(nb.smooth(0.12, 0.55, wmask), 0.8), col, thin)
    soil = nb.smooth(0.45, 0.9, wmask)
    col = nb.mix(soil, col, soil_col)

    if not natural:
        # black rubber crumb showing between fibres
        crumb = nb.smooth(0.62, 0.66, nb.noise(pos, 260.0, detail=1))
        col = nb.mix(nb.mul(crumb, 0.5), col, srgb("#1a1a1a"))

    wet = p["wetness"]
    col = nb.mix(1.0, col, (1 - 0.3 * wet,) * 3, blend="MULTIPLY")
    # wet turf stays fairly matte (droplets sparkle, blades don't turn into a mirror); mud goes shiny
    rough = nb.madd(nb.sub(n_fine, 0.5), 0.12, (0.86 if natural else 0.7) - 0.16 * wet)
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
    # blades shadow and mask each other, so a turf canopy never mirrors the sky the way a smooth
    # dielectric does at grazing angles: specular stays faint, a soft sheen carries the grazing look
    bump = nb.bump(nb.add(nb.mul(_blade_noise(nb, pos, p["mow"]), 0.7), nb.mul(nb.noise(pos, 180.0, detail=3), 0.3)),
                   0.55, 0.004)
    bsdf = nb.principled(Base_Color=s["Color"], Roughness=s["Roughness"], Specular_IOR_Level=0.04,
                         Sheen_Weight=0.35, Sheen_Roughness=0.3, Sheen_Tint=srgb("#a9c48a"), Normal=bump)
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
    bsdf = nb.principled(Base_Color=col, Roughness=nb.mixf(cover, s["Roughness"], 0.7),
                         Specular_IOR_Level=0.08, Sheen_Weight=0.25, Sheen_Roughness=0.35, Normal=bump)
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
    for i in range(12):
        hh = h * rng.uniform(0.7, 1.3)
        segs = 6
        t = np.linspace(0, 1, segs + 1)
        bend = rng.uniform(0.1, 0.6) * hh * (t ** 2)
        twist = rng.uniform(-0.6, 0.6) * t
        # mown blades end in a ragged, narrowing tip; synthetic fibres stay flat ribbons
        width = bw * np.clip(1 - 0.75 * t ** 1.5, 0.12, 1) if g["natural"] else np.full_like(t, bw)
        verts = []
        for ti, wi, bi, tw in zip(t, width, bend, twist):
            z = ti * hh
            dx, dy = math.cos(tw) * wi / 2, math.sin(tw) * wi / 2
            verts += [(-dx, bi - dy, z), (dx, bi + dy, z)]
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
    shade = nb.madd(zt, 0.75, 0.32)                        # deep shade at the root
    shade = nb.madd(nb.sub(rnd, 0.5), 0.35, shade)
    col = nb.mix(1.0, s["Color"], nb.comb(shade, shade, shade), blend="MULTIPLY")
    # every blade a slightly different green: younger ones yellower, older ones bluer
    hue = nb.attribute("hue", "INSTANCER")[2]
    col = nb.mix(hue, nb.mix(1.0, col, (1.10, 1.05, 0.72), blend="MULTIPLY"),
                 nb.mix(1.0, col, (0.88, 1.0, 1.15), blend="MULTIPLY"))
    if p["_grass"]["natural"]:
        dead = nb.smooth(0.955, 0.965, rnd)                 # the odd dead blade
        col = nb.mix(dead, col, tuple(0.75 * c for c in srgb(p["_grass"]["color_dry"])[:3]))
    paint_rgb = tuple(0.82 * c for c in srgb(p["line_color"])[:3])
    painted = nb.mix(0.22, paint_rgb, col)  # some green shows through
    col = nb.mix(nb.smooth(0.3, 0.7, paint), col, painted)
    natural = p["_grass"]["natural"]
    bsdf = nb.principled(Base_Color=col, Roughness=0.42 if natural else 0.5, Specular_IOR_Level=0.35)
    trans = nb.node("ShaderNodeBsdfTranslucent")
    # light through a blade comes out a warmer, brighter green
    nb.connect(nb.mix(1.0, col, (1.25, 1.35, 0.55), blend="MULTIPLY"), trans.inputs["Color"])
    mixs = nb.node("ShaderNodeMixShader")
    mixs.inputs[0].default_value = 0.4 if natural else 0.1
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
    group = bpy.data.node_groups[p["_surface_group"]]
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
    # thin out towards the patch edges so the blades blend into the shaded surface without a seam
    px, py, _ = nb.sep(posn)
    edge = nb.math("MINIMUM", nb.math("MINIMUM", nb.sub(px, x0), nb.sub(x1, px)),
                   nb.math("MINIMUM", nb.sub(py, y0), nb.sub(y1, py)))
    efade = nb.smooth(0.0, 2.5, edge)
    dp = nb.node("GeometryNodeDistributePointsOnFaces", distribute_method="RANDOM")
    nb.connect(gin.outputs[0], dp.inputs["Mesh"])
    nb.connect(nb.mul(nb.mul(fac.outputs["Result"], efade), density), dp.inputs["Density"])
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
    rh = nb.node("FunctionNodeRandomValue", data_type="FLOAT")
    rh.inputs["Seed"].default_value = 11
    sth = nb.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    sth.inputs["Name"].default_value = "hue"
    nb.connect(st2.outputs[0], sth.inputs["Geometry"])
    nb.connect(rh.outputs[1], sth.inputs["Value"])
    st2 = sth

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

def _rounded_rect_hit(dx, dy, hx, hy, r):
    """Distance from the centre to a rounded rectangle along the unit direction (dx, dy)."""
    lo, hi = 0.0, 4 * (hx + hy)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        x, y = abs(dx * mid), abs(dy * mid)
        qx, qy = max(x - (hx - r), 0), max(y - (hy - r), 0)
        inside = (x <= hx and y <= hy) and (qx * qx + qy * qy <= r * r)
        lo, hi = (mid, hi) if inside else (lo, mid)
    return lo


def clip_to_bowl(root, p, A, B, R):
    """Pull the run-off turf in to a curved bowl's stand fronts: a tight bowl's corner stands come in
    over the run-off corners, which would otherwise poke out behind them."""
    surf = next((c for c in root.children if c.name.startswith("PitchSurface")), None)
    if surf is None:
        return
    zf = crown_fn(p)
    for v in surf.data.vertices:
        x, y = v.co.x, v.co.y
        r = math.hypot(x, y)
        if r < 1e-9:
            continue
        f = _rounded_rect_hit(x / r, y / r, A + 0.3, B + 0.3, R)
        if r > f:
            fx, fy = x / r * f, y / r * f
            v.co = (fx, fy, float(zf(fx, fy)))


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
