"""Goal: Law 1 frame, style-specific supports, and a cloth-simulated net with real cords.

Local frame: origin at the centre of the goal line, +Y towards the pitch, net towards -Y.
"""
import math

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import srgb
from ..core.scene import collection, empty, link, mesh_object, set_params

DEFAULTS = {
    "style": "classic",
    "frame_color": "#f4f4f0",
    "net_colors": ["#f7f7f5", "#c8102e"],
    "net_pattern": "solid",      # solid | bands | checker
    "band_cells": 3,
    "mesh": "hex",               # hex | square
    "mesh_size": 0.12,
    "cord_mm": 4.0,
    "width": 7.32,               # inside the posts, metres (youth goals: catalog/goals.json)
    "height": 2.44,
    "seed": 1,
}

W, H = 7.32, 2.44               # the goal being built (build() sets them from its params)


def resolve(params):
    params = dict(params or {})
    style = params.get("style", DEFAULTS["style"])
    p = catalog.merge(DEFAULTS, catalog.preset("goals", style))
    return catalog.merge(p, params)


def _frame(p, col, root, mat):
    pw, pd = p["post_size"]
    xs = W / 2 + pw / 2
    zt = H + pw / 2
    path = M.densify(M.fillet([(-xs, 0, -0.3), (-xs, 0, zt), (xs, 0, zt), (xs, 0, -0.3)], pw * 0.6, 8), 0.3)
    V, F, UV = M.sweep(path, M.profile(p["post_profile"], pw, pd, 28), (0, 1, 0))
    return mesh_object("Frame", V, F, col, root, smooth=True, materials=[mat], uvs=UV)


def _net_shape(p):
    pw, pd = p["post_size"]
    yF = -pd / 2 - 0.012
    xN = W / 2 + pw / 2
    zT = H + pw / 2
    Dt, Db = p["net_top_depth"], p["net_bottom_depth"]
    FT, BT = np.array([yF, zT]), np.array([yF - Dt, zT])
    BB, FB = np.array([yF - Db, 0.0]), np.array([yF, 0.0])
    return dict(yF=yF, xN=xN, zT=zT, FT=FT, BT=BT, BB=BB, FB=FB)


def _tube(name, pts, radius, col, root, mat, fillet=0.0):
    pts = np.asarray(pts, float)
    if fillet:
        pts = M.fillet(pts, fillet, 10)
    pts = M.densify(pts, 0.3)
    hint = (1, 0, 0) if abs(pts[-1][0] - pts[0][0]) < 1e-6 else (0, 0, 1)
    V, F, UV = M.sweep(pts, M.profile_ellipse(2 * radius, 2 * radius, 16), hint)
    return mesh_object(name, V, F, col, root, smooth=True, materials=[mat], uvs=UV)


def _box_tube(name, pts, size, col, root, mat):
    """A square box-section tube (portable goals' ground frame)."""
    pts = M.densify(np.asarray(pts, float), 0.3)
    V, F, UV = M.sweep(pts, M.profile_rounded_rect(size, size, 0.006, 3), (0, 0, 1))
    return mesh_object(name, V, F, col, root, smooth=False, materials=[mat], uvs=UV)


def _supports(p, s, col, root, mat, sand):
    """The rear structure. classic: white bent stanchions (48 mm), ground bars and back bar. box: a
    full rectangular rear frame in 50 mm tube (thinner than the posts). portable: stanchions and a
    rear top bar on a 60 x 60 box-section ground frame."""
    from .seat import flat
    sup = p["support"]
    xN, yF, zT = s["xN"], s["yF"], s["zT"]
    FT, BT, BB = s["FT"], s["BT"], s["BB"]
    objs = []
    rmat = mat                                   # the whole frame in the one white paint
    r = {"stanchions": 0.024, "rear_frame": 0.025, "rear_poles": 0.025}[sup]
    bend = {"stanchions": 0.32, "rear_frame": 0.08, "rear_poles": 0.06}[sup]
    for side in (-1, 1):
        x = side * (xN + 0.035)
        path = [(x, FT[0] + 0.02, zT), (x, BT[0], zT), (x, BB[0], r)]
        objs.append(_tube(f"Stanchion_{side:+d}", path, r, col, root, rmat, fillet=bend))
        if sup == "rear_frame":
            objs.append(_box_tube(f"GroundBar_{side:+d}", [(x, yF + 0.03, 0.03), (x, BB[0] - 0.03, 0.03)], 0.06, col, root, mat))
        else:
            objs.append(_tube(f"GroundBar_{side:+d}", [(x, yF + 0.03, r), (x, BB[0], r)], r * 0.9, col, root, rmat))
    if sup == "rear_frame":
        objs.append(_box_tube("BackBar", [(-(xN + 0.035), BB[0] - 0.03, 0.03), (xN + 0.035, BB[0] - 0.03, 0.03)], 0.06,
                              col, root, mat))
    else:
        objs.append(_tube("BackBar", [(-(xN + 0.035), BB[0], r), (xN + 0.035, BB[0], r)], r * 0.9, col, root, rmat))
    if sup in ("rear_frame", "rear_poles"):
        # a rear top bar closes the box: the roof of the net is laced to it
        objs.append(_tube("RearTopBar", [(-(xN + 0.035), BT[0], zT), (xN + 0.035, BT[0], zT)], r * 0.9, col, root, mat))
    return objs


def _sandbag(name, loc, col, root, mat, seed=0):
    """Squashed, lumpy sack draped over the ground bar."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, np.pi, 14), np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
    rx, ry, rz = 0.17, 0.33, 0.1
    x = rx * np.sin(u) * np.cos(v)
    y = ry * np.sin(u) * np.sin(v)
    z = rz * np.cos(u)
    lump = 1 + 0.08 * np.sin(3 * v + rng.uniform(0, 6)) * np.sin(2 * u)
    z = np.where(z < 0, z * 0.25, z * lump) + 0.03
    V = np.stack([x * lump, y, z], -1).reshape(-1, 3) + np.array(loc)
    nu, nv = u.shape
    F = []
    for i in range(nu - 1):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            F.append((a, b, b + nv, a + nv))
    return mesh_object(name, V, F, col, root, smooth=True, materials=[mat])


def _clips(p, s, col, root, mat):
    """Net hooks: moulded C-hooks every 300 mm on the back of the posts and crossbar, and cord
    ties round the ground bars (and the rear top bar) every 400 mm."""
    from .props import PM
    pw, pd = p["post_size"]
    q = PM(0.0)
    prof = [(0.0028 * math.cos(a), 0.0028 * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    y0 = -pd / 2 + 0.004
    t = np.linspace(0, np.pi * 0.85, 6)
    for z in np.arange(0.25, H, 0.3):
        for side in (-1, 1):
            x = side * (W / 2 + pw / 2)
            q.sweep([(x + side * 0.016 * np.cos(a) * 0.4, y0 - 0.017 * np.sin(a), z + 0.016 * np.cos(a)) for a in t], prof, mat)
    for x in np.arange(-W / 2 + 0.2, W / 2, 0.3):
        q.sweep([(x, y0 - 0.017 * np.sin(a), s["zT"] + 0.016 * np.cos(a)) for a in t], prof, mat)
    sup = p["support"]
    xN, BB, BT, yF = s["xN"], s["BB"], s["BT"], s["yF"]
    rr = {"stanchions": 0.026, "rear_frame": 0.025, "rear_poles": 0.03}[sup] * 0.9
    tie = lambda a, b: q.cyl(a, b, rr + 0.0035, mat, sides=8)
    for x in np.arange(-xN + 0.2, xN, 0.4):
        tie((x - 0.006, BB[0], rr), (x + 0.006, BB[0], rr))
        if sup in ("rear_frame", "rear_poles"):
            tie((x - 0.006, BT[0], s["zT"]), (x + 0.006, BT[0], s["zT"]))
    for side in (-1, 1):
        for y in np.arange(yF - 0.25, BB[0], -0.4):
            tie((side * (xN + 0.035), y - 0.006, rr), (side * (xN + 0.035), y + 0.006, rr))
    o = q.obj("NetClips", col, bevel=False)
    o.parent = root
    return o


def _goal_hardware(p, s, col, root, mat):
    """Hardware on every goal: joint collars below the crossbar corners, ground sockets standing
    30 mm proud of the turf, and per style: welded rear-corner gussets and ground anchor plates
    (box), or the portable's running gear (200 mm wheels on rear-corner brackets with axle and
    hub, steel counterweight bars bolted to the side ground frames, cast corner connectors) and a
    maker's sticker on an upright."""
    from . import kit_detail as KD
    from .props import PM, tyre
    from .seat import _bolt, flat
    q = PM(0.003)
    pw, pd = p["post_size"]
    xN, BB, BT, zT, yF = s["xN"], s["BB"], s["BT"], s["zT"], s["yF"]
    galv = flat("Goal_Galv", "#8e9396", rough=0.4, metallic=0.5)
    sock = flat("GoalSocket", "#3b3c3a", rough=0.7)
    for side in (-1, 1):
        xs_ = side * (xN + 0.035)
        q.r = 0.006                                                                       # stanchion corner castings
        q.box((xs_ - 0.035, s["FT"][0] - 0.05, zT - 0.04), (xs_ + 0.035, s["FT"][0] + 0.04, zT + 0.04), mat)
        q.box((xs_ - 0.032, BB[0] - 0.05, 0.0), (xs_ + 0.032, BB[0] + 0.05, 0.065), mat)
        q.r = 0.003
    for side in (-1, 1):
        x = side * (W / 2 + pw / 2)
        q.cyl((x, 0, H - 0.12), (x, 0, H - 0.05), pw / 2 + 0.006, mat, sides=20)        # joint collars
        q.cyl((x - side * 0.12, 0, zT), (x - side * 0.05, 0, zT), pw / 2 + 0.006, mat, sides=20)
        q.cyl((x, 0, -0.01), (x, 0, 0.03), pw / 2 + 0.022, sock, sides=24)              # ground socket
    sup = p["support"]
    if sup in ("rear_poles", "stanchions"):
        peg = flat("Goal_Peg", "#2a2b2c", rough=0.6)
        for x in np.linspace(-xN + 0.6, xN - 0.6, 4):                                   # net pegs behind the back bar
            q.cyl((x, BB[0] - 0.22, -0.02), (x, BB[0] - 0.22, 0.03), 0.007, peg, sides=8)
            q.box((x - 0.006, BB[0] - 0.235, 0.025), (x + 0.006, BB[0] - 0.18, 0.035), peg)
    if sup == "rear_poles":
        for side in (-1, 1):
            x = side * (xN + 0.035)
            q.r = 0.0
            q.extrude_yz([(BB[0], zT - 0.16), (BB[0], zT - 0.02), (BB[0] + 0.16, zT - 0.02)], x - 0.003, x + 0.003, mat)
            q.r = 0.002
            q.box((x - 0.06, BB[0] - 0.06, 0.0), (x + 0.06, BB[0] + 0.06, 0.03), galv)       # anchor plate
            for dx in (-0.04, 0.04):
                _bolt(q, (x + dx, BB[0] + dx, 0.03), (0, 0, 1), 0.008, galv, h=0.006)
    if sup == "rear_frame":
        rubber = flat("Goal_Tyre", "#1a1a1a", rough=0.85)
        hub = flat("Goal_Hub", "#a4a8ab", rough=0.4, metallic=0.6)
        weight = flat("Goal_Weight", "#202122", rough=0.8)
        for side in (-1, 1):
            x = side * (xN + 0.035)
            yb = BB[0] - 0.03
            q.r = 0.003
            q.box((x - 0.035, yb - 0.11, 0.02), (x + 0.035, yb + 0.02, 0.14), mat)                  # wheel bracket
            tyre(q, (x + side * 0.075, yb - 0.08, 0.1), 0.1, 0.055, 0.06, rubber, hub, sides=20, axis="x")
            q.cyl((x, yb - 0.08, 0.1), (x + side * 0.11, yb - 0.08, 0.1), 0.012, hub, sides=8)
            q.r = 0.018                                                                          # moulded ballast
            q.box((x - 0.1, BB[0] + 0.3, 0.06), (x + 0.1, BB[0] + 0.9, 0.2), weight)
            q.r = 0.006
            for yy in (BB[0] + 0.48, BB[0] + 0.72):                                              # carry handle
                q.box((x - 0.012, yy - 0.012, 0.2), (x + 0.012, yy + 0.012, 0.245), weight)
            q.box((x - 0.012, BB[0] + 0.468, 0.235), (x + 0.012, BB[0] + 0.732, 0.258), weight)
            q.r = 0.004
            q.box((x - 0.045, yF - 0.02, zT - 0.06), (x + 0.045, yF + 0.07, zT + 0.055), mat)        # corner connectors
            q.box((x - 0.045, yF - 0.02, 0.0), (x + 0.045, yF + 0.09, 0.075), mat)
            q.box((x - 0.045, BT[0] - 0.055, zT - 0.05), (x + 0.045, BT[0] + 0.055, zT + 0.05), mat)
            q.box((x - 0.045, BB[0] - 0.08, 0.0), (x + 0.045, BB[0] + 0.03, 0.075), mat)
        sticker = KD.image_material("Goal_Sticker", albedo=KD.plate_image("Goal_Sticker_img", "MATCH GOAL",
                                                                          fg="#1d3f7a", bg="#efefea", px=(384, 72), border=4),
                                    rough=0.4, extension="EXTEND")
        x = -(W / 2 + pw / 2)
        q.decal((x, pd / 2 + 0.0005, 1.75), (0, 1, 0), (0, 0, 1), 0.075, 0.016, sticker, lift=0.0)
    o = q.obj("GoalHardware", col)
    o.parent = root
    return o


def _finish_frame(objs, p):
    """Occlusion and wear in the vertex colours (they survive the per-goal frame colour): grass
    staining and mud up the bottom 300 mm, ball scuffs at the posts, dull dust on top."""
    from . import kit_detail as KD
    rng = np.random.default_rng(p["seed"] + 11)
    ph = rng.uniform(0, 6.28, 4)

    def wear(P, N):
        z = P[:, 2]
        g = np.clip(1 - z / 0.32, 0, 1) ** 1.6 * (0.75 + 0.25 * np.sin(P[:, 0] * 7 + ph[0]) * np.sin(P[:, 1] * 9 + ph[1]))
        out = 1 - g[:, None] * (1 - np.array([0.7, 0.74, 0.55]))
        scuff = (np.sin(P[:, 0] * 31 + ph[2]) * np.sin(z * 23 + ph[3]) > 0.82) & (z > 0.2) & (z < 1.8)
        out[scuff] *= 0.9
        top = np.clip(N[:, 2], 0, 1) ** 2 * 0.06
        out *= (1 - top)[:, None]
        return out
    for o in objs:
        if o and o.type == "MESH" and len(o.data.vertices):
            KD.finish(o, distance=0.5, strength=0.6, floor=0.45)
            KD.tint_vertices(o, wear)


# -- net --------------------------------------------------------------------------------------

SIM_CELL = 0.2  # coarser than the mesh so the draped surface is smooth; cords are mapped onto it


def _panels(p, s):
    """Rest-pose quad grids for the main sheet (roof + back) and two sides."""
    cell = SIM_CELL
    xN, FT, BT, BB, FB = s["xN"], s["FT"], s["BT"], s["BB"], s["FB"]
    n_x = max(4, int(round(2 * xN / cell)))
    n_top = max(2, int(round(np.linalg.norm(BT - FT) / cell)))
    n_back = max(4, int(round(np.linalg.norm(BB - BT) / cell)))
    # main sheet: rows along FT->BT->BB, columns across x
    yz = np.concatenate([FT + (BT - FT) * t[:, None] for t in [np.linspace(0, 1, n_top + 1)]] +
                        [BT + (BB - BT) * t[:, None] for t in [np.linspace(0, 1, n_back + 1)[1:]]])
    xs = np.linspace(-xN, xN, n_x + 1)
    main = np.zeros((len(yz), n_x + 1, 3))
    main[..., 0] = xs[None, :]
    main[..., 1] = yz[:, None, 0]
    main[..., 2] = yz[:, None, 1]
    u = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(yz, axis=0), axis=1))])
    panels = {"main": dict(grid=main, u=u, x=xs, n_top=n_top)}
    a = np.linspace(0, 1, n_top + 1)[:, None]
    b = np.linspace(0, 1, n_back + 1)[None, :]
    yz_side = ((1 - b)[..., None] * ((1 - a)[..., None] * FT + a[..., None] * BT) +
               b[..., None] * ((1 - a)[..., None] * FB + a[..., None] * BB))
    for side, x in (("left", -xN), ("right", xN)):
        g = np.zeros((n_top + 1, n_back + 1, 3))
        g[..., 0] = x
        g[..., 1:] = yz_side
        panels[side] = dict(grid=g, quad=(FT, BT, BB, FB))
    return panels


def _weld(panels):
    """Merge panel grids into one vertex list; record each grid's vertex indices."""
    allv = np.concatenate([pn["grid"].reshape(-1, 3) for pn in panels.values()])
    keys = np.round(allv / 1e-5).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    verts = allv[first]
    off = 0
    faces = []
    for pn in panels.values():
        g = pn["grid"]
        n0, n1 = g.shape[:2]
        idx = inv[off:off + n0 * n1].reshape(n0, n1)
        off += n0 * n1
        pn["idx"] = idx
        a = idx[:-1, :-1]
        faces.append(np.stack([a, idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4))
    return verts, np.concatenate(faces)


def _pins(p, panels):
    main, L, R = panels["main"], panels["left"], panels["right"]
    pins = set(main["idx"][0].tolist())          # crossbar
    pins |= set(main["idx"][-1].tolist())        # back of the net on the ground
    for sd in (L, R):
        pins |= set(sd["idx"][0, :].tolist())    # post
        pins |= set(sd["idx"][:, -1].tolist())   # ground
    for sd in (L, R):
        pins |= set(sd["idx"][:, 0].tolist())   # along the top bar
        pins |= set(sd["idx"][-1, :].tolist())  # down the rear stanchion / upright
    if p["support"] in ("rear_frame", "rear_poles"):
        pins |= set(main["idx"][main["n_top"]].tolist())   # laced along the rear top bar
    return sorted(pins)


def _simulate(p, verts, faces, pins, col):
    obj = mesh_object("NetSim", verts, faces, col)
    vg = obj.vertex_groups.new(name="pin")
    vg.add(pins, 1.0, "REPLACE")
    ground = mesh_object("NetGround", [(-20, -20, 0.008), (20, -20, 0.008), (20, 20, 0.008), (-20, 20, 0.008)],
                         [(0, 1, 2, 3)], col)
    ground.modifiers.new("Collision", "COLLISION")
    ground.collision.thickness_outer = 0.01
    mod = obj.modifiers.new("Cloth", "CLOTH")
    st = mod.settings
    st.quality = 10
    st.mass = 0.01
    st.air_damping = 2.0
    st.tension_stiffness = 12.0
    st.compression_stiffness = 6.0
    st.shear_stiffness = 0.5
    st.bending_stiffness = 0.4
    st.shrink_min = -float(p["slack"])
    st.vertex_group_mass = "pin"
    mod.collision_settings.use_collision = True
    mod.collision_settings.distance_min = 0.006
    frames = 90
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, frames
    mod.point_cache.frame_start, mod.point_cache.frame_end = 1, frames
    for f in range(1, frames + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg).data
    co = np.zeros(len(ev.vertices) * 3, np.float32)
    ev.vertices.foreach_get("co", co)
    sc.frame_set(1)
    bpy.data.objects.remove(obj)
    bpy.data.objects.remove(ground)
    co = _relax(co.reshape(-1, 3).astype(float), faces, pins)
    # hand-hung nets are never perfectly even: a little irregularity in the free mesh
    rng = np.random.default_rng(p["seed"] + 7)
    free = np.ones(len(co), bool)
    free[pins] = False
    jit = rng.normal(0, 0.012, co.shape)
    jit[:, 2] = np.minimum(jit[:, 2], 0) * 0.5
    co[free] += jit[free]
    co[:, 2] = np.maximum(co[:, 2], 0.004)
    return _relax(co, faces, pins, iters=2, w=0.4)


def _relax(co, faces, pins, iters=6, w=0.5):
    """Laplacian smoothing of the free vertices: removes solver jitter, keeps the drape."""
    n = len(co)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 3]], faces[:, [3, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    free = np.ones(n, bool)
    free[pins] = False
    for _ in range(iters):
        acc = np.zeros_like(co)
        cnt = np.zeros(n)
        np.add.at(acc, e[:, 0], co[e[:, 1]])
        np.add.at(acc, e[:, 1], co[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        avg = acc / np.maximum(cnt, 1)[:, None]
        co[free] = co[free] * (1 - w) + avg[free] * w
    return co


def _bilinear(G, fi, fj):
    n0, n1 = G.shape[:2]
    i0 = np.clip(np.floor(fi).astype(int), 0, n0 - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, n1 - 2)
    t = np.clip(fi - i0, 0, 1)[:, None]
    s = np.clip(fj - j0, 0, 1)[:, None]
    return ((1 - t) * (1 - s) * G[i0, j0] + t * (1 - s) * G[i0 + 1, j0] +
            (1 - t) * s * G[i0, j0 + 1] + t * s * G[i0 + 1, j0 + 1])


def _inv_bilinear(q, quad):
    """(y,z) points -> (a,b) for P(a,b) = (1-b)[(1-a)FT + a BT] + b[(1-a)FB + a BB]."""
    FT, BT, BB, FB = quad
    a = np.full(len(q), 0.5)
    b = np.full(len(q), 0.5)
    for _ in range(12):
        P = ((1 - b)[:, None] * ((1 - a)[:, None] * FT + a[:, None] * BT) +
             b[:, None] * ((1 - a)[:, None] * FB + a[:, None] * BB))
        da = (1 - b)[:, None] * (BT - FT) + b[:, None] * (BB - FB)
        db = ((1 - a)[:, None] * FB + a[:, None] * BB) - ((1 - a)[:, None] * FT + a[:, None] * BT)
        r = P - q
        det = da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]
        a -= (r[:, 0] * db[:, 1] - r[:, 1] * db[:, 0]) / det
        b -= (da[:, 0] * r[:, 1] - da[:, 1] * r[:, 0]) / det
    return a, b


def _hex_segments(x0, x1, y0, y1, size):
    R = size / math.sqrt(3)
    dx, dy = size, 1.5 * R
    segs = {}
    for row in range(int((y1 - y0) / dy) + 3):
        for c in range(int((x1 - x0) / dx) + 3):
            cx = x0 - dx + c * dx + (dx / 2 if row % 2 else 0)
            cy = y0 - dy + row * dy
            corners = [(cx + R * math.cos(math.radians(30 + 60 * k)), cy + R * math.sin(math.radians(30 + 60 * k)))
                       for k in range(6)]
            for k in range(6):
                a, b = corners[k], corners[(k + 1) % 6]
                key = (round((a[0] + b[0]) * 500), round((a[1] + b[1]) * 500))
                segs[key] = (a, b)
    return np.array(list(segs.values()))


def _square_segments(x0, x1, y0, y1, size):
    segs = []
    for x in np.arange(x0, x1 + 1e-6, size):
        segs.append(((x, y0), (x, y1)))
    for y in np.arange(y0, y1 + 1e-6, size):
        segs.append(((x0, y), (x1, y)))
    return np.array(segs)


def _clip_subdivide(segs, inside, step=0.03):
    """Subdivide segments and keep the pieces whose ends are both inside."""
    out0, out1 = [], []
    for a, b in segs:
        n = max(1, int(np.linalg.norm(b - a) / step))
        t = np.linspace(0, 1, n + 1)[:, None]
        pts = a + (b - a) * t
        ok = inside(pts)
        keep = ok[:-1] & ok[1:]
        out0.append(pts[:-1][keep])
        out1.append(pts[1:][keep])
    return np.concatenate(out0), np.concatenate(out1)


def _band(values, width):
    return (np.floor(values / width).astype(int) % 2)


def _cords(p, s, panels, co):
    """Map the cord pattern drawn in each panel's rest space onto the simulated surface."""
    size = p["mesh_size"]
    pattern = _hex_segments if p["mesh"] == "hex" else _square_segments
    r = p["cord_mm"] / 2000.0
    bw = p["band_cells"] * size
    seg0, seg1, mat = [], [], []

    main = panels["main"]
    G = co[main["idx"]]
    u, xs = main["u"], main["x"]
    segs = pattern(xs[0], xs[-1], 0.0, u[-1], size)[:, :, ::-1]  # (u, x)
    inside = lambda q: (q[:, 0] >= 0) & (q[:, 0] <= u[-1]) & (np.abs(q[:, 1]) <= xs[-1])
    a, b = _clip_subdivide(segs, inside)
    for q, dst in ((a, seg0), (b, seg1)):
        fi = np.interp(q[:, 0], u, np.arange(len(u)))
        fj = (q[:, 1] - xs[0]) / (xs[1] - xs[0])
        dst.append(_bilinear(G, fi, fj))
    mid = (a + b) / 2
    mat.append(_pattern_index(p, mid[:, 0], mid[:, 1] + xs[-1], bw))

    for side in ("left", "right"):
        pn = panels[side]
        G = co[pn["idx"]]
        quad = pn["quad"]
        ys = [q[0] for q in quad]
        zs = [q[1] for q in quad]
        segs = pattern(min(ys), max(ys), 0.0, max(zs), size)
        poly = np.array(quad)

        def inside(q, poly=poly):
            ok = np.ones(len(q), bool)
            for i in range(4):
                e0, e1 = poly[i], poly[(i + 1) % 4]
                cr = (e1[0] - e0[0]) * (q[:, 1] - e0[1]) - (e1[1] - e0[1]) * (q[:, 0] - e0[0])
                ok &= cr <= 1e-9
            return ok
        if not inside(np.array([poly.mean(0)]))[0]:
            def inside(q, poly=poly):  # opposite winding
                ok = np.ones(len(q), bool)
                for i in range(4):
                    e0, e1 = poly[i], poly[(i + 1) % 4]
                    cr = (e1[0] - e0[0]) * (q[:, 1] - e0[1]) - (e1[1] - e0[1]) * (q[:, 0] - e0[0])
                    ok &= cr >= -1e-9
                return ok
        a, b = _clip_subdivide(segs, inside)
        for q, dst in ((a, seg0), (b, seg1)):
            aa, bb = _inv_bilinear(q, quad)
            n0, n1 = G.shape[:2]
            dst.append(_bilinear(G, aa * (n0 - 1), bb * (n1 - 1)))
        mid = (a + b) / 2
        mat.append(_pattern_index(p, quad[0][0] - mid[:, 0], quad[0][1] - mid[:, 1], bw))

    p0, p1 = np.concatenate(seg0), np.concatenate(seg1)
    V, F, keep = M.tubes(p0, p1, r, sides=4)
    mi = np.repeat(np.concatenate(mat)[keep], 4)

    # head ropes around every panel edge
    ropes0, ropes1 = [], []
    for name, pn in panels.items():
        G = co[pn["idx"]]
        edges = [G[0], G[-1], G[:, 0], G[:, -1]]
        for e in edges:
            ropes0.append(e[:-1])
            ropes1.append(e[1:])
    RV, RF, _ = M.tubes(np.concatenate(ropes0), np.concatenate(ropes1), r * 1.6, sides=6)
    return V, F, mi, RV, RF


def _pattern_index(p, along, across, bw):
    if p["net_pattern"] == "bands":
        return _band(along, bw)
    if p["net_pattern"] == "checker":
        return _band(along, bw) ^ _band(across, bw)
    return np.zeros(len(along), int)


def _drape(p, panels, co):
    """Gravity on the hung net beyond what the pinned solver gives: a catenary dip in the roof
    between the crossbar and the rear bar, the back panel bellying out and bagging at the bottom,
    the side panels sagging inwards between their ties."""
    main = panels["main"]
    idx = main["idx"]
    n0, n1 = idx.shape
    u = main["u"]
    nt = main["n_top"]
    xs = main["x"]
    ax = np.sin(np.pi * (xs - xs[0]) / (xs[-1] - xs[0])) ** 0.6
    roof = {"stanchions": 0.15, "rear_poles": 0.2, "rear_frame": 0.16}[p["support"]]
    done = set()
    for i in range(n0):
        for j in range(n1):
            v = idx[i, j]
            if v in done:
                continue
            done.add(v)
            if i <= nt and u[nt] > 0:
                t = u[i] / u[nt]
                co[v, 2] -= roof * 4 * t * (1 - t) * ax[j]
            else:
                t = (u[i] - u[nt]) / max(u[-1] - u[nt], 1e-6)
                co[v, 1] -= (0.15 * np.sin(np.pi * t) ** 1.3 + 0.06 * t ** 3) * ax[j]
                co[v, 2] -= 0.05 * np.sin(np.pi * t) * ax[j]
    for key in ("left", "right"):
        g = panels[key]["idx"]
        a0, b0 = g.shape
        for i in range(a0):
            for j in range(b0):
                v = g[i, j]
                if v in done:
                    continue
                done.add(v)
                k = np.sin(np.pi * i / (a0 - 1)) * np.sin(np.pi * j / (b0 - 1))
                co[v, 0] -= np.sign(co[v, 0]) * 0.08 * k
    co[:, 2] = np.maximum(co[:, 2], 0.004)


def build_net(p, col, root, mats):
    s = _net_shape(p)
    panels = _panels(p, s)
    verts, faces = _weld(panels)
    pins = _pins(p, panels)
    co = _simulate(p, verts, faces, pins, col)
    _drape(p, panels, co)
    V, F, mi, RV, RF = _cords(p, s, panels, co)
    net = mesh_object("Net", V, F, col, root, smooth=True, materials=mats[:2], mat_index=mi)
    rope = mesh_object("NetRope", RV, RF, col, root, smooth=True, materials=[mats[0]])
    for o in (net, rope):
        o["lod"] = 0
        o["has_lod1"] = True
    card = _net_card(p, panels, co, col, root)
    card.hide_render = True               # renders use the real cords; games use the card
    return net, rope, s


def _net_tag(p):
    """The cord pattern as a name: mesh, cell size in mm and cord in tenths of a mm (hex120c40)."""
    return f"{p['mesh']}{int(round(p['mesh_size'] * 1000))}c{int(round(p['cord_mm'] * 10))}"


def _pattern_image(p, px=128):
    """One repeat of the cord pattern with alpha: square cell (s x s) or hex cell (s x sqrt(3) s)."""
    size = p["mesh_size"]
    r = p["cord_mm"] / 2000.0 * 1.2        # a touch thicker so the card holds up at distance
    if p["mesh"] == "hex":
        W, H = size, math.sqrt(3) * size
        segs = _hex_segments(-W, 2 * W, -H, 2 * H, size)
    else:
        W, H = size, size
        segs = _square_segments(-W, 2 * W, -H, 2 * H, size)
    w, h = px, int(round(px * H / W))
    X, Y = np.meshgrid((np.arange(w) + 0.5) / w * W, (np.arange(h) + 0.5) / h * H)
    P = np.stack([X, Y], -1)
    d = np.full(X.shape, 1e9)
    for a, b in segs:
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (a + t[..., None] * ab), axis=-1))
    alpha = np.clip((r - d) / (0.5 * W / w) + 0.5, 0, 1)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., 3] = alpha
    img = MAT.image_from_array(f"NetPattern_{_net_tag(p)}", rgba)
    return img, W, H


def _net_card(p, panels, co, col, root):
    """LOD1: the simulated net surface as a card, textured with the cord pattern (alpha clip)."""
    img, pw, ph = _pattern_image(p)
    # one card material per cord pattern (the senior hex net keeps the plain name): goals with a
    # different net, such as the youth goals' square mesh, must not share the first goal's texture
    tag = _net_tag(p)
    name = "TINT_net_card" if tag == "hex120c40" else f"TINT_net_card_{tag}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        from ..core.nodes import material, srgb
        mat, nb = material(name)
        uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
        tex = nb.node("ShaderNodeTexImage", image=img, extension="REPEAT")
        nb.connect(uv, tex.inputs["Vector"])
        clip = nb.math("ROUND", tex.outputs["Alpha"])     # glTF exporter reads ROUND as alpha MASK
        c = np.minimum(np.array(srgb(p["net_colors"][0])[:3]), 0.74)    # polyester cord, not paper white
        nb.output(nb.principled(Base_Color=(*c, 1.0), Roughness=0.8, Alpha=clip))
    vs, fs, uvs, off = [], [], [], 0
    f = 3                                                     # finer than the solver grid
    pwid = p["post_size"][0]
    for key, pn in panels.items():
        G = co[pn["idx"]]
        n0, n1 = G.shape[:2]
        if key == "main":
            U = np.broadcast_to(pn["x"][None, :], (n0, n1))
            Vv = np.broadcast_to(pn["u"][:, None], (n0, n1))
        else:
            rest = pn["grid"][..., 1:]
            U, Vv = rest[..., 0], rest[..., 1]
        m0, m1 = f * (n0 - 1) + 1, f * (n1 - 1) + 1
        fi, fj = np.meshgrid(np.linspace(0, n0 - 1, m0), np.linspace(0, n1 - 1, m1), indexing="ij")
        Gr = _bilinear(G, fi.ravel(), fj.ravel()).reshape(m0, m1, 3)
        UV0 = np.stack([U, Vv], -1).astype(float)
        UVr = _bilinear(UV0, fi.ravel(), fj.ravel()).reshape(m0, m1, 2)
        if key == "main":
            # scallops: the net pulls in at every clip on the crossbar (every 300 mm)
            x = Gr[0, :, 0]
            dip = 0.035 * np.abs(np.sin(np.pi * (x - (-W / 2 + 0.2)) / 0.3))
            Gr[0, :, 2] -= dip
            Gr[1, :, 2] -= dip * 0.45
            # the bottom of the back lies on the grass for 160 mm
            last, uvl = Gr[-1], UVr[-1]
            ds = (0.1, 0.2, 0.3)
            ext = [last + np.array([0.0, -d, 0.0]) for d in ds]
            rng = np.random.default_rng(p["seed"] + 3)
            for k, e in enumerate(ext):
                e[:, 2] = 0.006 + 0.012 * (k == 0)
                e[:, 1] += 0.03 * np.sin(e[:, 0] * 2.1 + rng.uniform(0, 6))   # bunched, not ruled
            Gr = np.concatenate([Gr, np.stack(ext)])
            UVr = np.concatenate([UVr, np.stack([uvl + np.array([0.0, d]) for d in ds])])
        else:
            # scallops down the post (row 0 runs along the post: the clips are on its back)
            z = Gr[0, :, 2]
            dip = 0.02 * np.abs(np.sin(np.pi * (z - 0.25) / 0.3))
            Gr[0, :, 1] -= dip
            Gr[1, :, 1] -= dip * 0.45
            # the bottom of each side panel lies on the grass, turned outwards
            last, uvl = Gr[:, -1], UVr[:, -1]
            sgn = np.sign(last[:, 0:1])
            ds = (0.1, 0.2, 0.3)
            ext = [last + np.concatenate([sgn * d, np.zeros_like(sgn), np.zeros_like(sgn)], 1) for d in ds]
            for k, e in enumerate(ext):
                e[:, 2] = 0.006 + 0.012 * (k == 0)
            Gr = np.concatenate([Gr, np.stack(ext, 1)], 1)
            UVr = np.concatenate([UVr, np.stack([uvl + np.array([0.0, d]) for d in ds], 1)], 1)
        m0, m1 = Gr.shape[:2]
        vs.append(Gr.reshape(-1, 3))
        uvv = np.stack([UVr[..., 0] / pw, UVr[..., 1] / ph], -1).reshape(-1, 2)
        i, j = np.meshgrid(np.arange(m0 - 1), np.arange(m1 - 1), indexing="ij")
        a = i * m1 + j
        F = np.stack([a, a + m1, a + m1 + 1, a + 1], -1).reshape(-1, 4)
        fs.append(F + off)
        uvs.append(uvv[F.ravel()])
        off += m0 * m1
    o = mesh_object("Net_LOD1", np.concatenate(vs), np.concatenate(fs), col, root, smooth=True,
                    materials=[mat], uvs=np.concatenate(uvs))
    o["lod"] = 1
    # cord grime: greyer and dirtier towards the ground (vertex colours)
    Vt = np.concatenate(vs)
    g = np.clip(1 - Vt[:, 2] / 0.45, 0, 1) ** 1.5
    c = np.ones((len(Vt), 4), np.float32)
    c[:, :3] = (1 - g[:, None] * (1 - np.array([0.7, 0.69, 0.6])))
    attr = o.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", c.ravel())
    o.data.color_attributes.active_color = attr
    return o


# -- build ------------------------------------------------------------------------------------

def duplicate(template, col):
    """Linked copy of an already-built asset hierarchy (shares mesh data)."""
    def copy(o, parent):
        n = o.copy()
        link(n, col)
        n.parent = parent
        for c in o.children:
            copy(c, n)
        return n
    root = copy(template, None)
    root.matrix_world = template.matrix_basis.copy()
    return root


def build(params=None, col=None, template=None):
    col = col or collection("Goal")
    if template is not None:
        return duplicate(template, col)
    p = resolve(params)
    global W, H
    W, H = float(p["width"]), float(p["height"])
    root = empty("Goal", col, display="ARROWS", size=1.0)
    set_params(root, f"goal.{p['style']}", "goal", "goal", p)
    from .seat import flat
    c = np.minimum(np.array(srgb(p["frame_color"])[:3]), 0.62)       # powder coat white, albedo ~0.8
    frame_mat = flat("TINT_frame", "#" + "".join(f"{int(round(255 * (1.055 * v ** (1 / 2.4) - 0.055))):02x}" for v in c),
                     rough=0.4)
    net_mats = [MAT.net_cord("TINT_net_a", p["net_colors"][0]),
                MAT.net_cord("TINT_net_b", p["net_colors"][1 % len(p["net_colors"])])]
    fr = _frame(p, col, root, frame_mat)
    s = _net_shape(p)
    sups = _supports(p, s, col, root, frame_mat, None)
    _clips(p, s, col, root, flat("NetClip", "#e2e2dc", rough=0.5))
    sups.append(_goal_hardware(p, s, col, root, frame_mat))
    _finish_frame([fr] + sups, p)
    build_net(p, col, root, net_mats)
    root["inner_width"] = W
    root["inner_height"] = H
    return root
