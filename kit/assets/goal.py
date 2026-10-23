"""Goal: Law 1 frame, style-specific supports, and a cloth-simulated net with real cords.

Local frame: origin at the centre of the goal line, +Y towards the pitch, net towards -Y.
"""
import math

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
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
    "seed": 1,
}

W, H = 7.32, 2.44


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


def _supports(p, s, col, root, mat, sand):
    sup = p["support"]
    xN, yF, zT = s["xN"], s["yF"], s["zT"]
    FT, BT, BB = s["FT"], s["BT"], s["BB"]
    objs = []
    if sup in ("stanchions", "rear_frame"):
        r = 0.026 if sup == "stanchions" else 0.025
        bend = 0.32 if sup == "stanchions" else 0.08
        for side in (-1, 1):
            x = side * (xN + 0.035)
            path = [(x, FT[0] + 0.02, zT), (x, BT[0], zT), (x, BB[0], r)]
            objs.append(_tube(f"Stanchion_{side:+d}", path, r, col, root, mat, fillet=bend))
            objs.append(_tube(f"GroundBar_{side:+d}", [(x, yF + 0.03, r), (x, BB[0], r)], r * 0.9, col, root, mat))
        objs.append(_tube("BackBar", [(-(xN + 0.035), BB[0], r), (xN + 0.035, BB[0], r)], r * 0.9, col, root, mat))
        if sup == "rear_frame":
            for side in (-1, 1):
                objs.append(_sandbag(f"Sandbag_{side:+d}", (side * (xN + 0.035), BB[0] + 0.45, 0.0), col, root, sand,
                                     seed=p["seed"] + side))
    elif sup == "rear_poles":
        for side in (-1, 1):
            x, y = side * (xN + 0.12), BB[0] - 0.12
            objs.append(_tube(f"RearPole_{side:+d}", [(x, y, -0.4), (x, y, zT - 0.02)], 0.03, col, root, mat))
            objs.append(_tube(f"RearArm_{side:+d}", [(x, y, zT - 0.05), (side * xN, BT[0], zT - 0.01)], 0.012,
                              col, root, mat))
    return objs


def _sandbag(name, loc, col, root, mat, seed=0):
    """Squashed, lumpy sack draped over the ground bar."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, np.pi, 14), np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
    rx, ry, rz = 0.22, 0.32, 0.09
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
    """Plastic net clips on the back of the posts and crossbar."""
    pw, pd = p["post_size"]
    pts = []
    for z in np.arange(0.25, H, 0.3):
        for side in (-1, 1):
            pts.append((side * s["xN"], -pd / 2 - 0.006, z))
    for x in np.arange(-W / 2 + 0.2, W / 2, 0.3):
        pts.append((x, -pd / 2 - 0.006, s["zT"]))
    cube = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * [0.012, 0.007, 0.012]
    faces = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    V = np.concatenate([cube + q for q in pts])
    F = np.concatenate([faces + 8 * i for i in range(len(pts))])
    return mesh_object("NetClips", V, F, col, root, materials=[mat])


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
    if p["support"] in ("stanchions", "rear_frame"):
        for sd in (L, R):
            pins |= set(sd["idx"][:, 0].tolist())   # along the top bar
            pins |= set(sd["idx"][-1, :].tolist())  # down the rear stanchion
    else:  # rear poles: tension cords to the pole tops, rear corners laced down the poles
        for sd in (L, R):
            pins |= set(sd["idx"][:, 0].tolist())
            pins |= set(sd["idx"][-1, :].tolist())
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
    return _relax(co.reshape(-1, 3).astype(float), faces, pins)


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


def build_net(p, col, root, mats):
    s = _net_shape(p)
    panels = _panels(p, s)
    verts, faces = _weld(panels)
    pins = _pins(p, panels)
    co = _simulate(p, verts, faces, pins, col)
    V, F, mi, RV, RF = _cords(p, s, panels, co)
    net = mesh_object("Net", V, F, col, root, smooth=True, materials=mats[:2], mat_index=mi)
    rope = mesh_object("NetRope", RV, RF, col, root, smooth=True, materials=[mats[0]])
    return net, rope, s


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
    root = empty("Goal", col, display="ARROWS", size=1.0)
    set_params(root, f"goal.{p['style']}", "goal", "goal", p)
    frame_mat = (MAT.aluminium("TINT_frame") if p["frame_finish"] == "worn_aluminium"
                 else MAT.painted_metal("TINT_frame", p["frame_color"]))
    net_mats = [MAT.net_cord("TINT_net_a", p["net_colors"][0]),
                MAT.net_cord("TINT_net_b", p["net_colors"][1 % len(p["net_colors"])])]
    _frame(p, col, root, frame_mat)
    s = _net_shape(p)
    _supports(p, s, col, root, frame_mat, MAT.canvas("Sandbag"))
    _clips(p, s, col, root, MAT.plastic("NetClip", "#e9e9e4", 0.4))
    sleeve = MAT.rubber("GroundSleeve", "#1c1c1a")
    pw, pd = p["post_size"]
    for side in (-1, 1):
        V, F, _ = M.sweep([(side * s["xN"], 0, -0.02), (side * s["xN"], 0, 0.006)],
                          M.profile_ellipse(pw + 0.05, pd + 0.05, 24), (0, 1, 0))
        mesh_object(f"Sleeve_{side:+d}", V, F, col, root, materials=[sleeve])
    build_net(p, col, root, net_mats)
    root["inner_width"] = W
    root["inner_height"] = H
    return root
