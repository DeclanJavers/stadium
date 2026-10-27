"""The construction kit and derelict odds: welfare cabin, toilet, skip, tower crane, fence and
hoarding panels, pallets, pipes, survey pegs, the site board, shipping containers, oil drums,
tyres and a field gate. Real sizes; origin on the ground at the centre, +Y the front.
"""
import math

import numpy as np
from mathutils import Matrix, Vector

from . import site_materials, stand
from .props import PM, label, mesh_image, round_path, rrect, tyre, _paint, _plastic


# -- helpers ---------------------------------------------------------------------------------------

def _normal(P):
    """Newell normal of a polygon (unit)."""
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def _proj(q, n, roof_v0=0.0):
    """Metric UV of point q on a face with normal n: u along the face, read the right way round
    from outside, v = height; faces looking up or down take (x, y) shifted up by roof_v0."""
    ax = int(np.argmax(np.abs(n)))
    if ax == 2:
        return (q[1], q[0] + roof_v0)
    if ax == 0:
        return (q[1] * np.sign(n[0]), q[2])
    return (-q[0] * np.sign(n[1]), q[2])


def _auto_uv(p, start=0, roof_v0=0.0, only=None):
    """Metric UVs for every face (from start) that has none (only: restrict to these materials)."""
    for i in range(start, len(p.F)):
        if p.uv[i] is not None or (only is not None and p.mats[p.mi[i]] not in only):
            continue
        P = [np.array(p.V[k]) for k in p.F[i]]
        n = _normal(P)
        p.uv[i] = [_proj(q, n, roof_v0) for q in P]


def _uv01(p, start, end=None):
    """Each face (start..end) mapped 0-1 across itself (a cast fitting, a plate)."""
    for i in range(start, end if end is not None else len(p.F)):
        P = [np.array(p.V[k]) for k in p.F[i]]
        n = _normal(P)
        uv = np.array([_proj(q, n) for q in P])
        lo, hi = uv.min(0), uv.max(0)
        p.uv[i] = [tuple((t - lo) / np.maximum(hi - lo, 1e-9)) for t in uv]


def _overlay(p, start, end, mat, d=0.003):
    """Copy faces start..end a little off their surface in another material, same UVs (a wear or
    rust layer cut out with alpha over tinted paint)."""
    r, p.r = p.r, 0.0
    for i in range(start, end):
        P = [np.array(p.V[k]) for k in p.F[i]]
        n = _normal(P)
        p.face([q + n * d for q in P], mat, p.uv[i])
    p.r = r


def _corr_profile(length, pitch, depth, crest=0.25, shape=None):
    """One profile across the sheet: (s, offset) points. shape: [(fraction of a pitch, fraction of
    the depth)] for one pitch (default a trapezoid with crests `crest` of the pitch)."""
    n = max(1, int(round(length / pitch)))
    step = length / n
    if shape is None:
        f = (1 - 2 * crest) / 2
        shape = [(0.0, 0.0), (f / 2 + 1e-6, 1.0), (f / 2 + crest, 1.0), (f + crest, 0.0)]
    pts = []
    for k in range(n):
        pts += [(k * step + step * a, depth * b) for a, b in shape]
    pts.append((length, 0.0))
    return pts


def _corr_panel(p, O, U, N, length, height, mat, pitch, depth, off=0.0, s0=None, s1=None, h0=0.0, h1=None, uv01=None,
                crest=0.25, shape=None):
    """Profiled sheet: profile along U (ribs along V = N x U) from O, outward N, crests `depth`
    proud. s0..s1 / h0..h1 clip it (a stencil strip); uv01=(r0, r1, z0, z1) maps the clip 0-1 in
    the face's (right, up) frame."""
    O, U, N = (np.asarray(x, float) for x in (O, U, N))
    V = np.cross(N, U)
    prof = _corr_profile(length, pitch, depth, crest, shape)
    s0 = 0.0 if s0 is None else s0
    s1 = length if s1 is None else s1
    h1 = height if h1 is None else h1
    pts = []
    for (a, da), (b, db) in zip(prof[:-1], prof[1:]):
        lo, hi = max(a, s0), min(b, s1)
        if hi <= lo + 1e-7:
            continue
        lerp = lambda s: da + (db - da) * (s - a) / (b - a)
        if not pts:
            pts.append((lo, lerp(lo)))
        pts.append((hi, lerp(hi)))
    R = np.cross((0, 0, 1.0), N)
    if np.linalg.norm(R) < 1e-6:
        R = U
    for (a, da), (b, db) in zip(pts[:-1], pts[1:]):
        P = lambda s, d, h: O + U * s + N * (d + off) + V * h
        q = [P(a, da, h0), P(b, db, h0), P(b, db, h1), P(a, da, h1)]
        uv = None
        if uv01:
            r0, r1, z0, z1 = uv01
            uv = [((x @ R - r0) / (r1 - r0), (x[2] - z0) / (z1 - z0)) for x in q]
        p.face(q, mat, uv)


def _box01(p, lo, hi, mat):
    """A box whose every face maps 0-1."""
    i = len(p.F)
    p.box(lo, hi, mat)
    _uv01(p, i)


# -- shipping containers ------------------------------------------------------------------------------

def container(col, M, L=6.058, name="storage_container", tint="#2c5a8a"):
    """An ISO 668 shipping container (20 ft, or 40 ft with L = 12.192), doors at the +y end: cast
    corner fittings with their apertures, corner posts, top and bottom side rails (forklift pockets
    on the 20 ft), trapezoidal corrugated sides and end, a roof of shallow cross corrugations,
    two corrugated door leaves on hinges with four locking bars, cam keepers, handles and seals,
    the owner code, size-type code, weights and CSC plate stencilled on, rust bleeding from the
    rails and eating along the foot. Stands on its castings (the rails 15 mm off the ground)."""
    from . import site_textures as ST
    p = PM(0.0)
    W, H = 2.438, 2.591
    hl, hw = L / 2, W / 2
    big = L > 9
    body = ST.container_paint(tint)
    wear = ST.wear_soft("PS_ContainerWear40" if big else "PS_ContainerWear3", cell=ST.CONT_CELL, wall_h=2.62, seed=640 if big else 600,
                        chips=-0.3, foot=0.18, runs=0.8, bands=((0.17, 0.03), (2.5, 0.03)),
                        scrapes=7, top_bleed=0.55, roof=1.0, roof_v0=ST.CONT_ROOF_V0)
    cast = ST.container_casting()
    dark = M["steel"]
    rubber = M["rubber"]
    galv = ST.galvanised_img()
    cy, cx, cz = 0.178, 0.162, 0.118
    # corner castings
    for sy in (-1, 1):
        for sx in (-1, 1):
            for z0 in (0.0, H - cz):
                x0, x1 = sorted((sx * hw, sx * (hw - cx)))
                y0, y1 = sorted((sy * hl, sy * (hl - cy)))
                _box01(p, (x0, y0, z0), (x1, y1, z0 + cz), cast)
    i_body = len(p.F)
    # corner posts (door-end posts deeper)
    p.r = 0.006
    for sy in (-1, 1):
        dy = 0.2 if sy > 0 else 0.16
        for sx in (-1, 1):
            x0, x1 = sorted((sx * (hw - 0.004), sx * (hw - 0.15)))
            y0, y1 = sorted((sy * (hl - 0.004), sy * (hl - dy)))
            p.box((x0, y0, cz), (x1, y1, H - cz), body)
    # side rails: bottom (with forklift pockets on the 20 ft) and top
    p.r = 0.0
    pockets = [(-1.025 - 0.18, -1.025 + 0.18), (1.025 - 0.18, 1.025 + 0.18)] if not big else []
    zb0, zb1 = 0.015, 0.175
    for sx in (-1, 1):
        xo, xi = sorted((sx * (hw - 0.006), sx * (hw - 0.07)))
        cuts = [-hl + cy] + [v for a in pockets for v in a] + [hl - cy]
        for a, b in zip(cuts[0::2], cuts[1::2]):
            p.box((xo, a, zb0), (xi, b, zb1), body)
        for a, b in pockets:
            p.box((xo, a, zb0), (xi, b, 0.03), body)
            p.box((xo, a, 0.145), (xi, b, zb1), body)
            # reinforcing plate round the pocket and its dark tunnel
            q0, q1 = sorted((sx * (hw - 0.002), sx * (hw - 0.07)))
            for yy in (a - 0.03, b):
                p.box((q0, yy, zb0), (q1, yy + 0.03, zb1 + 0.004), body)
            p.box((-hw + 0.08, a, 0.03), (hw - 0.08, b, 0.032), dark)
            p.box((-hw + 0.08, a, 0.143), (hw - 0.08, b, 0.145), dark)
        t0, t1 = sorted((sx * (hw - 0.006), sx * (hw - 0.065)))
        p.box((t0, -hl + cy, H - 0.105), (t1, hl - cy, H - 0.006), body)
    # end rails: door sill and header (+y), front bottom and top rails (-y)
    p.box((-hw + cx, hl - 0.12, 0.0), (hw - cx, hl - 0.004, 0.2), body)
    p.box((-hw + cx, hl - 0.16, H - 0.22), (hw - cx, hl - 0.004, H - 0.006), body)
    p.box((-hw + cx, -hl + 0.004, zb0), (hw - cx, -hl + 0.1, zb1), body)
    p.box((-hw + cx, -hl + 0.004, H - 0.12), (hw - cx, -hl + 0.1, H - 0.006), body)
    i_walls = len(p.F)
    p.r = 0.0
    # corrugated sides (vertical ribs), front end, roof (cross ribs)
    for sx in (-1, 1):
        N = (sx, 0, 0)
        U = np.cross((0, 0, 1.0), N)
        y_start = -(hl - cy) * sx
        _corr_panel(p, (sx * (hw - 0.05), y_start, zb1), U, N, L - 2 * cy, H - 0.105 - zb1, body, 0.278, 0.042)
    _corr_panel(p, (-(hw - 0.15), -hl + 0.05, zb1), (1, 0, 0), (0, -1, 0), W - 0.3, H - 0.12 - zb1, body, 0.278, 0.042)
    _corr_panel(p, (hw - 0.06, -hl + cy, H - 0.03), (0, 1, 0), (0, 0, 1), L - 2 * cy, W - 0.12, body, 0.3, 0.018, crest=0.3)
    i_doors = len(p.F)
    # doors: two leaves, a flat frame round horizontal corrugations
    yd = hl - 0.035
    leaf = (W - 2 * 0.15) / 2 - 0.008
    for sx in (-1, 1):
        xa = sx * 0.004
        xb = sx * (0.004 + leaf)
        x0, x1 = sorted((xa, xb))
        for (a0, a1, b0, b1) in ((x0, x1, 0.2, 0.28), (x0, x1, H - 0.3, H - 0.22), (x0, x0 + 0.07, 0.28, H - 0.3), (x1 - 0.07, x1, 0.28, H - 0.3)):
            p.box((a0, yd, b0), (a1, yd + 0.03, b1), body)
        p.r = 0.0
        Odoor = np.array((x0 + 0.07, yd + 0.004, 0.28))
        _corr_panel(p, Odoor, (0, 0, 1), (0, 1, 0), H - 0.58, leaf - 0.14, body, 0.46, 0.022, crest=0.32)
    i_end = len(p.F)
    _auto_uv(p, i_body, roof_v0=ST.CONT_ROOF_V0 + hw)
    # wear over the walls, rails and posts; door seals; dark underside
    _overlay(p, i_body, i_end, wear, 0.003)
    p.r = 0.0
    p.box((-hw + 0.07, -hl + 0.1, 0.1), (hw - 0.07, hl - 0.12, 0.12), dark)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 0.0, sx * 0.012))
        p.box((x0, yd - 0.004, 0.2), (x1, yd + 0.026, H - 0.22), rubber)
        xe = sx * (0.004 + leaf + 0.004)
        x0, x1 = sorted((xe, xe + sx * 0.012))
        p.box((x0, yd - 0.004, 0.2), (x1, yd + 0.026, H - 0.22), rubber)
    # hinges: four a leaf, blades on the door, knuckles on the post
    for sx in (-1, 1):
        xe = sx * (hw - 0.15)
        for z in (0.45, 1.05, 1.65, 2.2):
            x0, x1 = sorted((xe, xe - sx * 0.16))
            p.box((x0, yd + 0.03, z - 0.035), (x1, yd + 0.045, z + 0.035), body)
            p.cyl((xe + sx * 0.02, yd + 0.04, z - 0.06), (xe + sx * 0.02, yd + 0.04, z + 0.06), 0.022, dark, 12)
    i_hw = len(p.F)
    # locking bars, guides, cam keepers and handles
    yb = yd + 0.07
    for x in (-0.86, -0.34, 0.34, 0.86):
        s = 1 if x > 0 else -1
        p.cyl((x, yb, 0.1), (x, yb, H - 0.1), 0.0135, galv, 16)
        for z in (0.1, H - 0.1):
            p.box((x - 0.035, yd + 0.03, z - 0.025), (x + 0.035, yb + 0.02, z + 0.025), dark)
            p.box((x - 0.02, yb - 0.015, z - 0.015), (x + 0.025, yb + 0.025, z + 0.015), galv)
        for z in (0.7, 1.6, 2.15):
            p.box((x - 0.03, yd + 0.03, z - 0.02), (x + 0.03, yb + 0.02, z + 0.02), galv)
        zh = 1.18 if abs(x) < 0.5 else 1.0
        p.cyl((x, yb - 0.02, zh - 0.05), (x, yb - 0.02, zh + 0.05), 0.026, galv, 12)
        hx = x - s * 0.36 if abs(x) > 0.5 else x + s * 0.36
        hx = max(min(hx, hw - 0.2), -hw + 0.2)
        p.r = 0.0
        p.tube([(x, yb + 0.01, zh), (x + (hx - x) * 0.15, yb + 0.035, zh), (hx, yb + 0.035, zh)], 0.011, galv, 8, bend=0.03)
        p.box((hx - 0.03, yd + 0.03, zh - 0.05), (hx + 0.03, yb + 0.045, zh + 0.05), dark)
    _auto_uv(p, i_hw)
    # stencils: owner code and size-type top right of the right door and of each side, weights on
    # the left door, the lessor's name along the sides, the CSC plate
    code = "KSTU 482913 5"
    st = "42G1" if big else "22G1"
    lbl = ST.stencil(f"PS_Label_ContainerCode_{st}", [(code, 0.36, 0.08), (st, 0.36, 0.56)], (512, 128))
    p.decal((-0.55, yd + 0.031, H - 0.48), (0, 1, 0), (0, 0, 1), 0.8, 0.2, lbl)
    wts = ST.stencil("PS_Label_ContainerWeights", [("MAX GROSS   30,480 KG", 0.1, 0.06), ("               67,200 LB", 0.1, 0.2),
                                                ("TARE          2,250 KG", 0.1, 0.38), ("               4,960 LB", 0.1, 0.52),
                                                ("PAYLOAD    28,230 KG", 0.1, 0.7), ("CU.CAP.     33.2 CU.M", 0.1, 0.84)], (256, 256), kind="narrow")
    p.decal((0.62, yd + 0.031, H - 0.75), (0, 1, 0), (0, 0, 1), 0.5, 0.5, wts)
    p.decal((0.5, yd + 0.031, 1.4), (0, 1, 0), (0, 0, 1), 0.2, 0.125, ST.csc_plate())
    logo = ST.stencil("PS_Label_ContainerLogo", [("KESTREL", 0.7, 0.15, "black")], (1024, 160), wear=0.42, seed=4)
    for sx in (-1, 1):
        N = np.array((sx, 0, 0.0))
        U = np.cross((0, 0, 1.0), N)
        y_start = -(hl - cy) * sx
        O = np.array((sx * (hw - 0.05), y_start, zb1))
        # owner code near the door end (top right seen from the side the doors are on the right)
        ln = L - 2 * cy
        s_door = ln if sx > 0 else 0.0
        a0, a1 = (ln - 1.05, ln - 0.15) if sx > 0 else (0.15, 1.05)
        z0, z1 = H - 0.105 - zb1 - 0.33, H - 0.105 - zb1 - 0.1
        r0 = (O + U * a0) @ np.cross((0, 0, 1.0), N)
        r1 = (O + U * a1) @ np.cross((0, 0, 1.0), N)
        _corr_panel(p, O, U, N, ln, H - 0.105 - zb1, lbl, 0.278, 0.042, off=0.004, s0=a0, s1=a1, h0=z0, h1=z1,
                    uv01=(min(r0, r1), max(r0, r1), zb1 + z0, zb1 + z1))
        # the lessor's name, big, along the middle
        mid = ln / 2
        lw = ln * 0.55 if big else min(3.4, ln * 0.6)
        a0, a1 = mid - lw / 2, mid + lw / 2
        z0, z1 = (0.75, 0.75 + lw * 160 / 1024) if big else (1.05, 1.05 + lw * 160 / 1024)
        r0 = (O + U * a0) @ np.cross((0, 0, 1.0), N)
        r1 = (O + U * a1) @ np.cross((0, 0, 1.0), N)
        _corr_panel(p, O, U, N, ln, H - 0.105 - zb1, logo, 0.278, 0.042, off=0.004, s0=a0, s1=a1, h0=z0, h1=z1,
                    uv01=(min(r0, r1), max(r0, r1), zb1 + z0, zb1 + z1))
    # ventilation grilles under the top rail near each end of both sides
    p.r = 0.002
    for sx in (-1, 1):
        for yv in (-hl + 1.7, hl - 1.7):
            x0, x1 = sorted((sx * (hw - 0.01), sx * (hw + 0.004)))
            p.box((x0, yv - 0.09, 2.3), (x1, yv + 0.09, 2.38), body)
            for k in range(4):
                p.box((x0 - 0.001 * sx, yv - 0.07, 2.31 + k * 0.017), (x1 + 0.001 * sx, yv + 0.07, 2.318 + k * 0.017), dark)
    return p.obj(name, col, 25)


def storage_container(col, M):
    return container(col, M)


def container_40(col, M):
    return container(col, M, L=12.192, name="container_40", tint="#7a2a22")


def _clad(p, O, U, N, length, height, mat, pitch, depth, holes=(), crest=0.25):
    """Profiled cladding with rectangular holes [(s0, s1, h0, h1)] in its own (along, up) frame."""
    cuts = sorted({0.0, length, *[min(max(c, 0.0), length) for h in holes for c in (h[0], h[1])]})
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-4:
            continue
        mid = (a + b) / 2
        z = 0.0
        for h0, h1 in sorted((h[2], h[3]) for h in holes if h[0] <= mid <= h[1]) + [(height, height)]:
            if h0 > z + 1e-4:
                _corr_panel(p, O, U, N, length, height, mat, pitch, depth, s0=a, s1=b, h0=z, h1=h0, crest=crest)
            z = max(z, h1)


def _merge(p, q, xf):
    """Copy PM q into p through the 4x4 transform xf (radii, materials and UVs kept)."""
    r = p.r
    for f, mi, uv, rad in zip(q.F, q.mi, q.uv, q.rad):
        p.r = rad
        p.face([xf @ Vector(q.V[i]) for i in f], q.mats[mi], uv)
    p.r = r


def _wall_xf(c, N):
    """Local wall frame (outward +y, x along, z up) to world, the wall plane through c."""
    yaw = math.atan2(-N[0], N[1])
    return Matrix.Translation(Vector(c)) @ Matrix.Rotation(yaw, 4, "Z")


def _window(M, frame, glass, w=1.2, z0=1.0, z1=2.1, shutter=0.0, box=True, slat=None):
    """A cabin window in its wall frame (opening centred on x = 0 in the plane y = 0): steel frame
    with a centre mullion, dark glass, sill, roller-shutter box and guides, the curtain `shutter`
    of the way down."""
    q = PM(0.0)
    h = z1 - z0
    for a, b in (((-w / 2 - 0.05, -0.07, z0 - 0.05), (w / 2 + 0.05, 0.015, z0)), ((-w / 2 - 0.05, -0.07, z1), (w / 2 + 0.05, 0.015, z1 + 0.05)),
                 ((-w / 2 - 0.05, -0.07, z0), (-w / 2, 0.015, z1)), ((w / 2, -0.07, z0), (w / 2 + 0.05, 0.015, z1))):
        q.box(a, b, frame)
    q.box((-0.025, -0.06, z0), (0.025, -0.015, z1), frame)
    q.face([(w / 2, -0.05, z0), (-w / 2, -0.05, z0), (-w / 2, -0.05, z1), (w / 2, -0.05, z1)], glass)
    from . import site_textures as ST
    q.face([(w / 2, -0.09, z1 - h * 0.55), (-w / 2, -0.09, z1 - h * 0.55), (-w / 2, -0.09, z1), (w / 2, -0.09, z1)], ST.blind(),
           [(0, 0), (w / 0.5, 0), (w / 0.5, 1), (0, 1)])
    q.face([(w / 2 + 0.2, -0.6, z0 - 0.3), (-w / 2 - 0.2, -0.6, z0 - 0.3), (-w / 2 - 0.2, -0.6, z1 + 0.2), (w / 2 + 0.2, -0.6, z1 + 0.2)], ST.room())
    for sx_ in (-1, 1):
        q.face([(sx_ * w / 2, -0.6, z0), (sx_ * w / 2, -0.06, z0), (sx_ * w / 2, -0.06, z1), (sx_ * w / 2, -0.6, z1)][::sx_], ST.room())
    q.face([(-w / 2, -0.6, z0), (w / 2, -0.6, z0), (w / 2, -0.06, z0), (-w / 2, -0.06, z0)], ST.room())
    q.face([(-w / 2, -0.06, z1), (w / 2, -0.06, z1), (w / 2, -0.6, z1), (-w / 2, -0.6, z1)], ST.room())
    q.box((-w / 2 - 0.08, -0.02, z0 - 0.09), (w / 2 + 0.08, 0.055, z0 - 0.05), frame)
    if box:
        q.r = 0.008
        q.box((-w / 2 - 0.08, -0.01, z1 + 0.05), (w / 2 + 0.08, 0.17, z1 + 0.27), frame)
        q.r = 0.0
        for s in (-1, 1):
            a, b = sorted((s * (w / 2 + 0.01), s * (w / 2 + 0.06)))
            q.box((a, 0.015, z0 - 0.05), (b, 0.08, z1 + 0.05), frame)
        if shutter > 0:
            hs = shutter * h + 0.05
            q.r = 0.0
            _corr_panel(q, (-w / 2 - 0.01, 0.045, z1 + 0.05 - hs), (0, 0, 1), (0, 1, 0), hs, w + 0.02, slat, 0.055, 0.009, crest=0.3)
            q.box((-w / 2 - 0.01, 0.035, z1 + 0.05 - hs - 0.04), (w / 2 + 0.01, 0.065, z1 + 0.05 - hs), frame)
    return q


def site_cabin(col, M):
    """An anti-vandal steel welfare/office cabin, 32 x 10 ft (9.75 x 3.05 m, 2.62 m high so two
    stack): cast corner fittings with lifting apertures, corner posts, a deep base frame with
    forklift pockets, a top frame standing up round a profiled roof, profiled cladding with
    windows cut in (frames, mullions, roller shutters at different heights), a steel door with
    trim, kick plate, lever and canopy under a bulkhead light, galvanised steps and handrails,
    a downpipe, power inlet and vents, hire stencils and site signs; grime and rust from the
    fixings. Front +y."""
    from . import site_textures as ST
    p = PM(0.005)
    L, D, Ht = 9.75, 3.05, 2.62
    hl, hd = L / 2, D / 2
    body = ST.paint_detail("TINT_cabin", "#d7d7cd", seed=70, foot=0.6, runs=0.6, patches=4)
    wear = ST.wear_soft("PS_CabinWear3", seed=81, chips=-0.4, bands=((0.32, 0.025),), foot=0.12, runs=0.9, scrapes=4, top_bleed=0.7,
                        fixings=((0.33, 0.4), (2.44, 0.4)), algae=0.8, mud=0.3)
    frame = ST.flat_steel("PS_CabinFrame", "#5a5d60", seed=90, dirt=0.25)
    roof = ST.cabin_roof()
    cast = ST.container_casting()
    glass = M["clear"]
    zf, zt = 0.3, 2.47                    # top of the base frame, bottom of the top frame
    # corner castings and posts
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, x1 = sorted((sx * hl, sx * (hl - 0.16)))
            y0, y1 = sorted((sy * hd, sy * (hd - 0.16)))
            for z0 in (0.0, Ht - 0.12):
                _box01(p, (x0, y0, z0), (x1, y1, z0 + 0.12), cast)
            x0, x1 = sorted((sx * (hl - 0.005), sx * (hl - 0.15)))
            y0, y1 = sorted((sy * (hd - 0.005), sy * (hd - 0.15)))
            p.box((x0, y0, 0.12), (x1, y1, Ht - 0.12), frame)
    # base frame (forklift pockets on the long sides) and top frame
    pk = [(-1.2, -0.85), (0.85, 1.2)]
    p.r = 0.0
    for sy in (-1, 1):
        y0, y1 = sorted((sy * hd, sy * (hd - 0.1)))
        cuts = [-hl + 0.16, *[v for a in pk for v in a], hl - 0.16]
        for a, b in zip(cuts[0::2], cuts[1::2]):
            p.box((a, y0, 0.12), (b, y1, zf), frame)
        for a, b in pk:
            p.box((a, y0, 0.12), (b, y1, 0.15), frame)
            yi = sy * (hd - 0.1)
            p.face([(a, yi, 0.15), (b, yi, 0.15), (b, yi, 0.27), (a, yi, 0.27)][::(-1 if sy > 0 else 1)], M["black"])
            p.box((a, y0, 0.27), (b, y1, zf), frame)
            p.box((a, -hd + 0.1, 0.15), (b, hd - 0.1, 0.152), M["steel"])
            p.box((a, -hd + 0.1, 0.268), (b, hd - 0.1, 0.27), M["steel"])
        p.box((-hl + 0.16, y0 - 0.01 * (sy < 0), zt), (hl - 0.16, y1 + 0.01 * (sy > 0), Ht - 0.005), frame)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * hl, sx * (hl - 0.1)))
        p.box((x0, -hd + 0.16, 0.12), (x1, hd - 0.16, zf), frame)
        x0, x1 = sorted((sx * (hl + 0.01), sx * (hl - 0.1)))
        p.box((x0, -hd + 0.16, zt), (x1, hd - 0.16, Ht - 0.005), frame)
    p.box((-hl + 0.1, -hd + 0.1, 0.12), (hl - 0.1, hd - 0.1, 0.14), M["steel"])
    p.r = 0.0
    # flat steel roof, cambered a touch to shed water, inside the 70 mm upstand of the top frame
    zr0, zr1 = Ht - 0.075, Ht - 0.045
    for sy in (-1, 1):
        q4 = [(-hl + 0.1, sy * (hd - 0.1), zr0), (hl - 0.1, sy * (hd - 0.1), zr0), (hl - 0.1, 0.0, zr1), (-hl + 0.1, 0.0, zr1)]
        p.face(q4 if sy < 0 else q4[::-1], roof)
    # cladding with the openings
    i_clad = len(p.F)
    hgt = zt - zf
    cv = 0.04                            # valley set back from the frame face
    front_w = [(-3.4, 1.2, 0.35), (-1.4, 1.2, 0.0), (3.2, 1.2, 0.6)]
    dx = 1.2
    # front (+y): along -x
    s_of = lambda x: (hl - 0.15) - x
    holes = [(s_of(x + w / 2 + 0.05), s_of(x - w / 2 - 0.05), 1.0 - 0.05 - zf, 2.1 + 0.05 - zf) for x, w, _ in front_w]
    holes.append((s_of(dx + 0.53), s_of(dx - 0.53), 0.0, 2.38 - zf))
    _clad(p, (hl - 0.15, hd - cv, zf), (-1, 0, 0), (0, 1, 0), L - 0.3, hgt, body, 0.2, 0.036, holes)
    back_w = [(-2.6, 1.2, 1.0), (2.2, 1.2, 1.0)]
    holes = [(x - w / 2 - 0.05 + hl - 0.15, x + w / 2 + 0.05 + hl - 0.15, 1.0 - 0.05 - zf, 2.1 + 0.05 - zf) for x, w, _ in back_w]
    _clad(p, (-hl + 0.15, -hd + cv, zf), (1, 0, 0), (0, -1, 0), L - 0.3, hgt, body, 0.2, 0.036, holes)
    _clad(p, (hl - cv, -hd + 0.15, zf), (0, 1, 0), (1, 0, 0), D - 0.3, hgt, body, 0.2, 0.036)
    holes = [((hd - 0.15) - 0.35 - 0.05, (hd - 0.15) + 0.35 + 0.05, 1.2 - 0.05 - zf, 2.0 + 0.05 - zf)]
    _clad(p, (-hl + cv, hd - 0.15, zf), (0, -1, 0), (-1, 0, 0), D - 0.3, hgt, body, 0.2, 0.036, holes)
    i_clad_end = len(p.F)
    _auto_uv(p, i_clad)
    _overlay(p, i_clad, i_clad_end, wear, 0.003)
    p.r = 0.004
    # windows: three on the front at different shutter heights, two shuttered on the back, one end
    slat = body
    for x, w, sh in front_w:
        _merge(p, _window(M, frame, glass, w, 1.0, 2.1, sh, slat=slat), _wall_xf((x, hd, 0), (0, 1, 0)))
    for x, w, sh in back_w:
        _merge(p, _window(M, frame, glass, w, 1.0, 2.1, sh, slat=slat), _wall_xf((x, -hd, 0), (0, -1, 0)))
    _merge(p, _window(M, frame, glass, 0.7, 1.2, 2.0, 0.15, slat=slat), _wall_xf((-hl, 0, 0), (-1, 0, 0)))
    # the door: frame trim, steel leaf, kick plate, lever and lock, canopy, light, signs
    q = PM(0.0)
    for a, b in (((-0.53, -0.07, zf), (-0.47, 0.02, 2.38)), ((0.47, -0.07, zf), (0.53, 0.02, 2.38)), ((-0.53, -0.07, 2.32), (0.53, 0.02, 2.38))):
        q.box(a, b, frame)
    q.box((-0.47, -0.04, zf + 0.02), (0.47, -0.02, 2.32), body)
    for z in (0.75, 1.55):
        q.box((-0.4, -0.02, z), (0.4, -0.012, z + 0.4), body)
    q.box((-0.44, -0.02, zf + 0.03), (0.44, -0.015, zf + 0.28), M["alu"])
    q.box((-0.47, -0.035, 0.6), (-0.44, -0.015, 2.1), M["steel"])
    q.r = 0.0
    q.tube([(0.36, -0.015, 1.15), (0.36, 0.035, 1.15), (0.22, 0.04, 1.15)], 0.009, M["alu"], 12, bend=0.02)
    q.r = 0.003
    q.box((0.34, -0.02, 1.08), (0.38, -0.01, 1.22), M["alu"])
    q.cyl((0.36, -0.01, 1.02), (0.36, 0.0, 1.02), 0.014, M["alu"], 12)
    q.r = 0.004
    q.obox((0, 0.22, 2.44), (1.4, 0.48, 0.03), frame, pitch=0.08)
    for s in (-1, 1):
        q.obox((s * 0.6, 0.12, 2.36), (0.03, 0.26, 0.12), frame, pitch=-0.5)
    q.box((0.68, 0.0, 2.08), (0.86, 0.09, 2.2), M["black"])
    q.decal((0.77, 0.091, 2.14), (0, 1, 0), (0, 0, 1), 0.15, 0.09, ST.sign_panel("PS_BulkheadLens", lambda im: im.__setitem__(Ellipsis, im * 0.95 + 0.05), (64, 32), 0.2, 0.05))
    q.decal((0, -0.012, 1.8), (0, 1, 0), (0, 0, 1), 0.46, 0.16,
            ST.sign_panel("PS_Label_SiteOffice", lambda im: (SP_fill(im, "#1f4e8a"), SP_text(im, "SITE OFFICE", 0.5, 0.25, 0.5, "#ffffff")), (384, 128)))
    q.decal((0.85, 0.02, 1.45), (0, 1, 0), (0, 0, 1), 0.3, 0.4, ST.ppe_sign())
    q.decal((-1.0, 0.02, 1.35), (0, 1, 0), (0, 0, 1), 0.3, 0.2,
            ST.sign_panel("PS_Label_FirstAid", lambda im: (SP_fill(im, "#0b7a3e"), SP_text(im, "FIRST AID", 0.5, 0.6, 0.18, "#ffffff"), SP_cross(im)), (300, 200)))
    _merge(p, q, _wall_xf((dx, hd, 0), (0, 1, 0)))
    # galvanised steps: two treads and a landing up to the raised floor, stringers, handrails
    steps = PM(0.0)
    grate = ST.grating()
    galv = ST.galvanised_img()
    for k, (z, d0, d1) in enumerate(((0.153, 0.62, 0.92), (0.306, 0.36, 0.64), (0.459, 0.02, 0.38))):
        steps.box((-0.5, d0, z - 0.03), (0.5, d1, z), grate)
    for s in (-1, 1):
        x0, x1 = sorted((s * 0.5, s * 0.506))
        steps.extrude_yz([(0.94, 0.0), (0.94, 0.12), (0.38, 0.47), (0.02, 0.47), (0.02, 0.0)], x0, x1, galv)
    for s in (-1, 1):
        x = s * 0.53
        steps.tube([(x, 0.9, 0.0), (x, 0.9, 1.15), (x, 0.1, 1.45), (x, 0.03, 1.45)], 0.02, galv, 16, bend=0.1)
        steps.tube([(x, 0.45, 0.32), (x, 0.45, 1.3)], 0.017, galv, 16)
        steps.tube([(x, 0.9, 0.62), (x, 0.06, 0.95)], 0.014, galv, 12)
    # downpipe from a roof outlet at the back corner, power inlet, vents
    p.r = 0.0
    xp = -hl + 0.3
    p.tube([(xp, -hd - 0.02, Ht - 0.08), (xp, -hd - 0.09, Ht - 0.16), (xp, -hd - 0.09, 0.25), (xp, -hd - 0.16, 0.12)], 0.034, M["black"], 16, bend=0.08)
    for z in (2.0, 1.0):
        p.box((xp - 0.04, -hd - 0.13, z), (xp + 0.04, -hd, z + 0.03), M["black"])
    p.r = 0.0
    p.r = 0.004
    for x in (-hl - 0.0, hl):
        s = 1 if x > 0 else -1
        y0, y1 = (0.5, 0.8)
        a, b = sorted((x, x + s * 0.03))
        p.box((a, y0, 2.2), (b, y1, 2.38), frame)
        for k in range(5):
            p.obox((x + s * 0.035, (y0 + y1) / 2, 2.22 + k * 0.032), (0.012, 0.28, 0.025), M["steel"], roll=s * 0.6)
    # hire stencils on the front
    hire = ST.stencil("PS_Label_CabinHire", [("ASHDOWN MODULAR", 0.3, 0.1, "black"), ("HIRE 01632 960418   UNIT AM-32/1147", 0.18, 0.6, "bold")],
                      (1024, 160), color="#2c3e66", wear=0.15)
    xs, ws = -2.4, 1.9
    _corr_panel(p, (hl - 0.15, hd - cv, zf), (-1, 0, 0), (0, 1, 0), L - 0.3, hgt, hire, 0.2, 0.036, off=0.004,
                s0=s_of(xs + ws / 2), s1=s_of(xs - ws / 2), h0=0.45 - zf, h1=0.45 - zf + ws * 160 / 1024,
                uv01=(-(xs + ws / 2), -(xs - ws / 2), 0.45, 0.45 + ws * 160 / 1024))
    _auto_uv(p)
    # the body stands 180 mm up on precast blocks (the dark gap under it), 2.62 m to the top so
    # two still stack; the steps, services and blocks are built at their real heights
    zb = 0.18
    fin = PM(0.0)
    _merge(fin, p, Matrix.Translation(Vector((0, 0, zb))) @ Matrix.Diagonal((1, 1, (Ht - zb) / Ht, 1)))
    _auto_uv(steps)
    _merge(fin, steps, _wall_xf((dx, hd, 0), (0, 1, 0)))
    blk = ST.concrete_block("PS_CabinPad", "#87827a", cell=1.0, seed=230, stains=0.4)
    fin.r = 0.01
    for x in (-hl + 0.2, -1.4, 1.4, hl - 0.2):
        for y in (-hd + 0.25, hd - 0.25):
            fin.box((x - 0.22, y - 0.22, -0.02), (x + 0.22, y + 0.22, zb), blk)
    # lifting eyes on the top corner castings
    fin.r = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * (hl - 0.08), sy * (hd - 0.08)
            fin.tube([(cx - 0.04, cy, Ht - 0.005), (cx - 0.04, cy, Ht + 0.03), (cx + 0.04, cy, Ht + 0.03), (cx + 0.04, cy, Ht - 0.005)],
                     0.009, M["steel"], 8, bend=0.03)
    # services on the +x end: hook-up box and armoured cable off across the ground, a waste pipe
    sz = (Ht - zb) / Ht
    fin.r = 0.006
    fin.box((hl, -0.62, zb + 0.55 * sz), (hl + 0.13, -0.36, zb + 0.82 * sz), M["grey"])
    fin.cyl((hl + 0.13, -0.49, zb + 0.64 * sz), (hl + 0.21, -0.49, zb + 0.64 * sz), 0.045, M["yellow"], 16)
    fin.r = 0.0
    cable = [(hl + 0.21, -0.49, zb + 0.64 * sz), (hl + 0.3, -0.49, zb + 0.55 * sz), (hl + 0.34, -0.5, 0.05), (hl + 0.7, -0.62, 0.035),
             (hl + 1.4, -0.4, 0.035), (hl + 2.2, -0.75, 0.035), (hl + 3.2, -0.55, 0.035), (hl + 4.2, -0.9, 0.035)]
    fin.tube(cable, 0.014, M["black"], 10, bend=0.25)
    fin.tube([(hl - 0.02, 0.6, zb + 0.15), (hl + 0.08, 0.6, zb + 0.1), (hl + 0.08, 0.6, 0.03), (hl + 0.5, 0.62, 0.03)], 0.028, M["grey"], 12, bend=0.06)
    fin.tube([(hl - 0.02, 0.85, zb + 0.2), (hl + 0.1, 0.85, zb + 0.16), (hl + 0.1, 0.85, 0.04), (hl + 0.3, 1.2, 0.035), (hl + 1.6, 1.6, 0.035)],
             0.012, M["blue"], 8, bend=0.08)
    _auto_uv(fin)
    return fin.obj("site_cabin", col, 30)


def SP_fill(im, c):
    from .site_paint import lin
    im[...] = lin(c)


def SP_text(im, s, x, y, cap, c, kind="bold"):
    from . import site_paint as SP
    H, W = im.shape[:2]
    SP.text(im, s, x * W, y * H, cap * H, SP.lin(c), kind, "c", max_w=W * 0.92)


def SP_cross(im):
    from . import site_paint as SP
    H, W = im.shape[:2]
    c = SP.lin("#ffffff")
    SP.put(im, SP.rect_mask(H, W, W * 0.44, H * 0.08, W * 0.56, H * 0.52), c)
    SP.put(im, SP.rect_mask(H, W, W * 0.35, H * 0.24, W * 0.65, H * 0.36), c)


def portaloo(col, M):
    """A hire-fleet portable toilet (1.16 x 1.26 m, 2.3 m to the roof): a moulded base tray with
    forklift slots and the tank bulging at the back, walls with deep vertical stiffening grooves
    and round corners, a door recessed in its frame on a full-height hinge, with louvres, a pull
    handle and a VACANT/ENGAGED latch, vents high on the sides, a translucent domed roof with a
    rim, the vent stack with its cowl; the hire firm's livery and notices, sun-faded, grimy at
    the foot and scuffed. Door on -y (the generator turns it to face the cabins)."""
    from . import site_textures as ST
    p = PM(0.0)
    blue = ST.moulded_plastic("PS_LooShell", "#2b5c9e", seed=140)
    roofm = ST.moulded_plastic("PS_LooRoof3", "#e1e1da", seed=141, grime=0.0, fade=0.0, scuffs=0, rough=0.5, emission=0.2, pools=0.45)
    base = ST.moulded_plastic("PS_LooBase", "#20406e", seed=142, grime=0.6, rough=0.7)
    hx, hy, r = 0.55, 0.6, 0.06
    z0, z1 = 0.15, 2.2
    groove = [(0.0, 0.0), (0.62, 0.0), (0.7, 1.0), (0.86, 1.0), (0.94, 0.0)]
    # base tray: rounded, with forklift slots front and back
    p.r = 0.012
    p.loft([rrect(2 * hx + 0.06, 2 * hy + 0.06, r + 0.03, z) for z in (0.0, z0)], base)
    p.r = 0.0
    for sy in (-1, 1):
        for x in (-0.3, 0.3):
            y = sy * (hy + 0.031)
            p.decal((x, y, 0.075), (0, sy, 0), (0, 0, 1), 0.26, 0.075, M["black"], lift=0.001)
    # walls: grooved sides and back, round corners, a front with the door opening
    # the grooves stop 120 mm short of the top and bottom into flat bands, their ends closed
    gb, gt = z0 + 0.12, z1 - 0.12
    def grooved(O, U, N, length):
        _corr_panel(p, O, U, N, length, z1 - z0, blue, 0.2, 0.0, shape=groove, h1=gb - z0)
        _corr_panel(p, O, U, N, length, z1 - z0, blue, 0.2, -0.025, shape=groove, h0=gb - z0, h1=gt - z0)
        _corr_panel(p, O, U, N, length, z1 - z0, blue, 0.2, 0.0, shape=groove, h0=gt - z0)
        O, U, N = (np.asarray(v, float) for v in (O, U, N))
        n = max(1, int(round(length / 0.2)))
        st = length / n
        for k in range(n):
            a_, b_ = k * st + st * 0.62, k * st + st * 0.94
            for z, up in ((gb, 1), (gt, -1)):
                q = [O + U * a_ + np.array((0, 0, z)), O + U * b_ + np.array((0, 0, z)),
                     O + U * (b_ - st * 0.08) - N * 0.025 + np.array((0, 0, z)), O + U * (a_ + st * 0.08) - N * 0.025 + np.array((0, 0, z))]
                p.face(q if up > 0 else q[::-1], blue)
    for sx in (-1, 1):
        N = (sx, 0, 0)
        U = np.cross((0, 0, 1.0), N)
        grooved((sx * hx, -sx * (hy - r), z0), U, N, 2 * (hy - r))
    grooved((hx - r, hy, z0), (-1, 0, 0), (0, 1, 0), 2 * (hx - r))
    # a skirt where the walls meet the base
    p.r = 0.008
    p.loft([rrect(2 * hx + 0.024, 2 * hy + 0.024, r + 0.012, z) for z in (z0 - 0.01, z0 + 0.05)], blue)
    p.r = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * (hx - r), sy * (hy - r)
            a0 = math.atan2(sy, sx) - math.pi / 4
            pts = [(cx + r * math.cos(a0 + math.pi / 2 * k / 4), cy + r * math.sin(a0 + math.pi / 2 * k / 4)) for k in range(5)]
            for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
                p.face([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], blue)
    # front (-y): wall round the doorway, a proud frame, the door set back in it
    y = -hy
    dw, dh = 0.82, 1.95
    p.face([(-hx + r, y, z0), (-dw / 2 - 0.04, y, z0), (-dw / 2 - 0.04, y, z1), (-hx + r, y, z1)], blue)
    p.face([(dw / 2 + 0.04, y, z0), (hx - r, y, z0), (hx - r, y, z1), (dw / 2 + 0.04, y, z1)], blue)
    p.face([(-dw / 2 - 0.04, y, z0 + dh + 0.04), (dw / 2 + 0.04, y, z0 + dh + 0.04), (dw / 2 + 0.04, y, z1), (-dw / 2 - 0.04, y, z1)], blue)
    p.r = 0.008
    for a, b in (((-dw / 2 - 0.04, y - 0.03, z0), (-dw / 2, y + 0.02, z0 + dh + 0.04)), ((dw / 2, y - 0.03, z0), (dw / 2 + 0.04, y + 0.02, z0 + dh + 0.04)),
                 ((-dw / 2 - 0.04, y - 0.03, z0 + dh), (dw / 2 + 0.04, y + 0.02, z0 + dh + 0.05))):
        p.box(a, b, blue)
    p.r = 0.006
    yd = y - 0.005
    p.box((-dw / 2 + 0.005, yd, z0 + 0.01), (dw / 2 - 0.005, yd + 0.04, z0 + dh - 0.005), blue)
    for zr in (0.45, 0.95, 1.4):
        p.box((-dw / 2 + 0.07, yd - 0.012, zr), (dw / 2 - 0.07, yd, zr + 0.05), blue)
    # louvres at the top of the door
    p.r = 0.0
    p.box((-0.3, yd - 0.004, 1.86), (0.3, yd, 2.05), M["black"])
    for k in range(6):
        z = 1.875 + k * 0.03
        p.obox((0, yd - 0.012, z), (0.6, 0.012, 0.02), blue, pitch=-0.6)
    # hinge strip, pull handle, latch with its indicator
    p.r = 0.003
    p.cyl((-dw / 2 - 0.005, yd - 0.01, z0 + 0.05), (-dw / 2 - 0.005, yd - 0.01, z0 + dh - 0.05), 0.012, M["grey"], 12)
    p.cyl((0.3, yd - 0.05, 0.98), (0.3, yd - 0.05, 1.18), 0.012, M["grey"], 12)
    for z in (0.98, 1.18):
        p.cyl((0.3, yd, z), (0.3, yd - 0.055, z), 0.012, M["grey"], 10)
    p.box((0.22, yd - 0.025, 1.26), (0.38, yd, 1.34), M["grey"])
    vac = ST.sign_panel("PS_Label_LooVacant", lambda im: (SP_fill(im, "#1c8a3e"), SP_text(im, "VACANT", 0.5, 0.28, 0.44, "#ffffff")),
                        (128, 48), grime=0.0)
    p.decal((0.3, yd - 0.026, 1.3), (0, -1, 0), (0, 0, 1), 0.1, 0.035, vac, lift=0.001)
    # livery and notices
    liv = ST.sign_panel("PS_Label_LooLivery", _loo_livery, (256, 320), grime=0.12)
    p.decal((0.0, yd - 0.013, 1.12), (0, -1, 0), (0, 0, 1), 0.32, 0.4, liv, lift=0.001)
    wash = ST.sign_panel("PS_Label_LooWash", lambda im: (SP_fill(im, "#1f5aa6"), SP_text(im, "NOW WASH", 0.5, 0.22, 0.2, "#ffffff"),
                                                    SP_text(im, "YOUR HANDS", 0.5, 0.56, 0.2, "#ffffff")), (192, 128))
    p.decal((-0.22, yd - 0.001, 1.7), (0, -1, 0), (0, 0, 1), 0.16, 0.1, wash, lift=0.001)
    num = ST.stencil("PS_Label_LooNumber", [("AW 214", 0.6, 0.15, "bold")], (256, 64), color="#f2f2ee", wear=0.15)
    for sx in (-1, 1):
        p.decal((sx * (hx + 0.001), 0.25 * sx, 1.6), (sx, 0, 0), (0, 0, 1), 0.3, 0.075, num, lift=0.001)
    # side vents, the tank at the back
    for sx in (-1, 1):
        p.box((sx * hx - 0.012 * (sx < 0), -0.15, 1.9), (sx * hx + 0.012 * (sx > 0), 0.15, 1.96), M["grey"])
        for k in range(3):
            p.obox((sx * (hx + 0.014), 0, 1.905 + k * 0.022), (0.012, 0.3, 0.016), blue, roll=sx * 0.6)
    # the door's self-closing arm, and scuffing round the handle
    p.r = 0.0
    p.box((-0.36, yd - 0.04, 1.98), (-0.12, yd - 0.025, 2.0), M["grey"])
    p.box((-0.44, yd - 0.035, 1.97), (-0.36, yd - 0.01, 2.02), M["grey"])
    # roof: rim and translucent dome, the vent stack and cowl
    p.r = 0.015
    p.loft([rrect(2 * hx + 0.08, 2 * hy + 0.08, r + 0.04, z) for z in (z1 - 0.02, z1 + 0.06)], roofm)
    p.r = 0.0
    p.loft([[(x * (1 - 0.55 * t), y_ * (1 - 0.55 * t), z1 + 0.04 + 0.1 * math.sin(t * math.pi / 2)) for x, y_, _ in rrect(2 * hx + 0.0, 2 * hy + 0.0, r + 0.02)]
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)], roofm, cap0=False)
    p.cyl((0.32, 0.38, z1), (0.32, 0.38, 2.62), 0.05, M["black"], 16)
    # a mushroom cowl on three short legs over the pipe
    p.lathe([(0.0, 2.7), (0.06, 2.69), (0.095, 2.665), (0.1, 2.655), (0.095, 2.65), (0.0, 2.66)], M["black"], (0.32, 0.38, 0), sides=20)
    for k in range(3):
        a = 2 * math.pi * k / 3
        p.cyl((0.32 + 0.045 * math.cos(a), 0.38 + 0.045 * math.sin(a), 2.6), (0.32 + 0.045 * math.cos(a), 0.38 + 0.045 * math.sin(a), 2.66), 0.006, M["black"], 6)
    # corner lifting lugs moulded into the roof rim
    p.r = 0.008
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((sx * (hx + 0.01) - 0.04, sy * (hy + 0.01) - 0.04, z1 + 0.04), (sx * (hx + 0.01) + 0.04, sy * (hy + 0.01) + 0.04, z1 + 0.1), roofm)
    # hinge leaves on the hinge side of the door
    p.r = 0.002
    for zh_ in (0.45, 1.2, 1.95):
        p.box((-dw / 2 - 0.03, yd - 0.016, zh_), (-dw / 2 + 0.06, yd - 0.008, zh_ + 0.09), M["grey"])
    _auto_uv(p)
    return p.obj("portaloo", col, 30)


def _loo_livery(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    img[...] = SP.lin("#f4f4f0")
    SP.put(img, SP.rect_mask(H, W, 0, 0, W, H * 0.34), SP.lin("#1f3f73"))
    SP.text(img, "ASHDOWN", W / 2, H * 0.07, H * 0.1, SP.lin("#ffffff"), "black", "c", max_w=W * 0.9)
    SP.text(img, "WELFARE HIRE", W / 2, H * 0.21, H * 0.055, SP.lin("#f2c200"), "bold", "c", max_w=W * 0.9)
    SP.text(img, "SERVICED WEEKLY", W / 2, H * 0.42, H * 0.05, SP.lin("#1f3f73"), "bold", "c", max_w=W * 0.9)
    SP.text(img, "Please report faults", W / 2, H * 0.55, H * 0.045, SP.lin("#333333"), "regular", "c", max_w=W * 0.9)
    SP.text(img, "01632 960733", W / 2, H * 0.66, H * 0.08, SP.lin("#1f3f73"), "bold", "c", max_w=W * 0.9)
    SP.sign_prohibit(img, W * 0.3, H * 0.87, H * 0.08)
    SP.text(img, "NO SMOKING", W * 0.62, H * 0.84, H * 0.05, SP.lin("#c8102e"), "bold", "c", max_w=W * 0.5)


def _prism(p, poly, z0, z1, mat):
    """A vertical prism over a counter-clockwise (x, y) polygon, faces outward."""
    n = len(poly)
    for j in range(n):
        (ax, ay), (bx, by) = poly[j], poly[(j + 1) % n]
        p.face([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], mat)
    p.face([(x, y, z0) for x, y in poly[::-1]], mat)
    p.face([(x, y, z1) for x, y in poly], mat)


def _shell(p, B, T, zb, zt, mat, inward=False):
    """Side walls between a bottom and a top ring (counter-clockwise from above)."""
    n = len(B)
    for j in range(n):
        q = [(*B[j], zb), (*B[(j + 1) % n], zb), (*T[(j + 1) % n], zt), (*T[j], zt)]
        p.face(q[::-1] if inward else q, mat)


def skip(col, M):
    """An 8-yard builder's skip (3.5 m long at the top, 2.3 m at the base, 1.68 m wide, 1.3 m to
    the rim): open steel body 6 mm thick with a box-section rim, pressed channel ribs on the sides
    and raked ends, a base channel, chunky lifting and tipping trunnions, a drop door with locking
    bars on one end, red/white reflective markers, the hire firm's name and number; battered,
    dented and rusting, with a mixed load (fill, bricks, block, timber, plasterboard, a black bag,
    a bent strap) sitting just below the rim."""
    from . import site_textures as ST
    p = PM(0.0)
    s = ST.paint_detail("TINT_skip", "#c8a01e", cell=4.0, wall_h=1.3, seed=100, foot=0.45, runs=0.55, patches=5, dents=0.009,
                        rough=0.6)
    wear = ST.wear_soft("PS_SkipWear3", cell=4.0, wall_h=1.3, seed=111, chips=-0.35, foot=0.25, runs=0.9, scrapes=30, top_bleed=0.7,
                        bands=((1.24, 0.06), (1.2, 0.03)), mud=0.35, scrape_band=(0.06, 0.45))
    inside = ST.rusty_steel("PS_SkipInside", seed=120, paint="#8a7430", amount=0.75)
    hw = 0.84
    zb, zt = 0.1, 1.24
    xb, xt = 1.15, 1.75
    slope = (xt - xb) / (zt - zb)
    B = [(-xb, -hw), (xb, -hw), (xb, hw), (-xb, hw)]
    T = [(-xt, -hw), (xt, -hw), (xt, hw), (-xt, hw)]
    i0 = len(p.F)
    # long sides subdivided and pushed in by a few soft dents between the ribs; ends plain
    drng = np.random.default_rng(41)
    dents = [(sy, drng.uniform(-1.45, 1.45), drng.uniform(0.3, 0.95), drng.uniform(0.12, 0.25), drng.uniform(0.015, 0.045))
             for sy in (-1, 1) for _ in range(3)]
    def dent(x, z, sy):
        # each panel between the channels dished in, plus a few impact dents
        f = ((x + 1.2) / 0.6) % 1.0
        dish = 0.025 * math.sin(math.pi * f) * math.sin(math.pi * min(max((z - 0.1) / 1.14, 0.0), 1.0))
        return dish + sum(d * math.exp(-((x - dx) ** 2 + (z - dz) ** 2) / (r * r)) for s_, dx, dz, r, d in dents if s_ == sy)
    def wall(Bq, Tq, z0, mat, inward, off=0.0):
        n = len(Bq)
        for j in range(n):
            (b0, b1), (t0, t1) = (Bq[j], Bq[(j + 1) % n]), (Tq[j], Tq[(j + 1) % n])
            long_side = abs(b0[1] - b1[1]) < 1e-6
            nu, nv = (18, 7) if long_side else (1, 1)
            sy = 1 if b0[1] > 0 else -1
            G = []
            for iv in range(nv + 1):
                fz = iv / nv
                z = z0 + (zt - z0) * fz
                row = []
                for iu in range(nu + 1):
                    fu = iu / nu
                    pb = np.array(b0) + (np.array(b1) - np.array(b0)) * fu
                    pt = np.array(t0) + (np.array(t1) - np.array(t0)) * fu
                    q = pb + (pt - pb) * fz
                    yy = q[1] - (sy * dent(q[0], z, sy) if long_side else 0.0)
                    row.append((q[0], yy, z))
                G.append(row)
            for iv in range(nv):
                for iu in range(nu):
                    f = [G[iv][iu], G[iv][iu + 1], G[iv + 1][iu + 1], G[iv + 1][iu]]
                    p.face(f[::-1] if inward else f, mat)
    wall(B, T, zb, s, False)
    p.face([(x, y, zb) for x, y in B[::-1]], s)
    t = 0.006
    Bi = [(x * (xb - t) / xb, y * (hw - t) / hw) for x, y in B]
    Ti = [(x * (xt - t) / xt, y * (hw - t) / hw) for x, y in T]
    i_in = len(p.F)
    wall(Bi, Ti, zb + t, inside, True)
    p.face([(x, y, zb + t) for x, y in Bi], inside)
    # rim: a box-section tube round the top
    i_rim = len(p.F)
    rim = []
    for j in range(4):
        a_, b_ = np.array(T[j]), np.array(T[(j + 1) % 4])
        for k in range(6):
            q = a_ + (b_ - a_) * k / 6
            sag = -0.03 * math.sin(math.pi * k / 6) if (j == 0 and 2 <= k <= 4) else 0.0
            rim.append((q[0] * (xt + 0.025) / xt, q[1] * (hw + 0.025) / hw, zt + 0.03 + sag))
    p.r = 0.006
    p.sweep(rim, [(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)], s, closed=True)
    # pressed channels: five up each side, two up each end, one along each side near the foot
    i_rib = len(p.F)
    p.r = 0.008
    for sy in (-1, 1):
        for k in range(5):
            xc = -1.2 + k * 0.6
            poly = [(xc - 0.07, sy * hw), (xc - 0.04, sy * (hw + 0.05)), (xc + 0.04, sy * (hw + 0.05)), (xc + 0.07, sy * hw)]
            _prism(p, poly if sy < 0 else poly[::-1], zb, zt - 0.005, s)
        poly = [(-xb - 0.0, sy * hw), (xb, sy * hw), (xb, sy * (hw + 0.045)), (-xb, sy * (hw + 0.045))]
        p.box((-xb + 0.12, min(sy * hw, sy * (hw + 0.045)), 0.24), (xb - 0.12, max(sy * hw, sy * (hw + 0.045)), 0.33), s)
    ang = math.atan(slope)
    Lf = (zt - zb) / math.cos(ang)
    for sx in (-1, 1):
        for yc in (-0.5, 0.5):
            zc = (zb + zt) / 2
            xc = sx * (xb + slope * (zc - zb))
            nrm = np.array((sx * math.cos(ang), 0, -math.sin(ang)))
            c = np.array((xc, yc, zc)) + nrm * 0.022
            p.obox(c, (0.045, 0.12, Lf - 0.02), s, roll=sx * ang)
    # base runners and bearers
    p.r = 0.005
    for y in (-0.55, 0.55):
        p.box((-xb + 0.05, y - 0.06, 0.0), (xb - 0.05, y + 0.06, zb), M["steel"])
    for x in (-0.8, 0.0, 0.8):
        p.box((x - 0.05, -0.62, 0.02), (x + 0.05, 0.62, zb), M["steel"])
    i_end = len(p.F)
    _auto_uv(p, i0)
    _overlay(p, i0, i_in, wear, 0.003)
    _overlay(p, i_rim, i_end - 5, wear, 0.003)
    # lifting trunnions near each end, tipping trunnions low on the +x end
    p.r = 0.004
    for sx in (-1, 1):
        for sy in (-1, 1):
            for x, z, big in ((sx * 1.38, 1.0, True), (sx * 1.0, 0.3, False)):
                if not big and sx < 0:
                    continue
                y0 = sy * hw
                p.box((x - 0.11, min(y0, y0 + sy * 0.035), z - 0.13), (x + 0.11, max(y0, y0 + sy * 0.035), z + 0.13), s)
                r = 0.045 if big else 0.04
                p.cyl((x, y0 + sy * 0.03, z), (x, y0 + sy * 0.13, z), r, M["steel"], 16)
                p.cyl((x, y0 + sy * 0.13, z), (x, y0 + sy * 0.155, z), r * 1.6, M["steel"], 16)
    # drop door on the +x end: frame, hinges at the foot, two locking bars with latches
    nrm = np.array((math.cos(ang), 0, -math.sin(ang)))
    up = np.array((math.sin(ang), 0, math.cos(ang)))
    o = np.array((xb, 0, zb))
    at = lambda u, y, d: o + up * u + np.array((0, y, 0)) + nrm * d
    for (u0, u1, y0, y1) in ((0.12, 1.1, -0.62, -0.57), (0.12, 1.1, 0.57, 0.62), (1.05, 1.1, -0.62, 0.62), (0.12, 0.17, -0.62, 0.62)):
        p.obox(at((u0 + u1) / 2, (y0 + y1) / 2, 0.012), (0.02, y1 - y0, u1 - u0), s, roll=ang)
    for yc in (-0.3, 0.3):
        p.cyl(at(0.05, yc - 0.12, 0.03), at(0.05, yc + 0.12, 0.03), 0.03, M["steel"], 12)
    for yc in (-0.42, 0.42):
        p.cyl(at(0.2, yc, 0.05), at(1.14, yc, 0.05), 0.016, M["steel"], 12)
        for u in (0.35, 0.9):
            p.obox(at(u, yc, 0.03), (0.04, 0.07, 0.05), M["steel"], roll=ang)
        p.obox(at(1.16, yc, 0.05), (0.08, 0.08, 0.1), M["steel"], roll=ang)
    # reflective markers at the top corners of each end, the firm's name and number on the sides
    chev = ST.sign_panel("PS_SkipReflector", _chevrons, (128, 256), rough=0.3, grime=0.25)
    for sx in (-1, 1):
        n = np.array((sx * math.cos(ang), 0, -math.sin(ang)))
        v = np.array((sx * math.sin(ang), 0, math.cos(ang)))
        for yc in (-0.68, 0.68):
            c = np.array((sx * (xb + slope * (1.0 - zb)), yc, 1.0))
            p.decal(c, n, v, 0.2, 0.36, chev, lift=0.004)
    name = ST.stencil("PS_Label_SkipName", [("BRAMLEY", 0.42, 0.08, "black"), ("SKIP HIRE  01632 960527", 0.2, 0.62, "bold")],
                      (1024, 256), color="#1d1e1f", wear=0.35, seed=3)
    # the name follows the side's profile: over the panels and up and over each pressed channel
    rib = [(0.0, 0.0), (0.05, 1.0), (0.1833, 1.0), (0.2333, 0.0)]
    for sy in (-1, 1):
        N = np.array((0, sy, 0.0))
        U = np.cross((0, 0, 1.0), N)
        O = np.array((-1.27 * U[0], sy * hw, 0.0))
        a0, a1 = 1.27 - 1.1, 1.27 + 1.1
        r0 = O @ U + a0
        _corr_panel(p, O, U, N, 3.0, 1.3, name, 0.6, 0.05, off=0.0025, s0=a0, s1=a1, h0=0.505, h1=1.055,
                    uv01=(r0, r0 + 2.2, 0.505, 1.055), shape=rib)
    # the load
    _skip_load(p, M, ST)
    return p.obj("skip", col, 30)


def _sack(p, c, r, mat, rng, nu=22, nv=12):
    """A filled plastic sack slumped on its side: a lumpy, flattened ball with a gathered neck."""
    c = np.asarray(c, float)
    ph = rng.uniform(0, 6.28, 5)
    rows = []
    for j in range(nv + 1):
        t = -math.pi / 2 + math.pi * j / nv
        ring = []
        for k in range(nu):
            a = 2 * math.pi * k / nu
            d = np.array((math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)))
            lump = 1 + 0.09 * math.sin(3 * a + ph[0]) * math.cos(2 * t + ph[1]) + 0.025 * math.sin(7 * a + 5 * t + ph[2])
            q = d * r * lump * np.array((1.35, 0.95, 0.62))
            # the neck: gathered to a twist at the +x end
            if d[0] > 0.75:
                q[1:] *= 1 - 2.2 * (d[0] - 0.75)
                q[0] += r * 0.5 * (d[0] - 0.75)
            q[2] = max(q[2], -r * 0.35)
            ring.append(c + q)
        rows.append(ring)
    for j in range(nv):
        for k in range(nu):
            k2 = (k + 1) % nu
            p.face([rows[j][k], rows[j][k2], rows[j + 1][k2], rows[j + 1][k]], mat)


def _chevrons(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    SP.hazard_stripes(img, 0, 0, W, H, "#f2f0ea", "#c8102e", pitch=70, angle=45)


def _skip_load(p, M, ST):
    """Rubble just under the rim: a lumpy fill and loose pieces on it."""
    fill = ST.rubble_img()
    rng = np.random.default_rng(7)
    nu, nv = 22, 11
    ph = rng.uniform(0, 6.28, 6)

    def hgt(u, v):
        bump = 0.5 + 0.25 * math.sin(3.1 * u + ph[0]) * math.sin(2.3 * v + ph[1]) + 0.15 * math.sin(7 * u + 5 * v + ph[2])
        return 0.95 + 0.62 * (1 - u * u) ** 0.8 * (1 - v * v) ** 0.6 * bump + 0.03 * math.sin(17 * u + ph[3]) * math.sin(13 * v + ph[4])
    G = [[(u * 1.6, v * 0.828, hgt(u, v)) for u in np.linspace(-1, 1, nu + 1)] for v in np.linspace(-1, 1, nv + 1)]
    p.r = 0.0
    for j in range(nv):
        for i in range(nu):
            p.face([G[j][i], G[j][i + 1], G[j + 1][i + 1], G[j + 1][i]], fill)
    brick = ST.flat_steel("PS_BrickBroken", "#93412c", cell=0.5, seed=130, dirt=0.25, rough=0.9, pools=True)
    block = ST.flat_steel("PS_ConcreteChunk", "#948f85", cell=0.8, seed=131, dirt=0.3, rough=0.92)
    wood = ST.flat_steel("PS_TimberOffcut", "#a37b4c", cell=0.6, seed=132, dirt=0.3, rough=0.8)
    board = ST.flat_steel("PS_Plasterboard", "#c4bfb4", cell=1.2, seed=133, dirt=0.35, rough=0.9)
    bag = ST.builders_bag()
    p.r = 0.0
    for k in range(40):
        u, v = rng.uniform(-0.78, 0.78), rng.uniform(-0.7, 0.7)
        z = hgt(u, v)
        x, y = u * 1.6, v * 0.828
        kind = k % 8
        yaw = rng.uniform(0, 6.28)
        tilt = rng.uniform(-0.5, 0.5, 2)
        if kind in (0, 1, 2):
            p.obox((x, y, z + 0.03), (0.215 * rng.uniform(0.4, 1.0), 0.1, 0.065), brick, yaw=yaw, pitch=tilt[0], roll=tilt[1])
        elif kind == 3:
            p.obox((x, y, z + 0.05), rng.uniform(0.12, 0.3, 3), block, yaw=yaw, pitch=tilt[0], roll=tilt[1])
        elif kind == 4:
            p.obox((x * 0.8, y * 0.5, z + 0.03), (rng.uniform(0.4, 0.7), 0.09, 0.045), wood, yaw=rng.uniform(-0.4, 0.4), pitch=tilt[0] * 0.4, roll=tilt[1] * 0.4)
        elif kind == 5 and k < 16:
            p.obox((x * 0.8, y * 0.6, z + 0.04), (rng.uniform(0.3, 0.5), rng.uniform(0.2, 0.35), 0.0125), board, yaw=yaw, pitch=tilt[0] * 0.6, roll=tilt[1] * 0.6)
        elif kind == 6:
            p.obox((x, y, z + 0.06), (0.44, 0.215, 0.1), block, yaw=yaw, pitch=tilt[0] * 0.5, roll=tilt[1] * 0.5)
    # a plank and a sheet of plasterboard sticking up out of the load
    p.r = 0.003
    p.obox((0.9, 0.35, 1.45), (1.3, 0.1, 0.035), wood, yaw=0.3, roll=0.55)
    p.obox((-0.25, -0.45, 1.42), (0.7, 0.5, 0.0125), board, yaw=-0.2, pitch=0.6)
    p.r = 0.0
    # a black builder's bag and a bent steel strap
    p.r = 0.0
    _sack(p, (-0.9, 0.3, hgt(-0.56, 0.36) + 0.06), 0.28, bag, rng)
    p.tube([(0.4, -0.5, hgt(0.25, -0.6) + 0.02), (0.75, -0.35, hgt(0.47, -0.42) + 0.15), (1.0, -0.1, hgt(0.62, -0.12) + 0.05),
            (1.2, 0.2, hgt(0.75, 0.24) + 0.2)], 0.012, ST.rusty_steel("PS_StrapRust", seed=134, amount=0.9), 6, bend=0.12)
    _auto_uv(p)


def tower_crane(col, M):
    """A flat-top tower crane (a Liebherr 172 EC-B class): 42 m of 1.6 m square lattice mast on a
    cast pad with fixing angles, an internal ladder and rest platforms; slewing ring and turntable,
    a glazed operator's cab on the side; a 50 m triangular jib (two bottom chords the trolley runs
    on, one top chord, laced on all three faces, tapering at the tip); a 15 m counter-jib with a
    grating walkway, handrails and the contractor's banner, hoist winch, cabinets and four concrete
    ballast slabs; a wheeled trolley with two falls of rope to a hook block; anemometer and red
    aviation lights."""
    from . import site_textures as ST
    from .props import emissive
    p = PM(0.0)
    y = ST.flat_steel("PS_CraneYellow2", "#d9a300", cell=3.0, seed=150, dirt=0.45, rough=0.5)
    grey = ST.flat_steel("PS_CraneGrey", "#4b4f53", cell=2.0, seed=151, dirt=0.3, rough=0.55)
    pad = ST.concrete_block("PS_CranePad", "#a29e95", cell=3.0, seed=152, stains=0.2)
    ballast = ST.concrete_block("PS_Ballast", "#8f8b83", seed=153, stains=0.6)
    grate = ST.grating()
    red = emissive("PS_AvLight", "#ff2a1a", 4.0)
    a, H = 0.8, 42.0
    chords = [(-a, -a), (a, -a), (a, a), (-a, a)]
    p.r = 0.03
    p.box((-3, -3, -0.7), (3, 3, 0.3), pad)
    p.r = 0.0
    for cx, cy in chords:
        p.box((cx - 0.26, cy - 0.26, 0.3), (cx + 0.26, cy + 0.26, 0.34), grey)
        for d in ((1, 0), (0, 1)):
            p.box((cx - 0.025 - 0.15 * d[0], cy - 0.025 - 0.15 * d[1], 0.34), (cx + 0.025 + 0.15 * d[0], cy + 0.025 + 0.15 * d[1], 0.75), grey)
        p.box((cx - 0.1, cy - 0.1, 0.34), (cx + 0.1, cy + 0.1, H), y)
    n = int(round((H - 0.75) / 1.95))
    panel = (H - 0.75) / n
    for k in range(n + 1):
        z = 0.75 + k * panel
        for i in range(4):
            (x0, y0), (x1, y1) = chords[i], chords[(i + 1) % 4]
            p.cyl((x0, y0, z), (x1, y1, z), 0.04, y, 4, cap=False)
            if k < n:
                if k % 2:
                    p.cyl((x0, y0, z), (x1, y1, z + panel), 0.034, y, 4, cap=False)
                else:
                    p.cyl((x1, y1, z), (x0, y0, z + panel), 0.034, y, 4, cap=False)
        if k % 2 == 0 and 0 < k < n:
            for cx, cy in chords:                       # bolted section joints
                p.box((cx - 0.11, cy - 0.11, z - 0.06), (cx + 0.11, cy + 0.11, z + 0.06), grey)
        if k % 4 == 0 and 0 < k < n:
            p.box((-0.72, -0.72, z - 0.03), (0.72, -0.1, z), grate)
            p.box((-0.72, 0.3, z - 0.03), (0.72, 0.72, z), grate)
    # the ladder up the inside of one face, with its hoop cage
    lx = -0.62
    for sy_ in (-0.2, 0.2):
        p.box((lx - 0.02, sy_ - 0.01, 0.75), (lx + 0.02, sy_ + 0.01, H), y)
    zr = 1.0
    while zr < H - 0.2:
        p.cyl((lx, -0.2, zr), (lx, 0.2, zr), 0.012, y, 4, cap=False)
        zr += 0.3
    zr = 3.0
    while zr < H - 0.5:
        hoop = [(lx + 0.02 + 0.33 * math.cos(a) * 0 + 0.35 * (math.cos(a)), 0.33 * math.sin(a), zr) for a in np.linspace(-math.pi / 2, math.pi / 2, 7)]
        for a_, b_ in zip(hoop[:-1], hoop[1:]):
            p.cyl(a_, b_, 0.012, y, 3, cap=False)
        zr += 1.2
    for a in np.linspace(-math.pi / 3, math.pi / 3, 3):
        x_, y_ = lx + 0.02 + 0.35 * math.cos(a), 0.33 * math.sin(a)
        p.cyl((x_, y_, 3.0), (x_, y_, H - 0.5), 0.01, y, 3, cap=False)
    # site electrics cabinet and its cable to the mast, a dirt apron round the pad
    p.r = 0.01
    p.box((3.4, -1.0, 0.0), (4.0, -0.6, 1.4), M["grey"])
    p.r = 0.0
    p.tube([(3.4, -0.8, 0.2), (3.3, -0.8, 0.03), (2.9, -0.6, 0.31), (0.9, -0.85, 0.31), (0.85, -0.85, 1.5)], 0.02, M["black"], 6, bend=0.2)
    p.face([(-4.5, -4.5, 0.02), (4.5, -4.5, 0.02), (4.5, 4.5, 0.02), (-4.5, 4.5, 0.02)], ST.scuff_ring(), [(0, 0), (1, 0), (1, 1), (0, 1)])
    # the compound round the base: Heras panels on rubber feet, a gap left for the ladder gate
    _fence_ring(p, 5.25, ST)
    # slewing ring, turntable, tower head
    p.r = 0.01
    p.cyl((0, 0, H), (0, 0, H + 0.35), 1.15, grey, 32)
    p.box((-1.4, -1.4, H + 0.35), (1.4, 1.4, H + 0.8), y)
    zj = H + 0.95
    p.box((-1.0, -1.2, H + 0.8), (1.0, 1.2, zj), y)
    p.r = 0.0
    # jib (+y): two bottom chords, a top chord, lacing on three faces, tapered tip
    J, hj, bw = 50.0, 1.3, 0.6
    top_z = lambda s_: zj + hj if s_ <= J - 4 else zj + hj * max(0.08, (J - s_) / 4)
    for sx in (-1, 1):
        p.box((sx * bw - 0.08, 1.0, zj - 0.18), (sx * bw + 0.08, J, zj), y)
    p.cyl((0, 1.0, zj + hj), (0, J - 4, zj + hj), 0.085, y, 8, cap=False)
    p.cyl((0, J - 4, zj + hj), (0, J, zj + 0.1), 0.085, y, 8, cap=False)
    # walkway between the bottom chords with a handrail rope, hoist and trolley ropes along the jib
    p.box((-0.3, 1.2, zj - 0.06), (0.3, J - 1.0, zj - 0.03), grate)
    for k in range(int((J - 2) / 4)):
        yy_ = 2.0 + k * 4.0
        p.cyl((bw, yy_, zj), (bw, yy_, zj + 1.0), 0.015, y, 6, cap=False)
    p.cyl((bw, 2.0, zj + 1.0), (bw, J - 2.0, zj + 1.0), 0.008, M["steel"], 4, cap=False)
    for dx in (-0.12, 0.12):
        p.cyl((dx, 0.8, zj - 0.12), (dx, J - 0.3, zj - 0.12), 0.008, M["steel"], 4, cap=False)
    s0 = 1.0
    k = 0
    while s0 < J - 0.1:
        s1 = min(J, s0 + 2.0)
        for sx in (-1, 1):
            if k % 2:
                p.cyl((sx * bw, s0, zj), (0, s1, top_z(s1)), 0.032, y, 4, cap=False)
            else:
                p.cyl((0, s0, top_z(s0)), (sx * bw, s1, zj), 0.032, y, 4, cap=False)
        p.cyl((-bw, s0, zj - 0.08), (bw, s0, zj - 0.08), 0.03, y, 4, cap=False)
        p.cyl((-bw if k % 2 else bw, s0, zj - 0.08), (bw if k % 2 else -bw, s1, zj - 0.08), 0.025, y, 4, cap=False)
        s0, k = s1, k + 1
    p.box((-bw - 0.1, J - 0.05, zj - 0.2), (bw + 0.1, J + 0.08, zj + 0.15), y)
    for x_, y_, z_ in ((0, 1.0, zj), (0, 1.0, zj + hj)):
        p.cyl((-1.0, 0.0, H + 0.9), (x_, y_ + 2.0, z_), 0.04, y, 4, cap=False)
        p.cyl((1.0, 0.0, H + 0.9), (x_, y_ + 2.0, z_), 0.04, y, 4, cap=False)
    # counter-jib (-y): girders, grating deck, handrails, winch, cabinets, ballast
    C = 15.0
    p.r = 0.006
    for sx in (-1, 1):
        p.box((sx * 0.9 - 0.07, -C, zj - 0.5), (sx * 0.9 + 0.07, -1.2, zj), y)
    p.r = 0.0
    p.box((-0.83, -C, zj - 0.04), (0.83, -1.2, zj), grate)
    yy = -1.6
    while yy > -C:
        p.box((-0.9, yy - 0.05, zj - 0.45), (0.9, yy + 0.05, zj - 0.35), y)
        yy -= 1.6
    for sx in (-1, 1):
        x = sx * 0.95
        yy = -1.3
        while yy > -C + 2.6:
            p.cyl((x, yy, zj), (x, yy, zj + 1.1), 0.022, y, 6, cap=False)
            yy -= 1.6
        for z in (0.55, 1.1):
            p.cyl((x, -1.3, zj + z), (x, -C + 2.6, zj + z), 0.022, y, 6, cap=False)
        p.cyl((sx * 1.0, -2.0, zj + 0.3), (sx * 0.95, -1.3, zj + 1.1), 0.022, y, 6, cap=False)
    p.r = 0.01
    p.cyl((-0.55, -4.0, zj + 0.55), (0.55, -4.0, zj + 0.55), 0.45, grey, 24)
    p.cyl((-0.6, -4.0, zj + 0.55), (-0.55, -4.0, zj + 0.55), 0.52, y, 24)
    p.cyl((0.55, -4.0, zj + 0.55), (0.6, -4.0, zj + 0.55), 0.52, y, 24)
    p.box((-0.75, -5.1, zj), (0.75, -4.6, zj + 0.35), y)
    p.box((0.15, -5.0, zj + 0.35), (0.7, -4.6, zj + 0.95), grey)
    p.cyl((-0.15, -4.8, zj + 0.65), (0.15, -4.8, zj + 0.65), 0.28, grey, 20)
    for x in (-0.55, 0.15):
        p.box((x, -7.3, zj), (x + 0.6, -6.8, zj + 1.5), M["grey"])
    for k in range(4):
        y0 = -C + 0.15 + k * 0.64
        dx = (0.03, -0.02, 0.04, -0.01)[k]
        p.r = 0.02
        p.box((-1.2 + dx, y0, zj), (1.2 + dx, y0 + 0.56, zj + 1.95 - 0.02 * k), ballast)
        p.r = 0.01
        p.cyl((-0.5, y0 + 0.29, zj + 1.95), (-0.5, y0 + 0.29, zj + 2.02), 0.05, M["rust"], 8)
        p.cyl((0.5, y0 + 0.29, zj + 1.95), (0.5, y0 + 0.29, zj + 2.02), 0.05, M["rust"], 8)
    ban = ST.banner("PS_CraneBanner", _crane_banner)
    maker = ST.stencil("PS_Label_CraneMaker", [("ARKLOW 172 EC", 0.7, 0.12, "black")], (1024, 128), color="#1b1b1b", wear=0.2, seed=6)
    for sx in (-1, 1):
        p.decal((sx * 0.975, -11.8, zj - 0.25), (sx, 0, 0), (0, 0, 1), 3.2, 0.4, maker, lift=0.002)
    for sx in (-1, 1):
        p.decal((sx * 0.975, -7.6, zj + 0.62), (sx, 0, 0), (0, 0, 1), 5.6, 1.0, ban, lift=0.01)
    # cab on the side of the turntable, its front looking down the jib
    cx0, cx1, cy0, cy1, cz0, cz1 = -2.9, -1.4, 0.0, 2.0, H + 0.25, H + 2.45
    p.r = 0.01
    p.box((cx0, cy0, cz0), (cx1, cy1, cz0 + 0.12), y)
    p.box((cx0 - 0.03, cy0, cz1 - 0.12), (cx1, cy1 + 0.06, cz1), y)
    p.box((cx0, cy0, cz0), (cx1, cy0 + 0.08, cz1), y)
    for xx, yy_ in ((cx0, cy1), (cx1, cy1), (cx0, (cy0 + cy1) / 2)):
        p.box((xx - 0.03, yy_ - 0.06, cz0), (xx + 0.03, yy_, cz1), y)
    for sx_, xx in ((-1, cx0), (1, cx1)):
        p.box((xx - 0.02, cy0, cz0), (xx + 0.02, cy1, cz0 + 0.95), y)
    p.r = 0.0
    tint = ST.tinted_glass()
    p.face([(cx0, cy1, cz0 + 0.12), (cx1, cy1, cz0 + 0.12), (cx1, cy1, cz1 - 0.12), (cx0, cy1, cz1 - 0.12)], tint)
    p.face([(cx0, cy0 + 0.08, cz0 + 0.95), (cx0, cy1, cz0 + 0.95), (cx0, cy1, cz1 - 0.12), (cx0, cy0 + 0.08, cz1 - 0.12)], tint)
    p.face([(cx1, cy1, cz0 + 0.95), (cx1, cy0 + 0.08, cz0 + 0.95), (cx1, cy0 + 0.08, cz1 - 0.12), (cx1, cy1, cz1 - 0.12)], tint)
    p.face([(cx1, cy1, cz0 + 0.12), (cx0, cy1, cz0 + 0.12), (cx0, cy1 + 0.25, cz0 + 0.02), (cx1, cy1 + 0.25, cz0 + 0.02)], tint)
    p.r = 0.004
    p.box((cx0 - 0.02, cy0 + 0.15, cz0 + 0.15), (cx0 + 0.005, cy0 + 0.85, cz0 + 0.93), M["white"])
    p.box((cx0 - 0.04, cy0 + 0.72, cz0 + 0.5), (cx0 - 0.02, cy0 + 0.8, cz0 + 0.54), M["black"])
    p.r = 0.0
    p.face([(cx0, cy1 + 0.01, cz1 - 0.08), (cx1, cy1 + 0.01, cz1 - 0.08), (cx1, cy1 + 0.35, cz1 - 0.02), (cx0, cy1 + 0.35, cz1 - 0.02)], y)
    p.box((cx0 + 0.45, 0.5, cz0 + 0.12), (cx1 - 0.45, 1.0, cz0 + 0.6), M["black"])
    p.box((cx0 + 0.45, 0.45, cz0 + 0.6), (cx1 - 0.45, 0.6, cz0 + 1.2), M["black"])
    p.box((cx0 + 0.1, 1.2, cz0 + 0.12), (cx0 + 0.4, 1.9, cz0 + 0.9), M["grey"])
    p.box((cx0 + 0.6, cy1 + 0.01, cz0 + 0.9), (cx0 + 0.62, cy1 + 0.02, cz0 + 1.6), M["black"])
    p.box((cx1, 0.2, H + 0.5), (-1.0, 1.0, H + 0.6), grate)
    # trolley, ropes and hook block
    ty, zh = J * 0.55, 14.0
    p.r = 0.008
    p.box((-0.75, ty - 0.7, zj - 0.75), (0.75, ty + 0.7, zj - 0.5), y)
    for sx in (-1, 1):
        p.box((sx * 0.6 - 0.03, ty - 0.6, zj - 0.5), (sx * 0.6 + 0.03, ty + 0.6, zj - 0.17), y)
        for dy in (-0.45, 0.45):
            p.cyl((sx * 0.68, ty + dy, zj - 0.25), (sx * 0.75, ty + dy, zj - 0.25), 0.12, grey, 16)
    for dy in (-0.25, 0.25):
        p.cyl((-0.2, ty + dy, zj - 0.85), (0.2, ty + dy, zj - 0.85), 0.2, grey, 16)
    p.r = 0.0
    for dx in (-0.12, 0.12):
        for dy in (-0.25, 0.25):
            p.cyl((dx, ty + dy, zj - 0.9), (dx, ty + dy * 0.8, zh + 1.05), 0.018, M["steel"], 6, cap=False)
    p.r = 0.01
    for sx in (-1, 1):
        p.box((sx * 0.2 - 0.03, ty - 0.35, zh + 0.25), (sx * 0.2 + 0.03, ty + 0.35, zh + 1.15), y)
    p.cyl((0, ty, zh + 0.25), (0, ty, zh + 0.05), 0.22, M["black"], 16)
    for dy in (-0.15, 0.15):
        p.cyl((-0.14, ty + dy, zh + 0.85), (0.14, ty + dy, zh + 0.85), 0.2, grey, 20)
    p.box((-0.2, ty - 0.12, zh + 0.1), (0.2, ty + 0.12, zh + 0.28), grey)
    p.cyl((0, ty, zh + 0.1), (0, ty, zh - 0.08), 0.05, grey, 12)
    stripes = ST.sign_panel("PS_HookStripes", lambda im: __import__("kit.assets.site_paint", fromlist=["x"]).hazard_stripes(im, 0, 0, im.shape[1], im.shape[0], pitch=40),
                            (128, 128), grime=0.2)
    for sx in (-1, 1):
        p.decal((sx * 0.23, ty, zh + 0.7), (sx, 0, 0), (0, 0, 1), 0.68, 0.88, stripes, lift=0.002)
    # load-rating boards hung on the jib
    for k, (yy, txt) in enumerate(((20.0, "5.6 t"), (32.0, "3.8 t"), (44.0, "2.4 t"))):
        brd = ST.stencil(f"PS_Label_Rating{k}", [(txt, 0.55, 0.2, "black"), (f"{int(yy)} m", 0.25, 0.7, "bold")], (256, 192),
                         color="#111111", align="c", wear=0.0, bg="#f2f2ee")
        p.decal((0.0, yy, zj + 0.55), (1, 0, 0), (0, 0, 1), 1.4, 1.0, brd, lift=0.05)
        p.decal((0.0, yy, zj + 0.55), (-1, 0, 0), (0, 0, 1), 1.4, 1.0, brd, lift=0.05)
    p.r = 0.0
    p.tube([(0, ty, zh - 0.08), (0, ty, zh - 0.35), (0.0, ty + 0.2, zh - 0.62), (0, ty + 0.0, zh - 0.82), (0, ty - 0.2, zh - 0.62),
            (0, ty - 0.18, zh - 0.44)], 0.06, M["red"], 12, bend=0.16)
    p.box((-0.015, ty - 0.18, zh - 0.44), (0.015, ty + 0.0, zh - 0.36), M["yellow"])
    # anemometer and aviation lights
    xa, ya = 1.0, -C + 0.4
    p.cyl((xa, ya, zj + 1.95), (xa, ya, zj + 3.2), 0.025, grey, 8)
    for k in range(3):
        an = 2 * math.pi * k / 3
        tip = (xa + 0.18 * math.cos(an), ya + 0.18 * math.sin(an), zj + 3.2)
        p.cyl((xa, ya, zj + 3.2), tip, 0.006, grey, 4)
        p.sphere(tip, 0.04, M["black"], nu=8, nv=4)
    for c in ((0, J + 0.05, zj + 0.25), (-0.6, -C + 0.4, zj + 2.2), (0, 1.2, zj + 0.25 + hj), (cx0 + 0.2, cy0 + 0.2, cz1 + 0.1)):
        p.cyl((c[0], c[1], c[2] - 0.35), (c[0], c[1], c[2] - 0.05), 0.02, grey, 6)
        p.sphere(c, 0.08, red, nu=10, nv=6)
    _auto_uv(p)
    return p.obj("tower_crane", col, 30)


def _fence_ring(p, h, ST):
    """A light Heras compound (square, half-size h) with a 3 m gap on the +y side: galvanised frame
    tubes with an alpha welded-mesh card and rubber feet (the cheap far-view version)."""
    galv = ST.galvanised_img()
    mesh = ST.weld_mesh("PS_HerasMeshCard", cell_u=0.075, cell_v=0.225, wire=0.005)
    foot = ST.recycled_rubber()
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    for i in range(4):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % 4])
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        n = int(round(L / 3.5))
        for k in range(n):
            p0, p1 = a + d * (k * L / n + 0.03), a + d * ((k + 1) * L / n - 0.03)
            if i == 2 and abs((p0 + p1)[0] / 2) < 2.0:
                continue
            q0, q1 = (p0[0], p0[1]), (p1[0], p1[1])
            p.tube([(q0[0], q0[1], 0.07), (q0[0], q0[1], 2.0), (q1[0], q1[1], 2.0), (q1[0], q1[1], 0.07)], 0.019, galv, 6, bend=0.1)
            p.tube([(q0[0], q0[1], 0.2), (q1[0], q1[1], 0.2)], 0.013, galv, 6)
            seg = float(np.linalg.norm(p1 - p0))
            p.face([(q0[0], q0[1], 0.2), (q1[0], q1[1], 0.2), (q1[0], q1[1], 1.98), (q0[0], q0[1], 1.98)], mesh,
                   [(0, 0), (seg, 0), (seg, 1.78), (0, 1.78)])
            for q in (q0, q1):
                p.box((q[0] - 0.35 * abs(d[0]) - 0.1 * abs(d[1]), q[1] - 0.35 * abs(d[1]) - 0.1 * abs(d[0]), 0.0),
                      (q[0] + 0.35 * abs(d[0]) + 0.1 * abs(d[1]), q[1] + 0.35 * abs(d[1]) + 0.1 * abs(d[0]), 0.14), foot)


def _crane_banner(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    img[...] = SP.lin("#123a5e")
    SP.put(img, SP.rect_mask(H, W, 0, H * 0.78, W, H), SP.lin("#f2c200"))
    SP.text(img, "KERR & MASON", W * 0.04, H * 0.2, H * 0.36, SP.lin("#ffffff"), "black", max_w=W * 0.5)
    SP.text(img, "BUILDING THE FUTURE", W * 0.96, H * 0.3, H * 0.18, SP.lin("#f2c200"), "bold", "r", max_w=W * 0.36)
    SP.text(img, "kerrmason.example   01632 960190", W / 2, H * 0.83, H * 0.1, SP.lin("#123a5e"), "bold", "c", max_w=W * 0.9)


def heras_panel(col, M):
    """A round-top temporary fence panel, 3.5 x 2.0 m: a 38 mm galvanised tube frame, anti-climb
    welded mesh (wires every 75 mm across, 300 mm up) cut out with alpha, its legs dropped into
    moulded recycled-rubber feet (700 x 200 x 140 mm, two holes, ribbed, chamfered: the next
    panel's leg takes the other hole), a two-piece coupler clamp at the top and foot of the joint
    and mud splashed up the bottom rail. Runs along x."""
    from . import site_textures as ST
    p = PM(0.0)
    galv = ST.galvanised_img()
    foot = ST.recycled_rubber()
    xl, H, zb = 1.71, 2.0, 0.2
    rr = 0.019
    frame = [(-xl, 0, 0.07), (-xl, 0, H), (xl, 0, H), (xl, 0, 0.07)]
    p.tube(frame, rr, galv, 12, bend=0.12)
    p.tube([(-xl, 0, zb), (xl, 0, zb)], 0.0127, galv, 10)
    p.tube([(0, 0, zb), (0, 0, H)], 0.0127, galv, 10)
    # the mesh: 4 mm wires every 75 mm across and 225 mm up, one soft-alpha card (blended, so it
    # neither breaks up close nor vanishes far off: its mipmaps fade to the mesh's grey haze)
    mesh = ST.weld_mesh("PS_HerasMeshSoft", cell_u=0.075, cell_v=0.225, wire=0.0042, px=(96, 288), blend=True)
    R = 0.12
    out = [np.array(q) for q in round_path([(-xl, 0, zb), (-xl, 0, H), (xl, 0, H), (xl, 0, zb)], R)]
    poly = [(max(-xl + 0.012, min(xl - 0.012, q[0])), 0.0, min(q[2], H - 0.012)) for q in out]
    p.face(poly, mesh, [(q[0] + xl - 0.0375, q[2] - zb) for q in poly])
    # couplers: two half-shells round the leg and the next panel's, one M12 bolt
    p.r = 0.002
    for z in (0.45, 1.75):
        for dx in (0.0, 0.08):
            p.cyl((xl + dx, 0, z - 0.04), (xl + dx, 0, z + 0.04), 0.027, galv, 12)
        p.box((xl, -0.012, z - 0.035), (xl + 0.08, 0.012, z + 0.035), galv)
        for dz in (-0.022, 0.022):
            p.cyl((xl + 0.04, -0.03, z + dz), (xl + 0.04, 0.03, z + dz), 0.006, M["steel"], 8)
            p.cyl((xl + 0.04, 0.03, z + dz), (xl + 0.04, 0.042, z + dz), 0.011, M["steel"], 6)
    # feet: chamfered rubber blocks centred on the joints, two leg holes, a ribbed top
    p.r = 0.01
    for x in (-xl - 0.04, xl + 0.04):
        p.loft([rrect(0.7, 0.2, 0.03, z, cx=x) for z in (0.0, 0.11)] + [rrect(0.66, 0.16, 0.02, 0.14, cx=x)], foot)
        p.r = 0.0
        for k in range(6):
            rx = x - 0.28 + k * 0.112
            if abs(rx - x) < 0.1:
                continue
            p.box((rx - 0.012, -0.075, 0.14), (rx + 0.012, 0.075, 0.152), foot)
        for hx in (x - 0.04, x + 0.04):
            p.decal((hx, 0, 0.1405), (0, 0, 1), (0, 1, 0), 0.05, 0.05, ST.sign_panel("PS_FootHole", _hole, (32, 32), grime=0.0), lift=0.001)
        mark = ST.stencil("PS_Label_FootMark", [("SITEGUARD  RECYCLED", 0.5, 0.25, "black")], (512, 64), color="#3a3b3c", wear=0.1)
        for sy in (-1, 1):
            p.decal((x, sy * 0.1, 0.06), (0, sy, 0), (0, 0, 1), 0.4, 0.05, mark, lift=0.0015)
        p.r = 0.01
    _auto_uv(p, roof_v0=2.0)
    return p.obj("heras_panel", col, 30)


def _hole(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    img[...] = SP.lin("#1e1f20")
    SP.put(img, SP.poly_mask(H, W, SP.circle_pts(W / 2, H / 2, W * 0.42, 24)), SP.lin("#050505"))


def hoarding_panel(col, M):
    """One 2.44 m module of site hoarding (tiles along x): two 1.22 x 2.44 m sheets of 18 mm ply,
    their faces painted in the club colour (TINT_hoarding) and printed with the scheme's graphic,
    the backs bare and stencilled, screwed to three 75 x 50 rails on 75 x 75 posts at 1.22 m
    centres, a capping rail, and one raking strut to a precast ballast block with a lifting eye,
    bolted through a steel shoe. Front +y; braces behind."""
    from . import site_textures as ST
    p = PM(0.0)
    tint = ST.hoarding_shade()
    prt = ST.hoarding_print()
    back = ST.plywood_back()
    tv = ST.timber("PS_TimberV", "#8a7457", seed=1, along="v")
    th = ST.timber("PS_TimberH", "#8a7457", seed=2, along="u")
    blk = ST.concrete_block("PS_LegoBlock", "#88837a", cell=1.0, seed=221, stains=0.7, chips=True)
    t, Hh = 0.018, 2.44
    for x0, x1 in ((-1.22, -0.0015), (0.0015, 1.22)):
        uv = lambda xs, zs: [(1.22 - x, z) for x, z in zip(xs, zs)]
        p.face([(x1, 0, 0), (x0, 0, 0), (x0, 0, Hh), (x1, 0, Hh)], tint, uv([x1, x0, x0, x1], [0, 0, Hh, Hh]))
        p.face([(x1, 0.0025, 0), (x0, 0.0025, 0), (x0, 0.0025, Hh), (x1, 0.0025, Hh)], prt, uv([x1, x0, x0, x1], [0, 0, Hh, Hh]))
        p.face([(x0, -t, 0), (x1, -t, 0), (x1, -t, Hh), (x0, -t, Hh)], back)
        p.face([(x0, 0, 0), (x0, -t, 0), (x0, -t, Hh), (x0, 0, Hh)], back)
        p.face([(x1, -t, 0), (x1, 0, 0), (x1, 0, Hh), (x1, -t, Hh)], back)
        p.face([(x0, -t, Hh), (x1, -t, Hh), (x1, 0, Hh), (x0, 0, Hh)], back)
        p.face([(x0, 0, 0), (x1, 0, 0), (x1, -t, 0), (x0, -t, 0)], back)
    p.r = 0.004
    p.box((-1.22, -0.12, Hh), (1.22, 0.012, Hh + 0.045), tint)
    # studwork: 100 x 50 studs at 600 mm centres (end studs flush so modules butt up), a sole
    # plate, and three 75 x 50 rails behind the studs
    for x in (-1.195, -0.61, 0.0, 0.61, 1.195):
        p.box((x - 0.025, -t - 0.1, 0.05), (x + 0.025, -t, Hh), tv)
    p.box((-1.22, -t - 0.1, 0.0), (1.22, -t, 0.05), th)
    for z in (0.3, 1.2, 2.1):
        p.box((-1.22, -t - 0.15, z), (1.22, -t - 0.1, z + 0.075), th)
    galv = ST.galvanised_img()
    # a kicker board along the foot of the studs, just off the ground
    p.box((-1.22, -t - 0.125, 0.0), (1.22, -t - 0.1, 0.15), th)
    # the raking prop (100 x 75 at about 50 degrees) to an interlocking concrete block
    for bx in (-0.61, 0.61):
        yb0, yb1 = -1.9, -1.5
        p.r = 0.012
        p.box((bx - 0.4, yb0, 0.0), (bx + 0.4, yb1, 0.4), blk)
        p.r = 0.006
        for sx in (-0.2, 0.2):
            p.cyl((bx + sx, (yb0 + yb1) / 2, 0.4), (bx + sx, (yb0 + yb1) / 2, 0.47), 0.09, blk, 16)
        top = np.array((bx, -t - 0.15 - 0.04, 1.85))
        foot = np.array((bx, -1.46, 0.44))
        d = top - foot
        L = float(np.linalg.norm(d))
        p.r = 0.004
        p.obox(tuple((top + foot) / 2), (0.075, 0.1, L), tv, pitch=-math.atan2(d[1], d[2]))
        p.r = 0.0
        # steel angle shoe bolted to the block face and the prop
        p.box((bx - 0.06, -1.5, 0.22), (bx + 0.06, -1.494, 0.42), galv)
        p.box((bx - 0.06, -1.5, 0.4), (bx + 0.06, -1.36, 0.406), galv)
        for sx in (-1, 1):
            p.obox((bx + sx * 0.043, -1.43, 0.5), (0.006, 0.18, 0.2), galv, pitch=-math.atan2(d[1], d[2]))
        for zz in (0.27, 0.36):
            p.cyl((bx, -1.494, zz), (bx, -1.48, zz), 0.012, M["steel"], 6)
        for z_, y_ in ((0.52, -1.42), (1.8, -0.21)):
            p.cyl((bx - 0.06, y_, z_), (bx + 0.06, y_, z_), 0.007, M["steel"], 6)
            p.cyl((bx + 0.045, y_, z_), (bx + 0.058, y_, z_), 0.014, M["steel"], 6)
    _auto_uv(p)
    return p.obj("hoarding_panel", col, 30)


def _pallet(p, wood, cx, cy, z, yaw=0.0, stamp=None, woods=None, dark=None):
    """A 1200 x 1000 mm block pallet (top z + 0.144): seven top boards with gaps, three cross
    boards, nine blocks, three bottom boards (darker, they sit in the dirt); boards in a few
    shades; nail heads along the deck; the IPPC mark on a corner block."""
    q = PM(0.002)
    woods = woods or [wood]
    dark = dark or wood
    rng = np.random.default_rng(abs(int(cx * 100)) + 7)
    for k in range(3):
        y = -0.45 + k * 0.45
        q.box((-0.6, y - 0.05, 0.0), (0.6, y + 0.05, 0.022), dark)
        for x in (-0.55, 0.0, 0.55):
            q.box((x - 0.05, y - 0.07, 0.022), (x + 0.05, y + 0.07, 0.1), woods[rng.integers(0, len(woods))])
    for x in (-0.55, 0.0, 0.55):
        q.box((x - 0.05, -0.5, 0.1), (x + 0.05, 0.5, 0.122), wood)
    for k in range(7):
        y = -0.5 + 0.05 + k * (0.9 / 6)
        w = 0.1 if k in (0, 6) else 0.08
        q.box((-0.6, y - w / 2, 0.122), (0.6, y + w / 2, 0.144), woods[rng.integers(0, len(woods))])
        q.r = 0.0
        for x in (-0.55, 0.0, 0.55):
            for dy in (-0.022, 0.022):
                q.cyl((x + rng.uniform(-0.01, 0.01), y + dy, 0.1441), (x, y + dy, 0.1449), 0.0045, ST_nail(), 6)
        q.r = 0.002
    if stamp is not None:
        q.decal((0.55, -0.52 - 0.0005, 0.061), (0, -1, 0), (0, 0, 1), 0.09, 0.06, stamp, lift=0.0)
    _auto_uv(q)
    _merge(p, q, Matrix.Translation(Vector((cx, cy, z))) @ Matrix.Rotation(yaw, 4, "Z"))


def pallet_stack(col, M):
    """A builders' merchant drop: dense concrete blocks (440 x 215 x 100) eight courses high on a
    pallet, strapped both ways, each block a little different and a few chipped; beside it,
    turned off square, a one-tonne woven bulk bag of sharp sand on its own pallet, open at the
    top with its four lifting loops up, slumped and bulging, a safety label stitched on."""
    from . import site_textures as ST
    p = PM(0.0)
    wood = ST.timber("PS_PalletWood2", "#a58c69", cell=0.6, seed=3, along="u", weathered=0.75)
    woods = [wood, ST.timber("PS_PalletWoodB2", "#ab9470", cell=0.6, seed=4, along="u", weathered=0.6),
             ST.timber("PS_PalletWoodC2", "#968061", cell=0.6, seed=5, along="u", weathered=0.9)]
    dark = ST.timber("PS_PalletWoodD", "#6e6150", cell=0.6, seed=6, along="u", weathered=0.9)
    stamp = ST.sign_panel("PS_Label_PalletStamp", _ippc, (96, 64), grime=0.2)
    _pallet(p, wood, -0.68, 0.0, 0.0, 0.0, stamp, woods, dark)
    blocks = [ST.concrete_block(f"PS_DenseBlock{k}", c, cell=0.5, seed=250 + k, stains=0.1, chips=False) for k, c in
              enumerate(("#858279", "#8e8b83", "#7b7871", "#938f86", "#77746c"))]
    rng = np.random.default_rng(11)
    z0 = 0.144
    p.r = 0.004
    for layer in range(8):
        p.r = 0.004 if layer == 7 else 0.0
        z = z0 + layer * 0.101
        for i in range(5):
            for j in range(2):
                x = -0.68 - 0.5375 + 0.215 * (i + 0.5) + rng.uniform(-0.006, 0.006)
                y = -0.225 + j * 0.45 + rng.uniform(-0.008, 0.008)
                lo = [x - 0.1065, y - 0.219, z]
                hi = [x + 0.1065, y + 0.219, z + 0.1]
                if layer == 7 and rng.random() < 0.3:          # a chipped top corner
                    hi[2] -= 0.015
                p.box(tuple(lo), tuple(hi), blocks[rng.integers(0, len(blocks))])
    # banding straps: two each way over the top and down to the pallet
    strap = ST.flat_steel("PS_StrapBlack", "#1b1c1d", cell=1.0, seed=260, dirt=0.1, rough=0.4)
    p.r = 0.0
    top = z0 + 8 * 0.101
    for x in (-0.68 - 0.3, -0.68 + 0.3):
        p.box((x - 0.008, -0.452, top), (x + 0.008, 0.452, top + 0.002), strap)
        for sy in (-1, 1):
            y0, y1 = sorted((sy * 0.45, sy * 0.452))
            p.box((x - 0.008, y0, 0.11), (x + 0.008, y1, top), strap)
    for x in (-0.68 - 0.3, -0.68 + 0.3):                  # seals where the bands cross
        for y in (-0.22, 0.22):
            p.box((x - 0.018, y - 0.014, top + 0.003), (x + 0.018, y + 0.014, top + 0.011), ST.galvanised_img())
    for y in (-0.22, 0.22):
        p.box((-0.68 - 0.54, y - 0.008, top + 0.002), (-0.68 + 0.54, y + 0.008, top + 0.004), strap)
        for sx in (-1, 1):
            x0, x1 = sorted((-0.68 + sx * 0.538, -0.68 + sx * 0.541))
            p.box((x0, y - 0.008, 0.11), (x1, y + 0.008, top + 0.004), strap)
    # the bulk bag, on its own pallet, turned a few degrees
    bx, by, yaw = 0.72, 0.08, math.radians(6)
    q = PM(0.0)
    _pallet(q, wood, 0.0, 0.0, 0.0, woods=woods, dark=dark)
    _bulk_bag(q, ST, 0.144)
    _merge(p, q, Matrix.Translation(Vector((bx, by, 0.0))) @ Matrix.Rotation(yaw, 4, "Z"))
    _auto_uv(p)
    return p.obj("pallet_stack", col, 35)


def ST_nail():
    from . import site_textures as ST
    return ST.flat_steel("PS_Nail", "#6b6a66", cell=0.2, seed=301, dirt=0.3, rough=0.6, metal=0.5)


def _ippc(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    img[...] = SP.lin("#a8916c")
    k = SP.lin("#1d1d1d")
    SP.put(img, SP.rect_mask(H, W, 3, 3, W - 3, 5), k)
    SP.put(img, SP.rect_mask(H, W, 3, H - 5, W - 3, H - 3), k)
    SP.put(img, SP.rect_mask(H, W, 3, 3, 5, H - 3), k)
    SP.put(img, SP.rect_mask(H, W, W - 5, 3, W - 3, H - 3), k)
    SP.text(img, "GB-49 1772", W / 2, H * 0.18, H * 0.2, k, "bold", "c", max_w=W * 0.85)
    SP.text(img, "HT    DB", W / 2, H * 0.55, H * 0.22, k, "bold", "c", max_w=W * 0.85)


def _bulk_bag(q, ST, z0):
    """A one-tonne FIBC on a pallet: woven body slumped and bulging, corner seams, the open top
    sagging round a mound of sand, four flat lifting loops, a label."""
    pp = ST.woven_pp()
    sand = ST.sand_img()
    rng = np.random.default_rng(5)
    H = 0.86
    levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, H]
    bulge = [0.98, 1.08, 1.1, 1.06, 1.0, 0.96, 0.94]
    rings = []
    nper = 6
    base = rrect(0.9, 0.9, 0.14, n=nper)
    ph = rng.uniform(0, 6.28, 4)
    for z, s_ in zip(levels, bulge):
        ring = []
        for k, (x, y, _) in enumerate(base):
            a = math.atan2(y, x)
            w = (1 + 0.025 * math.sin(3 * a + ph[0]) * (z / H) + (0.03 * math.sin(7 * a + ph[1]) if z > 0.75 else 0.0)
                 + 0.008 * math.sin(11 * a + ph[2]) * (z / H) ** 2)
            sag = -0.07 * (abs(math.cos(2 * a)) ** 2) if z >= H - 1e-6 else 0.0      # rim droops between the loops
            ring.append((x * s_ * w, y * s_ * w, z0 + z + sag))
        rings.append(ring)
    n = len(base)
    i_body = len(q.F)
    for i in range(len(rings) - 1):
        for k in range(n):
            k2 = (k + 1) % n
            q.face([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]], pp)
    _overlay(q, i_body, len(q.F), ST.bag_dirt(), 0.002)
    # inside face down to the sand, and the sand mound
    inner = [(x * 0.985, y * 0.985, zz) for x, y, zz in rings[-1]]
    zs = z0 + 0.74
    low = [(x * 0.96, y * 0.96, zs) for x, y, _ in rings[-1]]
    for k in range(n):
        k2 = (k + 1) % n
        q.face([inner[k2], inner[k], low[k], low[k2]], pp)
    c = (0.0, 0.0, zs + 0.09)
    for k in range(n):
        k2 = (k + 1) % n
        mid1 = tuple(np.add(np.multiply(low[k], 0.5), np.multiply(c, 0.5)) + np.array((0, 0, 0.06)))
        mid2 = tuple(np.add(np.multiply(low[k2], 0.5), np.multiply(c, 0.5)) + np.array((0, 0, 0.06)))
        q.face([low[k], low[k2], mid2, mid1], sand)
        q.face([mid1, mid2, c], sand)
    q.face([(x, y, z0 + 0.001) for x, y, _ in rings[0]][::-1], pp)
    # corner seams and lifting loops
    q.r = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            a = math.atan2(sy, sx)
            pts = []
            for z, s_ in zip(levels, bulge):
                r = 0.45 * s_ * 1.005
                pts.append((sx * (0.45 - 0.14 + 0.14 * 0.7071) * s_ * 1.01, sy * (0.45 - 0.14 + 0.14 * 0.7071) * s_ * 1.01, z0 + z))
            q.tube(pts[:-1], 0.006, pp, 4, bend=0.05)
            cx, cy = sx * 0.36, sy * 0.36
            ux, uy = -sy * 0.7071, sx * 0.7071
            # a webbing loop standing up off the corner and flopping over outward
            o = np.array((sx, sy, 0.0)) * 0.7071
            base = np.array((sx * 0.4, sy * 0.4, z0 + H - 0.1))
            fl = rng.uniform(0.8, 1.2)
            # up out of the corner seam, then folded over outward and hanging down the side
            pts = [base + o * (0.02 * t2 + fl * 0.16 * max(0.0, t2 - 0.3)) + np.array((0, 0, 0.16 * math.sin(min(t2, 0.45) / 0.45 * math.pi / 2)
                   - fl * 0.5 * max(0.0, t2 - 0.45) ** 1.6)) for t2 in np.linspace(0, 1, 12)]
            _ribbon(q, [tuple(v) for v in pts], [(ux, uy, 0)] * len(pts), 0.07, ST.woven_pp("PS_Webbing", color="#d8d2bf", cell=0.5))
    lbl = ST.sign_panel("PS_Label_BulkBag", _bag_label, (256, 320), grime=0.25)
    q.decal((0.0, -0.45 * 1.08 - 0.004, z0 + 0.4), (0, -1, 0), (0, 0, 1), 0.3, 0.375, lbl, lift=0.0)
    # sand spilt on the deck round the foot of the bag, dirt splashed up the bag's lower third
    for k in range(6):
        a = rng.uniform(0, 6.28)
        r = 0.5 + rng.uniform(0.0, 0.06)
        cx, cy = r * math.cos(a), r * math.sin(a) * 0.95
        if abs(cx) < 0.58 and abs(cy) < 0.48:
            q.sphere((cx, cy, z0), rng.uniform(0.04, 0.08), sand, nu=10, nv=5, scale=(1.4, 1.0, 0.25), z_min=0.0)


def _bag_label(img):
    from . import site_paint as SP
    H, W = img.shape[:2]
    img[...] = SP.lin("#f2f1ec")
    k = SP.lin("#1f3f73")
    SP.put(img, SP.rect_mask(H, W, 0, 0, W, H * 0.2), k)
    SP.text(img, "SHARP SAND", W / 2, H * 0.06, H * 0.08, SP.lin("#ffffff"), "black", "c", max_w=W * 0.9)
    for i, s in enumerate(["FIBC TYPE A", "SWL 1000 KG", "SF 5:1", "DO NOT STAND", "UNDER LOAD"]):
        SP.text(img, s, W / 2, H * (0.27 + i * 0.1), H * 0.05, SP.lin("#222222"), "bold", "c", max_w=W * 0.9)
    SP.sign_warning(img, W / 2, H * 0.84, H * 0.09)


def _pipe(p, x, z, y0, y1, R, t, socket_end, mats, rng):
    """A uPVC pipe along y from y0 to y1, hollow (dark bore) with a socket bell at one end and a
    rubber seal in its mouth, the ink line along its top."""
    pvc, bore, seal, ink = mats
    n = 20
    def ring(y, r):
        return [(x + r * math.cos(2 * math.pi * k / n), y, z + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    def band(A, B, mat, flip=False):
        for k in range(n):
            k2 = (k + 1) % n
            q = [A[k], A[k2], B[k2], B[k]]
            p.face(q if flip else q[::-1], mat)
    Rs = R * 1.11
    mids = lambda a, b: [(a + (b - a) * k / 6, R) for k in range(1, 6)]
    if socket_end > 0:
        prof = [(y0, R), *mids(y0, y1 - 0.15), (y1 - 0.15, R), (y1 - 0.13, Rs), (y1, Rs)]
        ends = ((y0, R, -1), (y1, Rs, 1))
    else:
        prof = [(y0, Rs), (y0 + 0.13, Rs), (y0 + 0.15, R), *mids(y0 + 0.15, y1), (y1, R)]
        ends = ((y0, Rs, -1), (y1, R, 1))
    rings = [ring(y, r) for y, r in prof]
    for A, B in zip(rings[:-1], rings[1:]):
        band(A, B, pvc)
    for y, r, s in ends:
        ri = r - t
        outer, inner = ring(y, r), ring(y, ri)
        band(outer, inner, pvc, flip=s < 0)
        mouth = inner
        if r > R * 1.05:
            # inside the bell: a short wall to the seal, the black seal ring stepping in to the bore
            ys = y - s * 0.035
            wall = ring(ys, ri)
            for k in range(n):
                k2 = (k + 1) % n
                q = [inner[k], inner[k2], wall[k2], wall[k]]
                p.face(q if s > 0 else q[::-1], bore, [(0, 1), (1, 1), (1, 0.92), (0, 0.92)])
            mouth = ring(ys, R - t)
            band(wall, mouth, seal, flip=s < 0)
            y = ys
        deep = ring(y - s * 0.45, R - t)
        for k in range(n):
            k2 = (k + 1) % n
            q = [mouth[k], mouth[k2], deep[k2], deep[k]]
            p.face(q if s > 0 else q[::-1], bore, [(0, 0.92 if mouth is not inner else 1), (1, 0.92 if mouth is not inner else 1), (1, 0), (0, 0)])
        p.face(deep if s > 0 else deep[::-1], bore, [(0.5, 0.0)] * n)
    # ink line along the top
    a = math.radians(70)
    c = np.array((x + R * math.cos(a), 0, z + R * math.sin(a)))
    nrm = np.array((math.cos(a), 0, math.sin(a)))
    up = np.array((-math.sin(a), 0, math.cos(a)))
    ya, yb = (y0, y1 - 0.15) if socket_end > 0 else (y0 + 0.15, y1)
    yy = ya + rng.uniform(0.05, 0.3)
    while yy + 1.0 < yb:
        p.decal(c + np.array((0, yy + 0.5, 0)), nrm, up, 1.0, 0.016, ink, lift=0.0008)
        yy += 1.0


def pipe_stack(col, M):
    """A pack of 3 m lengths of 160 mm orange uPVC drainage pipe (BS EN 1401, 4.7 mm wall),
    hex-stacked 5-4-3 on three 75 x 50 bearers with nailed chocks, two polyester bands round the
    pack; sockets alternating row by row, open bores going dark inside, seal rings in the
    sockets, the ink line along each pipe, dust on the top."""
    from . import site_textures as ST
    p = PM(0.0)
    mats = (ST.upvc(), ST.pipe_bore(), M["rubber"], ST.pipe_print())
    tv = ST.timber("PS_TimberVOld", "#8a7a62", seed=11, along="v", weathered=0.85)
    th = ST.timber("PS_TimberHOld", "#857560", seed=12, along="u", weathered=0.85)
    rng = np.random.default_rng(17)
    R, t, L = 0.08, 0.0047, 3.0
    D = 2 * R * 1.11 + 0.002
    zb = 0.1
    centres = []
    muddy = (ST.upvc("PS_uPVCMud", mud=0.7, seed=3),) + mats[1:]
    capm = ST.flat_steel("PS_PipeCap", "#b01e1e", cell=0.5, seed=291, dirt=0.2, rough=0.5)
    capped = {(1, 1), (2, 2), (0, 3)}
    for row, n in enumerate((5, 4, 3)):
        for i in range(n):
            x = (i - (n - 1) / 2) * D
            z = zb + R + row * D * 0.866
            off = rng.uniform(0.0, 0.06) * (1 if row % 2 else -1)
            Lp = L + rng.uniform(-0.05, 0.05)
            sk = 1 if row % 2 == 0 else -1
            _pipe(p, x, z, -Lp / 2 + off, Lp / 2 + off, R, t, sk, muddy if row == 0 else mats, rng)
            centres.append((x, z))
            if (row, i) in capped:                     # a push-fit cap on the spigot end
                ye = (-Lp / 2 + off) if sk > 0 else (Lp / 2 + off)
                d = -1 if sk > 0 else 1
                # a thin push-on cap: a 15 mm skirt with a lip and a slightly domed face
                p.cyl((x, ye - d * 0.015, z), (x, ye + d * 0.003, z), R + 0.003, capm, 20, cap=False)
                p.cyl((x, ye - d * 0.016, z), (x, ye - d * 0.011, z), R + 0.006, capm, 20, cap=False)
                q0 = [(x + (R + 0.003) * math.cos(2 * math.pi * k / 20), ye + d * 0.003, z + (R + 0.003) * math.sin(2 * math.pi * k / 20)) for k in range(20)]
                q1 = [(x + R * 0.6 * math.cos(2 * math.pi * k / 20), ye + d * 0.008, z + R * 0.6 * math.sin(2 * math.pi * k / 20)) for k in range(20)]
                for k in range(20):
                    k2 = (k + 1) % 20
                    f = [q0[k], q0[k2], q1[k2], q1[k]]
                    p.face(f if d > 0 else f[::-1], capm)
                p.face(q1 if d > 0 else q1[::-1], capm)
    # timber cassette frames: a bottom and a top rail, two posts, threaded rods through
    ztop = zb + R + 2 * D * 0.866 + R
    xw = 2 * D + R + 0.03
    p.r = 0.004
    for y in (-0.95, 0.95):
        p.box((-xw - 0.16, y - 0.05, 0.0), (xw + 0.16, y + 0.05, zb), th)
        p.box((-xw - 0.16, y - 0.05, ztop), (xw + 0.16, y + 0.05, ztop + 0.075), th)
        for sx in (-1, 1):
            x0, x1 = sorted((sx * xw, sx * (xw + 0.075)))
            p.box((x0, y - 0.0375, zb), (x1, y + 0.0375, ztop), tv)
        for sx in (-1, 1):
            xr = sx * (xw + 0.12)
            p.cyl((xr, y, zb - 0.01), (xr, y, ztop + 0.1), 0.006, ST.galvanised_img(), 6)
            p.cyl((xr, y, ztop + 0.075), (xr, y, ztop + 0.09), 0.012, M["steel"], 6)
    # bands round the pack, hugging the corner pipes
    strap = ST.flat_steel("PS_BandWhite", "#e9e8e2", cell=1.0, seed=290, dirt=0.25, rough=0.5)
    corners = [(-2 * D, zb + R), (2 * D, zb + R), (D, zb + R + 2 * D * 0.866), (-D, zb + R + 2 * D * 0.866)]
    Rb = R * 1.0 + 0.006
    path = []
    for i, (cx, cz) in enumerate(corners):
        pa, pb = corners[i - 1], corners[(i + 1) % 4]
        e_in = np.subtract((cx, cz), pa)
        e_out = np.subtract(pb, (cx, cz))
        n_in = np.array((e_in[1], -e_in[0])) / np.linalg.norm(e_in)
        n_out = np.array((e_out[1], -e_out[0])) / np.linalg.norm(e_out)
        a0, a1 = math.atan2(n_in[1], n_in[0]), math.atan2(n_out[1], n_out[0])
        while a1 > a0:
            a1 -= 2 * math.pi
        for k in range(5):
            a = a0 + (a1 - a0) * k / 4
            path.append((cx + Rb * math.cos(a), cz + Rb * math.sin(a)))
    p.r = 0.0
    for y in (-0.3, 0.45):
        P = [(x, y, z) for x, z in path]
        p.sweep(P, [(-0.0015, -0.0095), (0.0015, -0.0095), (0.0015, 0.0095), (-0.0015, 0.0095)], strap, closed=True, twist_ref=(0, 1, 0))
        cx, cz = path[len(path) // 2 + 2]
        p.r = 0.002
        p.box((cx - 0.012, y - 0.016, cz - 0.004), (cx + 0.012, y + 0.016, cz + 0.006), M["steel"])
        p.r = 0.0
    _auto_uv(p)
    return p.obj("pipe_stack", col, 40)


def _mound(p, mat, cx, cy, r, h, rng, n=14, rings=3):
    """A low lumpy mound of disturbed ground centred on (cx, cy)."""
    ph = rng.uniform(0, 6.28, 3)
    R = [[None] * n for _ in range(rings + 1)]
    for j in range(rings + 1):
        t = j / rings
        for k in range(n):
            a = 2 * math.pi * k / n
            rr = r * (1 - t) * (1 + 0.18 * math.sin(3 * a + ph[0]) + 0.1 * math.sin(5 * a + ph[1]))
            zz = h * math.sin(t * math.pi / 2) * (1 + 0.3 * math.sin(4 * a + ph[2])) + 0.002
            R[j][k] = (cx + rr * math.cos(a), cy + rr * math.sin(a), zz if j else -0.01)
    for j in range(rings):
        for k in range(n):
            k2 = (k + 1) % n
            p.face([R[j][k], R[j][k2], R[j + 1][k2], R[j + 1][k]], mat)


def _ribbon(p, pts, side, width, mat):
    """A flat strip of zero thickness through pts, `side` vectors giving its across direction at
    each point; UV u along (metres), v across."""
    s_ = 0.0
    prev = None
    rows = []
    for q, d in zip(pts, side):
        q, d = np.asarray(q, float), np.asarray(d, float)
        d = d / np.linalg.norm(d)
        if prev is not None:
            s_ += float(np.linalg.norm(q - prev))
        prev = q
        rows.append((q - d * width / 2, q + d * width / 2, s_))
    for (a0, a1, s0), (b0, b1, s1) in zip(rows[:-1], rows[1:]):
        p.face([a0, b0, b1, a1], mat, [(s0, 0), (s1, 0), (s1, width), (s0, width)])


def survey_peg(col, M):
    """A setting-out peg: 50 x 50 mm sawn softwood (pointed, driven in) standing 280 mm out of
    the ground with 2-3 mm arrises, its sawn top bruised by the hammer, dayglo marking paint over
    the top 70 mm with a hard edge, a galvanised survey nail, 25 mm pink flagging tape knotted
    round it with two thin tails hanging and twisting, and a low ring of disturbed soil."""
    from . import site_textures as ST
    p = PM(0.0)
    wood = ST.peg_wood()
    top_m = ST.end_grain()
    tape = ST.flag_tape()
    rng = np.random.default_rng(23)
    h, w = 0.25, 0.025
    p.r = 0.0025
    p.box((-w, -w, -0.04), (w, w, h - 0.006), wood)
    # a slightly mushroomed, hammered top
    p.r = 0.0
    n = 12
    ring = lambda sc, z, jit: [((w * sc + rng.uniform(0, jit)) * math.cos(a) / max(abs(math.cos(a)), abs(math.sin(a))) * 1.0,
                                (w * sc + rng.uniform(0, jit)) * math.sin(a) / max(abs(math.cos(a)), abs(math.sin(a))), z)
                               for a in [2 * math.pi * (k + 0.5) / n for k in range(n)]]
    rs = [ring(1.0, h - 0.006, 0.0), ring(1.04, h - 0.002, 0.0015), ring(0.97, h, 0.0005)]
    for i in range(2):
        for k in range(n):
            k2 = (k + 1) % n
            p.face([rs[i][k], rs[i][k2], rs[i + 1][k2], rs[i + 1][k]], wood)
    p.face(rs[2], top_m, [((x / w + 1) / 2, (y / w + 1) / 2) for x, y, _ in rs[2]])
    # survey nail
    p.cyl((0.005, -0.004, h - 0.002), (0.005, -0.004, h + 0.003), 0.0055, ST.galvanised_img(), 16)
    p.cyl((0.005, -0.004, h + 0.003), (0.005, -0.004, h + 0.0042), 0.0045, ST.galvanised_img(), 16)
    # flagging tape: a band pulled tight round the peg, a knot, two tails hanging and twisting
    zt = h - 0.085
    band = []
    sides = []
    for k in range(25):
        a = 2 * math.pi * k / 24 + 0.3
        c, s_ = math.cos(a), math.sin(a)
        m = max(abs(c), abs(s_))
        band.append(((w + 0.0012) * c / m, (w + 0.0012) * s_ / m, zt + 0.003 * math.sin(a * 2)))
        sides.append((0, 0, 1))
    _ribbon(p, band, sides, 0.025, tape)
    knot = np.array((w + 0.004, 0.006, zt))
    p.sphere(tuple(knot), 0.007, tape, nu=8, nv=5, scale=(0.9, 1.4, 1.2))
    for t_i, (ln, ph, dy) in enumerate(((0.13, 0.0, 0.004), (0.1, 1.7, -0.004))):
        pts, sd = [], []
        for k in range(16):
            t = k / 15
            z = zt - 0.004 - ln * t
            x = knot[0] + 0.012 * t + 0.02 * math.sin(t * 3.0 + ph) * t
            y = knot[1] + dy + 0.018 * math.sin(t * 4.2 + ph) * t
            tw = ph + t * 2.6
            pts.append((x, y, z))
            sd.append((math.cos(tw) * 0.25, math.sin(tw), 0.12))
        _ribbon(p, pts, sd, 0.025, tape)
    # a small scuffed ring of broken soil round its foot, fading into the turf
    ring = ST.scuff_ring()
    p.face([(-0.1, -0.1, 0.006), (0.1, -0.1, 0.006), (0.1, 0.1, 0.006), (-0.1, 0.1, 0.006)], ring, [(0, 0), (1, 0), (1, 1), (0, 1)])
    # chainage and level marked on one face in marker
    p.decal((0.0, -w - 0.0004, h - 0.12), (0, -1, 0), (0, 0, 1), 0.045, 0.05, ST.peg_marking(), lift=0.0)
    _auto_uv(p)
    return p.obj("survey_peg", col, 50)


def site_board(col, M):
    """The contractor's site board facing +y: a 2.4 x 1.2 m aluminium composite panel with a
    club-coloured header (TINT_club), the parties and contacts, PPE signs and a safety strip,
    screwed through two 50 x 25 battens to the faces of two 100 x 100 treated posts, coach-screw
    heads showing; weathered, splashed at the foot, the posts set in churned ground."""
    from . import site_textures as ST
    from . import site_materials
    p = PM(0.0)
    club = site_materials.build("TINT_club")
    face = ST.site_board_face()
    head = ST.stencil("PS_Label_SiteBoardHead", [("NEW TRAINING FACILITY", 0.42, 0.14, "black"), ("UNDER CONSTRUCTION", 0.22, 0.66, "bold")],
                      (1024, 128), color="#ffffff", align="c", wear=0.0)
    post = ST.treated_post("PS_PostTanalised", color="#6b6146", algae=0.35, seed=9)
    bat = ST.timber("PS_TimberH", "#8a7457", seed=2, along="u")
    back = ST.flat_steel("PS_ACMBackWhite", "#d6d8d6", cell=1.2, seed=352, dirt=0.08, rough=0.45, metal=0.0)
    trim = ST.galvanised_img("PS_AluTrim", base="#c5c8c9", mud=0.0, seed=13, rough_base=0.35, metal_base=0.8)
    z0, z1, hw = 0.9, 2.1, 1.2
    zh = z1 - 0.3
    y = 0.0
    # panel: header in the club colour, printed board under it, edges and back
    p.face([(hw, y, zh), (-hw, y, zh), (-hw, y, z1), (hw, y, z1)], club)
    p.face([(hw, y + 0.0015, zh + 0.02), (-hw, y + 0.0015, zh + 0.02), (-hw, y + 0.0015, z1 - 0.02), (hw, y + 0.0015, z1 - 0.02)], head,
           [(0, 0), (1, 0), (1, 1), (0, 1)])
    p.face([(hw, y, z0), (-hw, y, z0), (-hw, y, zh), (hw, y, zh)], face, [(0, 0), (1, 0), (1, 1), (0, 1)])
    t = 0.004
    p.face([(-hw, y - t, z0), (hw, y - t, z0), (hw, y - t, z1), (-hw, y - t, z1)], back)
    # 20 mm aluminium edge trim wrapping the panel edge
    p.r = 0.002
    for a, b in (((-hw - 0.006, -t - 0.004, z0 - 0.006), (hw + 0.006, 0.004, z0 + 0.014)),
                 ((-hw - 0.006, -t - 0.004, z1 - 0.014), (hw + 0.006, 0.004, z1 + 0.006)),
                 ((-hw - 0.006, -t - 0.004, z0), (-hw + 0.014, 0.004, z1)), ((hw - 0.014, -t - 0.004, z0), (hw + 0.006, 0.004, z1))):
        p.box(a, b, trim)
    lbl = ST.sign_panel("PS_Label_BoardMaker", lambda im: (SP_fill(im, "#f0efe8"), SP_text(im, "SIGNCRAFT UK", 0.5, 0.18, 0.22, "#1f3f73"),
                                                        SP_text(im, "ACM 4mm  Ref SB-2219", 0.5, 0.6, 0.15, "#333333")), (256, 96), grime=0.2)
    p.decal((0.4, y - t, 1.5), (0, -1, 0), (0, 0, 1), 0.22, 0.08, lbl, lift=0.001)
    # battens and posts behind, coach screws through the face
    p.r = 0.003
    for z in (z0 + 0.15, z1 - 0.15):
        p.box((-hw + 0.05, -0.029, z - 0.025), (hw - 0.05, -t, z + 0.025), bat)
    # dome-headed screw caps at the corners of the face (clear of the print)
    for sx in (-1, 1):
        for zc in (z0 + 0.035, z1 - 0.035):
            p.sphere((sx * (hw - 0.035), 0.0015, zc), 0.009, M["white"], nu=10, nv=6, scale=(1, 0.45, 1), z_min=-0.05)
    # a diagonal across the back between the battens
    a_, b_ = np.array((-0.85, -0.017, z0 + 0.15)), np.array((0.85, -0.017, z1 - 0.15))
    dd = b_ - a_
    p.obox(tuple((a_ + b_) / 2), (float(np.linalg.norm(dd)), 0.024, 0.05), bat, roll=-math.atan2(dd[2], dd[0]))
    p.r = 0.006
    galv = ST.galvanised_img()
    for x in (-0.9, 0.9):
        p.box((x - 0.05, -0.129, -0.05), (x + 0.05, -0.029, z1 + 0.05), post)
        p.r = 0.0
        p.obox((x, -0.079, z1 + 0.06), (0.1, 0.1, 0.03), post, pitch=0.3)
        # a raking brace behind each post to a short stake
        top = np.array((x, -0.129, 1.3))
        top = top + np.array((0.05, -0.01, 0))
        foot = np.array((x + 0.05, -0.975, 0.24))
        d = top - foot
        p.r = 0.003
        p.obox(tuple((top + foot) / 2), (0.047, 0.075, float(np.linalg.norm(d))), bat, pitch=-math.atan2(d[1], d[2]))
        p.box((x - 0.025, -1.0, -0.05), (x + 0.025, -0.95, 0.32), post)
        p.r = 0.0
        p.cyl((x - 0.04, -0.975, 0.27), (x + 0.09, -0.975, 0.27), 0.006, M["steel"], 6)
        p.r = 0.003
        p.r = 0.0
        p.cyl((x - 0.06, -0.15, 1.27), (x + 0.06, -0.15, 1.27), 0.006, M["steel"], 6)
        # coach bolts through the battens into the post (behind the board)
        for z in (z0 + 0.15, z1 - 0.15):
            p.cyl((x, -0.029, z), (x, -0.034, z), 0.012, galv, 6)
    rng = np.random.default_rng(31)
    soil = ST.soil_fine()
    for x in (-0.9, 0.9):
        _mound(p, soil, x, -0.08, 0.26, 0.045, rng)
    _auto_uv(p)
    return p.obj("site_board", col, 35)


def _lathe_uv(p, prof, mat, sides=40, C=1.838, cx=0.0, cy=0.0, top_v0=1.0, off=0.0, rows=None, disp=None):
    """A surface of revolution about the vertical through (cx, cy) with UVs in metres: u around
    (one turn = C), v = height on the walls; near-horizontal segments map v = radius + top_v0.
    off: push it out along its normals (an overlay); rows: only these profile segments."""
    for j in range(len(prof) - 1):
        if rows is not None and j not in rows:
            continue
        (r0, z0), (r1, z1) = prof[j], prof[j + 1]
        flat = abs(z1 - z0) < 0.35 * abs(r1 - r0)
        dz, dr = z1 - z0, r1 - r0
        L = math.hypot(dz, dr) or 1.0
        nr, nz = dz / L, -dr / L
        for k in range(sides):
            a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
            def P(r, z, a):
                rr = r + nr * off - (disp(a, z, r) if disp else 0.0)
                return (cx + rr * math.cos(a), cy + rr * math.sin(a), z + nz * off)
            def UV(r, z, a):
                return (a / (2 * math.pi) * C, r + top_v0 if flat else z)
            q = [(r0, z0, a0), (r0, z0, a1), (r1, z1, a1), (r1, z1, a0)]
            p.face([P(*t) for t in q], mat, [UV(*t) for t in q])


def _cyl_patch(p, r, a_mid, width, z0, z1, mat, segs=8, cx=0.0, cy=0.0, lift=0.0015):
    """A label wrapped round a cylinder (UV 0-1), centred on angle a_mid, read from outside."""
    da = width / r
    pts = []
    for i in range(segs + 1):
        a = a_mid - da / 2 + da * i / segs
        pts.append((cx + (r + lift) * math.cos(a), cy + (r + lift) * math.sin(a), i / segs))
    for (x0, y0, u0), (x1, y1, u1) in zip(pts[:-1], pts[1:]):
        p.face([(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)], mat, [(u0, 0), (u1, 0), (u1, 1), (u0, 1)])


def oil_drum(col, M):
    """A 205 L tight-head steel drum (585 mm across, 880 mm tall): rolled chimes standing above
    the recessed lid and below the base, two rolling hoops with grooves either side, the lid's
    pressed ring, a 2 in and a 3/4 in bung with flanges and caps; paint (TINT_drum) worn through
    to rust on the chimes and hoops, runs from the top, grime round the foot, a flammable-liquid
    diamond, a torn paper label and a stencilled contents line."""
    from . import site_textures as ST
    p = PM(0.0)
    C = 2 * math.pi * 0.2925
    paint = ST.paint_detail("TINT_drum", "#2e4256", cell=C, wall_h=0.89, seed=360, foot=0.5, runs=0.6, patches=2, dents=0.004, fade=0.12)
    wear = ST.wear_soft("PS_DrumWear4", cell=C, wall_h=0.89, seed=372, chips=-1.2, foot=0.08, runs=0.9, scrapes=10, top_bleed=0.6,
                        bands=((0.006, 0.02), (0.874, 0.007), (0.305, 0.003), (0.59, 0.003)), seam=True, scrape_band=(0.25, 0.65), px=1024)
    drng = np.random.default_rng(53)
    dents = [(drng.uniform(0, 6.28), drng.uniform(0.15, 0.75), drng.uniform(0.06, 0.12), drng.uniform(0.006, 0.014)) for _ in range(3)]
    flat_a = drng.uniform(0, 6.28)
    def disp(a, z, r):
        if r < 0.27:
            return 0.0
        d = 0.0
        for da, dz, rad, dep in dents:
            ang = (a - da + math.pi) % (2 * math.pi) - math.pi
            d += dep * math.exp(-((ang * 0.29) ** 2 + (z - dz) ** 2) / (rad * rad))
        if abs((a - flat_a + math.pi) % (2 * math.pi) - math.pi) < 0.25 and abs(z - 0.305) < 0.02:
            d += max(0.0, r - 0.288)                          # a hoop flattened where it was dropped
        return d
    hoop = lambda z: [(0.287, z - 0.03), (0.284, z - 0.02), (0.289, z - 0.012), (0.297, z - 0.004), (0.297, z + 0.004),
                      (0.289, z + 0.012), (0.284, z + 0.02), (0.287, z + 0.03)]
    prof = ([(0.0, 0.022), (0.268, 0.022), (0.276, 0.008), (0.284, 0.0), (0.294, 0.004), (0.297, 0.014), (0.292, 0.026), (0.287, 0.034)]
            + hoop(0.305) + hoop(0.59)
            + [(0.287, 0.85), (0.293, 0.858), (0.297, 0.868), (0.294, 0.878), (0.284, 0.882), (0.276, 0.876), (0.272, 0.866),
               (0.268, 0.862), (0.232, 0.862), (0.226, 0.867), (0.214, 0.867), (0.208, 0.862), (0.0, 0.862)])
    _lathe_uv(p, prof, paint, 48, C, disp=disp)
    side_rows = [j for j in range(len(prof) - 1) if abs(prof[j + 1][1] - prof[j][1]) >= 0.35 * abs(prof[j + 1][0] - prof[j][0])]
    _lathe_uv(p, prof, wear, 48, C, off=0.0012, rows=side_rows, disp=disp)
    # oily stain round the bungs on the lid
    p.decal((0.0, 0.02, 0.8625), (0, 0, 1), (0, 1, 0), 0.5, 0.5, ST.lid_stain(), lift=0.0006)
    # bungs
    p.r = 0.002
    for (x, y, rf, rc) in ((0.17, -0.02, 0.038, 0.03), (-0.18, 0.07, 0.024, 0.017)):
        p.cyl((x, y, 0.861), (x, y, 0.868), rf, M["steel"], 16)
        p.cyl((x, y, 0.868), (x, y, 0.88), rc, paint, 6)
    # labels
    p.r = 0.0
    _cyl_patch(p, 0.287, math.radians(-90), 0.17, 0.62, 0.79, ST.hazard_diamond())
    _cyl_patch(p, 0.287, math.radians(-40), 0.2, 0.36, 0.56, ST.paper_label())
    stn = ST.stencil("PS_Label_DrumStencil", [("WASTE OIL", 0.6, 0.15, "black")], (512, 96), color="#e8e6df", wear=0.12)
    _cyl_patch(p, 0.287, math.radians(-150), 0.42, 0.66, 0.74, stn, segs=12)
    return p.obj("oil_drum", col, 40)


_TYRE_PROF = [  # (r, z, v): bead lip, rounded top sidewall, tread with three grooves, bottom sidewall, liner
    (0.203, 0.072, 0.0), (0.208, 0.082, 0.02), (0.226, 0.093, 0.06), (0.25, 0.101, 0.11), (0.275, 0.103, 0.16),
    (0.295, 0.098, 0.2), (0.308, 0.088, 0.235), (0.315, 0.072, 0.26), (0.317, 0.058, 0.28), (0.317, 0.04, 0.31),
    (0.309, 0.038, 0.315), (0.309, 0.024, 0.33), (0.317, 0.022, 0.335), (0.317, 0.007, 0.36), (0.309, 0.005, 0.365),
    (0.309, -0.005, 0.38), (0.317, -0.007, 0.385), (0.317, -0.022, 0.41), (0.309, -0.024, 0.415), (0.309, -0.038, 0.43),
    (0.317, -0.04, 0.435), (0.317, -0.058, 0.47), (0.315, -0.072, 0.5), (0.308, -0.088, 0.53), (0.295, -0.098, 0.57),
    (0.275, -0.103, 0.61), (0.25, -0.101, 0.66), (0.226, -0.093, 0.71), (0.208, -0.082, 0.75), (0.203, -0.072, 0.77),
    (0.215, -0.058, 0.8), (0.262, -0.08, 0.84), (0.297, -0.055, 0.89), (0.303, 0.0, 0.92), (0.297, 0.055, 0.95),
    (0.262, 0.08, 0.98), (0.215, 0.058, 0.995), (0.203, 0.072, 1.0)]


def _old_tyre(p, xf, mat, sides=34, s=1.0):
    """An old car tyre (205/55 R16) lying flat at the origin of xf, open at the bead so the dark
    inside of the casing shows."""
    prof = _TYRE_PROF[::-1]
    for j in range(len(prof) - 1):
        (r0, z0, v0), (r1, z1, v1) = prof[j], prof[j + 1]
        for k in range(sides):
            a0, a1 = 2 * math.pi * k / sides, 2 * math.pi * (k + 1) / sides
            q = [(r0, z0, a0), (r0, z0, a1), (r1, z1, a1), (r1, z1, a0)]
            pts = [xf @ Vector((r * s * math.cos(a), r * s * math.sin(a), z * s)) for r, z, a in q]
            uv = [(1 - a / (2 * math.pi), 1 - (v0 if (r, z) == (r0, z0) else v1)) for r, z, a in q]
            p.face(pts, mat, uv)
    # shoulder blocks notching the silhouette, staggered top and bottom
    nb = 36
    for k in range(nb):
        a = 2 * math.pi * (k + 0.5 * (k % 2)) / nb
        da = math.pi / nb * 0.55
        for zs in (1, -1):
            if (k + (zs > 0)) % 2:
                continue
            z0, z1 = sorted((zs * 0.044, zs * 0.07))
            V = []
            for z in (z0, z1):
                for rr in (0.309, 0.3195):
                    for aa in (a - da, a + da):
                        V.append(xf @ Vector((rr * s * math.cos(aa), rr * s * math.sin(aa), z * s)))
            F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
            for f in F:
                p.face([V[i] for i in f], mat, [(0.5 / nb + k / nb, 0.6)] * 4)


def tyre_stack(col, M):
    """Four scrap car tyres stacked a little off true and one leaning against them: lathed
    205/55 R16 sections with bulging sidewalls and rounded shoulders, open beads showing the dark
    liner, block tread and moulded sidewall lettering, dusty on top, muddy at the foot."""
    from . import site_textures as ST
    p = PM(0.0)
    tyre_m = ST.tyre_img()
    mud = ST.tyre_img("PS_TyreMuddy", seed=5, mud=True)
    rng = np.random.default_rng(29)
    z = 0.0
    for k in range(4):
        sc = rng.uniform(0.97, 1.03)
        z += 0.1 * sc
        xf = (Matrix.Translation(Vector((rng.uniform(-0.035, 0.035), rng.uniform(-0.035, 0.035), z)))
              @ Matrix.Rotation(rng.uniform(0, 6.28), 4, "Z") @ Matrix.Rotation(rng.uniform(-0.04, 0.04), 4, "X")
              @ Matrix.Rotation(rng.uniform(-0.04, 0.04), 4, "Y"))
        _old_tyre(p, xf, mud if k == 0 else tyre_m, s=sc)
        z += 0.1 * sc - 0.004
    # one stood on its tread, leaning back against the stack
    t = 0.32
    xf = (Matrix.Translation(Vector((0.05, 0.505, 0.317 * math.cos(t) + 0.103 * math.sin(t) - 0.004))) @ Matrix.Rotation(t, 4, "X")
          @ Matrix.Rotation(math.pi / 2, 4, "X"))
    _old_tyre(p, xf, mud)
    return p.obj("tyre_stack", col, 40)


def gate_farm(col, M):
    """A 15 ft galvanised five-bar field gate standing open along +y, hung on a 225 mm treated
    hanging post at the origin (1.65 m out of the ground, weathered single-bevel top): hook-and-
    band hinges on through-bolts, a 42 mm top rail over four lighter rails, stiles with end caps,
    a diagonal brace and stay, weld collars at the joints, a spring latch; the free end, dropping
    a little, chained to a short stop post; mud splashed up the lower rails and trodden ground."""
    from . import site_textures as ST
    p = PM(0.0)
    galv = ST.galvanised_img("PS_GalvMatte", rough_base=0.58, metal_base=0.55, seed=8)
    post = ST.treated_post("PS_PostGreen", color="#5c5c3e", algae=0.5, seed=4)
    L, H = 4.5, 1.1
    # hanging post with a sloped top
    p.r = 0.006
    p.box((-0.1, -0.1, -0.05), (0.1, 0.1, 1.55), post)
    p.r = 0.0
    p.extrude([(-0.1, 1.55), (0.1, 1.55), (0.1, 1.6), (-0.1, 1.715)], -0.1, 0.1, post)
    # the gate, built level then dropped a touch about its hinges
    q = PM(0.0)
    ys, ye = 0.18, L
    rails = [(0.15, 0.0135), (0.3, 0.0135), (0.5, 0.0135), (0.75, 0.0135), (H, 0.025)]
    for z, r in rails[:-1]:
        q.cyl((0, ys, z), (0, ye, z), r, galv, 10, cap=False)
    # the top rail: a 50 x 35 mm oval section
    q.sweep([(0, ys, H), (0, ye, H)], [(0.0175 * math.cos(2 * math.pi * k / 12), 0.025 * math.sin(2 * math.pi * k / 12)) for k in range(12)],
            galv, cap=False, twist_ref=(0, 0, 1))
    q.cyl((0, ys, 0.08), (0, ys, H + 0.03), 0.021, galv, 12)
    q.cyl((0, ye, 0.1), (0, ye, H + 0.02), 0.021, galv, 12)
    q.cyl((0, ys, 0.15), (0, L * 0.5, H), 0.0135, galv, 10, cap=False)
    q.cyl((0, L * 0.5, 0.15), (0, L * 0.5, H), 0.0135, galv, 10, cap=False)
    q.cyl((0, L * 0.5, 0.15), (0, ye, 0.75), 0.0135, galv, 10, cap=False)
    for y in (ys, ye):
        q.cyl((0, y, H + 0.02), (0, y, H + 0.05), 0.026, M["black"], 12)
        for z, r in rails:
            q.cyl((0, y - 0.03 if y > 1 else y, z), (0, y + (0.0 if y > 1 else 0.03), z), r * 1.35, galv, 10)
    for z, r in rails[:-1]:
        q.cyl((0, L * 0.5 - 0.015, z), (0, L * 0.5 + 0.015, z), r * 1.35, galv, 10)
    # gate eyes round the hinge pins, straps along the rails
    for z in (0.15, H):
        q.cyl((0, ys - 0.07, z - 0.035), (0, ys - 0.07, z + 0.035), 0.03, galv, 14)
        q.box((-0.01, ys - 0.07, z - 0.025), (0.01, ys + 0.26, z + 0.025), galv)
        for yb in (ys + 0.1, ys + 0.21):
            q.cyl((-0.03, yb, z), (0.03, yb, z), 0.008, M["steel"], 6)
            q.cyl((0.03, yb, z), (0.042, yb, z), 0.013, M["steel"], 6)
    # Bristol-type latch on the slam stile: a body, a sliding bolt, a knob
    q.box((-0.035, ye - 0.12, 0.86), (0.035, ye - 0.01, 0.96), galv)
    q.cyl((0, ye - 0.12, 0.91), (0, ye + 0.09, 0.91), 0.012, galv, 10)
    q.cyl((0, ye - 0.06, 0.96), (0, ye - 0.06, 1.03), 0.009, galv, 8)
    q.sphere((0, ye - 0.06, 1.045), 0.018, M["red"], nu=10, nv=6)
    _merge(p, q, Matrix.Translation(Vector((0, ys, 0))) @ Matrix.Rotation(-0.008, 4, "X") @ Matrix.Translation(Vector((0, -ys, 0))))
    # hooks: a bolt through the post, the pin the eye drops over, washer and nut behind
    p.r = 0.0
    for z in (0.15, H):
        p.cyl((0, -0.13, z + 0.045), (0, ys - 0.07, z + 0.045), 0.012, galv, 10)
        p.cyl((0, ys - 0.07, z - 0.05), (0, ys - 0.07, z + 0.055), 0.01, galv, 10)
        p.cyl((0, -0.1, z + 0.045), (0, -0.106, z + 0.045), 0.028, galv, 14)
        p.cyl((0, -0.106, z + 0.045), (0, -0.124, z + 0.045), 0.02, M["steel"], 6)
        p.cyl((0, 0.1, z + 0.045), (0, 0.108, z + 0.045), 0.028, galv, 14)
    # stop post and chain holding the gate open
    sx, sy = 0.16, L + 0.06
    p.r = 0.004
    p.box((sx - 0.075, sy - 0.075, -0.05), (sx + 0.075, sy + 0.075, 1.0), post)
    p.r = 0.0
    p.box((sx - 0.08, sy - 0.08, 0.85), (sx - 0.074, sy + 0.06, 0.97), galv)
    for zb in (0.88, 0.94):
        p.cyl((sx - 0.086, sy + 0.02, zb), (sx - 0.08, sy + 0.02, zb), 0.01, M["steel"], 6)
    links = 6
    a0 = np.array((sx - 0.05, sy, 0.7))
    a1 = np.array((0.02, L - 0.01, 0.78))
    for k in range(links):
        t0, t1 = k / links, (k + 1) / links
        c = a0 * (1 - (t0 + t1) / 2) + a1 * (t0 + t1) / 2
        c[2] -= 0.05 * math.sin(math.pi * (t0 + t1) / 2)
        d = a1 - a0
        d /= np.linalg.norm(d)
        side = np.array((0, 0, 1.0)) if k % 2 else np.cross(d, (0, 0, 1.0))
        side /= np.linalg.norm(side)
        ring = [tuple(c + d * 0.02 * math.cos(2 * math.pi * j / 8) + side * 0.012 * math.sin(2 * math.pi * j / 8)) for j in range(8)]
        p.tube(ring, 0.003, galv, 4, bend=0.0, closed=True)
    # trodden ground at the post and the stop
    rng = np.random.default_rng(37)
    soil = ST.soil_fine()
    _mound(p, soil, 0.0, 0.0, 0.3, 0.035, rng)
    _mound(p, soil, sx, sy, 0.25, 0.03, rng)
    # one soft-edged strip of trodden mud along under the gate and round the posts
    mud = ST.mud_strip()
    p.face([(-0.45, -0.3, 0.025), (0.45, -0.3, 0.025), (0.45, L + 0.4, 0.025), (-0.45, L + 0.4, 0.025)], mud,
           [(0, 0), (1, 0), (1, 1), (0, 1)])
    _auto_uv(p, roof_v0=2.0)
    return p.obj("gate_farm", col, 35)


BUILDERS = {
    "prop.site_cabin": site_cabin, "prop.portaloo": portaloo, "prop.skip": skip, "prop.tower_crane": tower_crane,
    "prop.heras_panel": heras_panel, "prop.hoarding_panel": hoarding_panel, "prop.pallet_stack": pallet_stack,
    "prop.pipe_stack": pipe_stack, "prop.survey_peg": survey_peg, "prop.site_board": site_board,
    "prop.oil_drum": oil_drum, "prop.tyre_stack": tyre_stack, "prop.gate_farm": gate_farm,
    "prop.storage_container": storage_container, "prop.container_40": container_40,
}
