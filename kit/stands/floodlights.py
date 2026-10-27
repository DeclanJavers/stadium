"""Floodlight layouts. Pure numpy.

Three installations:
  corner_pylons  four masts outside the corners (UEFA: lamps >= 25 deg above the centre spot)
  roof           a ribbon of fixtures along the roof's leading edge (modern bowls)
  poles          grassroots poles along both touchlines

Every fixture gets an aim point on the pitch. Aim points form a grid and fixtures are dealt across it
so every point is lit from several directions, which is how real lighting designs avoid hard shadows.
Lamp power is set from a target horizontal illuminance (lux) over the pitch area.
"""
import math

import numpy as np

LUX_PER_W = 683.0      # Blender light units: 1 W/m^2 ~ 683 lux
MIN_ELEVATION = 25.0   # UEFA: lamp line at >= 25 deg from the centre of the pitch
POLE_CLEAR_X = 15.0    # poles on the main-stand side stay this far from halfway

DEFAULTS = {
    "corner_pylons": {"lux": 1200, "rows": 5, "cols": 7, "mast": "lattice", "fixture": "halide", "kelvin": 4300},
    "roof": {"lux": 2000, "spacing": 2.4, "fixture": "led", "kelvin": 5700},
    "poles": {"lux": 150, "per_side": 3, "height": 16.0, "lamps": 3, "fixture": "led", "kelvin": 5000},
}


def aim_grid(L, W, nx=9, ny=5):
    xs = np.linspace(-L / 2 + 4, L / 2 - 4, nx)
    ys = np.linspace(-W / 2 + 3, W / 2 - 3, ny)
    return np.array([(x, y, 0.0) for x in xs for y in ys])


def assign(fixtures, aims, seed=0):
    """Deal aim points to fixtures: nearest-first per installation, cycling so coverage overlaps."""
    rng = np.random.default_rng(seed)
    out = []
    order = rng.permutation(len(aims))
    for k, f in enumerate(fixtures):
        out.append(aims[order[k % len(aims)]])
    return np.array(out)


def pylons(L, W, A, B, reach, roof_top, spec, corner_radius=None):
    """Four masts on the diagonals, outside the stands. reach: how far out the structures go;
    corner_radius: a curved bowl's corner radius (the mast then stands behind the corner stand)."""
    out = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        # open corners: in the gap between the stands (which never reach past A, B)
        gap = 0.5 * min(reach, 30.0)
        p = np.array([sx * (A + gap), sy * (B + gap)])
        if corner_radius is not None:
            # a bowl: behind the corner stand, clear of its back and the mast head's width
            R = corner_radius
            c = np.array([sx * (A - R), sy * (B - R)])
            q = c + np.array([sx, sy]) / math.sqrt(2) * (R + reach + 8.0)
            if np.linalg.norm(q) > np.linalg.norm(p):
                p = q
        dist = np.linalg.norm(p)
        h = max(dist * math.tan(math.radians(MIN_ELEVATION + 3)), roof_top + 8.0, 30.0)
        out.append({"base": (float(p[0]), float(p[1])), "height": float(h)})
    return out


def head_layout(mast, spec):
    """Fixture positions on a mast head facing the pitch centre."""
    rows, cols = spec["rows"], spec["cols"]
    bx, by = mast["base"]
    H = mast["height"]
    yaw = math.atan2(-by, -bx)                     # facing the centre spot
    tilt = math.atan2(H, math.hypot(bx, by))       # head plane faces down at the centre
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0, 0, 1.0]) * math.cos(tilt) + fwd * math.sin(tilt)
    centre = np.array([bx, by, H]) + fwd * 1.0
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append(centre + side * (c - (cols - 1) / 2) * 0.8 + up * (r - (rows - 1) / 2) * 0.75)
    return np.array(pts), yaw, tilt


def edge_normal(line, p):
    """Horizontal unit normal of a roof edge polyline at its point nearest p, pointing away from the
    pitch centre."""
    best = (math.inf, np.array([1.0, 0.0]))
    q = np.asarray(p, float)[:2]
    for a, b in zip(np.asarray(line)[:-1, :2], np.asarray(line)[1:, :2]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = min(max(float((q - a) @ ab) / L2, 0.0), 1.0)
        dist = float(np.linalg.norm(a + ab * t - q))
        if dist < best[0]:
            best = (dist, ab / math.sqrt(L2))
    n = np.array([best[1][1], -best[1][0]])
    return -n if n @ q < 0 else n


def roof_ribbon(edge_lines, spacing):
    pts = []
    for line in edge_lines:
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(1, int(s[-1] / spacing))
        for t in (np.arange(n) + 0.5) * s[-1] / n:
            pts.append([np.interp(t, s, line[:, i]) for i in range(3)])
    return np.array(pts)


def poles(L, W, spec, offset, clear=POLE_CLEAR_X):
    """Poles along both touchlines; the main-stand side keeps `clear` metres from halfway (past the
    main stand's ends)."""
    out = []
    n = spec["per_side"]
    xs = np.linspace(-L / 2, L / 2, n + 2)[1:-1] if n > 1 else np.array([0.0])
    for sy in (1, -1):
        for x in xs:
            # the main-stand side keeps halfway clear for the dugouts, tunnel and TV camera
            if sy < 0 and abs(x) < clear:
                x = -clear if x < 0 else clear
            out.append({"base": (float(x), float(sy * (W / 2 + offset))), "height": spec["height"]})
    return out


def lamp_power(n_fixtures, spot_deg, lux, area, efficiency=0.65):
    """Blender spot power (W) per fixture to reach `lux` on average over `area`.

    A Blender spot of power P emits P/(4 pi) W/sr inside its cone; the cone holds 2 pi (1 - cos a) sr.
    """
    a = math.radians(spot_deg / 2)
    omega = 2 * math.pi * (1 - math.cos(a))
    per = (lux / LUX_PER_W) * area / (n_fixtures * efficiency)
    return per * 4 * math.pi / omega


def spot_angle(n_fixtures, area, throw, overlap=25.0):
    """Beam full-angle so each fixture's footprint, overlapped `overlap` times, covers the pitch."""
    r = math.sqrt(overlap * area / (n_fixtures * math.pi))
    return float(np.clip(2 * math.degrees(math.atan(r / max(throw, 1.0))), 14.0, 70.0))


def uniformity(lux_grid):
    """UEFA uniformity U1 = Emin / Emax (elite >= 0.7) and U2 = Emin / Eavg (>= 0.8)."""
    g = np.asarray(lux_grid, float)
    return float(g.min() / g.max()), float(g.min() / g.mean())


def illuminance(fixtures, targets, power, spot_deg, blend, points):
    """Horizontal illuminance (lux) at pitch points from every fixture: an analytic check of a design."""
    E = np.zeros(len(points))
    a = math.radians(spot_deg / 2)
    I = power / (4 * math.pi)
    for p, t in zip(np.asarray(fixtures), np.asarray(targets)):
        axis = (t - p) / np.linalg.norm(t - p)
        d = points - p
        r = np.linalg.norm(d, axis=1)
        u = d / r[:, None]
        cos_t = u @ axis
        # Blender's spot mask: smoothstep in cosine space across the blend band
        ca = math.cos(a)
        x = np.clip((cos_t - ca) / max(blend * (1 - ca), 1e-6), 0, 1)
        w = x * x * (3 - 2 * x)
        cos_inc = np.clip(-u[:, 2], 0, 1)
        E += I * w * cos_inc / r ** 2
    return E * LUX_PER_W


def contribution(fixture, aims, power, spot_deg, blend, points):
    """(len(aims), len(points)) lux from one fixture aimed at each candidate aim point."""
    return np.stack([illuminance([fixture], [a], power, spot_deg, blend, points) for a in aims])


def optimise_aims(fixtures, candidates, points, spot_deg, blend=0.9, iters=600, seed=3):
    """Greedy re-aiming, as lighting designers do: repeatedly move the fixture whose re-aim at the
    darkest region most improves uniformity (Emin/Emax). Returns targets and the lux grid (1 W units)."""
    fixtures = np.asarray(fixtures, float)
    C = np.stack([contribution(f, candidates, 1.0, spot_deg, blend, points) for f in fixtures])  # F, A, G
    F = len(fixtures)
    # a fixture never aims back over its own shoulder: its aim point lies in front of it, seen from the
    # pitch's spine (the long axis between the penalty spots' latitudes); the start deals candidates
    # round the fixtures (spread light), skipping those behind each fixture
    spine = max(float(candidates[:, 0].max() - candidates[:, 1].max()), 0.0)
    fx, fy = fixtures[:, 0][:, None], fixtures[:, 1][:, None]
    ix, iy = np.clip(fx, -spine, spine) - fx, -fy
    dx, dy = candidates[None, :, 0] - fx, candidates[None, :, 1] - fy
    allowed = (np.hypot(ix, iy) < 1.0) | (dx * ix + dy * iy >= 0.0)
    A = len(candidates)
    a = (np.arange(F) * 37 + seed) % A
    for f in range(F):
        a0 = a[f]
        a[f] = next((k for k in ((a0 + j * 7) % A for j in range(A)) if allowed[f, k]), a0)
    E = C[np.arange(F), a].sum(0)
    for _ in range(iters):
        g = int(np.argmin(E))
        star = int(np.argmin(np.linalg.norm(candidates[:, :2] - points[g, :2], axis=1)))
        new = E[None, :] - C[np.arange(F), a] + C[:, star]
        score = np.where(allowed[:, star], new.min(1) / new.max(1), -np.inf)
        f = int(np.argmax(score))
        if score[f] <= E.min() / E.max() + 1e-6:
            # try the second-darkest region before giving up
            order = np.argsort(E)[1:8]
            moved = False
            for g in order:
                star = int(np.argmin(np.linalg.norm(candidates[:, :2] - points[g, :2], axis=1)))
                new = E[None, :] - C[np.arange(F), a] + C[:, star]
                score = np.where(allowed[:, star], new.min(1) / new.max(1), -np.inf)
                f = int(np.argmax(score))
                if score[f] > E.min() / E.max() + 1e-6:
                    moved = True
                    break
            if not moved:
                break
        E = E - C[f, a[f]] + C[f, star]
        a[f] = star
    return candidates[a], E


def design_poles(spec, L, W, runoff_side):
    """A grassroots pole installation designed to a lux target (the Rust port's design() for
    "poles"): fixtures (position, aim target), tube masts and crossbars, lamp power and beam."""
    lamps = int(spec.get("lamps", 3))
    fixtures, masts = [], []
    for pole in poles(L, W, {"per_side": int(spec.get("per_side", 3)), "height": spec.get("height", 16.0)}, runoff_side + 1.2):
        bx, by = pole["base"]
        h = pole["height"]
        tubes, _ = pole_parts(pole, lamps)      # sectional pole, collars, crossbars (Rust design())
        masts += tubes
        for j in range(lamps):
            x = bx + (j - (lamps - 1.0) / 2.0) * 0.8
            fixtures.append((x, by - math.copysign(1.0, by) * 0.5, h + 0.3))
    d = {"fixture": spec.get("fixture", "led"), "kelvin": float(spec.get("kelvin", 5000.0)),
         "lux_target": float(spec.get("lux", 150.0)), "fixtures": [], "tube_masts": masts,
         "power": 0.0, "beam": 0.0}
    if not fixtures:
        return d
    gx, gy = np.meshgrid(np.linspace(-L / 2, L / 2, 22), np.linspace(-W / 2, W / 2, 15))
    grid = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cx, cy = np.meshgrid(np.linspace(-L / 2, L / 2, 15), np.linspace(-W / 2, W / 2, 9))
    cands = np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)])
    best = None
    for beam in spec.get("beams") or [70.0, 90.0, 120.0]:
        t, e = optimise_aims(np.array(fixtures), cands, grid, beam)
        u1, u2 = uniformity(e)
        if best is None or u1 > best[2]:
            best = (beam, t, u1, u2, e)
    beam, targets, u1, u2, e = best
    mean = float(np.mean(e))
    d["power"] = d["lux_target"] / mean
    d["beam"] = beam
    d["fixtures"] = [(tuple(map(float, f)), tuple(map(float, t))) for f, t in zip(fixtures, targets)]
    return d


# height of a fixture's trunnion axis (the housing centre) above its origin, the yoke's mounting
# plate (FIXTURE_PIVOT in kit/assets/floodlight.py)
PIVOT = {"led": 0.30, "halide": 0.33}


def mount_point(pos, target, pivot):
    """Where a fixture's origin goes so its housing centre sits at `pos` aimed at `target`: the pivot
    height back along its local -Z (local +Y aims, Blender's track-to Y with Z up)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    d = target - pos
    d = d / np.linalg.norm(d)
    yaw = math.atan2(-d[0], d[1])
    tilt = math.asin(min(max(d[2], -1.0), 1.0))
    st, ct = math.sin(tilt), math.cos(tilt)
    return pos - np.array([st * math.sin(yaw), -st * math.cos(yaw), ct]) * pivot


# -- structure detail (catwalks, ladders, platforms, plinths): shared with viewer/gen floodlights.rs

TAG_STEEL, TAG_CONCRETE, TAG_DARK = 0, 1, 2
GANTRY_DECK = -1.9         # catwalk deck (the truss's bottom chords) below the roof edge line (m)
LAMP_ROWS = (1.4, 0.4)     # the two lamp bars, above the deck (m)
LAMP_OUT = -0.3            # lamps hang this far in front of the catwalk's pitch-side face line (m)
LAMP_PAIR = 0.6            # lamps of a cluster sit this far either side of its centre (m)
LAMP_SCALE = 1.7           # roof luminaires are large LED floods (~1.05 x 0.8 m)


def roof_gantry(lines, spacing):
    """A catwalk truss slung under the roof's leading edge: top chords under the edge, bottom chords
    carrying a grating deck with toe boards, a cable tray and handrails, verticals and diagonal lacing
    on both faces, hangers to the roof, a PA speaker cluster every tenth bay, and on the pitch side two
    lamp bars carrying the fixtures in 2 x 2 clusters.
    Returns (steel polylines, MeshAcc, clusters [(design point, [lamp positions])])."""
    from .geometry import MeshAcc
    from .shapes import obox, offset, outward, sweep
    acc = MeshAcc()
    steel, clusters = [], []
    D = GANTRY_DECK
    for line in lines:
        line = np.asarray(line, float)
        if len(line) < 2:
            continue
        O = outward(line)
        sweep(acc, line, [(0.3, D), (1.5, D), (1.5, D + 0.07), (0.3, D + 0.07)], TAG_STEEL, O)
        for o in (0.3, 1.48):
            sweep(acc, line, [(o, D + 0.07), (o + 0.02, D + 0.07), (o + 0.02, D + 0.22), (o, D + 0.22)], TAG_STEEL, O)
        sweep(acc, line, [(1.2, D + 0.07), (1.45, D + 0.07), (1.45, D + 0.17), (1.2, D + 0.17)], TAG_DARK, O)
        for o, dz, r in ((0.3, -0.1, 0.06), (1.5, -0.1, 0.06), (0.3, D, 0.06), (1.5, D, 0.06),
                         (0.3, D + 1.05, 0.024), (1.5, D + 1.05, 0.024), (0.3, D + 0.55, 0.018),
                         (1.5, D + 0.55, 0.018), (0.0, D + LAMP_ROWS[0], 0.04), (0.0, D + LAMP_ROWS[1], 0.04)):
            steel.append((offset(line, O, o, dz), r))
        for k, p in enumerate(roof_ribbon([line], spacing)):
            n = edge_normal(line, p)
            tan = np.array([-n[1], n[0], 0.0])

            def at(o, dz, dt=0.0, p=p, n=n, tan=tan):
                return np.array([p[0] + n[0] * o, p[1] + n[1] * o, p[2] + dz]) + tan * dt
            for o in (0.3, 1.5):
                steel.append((np.array([at(o, D), at(o, -0.1)]), 0.04))
                steel.append((np.array([at(o, D, -0.5 * spacing), at(o, -0.1)]), 0.03))
                steel.append((np.array([at(o, -0.1), at(o, D, 0.5 * spacing)]), 0.03))
            steel.append((np.array([at(0.3, D), at(1.5, D)]), 0.035))
            steel.append((np.array([at(0.3, -0.1), at(1.5, -0.1)]), 0.035))
            if k % 2 == 0:
                steel.append((np.array([at(0.9, -0.1), at(0.9, 0.35)]), 0.04))
            if k % 10 == 5:
                obox(acc, at(0.9, D - 0.6), (tan[0], tan[1], 0.0), (n[0], n[1], 0.0), 0.3, 0.25, 0.45, TAG_DARK)
            c = at(LAMP_OUT, D + 0.5 * (LAMP_ROWS[0] + LAMP_ROWS[1]))
            members = [at(LAMP_OUT, D + row, dt) for row in LAMP_ROWS for dt in (-LAMP_PAIR, LAMP_PAIR)]
            for q in members:
                steel.append((np.array([q, q + np.array([n[0], n[1], 0.0]) * -LAMP_OUT]), 0.025))
            clusters.append((c, members))
    return steel, acc, clusters


def _lattice_half(top):
    w0, w1 = max(2.2, top / 14), 1.0
    return lambda z: (w0 + (w1 - w0) * z / top) / 2


def mast_parts(mast, pts, cols, lattice=True):
    """A corner mast's plinth, caged access ladder (on the face away from the pitch), rest platforms,
    the service walkway behind the lamp head and a heavier frame round the head.
    pts: head_layout's fixture points. Returns (steel polylines, MeshAcc)."""
    from .geometry import MeshAcc
    from .shapes import obox, zbox
    acc = MeshAcc()
    steel = []
    bx, by = mast["base"]
    H = mast["height"]
    top = H - 1.5
    if lattice:
        half = _lattice_half(top)
        face = half
        e = half(0.0) + 0.6
        zbox(acc, (bx - e, by - e, -0.04), (bx + e, by + e, 0.35), TAG_CONCRETE)
    else:
        def face(z):
            return 0.55
        zbox(acc, (bx - 1.3, by - 1.3, -0.04), (bx + 1.3, by + 1.3, 0.3), TAG_CONCRETE)
        zbox(acc, (bx - 0.85, by - 0.85, 0.3), (bx + 0.85, by + 0.85, 0.36), TAG_STEEL)
    pts = np.asarray(pts, float)
    yaw = math.atan2(-by, -bx)
    tilt = math.atan2(H, math.hypot(bx, by))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, math.cos(tilt)]) + fwd * math.sin(tilt)
    z_walk = float(pts[:, 2].min()) - 0.7
    sx = 1.0 if bx >= 0 else -1.0

    def lx(z, d=0.3):
        return bx + sx * (face(z) + d)
    # ladder: stiles and rungs every 0.3 m, a cage of hoops and straps from 2.5 m up
    z0 = 0.35
    for dy in (-0.22, 0.22):
        steel.append((np.array([(lx(z0), by + dy, z0), (lx(z_walk), by + dy, z_walk + 1.1)]), 0.025))
    n_r = int((z_walk - z0) / 0.3)
    for k in range(1, n_r + 1):
        z = z0 + k * 0.3
        steel.append((np.array([(lx(z), by - 0.22, z), (lx(z), by + 0.22, z)]), 0.012))
    angles = np.linspace(-math.pi / 2, math.pi / 2, 7)
    hoops = np.arange(2.5, z_walk, 1.0)
    for z in hoops:
        steel.append((np.array([(lx(z) + sx * 0.45 * math.cos(a), by + 0.38 * math.sin(a), z) for a in angles]), 0.014))
    if len(hoops) > 1:
        for a in angles[1:-1]:
            steel.append((np.array([(lx(z) + sx * 0.45 * math.cos(a), by + 0.38 * math.sin(a), z) for z in hoops]),
                          0.012))
    # rest platforms every 12 m beside the ladder
    z = 12.0
    while z < z_walk - 4.0:
        x0 = lx(z, 0.0)
        obox(acc, (x0 + sx * 0.55, by, z - 0.03), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.55, 0.65, 0.03, TAG_STEEL)
        rail = [(x0, by - 0.65, z + 1.0), (x0 + sx * 1.1, by - 0.65, z + 1.0), (x0 + sx * 1.1, by + 0.65, z + 1.0),
                (x0, by + 0.65, z + 1.0)]
        steel.append((np.array(rail), 0.022))
        for q in rail[1:3]:
            steel.append((np.array([(q[0], q[1], z), q]), 0.02))
        z += 12.0
    # service walkway behind and below the lamp head, railed all round
    hs = (cols - 1) / 2 * 0.8 + 0.5
    c = np.array([bx, by, z_walk - 0.03]) + fwd * 0.7
    obox(acc, c, side, fwd, hs, 0.9, 0.03, TAG_STEEL)
    ring = [c + side * a * hs + fwd * b * 0.9 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
    steel.append((np.array([q + np.array([0, 0, 1.1]) for q in ring]), 0.022))
    for q in ring[:4]:
        steel.append((np.array([q, q + np.array([0, 0, 1.1])]), 0.02))
    # a heavier frame round the lamp grid
    corners = [pts[0], pts[cols - 1], pts[-1], pts[-cols]]
    mid = np.mean(corners, 0)
    frame = []
    for q in corners + corners[:1]:
        d = q - mid
        frame.append(q + side * (0.45 if d @ side > 0 else -0.45) + up * (0.42 if d @ up > 0 else -0.42) - fwd * 0.1)
    steel.append((np.array(frame), 0.07))
    return steel, acc


def pole_parts(pole, lamps):
    """A grassroots pole: three telescoped sections with collars, the lamp crossbar, footing, base
    flange, access door, control gear at the head and a feeder pillar. Returns (tube segments
    [(a, b, r)], MeshAcc)."""
    from .geometry import MeshAcc
    from .shapes import obox, zbox
    acc = MeshAcc()
    bx, by = pole["base"]
    h = pole["height"]
    tubes = []
    cuts = [0.0, 0.4 * h, 0.75 * h, h]
    radii = [0.2, 0.165, 0.13]
    for k in range(3):
        tubes.append(((bx, by, cuts[k]), (bx, by, cuts[k + 1]), radii[k]))
    for k in (1, 2):
        tubes.append(((bx, by, cuts[k] - 0.1), (bx, by, cuts[k] + 0.06), radii[k - 1] + 0.012))
    bar = max(1.2, 0.4 * (lamps - 1.0) + 0.4)      # the crossbar spans every lamp
    tubes.append(((bx - bar, by, h), (bx + bar, by, h), 0.05))
    tubes.append(((bx - bar, by, h - 0.45), (bx + bar, by, h - 0.45), 0.035))
    for x in (bx - bar + 0.1, bx + bar - 0.1):
        tubes.append(((x, by, h - 0.45), (x, by, h), 0.03))
    sgn = 1.0 if by >= 0 else -1.0          # away from the pitch
    zbox(acc, (bx - 0.6, by - 0.6, -0.04), (bx + 0.6, by + 0.6, 0.15), TAG_CONCRETE)
    zbox(acc, (bx - 0.32, by - 0.32, 0.15), (bx + 0.32, by + 0.32, 0.19), TAG_STEEL)
    obox(acc, (bx, by + sgn * 0.2, 0.95), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.1, 0.025, 0.3, TAG_DARK)
    obox(acc, (bx, by + sgn * 0.3, h - 1.1), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.22, 0.14, 0.3, TAG_DARK)
    obox(acc, (bx + 0.9, by + sgn * 0.7, 0.6), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.3, 0.2, 0.6, TAG_DARK)
    return tubes, acc


LENS_PANEL = (0.27, 0.19, 0.076)   # half width, half height and forward offset (in front of the bezel) of an LED lens


def lens_panels(lamps, scale):
    """The glowing diffuser panel over each LED fixture's lens (a quad just in front of the lens,
    facing the aim): what makes a roof ring read as lit luminaires from across the ground.
    lamps: [(housing centre, aim target)]. Returns a MeshAcc."""
    from .geometry import MeshAcc
    acc = MeshAcc()
    hx, hz, fy = (v * scale for v in LENS_PANEL)
    for q, t in lamps:
        q, t = np.asarray(q, float), np.asarray(t, float)
        d = (t - q) / np.linalg.norm(t - q)
        yaw = math.atan2(-d[0], d[1])
        tilt = math.asin(min(max(d[2], -1.0), 1.0))
        st, ct = math.sin(tilt), math.cos(tilt)
        ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        az = np.array([st * math.sin(yaw), -st * math.cos(yaw), ct])
        c = q + d * fy
        base = acc.n
        acc.verts.append(np.array([c - ax * hx - az * hz, c + ax * hx - az * hz, c + ax * hx + az * hz,
                                   c - ax * hx + az * hz]))
        acc.n += 4
        acc.faces.append((base, base + 3, base + 2, base + 1))
        acc.tags.append(0)
        acc.uvs += [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    return acc
