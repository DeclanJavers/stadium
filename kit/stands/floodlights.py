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


def pylons(L, W, A, B, reach, roof_top, spec):
    """Four masts on the diagonals, outside the stands. reach: how far out the structures go."""
    out = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        d = np.array([sx * (L / 2), sy * (W / 2)])
        u = d / np.linalg.norm(d)
        base = np.array([sx * (A + reach * 0.5), sy * (B + reach * 0.5)])
        r = max(np.linalg.norm(base), np.linalg.norm(d) + 15.0)
        p = u * r if reach > 30 else base
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


def roof_ribbon(edge_lines, spacing):
    pts = []
    for line in edge_lines:
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(1, int(s[-1] / spacing))
        for t in (np.arange(n) + 0.5) * s[-1] / n:
            pts.append([np.interp(t, s, line[:, i]) for i in range(3)])
    return np.array(pts)


def poles(L, W, spec, offset):
    out = []
    n = spec["per_side"]
    xs = np.linspace(-L / 2, L / 2, n + 2)[1:-1] if n > 1 else np.array([0.0])
    for sy in (1, -1):
        for x in xs:
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
    a = (np.arange(F) * 37 + seed) % len(candidates)     # deterministic, portable start
    E = C[np.arange(F), a].sum(0)
    for _ in range(iters):
        g = int(np.argmin(E))
        star = int(np.argmin(np.linalg.norm(candidates[:, :2] - points[g, :2], axis=1)))
        new = E[None, :] - C[np.arange(F), a] + C[:, star]
        score = new.min(1) / new.max(1)
        f = int(np.argmax(score))
        if score[f] <= E.min() / E.max() + 1e-6:
            # try the second-darkest region before giving up
            order = np.argsort(E)[1:8]
            moved = False
            for g in order:
                star = int(np.argmin(np.linalg.norm(candidates[:, :2] - points[g, :2], axis=1)))
                new = E[None, :] - C[np.arange(F), a] + C[:, star]
                score = new.min(1) / new.max(1)
                f = int(np.argmax(score))
                if score[f] > E.min() / E.max() + 1e-6:
                    moved = True
                    break
            if not moved:
                break
        E = E - C[f, a[f]] + C[f, star]
        a[f] = star
    return candidates[a], E
