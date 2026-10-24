"""Facades: the outside of a stand run. Pure numpy, run coordinates (t, s, z).

The facade line sits behind everything (the stand's deepest structure or the roof's back-span tie,
whichever is further out), so the roof structure is enclosed as on a real stadium. Systems:

  concrete_frame  exposed columns on every bay, floor slabs every level, dark glazing between
  brick           heritage brick with stone string courses, windows per bay, turnstile entrances
  metal_bands     horizontal metal cladding bands in two tones, shadow gaps between them
  glass           curtain wall: glazing with mullions and transoms
  etfe            inflated diamond cushions on a white frame, can glow at night
  fins            vertical metal fins in front of a dark wall

Every system also gets ground-floor entrances, optional stair cores, and gable walls where a stand
end is exposed (open corners).
"""
import math

import numpy as np

from .geometry import MeshAcc, _world, section_hull, stand_depth

TAG_MAIN, TAG_ACCENT, TAG_DARK, TAG_TRIM, TAG_ETFE, TAG_CORE = range(6)
LEVEL = 4.5
DOOR_CLEAR, DOOR_TOP = 9.6, 3.55   # entrances: the cladding is cut this wide up to above the canopy
ETFE_BASE = 4.0                    # ETFE cushions start above a glazed ground floor this tall

DEFAULTS = {"type": "concrete_frame", "offset": 0.6, "cores_every": 4, "entrance_every": 2}


def line(tiers, roof_info, spec):
    s = stand_depth(tiers)
    top = max(t["h"][-1] for t in tiers) + 1.6
    if roof_info:
        s = max(s, roof_info["tail_s"])
        top = max(top, roof_info["tail_top"])
    return s + spec["offset"], top


def _rect(s0, s1, z0, z1):
    return np.array([(s0, z0), (s1, z0), (s1, z1), (s0, z1)])


def _prism(acc, run, ta, tb, s0, s1, z0, z1, tag, cap=None):
    acc.add_prism(run, ta, tb, (_rect(s0, s1, z0, z1), [tag] * 4), cap_tag=tag if cap is None else cap)


def _split(t0, t1, doors):
    """[t0, t1] cut at the door intervals: (ta, tb, inside a door)."""
    out, t = [], t0
    for a, b in doors:
        a, b = max(a, t0), min(b, t1)
        if b <= a or a < t:
            continue
        if a > t:
            out.append((t, a, False))
        out.append((a, b, True))
        t = b
    if t < t1:
        out.append((t, t1, False))
    return out


def _repeat(run, s, spacing, t0, t1, offset=0.5):
    """Positions along a run every `spacing` metres (measured at section position s)."""
    m = run.metric(s)
    n = max(1, int((t1 - t0) * m / spacing))
    step = (t1 - t0) / n
    return [t0 + (k + offset) * step for k in range(n)], step


def build(run, tiers, bay_list, spec, roof_info, ends=(False, False)):
    spec = dict(DEFAULTS, **(spec or {}))
    kind = spec["type"]
    acc = MeshAcc()
    tubes, sockets = [], []
    sw, H = line(tiers, roof_info, spec)
    t0, t1 = run.t0, run.t1
    levels = list(np.arange(LEVEL, H - 1.0, LEVEL))

    def hw(width, s):  # half-width in t units
        return 0.5 * width / run.metric(s)

    backing = TAG_DARK if kind in ("concrete_frame", "fins", "glass", "etfe") else TAG_MAIN
    # the inner face (seen from the back rows, under the roof) is the concourse's concrete wall
    acc.add_prism(run, t0, t1, (_rect(sw, sw + 0.3, 0.0, H), [backing] * 3 + [TAG_CORE]), cap_tag=backing)
    _prism(acc, run, t0, t1, sw - 0.1, sw + 0.5, H - 0.6, H + 0.3, TAG_TRIM)   # coping
    out_s = sw + 0.3
    # entrances: the cladding stops above the door canopy there
    doors = [(0.5 * (ta + tb) - hw(DOOR_CLEAR, out_s), 0.5 * (ta + tb) + hw(DOOR_CLEAR, out_s))
             for b, (ta, tb) in enumerate(bay_list) if b % spec["entrance_every"] == 0]

    def at_door(t):
        return any(a < t < b for a, b in doors)

    if kind == "concrete_frame":
        for ta, _ in bay_list + [(bay_list[-1][1], None)]:
            _prism(acc, run, ta - hw(0.7, out_s), ta + hw(0.7, out_s), sw, sw + 1.1, 0.0, H, TAG_MAIN)
        for z in levels:
            _prism(acc, run, t0, t1, sw, sw + 0.85, z - 0.25, z + 0.25, TAG_MAIN)
    elif kind == "brick":
        for z in levels:
            _prism(acc, run, t0, t1, sw, sw + 0.38, z - 0.15, z + 0.12, TAG_TRIM)
        for ta, tb in bay_list:
            tm = 0.5 * (ta + tb)
            for z in levels:
                if z + 3.0 > H - 0.8:
                    continue
                for dx in (-1.6, 1.6):
                    tc = tm + dx / run.metric(out_s)
                    _prism(acc, run, tc - hw(1.3, out_s), tc + hw(1.3, out_s), sw + 0.3, sw + 0.33, z + 0.5, z + 2.9,
                           TAG_DARK)
                    _prism(acc, run, tc - hw(1.6, out_s), tc + hw(1.6, out_s), sw + 0.3, sw + 0.4, z + 2.9, z + 3.2,
                           TAG_TRIM)
                    _prism(acc, run, tc - hw(1.5, out_s), tc + hw(1.5, out_s), sw + 0.3, sw + 0.42, z + 0.38, z + 0.5,
                           TAG_TRIM)
        _prism(acc, run, t0, t1, sw, sw + 0.42, 0.0, 0.6, TAG_TRIM)          # plinth
    elif kind == "metal_bands":
        z, k = 0.6, 0
        while z < H - 0.7:
            h = min(1.2, H - 0.7 - z)
            for ta, tb, door in _split(t0, t1, doors):
                lo = max(z + 0.03, DOOR_TOP) if door else z + 0.03
                if lo < z + h - 0.03:
                    _prism(acc, run, ta, tb, sw + 0.3, sw + 0.42, lo, z + h - 0.03,
                           TAG_MAIN if k % 2 == 0 else TAG_ACCENT)
            z += h
            k += 1
    elif kind == "glass":
        for ta, tb, door in _split(t0, t1, doors):
            _prism(acc, run, ta, tb, sw + 0.3, sw + 0.34, DOOR_TOP if door else 0.0, H - 0.6, TAG_DARK)
        ts, _ = _repeat(run, out_s, 1.6, t0, t1, 0.0)
        for t in ts + [t1]:
            _prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.5,
                   DOOR_TOP if at_door(t) else 0.0, H - 0.6, TAG_TRIM)
        for z in np.arange(3.6, H - 0.6, 3.6):
            _prism(acc, run, t0, t1, sw + 0.3, sw + 0.48, z - 0.06, z + 0.06, TAG_TRIM)
    elif kind == "fins":
        ts, _ = _repeat(run, out_s, spec.get("fin_spacing", 1.1), t0, t1)
        for t in ts:
            _prism(acc, run, t - hw(0.07, out_s), t + hw(0.07, out_s), sw + 0.35, sw + 1.35,
                   DOOR_TOP if at_door(t) else 0.5, H - 0.3, TAG_MAIN)
    elif kind == "etfe":
        # cushions above a glazed ground floor (mullions, and a fascia closing the cushion line)
        _etfe(acc, tubes, run, sw + 1.2, H, spec, ETFE_BASE)
        ts, _ = _repeat(run, out_s, 1.6, t0, t1)
        for t in ts:
            if not at_door(t):
                _prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.45, 0.0, ETFE_BASE - 0.4,
                       TAG_TRIM)
        _prism(acc, run, t0, t1, sw + 0.3, sw + 1.3, ETFE_BASE - 0.4, ETFE_BASE, TAG_TRIM)
    else:
        raise ValueError(f"unknown facade type {kind!r}")

    # ground-floor entrances (turnstiles / concourse doors) and stair cores
    for b, (ta, tb) in enumerate(bay_list):
        tm = 0.5 * (ta + tb)
        if b % spec["entrance_every"] == 0:
            _prism(acc, run, tm - hw(4.2, out_s), tm + hw(4.2, out_s), sw + 0.3, sw + 0.36, 0.0, 3.2, TAG_DARK)
            _prism(acc, run, tm - hw(4.8, out_s), tm + hw(4.8, out_s), sw + 0.3, sw + 1.4, 3.2, 3.55, TAG_TRIM)
            # door jambs and two mullions, so the opening reads as doors and not as more glazing
            for dx in (-4.3, -1.4, 1.4, 4.3):
                tc = tm + dx / run.metric(out_s)
                wd = 0.25 if abs(dx) > 4.0 else 0.1
                _prism(acc, run, tc - hw(wd, out_s), tc + hw(wd, out_s), sw + 0.3, sw + 0.55, 0.0, 3.2, TAG_TRIM)
        if spec["cores_every"] and b % spec["cores_every"] == spec["cores_every"] // 2 and H > 9:
            _core(acc, run, tm, sw, H, hw)
        p, n = run.frame(tm)
        q = p + n * (sw + 1.5)
        sockets.append(("facade_sign", f"{run.name}_B{b}", (float(q[0]), float(q[1]), round(0.62 * H, 3)), -n,
                        {"width": round((tb - ta) * run.metric(sw), 3), "max_height": round(0.3 * H, 3)}))
    for at_start, flag in zip((True, False), ends):
        if flag:
            _gable(acc, run, tiers, sw, H, at_start, kind)
    return {"mesh": acc.arrays(), "tubes": tubes, "sockets": sockets, "line_s": sw, "height": H}


def _core(acc, run, tm, sw, H, hw):
    s1 = sw + 6.0
    _prism(acc, run, tm - hw(7.0, s1), tm + hw(7.0, s1), sw, s1, 0.0, H + 1.5, TAG_CORE)
    for dx in (-2.0, 0.0, 2.0):
        tc = tm + dx / run.metric(s1)
        _prism(acc, run, tc - hw(0.35, s1), tc + hw(0.35, s1), s1, s1 + 0.03, 1.0, H - 0.5, TAG_DARK)


def _gable(acc, run, tiers, sw, H, at_start, kind):
    """End wall closing an exposed stand end, from the seating section out to the facade."""
    hull = section_hull(tiers).tolist() + [(sw + 0.3, 0.0), (sw + 0.3, H)]
    P = sorted(set(map(tuple, hull)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    poly = np.array(lo[:-1] + up[:-1])
    tag = TAG_MAIN if kind in ("brick", "metal_bands", "fins") else TAG_CORE
    d = 0.3 / run.metric(0.0)
    ta, tb = (run.t0 - d, run.t0) if at_start else (run.t1, run.t1 + d)
    acc.add_prism(run, ta, tb, (poly, [tag] * len(poly)), cap_tag=tag)


def _etfe(acc, tubes, run, s, H, spec, z0):
    """Diamond cushions, bulging outward, on a white frame."""
    wd, hd = spec.get("cushion", (4.0, 4.0))
    bulge = spec.get("bulge", 0.35)
    m = run.metric(s)
    length = (run.t1 - run.t0) * m
    n_u = max(1, int(round(length / wd)))
    wd = length / n_u
    n_z = max(1, int((H - z0 - 0.5) / hd))
    k = 5
    grid = np.linspace(0, 1, k + 1)
    A, B = np.meshgrid(grid, grid, indexing="ij")
    out = np.sin(np.pi * A) * np.sin(np.pi * B) * bulge
    zmax = z0 + n_z * hd
    # one extra row/column of cushions all round, clamped to the facade edge: the partial cushions
    # at the borders become the triangles a real ETFE skin has there
    for j in range(-1, 2 * n_z):
        cz = z0 + hd / 2 + j * hd / 2
        for i in range(-1, n_u + 1):
            cu = (i + 0.5) * wd if j % 2 == 0 else (i + 1.0) * wd
            V0 = np.array([cu, cz - hd / 2])
            V1 = np.array([cu + wd / 2, cz])
            V3 = np.array([cu - wd / 2, cz])
            uz = V0[None, None, :] + A[..., None] * (V1 - V0) + B[..., None] * (V3 - V0)
            if uz[..., 0].max() <= 0 or uz[..., 0].min() >= length or uz[..., 1].max() <= z0 or uz[..., 1].min() >= zmax:
                continue
            uz[..., 0] = np.clip(uz[..., 0], 0, length)
            uz[..., 1] = np.clip(uz[..., 1], z0, zmax)
            verts = []
            for a in range(k + 1):
                for b in range(k + 1):
                    t = run.t0 + uz[a, b, 0] / m
                    verts.append(_world(run, t, s + out[a, b], uz[a, b, 1]))
            base = acc.n
            acc.verts.append(np.array(verts))
            acc.n += len(verts)
            for a in range(k):
                for b in range(k):
                    q = (base + a * (k + 1) + b, base + (a + 1) * (k + 1) + b,
                         base + (a + 1) * (k + 1) + b + 1, base + a * (k + 1) + b + 1)
                    acc.faces.append(q)
                    acc.tags.append(TAG_ETFE)
                    acc.uvs += [(0, 0)] * 4
    # frame: the two diagonal families of cushion edges
    for i in range(-n_z, n_u + n_z + 1):
        for sign in (1, -1):
            pts = []
            for zz in np.linspace(z0, zmax, 12):
                u = (i + 0.5) * wd + sign * (zz - z0) * (wd / hd)
                if 0 <= u <= length:
                    pts.append(_world(run, run.t0 + u / m, s, zz))
            if len(pts) >= 2:
                tubes.append((np.array(pts), 0.045))
