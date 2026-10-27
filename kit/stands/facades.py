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
GROUND = 4.5                       # metal bands and fins start above a glazed ground floor this tall

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
             for b, (ta, tb) in enumerate(bay_list) if is_entrance(run, b, ta, tb, spec["entrance_every"], sw)]

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
        _shopfront(acc, run, sw, t0, t1, doors, hw)
        z, k = GROUND, 0
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
        _shopfront(acc, run, sw, t0, t1, doors, hw)
        zs = skin_base(H, GROUND)
        if zs > GROUND:
            _concourse(acc, run, sw, t0, t1, GROUND, zs, hw)
            _prism(acc, run, t0, t1, sw + 0.3, sw + 1.45, zs - 0.4, zs, TAG_TRIM)
        ts, _ = _repeat(run, out_s, spec.get("fin_spacing", 1.1), t0, t1)
        for t in ts:
            _prism(acc, run, t - hw(0.07, out_s), t + hw(0.07, out_s), sw + 0.35, sw + 1.35,
                   zs, H - 0.3, TAG_MAIN)
    elif kind == "etfe":
        # cushions above a glazed ground floor (mullions, and a fascia closing the cushion line)
        zs = skin_base(H, ETFE_BASE)
        _etfe(acc, tubes, run, sw + 1.2, H, spec, zs)
        if zs > ETFE_BASE:
            _concourse(acc, run, sw, t0, t1, ETFE_BASE, zs - 0.4, hw)
            _prism(acc, run, t0, t1, sw + 0.3, sw + 1.3, zs - 0.4, zs, TAG_TRIM)
        ts, _ = _repeat(run, out_s, 1.6, t0, t1)
        for t in ts:
            if not at_door(t):
                _prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.45, 0.0, ETFE_BASE - 0.4,
                       TAG_TRIM)
        _prism(acc, run, t0, t1, sw + 0.3, sw + 1.3, ETFE_BASE - 0.4, ETFE_BASE, TAG_TRIM)
    else:
        raise ValueError(f"unknown facade type {kind!r}")

    if H < SMALL_FACADE:
        _frontage(acc, run, sw, H, hw)
    # ground-floor entrances (turnstiles / concourse doors) and stair cores
    for b, (ta, tb) in enumerate(bay_list):
        tm = 0.5 * (ta + tb)
        core = bool(spec["cores_every"]) and b % spec["cores_every"] == spec["cores_every"] // 2 and H > 9
        if is_entrance(run, b, ta, tb, spec["entrance_every"], sw):
            _prism(acc, run, tm - hw(4.2, out_s), tm + hw(4.2, out_s), sw + 0.3, sw + 0.36, 0.0, 3.2, TAG_DARK)
            _prism(acc, run, tm - hw(4.8, out_s), tm + hw(4.8, out_s), sw + 0.3, sw + 1.4, 3.2, 3.55, TAG_TRIM)
            # door jambs and two mullions, so the opening reads as doors and not as more glazing
            for dx in (-4.3, -1.4, 1.4, 4.3):
                tc = tm + dx / run.metric(out_s)
                wd = 0.25 if abs(dx) > 4.0 else 0.1
                _prism(acc, run, tc - hw(wd, out_s), tc + hw(wd, out_s), sw + 0.3, sw + 0.55, 0.0, 3.2, TAG_TRIM)
            if not core:
                _turnstiles(acc, run, tm, sw, hw)
        if core:
            _core(acc, run, tm, sw, H, hw)
        p, n = run.frame(tm)
        q = p + n * (sw + 1.5)
        sockets.append(("facade_sign", f"{run.name}_B{b}", (float(q[0]), float(q[1]), round(0.62 * H, 3)), -n,
                        {"width": round((tb - ta) * run.metric(sw), 3), "max_height": round(0.3 * H, 3)}))
    for at_start, flag in zip((True, False), ends):
        if flag:
            _gable(acc, run, tiers, sw, H, at_start, kind)
    return {"mesh": acc.arrays(), "tubes": tubes, "sockets": sockets, "line_s": sw, "height": H,
            "props": plaza_props(run, bay_list, sw, spec["entrance_every"])}


PLAZA_LAMPS = 10.0       # lamp posts, bins and trees stand this far out from the facade line
LAMP_EVERY = 20.0


def plaza_props(run, bay_list, sw, entrance_every):
    """Street furniture on the plaza in front of a facade: lamp posts every ~20 m with a litter bin
    beside each, a tree or a bench between planters between each pair, and in front of each
    entrance a row of bollards, a sign post and cycle stands (a bicycle at one). Life on it:
    pigeons by the benches, gulls, matchday litter by the bins, leaves under the trees, a food van
    or a coffee cart out on the plaza. Its centrepiece (generated props): in front of the main
    stand (south, where the dugouts are; none on a curved corner) the club legend's statue with a memorial
    stone and flowers, on a long one the manager's statue and a fountain, the club shop's stall
    and a pillar box; the crest sculpture behind the north stand, a clock pillar by the east, a
    bronze ball by the west.
    Returns [(prop key, (x, y, z), yaw)], yaw turning the prop's +Y towards the facade."""
    out = []
    s_l = sw + PLAZA_LAMPS
    m = run.metric(s_l)
    n = max(1, int((run.t1 - run.t0) * m / LAMP_EVERY))
    step = (run.t1 - run.t0) / n

    def put(key, t, s, turn=0.0):
        p, nrm = run.frame(t)
        q = p + nrm * s
        out.append((key, (float(q[0]), float(q[1]), 0.0), math.atan2(nrm[0], -nrm[1]) + turn))
    for k in range(n):
        t = run.t0 + (k + 0.5) * step
        put("prop.lamp_post", t, s_l)
        put("prop.bin_litter", t + 1.2 / m, s_l)
        if k % 3 == 1:
            put("genprop.matchday_litter", t + 2.2 / m, s_l - 0.4, 0.7 * k)
        if k % 4 == 3:
            put("genprop.seagull_standing", t - 2.5 / m, s_l + 2.5, 1.9 * k)
        if k % 2 == 0 and k + 1 < n:
            put("prop.tree_birch", t + 0.5 * step, s_l + 0.5)
            if k % 4 == 0:
                put("genprop.leaf_litter", t + 0.5 * step + 1.2 / m, s_l + 1.8, 0.9 * k)
        elif k + 1 < n:
            # a bench between two planters in the bays without a tree
            tm = t + 0.5 * step
            put("prop.bench", tm, s_l + 0.5)
            for sg in (-1.0, 1.0):
                put("prop.planter", tm + sg * 2.4 / m, s_l + 0.5)
            if k % 4 == 1:
                put("genprop.pigeon_pecking", tm + 1.4 / m, s_l - 1.5, 2.3 * k)
                put("genprop.pigeon_standing", tm - 1.0 / m, s_l - 2.0, 1.1 * k + 0.5)
        if n >= 4 and k % 4 == 2:
            put("genprop.food_van" if (k // 4) % 2 == 0 else "genprop.coffee_cart", t, s_l + 15.0)
    for b, (ta, tb) in enumerate(bay_list):
        if not is_entrance(run, b, ta, tb, entrance_every, sw):
            continue
        tm = 0.5 * (ta + tb)
        mb = run.metric(sw + 5.5)
        for j in range(6):
            put("prop.bollard", tm + (j - 2.5) * 1.6 / mb, sw + 5.5)
        put("prop.sign_post", tm - 6.5 / run.metric(sw + 4.0), sw + 4.0)
        # cycle stands on the other side of the doors, a bicycle locked to the first
        for j in range(3):
            put("prop.bike_rack", tm + (6.0 + 0.9 * j) / run.metric(sw + 4.0), sw + 4.0)
        put("genprop.bicycle", tm + (6.0 + 1.85) / run.metric(sw + 4.0), sw + 4.0, math.pi / 2)
    # the centrepiece (facing out, to the people arriving)
    span = run.t1 - run.t0
    tm, ms = run.t0 + 0.5 * span, run.metric(s_l + 8.0)
    if n >= 2 and run.name in ("south", "north", "east", "west"):
        if run.name == "south":
            put("genprop.statue_legend", tm, s_l + 8.0, math.pi)
            put("genprop.plaque_post_flowers", tm + 2.6 / ms, s_l + 7.0, math.pi)
            put("genprop.memorial_stone", tm - 3.2 / ms, s_l + 7.5, math.pi)
            put("genprop.merch_stall", run.t0 + 0.38 * span, s_l + 5.0, math.pi)
            put("genprop.flags_tifo", run.t0 + 0.38 * span + 2.6 / run.metric(s_l + 5.0), s_l + 5.0, math.pi)
            put("genprop.pillar_box", run.t0 + 0.62 * span, s_l - 2.0, math.pi)
            if n >= 6:
                put("genprop.statue_manager", run.t0 + 0.25 * span, s_l + 8.0, math.pi)
                put("genprop.fountain_basin", run.t0 + 0.75 * span, s_l + 9.0)
        else:
            hero = {"north": "genprop.sculpture_crest", "east": "genprop.clock_pillar", "west": "genprop.bronze_football_plinth"}[run.name]
            put(hero, tm, s_l + 8.0, math.pi)
    return out


def _shopfront(acc, run, sw, t0, t1, doors, hw):
    """Glazed ground floor (concourse frontage) under a canopy, where the cladding starts above it."""
    out_s = sw + 0.3
    for ta, tb, door in _split(t0, t1, doors):
        if not door:
            _prism(acc, run, ta, tb, sw + 0.3, sw + 0.34, 0.0, GROUND - 0.35, TAG_DARK)
    ts, _ = _repeat(run, out_s, 1.6, t0, t1, 0.0)
    for t in ts + [t1]:
        if not any(a < t < b for a, b in doors):
            _prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.46, 0.0, GROUND - 0.35,
                   TAG_TRIM)
    _prism(acc, run, t0, t1, sw + 0.3, sw + 1.6, GROUND - 0.35, GROUND, TAG_TRIM)


def is_entrance(run, b, ta, tb, every, sw):
    """Every `every`-th bay has an entrance, where its doors fit on the run (not past a stand end)."""
    if b % every:
        return False
    tm, half = 0.5 * (ta + tb), 0.5 * DOOR_CLEAR / run.metric(sw + 0.3)
    return tm - half >= run.t0 and tm + half <= run.t1


def _turnstiles(acc, run, tm, sw, hw):
    """A turnstile block beside an entrance: a kiosk standing out from the facade with two turnstile
    openings in its face, a flat roof and a sign band in the accent colour."""
    s1 = sw + 2.5
    tc = tm + 6.6 / run.metric(s1)
    if tc + hw(3.9, s1) > run.t1:          # no room past the doors before the stand end
        return
    _kiosk(acc, run, tc, sw, hw)


def _kiosk(acc, run, tc, sw, hw):
    s1 = sw + 2.5
    _prism(acc, run, tc - hw(3.6, s1), tc + hw(3.6, s1), sw + 0.3, s1, 0.0, 2.9, TAG_CORE)
    _prism(acc, run, tc - hw(3.9, s1), tc + hw(3.9, s1), sw + 0.3, s1 + 0.2, 2.9, 3.1, TAG_TRIM)
    _prism(acc, run, tc - hw(3.4, s1), tc + hw(3.4, s1), s1, s1 + 0.04, 2.25, 2.8, TAG_ACCENT)
    for dx in (-0.8, 0.8):
        tk = tc + dx / run.metric(s1)
        _prism(acc, run, tk - hw(1.1, s1), tk + hw(1.1, s1), s1, s1 + 0.02, 0.0, 2.1, TAG_DARK)


SMALL_FACADE = 9.0      # a club stand's back wall lower than this gets doors, a tea bar, vents and downpipes


def _frontage(acc, run, sw, H, hw):
    """The back of a small club stand: a turnstile block at one end, steel doors in frames, a tea-bar
    hatch with a roller shutter, counter, canopy and sign in the middle, high louvre vents and
    downpipes."""
    out_s = sw + 0.3
    m = run.metric(out_s)
    L = (run.t1 - run.t0) * m
    tmid = 0.5 * (run.t0 + run.t1)
    kiosk = L > 14.0
    if kiosk:
        _kiosk(acc, run, run.t0 + 2.4 / run.metric(sw + 2.5), sw, hw)
    _prism(acc, run, tmid - hw(2.8, out_s), tmid + hw(2.8, out_s), sw + 0.3, sw + 0.38, 0.9, 2.45, TAG_TRIM)
    _prism(acc, run, tmid - hw(2.6, out_s), tmid + hw(2.6, out_s), sw + 0.3, sw + 0.4, 1.0, 2.3, TAG_CORE)
    _prism(acc, run, tmid - hw(2.8, out_s), tmid + hw(2.8, out_s), sw + 0.3, sw + 0.75, 0.95, 1.0, TAG_TRIM)
    _prism(acc, run, tmid - hw(3.2, out_s), tmid + hw(3.2, out_s), sw + 0.3, sw + 1.3, 2.45, 2.55, TAG_TRIM)
    _prism(acc, run, tmid - hw(2.6, out_s), tmid + hw(2.6, out_s), sw + 0.3, sw + 0.36, 2.6, 3.1, TAG_ACCENT)
    n = max(1, int(L / 8.0))
    for k in range(n):
        t = run.t0 + (k + 0.5) * (run.t1 - run.t0) / n
        if abs(t - tmid) * m < 3.6 or (kiosk and (t - run.t0) * m < 5.0):
            continue
        _prism(acc, run, t - hw(1.24, out_s), t + hw(1.24, out_s), sw + 0.3, sw + 0.38, 0.0, 2.27, TAG_TRIM)
        _prism(acc, run, t - hw(1.0, out_s), t + hw(1.0, out_s), sw + 0.3, sw + 0.42, 0.0, 2.1, TAG_DARK)
    nv = max(1, int(L / 6.0))
    for k in range(nv):
        t = run.t0 + (k + 0.5) * (run.t1 - run.t0) / nv
        _prism(acc, run, t - hw(0.6, out_s), t + hw(0.6, out_s), sw + 0.3, sw + 0.36, H - 1.9, H - 1.5, TAG_DARK)
    nd = max(1, int(L / 7.0))
    for k in range(nd + 1):
        t = run.t0 + (0.3 / m if k == 0 else (-0.3 / m if k == nd else 0.0)) + k * (run.t1 - run.t0) / nd
        _prism(acc, run, t - hw(0.1, out_s), t + hw(0.1, out_s), sw + 0.32, sw + 0.44, 0.0, H - 0.6, TAG_DARK)


def skin_base(H, ground):
    """Where the cladding starts on a tall facade: a glazed concourse (floor slabs showing through)
    fills about the lower 40%, the skin the rest."""
    if H <= 3 * LEVEL:
        return ground
    return max(ground, math.floor(0.4 * H / LEVEL) * LEVEL)


def _concourse(acc, run, sw, t0, t1, z0, z1, hw):
    """Concourse glazing between z0 and z1: dark glass, floor-slab edges every level, mullions."""
    out_s = sw + 0.3
    _prism(acc, run, t0, t1, sw + 0.3, sw + 0.34, z0, z1, TAG_DARK)
    for z in np.arange(z0 + LEVEL, z1 - 0.5, LEVEL):
        _prism(acc, run, t0, t1, sw + 0.3, sw + 0.7, z - 0.2, z + 0.2, TAG_TRIM)
    ts, _ = _repeat(run, out_s, 1.6, t0, t1, 0.0)
    for t in ts + [t1]:
        _prism(acc, run, t - hw(0.08, out_s), t + hw(0.08, out_s), sw + 0.3, sw + 0.45, z0, z1, TAG_TRIM)


def _core(acc, run, tm, sw, H, hw):
    """A stair tower: a solid core with a full-height glazed stair window, its landings showing as
    bands across the glass, and a cap."""
    s1 = sw + 6.0
    _prism(acc, run, tm - hw(7.0, s1), tm + hw(7.0, s1), sw, s1, 0.0, H + 1.5, TAG_CORE)
    _prism(acc, run, tm - hw(3.0, s1), tm + hw(3.0, s1), s1, s1 + 0.03, 0.5, H + 1.0, TAG_DARK)
    for z in np.arange(3.5, H + 1.0, 3.5):
        _prism(acc, run, tm - hw(3.2, s1), tm + hw(3.2, s1), s1, s1 + 0.12, z - 0.12, z + 0.12, TAG_TRIM)
    _prism(acc, run, tm - hw(7.6, s1), tm + hw(7.6, s1), sw, s1 + 0.3, H + 1.5, H + 1.8, TAG_TRIM)


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
    # a coping along the raking top and a plinth, both standing a little proud of the wall
    e = 0.06 / run.metric(0.0)
    upper = list(map(tuple, poly[len(lo) - 1:])) + [tuple(poly[0])]
    for a, b in zip(upper[:-1], upper[1:]):
        if abs(b[0] - a[0]) < 0.05:
            continue
        band = np.array([a, b, (b[0], b[1] + 0.15), (a[0], a[1] + 0.15)])
        acc.add_prism(run, ta - e, tb + e, (band, [TAG_TRIM] * 4), cap_tag=TAG_TRIM)
    s_lo = float(poly[0][0])
    acc.add_prism(run, ta - e, tb + e, (np.array([(s_lo, 0.0), (sw + 0.3, 0.0), (sw + 0.3, 0.6), (s_lo, 0.6)]),
                                         [TAG_TRIM] * 4), cap_tag=TAG_TRIM)


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
