"""The land round and between the plots (mirrors viewer/gen/src/site/landscape.rs): the fields or
river outside, the site lawn, driveways and footpaths (with a stub to each plot that needs one),
the boundary (fence, railings, wall or hedge, with tree or berm screening), the gate, and trees.
"""
import math

from ..stands.prng import uniform
from . import buildings, kernel, mats
from .kernel import along, ccw, dist_to_polyline, inside, seg_hit, strip
from .model import COVER_Z, FIELD_Z, LAWN_Z, PATH_Z, ROAD_Z, Placement, add, length, mul, normalize, num, sub, text

ROAD_W = 7.0
PATH_W = 3.0
GATE_W = 12.0
TAU = 2 * math.pi


def pts(v):
    return [(float(p[0]), float(p[1])) for p in v] if isinstance(v, list) else []


def lines(v):
    return [pts(x) for x in v] if isinstance(v, list) else []


def _along(path, spacing, offset):
    """kernel.along as (point, direction) pairs."""
    return [((x, y), (dx, dy)) for x, y, dx, dy in along(path, spacing, offset)]


def yaw_of(d):
    return math.atan2(-d[0], d[1])


def offset_line(path, off, closed):
    """Offsets of a polyline (mitred like kernel.strip): one point per vertex `off` metres to its left."""
    n = len(path)

    def seg(i):
        a, b = path[i % n], path[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy)
        return (-dy / ln, dx / ln) if ln > 1e-12 else (0.0, 1.0)
    out = []
    for i in range(n):
        if closed:
            prev, nxt = seg((i + n - 1) % n), seg(i)
        elif i == 0:
            prev, nxt = seg(0), seg(0)
        elif i == n - 1:
            prev, nxt = seg(n - 2), seg(n - 2)
        else:
            prev, nxt = seg(i - 1), seg(i)
        mx, my = prev[0] + nxt[0], prev[1] + nxt[1]
        ml = math.hypot(mx, my)
        if ml > 1e-12:
            mx /= ml
            my /= ml
        else:
            mx, my = nxt
        k = max(mx * nxt[0] + my * nxt[1], 0.3)
        out.append((path[i][0] + mx / k * off, path[i][1] + my / k * off))
    return out


def ribbon(m, path, a0, a1, tag, closed):
    """A ribbon between two offsets (o, z) of a polyline (banks, slopes), UVs in metres."""
    (o0, z0), (o1, z1) = a0, a1
    a = offset_line(path, o0, closed)
    b = offset_line(path, o1, closed)
    n = len(path)
    segs = n if closed else n - 1
    s = 0.0
    for i in range(segs):
        j = (i + 1) % n
        ln = length(sub(path[j], path[i]))
        w = math.hypot(length(sub(b[i], a[i])), z1 - z0)
        base = m.add_verts([(a[i][0], a[i][1], z0), (a[j][0], a[j][1], z0), (b[j][0], b[j][1], z1), (b[i][0], b[i][1], z1)])
        if o1 > o0:
            m.add_face([base, base + 1, base + 2, base + 3], tag, [(s, 0.0), (s + ln, 0.0), (s + ln, w), (s, w)])
        else:
            m.add_face([base + 3, base + 2, base + 1, base], tag, [(s, w), (s + ln, w), (s + ln, 0.0), (s, 0.0)])
        s += ln


def _bounds(boundary):
    lo = (min(p[0] for p in boundary), min(p[1] for p in boundary))
    hi = (max(p[0] for p in boundary), max(p[1] for p in boundary))
    return lo, hi


def context(o, tpl, boundary, seed):
    """Fields (hedgerows and trees) or a river meadow round the site, on a big plane of farm grass."""
    lo, hi = _bounds(boundary)
    e = 340.0
    m = kernel.Mesh()
    kernel.ground(m, kernel.rect(hi[0] - lo[0] + 2.0 * e, hi[1] - lo[1] + 2.0 * e, (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0), FIELD_Z, 0)
    o.part("World:fields", m, [mats.lawn("field", (0.0, 0.0), 0.0)])
    river = pts(tpl.get("river"))
    # the fields: boundaries every 110-240 m each way, kept 25 m clear of the site
    E = 300.0
    keep_out = ((lo[0] - 25.0, lo[1] - 25.0), (hi[0] + 25.0, hi[1] + 25.0))

    def clear(p):
        return not (keep_out[0][0] < p[0] < keep_out[1][0] and keep_out[0][1] < p[1] < keep_out[1][1])

    def cuts(a, b, salt):
        out, v, i = [], a + 40.0 * uniform(seed + salt, 0), 0
        while v < b:
            out.append(v)
            i += 1
            v += 110.0 + 130.0 * uniform(seed + salt, i)
        return out
    xs = cuts(lo[0] - E, hi[0] + E, 60)
    ys = cuts(lo[1] - E, hi[1] + E, 61)
    # each field is pasture (the plane), a hay meadow, stubble or ploughland
    covers = {"hay": kernel.Mesh(), "stubble": kernel.Mesh(), "plough": kernel.Mesh()}
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            c = j * 64 + i
            x0, x1, y0, y1 = xs[i] + 2.0, xs[i + 1] - 2.0, ys[j] + 2.0, ys[j + 1] - 2.0
            if x1 < keep_out[0][0] or x0 > keep_out[1][0] or y1 < keep_out[0][1] or y0 > keep_out[1][1]:
                r = uniform(seed + 62, c)
                kind = "hay" if 0.45 <= r < 0.65 else "stubble" if 0.65 <= r < 0.82 else "plough" if r >= 0.82 else None
                if kind:
                    kernel.ground(covers[kind], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], FIELD_Z + 0.006, 0)
                # a copse in some field corners: oak and birch with an understorey
                if uniform(seed + 63, c) < 0.16:
                    cx = x0 + 22.0 if uniform(seed + 64, c) < 0.5 else x1 - 22.0
                    cy = y0 + 22.0 if uniform(seed + 65, c) < 0.5 else y1 - 22.0
                    for t in range(9):
                        a = uniform(seed + 66, c * 16 + t) * 6.283
                        rr = 16.0 * math.sqrt(uniform(seed + 67, c * 16 + t))
                        q = (cx + rr * math.cos(a), cy + rr * math.sin(a))
                        u = uniform(seed + 68, c * 16 + t)
                        key = "prop.tree_oak" if u < 0.5 else "prop.tree_birch" if u < 0.72 else "prop.shrub"
                        if not river or dist_to_polyline(q, river) > 30.0:
                            tree(o, key, q, FIELD_Z, seed + 69, c * 16 + t)
    for kind, m2 in covers.items():
        if m2.faces:
            o.part(f"World:fields_{kind}", m2, [mats.named("Site_Dirt") if kind == "plough" else mats.lawn(kind, (0.0, 0.0), 0.0)])
    # hedgerows along every boundary, a little crooked, with oak standards in runs
    hedges = []
    for n, x in enumerate(xs):
        hedges.append([(x + 5.0 * (uniform(seed + 70, n * 16 + q) - 0.5), lo[1] - E + q * 100.0) for q in range(int((hi[1] - lo[1] + 2.0 * E) / 100.0) + 1)])
    for n, y in enumerate(ys):
        hedges.append([(lo[0] - E + q * 100.0, y + 5.0 * (uniform(seed + 71, n * 16 + q) - 0.5)) for q in range(int((hi[0] - lo[0] + 2.0 * E) / 100.0) + 1)])
    k = 0
    for h in hedges:
        for p, d in _along(h, 2.2, 0.5):
            k += 1
            if uniform(seed + 7, k // 9) < 0.08 or not clear(p) or roads_near(tpl, p, 9.0) or (river and dist_to_polyline(p, river) < 26.0):
                continue
            s = 0.85 + 0.35 * uniform(seed + 8, k)
            # a hedge segment runs along its local x (its front faces across the row)
            pl = Placement((p[0], p[1], FIELD_Z), yaw_of(d) - math.pi / 2 + (uniform(seed + 9, k) - 0.5) * 0.3, scale=(1.0, 1.0, s))
            o.prop("prop.hedge", pl)
            run = 0.11 if uniform(seed + 72, k // 24) < 0.35 else 0.012
            if uniform(seed + 10, k) < run:
                key = "prop.tree_oak" if uniform(seed + 11, k) < 0.8 else "prop.tree_birch"
                tree(o, key, p, FIELD_Z, seed + 12, k)
    if river:
        w = kernel.Mesh()
        strip(w, river, 30.0, FIELD_Z + 0.02, 0)
        o.part("World:river", w, [mats.named("Site_Water")])
        b = kernel.Mesh()
        ribbon(b, river, (15.0, FIELD_Z + 0.02), (21.0, FIELD_Z + 0.05), 0, False)
        ribbon(b, river, (-21.0, FIELD_Z + 0.05), (-15.0, FIELD_Z + 0.02), 0, False)
        o.part("World:riverbanks", b, [mats.named("Site_Dirt")])
        for j, (p, d) in enumerate(_along(river, 3.0, 0.5)):
            for side in (-1.0, 1.0):
                n = (-d[1] * side, d[0] * side)
                sk = j * 2 + (1 if side > 0.0 else 0)
                if uniform(seed + 13, sk) < 0.55:
                    q = add(p, mul(n, 14.5 + 3.0 * uniform(seed + 14, j)))
                    o.prop("prop.reeds", Placement((q[0], q[1], FIELD_Z + 0.02), uniform(seed + 15, j) * 6.283))
                if uniform(seed + 16, sk) < 0.09:
                    q = add(p, mul(n, 25.0 + 8.0 * uniform(seed + 17, j)))
                    tree(o, "prop.tree_willow" if uniform(seed + 18, j) < 0.5 else "prop.tree_oak", q, FIELD_Z, seed + 19, j)


def roads_near(tpl, p, d):
    return any(dist_to_polyline(p, r) < d for r in lines(tpl.get("roads")))


def tree(o, key, p, z, seed, k):
    """A tree prop with variation in size and turn, and its own proportions: spread and height
    scale separately (about 12 % each way), so no two crowns of a species share a silhouette."""
    s = 0.8 + 0.45 * uniform(seed, k)
    sx, sy, sz = (s * (0.88 + 0.24 * uniform(seed + 501 + i, k)) for i in range(3))
    o.prop(key, Placement((p[0], p[1], z), uniform(seed + 1, k) * TAU, scale=(sx, sy, sz)))


def site_ground(o, boundary):
    m = kernel.Mesh()
    kernel.ground(m, boundary, LAWN_Z, 0)
    o.part("Site:lawn", m, [mats.lawn("amenity", (0.0, 0.0), 0.0)])


def stub_of(p):
    """Where a plot's access stub starts (site xy), its direction, its width (0 = none), worn?"""
    st = p.state()
    fac = text(p.cfg, "facility", "")
    occ = text(p.cfg, "occupant", "")
    start = p.frame.pt(p.front[0] * p.w / 2.0, p.front[1] * p.d / 2.0)
    d = p.frame.vec(p.front[0], p.front[1])
    if st in ("built", "construction"):
        w, old = (6.0 if fac in ("car_park", "drop_off") else 0.0 if fac in ("pitch", "artificial_pitch", "youth_pitches", "showpiece_pitch") else PATH_W), False
    elif st in ("occupied", "demolition"):
        w = 5.0 if occ in ("old_car_park", "container_yard", "old_warehouse", "farm_barns", "derelict_mill", "abandoned_offices") else 0.0
        old = True
    else:
        w, old = 0.0, False
    return start, d, w, old


def roads(o, tpl, plots):
    """Driveways (kerbs, centre line), footpaths and the stubs to the plots."""
    rds = lines(tpl.get("roads"))
    paths = lines(tpl.get("paths"))
    asphalt, kerb, paint = kernel.Mesh(), kernel.Mesh(), kernel.Mesh()
    for r in rds:
        strip(asphalt, r, ROAD_W, ROAD_Z, 0)
        for side in (-1.0, 1.0):
            strip(kerb, offset_line(r, side * (ROAD_W / 2.0 + 0.1), False), 0.22, ROAD_Z + 0.09, 0)
        for p, d in _along(r, 9.0, 0.5):
            strip(paint, [sub(p, mul(d, 1.8)), add(p, mul(d, 1.8))], 0.12, ROAD_Z + 0.006, 0)
    path = kernel.Mesh()
    for p in paths:
        strip(path, p, PATH_W, PATH_Z, 0)
    old = kernel.Mesh()
    for p in plots:
        start, d, w, worn = stub_of(p)
        if w <= 0.0:
            continue
        best = None
        for ln, half in [(r, ROAD_W / 2.0) for r in rds] + [(r, PATH_W / 2.0) for r in paths]:
            for i in range(len(ln) - 1):
                t = seg_hit(start, d, ln[i], ln[i + 1])
                if t is not None and t < 90.0 and (best is None or t < best[0]):
                    best = (t, half)
        if best is not None:
            t, half = best
            line = [sub(start, mul(d, 0.5)), add(start, mul(d, t + half * 0.5))]
            if worn:
                strip(old, line, w, PATH_Z - 0.003, 0)
            elif w > PATH_W:
                strip(asphalt, line, w, ROAD_Z - 0.004, 0)
            else:
                strip(path, line, w, PATH_Z, 0)
    o.part("Site:roads", asphalt, [mats.named("Site_Asphalt")])
    o.part("Site:kerbs", kerb, [mats.named("Site_Kerb")])
    o.part("Site:road_paint", paint, [mats.tint("TINT_road_paint", "#ecebe4")])
    o.part("Site:paths", path, [mats.named("Site_Asphalt")])
    o.part("Site:old_tracks", old, [mats.named("Site_Asphalt_Old")])


def fence_line(boundary, gate):
    """The boundary as an open polyline that starts and ends either side of the gate opening."""
    b = ccw(boundary)
    n = len(b)
    best = (math.inf, 0)
    for i in range(n):
        d = dist_to_polyline(gate, [b[i], b[(i + 1) % n]])
        if d < best[0]:
            best = (d, i)
    i = best[1]
    a, c = b[i], b[(i + 1) % n]
    dr = normalize(sub(c, a))
    out = [add(gate, mul(dr, GATE_W / 2.0))]
    for k in range(1, n + 1):
        out.append(b[(i + k) % n])
    out.append(sub(gate, mul(dr, GATE_W / 2.0)))
    return out


def boundary(o, tpl, bnd, cfg, seed):
    gate = (num(tpl["gate"], "x", 0.0), num(tpl["gate"], "y", 0.0))
    line = fence_line(bnd, gate)
    kind = text(cfg, "type", "chainlink")
    if kind == "hedge":
        for k, (p, d) in enumerate(_along(line, 2.0, 0.5)):
            s = 0.9 + 0.25 * uniform(seed + 21, k)
            o.prop("prop.hedge", Placement((p[0], p[1], LAWN_Z), yaw_of(d) - math.pi / 2 + (uniform(seed + 22, k) - 0.5) * 0.2, scale=(1.0, 1.0, s)))
        # a stock fence inside the hedge, as farms do
        chainlink(o, offset_line(line, 0.9, False), 1.2, "Site:boundary")
    elif kind == "brick_wall":
        m = kernel.Mesh()
        for i in range(len(line) - 1):
            a, b = line[i], line[i + 1]
            d = sub(b, a)
            ln = length(d)
            if ln < 0.1:
                continue
            c = mul(add(a, b), 0.5)
            yaw = math.atan2(d[1], d[0])
            kernel.obox(m, c[0], c[1], LAWN_Z, ln, 0.34, 2.05, yaw, 0)
            kernel.obox(m, c[0], c[1], LAWN_Z + 2.05, ln + 0.1, 0.44, 0.09, yaw, 1)
        for p, d in _along(line, 3.6, 0.0):
            yaw = math.atan2(d[1], d[0])
            kernel.obox(m, p[0], p[1], LAWN_Z, 0.56, 0.56, 2.25, yaw, 0)
            kernel.obox(m, p[0], p[1], LAWN_Z + 2.25, 0.66, 0.66, 0.1, yaw, 1)
        o.part("Site:boundary", m, [mats.named("Facade_Brick"), mats.named("Facade_Stone")])
    elif kind == "palisade":
        m = kernel.Mesh()
        rails = [([(p[0], p[1], LAWN_Z + z) for p in line], 0.035) for z in (0.45, 2.0)]
        for p, d in _along(line, 2.75, 0.0):
            kernel.obox(m, p[0], p[1], LAWN_Z, 0.1, 0.1, 2.55, math.atan2(d[1], d[0]), 0)
        for k, (p, d) in enumerate(_along(line, 0.16, 0.5)):
            q = add(p, mul((-d[1], d[0]), 0.06))
            kernel.obox(m, q[0], q[1], LAWN_Z, 0.068, 0.022, 2.4 + 0.04 * (k % 2), math.atan2(d[1], d[0]), 0)
        o.part("Site:boundary", m, [mats.tint("TINT_fence", "#2b4433")])
        o.tubes("Site:boundary_rails", rails, mats.tint("TINT_fence", "#2b4433"), 6)
    elif kind == "railings":
        bars = [([(p[0], p[1], LAWN_Z + z) for p in line], 0.02) for z in (0.12, 1.65)]
        for p, _ in _along(line, 0.13, 0.5):
            bars.append(([(p[0], p[1], LAWN_Z), (p[0], p[1], LAWN_Z + 1.8)], 0.011))
        m = kernel.Mesh()
        for p, d in _along(line, 2.6, 0.0):
            kernel.obox(m, p[0], p[1], LAWN_Z, 0.08, 0.08, 1.9, math.atan2(d[1], d[0]), 0)
        o.part("Site:boundary", m, [mats.tint("TINT_fence", "#17191a")])
        o.tubes("Site:boundary_rails", bars, mats.tint("TINT_fence", "#17191a"), 5)
    else:
        chainlink(o, line, 2.4, "Site:boundary")
    screening = text(cfg, "screening", "none")
    if screening == "trees":
        # a planted belt, not a row: irregular spacing, clumps, mixed species and an understorey
        inner = offset_line(line, 5.5, False)
        for k, (p, d) in enumerate(_along(inner, 4.5, 0.5)):
            if length(sub(p, gate)) < 14.0 or roads_near(tpl, p, 6.0):
                continue
            dense = uniform(seed + 30, k // 5) < 0.6
            if uniform(seed + 31, k) > (0.62 if dense else 0.2):
                continue
            r = uniform(seed + 32, k)
            key = "prop.tree_conifer" if r < 0.3 else "prop.tree_birch" if r < 0.55 else "prop.tree_oak" if r < 0.75 else "prop.shrub"
            off = (uniform(seed + 33, k) - 0.5) * 5.0
            q = add(p, mul((-d[1], d[0]), off))
            tree(o, key, q, LAWN_Z, seed + 34, k)
    elif screening == "berm":
        m = kernel.Mesh()
        ribbon(m, line, (0.6, LAWN_Z), (4.5, 2.4), 0, False)
        ribbon(m, line, (4.5, 2.4), (8.6, LAWN_Z), 0, False)
        o.part("Site:berm", m, [mats.lawn("berm", (0.0, 0.0), 0.0)])
        for k, (p, _) in enumerate(_along(offset_line(line, 4.5, False), 11.0, 0.5)):
            if length(sub(p, gate)) < 16.0 or roads_near(tpl, p, 8.0) or uniform(seed + 35, k) < 0.4:
                continue
            key = "prop.shrub" if uniform(seed + 36, k) < 0.5 else "prop.tree_birch"
            tree(o, key, p, 2.3, seed + 37, k)


def chainlink(o, line, h, name):
    """Chain-link fence along a polyline: posts every 3 m, top rail, mesh panels."""
    panels = kernel.Mesh()
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        if length(sub(b, a)) < 0.05:
            continue
        kernel.quad(panels, (a[0], a[1], LAWN_Z + 0.03), (b[0], b[1], LAWN_Z + 0.03), (b[0], b[1], LAWN_Z + h), (a[0], a[1], LAWN_Z + h), 0)
    posts = [([(p[0], p[1], LAWN_Z + h - 0.03) for p in line], 0.022)]
    for p, _ in _along(line, 3.0, 0.0):
        posts.append(([(p[0], p[1], LAWN_Z), (p[0], p[1], LAWN_Z + h + 0.05)], 0.032))
    o.part(name, panels, [mats.chainlink()])
    o.tubes(f"{name}_posts", posts, mats.tint("TINT_fence", "#2b4433"), 6)


def gate(o, cat, tpl, cfg, _seed):
    """Farm gate, barrier and hut, or a gatehouse in the site's style."""
    g = (num(tpl["gate"], "x", 0.0), num(tpl["gate"], "y", 0.0))
    rds = lines(tpl.get("roads"))
    r = rds[0] if rds else []
    d = (0.0, 1.0)
    for i in range(max(len(r) - 1, 0)):
        if dist_to_polyline(g, [r[i], r[i + 1]]) < 1.0:
            d = normalize(sub(r[i + 1], r[i]))
    side = (d[1], -d[0])                      # to the right going in
    f = kernel.Frame(g[0], g[1], math.degrees(math.atan2(-d[0], d[1])))
    tier = int(num(cfg, "tier", 1.0))
    club = o.club
    m = kernel.Mesh()
    for s in (-1.0, 1.0):
        p = add(g, mul(side, s * (GATE_W / 2.0 + 0.2)))
        kernel.obox(m, p[0], p[1], LAWN_Z, 0.35, 0.35, 2.6 if tier >= 3 else 1.6, 0.0, 0)
    o.part("Gate:posts", m, [mats.named("Facade_Brick" if tier >= 3 else "Galvanised")])
    if tier == 1:
        for s in (-1.0, 1.0):
            o.prop_at(f, "prop.gate_farm", s * GATE_W / 2.0, 0.0, LAWN_Z, 0.0)
        o.prop_at(f, "prop.sign_post", GATE_W / 2.0 + 2.5, -3.0, LAWN_Z, 0.0, [("TINT_club", club.primary)])
    elif tier == 2:
        # one barrier on the way in (the left-hand lane): cabinet on the verge, 4 m boom to the centre line
        o.prop_at(f, "prop.barrier_arm", -ROAD_W / 2.0 - 0.45, 9.0, LAWN_Z, 0.0)
        o.prop_at(f, "prop.security_hut", -ROAD_W / 2.0 - 3.2, 10.0, COVER_Z, math.pi / 2, [("TINT_cabin", "#d9dbd6")])
        o.prop_at(f, "prop.sign_post", GATE_W / 2.0 + 2.5, -3.0, LAWN_Z, 0.0, [("TINT_club", club.primary)])
        o.prop_at(f, "prop.lamp_post", ROAD_W / 2.0 + 1.0, 4.0, LAWN_Z, math.pi / 2)
        o.prop_at(f, "prop.cctv_pole", -ROAD_W / 2.0 - 1.5, 3.0, LAWN_Z, math.pi)
    else:
        buildings.gatehouse(o, cat, f, text(cfg, "style", "classic"))
        # a barrier per lane, cabinets on the verges, booms meeting at the centre line
        o.prop_at(f, "prop.barrier_arm", -ROAD_W / 2.0 - 0.45, 10.0, LAWN_Z, 0.0)
        o.prop_at(f, "prop.barrier_arm", ROAD_W / 2.0 + 0.45, 14.0, LAWN_Z, math.pi)
        for s in (-1.0, 1.0):
            o.prop_at(f, "prop.planter", s * (GATE_W / 2.0 + 2.2), 1.5, LAWN_Z, 0.0)
            o.prop_at(f, "prop.lamp_post", s * (ROAD_W / 2.0 + 0.9), 20.0, LAWN_Z, s * math.pi / 2)
        o.prop_at(f, "prop.cctv_pole", ROAD_W / 2.0 + 1.6, 2.0, LAWN_Z, 0.0)
    o.socket_at(f, "gate", "main", 0.0, 0.0, 0.0, (0.0, 1.0), [("tier", float(tier))])


def trees(o, tpl, bnd, plots, density, seed):
    """Trees on the lawn between plots, roads and paths."""
    if density <= 0.0:
        return
    rds = lines(tpl.get("roads"))
    paths = lines(tpl.get("paths"))
    lo, hi = _bounds(bnd)
    rects = [[p.frame.pt(q[0], q[1]) for q in kernel.rect(p.w + 6.0, p.d + 6.0)] for p in plots]
    b = ccw(bnd)
    ring = b + [b[0]]
    step = 15.0
    nx, ny = int((hi[0] - lo[0]) / step), int((hi[1] - lo[1]) / step)
    k = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            k += 1
            p = (lo[0] + (i + uniform(seed + 41, k)) * step, lo[1] + (j + uniform(seed + 42, k)) * step)
            if uniform(seed + 43, k) > density * 0.6 or not inside(b, p[0], p[1]) or dist_to_polyline(p, ring) < 5.0:
                continue
            if any(inside(r, p[0], p[1]) for r in rects) or any(dist_to_polyline(p, r) < 8.0 for r in rds) \
                    or any(dist_to_polyline(p, r) < 4.5 for r in paths):
                continue
            r = uniform(seed + 44, k)
            key = "prop.tree_oak" if r < 0.4 else "prop.tree_birch" if r < 0.7 else "prop.tree_conifer" if r < 0.85 else "prop.shrub"
            tree(o, key, p, LAWN_Z, seed + 45, k)
