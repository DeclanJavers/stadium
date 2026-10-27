"""What stands on a plot before the club builds on it (mirrors viewer/gen/src/site/occupants.rs):
an old car park, a warehouse, a mill, offices, a container yard, farm barns, scrub, old tennis
courts, a bowling green, a rubble lot. Each fills its plot, varies with the plot's seed, and has a
half-demolished form for the demolition state.
"""
import math

from ..stands.prng import uniform
from . import buildings, facilities, kernel, landscape, mats
from .kernel import box as bx, cylinder, obox, quad, strip
from .model import COVER_Z, add, dot, length, mul, normalize, rround, sub

TAU = 2 * math.pi
CAR_COLORS = ("#1d1f22", "#e8e8e4", "#8c9096", "#2b3f66", "#7a1d22", "#3a4a3c", "#c9c4b8", "#4a4f57", "#b5b9bd", "#20304a")
DULL = ("#5b5148", "#6d6a62", "#4f5650", "#7a6e5a", "#5c4a3f", "#6b7072")
CONTAINERS = ("#7a3b2a", "#2f4f7a", "#3c5a3a", "#707479", "#b0592b", "#8a2a24")


def cover(o, p, name, inset, mat):
    """The plot's ground cover (inset from its edge), at COVER_Z."""
    m = kernel.Mesh()
    kernel.ground(m, kernel.rect(p.w - 2.0 * inset, p.d - 2.0 * inset), COVER_Z, 0)
    o.part_at(p.frame, f"{p.id}:{name}", m, [mat])


def mound(m, cx, cy, rx, ry, h, seed, tag):
    """A heap (rubble, earth, sand) as material settles: a slumped cone with a spread toe and a
    rounded crest, lumpy in broad lobes rather than vertex by vertex."""
    n = 24
    rings = ((1.0, 0.0), (0.86, 0.17), (0.7, 0.42), (0.52, 0.66), (0.34, 0.85), (0.16, 0.97))
    ph = [uniform(seed + 2, q) * TAU for q in range(4)]
    idx = []
    for ri, (rr, zz) in enumerate(rings):
        ring = []
        for k in range(n):
            a = TAU * k / n
            j = 1.0 + 0.12 * math.sin(2.0 * a + ph[0]) + 0.07 * math.sin(3.0 * a + ph[1]) + 0.05 * (uniform(seed, ri * n + k) - 0.5)
            lobe = 1.0 + 0.1 * math.sin(2.0 * a + ph[2]) + 0.06 * math.sin(5.0 * a + ph[3])
            z = COVER_Z if ri == 0 else COVER_Z + h * zz * lobe * (0.95 + 0.1 * uniform(seed + 1, ri * n + k))
            ring.append(len(m.verts))
            m.verts.append((cx + rx * rr * j * math.cos(a), cy + ry * rr * j * math.sin(a), z))
        idx.append(ring)
    top = len(m.verts)
    m.verts.append((cx, cy, COVER_Z + h))
    for r in range(len(rings) - 1):
        for k in range(n):
            f = [idx[r][k], idx[r][(k + 1) % n], idx[r + 1][(k + 1) % n], idx[r + 1][k]]
            m.add_face(f, tag, [(m.verts[i][0], m.verts[i][1]) for i in f])
    last = idx[-1]
    for k in range(n):
        f = [last[k], last[(k + 1) % n], top]
        m.add_face(f, tag, [(m.verts[i][0], m.verts[i][1]) for i in f])


def spot(p, salt, i, margin):
    """A scatter point inside the plot (local), keeping `margin` from the edge."""
    return ((uniform(p.seed + salt, i) - 0.5) * (p.w - 2.0 * margin), (uniform(p.seed + salt + 1, i) - 0.5) * (p.d - 2.0 * margin))


def weeds(o, p, per_m2, salt):
    for i in range(int(p.w * p.d * per_m2)):
        q = spot(p, salt, i, 1.5)
        s = 0.7 + 0.8 * uniform(p.seed + salt + 2, i)
        o.prop_full(p.frame, "prop.weeds", q[0], q[1], COVER_Z, uniform(p.seed + salt + 3, i) * TAU, 0.0, (s, s, s))


def bays(p_w, p_d, inset, paint, z, gaps):
    """Bay lines in rows across a plot (car parks old and new): the bay centres and their facing
    (local), the paint into `paint`."""
    bw, bd, aisle = 2.5, 5.0, 6.0
    out = []
    usable = p_d - 2.0 * inset
    rows = []
    y = -usable / 2.0
    up = True
    while not (y + bd > usable / 2.0 + 1e-9):
        rows.append((y, up))
        y += bd
        if up:
            y += aisle
        up = not up
    n = int(math.floor((p_w - 2.0 * inset) / bw))
    x0 = -n * bw / 2.0
    k = 0
    for y0, facing in rows:
        for i in range(n + 1):
            x = x0 + i * bw
            k += 1
            if not gaps(k):
                strip(paint, [(x, y0), (x, y0 + bd)], 0.1, z, 0)
        for i in range(n):
            out.append(((x0 + (i + 0.5) * bw, y0 + bd / 2.0), 0.0 if facing else math.pi))
    return out


def park(o, p, bays_, fill, dull, salt, reverse=True):
    """Parked cars in some of the bays (reverse: about half have backed in; off for a one-way
    lay-by, where every car faces the way the traffic goes)."""
    n = 0
    for i, (c, yaw) in enumerate(bays_):
        if uniform(p.seed + salt, i) >= fill:
            continue
        n += 1
        r = uniform(p.seed + salt + 1, i)
        key = "prop.car_hatch" if r < 0.45 else "prop.car_saloon" if r < 0.8 else "prop.car_suv"
        # one of five registrations per model, so no two cars in a row share a plate
        v = int(uniform(p.seed + salt + 7, i) * 5.0)
        key += f"_{v + 1}" if v > 0 else ""
        pal = DULL if dull else CAR_COLORS
        col = pal[int(uniform(p.seed + salt + 2, i) * len(pal)) % len(pal)]
        flip = math.pi if reverse and uniform(p.seed + salt + 3, i) < 0.45 else 0.0
        jx, jy = (uniform(p.seed + salt + 4, i) - 0.5) * 0.3, (uniform(p.seed + salt + 5, i) - 0.5) * 0.4
        o.prop_at(p.frame, key, c[0] + jx, c[1] + jy, COVER_Z, yaw + flip + (uniform(p.seed + salt + 6, i) - 0.5) * 0.06, [("TINT_car_paint", col)])
    return n


def build(o, p, occ, demolished):
    {"old_car_park": old_car_park, "old_warehouse": warehouse, "derelict_mill": mill, "abandoned_offices": offices,
     "container_yard": container_yard, "farm_barns": farm, "disused_courts": courts, "bowling_green": bowling_green,
     "rubble_lot": rubble}.get(occ, scrub)(o, p, demolished)


def old_car_park(o, p, demolished):
    cover(o, p, "ground", 1.0, mats.named("Site_Asphalt_Old"))
    paint = kernel.Mesh()
    seed = p.seed
    b = bays(p.w, p.d, 2.0, paint, COVER_Z + 0.004, lambda k: uniform(seed + 51, k) < 0.35)
    o.part_at(p.frame, f"{p.id}:paint", paint, [mats.tint("TINT_road_paint", "#a9a69c")])
    if not demolished:
        park(o, p, b, 0.1, True, 60)
        if uniform(p.seed, 70) < 0.6:
            q = spot(p, 71, 0, 6.0)
            o.prop_at(p.frame, "prop.scrap_car", q[0], q[1], COVER_Z, uniform(p.seed + 72, 0) * TAU)
    else:
        m = kernel.Mesh()
        for i in range(6):
            q = spot(p, 73, i, 8.0)
            mound(m, q[0], q[1], 3.0 + 2.0 * uniform(p.seed + 74, i), 2.5 + 2.0 * uniform(p.seed + 75, i), 1.0 + uniform(p.seed + 76, i), p.seed + 77 + i, 0)
        o.part_at(p.frame, f"{p.id}:rubble", m, [mats.named("Site_Rubble")], 75.0)
    n = int(max(math.floor(p.w / 30.0), 1.0))
    for i in range(n):
        u = -p.w / 2.0 + (i + 0.5) * p.w / n
        o.prop_full(p.frame, "prop.lamp_post", u, -p.d / 2.0 + 1.2, COVER_Z, 0.0, 0.12 if i == 0 else 0.0, (1.0, 1.0, 1.0))
    weeds(o, p, 1.0 / 60.0, 80)


def place(p, bw, bd):
    pl = facilities.place(p, bw, bd, 12.0)
    return pl.f, pl.w, pl.d


def derelict(w, d, storeys, sh, wall, wall_color, roof, roof_mat, window, seed):
    return buildings.Spec(
        w=w, d=d, storeys=storeys, storey_h=sh, roof=roof, pitch_deg=32.0, eaves=0.3, parapet=0.7, window=window, door="none",
        roller_doors=0, canopy="none", quoins=False, clock=False, fins=False, led_band=False, arcade=False, solar=False, frame=False,
        decay=0.8, sign=None, lit=0.0, seed=seed,
        mats=buildings.materials(wall, wall_color, "Facade_Concrete", roof_mat, None, "#4a4c4b", "#5f6360", False))


def warehouse(o, p, demolished):
    cover(o, p, "yard", 1.0, mats.named("Apron_Concrete"))
    f, w, d = place(p, 64.0, 42.0)
    if demolished:
        buildings.ruin(o, f, f"{p.id}:ruin", w, d, 4.5, mats.named("Bldg_Rusted"), p.seed)
        cols = []
        for i in range(int(w / 7.0)):
            x = -w / 2.0 + 3.5 + i * 7.0
            if uniform(p.seed, 300 + i) < 0.5:
                for sy in (-1.0, 1.0):
                    cols.append(([(x, sy * (d / 2.0 - 0.3), 0.0), (x, sy * (d / 2.0 - 0.3), 8.0)], 0.12))
                cols.append(([(x, -d / 2.0 + 0.3, 8.0), (x, 0.0, 9.5), (x, d / 2.0 - 0.3, 8.0)], 0.1))
        o.tubes_at(f, f"{p.id}:frames", cols, mats.named("Site_Rust"), 6)
    else:
        sp = derelict(w, d, 1, 8.4, "Bldg_Rusted", "#7c5a46", "barrel", "Roof_Cement", "strip", p.seed)
        sp.roller_doors = max(int(w / 18.0), 1)
        sp.mats[buildings.T_DOOR] = mats.tint("TINT_door", "#6b6f6a")
        sp.mats[buildings.T_BASE] = mats.named("Facade_Brick")
        buildings.build(o, f, f"{p.id}:warehouse", sp)
    front = add(mul(p.front, -0.0), mul(p.front, min(p.d, p.w) / 2.0 - 6.0))
    for i in range(4):
        side = mul((p.front[1], -p.front[0]), (i - 1.5) * 7.0 + 3.0 * (uniform(p.seed + 90, i) - 0.5))
        q = add(front, side)
        key = ("prop.pallet_stack", "prop.oil_drum", "prop.skip", "prop.tyre_stack")[i]
        tint = [("TINT_skip", "#a5562a")] if key == "prop.skip" else [("TINT_drum", "#2d4a6a")] if key == "prop.oil_drum" else []
        o.prop_at(p.frame, key, q[0], q[1], COVER_Z, uniform(p.seed + 91, i) * TAU, tint)
    weeds(o, p, 1.0 / 70.0, 92)


def mill(o, p, demolished):
    cover(o, p, "ground", 1.0, mats.named("Site_Gravel"))
    f, w, d = place(p, 62.0, 18.0)
    chimney_x = w / 2.0 + 4.0
    ch = kernel.Mesh()
    ch_h = 9.0 if demolished else 32.0
    cylinder(ch, min(chimney_x, p.w), d / 2.0 - 2.0, 0.0, ch_h, 1.6, 16, 0, True)
    cylinder(ch, min(chimney_x, p.w), d / 2.0 - 2.0, ch_h - 1.0, ch_h - 0.3, 1.85, 16, 0, True)
    o.part_at(f, f"{p.id}:chimney", ch, [mats.tint("Bldg_Brick_Old", "#6a3a2c")])
    if demolished:
        buildings.ruin(o, f, f"{p.id}:ruin", w, d, 7.0, mats.tint("Bldg_Brick_Old", "#6a3a2c"), p.seed)
    else:
        sp = derelict(w, d, 4, 3.6, "Bldg_Brick_Old", "#6a3a2c", "gable", "Roof_Slate", "punched", p.seed)
        sp.door = "timber"
        sp.mats[buildings.T_DOOR] = mats.named("Site_Plywood")
        sp.mats[buildings.T_TRIM] = mats.tint("TINT_trim", "#3b3a36")
        buildings.build(o, f, f"{p.id}:mill", sp)
    for i in range(int(p.w * p.d / 400.0)):
        q = spot(p, 95, i, 3.0)
        o.prop_at(p.frame, "prop.bramble", q[0], q[1], COVER_Z, uniform(p.seed + 96, i) * TAU)
    weeds(o, p, 1.0 / 50.0, 97)


def offices(o, p, demolished):
    cover(o, p, "ground", 1.0, mats.named("Site_Asphalt_Old"))
    f, w, d = place(p, 46.0, 15.0)
    if demolished:
        buildings.ruin(o, f, f"{p.id}:ruin", w, d, 5.0, mats.tint("Facade_Concrete", "#9a958b"), p.seed)
    else:
        sp = derelict(w, d, 3, 3.4, "Facade_Concrete", "#9a958b", "flat", "Roof_Flat", "ribbon", p.seed)
        sp.door = "glass"
        sp.canopy = "steel"
        buildings.build(o, f, f"{p.id}:offices", sp)
    weeds(o, p, 1.0 / 45.0, 98)


def _yard_ring(p, inset, opening):
    """A ring round the plot (local), starting and ending either side of an opening at the front."""
    hw, hd = p.w / 2.0 - inset, p.d / 2.0 - inset
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    mid = (p.front[0] * hw, p.front[1] * hd)
    tang = (-p.front[1], p.front[0])
    start = next((i for i, c in enumerate(corners) if dot(sub(c, mid), tang) > 0.0 and dot(sub(c, mid), p.front) > -0.1), 0)
    ring = [add(mid, mul(tang, opening))]
    for k in range(4):
        ring.append(corners[(start + k) % 4])
    ring.append(sub(mid, mul(tang, opening)))
    return ring


def container_yard(o, p, demolished):
    cover(o, p, "ground", 1.0, mats.named("Site_Gravel"))
    ring = _yard_ring(p, 1.2, 4.0)
    if not demolished:
        landscape.chainlink(o, [p.frame.pt(q[0], q[1]) for q in ring], 2.4, f"{p.id}:fence")
    long_x = p.w >= p.d
    la, lb = (p.w, p.d) if long_x else (p.d, p.w)
    rows = int(max(math.floor((lb - 14.0) / 6.0), 1.0))
    k = 0
    for r in range(rows):
        across = lb / 2.0 - 5.0 - r * 6.0
        a = -la / 2.0 + 4.0
        while a < la / 2.0 - 8.0:
            k += 1
            big = uniform(p.seed + 110, k) < 0.4
            ln = 12.19 if big else 6.06
            if a + ln > la / 2.0 - 3.0:
                break
            if uniform(p.seed + 111, k) < (0.6 if demolished else 0.18):
                a += ln + 1.0
                continue
            c = a + ln / 2.0
            u, v = (c, across) if long_x else (across, c)
            yaw = math.pi / 2 if long_x else 0.0
            col = CONTAINERS[int(uniform(p.seed + 112, k) * len(CONTAINERS)) % len(CONTAINERS)]
            key = "prop.container_40" if big else "prop.storage_container"
            o.prop_at(p.frame, key, u, v, COVER_Z, yaw, [("TINT_container", col)])
            if not demolished and uniform(p.seed + 113, k) < 0.35:
                col2 = CONTAINERS[int(uniform(p.seed + 114, k) * len(CONTAINERS)) % len(CONTAINERS)]
                o.prop_at(p.frame, key, u, v, COVER_Z + 2.59, yaw + (uniform(p.seed + 115, k) - 0.5) * 0.05, [("TINT_container", col2)])
            a += ln + 0.6 + 2.0 * uniform(p.seed + 116, k)
    m = kernel.Mesh()
    front = mul(p.front, (p.d if long_x else p.w) / 2.0 - 9.0)
    tang = (-p.front[1], p.front[0])
    for i in range(3):
        q = add(front, mul(tang, (i - 1.0) * 12.0))
        mound(m, q[0], q[1], 3.5, 2.8, 1.6 + uniform(p.seed + 117, i), p.seed + 118 + i, 0)
    o.part_at(p.frame, f"{p.id}:scrap", m, [mats.named("Site_Scrap")])
    for i in range(3):
        q = add(front, mul(tang, (i - 1.0) * 12.0 + 6.0))
        o.prop_at(p.frame, "prop.tyre_stack" if i == 1 else "prop.scrap_car", q[0], q[1], COVER_Z, uniform(p.seed + 119, i) * TAU)
    weeds(o, p, 1.0 / 90.0, 120)


def farm(o, p, demolished):
    cover(o, p, "yard", 1.0, mats.named("Site_Dirt"))
    f, w, d = place(p, 30.0, 14.0)
    fa = f.sub(-w * 0.3, 0.0, 0.0)
    if demolished:
        buildings.ruin(o, fa, f"{p.id}:ruin", w * 0.6, d, 3.0, mats.tint("TINT_timber", "#3b3029"), p.seed)
    else:
        sp = derelict(w * 0.6, d, 1, 5.5, "TINT_timber", "#3b3029", "gable", "Roof_Cement", "none", p.seed)
        sp.decay = 0.5
        sp.roller_doors = 1
        sp.mats[buildings.T_DOOR] = mats.tint("TINT_timber", "#4a3a30")
        buildings.build(o, fa, f"{p.id}:barn", sp)
    fb = f.sub(w * 0.33, 0.0, 0.0)
    bw, bd, bh = w * 0.32, d, 6.0
    cols = []
    nx = int(max(rround(bw / 4.5), 1.0))
    for i in range(nx + 1):
        x = -bw / 2.0 + bw * i / nx
        for sy in (-1.0, 1.0):
            cols.append(([(x, sy * bd / 2.0, 0.0), (x, sy * bd / 2.0, bh)], 0.1))
    o.tubes_at(fb, f"{p.id}:dutch_frame", cols, mats.named("Site_Rust"), 6)
    if not demolished:
        m = kernel.Mesh()
        n = 10
        rise = 2.6

        def z_at(y):
            q = y / (bd / 2.0)
            return bh + rise * (1.0 - q * q)
        for i in range(n):
            ya = -bd / 2.0 - 0.4 + (bd + 0.8) * i / n
            yb = -bd / 2.0 - 0.4 + (bd + 0.8) * (i + 1) / n
            quad(m, (-bw / 2.0 - 0.4, ya, z_at(ya)), (bw / 2.0 + 0.4, ya, z_at(ya)), (bw / 2.0 + 0.4, yb, z_at(yb)), (-bw / 2.0 - 0.4, yb, z_at(yb)), 0)
        o.part_at(fb, f"{p.id}:dutch_roof", m, [mats.named("Bldg_Rusted")])
        s = kernel.Mesh()
        cylinder(s, 0.0, 0.0, 0.0, 9.0, 1.6, 16, 0, True)
        o.part_at(f.sub(w / 2.0 + 3.0, d / 2.0 - 2.0, 0.0), f"{p.id}:silo", s, [mats.named("Galvanised")])
        q = mul(p.front, 8.0)
        o.prop_at(p.frame, "prop.tractor", q[0], q[1], COVER_Z, uniform(p.seed + 130, 0) * TAU, [("TINT_tractor", "#6d2a22")])
    for i in range(int(p.w * p.d / 300.0)):
        q = spot(p, 131, i, 2.0)
        o.prop_at(p.frame, "prop.bramble" if i % 3 == 0 else "prop.grass_tuft", q[0], q[1], COVER_Z, uniform(p.seed + 132, i) * TAU)


def scrub(o, p, demolished):
    cover(o, p, "rough", 0.5, mats.lawn("rough", p.frame.pt(0.0, 0.0), 0.0))
    area = p.w * p.d
    k_bramble = 600.0 if demolished else 160.0
    for i in range(int(area / k_bramble)):
        q = spot(p, 140, i, 1.5)
        s = 0.7 + 0.7 * uniform(p.seed + 141, i)
        o.prop_full(p.frame, "prop.bramble", q[0], q[1], COVER_Z, uniform(p.seed + 142, i) * TAU, 0.0, (s, s, s))
    for i in range(int(area / 45.0)):
        q = spot(p, 143, i, 1.0)
        s = 0.8 + 0.6 * uniform(p.seed + 145, i)
        o.prop_full(p.frame, "prop.grass_tuft", q[0], q[1], COVER_Z, uniform(p.seed + 144, i) * TAU, 0.0, (s, s, s))
    if not demolished:
        for i in range(int(area / 1400.0) + 1):
            q = spot(p, 146, i, 4.0)
            key = "prop.tree_dead" if uniform(p.seed + 147, i) < 0.25 else "prop.tree_birch" if uniform(p.seed + 148, i) < 0.6 else "prop.shrub"
            landscape.tree(o, key, p.frame.pt(q[0], q[1]), COVER_Z, p.seed + 149, i)
        q = spot(p, 150, 0, 5.0)
        o.prop_at(p.frame, "prop.tyre_stack", q[0], q[1], COVER_Z, 0.4)
    else:
        m = kernel.Mesh()
        for i in range(4):
            q = spot(p, 151, i, 6.0)
            mound(m, q[0], q[1], 2.5, 2.0, 1.2, p.seed + 152 + i, 0)
        o.part_at(p.frame, f"{p.id}:cuttings", m, [mats.named("Site_Dirt")])


def courts(o, p, demolished):
    cover(o, p, "rough", 0.5, mats.lawn("rough", p.frame.pt(0.0, 0.0), 0.0))
    cw, cd = 18.3, 36.6
    along_y = p.d >= cd + 2.0
    span = p.w if along_y else p.d
    n = min(max(int(math.floor((span - 4.0) / cw)), 1), 3)
    total = n * cw

    def to(a, b):
        return (a, b) if along_y else (b, a)
    surf, lines_, nets = kernel.Mesh(), kernel.Mesh(), kernel.Mesh()
    posts = []
    q0 = [to(-total / 2.0, -cd / 2.0), to(total / 2.0, -cd / 2.0), to(total / 2.0, cd / 2.0), to(-total / 2.0, cd / 2.0)]
    kernel.ground(surf, q0, COVER_Z + 0.01, 0)
    for c in range(n):
        a0 = -total / 2.0 + c * cw + cw / 2.0
        hl, hs, hd = 23.77 / 2.0, 10.97 / 2.0, 8.23 / 2.0

        def ln(a, b):
            strip(lines_, [a, b], 0.05, COVER_Z + 0.016, 0)
        for sx in (-1.0, 1.0):
            ln(to(a0 + sx * hs, -hl), to(a0 + sx * hs, hl))
            ln(to(a0 + sx * hd, -hl), to(a0 + sx * hd, hl))
            ln(to(a0 - hs, sx * hl), to(a0 + hs, sx * hl))
            ln(to(a0 - hd, sx * 6.4), to(a0 + hd, sx * 6.4))
        ln(to(a0, -6.4), to(a0, 6.4))
        pa, pb = to(a0 - hs - 0.9, 0.0), to(a0 + hs + 0.9, 0.0)
        for q in (pa, pb):
            posts.append(([(q[0], q[1], COVER_Z), (q[0], q[1], COVER_Z + 1.07)], 0.04))
        if not demolished:
            sag = 0.25 + 0.3 * uniform(p.seed + 160, c)
            mid = mul(add(pa, pb), 0.5)
            for a, b in ((pa, mid), (mid, pb)):
                za, zb = (1.0, 1.0 - sag) if a == pa else (1.0 - sag, 1.0)
                quad(nets, (a[0], a[1], COVER_Z + 0.05), (b[0], b[1], COVER_Z + 0.05), (b[0], b[1], COVER_Z + zb), (a[0], a[1], COVER_Z + za), 0)
    o.part_at(p.frame, f"{p.id}:courts", surf, [mats.tint("TINT_court", "#5e7356")])
    o.part_at(p.frame, f"{p.id}:court_lines", lines_, [mats.tint("TINT_road_paint", "#cfccc2")])
    o.part_at(p.frame, f"{p.id}:court_nets", nets, [mats.net("square", 45, 3.0, "#26282a")])
    o.tubes_at(p.frame, f"{p.id}:court_posts", posts, mats.named("Site_Rust"), 6)
    if not demolished:
        fl = [q0[0], q0[1], q0[2], q0[3], q0[0]]
        segs = []
        for i in range(4):
            a, b = fl[i], fl[i + 1]
            n2 = int(max(rround(length(sub(b, a)) / 3.0), 1.0))
            for j in range(n2):
                if uniform(p.seed + 161, i * 100 + j) < 0.15:
                    continue
                segs.append([add(a, mul(sub(b, a), j / n2)), add(a, mul(sub(b, a), (j + 1) / n2))])
        for j, s in enumerate(segs):
            landscape.chainlink(o, [p.frame.pt(q[0], q[1]) for q in s], 2.75, f"{p.id}:fence{j}")
    weeds(o, p, 1.0 / 40.0, 162)


def bowling_green(o, p, demolished):
    cover(o, p, "rough", 0.5, mats.lawn("rough", p.frame.pt(0.0, 0.0), 0.0))
    g = min(min(p.w, p.d) - 10.0, 34.0)
    c = (0.0, -2.0)
    m = kernel.Mesh()
    kernel.ground(m, kernel.rect(g, g, c[0], c[1]), COVER_Z + 0.03, 0)
    o.part_at(p.frame, f"{p.id}:green", m, [mats.lawn("fine", p.frame.pt(c[0], c[1]), p.frame.yaw)])
    dm, bm = kernel.Mesh(), kernel.Mesh()
    r = g / 2.0
    for a0, a1 in (((-r, -r), (r, -r)), ((r, -r), (r, r)), ((r, r), (-r, r)), ((-r, r), (-r, -r))):
        n = normalize(sub(a1, a0))
        out = (n[1], -n[0])
        strip(dm, [add(add(c, a0), mul(out, 0.15)), add(add(c, a1), mul(out, 0.15))], 0.3, COVER_Z + 0.01, 0)
        p0, p1 = add(add(c, a0), mul(out, 0.4)), add(add(c, a1), mul(out, 0.4))
        mid = mul(add(p0, p1), 0.5)
        obox(bm, mid[0], mid[1], COVER_Z, length(sub(p1, p0)) + 0.8, 0.12, 0.3, math.atan2(n[1], n[0]), 0)
    o.part_at(p.frame, f"{p.id}:ditch", dm, [mats.named("Site_Gravel")])
    o.part_at(p.frame, f"{p.id}:bank", bm, [mats.named("Site_Wood")])
    pf = p.frame.sub(0.0, p.d / 2.0 - 5.0, math.pi)
    if demolished:
        buildings.ruin(o, pf, f"{p.id}:ruin", 10.0, 5.5, 2.4, mats.tint("TINT_timber", "#5b4b3c"), p.seed)
    else:
        sp = derelict(10.0, 5.5, 1, 3.0, "TINT_timber", "#5b4b3c", "gable", "Roof_Slate", "large", p.seed)
        sp.decay = 0.5
        sp.door = "timber"
        sp.canopy = "timber"
        sp.mats[buildings.T_DOOR] = mats.tint("TINT_door", "#3f5a3f")
        buildings.build(o, pf, f"{p.id}:pavilion", sp)
        for i in range(4):
            o.prop_at(p.frame, "prop.bench", (i - 1.5) * 6.0, c[1] - r - 1.6, COVER_Z, 0.0)
    weeds(o, p, 1.0 / 120.0, 170)


def rubble(o, p, demolished):
    cover(o, p, "ground", 0.5, mats.named("Site_Rubble"))
    m = kernel.Mesh()
    n = min(max(int(p.w * p.d / 350.0), 3), 18) // (2 if demolished else 1)
    for i in range(n):
        q = spot(p, 180, i, 5.0)
        mound(m, q[0], q[1], 2.0 + 3.0 * uniform(p.seed + 181, i), 2.0 + 2.5 * uniform(p.seed + 182, i), 0.8 + 1.4 * uniform(p.seed + 183, i),
              p.seed + 184 + i, 0)
    o.part_at(p.frame, f"{p.id}:heaps", m, [mats.named("Site_Rubble")], 75.0)
    slabs = kernel.Mesh()
    rebar = []
    for i in range(n * 2):
        q = spot(p, 185, i, 4.0)
        obox(slabs, q[0], q[1], COVER_Z, 1.5 + 2.0 * uniform(p.seed + 186, i), 1.0 + 1.5 * uniform(p.seed + 187, i), 0.25,
             uniform(p.seed + 188, i) * math.pi, 0)
        if i % 3 == 0:
            rebar.append(([(q[0], q[1], COVER_Z + 0.2), (q[0] + 0.4, q[1] + 0.2, COVER_Z + 1.3)], 0.012))
    o.part_at(p.frame, f"{p.id}:slabs", slabs, [mats.named("Facade_Concrete")])
    o.tubes_at(p.frame, f"{p.id}:rebar", rebar, mats.named("Site_Rust"), 5)
    weeds(o, p, 1.0 / 35.0, 189)


_unused = (bx,)
