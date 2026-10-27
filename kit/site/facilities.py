"""Facilities by tier (mirrors viewer/gen/src/site/facilities.rs): training pitches (rough grass to
match replica, with ball-stop nets, floodlights, shelters, a filming tower and a viewing deck),
artificial pitches, goalkeeper areas, the training building (portacabins to a flagship), indoor
halls, recovery, car parks and grounds maintenance.
"""
import math

from ..stands import floodlights as FL
from ..stands.prng import uniform
from . import buildings, kernel, landscape, mats
from .kernel import box as bx, cylinder, quad, strip
from .model import (COVER_Z, LAWN_Z, Placement, SitePitch, add, dot, mul, num, option, option_bool, rround,
                    sub, text, tier_spec)
from .occupants import bays, cover, mound, park

TAU = 2 * math.pi
HALF_PI = math.pi / 2


def build(o, cat, p, fac, tier, style):
    opts = p.cfg.get("options") if isinstance(p.cfg.get("options"), dict) else {}
    if fac == "pitch":
        pitch(o, cat, p, tier, opts)
    elif fac == "youth_pitches":
        youth_pitches(o, cat, p, tier, opts)
    elif fac == "artificial_pitch":
        artificial(o, cat, p, tier, opts)
    elif fac == "cages":
        cages(o, cat, p, tier, opts)
    elif fac == "showpiece_pitch":
        showpiece(o, cat, p, tier, opts)
    elif fac == "goalkeeper_area":
        goalkeeper(o, cat, p, tier)
    elif fac in ("training_building", "academy_building"):
        training_building(o, cat, p, tier, style, opts, fac)
    elif fac == "indoor_hall":
        hall(o, cat, p, tier, style, opts)
    elif fac == "recovery":
        recovery(o, cat, p, tier, style)
    elif fac == "car_park":
        car_park(o, cat, p, tier, opts)
    elif fac == "drop_off":
        drop_off(o, cat, p, tier, opts)
    elif fac in ("medical_centre", "scouting_centre", "commercial_centre", "media_centre"):
        from . import interior
        interior.build(o, cat, p, tier, style, opts, fac)
    elif fac == "maintenance":
        maintenance(o, cat, p, tier, style)


# -- pitches ---------------------------------------------------------------------------------------

def pitch_dims(p, length, width, runoff_end, runoff_side):
    """Pitch size within the plot (length along its x, width along its y)."""
    return min(length, p.w - 2.0 * runoff_end - 10.0), min(width, p.d - 2.0 * runoff_side - 16.0)


def pitch_params(p, l, w, re, rs, grass, height, wear, wet, mow, hand, crown, markings):
    return {"length": l, "width": w, "runoff_end": re, "runoff_side": rs, "grass": grass, "height_mm": height,
            "wear": wear, "wetness": wet, "mow": {"pattern": mow, "width": 5.25, "angle": 0.0, "offset": 0.0},
            "hand_painted": hand, "crown": crown, "line_color": "#f5f5f0", "optional_marks": False,
            "markings": markings, "seed": p.seed % 997 + 1}


def kelvin_rgb(k):
    """Approximate blackbody colour (Tanner Helland), normalised to max 1, then ^2.2."""
    t = k / 100.0
    r = 255.0 if t <= 66.0 else 329.7 * (t - 60.0) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66.0 else 288.1 * (t - 60.0) ** -0.0755
    b = 255.0 if t >= 66.0 else 0.0 if t <= 19.0 else 138.5 * math.log(t - 10.0) - 305.0
    c = [min(max(v, 0.0), 255.0) / 255.0 for v in (r, g, b)]
    m = max(c)
    return tuple((v / m) ** 2.2 for v in c)


def pole_lights(o, f, name, l, w, runoff_side, per_side, height, lamps, lux, behind_n=0.0):
    """Floodlight poles along both touchlines, designed to a lux target (the stadium's pole design);
    behind_n moves the +y side's poles that much further back (behind a stand), still aimed at
    the same spots on the pitch."""
    d = FL.design_poles({"type": "poles", "per_side": per_side, "height": height, "lamps": lamps, "lux": lux, "fixture": "led",
                         "kelvin": 5000}, l, w, runoff_side)
    fixtures = [((pos[0], pos[1] + (behind_n if pos[1] > 0.0 else 0.0), pos[2]), t) for pos, t in d["fixtures"]]
    masts = [((a[0], a[1] + (behind_n if a[1] > 0.0 else 0.0), a[2]), (b[0], b[1] + (behind_n if a[1] > 0.0 else 0.0), b[2]), r)
             for a, b, r in d["tube_masts"]]
    for pos, t in fixtures:
        a, b = f.pt(pos[0], pos[1]), f.pt(t[0], t[1])
        pa, pb = (a[0], a[1], pos[2]), (b[0], b[1], t[2])
        dv = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        mount = tuple(float(v) for v in FL.mount_point(pa, pb, FL.PIVOT.get(d["fixture"], 0.0)))
        o.prop(f"floodlight_fixture.{d['fixture']}", Placement.aim(mount, dv))
        if o.night:
            ln = math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
            u = (dv[0] / ln, dv[1] / ln, dv[2] / ln)
            o.spot((pa[0] + u[0] * 0.3, pa[1] + u[1] * 0.3, pa[2] + u[2] * 0.3), u, d["power"], d["beam"], kelvin_rgb(d["kelvin"]))
    o.tubes_at(f, f"{name}:poles", [([a, b], r) for a, b, r in masts], mats.named("Galvanised"), 12)


def ball_stop(o, f, name, x, half, h):
    """Ball-stop netting behind a goal line: posts every 4 m and a net, `x` metres from the centre."""
    net = kernel.Mesh()
    posts = []
    for s in (-1.0, 1.0):
        xx = s * x
        quad(net, (xx, -half, LAWN_Z + 0.05), (xx, half, LAWN_Z + 0.05), (xx, half, h), (xx, -half, h), 0)
        n = int(max(rround(2.0 * half / 4.0), 1.0))
        for j in range(n + 1):
            y = -half + 2.0 * half * j / n
            posts.append(([(xx, y, LAWN_Z), (xx, y, h + 0.1)], 0.045))
        posts.append(([(xx, -half, h), (xx, half, h)], 0.02))
    o.part_at(f, f"{name}:ballstop", net, [mats.ball_stop()])
    o.tubes_at(f, f"{name}:ballstop_posts", posts, mats.named("Galvanised"), 8)


def filming_tower(o, f, name, top):
    """A scaffold filming tower with a covered platform at `top` metres, front towards +y."""
    t = []
    a = 1.3
    corners = [(-a, -a), (a, -a), (a, a), (-a, a)]
    for c in corners:
        t.append(([(c[0], c[1], LAWN_Z), (c[0], c[1], top + 2.3)], 0.03))
    z = LAWN_Z + 0.3
    k = 0
    while z <= top + 0.01:
        ring = [(c[0], c[1], z) for c in corners]
        ring.append(ring[0])
        t.append((ring, 0.024))
        if z + 2.0 <= top + 0.01:
            c0, c1 = corners[k % 4], corners[(k + 1) % 4]
            t.append(([(c0[0], c0[1], z), (c1[0], c1[1], z + 2.0)], 0.022))
        z += 2.0
        k += 1
    for x in (-0.25, 0.25):
        t.append(([(x, a + 0.1, LAWN_Z), (x, a + 0.1, top + 1.0)], 0.02))
    zz = LAWN_Z + 0.3
    while zz < top + 1.0:
        t.append(([(-0.25, a + 0.1, zz), (0.25, a + 0.1, zz)], 0.015))
        zz += 0.3
    m = kernel.Mesh()
    bx(m, -a - 0.2, -a - 0.2, top - 0.08, a + 0.2, a + 0.2, top + 0.04, 0)
    bx(m, -a - 0.3, -a - 0.3, top + 2.3, a + 0.3, a + 0.3, top + 2.42, 1)
    for x0, y0, x1, y1 in ((-a, -a - 0.2, a, -a - 0.15), (-a - 0.2, -a, -a - 0.15, a), (a + 0.15, -a, a + 0.2, a)):
        bx(m, x0, y0, top + 0.04, x1, y1, top + 1.05, 2)
    o.tubes_at(f, f"{name}:tower", t, mats.named("Galvanised"), 6)
    o.part_at(f, f"{name}:tower_deck", m, [mats.named("Site_Plywood"), mats.tint("TINT_trim", "#2b2d30"), mats.tint("TINT_club", o.club.primary)])


def viewing_deck(o, f, name, ln, roof_mat):
    """A raised viewing deck along +y of frame f: steel posts, a slab, glass balustrade, canopy."""
    dep, z = 4.5, 2.8
    m = kernel.Mesh()
    bx(m, -ln / 2.0, -dep / 2.0, z, ln / 2.0, dep / 2.0, z + 0.3, 0)
    n = int(rround(ln / 4.0))
    for j in range(n + 1):
        x = -ln / 2.0 + 0.2 + (ln - 0.4) * j / n
        for y in (-dep / 2.0 + 0.2, dep / 2.0 - 0.2):
            bx(m, x - 0.12, y - 0.12, LAWN_Z, x + 0.12, y + 0.12, z, 1)
            bx(m, x - 0.08, y - 0.08, z + 0.3, x + 0.08, y + 0.08, z + 3.2, 1)
    for s in range(14):
        x0 = -ln / 2.0 - 0.3 * (s + 1)
        bx(m, x0, -dep / 2.0, LAWN_Z, x0 + 0.3, -dep / 2.0 + 1.4, z + 0.3 - 0.2 * s, 0)
    quad(m, (-ln / 2.0, -dep / 2.0, z + 0.3), (ln / 2.0, -dep / 2.0, z + 0.3), (ln / 2.0, -dep / 2.0, z + 1.4), (-ln / 2.0, -dep / 2.0, z + 1.4), 2)
    bx(m, -ln / 2.0 - 0.3, -dep / 2.0 - 0.6, z + 3.2, ln / 2.0 + 0.3, dep / 2.0 + 0.3, z + 3.45, 3)
    o.part_at(f, f"{name}:deck", m, [mats.named("Facade_Concrete"), mats.tint("TINT_trim", "#2b2d30"), mats.named("Balustrade_Glass"), roof_mat])


DISC_PAIRS = (("#ffd21f", "#2c7be5"), ("#ff6a1a", "#f4f4f2"), ("#e53935", "#ffd21f"), ("#2c7be5", "#f4f4f2"), ("#7cd43a", "#ff6a1a"))
KIT_COLOURS = ("#ff6a1a", "#ffd21f", "#2c7be5", "#e53935")


def drills(o, f, l, w, seed):
    """Training gear set out for a session (pitch-local, x along the length). Every pitch has its
    own session: which half and side it uses, the cone colours, the size of the marker grid, and
    which drills are out (a slalom, a mannequin wall, a pole and ladder station, mini goals or a
    rondo circle), with the balls scattered."""
    def u(k, i=0):
        return uniform(seed + 10 + k, i)
    sx = 1.0 if u(0) < 0.5 else -1.0
    sy = 1.0 if u(1) < 0.5 else -1.0
    discs = DISC_PAIRS[int(u(2) * len(DISC_PAIRS)) % len(DISC_PAIRS)]
    tall = KIT_COLOURS[int(u(3) * len(KIT_COLOURS)) % len(KIT_COLOURS)]
    pole = KIT_COLOURS[int(u(4) * len(KIT_COLOURS)) % len(KIT_COLOURS)]

    def at(key, x, y, yaw, tints=()):
        o.prop_at(f, key, sx * x, sy * y, LAWN_Z, yaw, list(tints))

    def j(i):
        return (uniform(seed, i) - 0.5) * 0.3
    if u(5) < 0.8:
        for i in range(8):
            at("prop.cone_tall", -l / 4.0 - 12.0 + i * 3.0 + j(i), -w / 4.0 + (0.8 if i % 2 == 0 else -0.8), 0.0, [("TINT_cone", tall)])
    n = 3 if u(6) < 0.4 else 4
    step = 5.0 + 3.0 * u(7)
    for a in range(n):
        for b in range(n):
            at("prop.cone_disc", -l / 4.0 + a * step, w / 8.0 + b * step, 0.0, [("TINT_cone", discs[(a + b) % 2])])
    if u(8) < 0.7:
        for i in range(5):
            at("prop.mannequin", l / 2.0 - 20.0, -1.2 + i * 0.6, -HALF_PI * sx, [("TINT_mannequin", "#f4d21c")])
    if u(9) < 0.75:
        for i in range(6):
            at("prop.agility_pole", -8.0 + i * 1.8, w / 2.0 - 8.0 + (0.6 if i % 2 == 0 else -0.6), 0.0, [("TINT_pole_train", pole)])
        at("prop.agility_ladder", 6.0, w / 2.0 - 6.0, HALF_PI)
    if u(10) < 0.6:
        for s2 in (-1.0, 1.0):
            at("prop.mini_goal", -l / 2.0 + 16.0, s2 * 14.0, -s2 * sx * HALF_PI)
    else:
        for i in range(10):
            a = i / 10.0 * 2.0 * math.pi
            at("prop.cone_disc", -l / 2.0 + 18.0 + 7.0 * math.cos(a), 7.0 * math.sin(a), 0.0, [("TINT_cone", discs[0])])
    at("prop.ball_cart", 2.0, -w / 2.0 - 1.8, 0.3)
    at("prop.rebounder", -4.0, -w / 2.0 + 2.5, 0.0 if sy > 0.0 else math.pi)
    # the touchline station: a bag of balls, bibs, water carriers, the coaches' board, a kit bag
    if u(12) < 0.6:
        at("genprop.ball_bag", 3.4, -w / 2.0 - 1.6, 0.6)
    at("genprop.bibs_stack", 0.6, -w / 2.0 - 1.5, 0.2)
    for i in range(2):
        at("genprop.water_carrier", -1.0 - i * 0.5, -w / 2.0 - 1.4, 0.4 * i)
    if u(13) < 0.5:
        at("genprop.tactics_easel", -3.0, -w / 2.0 - 2.2, 0.0 if sy > 0.0 else math.pi)
    at("genprop.kit_bag", 5.0, -w / 2.0 - 1.5, 1.2)
    if u(14) < 0.5:
        at("genprop.gk_gloves", l / 2.0 - 2.0, 1.5, 0.8)
    # (one ball in three a muddy one)
    for i in range(8 + int(u(11) * 9.0)):
        x, y = (uniform(seed + 3, i) - 0.5) * l * 0.7, (uniform(seed + 4, i) - 0.5) * w * 0.7
        at("genprop.ball_muddy" if i % 3 == 0 else "prop.ball", x, y, uniform(seed + 5, i) * TAU, [("TINT_ball", "#f4f4f2")])
    # gulls on the grass
    if u(15) < 0.35:
        for i in range(1 + int(u(16) * 3.0)):
            at("genprop.seagull_standing", (u(17, i) - 0.5) * l * 0.6, (u(18, i) - 0.5) * w * 0.6, u(19, i) * TAU)


def _f(v, d):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else d


def pitch(o, cat, p, tier, opts):
    t = tier_spec(cat, "pitch", tier) or {}
    re, rs = 4.0, 3.0
    l, w = pitch_dims(p, _f(option(cat, "pitch", tier, opts, "length"), 105.0), _f(option(cat, "pitch", tier, opts, "width"), 68.0), re, rs)
    mv = option(cat, "pitch", tier, opts, "mow")
    mow = mv if isinstance(mv, str) else text(t, "mow", "stripes")
    grass = text(t, "grass", "perennial_ryegrass")
    params = pitch_params(p, l, w, re, rs, grass, num(t, "height_mm", 30.0), num(t, "wear", 0.4), num(t, "wetness", 0.0), mow,
                          num(t, "hand_painted", 0.0), num(t, "crown", 0.12), "full")
    lit = option_bool(cat, "pitch", tier, opts, "floodlights")
    o.pitches.append(SitePitch(p.id, params, p.frame.pt(0.0, 0.0), p.frame.yaw, text(t, "goals", "classic"), True,
                               text(t, "flags", "pro_spring"), lit, "hybrid" if grass.startswith("hybrid") else "grass"))
    f = p.frame
    if option_bool(cat, "pitch", tier, opts, "ball_stop"):
        ball_stop(o, f, p.id, l / 2.0 + re + 2.0, min(w / 2.0 + 8.0, p.d / 2.0 - 2.0), 8.0 if tier >= 3 else 6.0)
    if lit:
        per, h, lamps, lux = (3, 15.0, 2, 200.0) if tier in (1, 2) else (4, 18.0, 3, 400.0) if tier == 3 else (4, 20.0, 4, 600.0)
        pole_lights(o, f, p.id, l, w, rs, per, h, lamps, lux)
    if option_bool(cat, "pitch", tier, opts, "shelters"):
        for s in (-1.0, 1.0):
            o.prop_at(f, "prop.team_shelter", s * 9.0, -(w / 2.0 + rs + 1.6), LAWN_Z, 0.0, [("TINT_club", o.club.primary)])
    if option_bool(cat, "pitch", tier, opts, "filming_tower"):
        filming_tower(o, f.sub(0.0, -min(w / 2.0 + rs + 6.0, p.d / 2.0 - 2.0), 0.0), p.id, 8.0)
    if option_bool(cat, "pitch", tier, opts, "viewing_deck"):
        viewing_deck(o, f.sub(0.0, min(w / 2.0 + rs + 4.8, p.d / 2.0 - 2.4), math.pi), p.id, 24.0, mats.named("Roof_Flat"))
    if option_bool(cat, "pitch", tier, opts, "equipment"):
        drills(o, f, l, w, p.seed + 500)


# FA youth formats: pitch (length, width), the pitch generator's markings, the goal style (the
# FA's goalpost size for the format) and the FA line colour used where formats share ground.
YOUTH_FORMATS = {
    "5v5": (36.6, 27.4, "5v5", "youth_mini", "#d42a2a"),
    "7v7": (54.9, 36.6, "7v7", "youth_mini", "#f0c419"),
    "9v9": (73.2, 45.7, "9v9", "youth_9v9", "#2a62c9"),
    "11v11_u13": (82.3, 50.3, "full", "youth_u13", "#f5f5f0"),
}
YOUTH_RUNOFF = 3.0      # FA: 3 m clear round a youth pitch


def youth_grid(pw, pd, l, w, n_max):
    """How youth pitches of l x w (plus run-off) pack into a pw x pd plot: (rotated, nx, ny) with
    the most pitches, length along the plot when that fits as many; at most n_max."""
    best = None
    for rot in (False, True):
        cl, cw = l + 2.0 * YOUTH_RUNOFF, w + 2.0 * YOUTH_RUNOFF
        cx, cy = (cw, cl) if rot else (cl, cw)
        nx, ny = int(max(pw - 4.0, 0.0) / cx), int(max(pd - 4.0, 0.0) / cy)
        if best is None or nx * ny > best[1] * best[2]:
            best = (rot, nx, ny)
    rot, nx, ny = best
    while nx * ny > n_max:
        if nx >= ny:
            nx -= 1
        else:
            ny -= 1
    return rot, nx, ny


def youth_pitches(o, cat, p, tier, opts):
    """Youth pitches in one of the FA formats, as many as fit the plot (or `count`), each with its
    own goals, corner flags and session; the grass, line finish and kit improve by tier. Parents
    watch from behind a post-and-rail on one touchline, the teams have the other."""
    t = tier_spec(cat, "youth_pitches", tier) or {}
    fv = option(cat, "youth_pitches", tier, opts, "format")
    l, w, marks, goals, fa_line = YOUTH_FORMATS.get(fv if isinstance(fv, str) else "9v9", YOUTH_FORMATS["9v9"])
    cv = option(cat, "youth_pitches", tier, opts, "count")
    n_max = int(max(_f(cv, 99.0), 1.0))
    rot, nx, ny = youth_grid(p.w, p.d, l, w, n_max)
    if nx * ny == 0:
        return
    mv = option(cat, "youth_pitches", tier, opts, "mow")
    mow = mv if isinstance(mv, str) else text(t, "mow", "stripes")
    line = fa_line if option(cat, "youth_pitches", tier, opts, "lines") == "fa_colours" else "#f5f5f0"
    lit = option_bool(cat, "youth_pitches", tier, opts, "floodlights")
    r = YOUTH_RUNOFF
    cl, cw = l + 2.0 * r, w + 2.0 * r
    cx, cy = (cw, cl) if rot else (cl, cw)
    k = 0
    for j in range(ny):
        for i in range(nx):
            u, v = (i - (nx - 1) / 2.0) * cx, (j - (ny - 1) / 2.0) * cy
            f = p.frame.sub(u, v, HALF_PI if rot else 0.0)
            pid = f"{p.id}-{k + 1}"
            seed = p.seed + 37 * k
            params = pitch_params(p, l, w, r, r, text(t, "grass", "perennial_ryegrass"), num(t, "height_mm", 30.0),
                                  num(t, "wear", 0.4), num(t, "wetness", 0.0), mow, num(t, "hand_painted", 0.0),
                                  num(t, "crown", 0.1), marks)
            params["line_width"] = 0.1
            params["line_color"] = line
            params["seed"] = seed % 997 + 1
            o.pitches.append(SitePitch(pid, params, f.pt(0.0, 0.0), f.yaw, goals, True, text(t, "flags", "grassroots"), lit, "grass"))
            # side by side, alternate pitches are mirrored: parents share one gap, teams the next
            side = -1.0 if (i if rot else j) % 2 == 1 else 1.0
            if option_bool(cat, "youth_pitches", tier, opts, "rails"):
                spectator_rail(o, f, pid, l / 2.0 + 1.0, -side * (w / 2.0 + 2.5))
            if option_bool(cat, "youth_pitches", tier, opts, "shelters"):
                for s in (-1.0, 1.0):
                    o.prop_at(f, "prop.team_shelter", s * min(6.0, l / 6.0), side * (w / 2.0 + 2.2), LAWN_Z,
                              math.pi if side > 0.0 else 0.0, [("TINT_club", o.club.primary)])
            if option_bool(cat, "youth_pitches", tier, opts, "ball_stop"):
                ball_stop(o, f, pid, l / 2.0 + 2.8, w / 2.0 + 2.0, 4.5)
            if lit:
                pole_lights(o, f, pid, l, w, r, 2, 12.0, 2, 120.0)
            if option_bool(cat, "youth_pitches", tier, opts, "equipment") and k % 2 == 0:
                youth_session(o, f, l, w, seed + 500)
            k += 1


def spectator_rail(o, f, name, half, y):
    """A post-and-rail spectator barrier along a touchline (pitch-local y): 48 mm galvanised posts
    every 2.5 m, a top rail at 1.1 m and a mid rail."""
    posts = []
    n = int(max(rround(2.0 * half / 2.5), 1.0))
    for j in range(n + 1):
        x = -half + 2.0 * half * j / n
        posts.append(([(x, y, LAWN_Z), (x, y, LAWN_Z + 1.1)], 0.024))
    for z in (1.1, 0.55):
        posts.append(([(-half, y, LAWN_Z + z), (half, y, LAWN_Z + z)], 0.021))
    o.tubes_at(f, f"{name}:rail", posts, mats.named("Galvanised"), 8)


def youth_session(o, f, l, w, seed):
    """A youth session set out on a small pitch: a grid of disc cones, a cone slalom, pop-up mini
    goals at one end and balls about (pitch-local, sized to the pitch)."""
    def u(k):
        return uniform(seed + 10 + k, 0)
    sx = 1.0 if u(0) < 0.5 else -1.0
    discs = DISC_PAIRS[int(u(1) * len(DISC_PAIRS)) % len(DISC_PAIRS)]
    tall = KIT_COLOURS[int(u(2) * len(KIT_COLOURS)) % len(KIT_COLOURS)]

    def at(key, x, y, yaw, tints=()):
        o.prop_at(f, key, sx * x, y, LAWN_Z, yaw, list(tints))
    step = min(4.0, w / 8.0)
    for a in range(3):
        for b in range(3):
            at("prop.cone_disc", -l / 4.0 + (a - 1) * step, (b - 1) * step, 0.0, [("TINT_cone", discs[(a + b) % 2])])
    for i in range(6):
        at("prop.cone_tall", l / 8.0 + i * 1.5, -w / 4.0 + (0.5 if i % 2 == 0 else -0.5), 0.0, [("TINT_cone", tall)])
    if u(3) < 0.6:
        for s2 in (-1.0, 1.0):
            at("prop.mini_goal", l / 2.0 - 6.0, s2 * w / 5.0, -sx * HALF_PI)
    for i in range(5 + int(u(4) * 6.0)):
        x, y = (uniform(seed + 3, i) - 0.5) * l * 0.6, (uniform(seed + 4, i) - 0.5) * w * 0.6
        at("prop.ball", x, y, uniform(seed + 5, i) * TAU, [("TINT_ball", "#f4f4f2")])
    at("prop.ball_cart", 0.0, -w / 2.0 - 1.2, 0.3)


def artificial(o, cat, p, tier, opts):
    t = tier_spec(cat, "artificial_pitch", tier) or {}
    re, rs = (2.0, 2.0) if tier == 1 else (3.0, 3.0)
    l, w = pitch_dims(p, num(t, "length", 60.0), num(t, "width", 40.0), re, rs)
    params = pitch_params(p, l, w, re, rs, "artificial_3g", 40.0, 0.1, 0.0, "stripes", 0.0, 0.03, "outline" if tier == 1 else "full")
    lit = option_bool(cat, "artificial_pitch", tier, opts, "floodlights")
    o.pitches.append(SitePitch(p.id, params, p.frame.pt(0.0, 0.0), p.frame.yaw, text(t, "goals", "portable"), True, None, lit, "artificial"))
    f = p.frame
    hx, hy = l / 2.0 + re + 1.0, w / 2.0 + rs + 1.0
    apron = kernel.Mesh()
    kernel.ground(apron, kernel.rect(2.0 * hx, 2.0 * hy), LAWN_Z + 0.01, 0)
    o.part_at(f, f"{p.id}:apron", apron, [mats.turf_3g(f.pt(0.0, 0.0), f.yaw)])
    gate = (0.0, -hy)
    ring = [add(gate, (1.2, 0.0)), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy), sub(gate, (1.2, 0.0))]
    landscape.chainlink(o, [f.pt(q[0], q[1]) for q in ring], num(t, "fence", 4.5), f"{p.id}:cage")
    if lit:
        if tier == 1:
            pole_lights(o, f, p.id, l, w, rs, 2, 8.0, 2, 250.0)
        else:
            pole_lights(o, f, p.id, l, w, rs, 3, 15.0, 3, 350.0)
    if option_bool(cat, "artificial_pitch", tier, opts, "equipment"):
        for i in range(6):
            o.prop_at(f, "prop.cone_tall", -6.0 + i * 2.5, -w / 4.0, LAWN_Z, 0.0, [("TINT_cone", "#ff6a1a")])
        o.prop_at(f, "prop.ball_cart", 3.0, -w / 2.0 - 1.2, LAWN_Z, 0.0)


def showpiece(o, cat, p, tier, opts):
    """The academy's full-size match pitch (U18 and U21 games): a perimeter rail round a railed
    hardstanding, dugouts in front of a covered seated stand along one touchline, floodlights and
    ball-stop nets; the stand grows with the tier, the top tier adds a filming tower."""
    t = tier_spec(cat, "showpiece_pitch", tier) or {}
    rows = int(num(t, "stand_rows", 0.0))
    re, rs = 4.0, 3.0
    l, w = pitch_dims(p, num(t, "length", 100.0), num(t, "width", 64.0), re, rs)
    # the pitch sits off-centre, leaving room for the stand on the +y side
    pf = p.frame.sub(0.0, -(rows * 0.8 + 1.0) / 2.0, 0.0)
    grass = text(t, "grass", "perennial_ryegrass")
    params = pitch_params(p, l, w, re, rs, grass, num(t, "height_mm", 25.0), num(t, "wear", 0.2), 0.0, "stripes", 0.0,
                          num(t, "crown", 0.15), "full")
    lit = option_bool(cat, "showpiece_pitch", tier, opts, "floodlights")
    o.pitches.append(SitePitch(p.id, params, pf.pt(0.0, 0.0), pf.yaw, text(t, "goals", "classic"), True,
                               text(t, "flags", "striped"), lit, "hybrid" if grass.startswith("hybrid") else "grass"))
    hx, hy = l / 2.0 + re - 0.5, w / 2.0 + rs - 0.5
    spectator_rail(o, pf, f"{p.id}-n", hx, hy)
    spectator_rail(o, pf, f"{p.id}-s", hx, -hy)
    for sx in (-1.0, 1.0):
        spectator_rail(o, pf.sub(sx * hx, 0.0, HALF_PI), f"{p.id}-{'e' if sx > 0.0 else 'w'}", hy, 0.0)
    walk = kernel.Mesh()
    r_out, r_in = kernel.rect(2.0 * hx + 4.0, 2.0 * hy + 4.0), kernel.rect(2.0 * hx, 2.0 * hy)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        kernel.ground(walk, [r_in[a], r_out[a], r_out[b], r_in[b]], COVER_Z + 0.01, 0)
    o.part_at(pf, f"{p.id}:walkway", walk, [mats.named("Apron_Concrete")])
    for sx in (-1.0, 1.0):
        o.prop_at(pf, "prop.team_shelter", sx * 6.0, w / 2.0 + 1.4, LAWN_Z, math.pi, [("TINT_club", o.club.primary)])
    ball_stop(o, pf, p.id, l / 2.0 + re + 2.5, w / 2.0 + 2.0, 6.0)
    if lit:
        # on the stand side the poles stand behind the stand, clear of its roof
        pole_lights(o, pf, p.id, l, w, rs, int(num(t, "poles", 3.0)), num(t, "pole_h", 15.0), int(num(t, "lamps", 2.0)), num(t, "lux", 200.0),
                    rows * 0.8 + 3.5 if rows > 0 else 0.0)
    if rows > 0:
        sv = option(cat, "showpiece_pitch", tier, opts, "seat")
        seat = sv if isinstance(sv, str) else "tipup_classic"
        cv = option(cat, "showpiece_pitch", tier, opts, "seat_color")
        stand(o, pf, p.id, num(t, "stand_length", 40.0), rows, w / 2.0 + rs + 1.5, seat, cv if isinstance(cv, str) else o.club.primary,
              o.club.name if option_bool(cat, "showpiece_pitch", tier, opts, "sign") else None)
    if option_bool(cat, "showpiece_pitch", tier, opts, "filming_tower"):
        filming_tower(o, pf.sub(0.0, -min(w / 2.0 + rs + 5.0, p.d / 2.0 - 2.0), 0.0), p.id, 7.0)


def stand(o, f, name, ln, rows, y0, seat, seat_color, sign):
    """A covered seated stand along a touchline (frame-local, the pitch at -y): concrete terraces
    (0.8 m going, 0.3 m risers, the front row 0.6 m up behind a club-coloured wall), seats with an
    aisle every 13 m and intermediate steps in it, a walkway behind, a profiled steel back wall
    and end screens, and a roof sloping to the rear on white steel columns with raking rafters,
    its fascia carrying the club's name."""
    g, rr, z0 = 0.8, 0.3, 0.6
    m = kernel.Mesh()
    hl = ln / 2.0
    # (the concrete stops 3 cm short of the end screens and the back wall: never in their plane)
    for r in range(rows):
        bx(m, -hl + 0.03, y0 + r * g, LAWN_Z, hl - 0.03, y0 + (r + 1) * g, LAWN_Z + z0 + r * rr, 0)
    z_top = LAWN_Z + z0 + rows * rr
    yb = y0 + rows * g + 1.5
    bx(m, -hl + 0.03, y0 + rows * g, LAWN_Z, hl - 0.03, yb - 0.03, z_top - rr, 0)
    bx(m, -hl, y0 - 0.15, LAWN_Z, hl, y0, LAWN_Z + z0 + 0.8, 4)
    zr_f, zr_b = z_top + 3.4, z_top + 2.8
    yf = y0 - 1.2
    quad(m, (-hl, yb, LAWN_Z), (hl, yb, LAWN_Z), (hl, yb, zr_b), (-hl, yb, zr_b), 1)
    # the way in from behind: three entrances (stairs up inside to the walkway) in white frames,
    # and a refreshment kiosk with a club-coloured roller shutter over its counter
    for x in (-hl / 2.0, 0.0, hl / 2.0):
        quad(m, (x + 0.9, yb + 0.02, LAWN_Z), (x - 0.9, yb + 0.02, LAWN_Z), (x - 0.9, yb + 0.02, LAWN_Z + 2.3), (x + 0.9, yb + 0.02, LAWN_Z + 2.3), 6)
        for x0, x1, z1, z2 in ((x - 1.05, x - 0.9, LAWN_Z, LAWN_Z + 2.45), (x + 0.9, x + 1.05, LAWN_Z, LAWN_Z + 2.45),
                               (x - 1.05, x + 1.05, LAWN_Z + 2.3, LAWN_Z + 2.45)):
            bx(m, x0, yb, z1, x1, yb + 0.1, z2, 5)
    kernel.ground(m, kernel.rect(ln + 4.0, 3.0, 0.0, yb + 1.5), COVER_Z + 0.01, 0)
    xk = -hl / 4.0
    bx(m, xk - 1.6, yb, LAWN_Z + 0.95, xk + 1.6, yb + 0.45, LAWN_Z + 1.05, 0)
    bx(m, xk - 1.5, yb, LAWN_Z + 1.05, xk + 1.5, yb + 0.06, LAWN_Z + 2.25, 4)
    bx(m, xk - 1.6, yb, LAWN_Z + 2.25, xk + 1.6, yb + 0.25, LAWN_Z + 2.45, 5)
    for sx in (-1.0, 1.0):
        x = sx * hl
        pts = [(x, y0, LAWN_Z), (x, yb, LAWN_Z), (x, yb, zr_b), (x, yf + 1.4, zr_f - 0.1)]
        quad(m, *(pts if sx > 0.0 else pts[::-1]), 1)
    e = 0.6
    quad(m, (-hl - e, yf, zr_f), (hl + e, yf, zr_f), (hl + e, yb + 0.3, zr_b), (-hl - e, yb + 0.3, zr_b), 2)
    quad(m, (-hl - e, yb + 0.3, zr_b - 0.12), (hl + e, yb + 0.3, zr_b - 0.12), (hl + e, yf, zr_f - 0.12), (-hl - e, yf, zr_f - 0.12), 3)
    bx(m, -hl - e, yf - 0.25, zr_f - 0.75, hl + e, yf, zr_f + 0.08, 4)
    if sign:
        buildings.letters(m, ln + 2.0 * e, -2.0 * (yf - 0.25), 0, sign, (ln + 2.0 * e) / 2.0, zr_f - 0.62, ln * 0.45, 5, 0.5 / 7.0)
    steel = []
    n = int(max(rround(ln / 7.5), 1.0))
    for k in range(n + 1):
        x = -hl + ln * k / n
        steel.append(([(x, yb - 0.3, LAWN_Z), (x, yb - 0.3, zr_b - 0.15)], 0.14))
        steel.append(([(x, yb - 0.3, zr_b - 0.15), (x, yf + 0.3, zr_f - 0.15)], 0.09))
        steel.append(([(x, yb - 0.3, z_top + 0.6), (x, (yb + yf) / 2.0, (zr_b + zr_f) / 2.0 - 0.15)], 0.06))
    o.tubes_at(f, f"{name}:stand_steel", steel, mats.tint("Roof_Steel", "#eceeee"), 8)
    # seats, an aisle every 13 m with a half step at each row
    na = int(max(rround(ln / 13.0), 1.0))
    aisles = [-hl + ln * (k + 0.5) / na for k in range(na)]
    for r in range(rows):
        y, z = y0 + r * g, LAWN_Z + z0 + r * rr
        for xa in aisles:
            bx(m, xa - 0.6, y, z, xa + 0.6, y + 0.4, z + rr / 2.0, 0)
        x = -hl + 0.6
        while x <= hl - 0.6:
            if all(abs(x - xa) > 0.95 for xa in aisles):
                o.prop_at(f, f"seat.{seat}.std", x, y + 0.45, z, math.pi, [("TINT_seat", seat_color)])
            x += 0.5
    o.part_at(f, f"{name}:stand", m, [mats.named("Stand_Concrete"), mats.tint("TINT_corrugated", "#59616a"),
                                     mats.tint("TINT_roof", "#d9dbdc"), mats.named("Roof_Underside"), mats.tint("TINT_club", o.club.primary),
                                     mats.tint("TINT_trim", "#f2f2f0"), mats.named("Bldg_Opening")])


def cages(o, cat, p, tier, opts):
    """Small-sided 3G cages for the youngest age groups: 3G with 5v5 markings and youth goals, steel
    rebound boards round the base (1.2 m, the ball plays off them), chain-link above, a gate on one
    side and LED floodlights on short columns; one cage, or as many as fit the plot."""
    t = tier_spec(cat, "cages", tier) or {}
    l, w = num(t, "length", 30.0), num(t, "width", 20.0)
    rot, nx, ny = youth_grid(p.w, p.d, l, w, int(num(t, "max", 1.0)))
    lit = option_bool(cat, "cages", tier, opts, "floodlights")
    cl, cw = l + 2.0 * YOUTH_RUNOFF, w + 2.0 * YOUTH_RUNOFF
    cx, cy = (cw, cl) if rot else (cl, cw)
    k = 0
    for j in range(ny):
        for i in range(nx):
            f = p.frame.sub((i - (nx - 1) / 2.0) * cx, (j - (ny - 1) / 2.0) * cy, HALF_PI if rot else 0.0)
            cid = f"{p.id}-{k + 1}"
            params = pitch_params(p, l, w, 1.0, 1.0, "artificial_3g", 40.0, 0.15, 0.0, "stripes", 0.0, 0.02, "5v5")
            params["line_width"] = 0.08
            params["seed"] = (p.seed + 37 * k) % 997 + 1
            o.pitches.append(SitePitch(cid, params, f.pt(0.0, 0.0), f.yaw, "youth_mini", True, None, lit, "artificial"))
            hx, hy = l / 2.0 + 1.0, w / 2.0 + 1.0
            boards = kernel.Mesh()
            gate = 1.2
            # (just inside the chain-link, mitred at the corners: no face shares another's plane)
            a, b = 0.03, 0.08
            for x0, y0, x1, y1 in ((-hx + b, -hy + a, -gate, -hy + b), (gate, -hy + a, hx - b, -hy + b), (-hx + b, hy - b, hx - b, hy - a),
                                   (-hx + a, -hy + a, -hx + b, hy - a), (hx - b, -hy + a, hx - a, hy - a)):
                bx(boards, x0, y0, LAWN_Z, x1, y1, LAWN_Z + 1.2, 0)
            o.part_at(f, f"{cid}:boards", boards, [mats.tint("TINT_fence", "#2b4433")])
            ring = [(-gate, -hy), (-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy), (gate, -hy)]
            landscape.chainlink(o, [f.pt(q[0], q[1]) for q in ring], num(t, "fence", 4.0), f"{cid}:cage")
            if lit:
                pole_lights(o, f, cid, l, w, 1.0, 2, 8.0, 2, 250.0)
            k += 1


def goalkeeper(o, cat, p, tier):
    t = tier_spec(cat, "goalkeeper_area", tier) or {}
    l, w = min(num(t, "length", 34.0), p.w - 8.0), min(num(t, "width", 26.0), p.d - 6.0)
    params = pitch_params(p, l, w, 2.0, 2.0, "perennial_ryegrass", 26.0, 0.5, 0.0, "stripes_long", 0.0, 0.03, "none")
    o.pitches.append(SitePitch(f"{p.id}-gk", params, p.frame.pt(0.0, 0.0), p.frame.yaw, "box", tier >= 2, None, False, "grass"))
    f = p.frame
    for i in range(3):
        o.prop_at(f, "prop.mannequin", l / 2.0 - 9.0, -2.0 + i * 2.0, LAWN_Z, -HALF_PI, [("TINT_mannequin", "#f4d21c")])
    if tier >= 2:
        for i in range(3):
            o.prop_at(f, "prop.rebounder", -6.0 + i * 6.0, -w / 2.0 + 1.0, LAWN_Z, 0.0)
        ball_stop(o, f, p.id, l / 2.0 + 3.0, w / 2.0 + 1.0, 6.0)
        o.prop_at(f, "prop.storage_container", -l / 2.0 - 1.0, w / 2.0 + 0.5, COVER_Z, HALF_PI, [("TINT_container", "#3c5a3a")])
    o.prop_at(f, "prop.ball_cart", 0.0, -w / 2.0 - 1.0, LAWN_Z, 0.0)


# -- buildings --------------------------------------------------------------------------------------

class Placed:
    """A building's place on a plot: its frame (front, local -y, to the plot's frontage), its size
    (clamped to the plot) and the gap left in front of it for a forecourt or yard."""
    __slots__ = ("f", "w", "d", "gap")

    def __init__(self, f, w, d, gap):
        self.f, self.w, self.d, self.gap = f, w, d, gap


def place(p, bw, bd, forecourt_):
    yaw = math.atan2(p.front[0], -p.front[1])
    along_x = abs(p.front[1]) > 0.5
    pw, pd = (p.w, p.d) if along_x else (p.d, p.w)
    w = min(bw, pw - 6.0)
    d = max(min(bd, pd - forecourt_ - 2.0), 4.0)
    lim = max(pd / 2.0 - d / 2.0 - 1.0, 0.0)
    c = min(max(pd / 2.0 - forecourt_ - d / 2.0, -lim), lim)
    off = mul(p.front, c)
    return Placed(p.frame.sub(off[0], off[1], yaw), w, d, pd / 2.0 - c - d / 2.0)


def forecourt(o, p, b):
    """Paving round a building and out to the plot frontage."""
    m = kernel.Mesh()
    y0, y1 = -b.d / 2.0 - b.gap + 0.5, b.d / 2.0 + 2.0
    kernel.ground(m, kernel.rect(b.w + 4.0, y1 - y0, 0.0, (y0 + y1) / 2.0), COVER_Z, 0)
    o.part_at(b.f, f"{p.id}:forecourt", m, [mats.named("Plaza_Paving")])


def footprint(cat, p, fac, tier):
    """The footprint a facility's main building will have (for its construction site): Placed,
    storeys and storey height, or None."""
    t = tier_spec(cat, fac, tier)
    if isinstance(cat["facilities"].get(fac, {}).get("sizes"), list):
        sizes = cat["facilities"][fac]["sizes"]
        sv = option(cat, fac, tier, p.cfg.get("options") if isinstance(p.cfg.get("options"), dict) else {}, "size")
        si = min(max(int(sv) if isinstance(sv, (int, float)) and not isinstance(sv, bool) else 2, 1), max(len(sizes), 1))
        sz = sizes[si - 1] if sizes else {"w": 28.8, "d": 15.6, "storeys": 1}
        return place(p, num(sz, "w", 28.8), num(sz, "d", 15.6), 16.0), int(num(sz, "storeys", 1.0)), 3.6
    if not isinstance(t, dict) or not isinstance(t.get("size"), list):
        return None
    bw, bd = float(t["size"][0]), float(t["size"][1])
    if fac in ("training_building", "academy_building"):
        return place(p, bw, bd, 14.0), int(num(t, "storeys", 1.0)), 4.0
    if fac == "recovery":
        return place(p, bw, bd, 8.0), 1, 4.0
    if fac == "maintenance":
        return place(p, bw, bd, 9.0), 1, 5.6
    if fac == "indoor_hall":
        yaw = math.atan2(p.front[0], -p.front[1])
        along_x = abs(p.front[1]) > 0.5
        w, d = min(bw, p.w - 8.0), min(bd, p.d - 10.0)
        ww, dd = (w, d) if along_x else (d, w)
        return Placed(p.frame.sub(0.0, 0.0, yaw), ww, dd, 4.0), 1, num(t, "height", 13.0) - buildings.PLINTH
    return None


def training_building(o, cat, p, tier, style, opts, fac="training_building"):
    """The training building, or (fac "academy_building") an academy's: the same building tiers
    (portacabins to a flagship) with the academy's own modules: classrooms, sports science, a
    parents' lounge and a scholars' lodge, and the academy's name on the sign."""
    t = tier_spec(cat, fac, tier) or {}
    club = o.club
    academy = fac == "academy_building"
    mv = option(cat, fac, tier, opts, "modules")
    modules = [m for m in mv if isinstance(m, str)] if isinstance(mv, list) else []
    if tier == 1:
        cover(o, p, "ground", 1.0, mats.named("Site_Gravel"))
        f = place(p, 20.0, 7.0, 5.0).f
        for i, (x, z) in enumerate(((-5.0, 0.0), (5.0, 0.0), (-5.0, 2.62), (5.0, 2.62))):
            if i >= int(num(t, "cabins", 4.0)):
                break
            o.prop_at(f, "prop.site_cabin", x, 0.0, COVER_Z + z, math.pi, [("TINT_cabin", "#e3e4df")])
        # the upper cabins' access: a galvanised stair climbing to a landing at the end of a
        # walkway along the doors, stringers, treads, handrails and a guard rail with posts
        top, n = 2.62, 12
        rise = top / n
        walk = kernel.Mesh()
        bx(walk, -10.0, -2.4, COVER_Z + top - 0.07, 10.0, -1.5, COVER_Z + top, 0)
        bx(walk, 10.0, -2.4, COVER_Z + top - 0.07, 11.0, -1.4, COVER_Z + top, 0)
        for k in range(n - 1):
            x1 = 11.0 + (k + 1) * 0.26
            z = COVER_Z + top - (k + 1) * rise
            bx(walk, x1 - 0.26, -2.4, z - 0.04, x1, -1.4, z, 0)
        o.part_at(f, f"{p.id}:walkway", walk, [mats.named("Galvanised")])
        foot = 11.0 + n * 0.26
        st = []
        for y in (-2.42, -1.38):
            st.append(([(foot, y, COVER_Z), (11.0, y, COVER_Z + top - 0.07)], 0.06))
            st.append(([(foot + 0.1, y, COVER_Z + 1.0), (11.0, y, COVER_Z + top + 1.0), (10.0, y, COVER_Z + top + 1.0)], 0.024))
            st.append(([(foot, y, COVER_Z), (foot + 0.1, y, COVER_Z + 1.0)], 0.024))
        st.append(([(10.0, -2.42, COVER_Z + top + 1.1), (-10.0, -2.42, COVER_Z + top + 1.1)], 0.024))
        st.append(([(10.0, -2.42, COVER_Z + top + 0.55), (-10.0, -2.42, COVER_Z + top + 0.55)], 0.018))
        for k in range(11):
            x = -10.0 + k * 2.0
            st.append(([(x, -2.42, COVER_Z + top), (x, -2.42, COVER_Z + top + 1.1)], 0.024))
            st.append(([(x, -2.4, COVER_Z), (x, -2.4, COVER_Z + top - 0.07)], 0.05))
        o.tubes_at(f, f"{p.id}:stairs", st, mats.named("Galvanised"), 6)
        for i, key in enumerate(("prop.bin_wheelie", "prop.bin_wheelie", "prop.picnic_table")):
            # bins turned to face the yard (their front is +y), backs to the cabins
            yaw = math.pi if i < 2 else 0.0
            o.prop_at(f, key, -12.0 + i * 1.0 + (-3.0 if i == 2 else 0.0), -4.5, COVER_Z, yaw, [("TINT_bin", "#2f3a2f")])
        o.entrance = f.pt(0.0, -3.0)
        return
    size = t.get("size") or [40.0, 18.0]
    bw, bd = _f(size[0], 40.0), _f(size[1], 18.0)
    storeys = int(num(t, "storeys", 1.0))
    pl = place(p, bw, bd, 14.0)
    f, w, d = pl.f, pl.w, pl.d
    forecourt(o, p, pl)
    sp = buildings.styled(cat, style, w, d, storeys, club.primary, o.night, opts, p.seed)
    sg = option(cat, fac, tier, opts, "sign")
    signed = sg if isinstance(sg, bool) else True
    name = f"{club.name} ACADEMY" if academy else club.name
    # a flagship carries its name on the glass atrium in front (the main facade is behind it)
    if signed and tier < 4:
        sp.sign = name
    b = buildings.build(o, f, f"{p.id}:building", sp)
    o.entrance = f.pt(b.entrance[0], b.entrance[1])

    def wing(name, ww, dd, side, window):
        s2 = buildings.styled(cat, style, ww, dd, 1, club.primary, o.night, opts, p.seed + 7)
        s2.roof = "hip" if s2.roof in ("gable", "hip") else s2.roof
        s2.window = window
        s2.door = "none"
        s2.canopy = "none"
        s2.clock = False
        s2.arcade = False
        wf = f.sub(side * (w / 2.0 + ww / 2.0 - 0.1), (dd - d) / 2.0 + 0.6, 0.0)
        buildings.build(o, wf, f"{p.id}:{name}", s2)
    if "gym" in modules:
        wing("gym", 16.0, d * 0.8, 1.0, "ribbon")
    elif "sports_science" in modules:
        wing("sports_science", 16.0, d * 0.8, 1.0, "ribbon")
    if "rehab" in modules:
        wing("pool", 18.0, d * 0.75, -1.0, "strip")
    elif "classrooms" in modules:
        # the education wing: classrooms with big windows (full-time scholars are schooled on site)
        wing("classrooms", 20.0, d * 0.8, -1.0, "large")
    if tier >= 4:
        atrium = buildings.styled(cat, "ultra_modern", 14.0, 7.0, min(storeys, 2), club.primary, o.night, opts, p.seed + 11)
        atrium.window = "ribbon"
        atrium.door = "glass"
        atrium.canopy = "none"
        atrium.fins = False
        atrium.mats = list(atrium.mats)
        atrium.mats[buildings.T_WALL] = mats.named("Facade_DarkGlass")
        atrium.roof = "flat"
        if signed:
            atrium.sign = name
        af = f.sub(0.0, -d / 2.0 - 3.4, 0.0)
        a = buildings.build(o, af, f"{p.id}:atrium", atrium)
        o.entrance = af.pt(a.entrance[0], a.entrance[1])
        rear = buildings.styled(cat, style, 22.0, 26.0, 2, club.primary, o.night, opts, p.seed + 13)
        rear.door = "none"
        rear.canopy = "none"
        rear.clock = False
        rear.arcade = False
        buildings.build(o, f.sub(w / 2.0 - 14.0, d / 2.0 + 13.0 - 0.2, 0.0), f"{p.id}:rear_wing", rear)
    if "canteen" in modules or "parents_lounge" in modules:
        tf = f.sub(-w / 4.0, d / 2.0 + 5.0, 0.0)
        m = kernel.Mesh()
        kernel.ground(m, kernel.rect(16.0, 9.0), COVER_Z + 0.01, 0)
        o.part_at(tf, f"{p.id}:terrace", m, [mats.named("Site_Wood")])
        for i in range(4):
            x = -5.4 + i * 3.6
            o.prop_at(tf, "prop.picnic_table", x, 0.5, COVER_Z, 0.0)
            o.prop_at(tf, "prop.parasol", x, 0.5, COVER_Z, 0.0, [("TINT_club", club.primary)])
    if "media" in modules:
        h = buildings.PLINTH + storeys * sp.storey_h
        z = h if sp.roof == "flat" else h + 0.3
        o.prop_at(f, "prop.satellite_dish", w / 2.0 - 4.0, d / 2.0 - 3.0, z, math.pi * 0.8)
        o.prop_at(f, "prop.satellite_dish", w / 2.0 - 7.0, d / 2.0 - 3.0, z, math.pi * 0.8)
    for i, s in enumerate((-1.0, 0.0, 1.0)):
        col = club.secondary if i == 1 else club.primary
        o.prop_at(f, "prop.flagpole", s * 3.0 - w / 2.0 + 6.0, -d / 2.0 - 9.0, COVER_Z, 0.0, [("TINT_flag", col)])
    o.prop_at(f, "prop.bike_rack", w / 2.0 - 4.0, -d / 2.0 - 3.0, COVER_Z, 0.0)
    if academy and "boarding" in modules and tier >= 3:
        # the scholars' lodge: a two-storey residential block behind, its own front door
        lodge = buildings.styled(cat, style, 26.0, 11.0, 2, club.primary, o.night, opts, p.seed + 17)
        lodge.window = "punched"
        lodge.roof = "hip" if lodge.roof in ("gable", "hip", "shallow", "barrel") else lodge.roof
        lodge.clock = False
        lodge.arcade = False
        lodge.canopy = "none"
        lodge.sign = None
        buildings.build(o, f.sub(-(w / 2.0 - 13.0), d / 2.0 + 22.0, math.pi), f"{p.id}:lodge", lodge)
    o.prop_at(f, "prop.bin_litter", 3.6, -d / 2.0 - 2.4, COVER_Z, 0.0, [("TINT_bin", "#2a2c2e")])
    o.prop_at(f, "prop.bench", -5.0, -d / 2.0 - 3.2, COVER_Z, math.pi)
    o.prop_at(f, "genprop.pigeon_pecking", -3.9, -d / 2.0 - 4.4, COVER_Z, 2.1)
    o.prop_at(f, "genprop.bicycle", w / 2.0 - 5.85, -d / 2.0 - 3.0, COVER_Z, HALF_PI)
    if tier >= 3:
        # a bust of the club's great manager by the door; a bronze ball at an academy
        o.prop_at(f, "genprop.bronze_football_plinth" if academy else "genprop.bust_pedestal", -9.0, -d / 2.0 - 4.5, COVER_Z, math.pi)


def air_dome(o, f, name, a, b, h):
    """An inflated dome over a half pitch: an ellipsoid of membrane on a ring beam."""
    m = kernel.Mesh()
    nu, nv = 36, 9

    def pt(i, j):
        phi = TAU * i / nu
        th = HALF_PI * j / nv
        c = math.cos(th) ** 0.55
        return (a * c * math.cos(phi), b * c * math.sin(phi), COVER_Z + 0.4 + h * math.sin(th))
    for j in range(nv):
        for i in range(nu):
            p0, p1, p2, p3 = pt(i, j), pt((i + 1) % nu, j), pt((i + 1) % nu, j + 1), pt(i, j + 1)
            if j == nv - 1:
                kernel.tri(m, p0, p1, p2, 0)
            else:
                quad(m, p0, p1, p2, p3, 0)
    base = kernel.Mesh()
    kernel.prism(base, [(pt(i, 0)[0] * 1.02, pt(i, 0)[1] * 1.02) for i in range(nu)], COVER_Z, COVER_Z + 0.45, 1, 1, None)
    bx(base, -2.0, -b - 4.0, COVER_Z, 2.0, -b + 1.0, COVER_Z + 3.0, 2)
    bx(base, a - 2.0, -6.0, COVER_Z, a + 3.0, -2.0, COVER_Z + 2.2, 2)
    o.part_at(f, f"{name}:dome", m, [mats.tint("TINT_roof_membrane", "#f1f1ec")], 25.0)
    o.part_at(f, f"{name}:dome_base", base, [mats.tint("TINT_roof_membrane", "#f1f1ec"), mats.named("Facade_Concrete"), mats.tint("TINT_trim", "#3b3e42")])


def hall(o, cat, p, tier, style, opts):
    t = tier_spec(cat, "indoor_hall", tier) or {}
    club = o.club
    size = t.get("size") or [64.0, 44.0]
    bw, bd = _f(size[0], 64.0), _f(size[1], 44.0)
    h = num(t, "height", 13.0)
    f = p.frame.sub(0.0, 0.0, 0.0)
    w, d = min(bw, p.w - 8.0), min(bd, p.d - 10.0)
    apron = kernel.Mesh()
    kernel.ground(apron, kernel.rect(w + 6.0, d + 6.0), COVER_Z - 0.005, 0)
    o.part_at(f, f"{p.id}:apron", apron, [mats.named("Apron_Concrete")])
    if tier == 1:
        air_dome(o, f, p.id, w / 2.0, d / 2.0, h)
        return
    yaw = math.atan2(p.front[0], -p.front[1])
    along_x = abs(p.front[1]) > 0.5
    ww, dd = (w, d) if along_x else (d, w)
    hf = p.frame.sub(0.0, 0.0, yaw)
    if tier == 2:
        sp = buildings.styled(cat, "industrial", ww, dd, 1, club.primary, o.night, opts, p.seed)
        sp.mats = list(sp.mats)
        sp.mats[buildings.T_WALL] = mats.tint("TINT_corrugated", "#7b8a83")
    else:
        sp = buildings.styled(cat, style, ww, dd, 1, club.primary, o.night, opts, p.seed)
    sp.storey_h = h - buildings.PLINTH
    sp.roof = "shallow" if style == "industrial" or tier == 2 else "barrel"
    sp.pitch_deg = 6.0
    sp.window = "strip"
    sp.door = "glass"
    sp.canopy = "none"
    sp.clock = False
    sp.arcade = False
    sp.quoins = False
    sp.fins = False
    sp.roller_doors = 2
    sp.mats[buildings.T_DOOR] = mats.tint("TINT_door", "#6e7377")
    sg = option(cat, "indoor_hall", tier, opts, "sign")
    if sg if isinstance(sg, bool) else True:
        sp.sign = club.name
    buildings.build(o, hf, f"{p.id}:hall", sp)
    if tier >= 3:
        e = buildings.styled(cat, style, 28.0, 9.0, 2, club.primary, o.night, opts, p.seed + 3)
        e.roof = "hip" if e.roof in ("gable", "hip") else e.roof
        e.clock = False
        e.arcade = False
        buildings.build(o, hf.sub(0.0, -dd / 2.0 - 4.4, 0.0), f"{p.id}:entrance", e)


def recovery(o, cat, p, tier, style):
    t = tier_spec(cat, "recovery", tier) or {}
    club = o.club
    if tier == 1:
        cover(o, p, "deck", 2.0, mats.named("Site_Wood"))
        f = place(p, 14.0, 6.0, 4.0).f
        o.prop_at(f, "prop.site_cabin", -3.0, 1.0, COVER_Z, HALF_PI, [("TINT_cabin", "#f2f3f0")])
        m = kernel.Mesh()
        cylinder(m, 6.5, 0.5, COVER_Z, COVER_Z + 2.4, 1.1, 20, 0, True)
        for i in range(3):
            bx(m, -9.0 + i * 2.2, -4.5, COVER_Z, -7.4 + i * 2.2, -2.6, COVER_Z + 0.75, 1)
        o.part_at(f, f"{p.id}:cryo", m, [mats.named("Galvanised"), mats.tint("TINT_trim", "#e9eae6")])
        return
    size = t.get("size") or [24.0, 14.0]
    pl = place(p, _f(size[0], 24.0), _f(size[1], 14.0), 8.0)
    f, w, d = pl.f, pl.w, pl.d
    forecourt(o, p, pl)
    sp = buildings.styled(cat, style, w, d, 1, club.primary, o.night, None, p.seed)
    sp.window = "large" if tier == 3 else sp.window
    sp.roof = "hip" if sp.roof == "gable" else sp.roof
    sp.clock = False
    sp.arcade = False
    buildings.build(o, f, f"{p.id}:building", sp)
    pf = f.sub(w / 2.0 + 4.0, 0.0, 0.0)
    deck = kernel.Mesh()
    bx(deck, -2.6, -d / 2.0 + 1.0, COVER_Z, 2.6, d / 2.0 - 1.0, COVER_Z + 0.5, 0)
    water = kernel.Mesh()
    for i in range(2):
        y = -d / 4.0 + i * d / 2.0 - 1.4
        kernel.ground(water, kernel.rect(3.0, 2.8, 0.0, y + 1.4), COVER_Z + 0.45, 0)
    o.part_at(pf, f"{p.id}:pool_deck", deck, [mats.named("Site_Wood")])
    o.part_at(pf, f"{p.id}:pools", water, [mats.named("Site_Water")])



def arrow(m, x, y, sx, z):
    """A painted lane arrow pointing along +x (sx = 1) or -x: a 3 m shaft and a head."""
    kernel.ground(m, kernel.rect(3.0, 0.22, x - sx * 0.8, y), z, 0)
    a, b, c = (x + sx * 0.7, y - 0.6), (x + sx * 1.9, y), (x + sx * 0.7, y + 0.6)
    kernel.ground(m, [a, b, c] if sx > 0.0 else [c, b, a], z, 0)


def drop_off(o, cat, p, tier, opts):
    """Arrival for an academy (players are dropped off, not parked): a lay-by for parents along a
    footway, then a one-way loop round a planted island with minibus bays in yellow boxes, zebra
    crossings and lane arrows; at the top tier a security hut, an entry barrier and CCTV (plot
    local, the long side along x, entry at -x)."""
    t = tier_spec(cat, "drop_off", tier) or {}
    cover(o, p, "surface", 1.0, mats.named("Site_Asphalt"))
    # turned round when the road is at +x, so the way in faces it
    flip = p.front[0] > 0.5
    f = p.frame.sub(0.0, 0.0, math.pi) if flip else p.frame
    hl, hd = (p.w - 2.0) / 2.0, (p.d - 2.0) / 2.0
    z = COVER_Z + 0.004
    paint, yellow, kerb, paving, beds = kernel.Mesh(), kernel.Mesh(), kernel.Mesh(), kernel.Mesh(), kernel.Mesh()
    # kerbed footways along both long edges
    for s in (-1.0, 1.0):
        kernel.ground(paving, kernel.rect(2.0 * hl, 2.5, 0.0, s * (hd - 1.25)), COVER_Z + 0.12, 0)
        strip(kerb, [(-hl, s * (hd - 2.5)), (hl, s * (hd - 2.5))], 0.2, COVER_Z + 0.125, 0)
    o.part_at(f, f"{p.id}:footway", paving, [mats.named("Plaza_Paving")])
    # the parents' lay-by: 6 m bays along the north footway, cars stopping in some of them
    yb = hd - 2.5 - 2.4
    nb = int(max(2.0 * hl - 24.0, 6.0) / 6.0)
    x0 = -nb * 3.0
    strip(paint, [(x0, yb), (x0 + nb * 6.0, yb)], 0.1, z, 0)
    for k in range(nb + 1):
        strip(paint, [(x0 + k * 6.0, yb), (x0 + k * 6.0, hd - 2.6)], 0.1, z, 0)
    lay = [((x0 + (k + 0.5) * 6.0, yb + 1.2), -HALF_PI) for k in range(nb)]
    if flip:
        lay = [((-c[0], -c[1]), a + math.pi) for c, a in lay]
    park(o, p, lay, num(t, "cars", 0.4), False, 620, reverse=False)
    if tier >= 2:
        # the island: kerbed bed with trees, the loop round it one way (east on the north lane)
        il, iw = 2.0 * hl - 36.0, 8.0
        r = kernel.rect(il, iw)
        strip(kerb, [r[0], r[1], r[2], r[3], r[0]], 0.25, COVER_Z + 0.125, 0)
        kernel.ground(beds, kernel.rect(il - 0.4, iw - 0.4), COVER_Z + 0.11, 0)
        n = int(max(il / 12.0, 1.0))
        for k in range(n):
            x = -il / 2.0 + (k + 0.5) * il / n
            if tier >= 3 and k == 0:
                continue            # the security hut's place
            landscape.tree(o, "prop.tree_birch", f.pt(x, 0.0), COVER_Z + 0.11, p.seed + 640, k)
            o.prop_at(f, "prop.shrub", x + il / n / 2.0 - 1.5, 1.6, COVER_Z + 0.11, uniform(p.seed + 641, k) * TAU)
        for sx, yl in ((1.0, (iw / 2.0 + yb) / 2.0), (-1.0, -(iw / 2.0 + hd - 2.5) / 2.0)):
            for x in (-il / 4.0, il / 4.0):
                arrow(paint, x, yl, sx, z)
        # zebra crossings from each end of the island to both footways
        for x in (-il / 2.0 + 3.0, il / 2.0 - 3.0):
            for y_a, y_b in ((iw / 2.0, yb), (-(hd - 2.5), -iw / 2.0)):
                y = y_a + 0.3
                while y + 0.5 <= y_b - 0.2:
                    kernel.ground(paint, kernel.rect(2.4, 0.5, x, y + 0.25), z, 0)
                    y += 1.0
        # minibus bays on the south side, boxed in yellow, the club's minibuses in some
        yc = -(hd - 2.5 - 1.75)
        for k in range(3):
            x = -13.0 + k * 13.0
            q = kernel.rect(10.0, 3.3, x, yc)
            strip(yellow, [q[0], q[1], q[2], q[3], q[0]], 0.12, z, 0)
            if uniform(p.seed + 650, k) < 0.7:
                o.prop_at(f, "prop.minibus", x, yc, COVER_Z, -HALF_PI if k % 2 == 0 else HALF_PI, [("TINT_van_paint", o.club.primary)])
        for x in (-il / 2.0 + 8.0, il / 2.0 - 8.0):
            o.prop_at(f, "prop.lamp_post", x, 0.0, COVER_Z + 0.11, 0.0)
            if o.night:
                q = f.pt(x, 1.2)
                o.spot((q[0], q[1], 7.6), (0.0, 0.0, -1.0), 400.0, 120.0, (1.0, 0.86, 0.7))
        for x in (-6.0, 6.0):
            o.prop_at(f, "prop.bench", x, hd - 1.0, COVER_Z + 0.12, math.pi)
        o.prop_at(f, "prop.planter", 0.0, hd - 1.0, COVER_Z + 0.12, 0.0)
        o.prop_at(f, "prop.sign_post", -hl + 3.0, hd - 2.0, COVER_Z + 0.12, -HALF_PI)
    if tier >= 3:
        # controlled entry: a security hut on the island nose, a barrier across the way in, CCTV
        # (on the island's nose, past the zebra's landing; the arm swings across the way in)
        o.prop_at(f, "prop.security_hut", -il / 2.0 + 6.5, 0.6, COVER_Z + 0.11, 0.0)
        o.prop_at(f, "prop.barrier_arm", -il / 2.0 + 9.5, iw / 2.0 - 0.5, COVER_Z + 0.11, HALF_PI)
        for x, y in ((-hl + 1.0, hd - 0.6), (hl - 1.0, hd - 0.6), (-hl + 1.0, -hd + 0.6), (hl - 1.0, -hd + 0.6)):
            o.prop_at(f, "prop.cctv_pole", x, y, COVER_Z + 0.12, math.atan2(-x, y))
        for k in range(3):
            o.prop_at(f, "prop.bike_rack", hl - 10.0 + k * 1.0, -hd + 1.2, COVER_Z + 0.12, 0.0)
    o.part_at(f, f"{p.id}:paint", paint, [mats.tint("TINT_road_paint", "#ecebe4")])
    o.part_at(f, f"{p.id}:yellow", yellow, [mats.tint("TINT_road_paint", "#e3b92a")])
    o.part_at(f, f"{p.id}:kerb", kerb, [mats.named("Site_Kerb")])
    o.part_at(f, f"{p.id}:beds", beds, [mats.lawn("amenity", f.pt(0.0, 0.0), f.yaw)])


def car_park(o, cat, p, tier, opts):
    t = tier_spec(cat, "car_park", tier) or {}
    cover(o, p, "surface", 1.0, mats.named(text(t, "surface", "Site_Asphalt")))

    def b(k):
        return t.get(k) is True
    paint = kernel.Mesh()
    spots = bays(p.w, p.d, 2.2, paint, COVER_Z + 0.004, lambda _k: tier == 1)
    if tier >= 2:
        o.part_at(p.frame, f"{p.id}:paint", paint, [mats.tint("TINT_road_paint", "#ecebe4")])
        k = kernel.Mesh()
        r = kernel.rect(p.w - 2.0, p.d - 2.0)
        strip(k, [r[0], r[1], r[2], r[3], r[0]], 0.25, COVER_Z + 0.1, 0)
        o.part_at(p.frame, f"{p.id}:kerb", k, [mats.named("Site_Kerb")])
    else:
        r = kernel.rect(p.w - 2.0, p.d - 2.0)
        for k, (x, y, _dx, _dy) in enumerate(kernel.along([r[0], r[1], r[2], r[3], r[0]], 4.0, 0.5)):
            if k % 2 == 0:
                o.prop_at(p.frame, "prop.bollard", x, y, COVER_Z, 0.0)
    if b("islands"):
        # kerbed planting beds at the ends of each row of bays (and one every 12 bays along a long
        # row): a tree, or under a solar canopy low shrubs
        islands, beds = kernel.Mesh(), kernel.Mesh()
        kept = []
        rows = {}
        for i, s in enumerate(spots):
            rows.setdefault(rround(s[0][1] * 10.0), []).append(i)
        bed = set()
        for idx in rows.values():
            if len(idx) >= 6:
                bed.update((idx[0], idx[-1]))
                bed.update(idx[k] for k in range(12, len(idx) - 6, 12))
        for i, s in enumerate(spots):
            if i in bed:
                cx, cy = s[0]
                for x0, y0, x1, y1 in ((cx - 1.2, cy - 2.4, cx + 1.2, cy - 2.25), (cx - 1.2, cy + 2.25, cx + 1.2, cy + 2.4),
                                       (cx - 1.2, cy - 2.25, cx - 1.05, cy + 2.25), (cx + 1.05, cy - 2.25, cx + 1.2, cy + 2.25)):
                    bx(islands, x0, y0, COVER_Z, x1, y1, COVER_Z + 0.15, 0)
                kernel.ground(beds, kernel.rect(2.1, 4.5, cx, cy), COVER_Z + 0.11, 0)
                if b("canopy"):
                    for k in (-1, 1):
                        o.prop_at(p.frame, "prop.shrub", cx, cy + k * 1.15, COVER_Z + 0.11, uniform(p.seed + 601, 2 * i + (k + 1) // 2) * 6.283)
                else:
                    landscape.tree(o, "prop.tree_birch", p.frame.pt(cx, cy), COVER_Z + 0.11, p.seed + 600, i)
            else:
                kept.append(s)
        spots = kept
        o.part_at(p.frame, f"{p.id}:islands", islands, [mats.named("Site_Kerb")])
        o.part_at(p.frame, f"{p.id}:beds", beds, [mats.named("Site_Dirt")])
    if b("ev"):
        ev = kernel.Mesh()
        for s in spots[:6]:
            kernel.ground(ev, kernel.rect(2.3, 4.8, s[0][0], s[0][1]), COVER_Z + 0.005, 0)
            back = (0.0, -2.7 if s[1] == 0.0 else 2.7)
            o.prop_at(p.frame, "prop.ev_charger", s[0][0] + back[0], s[0][1] + back[1], COVER_Z, s[1])
        o.part_at(p.frame, f"{p.id}:ev_bays", ev, [mats.tint("TINT_road_paint", "#2f8a4f")])
    fill = min(max(_f(option(cat, "car_park", tier, opts, "fill"), 0.6), 0.0), 1.0)
    park(o, p, spots, fill, False, 610)
    o.parking += len(spots)
    if b("lamps"):
        n = int(max(math.floor(p.w / 24.0), 1.0))
        for i in range(n):
            x = -p.w / 2.0 + (i + 0.5) * p.w / n
            for y in (-p.d / 2.0 + 1.4, p.d / 2.0 - 1.4):
                o.prop_at(p.frame, "prop.lamp_post", x, y, COVER_Z, 0.0 if y < 0.0 else math.pi)
                if o.night:
                    q = p.frame.pt(x, y + (1.2 if y < 0.0 else -1.2))
                    o.spot((q[0], q[1], 7.6), (0.0, 0.0, -1.0), 400.0, 120.0, (1.0, 0.86, 0.7))
        o.prop_at(p.frame, "prop.ticket_machine", p.w / 2.0 - 3.0, -p.d / 2.0 + 2.0, COVER_Z, 0.0)
    if b("canopy"):
        deck = kernel.Mesh()
        cols = []
        ys = sorted({rround(s[0][1] * 10.0) / 10.0 for s in spots})
        seen = set()
        for y in ys:
            row = [s for s in spots if abs(s[0][1] - y) < 0.2]
            x0 = min(s[0][0] for s in row) - 1.25
            x1 = max(s[0][0] for s in row) + 1.25
            quad(deck, (x0, y - 2.6, 3.0), (x1, y - 2.6, 3.0), (x1, y + 2.6, 3.7), (x0, y + 2.6, 3.7), 0)
            # columns at the back of the bays, on the lines between cars (every third), never in a bay
            yb = y - 2.45 if row[0][1] == 0.0 else y + 2.45
            zb = 3.0 + 0.7 * (yb - (y - 2.6)) / 5.2
            n = int(max(rround((x1 - x0) / 7.5), 1.0))
            xs = [x0 + min(j * 7.5, x1 - x0) for j in range(n)] + [x1]
            for x in xs:
                cols.append(([(x, y - 2.6, 2.95), (x, y + 2.6, 3.65)], 0.06))
                key = (rround(x * 10.0), rround(yb * 10.0))
                if key not in seen:
                    seen.add(key)
                    cols.append(([(x, yb, COVER_Z), (x, yb, zb - 0.08)], 0.11))
            cols.append(([(x0, yb, zb - 0.1), (x1, yb, zb - 0.1)], 0.07))
        o.part_at(p.frame, f"{p.id}:canopy", deck, [mats.named("Solar_Panel")])
        o.tubes_at(p.frame, f"{p.id}:canopy_steel", cols, mats.named("Galvanised"), 8)


def maintenance(o, cat, p, tier, style):
    t = tier_spec(cat, "maintenance", tier) or {}
    cover(o, p, "yard", 1.0, mats.named("Site_Gravel" if tier == 1 else "Apron_Concrete"))
    heaps, sand = kernel.Mesh(), kernel.Mesh()
    if tier == 1:
        f = place(p, 14.0, 6.0, 6.0).f
        for i in range(2):
            o.prop_at(f, "prop.storage_container", -3.5 + i * 7.0, 0.0, COVER_Z, HALF_PI * 0.0, [("TINT_container", "#3c5a3a")])
        w, d = 14.0, 6.0
    else:
        size = t.get("size") or [20.0, 12.0]
        pl = place(p, _f(size[0], 20.0), _f(size[1], 12.0), 9.0)
        f, w, d = pl.f, pl.w, pl.d
        sp = buildings.styled(cat, "industrial" if style == "ultra_modern" else style, w, d, 1, o.club.primary, o.night, None, p.seed)
        sp.storey_h = 5.6
        sp.window = "strip"
        sp.roller_doors = 3 if tier == 3 else 2
        sp.door = "none"
        sp.canopy = "none"
        sp.clock = False
        sp.arcade = False
        sp.quoins = False
        sp.mats[buildings.T_DOOR] = mats.tint("TINT_door", "#6e7377")
        sp.roof = sp.roof if sp.roof in ("flat", "mono") else "shallow"
        buildings.build(o, f, f"{p.id}:shed", sp)
        if tier == 3:
            cf = f.sub(w / 2.0 + 6.0, 0.0, 0.0)
            c = kernel.Mesh()
            bx(c, -5.0, -d / 2.0, 4.5, 5.0, d / 2.0, 4.7, 0)
            o.part_at(cf, f"{p.id}:bays_roof", c, [mats.tint("TINT_roof", "#5d6166")])
            cols = []
            for x in (-4.8, 0.0, 4.8):
                for y in (-d / 2.0 + 0.2, d / 2.0 - 0.2):
                    cols.append(([(x, y, COVER_Z), (x, y, 4.5)], 0.1))
            o.tubes_at(cf, f"{p.id}:bays_steel", cols, mats.named("Galvanised"), 8)
            o.prop_at(cf, "prop.tractor", 0.0, 0.0, COVER_Z, HALF_PI, [("TINT_tractor", "#2f6b2f")])
    yard = f.sub(0.0, -d / 2.0 - 5.5, 0.0)
    o.prop_at(yard, "prop.ride_mower", -w / 2.0 + 2.0, 0.0, COVER_Z, 0.4)
    if tier >= 2:
        o.prop_at(yard, "prop.ride_mower", -w / 2.0 + 4.5, 0.3, COVER_Z, 0.5)
        o.prop_at(yard, "prop.tractor", w / 2.0 - 3.0, -0.5, COVER_Z, -0.3, [("TINT_tractor", "#2f6b2f")])
        o.prop_at(f, "prop.fuel_tank", -w / 2.0 - 2.5, d / 2.0 - 2.0, COVER_Z, 0.0)
    # the groundsman's hand kit along the front of the shed
    o.prop_at(yard, "genprop.line_marker", -2.5, 3.9, COVER_Z, 0.3)
    o.prop_at(yard, "genprop.wheelbarrow_tools", -0.6, 4.0, COVER_Z, -0.2)
    o.prop_at(yard, "genprop.hose_reel", 1.4, 4.4, COVER_Z, 0.0)
    o.prop_at(yard, "genprop.divot_bucket", 2.7, 4.5, COVER_Z, 0.0)
    if tier >= 2:
        o.prop_at(yard, "genprop.cylinder_mower", 4.6, 4.0, COVER_Z, 0.2)
        o.prop_at(yard, "genprop.pitch_sprinkler", -4.4, 4.2, COVER_Z, 0.0)
    mound(sand, 2.0, -1.0, 2.6, 2.0, 1.3, p.seed + 700, 0)
    mound(heaps, 6.5, -1.0, 2.4, 2.0, 1.2, p.seed + 701, 0)
    o.part_at(yard, f"{p.id}:sand", sand, [mats.named("Site_Sand")], 75.0)
    o.part_at(yard, f"{p.id}:topsoil", heaps, [mats.named("Site_Dirt")], 75.0)
    if tier == 1:
        hw, hd = p.w / 2.0 - 1.5, p.d / 2.0 - 1.5
        mid = (p.front[0] * hw, p.front[1] * hd)
        tang = (-p.front[1], p.front[0])
        corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        start = next((i for i, c in enumerate(corners) if dot(sub(c, mid), tang) > 0.0 and dot(sub(c, mid), p.front) > -0.1), 0)
        ring = [add(mid, mul(tang, 3.0))] + [corners[(start + k) % 4] for k in range(4)] + [sub(mid, mul(tang, 3.0))]
        landscape.chainlink(o, [p.frame.pt(q[0], q[1]) for q in ring], 2.4, f"{p.id}:fence")
