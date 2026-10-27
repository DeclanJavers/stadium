"""Demolition, cleared ground and construction sites (mirrors viewer/gen/src/site/construction.rs).

Demolition: the occupant half knocked down (occupants.py), welded-mesh fencing round the plot, an
excavator and skips. Cleared: levelled topsoil and survey pegs. Construction: branded hoardings,
site cabins and a toilet, and the works: for a building its frame rising inside scaffolding with a
tower crane; for a pitch the earthworks with drainage; for a car park the sub-base going down.
"""
import math

from ..stands.prng import uniform
from . import kernel, mats
from .facilities import footprint, pitch_dims
from .kernel import along, box as bx, quad, strip
from .model import COVER_Z, Placement, add, dot, mul, normalize, sub
from .occupants import cover, mound

HALF_PI = math.pi / 2


def perimeter(o, p, key, pitch, inset, opening, tints):
    """Panels (a prop every `pitch` metres) round the plot, inset, with an opening at the frontage."""
    hw, hd = p.w / 2.0 - inset, p.d / 2.0 - inset
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    mid = (p.front[0] * hw, p.front[1] * hd)
    tang = (-p.front[1], p.front[0])
    start = next((i for i, c in enumerate(corners) if dot(sub(c, mid), tang) > 0.0 and dot(sub(c, mid), p.front) > -0.1), 0)
    ring = [add(mid, mul(tang, opening / 2.0))] + [corners[(start + k) % 4] for k in range(4)] + [sub(mid, mul(tang, opening / 2.0))]
    for x, y, dx, dy in along(ring, pitch, 0.5):
        o.prop_at(p.frame, key, x, y, COVER_Z, math.atan2(-dx, dy) + HALF_PI, tints)


def demolition(o, p):
    perimeter(o, p, "prop.heras_panel", 3.5, 1.5, 8.0, [])
    inner = mul(p.front, min(p.d, p.w) / 2.0 - 9.0)
    o.prop_at(p.frame, "prop.excavator", inner[0], inner[1], COVER_Z, uniform(p.seed + 800, 0) * 6.283)
    tang = (-p.front[1], p.front[0])
    for i in range(2):
        q = add(inner, mul(tang, 8.0 + i * 4.0))
        o.prop_at(p.frame, "prop.skip", q[0], q[1], COVER_Z, math.atan2(-p.front[0], p.front[1]), [("TINT_skip", "#c9a227")])


def cleared(o, p):
    cover(o, p, "ground", 1.0, mats.named("Site_Dirt"))
    for i, (x, y) in enumerate(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))):
        o.prop_at(p.frame, "prop.survey_peg", x * (p.w / 2.0 - 3.0), y * (p.d / 2.0 - 3.0), COVER_Z, float(i))
    q = mul(p.front, (p.d if abs(p.front[1]) > 0.5 else p.w) / 2.0 - 2.5)
    o.prop_at(p.frame, "prop.site_board", q[0], q[1], COVER_Z, math.atan2(-p.front[0], p.front[1]), [("TINT_club", o.club.primary)])


def frame_and_scaffold(o, f, name, w, d, levels, sh):
    """A concrete frame (slabs and columns) up to `levels` storeys, with scaffolding round it."""
    m = kernel.Mesh()
    z0 = 0.45
    bx(m, -w / 2.0, -d / 2.0, 0.0, w / 2.0, d / 2.0, z0, 0)
    nx = int(max(_round(w / 6.0), 1.0))
    ny = int(max(_round(d / 6.0), 1.0))
    for lv in range(levels):
        z = z0 + lv * sh
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = -w / 2.0 + 0.25 + (w - 0.5) * i / nx
                y = -d / 2.0 + 0.25 + (d - 0.5) * j / ny
                bx(m, x - 0.2, y - 0.2, z, x + 0.2, y + 0.2, z + sh - 0.3, 0)
        bx(m, -w / 2.0, -d / 2.0, z + sh - 0.3, w / 2.0, d / 2.0, z + sh, 0)
    o.part_at(f, f"{name}:frame", m, [mats.named("Facade_Concrete")])
    top = z0 + levels * sh + 1.0
    ax, ay = w / 2.0 + 1.2, d / 2.0 + 1.2
    ring = [(-ax, -ay), (ax, -ay), (ax, ay), (-ax, ay), (-ax, -ay)]
    t = [([(x, y, COVER_Z), (x, y, top)], 0.024) for x, y, _dx, _dy in along(ring, 2.5, 0.0)]
    boards = kernel.Mesh()
    z = 2.0
    while z < top:
        t.append(([(q[0], q[1], z) for q in ring], 0.024))
        t.append(([(q[0], q[1], z + 1.0) for q in ring], 0.02))
        for i in range(4):
            a, b = ring[i], ring[i + 1]
            n = normalize(sub(b, a))
            inw = (-n[1], n[0])
            a2, b2 = add(a, mul(inw, 0.95)), add(b, mul(inw, 0.95))
            quad(boards, (a[0], a[1], z), (b[0], b[1], z), (b2[0], b2[1], z), (a2[0], a2[1], z), 0)
        z += 2.0
    o.tubes_at(f, f"{name}:scaffold", t, mats.named("Galvanised"), 5)
    o.part_at(f, f"{name}:boards", boards, [mats.named("Site_Plywood")])


def _round(x):
    from .model import rround
    return rround(x)


def site(o, cat, p, fac, tier, _style):
    club = o.club
    perimeter(o, p, "prop.hoarding_panel", 2.44, 1.2, 8.0, [("TINT_hoarding", club.primary)])
    tang = (-p.front[1], p.front[0])
    along_x = abs(p.front[1]) > 0.5
    pw, pd = (p.w, p.d) if along_x else (p.d, p.w)
    corner = add(mul(p.front, pd / 2.0 - 5.0), mul(tang, pw / 2.0 - 9.0))
    yaw = math.atan2(-tang[0], tang[1])
    o.prop_at(p.frame, "prop.site_cabin", corner[0], corner[1], COVER_Z, yaw, [("TINT_cabin", "#d7d9d4")])
    o.prop_at(p.frame, "prop.site_cabin", corner[0], corner[1], COVER_Z + 2.62, yaw, [("TINT_cabin", "#d7d9d4")])
    wc = sub(corner, mul(tang, 7.0))
    o.prop_at(p.frame, "prop.portaloo", wc[0], wc[1], COVER_Z, yaw + math.pi)
    pal = sub(corner, mul(p.front, 8.0))
    o.prop_at(p.frame, "prop.pallet_stack", pal[0], pal[1], COVER_Z, 0.3)
    o.prop_at(p.frame, "prop.pipe_stack", pal[0] - tang[0] * 4.0, pal[1] - tang[1] * 4.0, COVER_Z, yaw)
    if fac in ("pitch", "artificial_pitch", "goalkeeper_area", "youth_pitches", "cages", "showpiece_pitch"):
        cover(o, p, "earthworks", 1.0, mats.named("Site_Dirt"))
        l, w = (min(34.0, p.w - 8.0), min(26.0, p.d - 6.0)) if fac == "goalkeeper_area" else pitch_dims(p, 105.0, 68.0, 4.0, 3.0)
        d = kernel.Mesh()
        strip(d, [(-l / 2.0, 0.0), (l / 2.0, 0.0)], 0.6, COVER_Z + 0.01, 0)
        n = int(l / 6.0)
        for i in range(n + 1):
            x = -l / 2.0 + l * i / max(n, 1)
            for s in (-1.0, 1.0):
                strip(d, [(x, 0.0), (x + 4.0, s * w / 2.0)], 0.35, COVER_Z + 0.008, 0)
        o.part_at(p.frame, f"{p.id}:drains", d, [mats.named("Site_Gravel")])
        heaps = kernel.Mesh()
        for i in range(4):
            mound(heaps, -l / 2.0 + 8.0 + i * 9.0, w / 2.0 + 2.0, 4.0, 3.0, 2.0, p.seed + 810 + i, 0)
        o.part_at(p.frame, f"{p.id}:topsoil", heaps, [mats.named("Site_Dirt")], 75.0)
        o.prop_at(p.frame, "prop.excavator", -l / 4.0, -w / 4.0, COVER_Z, 0.6)
        o.prop_at(p.frame, "prop.dumper", l / 4.0, w / 6.0, COVER_Z, 2.2)
        o.prop_at(p.frame, "prop.roller", 0.0, -w / 3.0, COVER_Z, HALF_PI)
    elif fac == "car_park":
        cover(o, p, "subbase", 1.0, mats.named("Site_Gravel"))
        k = kernel.Mesh()
        r = kernel.rect(p.w - 2.0, p.d - 2.0)
        strip(k, [r[0], r[1], r[2]], 0.25, COVER_Z + 0.1, 0)
        o.part_at(p.frame, f"{p.id}:kerb", k, [mats.named("Site_Kerb")])
        o.prop_at(p.frame, "prop.roller", 0.0, 0.0, COVER_Z, 0.0)
        o.prop_at(p.frame, "prop.dumper", p.w / 4.0, -p.d / 6.0, COVER_Z, 1.0)
    else:
        cover(o, p, "ground", 1.0, mats.named("Site_Dirt"))
        fp = footprint(cat, p, fac, tier)
        if fp is not None:
            pl, storeys, sh = fp
            if fac == "indoor_hall" and tier == 1:
                m = kernel.Mesh()
                bx(m, -pl.w / 2.0, -pl.d / 2.0, COVER_Z, pl.w / 2.0, pl.d / 2.0, COVER_Z + 0.4, 0)
                bx(m, -6.0, -2.0, COVER_Z + 0.4, 6.0, 2.0, COVER_Z + 2.2, 1)
                o.part_at(pl.f, f"{p.id}:slab", m, [mats.named("Facade_Concrete"), mats.tint("TINT_roof_membrane", "#eeeeea")])
            elif fac == "indoor_hall":
                t = []
                n = int(_round(pl.w / 6.0))
                h = sh + 0.45
                for i in range(n // 2 + 1):
                    x = -pl.w / 2.0 + pl.w * i / n
                    rise = 0.14 * pl.d
                    arc = [(x, -pl.d / 2.0, 0.0)]
                    for k in range(11):
                        y = -pl.d / 2.0 + pl.d * k / 10.0
                        q = y / (pl.d / 2.0)
                        arc.append((x, y, h + rise * (1.0 - q * q)))
                    arc.append((x, pl.d / 2.0, 0.0))
                    t.append((arc, 0.25))
                o.tubes_at(pl.f, f"{p.id}:portals", t, mats.tint("TINT_trim", "#d6d9d6"), 6)
                m = kernel.Mesh()
                bx(m, -pl.w / 2.0, -pl.d / 2.0, COVER_Z, pl.w / 2.0, pl.d / 2.0, COVER_Z + 0.3, 0)
                o.part_at(pl.f, f"{p.id}:slab", m, [mats.named("Facade_Concrete")])
                c = pl.f.pt(pl.w / 2.0 + 6.0, pl.d / 2.0 + 3.0)
                o.prop("prop.tower_crane", Placement((c[0], c[1], COVER_Z), pl.f.yaw + math.pi * 1.2))
            else:
                levels = storeys - 1 if storeys > 1 else 1
                frame_and_scaffold(o, pl.f, p.id, pl.w, pl.d, levels, sh)
                c = pl.f.pt(pl.w / 2.0 + 5.0, pl.d / 2.0 + 2.0)
                o.prop("prop.tower_crane", Placement((c[0], c[1], COVER_Z), pl.f.yaw + math.pi * 1.25))
        else:
            cover(o, p, "pad", 4.0, mats.named("Site_Gravel"))
