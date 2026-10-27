"""Parametric buildings (mirrors viewer/gen/src/site/buildings.rs): a rectangular block of storeys
with a plinth, windows and doors laid out per facade, a roof (flat with parapet, gable, hip,
mono-pitch, barrel), and the details that make a style (quoins and a portico, exposed frames and
roller doors, timber and solar panels, render and an arcade, fins and an LED band). The same
generator makes the club's buildings and, with decay, the derelict ones the club inherits.

A building's local frame: centred on its footprint, front face towards -y, z up from the ground.
"""
import copy
import json
import math

from ..stands.prng import uniform
from . import kernel, mats
from .kernel import box as bx, cylinder, quad, tri
from .model import num, rround, text

T_WALL, T_BASE, T_ROOF, T_GLASS, T_TRIM, T_DOOR, T_ACCENT, T_METAL = 0, 1, 2, 3, 4, 5, 6, 7
T_STONE, T_LED, T_SOLAR, T_BOARD, T_DARK, T_LIT, T_BLIND, T_GRAVEL = 8, 9, 10, 11, 12, 13, 14, 15
PLINTH = 0.45


class Spec:
    """A building: size, storeys, roof (flat | gable | hip | mono | shallow | barrel | none), window
    (ribbon | punched | strip | large | shuttered | none), door (glass | timber | none), canopy
    (cantilever | portico | steel | timber | arcade | none), style flags, decay (0 kept, 1 derelict),
    sign text, the lit share of windows at night, seed and the materials by tag."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def copy(self):
        return copy.copy(self)


def materials(wall, wall_color, base, roof, roof_color, trim_color, club, night):
    """The materials of a building, by tag."""
    return [
        mats.tint(wall, wall_color),
        mats.named(base),
        mats.tint(roof, roof_color) if roof_color else mats.named(roof),
        mats.named("Facade_DarkGlass"),
        mats.tint("TINT_trim", trim_color),
        mats.tint("TINT_door", club),
        mats.tint("TINT_club", club),
        mats.tint("TINT_trim", "#8b8e91"),
        mats.named("Facade_Stone"),
        mats.glow("TINT_club", club, 3.0 if night else 0.4),
        mats.named("Solar_Panel"),
        mats.named("Site_Plywood"),
        mats.named("Bldg_Opening"),
        mats.glow("Bldg_Glass_Lit", "#ffd8a6", 1.6) if night else mats.named("Facade_DarkGlass"),
        mats.tint("Facade_DarkGlass", "#8f8b82"),   # blinds behind the glazing (its reflection over them)
        mats.named("Site_Gravel"),                  # pebble margins round a green roof
    ]


def styled(cat, style, w, d, storeys, club, night, opts, seed):
    """A building spec in a catalogued style (catalog/styles.json) with the player's colours."""
    st = cat["styles"].get(style)
    st = st if isinstance(st, dict) else {}
    opts = opts if isinstance(opts, dict) else {}
    wall_color = opts["wall_color"] if isinstance(opts.get("wall_color"), str) else text(st, "wall_color", "#c8c4bb")
    trim_color = opts["trim_color"] if isinstance(opts.get("trim_color"), str) else text(st, "trim_color", "#2a2c30")
    roof_mat = text(st, "roof_mat", "Roof_Flat")
    roof_color = "#5d6166" if roof_mat == "TINT_roof" else None

    def b(k):
        return st.get(k) is True
    return Spec(
        w=w, d=d, storeys=storeys, storey_h=num(st, "storey_h", 4.0), roof=text(st, "roof", "flat"),
        pitch_deg=num(st, "pitch_deg", 30.0), eaves=num(st, "eaves", 0.4), parapet=num(st, "parapet", 0.8),
        window=text(st, "window", "punched"), door="timber" if style == "classic" else "glass", roller_doors=0,
        canopy=text(st, "canopy", "none"), quoins=b("quoins"), clock=b("clock"), fins=b("fins"), led_band=b("led_band"),
        arcade=b("arcade"), solar=b("solar"), frame=b("frame"), decay=0.0, sign=None, lit=0.7 if night else 0.0, seed=seed,
        mats=materials(text(st, "wall", "Facade_Concrete"), wall_color, text(st, "base", "Facade_Concrete"), roof_mat, roof_color,
                       trim_color, club, night))


# -- facade coordinates: side 0 front (-y), 1 right (+x), 2 back (+y), 3 left (-x); s runs along the
# side seen from outside (left to right), `out` is the distance in front of the wall

def side_len(w, d, side):
    return w if side % 2 == 0 else d


def fpt(w, d, side, s, z, out):
    if side == 0:
        return (-w / 2.0 + s, -d / 2.0 - out, z)
    if side == 1:
        return (w / 2.0 + out, -d / 2.0 + s, z)
    if side == 2:
        return (w / 2.0 - s, d / 2.0 + out, z)
    return (-w / 2.0 - out, d / 2.0 - s, z)


def fbox(m, w, d, side, s0, s1, z0, z1, out0, out1, tag):
    a = fpt(w, d, side, s0, z0, out0)
    b = fpt(w, d, side, s1, z1, out1)
    bx(m, min(a[0], b[0]), min(a[1], b[1]), z0, max(a[0], b[0]), max(a[1], b[1]), z1, tag)


def fquad(m, w, d, side, s0, s1, z0, z1, out, tag):
    quad(m, fpt(w, d, side, s0, z0, out), fpt(w, d, side, s1, z0, out), fpt(w, d, side, s1, z1, out), fpt(w, d, side, s0, z1, out), tag)


def fpoly(m, w, d, side, pts, out0, out1, tag):
    """A polygon in a facade's plane, counter-clockwise seen from outside, extruded between two offsets."""
    n = len(pts)
    base = m.add_verts([fpt(w, d, side, p[0], p[1], out1) for p in pts])
    m.add_face(list(range(base, base + n)), tag, [tuple(p) for p in pts])
    base = m.add_verts([fpt(w, d, side, p[0], p[1], out0) for p in reversed(pts)])
    m.add_face(list(range(base, base + n)), tag, [tuple(p) for p in reversed(pts)])
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        quad(m, fpt(w, d, side, a[0], a[1], out0), fpt(w, d, side, b[0], b[1], out0), fpt(w, d, side, b[0], b[1], out1),
             fpt(w, d, side, a[0], a[1], out1), tag)


def pane(m, sp, side, s0, s1, z0, z1, tag, k, bars):
    """One pane of glazing: occupied rooms have their blinds part-way down (a pale band at the top
    of the pane, a different drop in every window), classic windows get glazing bars."""
    w, d = sp.w, sp.d
    top = z1
    if tag == T_GLASS and sp.decay == 0.0:
        r = uniform(sp.seed + 303, k)
        if r > 0.62:
            top = z1 - (z1 - z0) * (0.1 + 0.4 * (r - 0.62) / 0.38)
            fquad(m, w, d, side, s0, s1, top, z1, 0.015, T_BLIND)
            fbox(m, w, d, side, s0, s1, top - 0.025, top, 0.0, 0.02, T_TRIM)    # the blind's bottom rail
    fquad(m, w, d, side, s0, s1, z0, top, 0.01, tag)
    nv, nh = bars
    for j in range(1, nv + 1):
        s = s0 + (s1 - s0) * j / (nv + 1)
        fbox(m, w, d, side, s - 0.02, s + 0.02, z0, z1, 0.0, 0.05, T_TRIM)
    for j in range(1, nh + 1):
        z = z0 + (z1 - z0) * j / (nh + 1)
        fbox(m, w, d, side, s0, s1, z - 0.025, z + 0.025, 0.0, 0.05, T_TRIM)


def window_tag(sp, k):
    r = uniform(sp.seed + 101, k)
    if sp.decay > 0.0:
        if r < sp.decay * 0.45:
            return T_DARK
        if r < sp.decay * 0.8:
            return T_BOARD
        return T_GLASS
    return T_LIT if r < sp.lit else T_GLASS


class _K:
    """A window counter shared across facades (Rust passes it as &mut u64)."""
    __slots__ = ("k",)

    def __init__(self):
        self.k = 0


def facade(m, sp, side, zs, door, kc):
    """Windows, doors and facade details on one side."""
    w, d = sp.w, sp.d
    ln = side_len(w, d, side)
    sh = sp.storey_h

    def clear(s0, s1, storey):
        if door is not None and side == 0 and storey == 0:
            a, b = door
            return s1 < a - 0.4 or s0 > b + 0.4
        return True
    for si, z in enumerate(zs):
        if sp.window == "ribbon":
            z0, z1 = z + 0.85, z + sh - 0.55
            runs = [(0.6, ln - 0.6)]
            if door is not None and side == 0 and si == 0:
                a, b = door
                runs = [(0.6, a - 0.5), (b + 0.5, ln - 0.6)]
            for s0, s1 in runs:
                if s1 - s0 < 1.0:
                    continue
                # the run sits back in a reveal: a deep head (a shading lip), sill and end jambs
                fbox(m, w, d, side, s0 - 0.05, s1 + 0.05, z0 - 0.12, z0, 0.0, 0.18, T_TRIM)
                fbox(m, w, d, side, s0 - 0.05, s1 + 0.05, z1, z1 + 0.12, 0.0, 0.3, T_TRIM)
                fbox(m, w, d, side, s0 - 0.05, s0, z0, z1, 0.0, 0.3, T_TRIM)
                fbox(m, w, d, side, s1, s1 + 0.05, z0, z1, 0.0, 0.3, T_TRIM)
                n = int(max(rround((s1 - s0) / 1.5), 1.0))
                for j in range(n + 1):
                    s = s0 + (s1 - s0) * j / n
                    fbox(m, w, d, side, s - 0.035, s + 0.035, z0, z1, 0.0, 0.09, T_TRIM)
                    if j < n:
                        kc.k += 1
                        s2 = s0 + (s1 - s0) * (j + 1) / n
                        pane(m, sp, side, s, s2, z0, z1, window_tag(sp, kc.k), kc.k, (0, 0))
        elif sp.window == "strip":
            z0, z1 = z + sh - 1.6, z + sh - 0.6
            n = int(max(math.floor((ln - 2.0) / 2.0), 1.0))
            s0 = (ln - n * 2.0) / 2.0
            kc.k += 1
            fquad(m, w, d, side, s0, ln - s0, z0, z1, 0.01, window_tag(sp, kc.k))
            for j in range(n + 1):
                s = s0 + j * 2.0
                fbox(m, w, d, side, s - 0.04, s + 0.04, z0, z1, 0.0, 0.07, T_TRIM)
            fbox(m, w, d, side, s0, ln - s0, z0 - 0.08, z0, 0.0, 0.1, T_TRIM)
            fbox(m, w, d, side, s0, ln - s0, z1, z1 + 0.08, 0.0, 0.1, T_TRIM)
        elif sp.window != "none":
            kind = sp.window
            if kind == "large":
                bay, ww, z0, z1, frame = 3.8, 2.6, z + 0.35, z + sh - 0.6, 0.12
            elif kind == "shuttered":
                bay, ww, z0, z1, frame = 3.0, 1.1, z + 0.95, z + 0.95 + 1.7, 0.07
            else:
                bay, ww, z0, z1, frame = 3.2, 1.3, z + 0.9, z + 0.9 + min(2.0, sh - 1.4), 0.08
            n = int(max(math.floor((ln - 1.6) / bay), 1.0))
            first = (ln - (n - 1.0) * bay) / 2.0
            for j in range(n):
                c = first + j * bay
                s0, s1 = c - ww / 2.0, c + ww / 2.0
                if s0 < 0.4 or s1 > ln - 0.4 or not clear(s0, s1, si):
                    continue
                kc.k += 1
                bars = (1, 1) if kind == "punched" and sp.quoins else (1, 2) if kind == "shuttered" else (0, 0)
                pane(m, sp, side, s0, s1, z0, z1, window_tag(sp, kc.k), kc.k, bars)
                # the window sits back in its surround (a reveal deep enough to cast a shadow line)
                fbox(m, w, d, side, s0 - frame, s0, z0, z1, 0.0, 0.2, T_TRIM)
                fbox(m, w, d, side, s1, s1 + frame, z0, z1, 0.0, 0.2, T_TRIM)
                fbox(m, w, d, side, s0 - frame, s1 + frame, z1, z1 + frame, 0.0, 0.2, T_TRIM)
                fbox(m, w, d, side, s0 - frame - 0.05, s1 + frame + 0.05, z0 - 0.08, z0, 0.0, 0.24, T_STONE if sp.quoins else T_TRIM)
                if kind == "large":
                    fbox(m, w, d, side, c - 0.04, c + 0.04, z0, z1, 0.0, 0.08, T_TRIM)
                if kind == "punched" and sp.quoins:
                    fbox(m, w, d, side, s0 - 0.15, s1 + 0.15, z1 + frame, z1 + frame + 0.22, 0.0, 0.05, T_STONE)
                if kind == "shuttered":
                    for a, b in ((s0 - 0.6, s0 - 0.05), (s1 + 0.05, s1 + 0.6)):
                        fbox(m, w, d, side, a, b, z0, z1, 0.02, 0.06, T_TRIM)
        if si > 0 and (sp.quoins or sp.window == "shuttered"):
            fbox(m, w, d, side, -0.05, ln + 0.05, z - 0.1, z + 0.08, 0.0, 0.07, T_STONE)
    if sp.fins and side == 0:
        top = zs[0] + sp.storey_h * len(zs)
        n = int(ln / 1.25)
        for j in range(n + 1):
            s = j * ln / max(n, 1)
            fbox(m, w, d, side, s - 0.06, s + 0.06, zs[0] + 0.2, top, 0.0, 0.55, T_TRIM)
    if sp.frame:
        # a clad portal frame: the steel is inside; outside are the folded flashings in the trim
        # colour (corners, the drip over the plinth, the gutter along the eaves)
        top = zs[0] + sp.storey_h * len(zs)
        for a, b in ((0.0, 0.16), (ln - 0.16, ln)):
            fbox(m, w, d, side, a, b, zs[0] - 0.05, top, 0.0, 0.05, T_TRIM)
        fbox(m, w, d, side, 0.0, ln, zs[0] - 0.04, zs[0] + 0.04, 0.0, 0.05, T_TRIM)
        fbox(m, w, d, side, -0.05, ln + 0.05, top - 0.22, top, 0.0, 0.16, T_TRIM)
        if ln >= max(w, d) - 1e-6:
            # rainwater downpipes off the eaves gutter, one every 15-20 m, into a shoe at the foot
            n = int(max(rround(ln / 18.0), 1.0))
            for j in range(n):
                s = (j + 0.5) * ln / n
                fbox(m, w, d, side, s - 0.05, s + 0.05, 0.15, top - 0.22, 0.03, 0.13, T_TRIM)
                fbox(m, w, d, side, s - 0.06, s + 0.06, 0.0, 0.15, 0.03, 0.22, T_TRIM)
            # fire exits between every other pair of downpipes (not on the front with its doors):
            # steel double doors in a frame under a hood, out onto a concrete landing and step
            z0 = zs[0]
            for j in range(1, n, 2) if side != 0 else []:
                c = j * ln / n
                fquad(m, w, d, side, c - 0.9, c - 0.01, z0, z0 + 2.1, 0.03, T_DOOR)
                fquad(m, w, d, side, c + 0.01, c + 0.9, z0, z0 + 2.1, 0.03, T_DOOR)
                fbox(m, w, d, side, c - 1.0, c - 0.9, z0, z0 + 2.2, 0.0, 0.08, T_TRIM)
                fbox(m, w, d, side, c + 0.9, c + 1.0, z0, z0 + 2.2, 0.0, 0.08, T_TRIM)
                fbox(m, w, d, side, c - 1.0, c + 1.0, z0 + 2.1, z0 + 2.2, 0.0, 0.08, T_TRIM)
                fbox(m, w, d, side, c - 1.3, c + 1.3, z0 + 2.45, z0 + 2.55, 0.0, 0.7, T_TRIM)
                fbox(m, w, d, side, c - 1.4, c + 1.4, 0.0, z0, 0.04, 1.6, T_BASE)    # (from the plinth's face)
                fbox(m, w, d, side, c - 1.4, c + 1.4, 0.0, z0 * 0.5, 1.6, 1.95, T_BASE)



def rot90(m):
    """Rotate a building-local mesh by 90 degrees about z (x -> y)."""
    m.verts = [(-p[1], p[0], p[2]) for p in m.verts]


def roof(m, sp, h, tubes):
    """The roof over walls of height h. Ridges run along the longer side."""
    w, d = sp.w, sp.d
    e = sp.eaves
    if sp.roof in ("flat", "none"):
        if sp.roof == "flat":
            quad(m, (-w / 2.0, -d / 2.0, h), (w / 2.0, -d / 2.0, h), (w / 2.0, d / 2.0, h), (-w / 2.0, d / 2.0, h), T_ROOF)
            p = sp.parapet
            t = 0.3
            for x0, y0, x1, y1 in ((-w / 2.0, -d / 2.0, w / 2.0, -d / 2.0 + t), (-w / 2.0, d / 2.0 - t, w / 2.0, d / 2.0),
                                   (-w / 2.0, -d / 2.0 + t, -w / 2.0 + t, d / 2.0 - t), (w / 2.0 - t, -d / 2.0 + t, w / 2.0, d / 2.0 - t)):
                bx(m, x0, y0, h, x1, y1, h + p, T_WALL)
                bx(m, x0 - 0.03, y0 - 0.03, h + p, x1 + 0.03, y1 + 0.03, h + p + 0.06, T_METAL)
            if sp.decay <= 0.0 and w > 12.0:
                bx(m, w * 0.12, -d * 0.15, h, w * 0.12 + 3.2, -d * 0.15 + 2.0, h + 1.6, T_METAL)
                bx(m, -w * 0.3, d * 0.05, h, -w * 0.3 + 2.4, d * 0.05 + 2.4, h + 1.1, T_METAL)
            if sp.solar:
                solar_rows(m, -w / 2.0 + 1.5, w / 2.0 - 1.5, -d / 2.0 + 1.5, d / 2.0 - 1.5, h + 0.3, 0.0)
    elif sp.roof == "hip":
        l, s, swap = (w, d, False) if w >= d else (d, w, True)
        r = kernel.Mesh()
        tp = math.tan(math.radians(sp.pitch_deg))
        lx, dy = l / 2.0 + e, s / 2.0 + e
        ze = h - e * tp
        zr = h + s / 2.0 * tp
        rx = max(l / 2.0 - s / 2.0, 0.0)
        quad(r, (-lx, -dy, ze), (lx, -dy, ze), (rx, 0.0, zr), (-rx, 0.0, zr), T_ROOF)
        quad(r, (lx, dy, ze), (-lx, dy, ze), (-rx, 0.0, zr), (rx, 0.0, zr), T_ROOF)
        tri(r, (lx, -dy, ze), (lx, dy, ze), (rx, 0.0, zr), T_ROOF)
        tri(r, (-lx, dy, ze), (-lx, -dy, ze), (-rx, 0.0, zr), T_ROOF)
        for x0, y0, x1, y1 in ((-lx, -dy - 0.03, lx, -dy + 0.12), (-lx, dy - 0.12, lx, dy + 0.03), (-lx - 0.03, -dy, -lx + 0.12, dy),
                               (lx - 0.12, -dy, lx + 0.03, dy)):
            bx(r, x0, y0, ze - 0.22, x1, y1, ze + 0.02, T_TRIM)
        if rx > 0.0:
            bx(r, -rx, -0.1, zr - 0.05, rx, 0.1, zr + 0.1, T_ROOF)
        if swap:
            rot90(r)
        m.append(r)
    elif sp.roof == "mono":
        tp = math.tan(math.radians(sp.pitch_deg))
        rise = d * tp
        lx, dy = w / 2.0 + e, d / 2.0 + e
        zf = h + (d + e) * tp
        zb = h - e * tp
        quad(m, (-lx, -dy, zf), (lx, -dy, zf), (lx, dy, zb), (-lx, dy, zb), T_ROOF)
        fquad(m, w, d, 0, 0.0, w, h, h + rise, 0.0, T_WALL)
        tri(m, (w / 2.0, -d / 2.0, h), (w / 2.0, d / 2.0, h), (w / 2.0, -d / 2.0, h + rise), T_WALL)
        tri(m, (-w / 2.0, d / 2.0, h), (-w / 2.0, -d / 2.0, h), (-w / 2.0, -d / 2.0, h + rise), T_WALL)
        bx(m, -lx, -dy - 0.05, zf - 0.3, lx, -dy + 0.1, zf + 0.05, T_TRIM)
        bx(m, -lx, dy - 0.1, zb - 0.25, lx, dy + 0.05, zb + 0.05, T_TRIM)
        if sp.mats[T_ROOF].key == "Roof_Sedum":
            # a pebble margin round the green roof (a fire break, and drainage at the eaves)
            def zr(y):
                return zf + (zb - zf) * (y + dy) / (2.0 * dy) + 0.02
            b = 0.45
            for x0, y0, x1, y1 in ((-lx, -dy, lx, -dy + b), (-lx, dy - b, lx, dy), (-lx, -dy + b, -lx + b, dy - b),
                                   (lx - b, -dy + b, lx, dy - b)):
                quad(m, (x0, y0, zr(y0)), (x1, y0, zr(y0)), (x1, y1, zr(y1)), (x0, y1, zr(y1)), T_GRAVEL)
        if sp.solar:
            solar_rows(m, -w / 2.0 + 1.0, w / 2.0 - 1.0, -d / 2.0 + 1.2, d / 2.0 - 1.2, h + 0.15, sp.pitch_deg)
    elif sp.roof == "barrel":
        l, s, swap = (w, d, False) if w >= d else (d, w, True)
        r = kernel.Mesh()
        rise = 0.14 * s
        n = 12

        def z_at(y):
            q = y / (s / 2.0)
            return h + rise * (1.0 - q * q)
        ys = [-(s / 2.0 + e) + (s + 2.0 * e) * i / n for i in range(n + 1)]
        segs = int(max(math.ceil(l / 3.0), 1.0))
        for g in range(segs):
            x0 = -l / 2.0 - e + (l + 2.0 * e) * g / segs
            x1 = -l / 2.0 - e + (l + 2.0 * e) * (g + 1) / segs
            hole = sp.decay > 0.3 and uniform(sp.seed + 202, g) < sp.decay * 0.35
            for i in range(n):
                ya, yb = ys[i], ys[i + 1]
                if hole and 2 < i < n - 2:
                    continue
                quad(r, (x0, ya, z_at(ya)), (x1, ya, z_at(ya)), (x1, yb, z_at(yb)), (x0, yb, z_at(yb)), T_ROOF)
            if hole:
                xm = (x0 + x1) / 2.0
                tubes.append(([(xm, y, z_at(y) - 0.15) for y in ys], 0.06))
        for sx in (1.0, -1.0):
            x = sx * l / 2.0
            pts = [(x, -s / 2.0, h), (x, s / 2.0, h)]
            for i in range(1, n):
                y = s / 2.0 - s * i / n
                pts.append((x, y, z_at(y)))
            if sx < 0.0:
                pts.reverse()
            base = r.add_verts(pts)
            r.add_face(list(range(base, base + len(pts))), T_WALL, kernel.planar_uv(pts))
        if swap:
            rot90(r)
        m.append(r)
    else:   # gable / shallow
        l, s, swap = (w, d, False) if w >= d else (d, w, True)
        r = kernel.Mesh()
        tp = math.tan(math.radians(sp.pitch_deg))
        lx, dy = l / 2.0 + e, s / 2.0 + e
        ze = h - e * tp
        rise = s / 2.0 * tp
        zr = h + rise
        segs = int(max(math.ceil(l / 3.0), 1.0))
        for g in range(segs):
            x0, x1 = -lx + 2.0 * lx * g / segs, -lx + 2.0 * lx * (g + 1) / segs
            hole = sp.decay > 0.3 and uniform(sp.seed + 203, g) < sp.decay * 0.35
            if not hole or uniform(sp.seed + 204, g) < 0.5:
                quad(r, (x0, -dy, ze), (x1, -dy, ze), (x1, 0.0, zr), (x0, 0.0, zr), T_ROOF)
            if not hole or uniform(sp.seed + 205, g) < 0.5:
                quad(r, (x1, dy, ze), (x0, dy, ze), (x0, 0.0, zr), (x1, 0.0, zr), T_ROOF)
            if hole:
                xm = (x0 + x1) / 2.0
                for k in (-1.0, 1.0):
                    tubes.append(([(xm + k * 0.6, -dy + 0.2, ze + 0.05), (xm + k * 0.6, 0.0, zr - 0.1), (xm + k * 0.6, dy - 0.2, ze + 0.05)], 0.07))
        for sx in (1.0, -1.0):
            x = sx * l / 2.0
            if sx > 0.0:
                tri(r, (x, -s / 2.0, h), (x, s / 2.0, h), (x, 0.0, zr), T_WALL)
            else:
                tri(r, (x, s / 2.0, h), (x, -s / 2.0, h), (x, 0.0, zr), T_WALL)
            xb = sx * lx
            for sy in (-1.0, 1.0):
                a, b = (xb, sy * dy, ze), (xb, 0.0, zr)
                ad, bd = (a[0], a[1], a[2] - 0.24), (b[0], b[1], b[2] - 0.24)
                if sx * sy > 0.0:
                    quad(r, ad, bd, b, a, T_TRIM)
                else:
                    quad(r, bd, ad, a, b, T_TRIM)
        for sy in (-1.0, 1.0):
            bx(r, -lx, sy * dy - 0.06, ze - 0.24, lx, sy * dy + 0.06, ze + 0.02, T_TRIM)
        bx(r, -lx, -0.12, zr - 0.04, lx, 0.12, zr + 0.12, T_METAL)
        if swap:
            rot90(r)
        m.append(r)


def solar_rows(m, x0, x1, y0, y1, z, pitch_deg):
    """Rows of solar panels on a roof plane rising towards -y at pitch_deg."""
    tp = math.tan(math.radians(pitch_deg))
    rows = int(max(math.floor((y1 - y0) / 2.2), 1.0))
    for r in range(rows):
        ya = y0 + r * 2.2
        yb = ya + 1.7
        if pitch_deg > 0.0:
            za, zb = z + (y1 - ya) * tp, z + (y1 - yb) * tp
        else:
            za, zb = z + 0.55, z
        n = int(max(math.floor((x1 - x0) / 1.05), 1.0))
        for c in range(n):
            xa = x0 + c * 1.05
            quad(m, (xa, ya, za), (xa + 1.0, ya, za), (xa + 1.0, yb, zb), (xa, yb, zb), T_SOLAR)


_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        from pathlib import Path
        _FONT = json.loads((Path(__file__).resolve().parents[2] / "catalog" / "font_strokes.json").read_text())
    return _FONT


def _stroke_points(st):
    """A stroke's polyline (glyph units); arcs become segments of at most 15 degrees."""
    if len(st) == 1 and isinstance(st[0][0], str):
        _, cx, cy, rx, ry, a0, a1 = st[0]
        n = max(4, math.ceil(abs(a1 - a0) / 15.0))
        return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * t / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * t / n)))
                for t in range(n + 1)]
    return [(p[0], p[1]) for p in st]


def letters(m, w, d, side, txt, s_mid, z, max_w, tag, max_px=0.16, out=0.0):
    """Raised letters for the club's name (a geometric sans stroke font, catalog/font_strokes.json):
    every stroke segment is a small prism standing proud of the wall, centred at facade position
    s_mid, bottom at z, at most max_w wide and 7 * max_px tall."""
    font = _font()
    glyphs, widths = font["glyphs"], font["widths"]
    chars = [c.upper() for c in txt]
    adv = [float(widths.get(c, 4.0 if c in glyphs else 2.4)) for c in chars]
    total = sum(adv) + 1.2 * max(len(chars) - 1, 0)
    if total <= 0.0:
        return
    u = min(max(max_w / total, 0.04), max_px * 7.0 / 6.0)
    hw = 0.36 * u
    x = s_mid - total * u / 2.0
    for c, a in zip(chars, adv):
        for st in glyphs.get(c, []):
            pts = _stroke_points(st)
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                ds, dz = (x1 - x0) * u, (y1 - y0) * u
                ln = math.hypot(ds, dz)
                if ln < 1e-9:
                    ds, dz, ln = 0.0, 1.0, 1.0
                ts, tz = ds / ln, dz / ln
                ns, nz = -tz * hw, ts * hw
                ps, pz = x + x0 * u - ts * hw, z + y0 * u - tz * hw
                qs, qz = x + x1 * u + ts * hw, z + y1 * u + tz * hw
                ring = [(ps - ns, pz - nz), (qs - ns, qz - nz), (qs + ns, qz + nz), (ps + ns, pz + nz)]
                back = [fpt(w, d, side, a_, b_, out + 0.02) for a_, b_ in ring]
                front = [fpt(w, d, side, a_, b_, out + 0.09) for a_, b_ in ring]
                quad(m, front[0], front[1], front[2], front[3], tag)
                for k in range(4):
                    k2 = (k + 1) % 4
                    quad(m, back[k], back[k2], front[k2], front[k], tag)
        x += (a + 1.2) * u


class Built:
    __slots__ = ("entrance", "height")

    def __init__(self, entrance, height):
        self.entrance, self.height = entrance, height


def build(o, f, name, sp):
    """Generate a building at frame f (its centre, front towards local -y)."""
    w, d = sp.w, sp.d
    m = kernel.Mesh()
    tubes = []
    h = PLINTH + sp.storeys * sp.storey_h
    foot = kernel.rect(w, d)
    kernel.prism(m, kernel.rect(w + 0.08, d + 0.08), 0.0, PLINTH, T_BASE, None, None)
    kernel.prism(m, foot, PLINTH, h, T_WALL, None, None)
    zs = [PLINTH + i * sp.storey_h for i in range(sp.storeys)]
    door_w = 1.8 if sp.door == "timber" else 2.8
    door = (w / 2.0 - door_w / 2.0, w / 2.0 + door_w / 2.0) if sp.door != "none" else None
    kc = _K()
    for side in range(4):
        facade(m, sp, side, zs, door, kc)
    if door is not None:
        a, b = door
        dh = min(2.7, sp.storey_h - 0.6)
        tag = T_DOOR if sp.door == "timber" else T_BOARD if sp.decay > 0.5 else T_GLASS
        fquad(m, w, d, 0, a, b, PLINTH, PLINTH + dh, 0.02, tag)
        fbox(m, w, d, 0, a - 0.12, a, PLINTH, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM)
        fbox(m, w, d, 0, b, b + 0.12, PLINTH, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM)
        fbox(m, w, d, 0, a - 0.12, b + 0.12, PLINTH + dh, PLINTH + dh + 0.12, 0.0, 0.14, T_TRIM)
        fbox(m, w, d, 0, (a + b) / 2.0 - 0.04, (a + b) / 2.0 + 0.04, PLINTH, PLINTH + dh, 0.0, 0.08, T_TRIM)
        if sp.door == "timber" and sp.storey_h - dh > 1.0:
            fquad(m, w, d, 0, a, b, PLINTH + dh + 0.12, PLINTH + dh + 0.7, 0.02, T_GLASS)
            fbox(m, w, d, 0, a - 0.12, b + 0.12, PLINTH + dh + 0.7, PLINTH + dh + 0.82, 0.0, 0.14, T_TRIM)
        fbox(m, w, d, 0, a - 0.6, b + 0.6, 0.0, PLINTH * 0.5, 0.0, 1.4, T_BASE)
        fbox(m, w, d, 0, a - 0.6, b + 0.6, 0.0, PLINTH, 0.0, 0.7, T_BASE)
    if sp.roller_doors > 0:
        n = sp.roller_doors
        span = w / n
        for j in range(n):
            c = span * (j + 0.5)
            a, b = c - 2.0, c + 2.0
            if door is not None and b > door[0] - 0.5 and a < door[1] + 0.5:
                continue
            top = PLINTH + min(4.4, sp.storey_h * sp.storeys - 0.6)
            fquad(m, w, d, 0, a, b, 0.05, top, 0.03, T_DOOR)
            ribs = int((top - 0.05) / 0.32)
            for r in range(ribs):
                z = 0.05 + r * 0.32
                fbox(m, w, d, 0, a, b, z, z + 0.04, 0.03, 0.06, T_DOOR)
            fbox(m, w, d, 0, a - 0.15, a, 0.0, top + 0.15, 0.0, 0.2, T_TRIM)
            fbox(m, w, d, 0, b, b + 0.15, 0.0, top + 0.15, 0.0, 0.2, T_TRIM)
            fbox(m, w, d, 0, a - 0.15, b + 0.15, top, top + 0.5, 0.0, 0.3, T_TRIM)
    if sp.quoins:
        z = PLINTH
        i = 0
        while z + 0.32 < h:
            long = 0.6 if i % 2 == 0 else 0.35
            for side in range(4):
                ln = side_len(w, d, side)
                fbox(m, w, d, side, -0.02, long, z, z + 0.3, 0.0, 0.05, T_STONE)
                fbox(m, w, d, side, ln - long, ln + 0.02, z, z + 0.3, 0.0, 0.05, T_STONE)
            z += 0.34
            i += 1
    if sp.led_band:
        for side in range(4):
            fbox(m, w, d, side, 0.0, side_len(w, d, side), h - 0.32, h - 0.1, 0.0, 0.05, T_LED)
    roof(m, sp, h, tubes)
    dz = PLINTH + min(2.7, sp.storey_h - 0.6) + 0.4
    if sp.canopy == "cantilever":
        fbox(m, w, d, 0, w / 2.0 - 5.0, w / 2.0 + 5.0, dz + 0.4, dz + 0.75, 0.0, 3.6, T_TRIM)
        if sp.led_band:
            fbox(m, w, d, 0, w / 2.0 - 5.0, w / 2.0 + 5.0, dz + 0.32, dz + 0.4, 3.45, 3.6, T_LED)
    elif sp.canopy == "portico":
        depth = 3.0
        for sx in (-1.0, 1.0):
            p = fpt(w, d, 0, w / 2.0 + sx * 2.2, 0.0, depth - 0.35)
            cylinder(m, p[0], p[1], 0.0, dz + 0.9, 0.24, 12, T_STONE, True)
        fbox(m, w, d, 0, w / 2.0 - 2.7, w / 2.0 + 2.7, dz + 0.9, dz + 1.35, 0.0, depth, T_STONE)
        pts = [(w / 2.0 - 2.9, dz + 1.35), (w / 2.0 + 2.9, dz + 1.35), (w / 2.0, dz + 2.45)]
        fpoly(m, w, d, 0, pts, depth - 0.25, depth + 0.05, T_STONE)
        fpoly(m, w, d, 0, pts, 0.0, 0.3, T_STONE)

        def v(s, z, out):
            return fpt(w, d, 0, s, z, out)
        quad(m, v(w / 2.0 - 3.0, dz + 1.3, depth + 0.1), v(w / 2.0, dz + 2.5, depth + 0.1), v(w / 2.0, dz + 2.5, 0.0), v(w / 2.0 - 3.0, dz + 1.3, 0.0), T_ROOF)
        quad(m, v(w / 2.0, dz + 2.5, depth + 0.1), v(w / 2.0 + 3.0, dz + 1.3, depth + 0.1), v(w / 2.0 + 3.0, dz + 1.3, 0.0), v(w / 2.0, dz + 2.5, 0.0), T_ROOF)
    elif sp.canopy == "steel":
        fbox(m, w, d, 0, w / 2.0 - 3.5, w / 2.0 + 3.5, dz + 0.3, dz + 0.45, 0.0, 3.0, T_METAL)
        for sx in (-1.0, 1.0):
            fbox(m, w, d, 0, w / 2.0 + sx * 3.2 - 0.08, w / 2.0 + sx * 3.2 + 0.08, 0.0, dz + 0.3, 2.7, 2.9, T_METAL)
    elif sp.canopy == "timber":
        fbox(m, w, d, 0, w / 2.0 - 4.0, w / 2.0 + 4.0, dz + 0.5, dz + 0.75, 0.0, 3.4, T_ROOF)
        fbox(m, w, d, 0, w / 2.0 - 4.0, w / 2.0 + 4.0, dz + 0.35, dz + 0.5, 0.0, 3.4, T_WALL)
        for sx in (-1.0, 1.0):
            fbox(m, w, d, 0, w / 2.0 + sx * 3.6 - 0.1, w / 2.0 + sx * 3.6 + 0.1, 0.0, dz + 0.35, 3.0, 3.2, T_WALL)
    if sp.arcade:
        depth = 3.0
        zt = PLINTH + sp.storey_h - 0.2
        n = int(max(math.floor((w - 1.0) / 3.6), 2.0))
        bay = (w - 0.6) / n
        cols = [0.3 + j * bay for j in range(n + 1)]
        for c in cols:
            fbox(m, w, d, 0, c - 0.24, c + 0.24, 0.0, zt, depth - 0.48, depth, T_WALL)
        for j in range(n):
            s0, s1 = cols[j] + 0.24, cols[j + 1] - 0.24
            rr = (s1 - s0) / 2.0
            zs_ = zt - 0.45 - rr
            sm = (s0 + s1) / 2.0
            pts = [(s0, zs_)]
            for q in range(1, 10):
                a = math.pi - math.pi * q / 10.0
                pts.append((sm + rr * math.cos(a), zs_ + rr * math.sin(a)))
            pts += [(s1, zs_), (s1, zt), (s0, zt)]
            fpoly(m, w, d, 0, pts, depth - 0.48, depth, T_WALL)
        fbox(m, w, d, 0, 0.0, w, zt, zt + 0.35, 0.0, depth + 0.1, T_WALL)
        quad(m, fpt(w, d, 0, -0.2, zt + 0.35, depth + 0.4), fpt(w, d, 0, w + 0.2, zt + 0.35, depth + 0.4),
             fpt(w, d, 0, w + 0.2, zt + 1.1, 0.0), fpt(w, d, 0, -0.2, zt + 1.1, 0.0), T_ROOF)
    if sp.clock:
        pts = [(w / 2.0 - 3.4, h - 0.2), (w / 2.0 + 3.4, h - 0.2), (w / 2.0, h + 2.0)]
        fpoly(m, w, d, 0, pts, 0.0, 0.25, T_STONE)
        c = (w / 2.0, h + 0.55)
        ring = [(c[0] + 0.55 * math.cos(2 * math.pi * q / 16.0), c[1] + 0.55 * math.sin(2 * math.pi * q / 16.0)) for q in range(16)]
        fpoly(m, w, d, 0, ring, 0.25, 0.3, T_TRIM)
    if sp.sign is not None:
        # the name goes where the wall is blank: on a flat roof's parapet, else in the band between
        # the top windows' heads and the eaves (never across windows or under the cornice)
        # (else on the entrance portico's entablature or the canopy fascia; with neither, the name is
        # left to the sign at the gate)
        if sp.roof == "flat":
            room = sp.parapet - 0.2
            letters(m, w, d, 0, sp.sign, w / 2.0, h + 0.1, min(w * 0.45, 14.0), T_ACCENT, room / 7.0)
        elif sp.canopy == "portico":
            letters(m, w, d, 0, sp.sign, w / 2.0, dz + 0.95, 4.8, T_ACCENT, 0.36 / 7.0, 3.0)
        elif sp.canopy == "cantilever":
            letters(m, w, d, 0, sp.sign, w / 2.0, dz + 0.43, 9.0, T_ACCENT, 0.29 / 7.0, 3.6)
    if sp.decay > 0.0:
        for j, side in enumerate((0, 2)):
            ln = side_len(w, d, side)
            p = fpt(w, d, side, ln * 0.15, 0.0, 0.08)
            tubes.append(([(p[0], p[1], 0.0), (p[0], p[1], h - 0.1 - uniform(sp.seed + 301, j) * 2.0)], 0.05))
    o.part_at(f, name, m, list(sp.mats), 0.0)
    if sp.decay <= 0.0 and sp.roof in ("gable", "shallow", "hip"):
        o.tubes_at(f, f"{name}_rainwater", rainwater(sp, h), mats.tint("TINT_trim", "#26282b"), 8)
    if tubes:
        o.tubes_at(f, f"{name}_steel", tubes, mats.named("Site_Rust"), 6)
    return Built((0.0, -d / 2.0 - 1.5), h)


def rainwater(sp, h):
    """Rainwater goods for a pitched roof: a gutter under each long eave and a downpipe from each
    end of it, set back to the wall and down to a shoe at the ground."""
    w, d = sp.w, sp.d
    l, s, swap = (w, d, False) if w >= d else (d, w, True)
    e = sp.eaves
    zf = h - e * math.tan(math.radians(sp.pitch_deg))
    lx = l / 2.0 + e

    def place(x, y, z):
        return (-y, x, z) if swap else (x, y, z)
    out = []
    for sy in (-1.0, 1.0):
        yg = sy * (s / 2.0 + e + 0.07)
        out.append(([place(-lx, yg, zf - 0.12), place(lx, yg, zf - 0.12)], 0.065))
        for sx in (-1.0, 1.0):
            x = sx * (l / 2.0 - 0.8)
            yw = sy * (s / 2.0 + 0.07)
            out.append(([place(x, yg, zf - 0.16), place(x, yg, zf - 0.32), place(x, yw, zf - 0.7), place(x, yw, 0.25),
                         place(x, yw + sy * 0.12, 0.08)], 0.035))
    return out


def ruin(o, f, name, w, d, h_max, wall, seed):
    """A ruin: walls broken down to uneven heights, no roof (a building half demolished)."""
    m = kernel.Mesh()
    bx(m, -w / 2.0, -d / 2.0, 0.0, w / 2.0, d / 2.0, 0.25, 1)
    k = 0
    for side in range(4):
        ln = side_len(w, d, side)
        n = int(max(math.ceil(ln / 2.4), 1.0))
        for j in range(n):
            k += 1
            if uniform(seed + 401, k) < 0.25:
                continue
            s0, s1 = ln * j / n, ln * (j + 1) / n
            top = 0.4 + (h_max - 0.4) * uniform(seed + 402, k) ** 1.6
            fbox(m, w, d, side, s0, s1, 0.0, top, -0.35, 0.0, 0)
    o.part_at(f, name, m, [wall, mats.named("Facade_Concrete")], 0.0)


def gatehouse(o, cat, f, style):
    """A small gatehouse beside the drive in a style, with the club's name on a sign wall."""
    from .landscape import GATE_W, ROAD_W
    club = o.club
    sp = styled(cat, style, 7.5, 5.0, 1, club.primary, o.night, None, 9)
    sp.canopy = "none"
    sp.arcade = False
    sp.clock = False
    sp.solar = False
    sp.window = "ribbon" if sp.window == "ribbon" else "large"
    g = f.sub(-(ROAD_W / 2.0 + 6.0), 12.0, -math.pi / 2)
    build(o, g, "Gate:gatehouse", sp)
    m = kernel.Mesh()
    ww, dd = 6.0, 0.6
    bx(m, -ww / 2.0, -dd / 2.0, 0.0, ww / 2.0, dd / 2.0, 1.4, T_WALL)
    bx(m, -ww / 2.0 - 0.05, -dd / 2.0 - 0.05, 1.4, ww / 2.0 + 0.05, dd / 2.0 + 0.05, 1.5, T_STONE)
    letters(m, ww, dd, 0, club.name, ww / 2.0, 0.45, 5.0, T_ACCENT, 0.16)
    s = f.sub(GATE_W / 2.0 + 5.5, -1.0, 0.0)
    o.part_at(s, "Gate:sign", m, list(sp.mats), 0.0)
