"""Faces and fitted details of the road vehicles (kit/assets/vehicles.py).

The nose and tail of each car are dressed with one painted sheet each (lamps with reflectors,
projectors and LED light guides, grilles with honeycomb or slats, intakes, badges, reflectors,
bumper and tailgate splits), drawn in metric (arc length, height) coordinates and wrapped onto
the body by casting rays round the end of the car; quads the sheet leaves bare are dropped, so
the body paint shows between the parts. Number plates use the stroke font of vehicle_paint
(close to the UK plate face). Shut lines, handles, mirrors, wipers, spoilers, aerials and roof
rails are fitted onto the body the same way. Vertex colours carry baked occlusion and grime.
"""
import math

import bpy
import numpy as np

from . import vehicle_paint as VP
from .vehicle_paint import Canvas, arc, mirror_x, rrect, round_poly

TYRE_V0, TYRE_V1 = 0.0, 1.0
TYRE_LEN = 0.42           # metres of profile the tyre sheet covers (v = 0..1)
TYRE_CIRC = 0.85          # metres of circumference per repeat

# linear colours
GB = (0.009, 0.0095, 0.010)      # gloss black
TB = (0.022, 0.022, 0.023)       # textured black plastic
CH = (0.62, 0.63, 0.64)          # chrome
SATIN = (0.30, 0.31, 0.32)       # satin aluminium
WHITE_LED = (0.92, 0.94, 0.96)
AMBER = (0.75, 0.30, 0.02)
GAP = (0.004, 0.004, 0.004)

_CANVAS = {}


# -- painting helpers -----------------------------------------------------------------------------

def hexmesh(cell, rib=0.16, hole=(0.004, 0.004, 0.0045), ribc=(0.03, 0.03, 0.032)):
    """Colour function of a honeycomb grille (pointy-top hexes of inradius cell/2)."""
    R = cell / math.sqrt(3)

    def f(X, Y):
        q = (math.sqrt(3) / 3 * X - Y / 3) / R
        r = (2 / 3 * Y) / R
        x, z = q, r
        y = -x - z
        rx, ry, rz = np.round(x), np.round(y), np.round(z)
        dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
        c1 = (dx > dy) & (dx > dz)
        rx = np.where(c1, -ry - rz, rx)
        c2 = ~c1 & (dy > dz)
        rz = np.where(~c1 & ~c2, -rx - ry, rz)
        cx = R * math.sqrt(3) * (rx + rz / 2)
        cy = R * 1.5 * rz
        ex, ey = X - cx, Y - cy
        ir = R * math.sqrt(3) / 2
        d = np.maximum.reduce([np.abs(ex), np.abs(ex * 0.5 + ey * math.sqrt(3) / 2), np.abs(ex * 0.5 - ey * math.sqrt(3) / 2)]) / ir
        rimm = np.clip((d - (1 - rib)) / 0.06, 0, 1)
        # a little light on the lower inside of each cell (it reads as depth)
        lit = np.clip(-ey / ir, 0, 1) * 0.012
        col = np.asarray(hole)[None, None, :] + lit[..., None]
        top = np.clip(ey / ir, 0, 1)[..., None] * 0.03
        return col * (1 - rimm[..., None]) + (np.asarray(ribc) + top) * rimm[..., None]
    return f


def diamond_mesh(cell):
    """Colour function of a 3D diamond-block grille (lit upper faces, dark lower)."""
    def f(X, Y):
        u = (X / cell + Y / cell)
        v = (X / cell - Y / cell)
        fu, fv = u - np.floor(u), v - np.floor(v)
        edge = (np.minimum(fu, 1 - fu) < 0.08) | (np.minimum(fv, 1 - fv) < 0.08)
        upper = (fu > 0.5) ^ (fv > 0.5)
        c = np.where(upper, 0.045, 0.012)
        c = np.where(edge, 0.003, c)
        return np.repeat(c[..., None], 3, -1)
    return f


def slats(spacing, vertical=True, base=0.004, slat=0.02, tip=0.12, width=0.45):
    def f(X, Y):
        t = (X if vertical else Y) / spacing
        fr = t - np.floor(t)
        s = np.abs(fr - 0.5) < width / 2
        edge = np.abs(fr - 0.5 + width / 2) < 0.08
        c = np.where(s, slat, base)
        c = np.where(edge & s, tip, c)
        return np.repeat(c[..., None], 3, -1)
    return f


def vgrad(c0, c1, y0, y1):
    c0, c1 = np.asarray(c0, np.float32), np.asarray(c1, np.float32)

    def f(X, Y):
        t = np.clip((Y - y0) / (y1 - y0), 0, 1)[..., None]
        return c0 * (1 - t) + c1 * t
    return f


def sym(poly):
    """A polygon given for s >= 0 and its mirror image."""
    return [poly, mirror_x(poly)]


def outline(cv, poly, width, color, **kw):
    P = list(poly) + [poly[0]]
    cv.lines([P], width, color, **kw)


def lamp_housing(cv, poly, smoke=(0.02, 0.021, 0.023), rim=CH, rim_w=0.004, inner=None):
    """A headlamp: chrome-edged smoky housing; inner() draws the modules."""
    P = round_poly(poly, 0.015)
    ys = [p[1] for p in P]
    cv.poly(P, vgrad((smoke[0] * 2.2, smoke[1] * 2.2, smoke[2] * 2.4), smoke, min(ys), max(ys)), rough=0.04, metal=0.0)
    outline(cv, P, rim_w, rim, rough=0.1, metal=1.0)
    if inner:
        inner(cv)
    # lens highlight: a soft streak along the top
    top = sorted(P, key=lambda p: -p[1])[:2]
    return P


def projector(cv, x, y, r):
    cv.ellipse(x, y, r * 1.25, r * 1.25, (0.12, 0.125, 0.13), rough=0.1, metal=1.0)
    cv.ring(x, y, r * 0.95, r * 1.25, CH, rough=0.08, metal=1.0)
    cv.ellipse(x, y, r * 0.8, r * 0.8, (0.01, 0.012, 0.016), rough=0.02)
    cv.ellipse(x - r * 0.25, y + r * 0.3, r * 0.22, r * 0.16, (0.5, 0.52, 0.56), rough=0.02)


def tail_lens(cv, poly, guides=(), bright=(0.75, 0.015, 0.012), base=(0.06, 0.003, 0.004)):
    P = round_poly(poly, 0.012)
    ys = [p[1] for p in P]
    cv.poly(P, vgrad((base[0] * 1.8, base[1], base[2]), base, min(ys), max(ys)), rough=0.04)
    for g, w in guides:
        cv.lines([g], w * 2.4, (bright[0] * 0.35, bright[1] * 0.35, bright[2] * 0.35), rough=0.04)
        cv.lines([g], w, bright, rough=0.05)
    outline(cv, P, 0.003, (0.03, 0.003, 0.003), rough=0.1)


def badge(cv, x, y, h, style="shield"):
    """The kit's own (fictional) maker badges: never a real roundel or oval."""
    k = h / 2
    if style == "shield":
        out = [(x - 0.8 * k, y + k), (x + 0.8 * k, y + k), (x + 0.7 * k, y - 0.2 * k), (x, y - k), (x - 0.7 * k, y - 0.2 * k)]
    elif style == "wing":
        out = [(x - 2.0 * k, y), (x - 1.4 * k, y + 0.6 * k), (x + 1.4 * k, y + 0.6 * k), (x + 2.0 * k, y), (x + 1.4 * k, y - 0.6 * k),
               (x - 1.4 * k, y - 0.6 * k)]
    else:
        out = [(x - 1.2 * k, y), (x, y + k), (x + 1.2 * k, y), (x, y - k)]
    P = round_poly(out, 0.15 * k)
    cv.poly(P, CH, rough=0.08, metal=1.0)
    c = np.array([x, y])
    inner = [tuple(c + (np.array(p) - c) * 0.8) for p in P]
    cv.poly(inner, (0.015, 0.02, 0.03), rough=0.1)
    if style == "shield":
        cv.lines([[(x - 0.4 * k, y + 0.45 * k), (x, y - 0.25 * k), (x + 0.4 * k, y + 0.45 * k)]], 0.16 * k, CH, rough=0.08, metal=1.0)
    elif style == "wing":
        cv.lines([[(x - 1.1 * k, y), (x + 1.1 * k, y)]], 0.22 * k, CH, rough=0.08, metal=1.0)
    else:
        cv.lines([[(x - 0.5 * k, y), (x, y + 0.42 * k), (x + 0.5 * k, y), (x, y - 0.42 * k), (x - 0.5 * k, y)]], 0.14 * k, CH, rough=0.08, metal=1.0)


def plate_text_canvas(text, front, worn=False):
    W, H = 1024, 219
    cv = Canvas(W, H, 0, 0.52, 0, 0.111, ss=3, rough=0.32)
    bg = (0.86, 0.86, 0.84) if front else (0.88, 0.52, 0.0)
    cv.poly(rrect(0, 0, 0.52, 0.111, 0.008), bg, rough=0.32)
    outline(cv, rrect(0.003, 0.003, 0.517, 0.108, 0.006), 0.002, (0.02, 0.02, 0.02))
    cv.text(text, 0.26, 0.016, 0.079, (0.012, 0.012, 0.012), cw=50 / 79, gap=11 / 79, stroke=14 / 79, rough=0.4)
    cv.text("KIT MOTORS", 0.26, 0.004, 0.006, (0.05, 0.05, 0.05), stroke=0.16)
    if worn:
        n = VP.noise2(H, W, 6, seed=3, octaves=4)
        d = np.clip((n - 0.45) * 2.0, 0, 1)[..., None] * 0.55
        cv.P[...] = cv.P * (1 - d) + np.array([0.12, 0.08, 0.04]) * d * cv.A[..., None]
    return cv


def plate_material(text, front, worn=False):
    name = f"Plate_{text.replace(' ', '')}_{'F' if front else 'R'}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    return VP.image_material(name, plate_text_canvas(text, front, worn), mask=False)


# -- tyres ----------------------------------------------------------------------------------------

def tyre_material(size="225/45 R17 94W"):
    name = "Car_Tyre_" + size.split()[0].replace("/", "_") + size.split()[1]
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    W, H = 1024, 512
    cv = Canvas(W, H, 0, TYRE_CIRC, 0, TYRE_LEN, ss=2, rough=0.86)
    rub = (0.024, 0.024, 0.025)
    cv.poly([(0, 0), (TYRE_CIRC, 0), (TYRE_CIRC, TYRE_LEN), (0, TYRE_LEN)], rub, rough=0.88)
    # tread: four circumferential grooves and sipes across the blocks (v 0.155 .. 0.265)
    t0, t1 = 0.16, 0.26
    for v in np.linspace(t0 + 0.015, t1 - 0.015, 4):
        cv.lines([[(0, v), (TYRE_CIRC, v)]], 0.009, (0.006, 0.006, 0.006), rough=0.95)
    for u in np.arange(0, TYRE_CIRC, 0.028):
        cv.lines([[(u, t0 - 0.01), (u + 0.006, t0 + 0.012)]], 0.003, (0.008, 0.008, 0.008))
        cv.lines([[(u + 0.014, t1 - 0.012), (u + 0.02, t1 + 0.01)]], 0.003, (0.008, 0.008, 0.008))
        for v0, v1 in ((t0 + 0.02, t0 + 0.036), (t1 - 0.036, t1 - 0.02)):
            cv.lines([[(u + 0.004, v0), (u + 0.01, v1)]], 0.0016, (0.01, 0.01, 0.01))
    # outer sidewall lettering (v 0.29 .. 0.40; letters upright when the tyre is fitted)
    let = (0.05, 0.05, 0.052)
    for k in range(2):
        u0 = k * TYRE_CIRC / 2
        cv.text("ROADSTAR", u0 + 0.20, 0.335, 0.026, let, stroke=0.2, flip_y=True, rough=0.7)
        cv.text(size, u0 + 0.20 + 0.3, 0.338, 0.016, let, stroke=0.18, flip_y=True, rough=0.7)
        cv.text("ECO TOURING", u0 + 0.06, 0.342, 0.012, let, stroke=0.18, flip_y=True, rough=0.7)
    # a moulded rib near the rim
    cv.lines([[(0, 0.392), (TYRE_CIRC, 0.392)]], 0.003, (0.04, 0.04, 0.042))
    cv.lines([[(0, 0.028), (TYRE_CIRC, 0.028)]], 0.003, (0.04, 0.04, 0.042))
    m = VP.image_material(name, cv, mask=False, extension="REPEAT")
    return m


# -- scrap paint ----------------------------------------------------------------------------------

def scrap_paint():
    name = "Car_ScrapPaint"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    W = H = 1024
    T = 2.0
    cv = Canvas(W, H, 0, T, 0, T, ss=1, rough=0.62)
    n1 = VP.noise2(H, W, 5, seed=11, octaves=5)
    n2 = VP.noise2(H, W, 18, seed=12, octaves=4)
    n3 = VP.noise2(H, W, 60, seed=13, octaves=3)
    fade = np.clip((n1 - 0.3) * 1.6, 0, 1)
    paint = np.array([0.30, 0.08, 0.055]) * (1 - fade[..., None]) + np.array([0.42, 0.17, 0.12]) * fade[..., None]
    paint *= (0.92 + 0.16 * n3)[..., None]
    rust_m = np.clip((n2 * 0.6 + n1 * 0.4 - 0.70) * 9, 0, 1)
    # drips: rust runs downwards from the blotches and from random seams
    rng = np.random.default_rng(5)
    drip = np.zeros((H, W), np.float32)
    for _ in range(45):
        x = rng.integers(0, W)
        y0 = rng.integers(0, H)
        ln = rng.integers(30, 260)
        w = rng.integers(1, 4)
        ys = np.arange(max(0, y0 - ln), y0)
        a = ((ys - (y0 - ln)) / ln) ** 1.5 * rng.uniform(0.25, 0.7)
        xs = np.clip(x + (np.sin(ys * 0.05 + x) * 1.5).astype(int), 0, W - 1)
        for dx in range(-w, w + 1):
            drip[ys, np.clip(xs + dx, 0, W - 1)] = np.maximum(drip[ys, np.clip(xs + dx, 0, W - 1)], a * (1 - abs(dx) / (w + 1)))
    rust_c = np.array([0.16, 0.06, 0.022]) * (0.6 + 0.8 * n3[..., None])
    dark = np.array([0.05, 0.025, 0.012])
    pits = np.clip((n3 - 0.6) * 5, 0, 1) * rust_m
    col = paint * (1 - rust_m[..., None]) + rust_c * rust_m[..., None]
    col = col * (1 - pits[..., None]) + dark * pits[..., None]
    col = col * (1 - drip[..., None] * 0.8) + np.array([0.13, 0.05, 0.02]) * drip[..., None] * 0.8
    # chalky bloom and scratches
    for _ in range(18):
        x, y = rng.uniform(0, T, 2)
        a = rng.uniform(0, math.pi)
        l = rng.uniform(0.04, 0.18)
        cv.lines([[(x, y), (x + l * math.cos(a), y + l * math.sin(a))]], 0.0015, (0.3, 0.22, 0.18), alpha=0.35)
    sc = cv.rgba()[..., :3]
    scr = cv.A[..., None]
    col = col * (1 - scr) + sc * scr
    cv.P[...] = col
    cv.A[...] = 1.0
    cv.R[...] = 0.55 + 0.35 * rust_m + 0.1 * drip
    m = VP.image_material(name, cv, mask=False, extension="REPEAT")
    m["uv_tile"] = T
    return m


def box_uv_for(obj, mat, tile=2.0):
    """Metric box UVs (in tiles) for the faces using mat only."""
    me = obj.data
    idx = [i for i, m in enumerate(me.materials) if m == mat]
    if not idx:
        return
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    layer = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    uv = np.zeros(len(me.loops) * 2, np.float32)
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    lv = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", lv)
    for p in me.polygons:
        if p.material_index not in idx:
            continue
        ax = int(np.argmax(np.abs(p.normal)))
        keep = [i for i in range(3) if i != ax]
        for li in range(p.loop_start, p.loop_start + p.loop_total):
            uv[li] = co[lv[li], keep] / tile
    layer.data.foreach_set("uv", uv.ravel())


# -- fascia sheets --------------------------------------------------------------------------------

def _gap_line(cv, pts, w=0.0035):
    cv.lines([pts], w, GAP, rough=0.9)


def front_golf(cv, old=False, smashed=False):
    for m in (1, -1):
        L = [(0.30, 0.768), (0.30, 0.712), (0.42, 0.700), (0.62, 0.694), (0.78, 0.705), (0.90, 0.738), (0.92, 0.775),
             (0.80, 0.787), (0.55, 0.785)]
        L = [(m * x, y) for x, y in L]

        def inner(cv, m=m):
            for x in (0.46, 0.575):
                projector(cv, m * x, 0.740, 0.022)
            cv.lines([[(m * 0.31, 0.716), (m * 0.6, 0.703), (m * 0.80, 0.713), (m * 0.90, 0.745)]], 0.0055, WHITE_LED, rough=0.1)
            cv.lines([[(m * 0.68, 0.770), (m * 0.84, 0.776)]], 0.006, AMBER, rough=0.1)
            cv.ring(m * 0.73, 0.742, 0.018, 0.024, (0.25, 0.26, 0.27), rough=0.1, metal=1.0)
        lamp_housing(cv, L, inner=inner)
        # bonnet edge above the lamp, wing / bumper split down to the arch
        _gap_line(cv, [(m * 0.30, 0.776), (m * 0.55, 0.792), (m * 0.80, 0.795), (m * 0.93, 0.786)])
        _gap_line(cv, [(m * 0.92, 0.735), (m * 0.98, 0.62), (m * 1.04, 0.50), (m * 1.08, 0.42)])
        # side intakes with a horizontal blade
        S = round_poly([(m * 0.58, 0.29), (m * 0.75, 0.29), (m * 0.82, 0.50), (m * 0.65, 0.50)], 0.02)
        cv.poly(S, hexmesh(0.02), rough=0.45)
        outline(cv, S, 0.008, GB, rough=0.12)
        cv.lines([[(m * 0.62, 0.40), (m * 0.79, 0.40)]], 0.014, GB, rough=0.12)
    # upper grille between the lamps with the light bar and badge
    G = rrect(-0.30, 0.704, 0.30, 0.770, 0.01)
    cv.poly(G, GB, rough=0.1)
    cv.lines([[(-0.30, 0.723), (-0.08, 0.723)], [(0.08, 0.723), (0.30, 0.723)]], 0.0035, WHITE_LED, rough=0.1)
    badge(cv, 0, 0.737, 0.07, "shield")
    # lower intake and lip
    I = round_poly([(-0.52, 0.28), (0.52, 0.28), (0.42, 0.425), (-0.42, 0.425)], 0.03)
    cv.poly(I, hexmesh(0.024), rough=0.45)
    outline(cv, I, 0.01, GB, rough=0.12)
    cv.poly([(-0.86, 0.222), (0.86, 0.222), (0.84, 0.262), (-0.84, 0.262)], TB, rough=0.65)
    # plate mount and tow-eye cover
    _gap_line(cv, [(-0.46, 0.475), (-0.42, 0.475)])
    outline(cv, rrect(0.47, 0.34, 0.52, 0.37, 0.012), 0.0025, GAP)


def front_scrap(cv):
    """An older hatch: deep rectangular lamps (one smashed), slatted grille, black bumper."""
    rng = np.random.default_rng(4)
    for m in (1, -1):
        L = [(m * x, y) for x, y in [(0.32, 0.69), (0.74, 0.685), (0.86, 0.72), (0.88, 0.78), (0.32, 0.785)]]

        def inner(cv, m=m):
            refl = lambda X, Y, cx=0.0: np.repeat((0.10 + 0.22 * np.clip(1 - np.hypot(X - cx, Y - 0.737) / 0.05, 0, 1))[..., None], 3, -1)
            cv.poly([(m * 0.33, 0.70), (m * 0.74, 0.695), (m * 0.74, 0.778), (m * 0.33, 0.778)],
                    lambda X, Y: refl(X, Y, m * 0.53), rough=0.15, metal=0.8)
            cv.lines([[(m * x, 0.698), (m * x, 0.776)] for x in np.arange(0.36, 0.74, 0.03)], 0.002, (0.18, 0.18, 0.19), alpha=0.6)
            cv.lines([[(m * 0.33, y), (m * 0.74, y)] for y in np.arange(0.71, 0.775, 0.02)], 0.0015, (0.2, 0.2, 0.21), alpha=0.5)
            cv.poly([(m * 0.76, 0.70), (m * 0.86, 0.72), (m * 0.87, 0.775), (m * 0.76, 0.775)], (0.5, 0.2, 0.02), rough=0.15)
        lamp_housing(cv, L, smoke=(0.06, 0.06, 0.062), inner=inner)
        _gap_line(cv, [(m * 0.30, 0.79), (m * 0.88, 0.79)])
        _gap_line(cv, [(m * 0.90, 0.70), (m * 1.0, 0.55), (m * 1.06, 0.42)])
    # smashed left lamp (car's left = -x): a dark hole with shards and cracks
    hole = [(-0.40, 0.70), (-0.47, 0.715), (-0.55, 0.70), (-0.60, 0.73), (-0.66, 0.705), (-0.70, 0.735), (-0.66, 0.765),
            (-0.58, 0.75), (-0.50, 0.775), (-0.43, 0.76)]
    cv.poly(hole, (0.02, 0.018, 0.016), rough=0.8)
    for _ in range(14):
        a = rng.uniform(0, 2 * math.pi)
        x0, y0 = -0.55 + 0.1 * math.cos(a), 0.737 + 0.03 * math.sin(a)
        cv.lines([[(x0, y0), (x0 + 0.06 * math.cos(a), y0 + 0.03 * math.sin(a))]], 0.0015, (0.5, 0.5, 0.5), alpha=0.7)
    G = rrect(-0.30, 0.69, 0.30, 0.785, 0.008)
    cv.poly(G, slats(0.022, vertical=False, width=0.5), rough=0.5)
    outline(cv, G, 0.006, (0.05, 0.05, 0.05))
    cv.ring(0, 0.737, 0.03, 0.04, (0.05, 0.05, 0.05))       # the badge has gone, its mount ring is left
    # unpainted bumper
    B = [(-1.0, 0.24), (1.0, 0.24), (1.0, 0.54), (0.6, 0.555), (-0.6, 0.555), (-1.0, 0.54)]
    nz = VP.noise2(cv.H, cv.W, 40, seed=8, octaves=3)

    def n(X, Y):
        c = np.clip(((X - cv.x0) / cv.sx).astype(int), 0, cv.W - 1)
        r = np.clip(((Y - cv.y0) / cv.sy).astype(int), 0, cv.H - 1)
        g = 0.035 + 0.03 * nz[r, c] + 0.04 * np.clip((0.33 - Y) / 0.1, 0, 1)
        return np.stack([g * 1.05, g, g * 0.92], -1)
    cv.poly(B, n, rough=0.85)
    cv.poly(rrect(-0.42, 0.29, 0.42, 0.38, 0.02), slats(0.03, vertical=False), rough=0.6)
    for m in (1, -1):
        cv.ellipse(m * 0.66, 0.36, 0.045, 0.03, (0.08, 0.08, 0.08), rough=0.2)
    cv.lines([[(0.2, 0.30), (0.33, 0.45), (0.36, 0.52)]], 0.002, (0.1, 0.1, 0.1))   # a crack in the bumper


def front_bmw(cv):
    # one wide hexagonal grille (the kit's own design) with a satin frame and a wing badge
    Gx = round_poly([(-0.25, 0.672), (0.25, 0.672), (0.285, 0.60), (0.24, 0.49), (-0.24, 0.49), (-0.285, 0.60)], 0.02)
    cv.poly(Gx, SATIN, rough=0.2, metal=0.8)
    Gi = round_poly([(-0.237, 0.660), (0.237, 0.660), (0.27, 0.60), (0.228, 0.502), (-0.228, 0.502), (-0.27, 0.60)], 0.016)
    cv.poly(Gi, hexmesh(0.022), rough=0.4)
    cv.lines([[(-0.26, 0.60), (0.26, 0.60)]], 0.012, GB, rough=0.12)
    badge(cv, 0, 0.60, 0.04, "wing")
    for m in (1, -1):
        L = [(m * x, y) for x, y in [(0.29, 0.605), (0.40, 0.598), (0.445, 0.628), (0.80, 0.632), (0.885, 0.66), (0.87, 0.693),
                                      (0.60, 0.70), (0.30, 0.68)]]

        def inner(cv, m=m):
            for x in (0.47, 0.61):
                cv.ring(m * x, 0.663, 0.024, 0.033, (0.45, 0.46, 0.47), rough=0.08, metal=1.0)
                cv.ellipse(m * x, 0.663, 0.024, 0.024, (0.03, 0.033, 0.04), rough=0.03)
                cv.lines([arc(m * x, 0.663, 0.029, 0.029, 100 if m > 0 else 80, -40 if m > 0 else 220, 12)], 0.004, WHITE_LED, rough=0.1)
            cv.lines([[(m * 0.30, 0.612), (m * 0.40, 0.606), (m * 0.44, 0.635)]], 0.005, WHITE_LED)
            cv.lines([[(m * 0.72, 0.640), (m * 0.86, 0.666)]], 0.005, AMBER)
        lamp_housing(cv, L, inner=inner)
        _gap_line(cv, [(m * 0.30, 0.688), (m * 0.6, 0.708), (m * 0.88, 0.70)])
        _gap_line(cv, [(m * 0.885, 0.655), (m * 0.96, 0.52), (m * 1.02, 0.36)])
        S = round_poly([(m * x, y) for x, y in [(0.52, 0.245), (0.80, 0.255), (0.84, 0.50), (0.75, 0.515), (0.70, 0.34), (0.52, 0.335)]], 0.02)
        cv.poly(S, hexmesh(0.02), rough=0.45)
        outline(cv, S, 0.009, GB, rough=0.12)
    I = round_poly([(-0.44, 0.245), (0.44, 0.245), (0.40, 0.37), (-0.40, 0.37)], 0.025)
    cv.poly(I, hexmesh(0.022), rough=0.45)
    outline(cv, I, 0.009, GB, rough=0.12)
    cv.lines([[(-0.40, 0.31), (0.40, 0.31)]], 0.012, GB, rough=0.12)
    cv.poly([(-0.9, 0.212), (0.9, 0.212), (0.88, 0.245), (-0.88, 0.245)], TB, rough=0.65)


def front_qashqai(cv):
    G = round_poly([(-0.42, 0.865), (0.42, 0.865), (0.37, 0.60), (0.22, 0.475), (-0.22, 0.475), (-0.37, 0.60)], 0.03)
    cv.poly(G, diamond_mesh(0.032), rough=0.15)
    for m in (1, -1):
        cv.lines([[(m * 0.46, 0.85), (m * 0.40, 0.60), (m * 0.235, 0.46), (m * 0.0, 0.445)]], 0.026, CH, rough=0.08, metal=1.0)
        L = [(m * x, y) for x, y in [(0.44, 0.835), (0.46, 0.885), (0.65, 0.90), (0.88, 0.89), (0.95, 0.865), (0.93, 0.83), (0.62, 0.818)]]

        def inner(cv, m=m):
            for x in (0.58, 0.66, 0.74):
                cv.poly(rrect(m * x - 0.025, 0.842, m * x + 0.025, 0.875, 0.006), (0.25, 0.26, 0.27), rough=0.08, metal=1.0)
                cv.poly(rrect(m * x - 0.015, 0.848, m * x + 0.015, 0.868, 0.004), (0.02, 0.022, 0.03), rough=0.03)
            cv.lines([[(m * 0.92, 0.836), (m * 0.62, 0.826), (m * 0.47, 0.83)]], 0.006, WHITE_LED)
            cv.lines([[(m * 0.86, 0.88), (m * 0.92, 0.868)]], 0.005, AMBER)
        lamp_housing(cv, L, inner=inner)
        _gap_line(cv, [(m * 0.44, 0.905), (m * 0.7, 0.915), (m * 0.95, 0.90)])
        _gap_line(cv, [(m * 0.95, 0.83), (m * 1.0, 0.70), (m * 1.04, 0.56)])
        cv.poly(round_poly([(m * x, y) for x, y in [(0.70, 0.56), (0.80, 0.56), (0.86, 0.74), (0.77, 0.74)]], 0.015), GB, rough=0.12)
        F = rrect(m * 0.66 - 0.07, 0.445, m * 0.66 + 0.07, 0.535, 0.02) if m > 0 else rrect(-0.73, 0.445, -0.59, 0.535, 0.02)
        cv.poly(F, GB, rough=0.12)
        projector(cv, m * 0.66, 0.49, 0.022)
    # cladding, skid plate, lower intake
    cv.poly([(-1.1, 0.36), (1.1, 0.36), (1.1, 0.44), (0.62, 0.445), (0.5, 0.50), (-0.5, 0.50), (-0.62, 0.445), (-1.1, 0.44)], TB, rough=0.7)
    cv.poly(round_poly([(-0.36, 0.37), (0.36, 0.37), (0.32, 0.43), (-0.32, 0.43)], 0.015), SATIN, rough=0.35, metal=0.7)
    I = round_poly([(-0.30, 0.44), (0.30, 0.44), (0.28, 0.495), (-0.28, 0.495)], 0.012)
    cv.poly(I, hexmesh(0.02), rough=0.45)
    badge(cv, 0, 0.745, 0.08, "diamond")


def front_transit(cv):
    """Large trapezoid grille with three bars and a generic badge, swept lamp clusters, black bumper."""
    G = round_poly([(-0.50, 1.04), (0.50, 1.04), (0.44, 0.66), (-0.44, 0.66)], 0.05)
    cv.poly(G, (0.02, 0.02, 0.021), rough=0.2)
    Gi = round_poly([(-0.475, 1.02), (0.475, 1.02), (0.418, 0.68), (-0.418, 0.68)], 0.04)
    cv.poly(Gi, hexmesh(0.024), rough=0.45)
    for z in (0.765, 0.865, 0.965):
        hw = 0.42 + (z - 0.68) * 0.15
        cv.poly(rrect(-hw, z - 0.026, hw, z + 0.026, 0.012),
                vgrad((0.012, 0.012, 0.013), (0.06, 0.062, 0.065), z - 0.026, z + 0.026), rough=0.15)
        cv.lines([[(-hw + 0.01, z + 0.022), (hw - 0.01, z + 0.022)]], 0.003, (0.12, 0.12, 0.125), rough=0.1)
    # a small generic badge (chrome chevron on a dark shield)
    cv.poly(round_poly([(-0.06, 0.885), (0.06, 0.885), (0.045, 0.848), (0.0, 0.838), (-0.045, 0.848)], 0.008), CH, rough=0.08, metal=1.0)
    cv.poly(round_poly([(-0.05, 0.879), (0.05, 0.879), (0.038, 0.852), (0.0, 0.845), (-0.038, 0.852)], 0.006), (0.02, 0.02, 0.025), rough=0.1)
    cv.lines([[(-0.025, 0.872), (0.0, 0.856), (0.025, 0.872)]], 0.005, CH, rough=0.08, metal=1.0)
    for m in (1, -1):
        L = [(m * x, y) for x, y in [(0.52, 0.93), (0.52, 1.045), (0.66, 1.10), (0.86, 1.16), (1.02, 1.205), (1.09, 1.18),
                                      (1.05, 1.08), (0.84, 0.995), (0.64, 0.94)]]

        def inner(cv, m=m):
            for x0, x1, y0, y1 in ((0.56, 0.68, 0.96, 1.035), (0.71, 0.83, 1.0, 1.08)):
                P = [(m * x0, y0), (m * x1, y0 + 0.03), (m * x1, y1 + 0.03), (m * x0, y1)]
                cv.poly(P, lambda X, Y: np.repeat((0.18 + 0.25 * np.clip(1 - np.abs(Y - (y0 + y1) / 2 - 0.015) / 0.05, 0, 1))[..., None], 3, -1),
                        rough=0.08, metal=0.9)
                cv.lines([[(m * (x0 + 0.015), y0 + 0.035), (m * (x1 - 0.015), y0 + 0.055)]], 0.01, (0.02, 0.022, 0.03), rough=0.03)
            cv.poly([(m * 0.88, 1.10), (m * 1.02, 1.165), (m * 1.07, 1.155), (m * 1.03, 1.09), (m * 0.88, 1.035)], AMBER, rough=0.1)
            cv.lines([[(m * 0.54, 0.95), (m * 0.66, 0.965), (m * 0.84, 1.02), (m * 1.04, 1.095)]], 0.007, WHITE_LED)
        lamp_housing(cv, L, smoke=(0.06, 0.062, 0.066), inner=inner)
        _gap_line(cv, [(m * 0.51, 1.06), (m * 0.66, 1.115), (m * 0.86, 1.175)])
        cv.poly(rrect(m * 0.74 - 0.07, 0.50, m * 0.74 + 0.07, 0.565, 0.02) if m > 0 else rrect(-0.81, 0.50, -0.67, 0.565, 0.02), GB, rough=0.2)
        cv.ellipse(m * 0.74, 0.532, 0.028, 0.022, (0.3, 0.31, 0.32), rough=0.05, metal=0.6)
    B = [(-1.25, 0.40), (1.25, 0.40), (1.25, 0.655), (0.9, 0.64), (0.5, 0.645), (-0.5, 0.645), (-0.9, 0.64), (-1.25, 0.655)]
    cv.poly(B, TB, rough=0.7)
    cv.lines([[(-1.2, 0.648), (-0.45, 0.648)], [(0.45, 0.648), (1.2, 0.648)]], 0.004, (0.045, 0.045, 0.047))
    I = rrect(-0.38, 0.425, 0.38, 0.465, 0.012)
    cv.poly(I, hexmesh(0.018), rough=0.45)


def rear_golf(cv, old=False):
    for m in (1, -1):
        if old:
            L = [(m * x, y) for x, y in [(0.62, 0.78), (0.93, 0.78), (0.96, 0.96), (0.62, 0.975)]]
            tail_lens(cv, L, guides=[([(m * 0.66, 0.84), (m * 0.92, 0.84)], 0.004)], base=(0.12, 0.006, 0.006))
            cv.poly([(m * 0.64, 0.79), (m * 0.92, 0.79), (m * 0.925, 0.825), (m * 0.64, 0.825)], (0.35, 0.36, 0.37), rough=0.1)
            continue
        L = [(m * x, y) for x, y in [(0.22, 0.905), (0.42, 0.888), (0.70, 0.882), (0.90, 0.89), (0.97, 0.925), (0.97, 0.975),
                                      (0.70, 0.988), (0.40, 0.978), (0.22, 0.95)]]
        gl = [([(m * 0.25, 0.925), (m * 0.42, 0.912), (m * 0.70, 0.908), (m * 0.92, 0.918)], 0.004),
              ([(m * 0.26, 0.945), (m * 0.42, 0.962), (m * 0.70, 0.968), (m * 0.93, 0.958)], 0.004),
              ([(m * 0.70, 0.908), (m * 0.74, 0.938), (m * 0.70, 0.968)], 0.004)]
        tail_lens(cv, L, guides=gl)
        cv.poly([(m * 0.47, 0.927), (m * 0.55, 0.926), (m * 0.55, 0.944), (m * 0.47, 0.945)], (0.07, 0.07, 0.075), rough=0.05)
        _gap_line(cv, [(m * 0.56, 0.56), (m * 0.56, 1.0)], 0.004)
        cv.poly([(m * 0.62, 0.425), (m * 0.80, 0.43), (m * 0.80, 0.445), (m * 0.62, 0.44)], (0.3, 0.01, 0.01), rough=0.2)
    _gap_line(cv, [(-0.56, 0.56), (0.56, 0.56)], 0.004)
    D = [(-0.88, 0.30), (0.88, 0.30), (0.86, 0.40), (-0.86, 0.40)]
    cv.poly(D, TB, rough=0.7)
    cv.lines([[(-0.6, 0.385), (0.6, 0.385)]], 0.006, GB, rough=0.12)
    cv.poly(rrect(0.66, 0.335, 0.74, 0.365, 0.006), (0.22, 0.01, 0.01), rough=0.1)
    if not old:
        badge(cv, 0, 0.86, 0.06, "shield")
        cv.text("1.5", -0.40, 0.80, 0.03, CH, stroke=0.17, rough=0.1, metal=1.0)
    else:
        cv.text("1.4", -0.40, 0.80, 0.03, (0.3, 0.3, 0.3), stroke=0.17)
    cv.poly(rrect(-0.06, 0.69, 0.06, 0.705, 0.004), (0.06, 0.06, 0.06), rough=0.2)


def rear_scrap(cv):
    rear_golf(cv, old=True)
    B = [(-1.0, 0.30), (1.0, 0.30), (1.0, 0.52), (-1.0, 0.52)]
    cv.poly(B, lambda X, Y: np.repeat((0.03 + 0.015 * np.sin(X * 41) ** 2)[..., None], 3, -1), rough=0.8)
    cv.poly([(0.70, 0.86), (0.78, 0.90), (0.84, 0.86), (0.80, 0.93), (0.74, 0.92)], (0.02, 0.015, 0.015), rough=0.8)   # broken lamp


def rear_bmw(cv):
    for m in (1, -1):
        L = [(m * x, y) for x, y in [(0.30, 0.905), (0.55, 0.895), (0.55, 0.84), (0.88, 0.85), (0.955, 0.90), (0.955, 0.975),
                                      (0.55, 0.99), (0.30, 0.965)]]
        gl = [([(m * 0.62, 0.972), (m * 0.62, 0.872), (m * 0.90, 0.882)], 0.009),
              ([(m * 0.66, 0.955), (m * 0.66, 0.893), (m * 0.90, 0.90)], 0.004),
              ([(m * 0.33, 0.945), (m * 0.53, 0.955)], 0.004)]
        tail_lens(cv, L, guides=gl)
        _gap_line(cv, [(m * 0.55, 0.80), (m * 0.55, 1.03)], 0.004)
        cv.poly([(m * 0.84, 0.46), (m * 0.87, 0.46), (m * 0.875, 0.56), (m * 0.845, 0.56)], (0.3, 0.01, 0.01), rough=0.2)
        E = [(m * x, y) for x, y in [(0.42, 0.33), (0.66, 0.33), (0.68, 0.42), (0.40, 0.42)]]
        cv.poly(round_poly(E, 0.015), CH, rough=0.1, metal=1.0)
        cv.poly(round_poly([(m * x, y) for x, y in [(0.435, 0.342), (0.645, 0.342), (0.662, 0.408), (0.418, 0.408)]], 0.01),
                (0.006, 0.006, 0.006), rough=0.6)
    _gap_line(cv, [(-0.55, 0.80), (0.55, 0.80)], 0.004)
    D = [(-0.9, 0.30), (0.9, 0.30), (0.88, 0.44), (-0.88, 0.44)]
    cv.poly(D, TB, rough=0.7)
    for m in (1, -1):
        E = [(m * x, y) for x, y in [(0.42, 0.33), (0.66, 0.33), (0.68, 0.42), (0.40, 0.42)]]
        cv.poly(round_poly(E, 0.015), CH, rough=0.1, metal=1.0)
        cv.poly(round_poly([(m * x, y) for x, y in [(0.435, 0.342), (0.645, 0.342), (0.662, 0.408), (0.418, 0.408)]], 0.01),
                (0.006, 0.006, 0.006), rough=0.6)
    badge(cv, 0, 0.935, 0.045, "wing")
    cv.text("2.0d", -0.40, 0.84, 0.028, CH, stroke=0.16, rough=0.1, metal=1.0)
    cv.poly(rrect(-0.06, 0.672, 0.06, 0.685, 0.004), (0.06, 0.06, 0.06), rough=0.2)


def rear_qashqai(cv):
    for m in (1, -1):
        L = [(m * x, y) for x, y in [(0.26, 0.995), (0.50, 0.988), (0.86, 0.993), (0.965, 1.02), (0.965, 1.065), (0.80, 1.08),
                                      (0.45, 1.075), (0.24, 1.045)]]
        gl = [([(m * (0.70 + 0.05 * k), 1.005), (m * (0.73 + 0.05 * k), 1.035), (m * (0.70 + 0.05 * k), 1.065)], 0.0045) for k in range(5)]
        gl += [([(m * 0.28, 1.03), (m * 0.62, 1.035)], 0.004)]
        tail_lens(cv, L, guides=gl)
        _gap_line(cv, [(m * 0.60, 0.62), (m * 0.60, 1.12)], 0.004)
        cv.poly([(m * 0.80, 0.50), (m * 0.93, 0.50), (m * 0.93, 0.53), (m * 0.80, 0.53)], (0.3, 0.01, 0.01), rough=0.2)
    _gap_line(cv, [(-0.60, 0.62), (0.60, 0.62)], 0.004)
    cv.poly([(-1.05, 0.38), (1.05, 0.38), (1.05, 0.565), (-1.05, 0.565)], TB, rough=0.7)
    cv.poly(round_poly([(-0.40, 0.39), (0.40, 0.39), (0.36, 0.465), (-0.36, 0.465)], 0.015), SATIN, rough=0.35, metal=0.7)
    badge(cv, 0, 0.93, 0.06, "diamond")
    cv.text("HYBRID", -0.42, 0.86, 0.026, CH, stroke=0.16, rough=0.1, metal=1.0)
    cv.poly(rrect(-0.06, 0.815, 0.06, 0.828, 0.004), (0.06, 0.06, 0.06), rough=0.2)


def rear_transit(cv):
    """Twin rear doors (seam, handle, hinges), tall corner lamp clusters, step bumper, high brake light."""
    _gap_line(cv, [(0, 0.63), (0, 2.63)], 0.008)
    for m in (1, -1):
        _gap_line(cv, [(m * 0.915, 0.63), (m * 0.915, 2.63)], 0.006)
        L = round_poly([(m * x, y) for x, y in [(0.93, 0.84), (1.07, 0.84), (1.08, 1.62), (0.93, 1.62)]], 0.02)
        tail_lens(cv, L, guides=[([(m * 0.95, 1.50), (m * 1.06, 1.50)], 0.006), ([(m * 0.95, 0.90), (m * 1.06, 0.90)], 0.006)],
                  base=(0.12, 0.005, 0.006))
        cv.poly([(m * 0.94, 1.15), (m * 1.07, 1.15), (m * 1.072, 1.29), (m * 0.94, 1.29)], AMBER, rough=0.06)
        cv.poly([(m * 0.94, 0.99), (m * 1.07, 0.99), (m * 1.07, 1.13), (m * 0.94, 1.13)], (0.5, 0.51, 0.52), rough=0.04)
        outline(cv, L, 0.006, (0.02, 0.02, 0.02))
        for z in (0.95, 1.95):
            cv.poly(rrect(m * 0.87, z, m * 0.905, z + 0.13, 0.006) if m > 0 else rrect(-0.905, z, -0.87, z + 0.13, 0.006), GB, rough=0.4)
    cv.poly(rrect(-0.25, 2.60, 0.25, 2.645, 0.01), (0.35, 0.01, 0.01), rough=0.08)
    cv.poly(rrect(-0.20, 1.17, -0.03, 1.235, 0.014), (0.02, 0.02, 0.021), rough=0.5)
    cv.poly(rrect(-0.18, 1.19, -0.05, 1.215, 0.008), (0.05, 0.05, 0.052), rough=0.3)
    cv.poly([(-1.15, 0.44), (1.15, 0.44), (1.15, 0.625), (-1.15, 0.625)], TB, rough=0.7)
    for k in range(9):
        x = -0.4 + k * 0.1
        cv.lines([[(x, 0.59), (x + 0.06, 0.59)]], 0.012, (0.05, 0.05, 0.052), rough=0.5)
    cv.poly([(-0.95, 0.65), (-0.83, 0.65), (-0.83, 0.67), (-0.95, 0.67)], (0.3, 0.01, 0.01), rough=0.2)
    cv.poly([(0.83, 0.65), (0.95, 0.65), (0.95, 0.67), (0.83, 0.67)], (0.3, 0.01, 0.01), rough=0.2)
    cv.poly(rrect(0.62, 0.47, 0.74, 0.51, 0.008), (0.25, 0.008, 0.008), rough=0.1)        # rear fog lamp


FRONTS = {"golf": front_golf, "bmw": front_bmw, "qashqai": front_qashqai, "transit": front_transit, "scrap": front_scrap}
REARS = {"golf": rear_golf, "bmw": rear_bmw, "qashqai": rear_qashqai, "transit": rear_transit, "scrap": rear_scrap}


def fascia(s, end):
    name = f"Car_Fascia_{s.kind}_{'F' if end > 0 else 'R'}"
    if name in _CANVAS and bpy.data.materials.get(name):
        return _CANVAS[name], bpy.data.materials[name]
    z0, z1 = (s.fascia[0], s.fascia[1]) if end > 0 else (s.fascia[2], s.fascia[3])
    half = 1.3
    W = 1024
    H = int(round(W * (z1 - z0) / (2 * half) / 8)) * 8
    H = max(H, 128)
    W2 = W if H < 400 else 1024
    cv = Canvas(W2, H, -half, half, z0, z1, ss=3, rough=0.2)
    (FRONTS if end > 0 else REARS)[s.kind](cv)
    _CANVAS[name] = cv
    return cv, VP.image_material(name, cv, mask=True)


# -- fitting --------------------------------------------------------------------------------------

def _auto_grid(soup, G, Nref, mat, uv=None, keep=None):
    """Grid oriented so its faces point along the reference normals."""
    nu, nv = G.shape[:2]
    i, j = nu // 2 - (1 if nu > 2 else 0), nv // 2 - (1 if nv > 2 else 0)
    i, j = max(0, min(i, nu - 2)), max(0, min(j, nv - 2))
    n = np.cross(G[i + 1, j] - G[i, j], G[i, j + 1] - G[i, j])
    flip = np.dot(n, Nref[i, j]) < 0
    soup.grid(G, mat, uv=uv, keep=keep, flip=flip)


def wrap_patch(fit, soup, s, end, cv, mat, lod):
    z0, z1 = (s.fascia[0], s.fascia[1]) if end > 0 else (s.fascia[2], s.fascia[3])
    yE = s.yF if end > 0 else s.yR
    Cy = yE - end * s.wrap
    thm = math.radians(64)
    big = s.kind == "transit"
    nu = 30 if lod else (60 if big else 88)
    nv = max(4, int((z1 - z0) / (0.04 if lod else (0.022 if big else 0.014))))
    th = np.linspace(-thm, thm, nu + 1)
    zs = np.linspace(z0, z1, nv + 1)
    O = np.zeros((nu + 1, nv + 1, 3))
    D = np.zeros_like(O)
    for i, t in enumerate(th):
        d = np.array([math.sin(t), end * math.cos(t), 0.0])
        for j, z in enumerate(zs):
            O[i, j] = np.array([0.0, Cy, z]) + d * 6.0
            D[i, j] = -d
    H, N, ok = fit.cast_grid(O, D)
    for i in range(nu + 1):
        for j in range(nv + 1):
            if ok[i, j] and np.dot(N[i, j], -D[i, j]) < 0.1:
                ok[i, j] = False
    c = nu // 2
    U = np.zeros((nu + 1, nv + 1))
    Vv = np.zeros((nu + 1, nv + 1))
    # surface distances from a reference row (mid face) and the centre column, along runs of
    # hits that stay on the same surface
    zmid = 0.5 * (z0 + z1)
    jref = int(np.argmin(np.abs(zs - zmid)))
    for i in range(nu + 1):
        if not ok[i, jref]:
            ok[i, :] = False
            continue
        Vv[i, jref] = zs[jref] + getattr(s, "fascia_dz", (0.0, 0.0))[0 if end > 0 else 1]
        for rng_, st in ((range(jref + 1, nv + 1), -1), (range(jref - 1, -1, -1), 1)):
            dead = False
            for j in rng_:
                if dead or not ok[i, j] or np.linalg.norm(H[i, j] - H[i, j + st]) > 0.05:
                    dead = True
                    ok[i, j] = False
                    continue
                Vv[i, j] = Vv[i, j + st] - st * np.linalg.norm(H[i, j] - H[i, j + st])
    for j in range(nv + 1):
        for rng_, st in ((range(c + 1, nu + 1), -1), (range(c - 1, -1, -1), 1)):
            dead = not ok[c, j]
            last = 0.03
            for i in rng_:
                dd = np.linalg.norm(H[i, j] - H[i + st, j]) if ok[i, j] else 1e9
                if dead or dd > 2.5 * last + 0.02:
                    dead = True
                    ok[i, j] = False
                    continue
                last = dd
                U[i, j] = U[i + st, j] - st * dd
        if not ok[c, j]:
            ok[:, j] = False
    uvn = np.stack([(U - cv.x0) / (cv.x1 - cv.x0), (Vv - cv.y0) / (cv.y1 - cv.y0)], -1)
    step = 2 * thm / nu * 1.2 * 3

    def keep(i, j):
        if not (ok[i, j] and ok[i + 1, j] and ok[i, j + 1] and ok[i + 1, j + 1]):
            return False
        if np.linalg.norm(H[i + 1, j] - H[i, j]) > 0.12 or np.linalg.norm(H[i, j + 1] - H[i, j]) > 0.12:
            return False
        u0, u1 = sorted((U[i, j], U[i + 1, j + 1]))
        v0, v1 = sorted((Vv[i, j], Vv[i + 1, j + 1]))
        c0 = int(np.clip((min(U[i, j], U[i + 1, j], U[i, j + 1], U[i + 1, j + 1]) - cv.x0) / cv.sx, 0, cv.W - 1))
        c1 = int(np.clip((max(U[i, j], U[i + 1, j], U[i, j + 1], U[i + 1, j + 1]) - cv.x0) / cv.sx, 0, cv.W - 1))
        r0 = int(np.clip((min(Vv[i, j], Vv[i, j + 1], Vv[i + 1, j], Vv[i + 1, j + 1]) - cv.y0) / cv.sy, 0, cv.H - 1))
        r1 = int(np.clip((max(Vv[i, j], Vv[i, j + 1], Vv[i + 1, j], Vv[i + 1, j + 1]) - cv.y0) / cv.sy, 0, cv.H - 1))
        return cv.A[r0:r1 + 1, c0:c1 + 1].max() > 0.45
    _auto_grid(soup, H + N * 0.0025, N, mat, uv=uvn, keep=keep)


def cast_rect(fit, soup, axes, c, sign, rect, mat, lift=0.003, nu=4, nv=2, uv=False, rot180=False):
    x0, x1, y0, y1 = rect
    G2 = np.zeros((nu + 1, nv + 1, 2))
    for i in range(nu + 1):
        for j in range(nv + 1):
            G2[i, j] = (x0 + (x1 - x0) * i / nu, y0 + (y1 - y0) * j / nv)
    O = np.zeros((nu + 1, nv + 1, 3))
    O[..., axes[0]] = G2[..., 0]
    O[..., axes[1]] = G2[..., 1]
    O[..., c] = sign * 6.0
    d = np.zeros(3)
    d[c] = -sign
    H, N, ok = fit.cast_grid(O, np.broadcast_to(d, O.shape))
    if not ok.all():
        return False
    uvs = None
    if uv:
        uvs = np.zeros((nu + 1, nv + 1, 2))
        for i in range(nu + 1):
            for j in range(nv + 1):
                uvs[i, j] = (i / nu, j / nv)
        if axes[0] == 0 and c == 1 and sign > 0:        # seen from the front, +x is on the viewer's left
            uvs[..., 0] = 1 - uvs[..., 0]
        if rot180:
            uvs = 1 - uvs
    _auto_grid(soup, H + N * lift, N, mat, uv=uvs)
    return True


def plates(fit, soup, s, M, lod):
    t = s.plate_text
    for end, z in ((1, s.plates[0]), (-1, s.plates[1])):
        mat = plate_material(t, end > 0, worn=s.kind == "scrap")
        cast_rect(fit, soup, (0, 2), 1, end, (-0.268, 0.268, z - 0.062, z + 0.062), M["trim"], lift=0.005, nu=4, nv=1)
        cast_rect(fit, soup, (0, 2), 1, end, (-0.26, 0.26, z - 0.0555, z + 0.0555), mat, lift=0.008, nu=4, nv=1, uv=True)


def mirror(fit, soup, s, M, side, lod):
    if s.kind == "scrap" and side < 0:
        # the nearside mirror has been snapped off: only the black mounting base is left
        y, z = s.mirror
        loc, n = fit.hit((side * 3.0, y, s.belt(y) + 0.015), (-side, 0.0, 0.0))
        if loc is not None:
            soup.box(loc + np.array([side * 0.02, -0.01, 0.01]), (0.04, 0.07, 0.05), M["trim"])
        return
    y, z = s.mirror
    zh = s.belt(y) + 0.015
    loc, n = fit.hit((side * 3.0, y, zh), (-side, 0.0, 0.0))
    if loc is None:
        return
    van = s.kind == "transit"
    ax = (0.11, 0.06, 0.15) if van else (0.1, 0.058, 0.07)
    out = 0.07 if van else 0.04
    c = loc + np.array([side * (out + ax[0]), -0.02, (z - zh) + (0.04 if not van else 0.0)])
    Fm = np.array([[side, 0, 0], [0, 1, 0], [0, 0, 1]], float).T
    paint, base = (M["clad"], M["clad"]) if van else (M["paint"], M["trim"])

    def matf(ph, al):
        return base if math.sin(ph) < -0.55 or (math.cos(ph) < -0.7) else paint
    soup.blob(c, ax, paint, axis=Fm, e=(0.5, 0.45), nu=10 if lod else 16, nv=5 if lod else 9, taper=(0.22, 0.18), matf=matf)
    # glass on the rear face
    if not lod:
        pts = []
        for k in range(20):
            a = 2 * math.pi * k / 20
            px = np.sign(math.cos(a)) * abs(math.cos(a)) ** 0.4 * ax[0] * 0.82
            pz = np.sign(math.sin(a)) * abs(math.sin(a)) ** 0.4 * ax[2] * 0.78
            t = (px / ax[0] + 1) / 2
            pts.append(c + np.array([side * px, -ax[1] * 0.88, pz * (1 - 0.18 * t)]))
        nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        soup.face(pts if nrm[1] < 0 else pts[::-1], M["mirror"])
    # arm to the door
    arm_len = out + 0.03
    soup.box(loc + np.array([side * arm_len / 2, -0.01, 0.0 if van else 0.01]), (arm_len, 0.05, 0.035 if not van else 0.03),
             M["trim"] if not van else M["clad"])
    if van:
        soup.box(loc + np.array([side * arm_len / 2, -0.01, -0.11]), (arm_len, 0.03, 0.025), M["clad"])


def door_lines(fit, soup, s, M):
    ar = s.axles[0]
    R = s.arch + 0.055
    for side in (-1, 1):
        d = (-side, 0, 0)
        X = side * 1.5
        for yf, yb in s.doors:
            zlow = s.sill(0) + s.rocker + 0.02
            for y in (yf, yb):
                pts = []
                if abs(y - ar) < R:
                    zm = s.wheel_r + math.sqrt(max(R * R - (y - ar) ** 2, 0))
                    pts = [(X, y, z) for z in np.linspace(s.belt(y) - 0.006, zm, 8)]
                    a0 = math.atan2(zm - s.wheel_r, y - ar)
                    a1 = math.atan2(zlow - s.wheel_r, math.sqrt(max(R * R - (zlow - s.wheel_r) ** 2, 0)))
                    for a in np.linspace(a0, a1, 10)[1:]:
                        pts.append((X, ar + R * math.cos(a), s.wheel_r + R * math.sin(a)))
                else:
                    pts = [(X, y + 0.01 * (z - zlow), z) for z in np.linspace(zlow, s.belt(y) - 0.006, 10)]
                fit.strip(pts, 0.0045, d, M["frit"])
            ylo = min(yf, yb)
            if abs(ylo - ar) < R:
                ylo = ar + math.sqrt(max(R * R - (zlow - s.wheel_r) ** 2, 0))
            fit.strip([(X, y, zlow) for y in np.linspace(ylo, max(yf, yb), 12)], 0.0045, d, M["frit"])


def bonnet_lines(fit, soup, s, M):
    yc = s.screens[0]["y"][0] - 0.06
    yn = s.yF - 0.3
    for sx in (-1, 1):
        pts = [(sx * (s.halfw(y) - 0.05 * s.sculpt - 0.06), y, 4.0) for y in np.linspace(yc, yn, 14)]
        fit.strip(pts, 0.0045, (0, 0, -1), M["frit"])
    fit.strip([(sx * (s.halfw(yc) - 0.05 * s.sculpt - 0.06), yc, 4.0) for sx in np.linspace(-1, 1, 16)], 0.0045, (0, 0, -1), M["frit"])


def handles(fit, soup, s, M):
    van = s.kind == "transit"
    for side in (-1, 1):
        for y in s.handles:
            loc, n = fit.hit((side * 3.0, y, s.handle_z), (-side, 0, 0))
            if loc is None:
                continue
            cast_rect(fit, soup, (1, 2), 0, side, (y - 0.07, y + 0.07, s.handle_z - 0.022, s.handle_z + 0.008), M["frit"],
                      lift=0.0015, nu=3, nv=1)
            Fm = np.array([[0, 1, 0], [side, 0, 0], [0, 0, 1]], float).T
            soup.blob(loc + n * 0.012, (0.085, 0.014, 0.019), M["clad"] if van else M["paint"], axis=Fm, e=(0.35, 0.6), nu=10, nv=6)


def wipers(fit, soup, s, M):
    from .vehicles import section
    sc = s.screens[0]
    yb = sc["y"][0]
    # the scuttle panel spans the foot of the glass between the A-pillars (not out over the wings)
    h, keys, _ = section(s, yb, False)
    xb = min(sc["x"][0], h[keys[10]][0] - sc.get("inset_x", 0.0))
    cast_rect(fit, soup, (0, 1), 2, 1, (-xb - 0.02, xb + 0.02, yb - 0.01, yb + 0.05), M["clad"], lift=0.004, nu=12, nv=2)
    for x0, x1, yo in ((-0.6, -0.04, 0.02), (0.0, 0.55, 0.035)):
        pts, nrm = [], []
        for x in np.linspace(x0, x1, 8):
            loc, n = fit.hit((x, yb - yo - 0.02 * (x - x0), 4.0), (0, 0, -1))
            if loc is not None:
                pts.append(loc + n * 0.002)
                nrm.append(n)
        if len(pts) > 2:
            soup.tube(pts, nrm, 0.013, 0.011, M["seal"])


def spoiler(fit, soup, s, M):
    sc = s.screens[1]
    yt = sc["y"][1]
    rw = s.roofw(yt) - 0.015
    xs = np.linspace(-rw, rw, 13)
    G = []
    for x in xs:
        loc, n = fit.hit((x, yt + 0.05, 4.0), (0, 0, -1))
        if loc is None:
            return
        z = loc[2]
        G.append([(x, yt + 0.05, z + 0.002), (x, yt - 0.06, z + 0.006), (x, yt - 0.11, z - 0.003), (x, yt - 0.105, z - 0.028),
                  (x, yt - 0.01, z - 0.026)])
    G = np.array(G)
    soup.grid(G, lambda i, j: M["trim"] if j >= 2 else M["paint"])
    for k in (0, -1):
        soup.face(list(G[k]) if k == 0 else list(G[k])[::-1], M["trim"])


def rear_wiper(fit, soup, s, M):
    sc = s.screens[1]
    yb = sc["y"][0]
    pts, nrm = [], []
    for t in np.linspace(0, 1, 6):
        loc, n = fit.hit((-0.03 - 0.36 * t, yb + 0.05 + 0.03 * t, 4.0), (0, 0, -1))
        if loc is not None:
            pts.append(loc + n * 0.004)
            nrm.append(n)
    if len(pts) > 2:
        soup.tube(pts, nrm, 0.012, 0.01, M["seal"])
    loc, n = fit.hit((0.0, yb + 0.03, 4.0), (0, 0, -1))
    if loc is not None:
        soup.blob(loc + n * 0.01, (0.03, 0.025, 0.02), M["trim"], e=(0.6, 0.6), nu=8, nv=4)


def fin(fit, soup, s, M):
    loc, n = fit.hit((0.0, s.fin, 4.0), (0, 0, -1))
    if loc is None:
        return
    Fm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float).T
    soup.blob(loc + np.array([0, 0, 0.02]), (0.09, 0.03, 0.045), M["trim"], axis=Fm, e=(0.5, 0.55), nu=10, nv=6, taper=(0.5, 0.8))


def roof_rails(fit, soup, s, M):
    for sx in (-1, 1):
        pts, nrm = [], []
        for y in np.linspace(0.05, -1.62, 10):
            x = sx * (s.roofw(y) - 0.09)
            loc, n = fit.hit((x, y, 4.0), (0, 0, -1))
            if loc is not None:
                pts.append(loc + np.array([0, 0, 0.045]))
                nrm.append(np.array([0, 0, 1.0]))
        if len(pts) > 3:
            soup.tube(pts, nrm, 0.03, 0.022, M["rail"])
            for p in (pts[0], pts[-1]):
                soup.box(p - np.array([0, 0, 0.025]), (0.035, 0.12, 0.05), M["trim"])


def b_pillars(fit, soup, s, M):
    from .vehicles import Pane
    W = sorted([w for w in s.windows if not w.get("only")], key=lambda w: -w["bot"][1])
    for a, b in zip(W[:-1], W[1:]):
        y1 = min(a["bot"][0], a["top"][0])
        y0 = max(b["bot"][1], b["top"][1])
        if y1 - y0 > 0.25 or y1 <= y0:
            continue
        for side in (-1, 1):
            zb = lambda y: s.belt(y) + 0.012
            zt = lambda y: (b.get("top_z") or s.rail_z(y) + 0.004)
            B = [(y, zb(y)) for y in np.linspace(y0 - 0.03, y1 + 0.03, 3)]
            T = [(y, zt(y)) for y in np.linspace(y0 - 0.03, y1 + 0.03, 3)]
            p = Pane(B, [B[-1], T[-1]], T, [B[0], T[0]], (1, 2), 0, side, M["trim"], see=False, inset=-0.0035, frame=0)
            from .vehicles import build_pane
            build_pane(fit, soup, p, M)


def rubbing_strip(fit, soup, s, M):
    """The van's black side protection strip, broken by the wheel arches."""
    ar, af = s.axles
    R = s.arch + 0.02
    for side in (-1, 1):
        for y0, y1 in ((s.yR + 0.12, ar - R), (ar + R, af - R)):
            pts = [(side * 2.0, y, 0.66) for y in np.linspace(y0, y1, max(3, int((y1 - y0) / 0.25)))]
            fit.strip(pts, 0.07, (-side, 0, 0), M["clad"], lift=0.006)


def van_extras(fit, soup, s, M):
    """Roof marker lamps, high-roof join line, sliding-door rail, mudflaps and exhaust."""
    from .vehicles import _mat
    amber = _mat("Car_MarkerAmber", "#d07a10", 0.2)
    red = _mat("Car_MarkerRed", "#a01010", 0.2)
    for sx in (-1, 1):
        for y, m in ((1.25, amber), (s.yR + 0.05, red)):
            loc, n = fit.hit((sx * 0.85, y, 4.0), (0, 0, -1))
            if loc is not None:
                soup.blob(loc + np.array([0, 0, 0.012]), (0.04, 0.025, 0.015), m, e=(0.5, 0.5), nu=8, nv=4)
        # join between the steel body and the high roof
        fit.strip([(sx * 2.0, y, 2.36) for y in np.linspace(1.2, s.yR + 0.1, 16)], 0.006, (-sx, 0, 0), M["frit"])
        # mudflaps behind each wheel
        for ai, ax in enumerate(s.axles):
            x = sx * (s.track[ai] - (0.12 if ai == 0 else 0.0))
            soup.box((x, ax - s.arch - 0.03, 0.27), (0.30 if ai == 0 else 0.24, 0.01, 0.32), M["clad"])
    # sliding door rail on the nearside behind the door
    fit.strip([(-2.0, y, 1.31) for y in np.linspace(0.1, -0.95, 8)], 0.026, (1, 0, 0), M["clad"], lift=0.008)
    # exhaust tailpipe under the rear, offside
    c = np.array([0.55, s.yR + 0.55, 0.33])
    Fm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], float).T
    soup.blob(c, (0.12, 0.03, 0.03), M["alloy_dark"], axis=Fm, e=(0.2, 1.0), nu=10, nv=3)


def fuel_flap(fit, soup, s, M):
    y, z = s.fuel
    P = rrect(y - 0.085, z - 0.06, y + 0.085, z + 0.06, 0.02, 4)
    P = P + [P[0]]
    fit.strip([(1.5, py, pz) for py, pz in P], 0.0035, (-1, 0, 0), M["frit"])


def scrap_extras(fit, soup, s, M):
    """A 'POLICE AWARE' notice on the windscreen and a crazed impact in it."""
    yb, yt = s.screens[0]["y"]
    ym = yb * 0.55 + yt * 0.45
    if not bpy.data.materials.get("Car_PoliceAware"):
        cv = Canvas(512, 320, 0, 0.2, 0, 0.125, ss=3, rough=0.4)
        cv.poly(rrect(0, 0, 0.2, 0.125, 0.006), (0.85, 0.62, 0.0))
        cv.text("POLICE", 0.1, 0.082, 0.026, (0.01, 0.01, 0.01), stroke=0.2)
        cv.text("AWARE", 0.1, 0.048, 0.026, (0.01, 0.01, 0.01), stroke=0.2)
        cv.text("VEHICLE REPORTED", 0.1, 0.022, 0.009, (0.01, 0.01, 0.01), stroke=0.18)
        cv.text("DO NOT REMOVE", 0.1, 0.008, 0.008, (0.01, 0.01, 0.01), stroke=0.18)
        VP.image_material("Car_PoliceAware", cv, mask=False)
    mat = bpy.data.materials["Car_PoliceAware"]
    cast_rect(fit, soup, (0, 1), 2, 1, (-0.40, -0.18, ym - 0.06, ym + 0.06), mat, lift=-0.002, nu=3, nv=2, uv=True, rot180=True)
    if not bpy.data.materials.get("Car_GlassCrack"):
        cv = Canvas(512, 512, -0.4, 0.4, -0.4, 0.4, ss=2, rough=0.1)
        rng = np.random.default_rng(9)
        cx, cy = -0.05, 0.02
        for k in range(26):
            a = rng.uniform(0, 2 * math.pi)
            pts = [(cx, cy)]
            r = 0
            ang = a
            for _ in range(int(rng.integers(3, 9))):
                r += rng.uniform(0.02, 0.07)
                ang += rng.normal(0, 0.18)
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            cv.lines([pts], 0.0016, (0.55, 0.58, 0.6), rough=0.05)
        for rr in (0.025, 0.05, 0.085):
            cv.lines([arc(cx, cy, rr, rr * 0.9, 0, 360, 30)], 0.0012, (0.5, 0.52, 0.55), alpha=0.8)
        cv.ellipse(cx, cy, 0.012, 0.012, (0.7, 0.72, 0.74))
        VP.image_material("Car_GlassCrack", cv, mask=True)
    mat = bpy.data.materials["Car_GlassCrack"]
    cast_rect(fit, soup, (0, 1), 2, 1, (-0.62, 0.18, ym - 0.32, ym + 0.32), mat, lift=-0.003, nu=8, nv=6, uv=True)
    # dead leaves in the scuttle and bird droppings on the roof
    rng = np.random.default_rng(31)
    if not bpy.data.materials.get("Car_Leaves"):
        cv = Canvas(512, 128, -0.8, 0.8, 0, 0.4, ss=2, rough=0.8)
        for _ in range(45):
            x, y = rng.uniform(-0.78, 0.78), rng.uniform(0.0, 0.4) ** 1.6 / 0.4 ** 0.6
            a = rng.uniform(0, 180)
            r = rng.uniform(0.012, 0.03)
            c = np.array([0.12, 0.06, 0.02]) * rng.uniform(0.6, 1.5) + np.array([0.0, 0.02, 0.0]) * rng.random()
            pts = [(x + r * math.cos(math.radians(a + t)) * (1 if t % 180 == 0 else 0.45), y + r * math.sin(math.radians(a + t)) * (1 if t % 180 == 0 else 0.45))
                   for t in range(0, 360, 45)]
            cv.poly(pts, tuple(c), rough=0.8)
        VP.image_material("Car_Leaves", cv, mask=True)
    cast_rect(fit, soup, (0, 1), 2, 1, (-0.8, 0.8, yb + 0.0, yb + 0.12), bpy.data.materials["Car_Leaves"], lift=0.006, nu=10, nv=2,
              uv=True, rot180=True)
    if not bpy.data.materials.get("Car_Droppings"):
        cv = Canvas(256, 256, 0, 1, 0, 1, ss=2, rough=0.7)
        for _ in range(9):
            x, y = rng.uniform(0.1, 0.9, 2)
            r = rng.uniform(0.012, 0.03)
            cv.ellipse(x, y, r, r * rng.uniform(0.6, 1.0), (0.62, 0.62, 0.58), rough=0.7)
            for _ in range(5):
                a = rng.uniform(0, 2 * math.pi)
                cv.ellipse(x + 1.8 * r * math.cos(a), y + 1.8 * r * math.sin(a), r * 0.3, r * 0.3, (0.55, 0.55, 0.5), rough=0.7)
            cv.ellipse(x, y, r * 0.35, r * 0.35, (0.12, 0.11, 0.09))
        VP.image_material("Car_Droppings", cv, mask=True)
    cast_rect(fit, soup, (0, 1), 2, 1, (-0.4, 0.4, -1.1, -0.3), bpy.data.materials["Car_Droppings"], lift=0.003, nu=4, nv=4, uv=True)


def details(fit, soup, s, M, lod, scrap):
    for end in (1, -1):
        cv, mat = fascia(s, end)
        wrap_patch(fit, soup, s, end, cv, mat, lod)
    plates(fit, soup, s, M, lod)
    for side in (-1, 1):
        mirror(fit, soup, s, M, side, lod)
    if lod:
        return
    door_lines(fit, soup, s, M)
    bonnet_lines(fit, soup, s, M)
    handles(fit, soup, s, M)
    wipers(fit, soup, s, M)
    fuel_flap(fit, soup, s, M)
    b_pillars(fit, soup, s, M)
    if s.spoiler:
        spoiler(fit, soup, s, M)
    if s.rear_wiper:
        rear_wiper(fit, soup, s, M)
    if s.fin is not None:
        fin(fit, soup, s, M)
    if s.roof_rails:
        roof_rails(fit, soup, s, M)
    if s.kind == "transit":
        rubbing_strip(fit, soup, s, M)
        van_extras(fit, soup, s, M)
    if scrap:
        scrap_extras(fit, soup, s, M)


# -- the scrapped car's blocks --------------------------------------------------------------------

def _frame(d):
    """3x3 frame with its first column along d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    a = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    return np.stack([d, a, b], 1)


def blocks(soup, M, s):
    from .vehicles import _mat, lathe
    rng = np.random.default_rng(21)
    bricks = [_mat("Car_Brick", "#7e3626", 0.9), _mat("Car_BrickDark", "#5e2a20", 0.92), _mat("Car_BrickPale", "#9a5a44", 0.9)]
    rust = _mat("Car_RustDisc", "#4e3322", 0.75, 0.3)
    caliper = _mat("Car_RustCaliper", "#3a2a22", 0.8, 0.2)
    for ai, ax in enumerate(s.axles):
        for sx in (-1, 1):
            x = sx * s.track[ai]
            # three engineering bricks crossed under the hub, the top one knocked askew
            for k in range(3):
                a = rng.uniform(-0.12, 0.12) + (math.pi / 2 if k % 2 else 0)
                ca, sa = math.cos(a), math.sin(a)
                R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
                if k == 2:
                    tl = rng.uniform(0.05, 0.12)
                    R = R @ np.array([[1, 0, 0], [0, math.cos(tl), -math.sin(tl)], [0, math.sin(tl), math.cos(tl)]])
                soup.box((x + rng.uniform(-0.02, 0.02), ax + rng.uniform(-0.02, 0.02), 0.0325 + 0.065 * k + 0.001),
                         (0.215, 0.1025, 0.065), bricks[int(rng.integers(0, 3))], R=R)
            c = np.array([x, ax, s.wheel_r + s.lift])
            # strut and coil spring up into the arch, calliper on the disc
            top = np.array([sx * (s.track[ai] - 0.16), ax - 0.02, s.wheel_r + s.arch * 0.85])
            base = c + np.array([-sx * 0.12, 0, 0.05])
            soup.blob(0.5 * (top + base), (np.linalg.norm(top - base) / 2, 0.025, 0.025), rust,
                      axis=_frame(top - base), e=(0.15, 1.0), nu=8, nv=3)
            pts, nrm = [], []
            d = (top - base) / np.linalg.norm(top - base)
            u = np.cross(d, [1.0, 0, 0]); u /= np.linalg.norm(u)
            v = np.cross(d, u)
            for i in range(41):
                ang = i / 40 * 2 * math.pi * 4.5
                p0 = base + (top - base) * (0.35 + 0.55 * i / 40)
                pts.append(p0 + 0.065 * (u * math.cos(ang) + v * math.sin(ang)))
                nrm.append(u * math.cos(ang) + v * math.sin(ang))
            soup.tube(pts, nrm, 0.013, 0.013, rust)
            soup.blob(c + np.array([-sx * 0.06, -0.08, 0.1]), (0.03, 0.08, 0.04), caliper, e=(0.4, 0.4), nu=8, nv=4)
            dr = 0.142
            lathe(soup, c, sx, [(dr, -0.012), (dr, 0.012), (0.08, 0.012), (0.075, 0.03), (0.035, 0.03), (0.0, 0.032)], rust, 24)
            lathe(soup, c, sx, [(0.06, -0.15), (0.06, -0.012)], rust, 12)
            for k in range(5):
                a = 2 * math.pi * k / 5
                p = c + np.array([sx * 0.03, 0.05 * math.cos(a), 0.05 * math.sin(a)])
                lathe(soup, p, sx, [(0.006, 0.0), (0.006, 0.03), (0.0, 0.03)], rust, 6)


# -- vertex colours: occlusion, grime, weathering --------------------------------------------------

def vertex_shading(obj, s, scrap):
    from ..core import ao
    ao.bake(obj)
    me = obj.data
    attr = me.color_attributes["Col"]
    c = np.zeros(len(attr.data) * 4, np.float32)
    attr.data.foreach_get("color", c)
    c = c.reshape(-1, 4)
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nr = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", nr)
    nr = nr.reshape(-1, 3)
    # smooth the baked occlusion over the mesh (a few passes along edges): no per-vertex speckle
    ev = np.zeros(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    for _ in range(4):
        acc = np.zeros((len(c), 3))
        cnt = np.zeros(len(c))
        np.add.at(acc, ev[:, 0], c[ev[:, 1], :3])
        np.add.at(acc, ev[:, 1], c[ev[:, 0], :3])
        np.add.at(cnt, ev[:, 0], 1)
        np.add.at(cnt, ev[:, 1], 1)
        m = cnt > 0
        c[m, :3] = 0.5 * c[m, :3] + 0.5 * acc[m] / cnt[m, None]
    z = co[:, 2] - s.lift
    g = np.clip((0.5 - z) / 0.32, 0, 1) ** 1.5 * (0.32 if scrap else 0.2)
    for ax in s.axles:
        d = np.abs(np.hypot(co[:, 1] - ax, z - s.wheel_r) - s.arch)
        g += np.clip(1 - d / 0.14, 0, 1) * (np.abs(co[:, 0]) > s.halfw(0) - 0.2) * (0.35 if scrap else 0.16)
    tail = np.clip((s.yR + 0.12 - co[:, 1]) / 0.12, 0, 1) * np.clip((1.0 - z) / 0.6, 0, 1)
    g += tail * 0.18
    g = np.clip(g, 0, 0.6)
    tint = np.array([0.30, 0.38, 0.50])          # grime darkens blue most: a warm brown film
    c[:, :3] *= 1 - g[:, None] * tint[None, :]
    if scrap:
        # vertical panels keep a deeper red; tops are bleached by the sun
        vert = np.clip(1 - np.abs(nr[:, 2]) * 1.5, 0, 1)
        c[:, :3] *= 1 - vert[:, None] * np.array([0.30, 0.58, 0.58])[None, :]
        rust = np.clip((0.55 - z) / 0.3, 0, 1) ** 1.3 * 0.55
        for ax in s.axles:
            d = np.abs(np.hypot(co[:, 1] - ax, z - s.wheel_r) - s.arch)
            rust = np.maximum(rust, np.clip(1 - d / 0.1, 0, 1) * 0.6)
        c[:, :3] *= 1 - rust[:, None] * np.array([0.35, 0.55, 0.72])[None, :]
        moss = np.clip((0.28 - z) / 0.12, 0, 1) * 0.3
        c[:, :3] *= 1 - moss[:, None] * np.array([0.45, 0.2, 0.5])[None, :]
    attr.data.foreach_set("color", c.ravel())
