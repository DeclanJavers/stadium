"""Retail, café, bar, dining and back-of-house fittings for a club's commercial and hospitality
building (kit/site/interior.py): the club store (garment rails of shirts, a slatwall of face-out
kits and scarves, gondola shelving, a mannequin in the home kit, the till counter, the shirt
printing station, stock racking), the ticket office, a café counter with its back bar and menu
board, bistro tables, the sports bar (counter with taps and stools, back bar of bottles), dining,
banqueting and lounge furniture, the trophy room's cabinets and plinth, a lectern, the commercial
kitchen line with its extraction canopy and prep table, and public toilets (cubicles, basins,
mirrors and hand dryers). Real sizes; origin on the floor at the footprint centre, +Y the front
(the customer's or user's side; wall units: the back against the wall at y = 0, facing +y).
"""
import math

import numpy as np

from .texpaint import Canvas, box_uv, decal, noise, project
from .props import PM, _paint, _plastic, clear_glass
from .props_amenity import _loft, _tube
from .props_medical import _cushion_canvas, _monitor, _white
from .props_office import NAMES, _display, _mug, _place, _task_chair


# -- shared bits ------------------------------------------------------------------------------------

SHIRT = [(-0.2, 0.0), (0.2, 0.0), (0.2, 0.46), (0.32, 0.38), (0.38, 0.5), (0.24, 0.66), (0.07, 0.7), (0.0, 0.66),
         (-0.07, 0.7), (-0.24, 0.66), (-0.38, 0.5), (-0.32, 0.38), (-0.2, 0.46)]


def _shirt_mat():
    cv = Canvas("Off_ShirtCloth", 256, 256, col="#ffffff", rough=0.8)
    r = cv.alloc(0.6, 0.7, 300)
    r.fill(col="#ffffff")
    r.field(lambda X, Y: noise(X.shape, 30, 81, 2) * 0.12, col="#a0a0a0")
    return cv.tint_material("TINT_club_shirt", "#c8202a", rough=0.8)


def _shirt_side(p, mat, x, y, z_top, s=1.0):
    """A shirt on a hanger seen side-on along a rail (its width across y), hanging from z_top."""
    poly = [(y + a * s, z_top - 0.72 * s + b * s) for a, b in SHIRT]
    p.extrude_yz(poly, x - 0.012, x + 0.012, mat)


def _shirt_face(p, mat, x, y, z_top, s=1.0):
    """A shirt hung face-out (facing +y) from z_top."""
    poly = [(x + a * s, z_top - 0.72 * s + b * s) for a, b in SHIRT]
    p.extrude(poly, y - 0.012, y + 0.012, mat)


def _hanger(p, M, x, y, z):
    p.r = 0.0
    _tube(p, [(x, y - 0.2, z - 0.06), (x, y, z), (x, y + 0.2, z - 0.06), (x, y - 0.2, z - 0.06)], 0.004, M["black"], 4)
    _tube(p, [(x, y, z), (x, y, z + 0.04), (x, y - 0.02, z + 0.055)], 0.003, M["chrome"], 4)


def _bistro_chair(M, wood):
    """A bentwood café chair facing +y: four splayed legs, a round seat, a hoop back."""
    p = PM(0.004)
    p.r = 0.0
    for x, y in ((-0.18, -0.18), (0.18, -0.18), (-0.18, 0.18), (0.18, 0.18)):
        p.cyl((x * 1.15, y * 1.15, 0.0), (x * 0.85, y * 0.85, 0.45), 0.015, wood, 8)
    _tube(p, [(0.21 * math.cos(t), 0.21 * math.sin(t), 0.2) for t in np.linspace(0, 2 * math.pi, 17)], 0.008, wood, 6, closed=True)
    p.cyl((0, 0, 0.44), (0, 0, 0.47), 0.21, wood, 20)
    _tube(p, [(-0.17, -0.15, 0.46), (-0.18, -0.22, 0.85), (0.18, -0.22, 0.85), (0.17, -0.15, 0.46)], 0.012, wood, 6, bend=0.08)
    for x in (-0.06, 0.06):
        p.cyl((x, -0.16, 0.47), (x, -0.21, 0.82), 0.007, wood, 6)
    return p


def _dining_chair(M, fabric, wood):
    """An upholstered dining chair facing +y: tapered legs, a padded seat and back."""
    p = PM(0.004)
    p.r = 0.0
    for x, y in ((-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)):
        p.cyl((x, y, 0.0), (x, y, 0.45), 0.017, wood, 8, r2=0.022)
    p.r = 0.03
    p.box((-0.23, -0.23, 0.44), (0.23, 0.24, 0.52), fabric)
    p.obox((0, -0.23, 0.78), (0.44, 0.06, 0.5), fabric, pitch=math.radians(-6))
    return p


def _bottle(p, x, y, z, h, mat, cap):
    r = h * 0.12
    p.r = 0.0
    p.lathe([(0.0, 0.0), (r, 0.0), (r, h * 0.62), (r * 0.35, h * 0.82), (r * 0.32, h), (0.0, h)], mat, c=(x, y, z), sides=10)
    p.cyl((x, y, z + h), (x, y, z + h + 0.012), r * 0.36, cap, 8)


def _glass(p, x, y, z, h=0.15, r=0.035):
    p.r = 0.0
    p.lathe([(0.0, 0.0), (r * 0.85, 0.0), (r, h), (0.0, h * 0.98)], clear_glass(), c=(x, y, z), sides=12)


def _plate(p, M, x, y, z, r=0.13):
    p.r = 0.0
    p.lathe([(0.0, 0.0), (r * 0.6, 0.0), (r, 0.018), (r * 0.95, 0.02), (0.0, 0.008)], M["white"], c=(x, y, z), sides=20)


def _mirror_glass(name="Hos_MirrorGlass"):
    """Silvered mirror glass, as the editor can show it (it has no reflections, so a pure metal
    mirror renders as dark sky): a pale blue-grey gloss, part metallic."""
    import bpy
    from ..core.nodes import material, srgb
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#c3ccd1"), Metallic=0.35, Roughness=0.08))
    return m


def _place_setting(p, M, x, y, z, yaw):
    """A plate, knife and fork either side, a wine glass and a water glass, the napkin."""
    ca, sa = math.cos(yaw), math.sin(yaw)

    def at(u, v):
        return (x + u * ca - v * sa, y + u * sa + v * ca)
    _plate(p, M, *at(0, 0), z)
    p.r = 0.001
    for u in (-0.17, 0.17):
        cx, cy = at(u, 0.0)
        p.obox((cx, cy, z + 0.003), (0.02, 0.2, 0.004), M["chrome"], yaw=yaw)
    gx, gy = at(0.16, -0.17)
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.035, 0.0), (0.004, 0.01), (0.004, 0.1), (0.035, 0.13), (0.04, 0.19), (0.0, 0.17)], clear_glass(), c=(gx, gy, z), sides=12)
    wx, wy = at(0.05, -0.2)
    _glass(p, wx, wy, z, 0.11, 0.032)
    nx, ny = at(0, 0)
    p.r = 0.004
    p.obox((nx, ny, z + 0.03), (0.1, 0.1, 0.02), _paint("Hos_Napkin", "#f2f0ea", 0.8), yaw=yaw + 0.785)


# -- the store -------------------------------------------------------------------------------------

def shirt_rail(col, M):
    """A chrome garment rail (1.2 m, 1.5 m high on a twin-wheeled base) of replica home shirts on
    black hangers, size dividers between the runs, a price card on the end. Long axis along x;
    the shirts' fronts face +y."""
    shirt = _shirt_mat()
    cv = Canvas("Hos_RailCard", 256, 256, col="#ffffff", rough=0.4)
    card = cv.alloc(0.15, 0.1, 1200)
    card.fill(col="#ffffff")
    card.text("HOME SHIRT", 0.075, 0.065, 0.016, weight="black", align="c", col="#1a1a1a")
    card.text("£75", 0.075, 0.018, 0.04, weight="black", align="c", col="#c8202a")
    cmat = cv.material("Hos_RailCardMat")
    p = PM(0.004)
    p.r = 0.0
    for x in (-0.62, 0.62):
        p.cyl((x, 0.0, 0.08), (x, 0.0, 1.5), 0.016, M["chrome"], 12)
        p.cyl((x, -0.25, 0.06), (x, 0.25, 0.06), 0.016, M["chrome"], 12)
        for y in (-0.25, 0.25):
            p.cyl((x, y, 0.03), (x, y, 0.06), 0.03, M["black"], 12)
    p.cyl((-0.64, 0.0, 1.5), (0.64, 0.0, 1.5), 0.014, M["chrome"], 12)
    n = 16
    for k in range(n):
        x = -0.52 + k * 1.04 / (n - 1)
        _hanger(p, M, x, 0.0, 1.44)
        _shirt_side(p, shirt, x, 0.0, 1.42, 0.95)
    for x in (-0.19, 0.19):
        p.r = 0.0
        p.cyl((x - 0.002, 0.0, 1.47), (x + 0.002, 0.0, 1.47), 0.06, M["white"], 16)
    p.r = 0.002
    p.box((0.62, -0.003, 1.55), (0.64, 0.003, 1.65), M["chrome"])
    decal(p, (0.66, 0.0, 1.62), (1, 0, 0), (0, 0, 1), 0.15, 0.1, cmat, card, lift=0.002)
    return p.obj("shirt_rail", col)


def merch_wall(col, M):
    """A slatwall merchandise bay (2.4 x 2.4 m) under a lit header: face-out shirts on arms (home,
    away and the goalkeeper kit), scarves hanging in a row, a shelf of caps and folded hoodies,
    boxed gifts and mugs on the lower shelves. The back against the wall at y = 0; front +y."""
    cv = Canvas("Hos_Slatwall", 1024, 1024, col="#d8d8d4", rough=0.5)
    sl = cv.alloc(2.4, 2.0, 200)
    sl.fill(col="#dcdcd8")
    sl.field(lambda X, Y: (np.mod(Y, 0.1) < 0.012) * 1.0, col="#7a7a76", dh=-0.004)
    hd = cv.alloc(2.4, 0.3, 300)
    hd.fill(col="#141a28", emit="#141a28")
    hd.text("OFFICIAL CLUB STORE", 1.2, 0.09, 0.12, weight="black", align="c", col="#ffffff", emit="#ffffff")
    sc = cv.alloc(0.18, 1.2, 300)
    sc.fill(col="#ffffff")
    for k in range(6):
        sc.rect(0.0, 0.08 + k * 0.19, 0.18, 0.15 + k * 0.19, col="#202020")
    sc.text("UNITED", 0.09, 0.5, 0.05, weight="black", align="c", col="#ffffff", cond=0.6)
    mat = cv.material("Hos_SlatwallFace")
    hmat = cv.material("Hos_StoreHeader", emission=1.4)
    scarf = cv.tint_material("TINT_club_scarf", "#c8202a", rough=0.8)
    shirt = _shirt_mat()
    away = _paint("Hos_AwayKit", "#f0f0ea", 0.8)
    gk = _paint("Hos_KeeperKit", "#2a9a4a", 0.8)
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-1.2, 0.0, 0.1), (1.2, 0.03, 2.1), mat)
    project(p, i0, sl, (1.2, 0.03, 0.1), (-1, 0, 0), (0, 0, 1))
    p.box((-1.2, 0.0, 0.0), (1.2, 0.05, 0.1), M["black"])
    p.r = 0.004
    p.box((-1.2, 0.0, 2.1), (1.2, 0.12, 2.4), M["black"])
    p.r = 0.0
    decal(p, (0.0, 0.12, 2.25), (0, 1, 0), (0, 0, 1), 2.4, 0.3, hmat, hd, lift=0.001)
    # face-out arms: three shirts each in a column
    for j, (x, m_) in enumerate(((-0.8, shirt), (0.0, away), (0.8, gk))):
        p.r = 0.0
        p.cyl((x, 0.03, 1.95), (x, 0.42, 1.95), 0.008, M["chrome"], 8)
        for k in range(3):
            _shirt_face(p, m_, x, 0.12 + k * 0.1, 1.92, 0.9)
        p.r = 0.002
        p.box((x - 0.12, 0.4, 1.96), (x + 0.12, 0.42, 2.02), M["white"])
    # scarves, then a shelf of caps and hoodies, then gifts
    for k in range(9):
        x = -1.1 + k * 0.275
        p.r = 0.0
        p.cyl((x, 0.03, 1.05), (x, 0.12, 1.05), 0.005, M["chrome"], 6)
        i0 = len(p.F)
        p.box((x - 0.09, 0.1, 0.45), (x + 0.09, 0.115, 1.05), scarf)
        project(p, i0, sc, (x + 0.09, 0.115, 0.0), (-1, 0, 0), (0, 0, 1))
    for z, kind in ((0.32, "gift"), (0.04, "gift")):
        p.r = 0.003
        p.box((-1.18, 0.03, z), (1.18, 0.35, z + 0.02), M["white"])
        x = -1.12
        k = 0
        while x < 1.05:
            w = 0.16 if k % 3 else 0.22
            c = ("#c8202a", "#141a28", "#f0f0ea", "#c8202a", "#2a2c30")[k % 5]
            p.r = 0.004
            p.box((x, 0.06, z + 0.02), (x + w - 0.02, 0.3, z + 0.02 + (0.18 if k % 2 else 0.12)), _paint("Hos_Gift" + c[1:], c, 0.5))
            x += w
            k += 1
    return p.obj("merch_wall", col)


def gondola(col, M):
    """A double-sided gondola shelving unit (1.2 x 0.9 m, 1.5 m high): a pegboard spine with
    three shelves each side of footballs in boxes, mugs, mascot toys, kit bags and boxed gifts, a
    wire basket of size 5 balls on the end. Long axis along x; faces +y and -y."""
    shirt = _shirt_mat()
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.6, -0.42, 0.0), (0.6, 0.42, 0.12), M["black"])
    p.box((-0.6, -0.02, 0.12), (0.6, 0.02, 1.5), _paint("Hos_Pegboard", "#e4e4e0", 0.5))
    rng = np.random.default_rng(31)
    for s in (-1, 1):
        for z in (0.12, 0.55, 0.98):
            p.r = 0.003
            p.box((-0.6, 0.02 * s if s > 0 else -0.42, z), (0.6, 0.42 if s > 0 else -0.02, z + 0.02), M["white"])
            x = -0.56
            while x < 0.5:
                kind = int(rng.integers(0, 4))
                yc = s * 0.22
                if kind == 0:     # mugs
                    for j in range(3):
                        _mug(p, M, x + 0.05, yc + (j - 1) * 0.1, z + 0.02, ("#f0f0ee", "#c8202a", "#141a28")[j])
                    x += 0.12
                elif kind == 1:   # boxed balls
                    p.r = 0.004
                    p.box((x, yc - 0.12, z + 0.02), (x + 0.24, yc + 0.12, z + 0.26), _paint("Hos_BallBox", "#f2f2ee", 0.5))
                    x += 0.26
                elif kind == 2:   # mascot toys
                    for j in range(2):
                        p.r = 0.0
                        p.sphere((x + 0.07, yc + (j - 0.5) * 0.15, z + 0.09), 0.07, shirt, 10, 8)
                        p.sphere((x + 0.07, yc + (j - 0.5) * 0.15, z + 0.2), 0.055, _paint("Hos_Plush", "#b8865a", 0.9), 10, 8)
                    x += 0.15
                else:             # kit bags
                    p.r = 0.03
                    p.box((x, yc - 0.12, z + 0.02), (x + 0.3, yc + 0.12, z + 0.22), shirt)
                    x += 0.32
    # the ball basket on the end
    p.r = 0.0
    for k in range(9):
        a = 2 * math.pi * k / 9
        p.cyl((0.85 + 0.2 * math.cos(a), 0.2 * math.sin(a), 0.0), (0.85 + 0.2 * math.cos(a), 0.2 * math.sin(a), 0.7), 0.004, M["chrome"], 4)
    for z in (0.05, 0.4, 0.7):
        _tube(p, [(0.85 + 0.2 * math.cos(t), 0.2 * math.sin(t), z) for t in np.linspace(0, 2 * math.pi, 19)], 0.004, M["chrome"], 4, closed=True)
    for j, (dx, dy, dz) in enumerate(((0, 0, 0.5), (-0.08, 0.06, 0.62), (0.08, -0.05, 0.64), (0.02, 0.09, 0.78), (-0.06, -0.08, 0.8))):
        p.sphere((0.85 + dx, dy, dz), 0.11, M["white"] if j % 2 else shirt, 14, 10)
    return p.obj("gondola", col)


def _ellipse(cx, cy, z, a, b, n=16):
    return [(cx + a * math.cos(2 * math.pi * k / n), cy + b * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def shop_mannequin(col, M):
    """A full-figure display mannequin (1.85 m, a matt white sculpted form with an abstract head)
    in the home kit: a fitted short-sleeved shirt (tinted: the club colour) with a white collar,
    white shorts, socks in the club colour turned down below the knee and black boots, arms
    relaxed at the sides; on a chrome rod into one calf from a round glass base. Front +y."""
    shirt = _shirt_mat()
    form = _paint("Hos_Form", "#ecebe6", 0.3)
    shorts = _paint("Hos_Shorts", "#f4f4f0", 0.7)
    boot = _plastic("Hos_Boot", "#141516", 0.35)
    p = PM(0.004)
    p.r = 0.0
    p.cyl((0, 0, 0.0), (0, 0, 0.012), 0.25, clear_glass(), 28)
    p.cyl((0.095, -0.05, 0.012), (0.095, -0.035, 0.3), 0.007, M["chrome"], 8)
    for s in (-1, 1):
        x = s * 0.095
        # boot: a sole and a lofted upper round the foot, toe to +y
        p.r = 0.004
        p.box((x - 0.045, -0.06, 0.0), (x + 0.045, 0.17, 0.018), boot)
        p.r = 0.0
        _loft(p, [_ellipse(x, 0.055, 0.018, 0.046, 0.112), _ellipse(x, 0.045, 0.06, 0.044, 0.095), _ellipse(x, 0.0, 0.1, 0.037, 0.045)], boot)
        # sock to below the knee (a turned-down top), then the form's knee and thigh
        _loft(p, [_ellipse(x, 0.0, 0.1, 0.036, 0.042), _ellipse(x, 0.0, 0.2, 0.041, 0.046), _ellipse(x, -0.004, 0.36, 0.056, 0.059),
                  _ellipse(x, 0.0, 0.44, 0.049, 0.053), _ellipse(x, 0.0, 0.47, 0.053, 0.057)], shirt, cap0=False)
        _loft(p, [_ellipse(x, 0.0, 0.47, 0.046, 0.05), _ellipse(x, 0.004, 0.52, 0.05, 0.052), _ellipse(x, 0.0, 0.7, 0.07, 0.072),
                  _ellipse(s * 0.1, 0.0, 0.8, 0.08, 0.08)], form, cap0=False)
        # short sleeve, then the arm down to the hand by the thigh
        # (the sleeve angles down from inside the shoulder; the arm runs through its hem)
        _loft(p, [_ellipse(s * 0.252, 0.0, 1.24, 0.06, 0.057), _ellipse(s * 0.222, 0.0, 1.32, 0.064, 0.06), _ellipse(s * 0.165, 0.0, 1.41, 0.062, 0.06)], shirt)
        _loft(p, [_ellipse(s * 0.265, 0.06, 0.86, 0.026, 0.028), _ellipse(s * 0.27, 0.04, 0.98, 0.034, 0.035), _ellipse(s * 0.27, 0.02, 1.08, 0.036, 0.037),
                  _ellipse(s * 0.245, 0.0, 1.27, 0.042, 0.042)], form, cap1=False)
        p.sphere((s * 0.265, 0.07, 0.8), 0.045, form, 12, 8, scale=(0.6, 0.9, 1.3))
    # shorts round the hips, the shirt over them
    # (a leg each, flaring a little at the hem, under the seat and waistband)
    for s in (-1, 1):
        _loft(p, [_ellipse(s * 0.1, 0.0, 0.72, 0.098, 0.096), _ellipse(s * 0.097, 0.0, 0.8, 0.094, 0.094), _ellipse(s * 0.09, 0.0, 0.88, 0.098, 0.11)],
              shorts, cap0=False)
    _loft(p, [_ellipse(0.0, 0.0, 0.84, 0.186, 0.118), _ellipse(0.0, 0.0, 0.92, 0.185, 0.12), _ellipse(0.0, 0.0, 1.0, 0.17, 0.115)], shorts)
    _loft(p, [_ellipse(0.0, 0.0, 0.96, 0.18, 0.125), _ellipse(0.0, 0.0, 1.05, 0.168, 0.113), _ellipse(0.0, 0.005, 1.18, 0.177, 0.121),
              _ellipse(0.0, 0.012, 1.3, 0.192, 0.126), _ellipse(0.0, 0.0, 1.4, 0.207, 0.112), _ellipse(0.0, 0.0, 1.46, 0.16, 0.09),
              _ellipse(0.0, 0.0, 1.5, 0.072, 0.062)], shirt)
    _tube(p, [(0.074 * math.cos(t), 0.064 * math.sin(t), 1.5) for t in np.linspace(0, 2 * math.pi, 17)], 0.008, shorts, 6, closed=True)
    # neck and an abstract head
    p.cyl((0, 0, 1.46), (0, 0.004, 1.6), 0.052, form, 14)
    p.sphere((0.0, 0.01, 1.7), 0.1, form, 18, 12, scale=(0.82, 0.95, 1.18))
    return p.obj("shop_mannequin", col)


def shop_counter(col, M):
    """The store's till counter (2.0 x 0.7 m, white solid surface with a club-coloured fascia):
    two POS screens and a customer display, card terminals, a bag rack of branded carriers under,
    a spinner of keyrings and badges at the customer end. Front +y (the customers' side)."""
    cv = Canvas("Hos_Till", 1024, 512, col="#202020", rough=0.4)
    pos = cv.alloc(0.3, 0.22, 900)
    pos.fill(col="#f4f6f9", emit="#f4f6f9")
    pos.rect(0, 0.19, 0.3, 0.22, col="#1d4f91", emit="#1d4f91")
    for i in range(6):
        pos.text(("Home shirt M", "Printing  9", "Scarf", "Mug", "Ball size 5", "TOTAL")[i], 0.01, 0.165 - i * 0.026, 0.014, col="#1a1a1a", emit="#1a1a1a",
                 weight="bold" if i == 5 else "regular")
        pos.text(("75.00", "15.00", "18.00", "10.00", "25.00", "143.00")[i], 0.29, 0.165 - i * 0.026, 0.014, align="r", col="#1a1a1a", emit="#1a1a1a")
    smat = cv.material("Hos_TillScreen", emission=1.0)
    surf = _paint("Hos_Solid", "#f3f3f1", 0.25)
    band = _plastic("TINT_club_counter", "#c8202a", 0.4)
    p = PM(0.006)
    p.r = 0.015
    p.box((-1.0, -0.35, 0.0), (1.0, 0.35, 0.95), surf)
    p.box((-1.03, -0.38, 0.95), (1.03, 0.38, 0.98), surf)
    p.r = 0.0
    p.face([(1.0, 0.351, 0.1), (-1.0, 0.351, 0.1), (-1.0, 0.351, 0.85), (1.0, 0.351, 0.85)], band)
    p.box((-0.98, 0.2, 0.0), (0.98, 0.35, 0.08), M["black"])
    for x in (-0.5, 0.4):
        p.r = 0.006
        p.cyl((x, -0.1, 0.98), (x, -0.1, 1.08), 0.02, M["black"], 10)
        p.obox((x, -0.08, 1.16), (0.32, 0.03, 0.24), M["black"], pitch=math.radians(-15))
        decal(p, (x, -0.096, 1.16), (0, -math.cos(math.radians(15)), math.sin(math.radians(15))), (0, math.sin(math.radians(15)), math.cos(math.radians(15))), 0.3, 0.22, smat, pos, lift=0.002)
        p.obox((x + 0.25, 0.18, 1.0), (0.08, 0.16, 0.04), M["black"], pitch=math.radians(20))
        p.box((x - 0.25, -0.3, 0.98), (x - 0.1, -0.1, 1.05), _plastic("Hos_Drawer", "#2a2c30", 0.5))
    # bags under, the spinner at the end
    for k in range(6):
        p.r = 0.002
        p.box((-0.9 + k * 0.02, -0.33, 0.1), (-0.89 + k * 0.02, -0.1, 0.45), _paint("Hos_Bag", "#141a28", 0.6))
    p.r = 0.0
    p.cyl((1.25, 0.2, 0.0), (1.25, 0.2, 1.5), 0.012, M["chrome"], 8)
    p.cyl((1.25, 0.2, 0.0), (1.25, 0.2, 0.02), 0.18, M["black"], 16)
    for z in (0.9, 1.1, 1.3):
        for k in range(8):
            a = k * math.pi / 4
            p.box((1.25 + 0.12 * math.cos(a) - 0.02, 0.2 + 0.12 * math.sin(a) - 0.002, z - 0.08), (1.25 + 0.12 * math.cos(a) + 0.02, 0.2 + 0.12 * math.sin(a) + 0.002, z),
                  _paint("Hos_Keyring" + str(k % 3), ("#c8202a", "#141a28", "#e3b92a")[k % 3], 0.4))
    return p.obj("shop_counter", col)


def print_station(col, M):
    """The shirt-printing station (2.0 m, the back against the wall at y = 0): a pegboard of vinyl
    letters and numbers in the league font, a workbench with a clamshell heat press closing on a
    shirt, a roll of transfer sheet, a price list. Front +y."""
    cv = Canvas("Hos_Print", 1024, 1024, col="#ffffff", rough=0.5)
    b = cv.alloc(2.0, 1.0, 400)
    b.fill(col="#e8e6e0")
    b.text("NAME & NUMBER PRINTING", 1.0, 0.9, 0.06, weight="black", align="c", col="#141a28")
    for i in range(10):
        for j in range(3):
            x, y = 0.08 + i * 0.19, 0.12 + j * 0.24
            b.rect(x, y, x + 0.15, y + 0.2, col="#141a28" if j < 2 else "#ffffff")
            b.text(str(i) if j < 2 else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[(i * 3 + j) % 26], x + 0.075, y + 0.03, 0.15, weight="black", align="c",
                   col="#ffffff" if j < 2 else "#141a28", cond=0.8)
    b.text("Name  £12     Number  £9     Badges  £5", 1.0, 0.04, 0.045, weight="bold", align="c", col="#c8202a")
    mat = cv.material("Hos_PrintBoard")
    shirt = _shirt_mat()
    steel = _paint("Hos_Steel", "#3a3d42", 0.4)
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-1.0, 0.0, 1.05), (1.0, 0.02, 2.05), mat)
    project(p, i0, b, (1.0, 0.02, 1.05), (-1, 0, 0), (0, 0, 1))
    p.r = 0.006
    p.box((-1.0, 0.0, 0.88), (1.0, 0.7, 0.92), _paint("Hos_Bench", "#d8d4cc", 0.5))
    for x in (-0.95, 0.95):
        for y in (0.05, 0.65):
            p.box((x - 0.02, y - 0.02, 0.0), (x + 0.02, y + 0.02, 0.88), steel)
    p.box((-0.95, 0.05, 0.2), (0.95, 0.65, 0.22), steel)
    # heat press: base platen with a shirt laid on it, the head on its arm, the controller
    p.box((-0.6, 0.15, 0.92), (-0.1, 0.6, 1.0), steel)
    p.r = 0.0
    p.box((-0.55, 0.18, 1.0), (-0.15, 0.57, 1.008), shirt)
    p.r = 0.006
    p.box((-0.62, 0.1, 1.12), (-0.08, 0.62, 1.18), _paint("Hos_Platen", "#c8ccd0", 0.3))
    p.box((-0.38, 0.06, 1.0), (-0.32, 0.12, 1.4), steel)
    p.box((-0.38, 0.06, 1.35), (-0.32, 0.4, 1.4), steel)
    p.box((0.0, 0.3, 0.92), (0.2, 0.5, 1.05), M["black"])
    p.box((0.03, 0.5, 0.97), (0.17, 0.505, 1.03), _plastic("Hos_PressLed", "#e04030", 0.2))
    p.r = 0.0
    p.cyl((0.4, 0.2, 0.98), (0.9, 0.2, 0.98), 0.05, _white("Hos_Transfer"), 16)
    return p.obj("print_station", col)


def stock_racking(col, M):
    """Steel stock racking (1.8 x 0.6 x 2.4 m, four beam levels): cardboard cartons labelled by
    product and size, shrink-wrapped bundles, a stepladder hooked on the end. Front +y."""
    cv = Canvas("Hos_Cartons", 512, 512, col="#b08a5a", rough=0.8)
    labs = []
    for k in range(6):
        r = cv.alloc(0.12, 0.08, 1500)
        r.fill(col="#ffffff")
        r.text(("SHIRT H  M", "SHIRT A  L", "SCARF", "HOODIE XL", "BALL 5", "MUG")[k], 0.06, 0.045, 0.014, weight="black", align="c", col="#1a1a1a")
        r.rect(0.01, 0.01, 0.11, 0.03, col="#1a1a1a")
        labs.append(r)
    lmat = cv.material("Hos_CartonLabels")
    kraft = _paint("Hos_Kraft", "#b58c5c", 0.85)
    orange = _paint("Hos_RackBeam", "#e5641c", 0.4)
    blue = _paint("Hos_RackFrame", "#2457a6", 0.4)
    p = PM(0.004)
    for x in (-0.9, 0.9):
        for y in (-0.3, 0.3):
            p.r = 0.002
            p.box((x - 0.04, y - 0.025, 0.0), (x + 0.04, y + 0.025, 2.4), blue)
        p.box((x - 0.02, -0.3, 0.3), (x + 0.02, 0.3, 0.33), blue)
        p.box((x - 0.02, -0.3, 1.6), (x + 0.02, 0.3, 1.63), blue)
    rng = np.random.default_rng(41)
    for z in (0.12, 0.72, 1.32, 1.92):
        for y in (-0.3, 0.3):
            p.r = 0.002
            p.box((-0.86, y - 0.025, z), (0.86, y + 0.025, z + 0.08), orange)
        p.box((-0.86, -0.28, z + 0.08), (0.86, 0.28, z + 0.095), _paint("Hos_Deck", "#8a8f94", 0.5))
        x = -0.84
        k = 0
        while x < 0.7:
            w = float(rng.choice([0.3, 0.4, 0.5]))
            h = float(rng.uniform(0.25, 0.42))
            p.r = 0.004
            p.box((x, -0.25, z + 0.095), (x + w - 0.02, 0.25, z + 0.095 + h), kraft)
            p.r = 0.0
            decal(p, (x + w / 2 - 0.01, 0.251, z + 0.095 + h * 0.6), (0, 1, 0), (0, 0, 1), 0.12, 0.08, lmat, labs[int(rng.integers(0, 6))], lift=0.001)
            x += w
            k += 1
    p.r = 0.0
    for y in (-0.15, 0.15):
        p.cyl((0.98, y, 0.0), (0.98, y, 1.6), 0.015, M["alu"], 8)
    for z in np.arange(0.25, 1.6, 0.27):
        p.box((0.97, -0.16, z), (0.99, 0.16, z + 0.03), M["alu"])
    return p.obj("stock_racking", col)


# -- tickets ----------------------------------------------------------------------------------------

def ticket_desk(col, M):
    """A two-window ticket and membership counter (3.0 m, the customer top 1.05 m up): toughened
    glass screens with speaking slots and a deal tray under each, staff monitors, ticket printers
    and card terminals, two task chairs. Front +y (the customers' side)."""
    cv = Canvas("Hos_Tickets", 1024, 512, col="#202020", rough=0.4)
    regs = []
    for k in range(2):
        r = cv.alloc(0.48, 0.27, 700)
        _display(r, "sales" if k else "sheet", 90 + k)
        regs.append(r)
    sign = cv.alloc(0.6, 0.15, 800)
    sign.fill(col="#141a28", emit="#141a28")
    sign.text("TICKETS  &  MEMBERSHIP", 0.3, 0.05, 0.05, weight="black", align="c", col="#ffffff", emit="#ffffff")
    smat = cv.material("Hos_TicketScreens", emission=1.0)
    surf = _paint("Hos_Solid", "#f3f3f1", 0.25)
    band = _plastic("TINT_club_counter", "#c8202a", 0.4)
    p = PM(0.006)
    p.r = 0.015
    p.box((-1.5, 0.0, 0.0), (1.5, 0.35, 1.02), surf)
    p.box((-1.52, 0.0, 1.02), (1.52, 0.42, 1.05), surf)
    p.box((-1.5, -0.6, 0.72), (1.5, 0.0, 0.75), surf)
    p.r = 0.0
    p.face([(1.5, 0.351, 0.1), (-1.5, 0.351, 0.1), (-1.5, 0.351, 0.95), (1.5, 0.351, 0.95)], band)
    p.box((-1.52, 0.03, 1.05), (1.52, 0.05, 2.1), clear_glass())
    p.r = 0.004
    p.box((-1.52, 0.0, 2.1), (1.52, 0.1, 2.25), M["black"])
    p.r = 0.0
    decal(p, (0.0, 0.1, 2.175), (0, 1, 0), (0, 0, 1), 0.6, 0.15, smat, sign, lift=0.001)
    for j, x in enumerate((-0.75, 0.75)):
        p.r = 0.003
        p.box((x - 0.15, 0.0, 1.05), (x + 0.15, 0.42, 1.07), M["chrome"])
        p.box((x - 0.12, 0.025, 1.3), (x + 0.12, 0.055, 1.36), M["grey"])
        _monitor(p, (x - 0.25, -0.35, 0.75), math.pi, 0.48, 0.27, smat, regs[j], M)
        p.r = 0.006
        p.box((x + 0.12, -0.45, 0.75), (x + 0.32, -0.25, 0.87), M["black"])
        p.box((x + 0.15, -0.25, 0.83), (x + 0.29, -0.24, 0.84), _white("Hos_Ticket"))
        p.obox((x + 0.05, -0.15, 0.77), (0.08, 0.16, 0.04), M["black"], pitch=math.radians(15))
        _place(p, _task_chair(M), x - 0.1, -0.95, 0.0)
    return p.obj("ticket_desk", col)


# -- the café ---------------------------------------------------------------------------------------

def _bakes(p, M):
    """The chilled display's food: on the top shelf a layered sponge on a glass cake stand,
    cupcakes piped in three flavours in pleated cases and custard tarts; on the lower one sandwich
    wedges in clear packs, glazed croissants on a plate and a tray of brownies cut into squares."""
    sponge = _paint("Hos_Sponge", "#d9b072", 0.85)
    cream = _paint("Hos_Cream", "#f5efe2", 0.6)
    choc = _paint("Hos_Choc", "#4a2a1a", 0.35)
    pastry = _paint("Hos_Pastry", "#c98a3c", 0.35)
    bread = _paint("Hos_Bread", "#e6cf9e", 0.9)
    crust = _paint("Hos_Crust", "#b0773a", 0.8)
    berry = _plastic("Hos_Berry", "#a8142a", 0.25)
    case = _paint("Hos_CakeCase", "#f2f2ee", 0.7)
    zt, zb = 1.19, 1.01
    # top shelf: the layer cake on its stand, cupcakes, tarts
    cx = 0.38
    p.r = 0.0
    p.cyl((cx, 0.0, zt), (cx, 0.0, zt + 0.04), 0.03, clear_glass(), 12)
    p.cyl((cx, 0.0, zt + 0.04), (cx, 0.0, zt + 0.048), 0.12, clear_glass(), 24)
    z = zt + 0.048
    for h, m in ((0.032, sponge), (0.008, cream), (0.032, sponge), (0.01, choc)):
        p.lathe([(0.0, 0.0), (0.1, 0.0), (0.1, h), (0.0, h)], m, c=(cx, 0.0, z), sides=28)
        z += h
    for k in range(8):
        a = k * math.pi / 4 + 0.4
        p.sphere((cx + 0.075 * math.cos(a), 0.075 * math.sin(a), z + 0.008), 0.011, berry, 8, 6)
    for k, (x, y) in enumerate(((0.6, -0.12), (0.71, -0.12), (0.82, -0.12), (0.6, 0.0), (0.71, 0.0), (0.82, 0.0), (0.66, 0.11), (0.77, 0.11))):
        top = (_paint("Hos_FrostPink", "#f0b8c8", 0.5), cream, choc)[k % 3]
        p.lathe([(0.0, 0.0), (0.026, 0.0), (0.032, 0.038), (0.0, 0.038)], case, c=(x, y, zt), sides=16, flutes=16, depth=0.0015)
        p.lathe([(0.0, 0.034), (0.035, 0.038), (0.03, 0.055), (0.02, 0.068), (0.008, 0.08), (0.0, 0.082)], top, c=(x, y, zt), sides=16, flutes=6, depth=0.003)
    for k in range(3):
        x = 1.0
        p.r = 0.0
        p.cyl((x, -0.12 + k * 0.12, zt), (x, -0.12 + k * 0.12, zt + 0.012), 0.045, cream, 16)
        p.sphere((x, -0.12 + k * 0.12, zt + 0.015), 0.04, pastry, 12, 6, scale=(1.0, 1.0, 0.5), z_min=0.0)
    # lower shelf: sandwich wedges in clear packs
    for k in range(4):
        x, y = 0.3 + (k % 2) * 0.13, -0.1 + (k // 2) * 0.17
        tri = [(-0.06, 0.0), (0.06, 0.0), (0.0, 0.085)]
        for j, (y0, y1, m) in enumerate(((-0.03, -0.012, bread), (-0.012, -0.004, _paint("Hos_Filling" + str(k % 2), ("#c84a3a", "#7aa04a")[k % 2], 0.6)), (-0.004, 0.012, bread))):
            p.extrude([(x + a, zb + 0.004 + b) for a, b in tri], y + y0, y + y1, m)
        p.r = 0.003
        p.box((x - 0.065, y - 0.034, zb), (x + 0.065, y + 0.016, zb + 0.004), clear_glass())
        p.r = 0.0
        p.face([(x - 0.065, y - 0.034, zb + 0.004), (x + 0.065, y - 0.034, zb + 0.004), (x, y - 0.034, zb + 0.094)], clear_glass())
    # glazed croissants on a plate
    p.cyl((0.68, 0.0, zb), (0.68, 0.0, zb + 0.01), 0.12, cream, 24)
    for k, (dx, dy, yaw) in enumerate(((-0.05, -0.04, 0.3), (0.05, -0.03, -0.4), (0.0, 0.06, 1.4))):
        for j in range(5):
            t = (j - 2) / 2.0
            r = 0.022 * (1.0 - 0.45 * abs(t))
            px = 0.68 + dx + math.cos(yaw) * t * 0.05 - math.sin(yaw) * (0.012 * t * t)
            py = dy + math.sin(yaw) * t * 0.05 + math.cos(yaw) * (0.012 * t * t)
            p.sphere((px, py, zb + 0.01 + r * 0.8), r, pastry, 10, 6, scale=(1.3, 1.0, 0.85))
    # brownies: a tray cut into squares
    p.r = 0.002
    p.box((0.86, -0.13, zb), (1.1, 0.13, zb + 0.008), M["chrome"])
    for i in range(4):
        for j in range(4):
            x0, y0 = 0.87 + i * 0.058, -0.12 + j * 0.06
            p.box((x0, y0, zb + 0.008), (x0 + 0.054, y0 + 0.056, zb + 0.038), choc)
            p.box((x0 + 0.004, y0 + 0.004, zb + 0.038), (x0 + 0.05, y0 + 0.052, zb + 0.041), crust)


def cafe_counter(col, M):
    """A café counter (2.4 x 0.75 m, slatted oak front, a quartz top): a curved-glass chilled cake
    display with cakes and sandwiches, a two-group espresso machine with its grinder and knock box,
    the till, stacked cups, a tip jar. Front +y (the customers' side)."""
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    quartz = _paint("Hos_Quartz", "#e8e6e2", 0.15)
    steel = _paint("Hos_Stainless", "#b8bcc0", 0.3)
    p = PM(0.004)
    p.r = 0.006
    p.box((-1.2, -0.37, 0.0), (1.2, 0.37, 0.9), _paint("Hos_CounterBody", "#2a2c30", 0.6))
    for k in range(40):
        x = -1.19 + k * 0.06
        p.r = 0.002
        p.box((x, 0.37, 0.05), (x + 0.045, 0.39, 0.9), oak)
    p.r = 0.006
    p.box((-1.22, -0.39, 0.9), (1.22, 0.41, 0.94), quartz)
    # the chilled display at the customer end
    p.r = 0.0
    p.box((0.2, -0.3, 0.94), (1.15, 0.36, 1.0), steel)
    for z in (1.0, 1.18):
        p.box((0.22, -0.28, z), (1.13, 0.3, z + 0.01), clear_glass())
    p.face([(1.15, 0.36, 1.0), (0.2, 0.36, 1.0), (0.2, 0.18, 1.38), (1.15, 0.18, 1.38)], clear_glass())
    p.box((0.2, -0.3, 1.36), (1.15, 0.18, 1.38), steel)
    _bakes(p, M)
    # espresso machine, grinder, knock box, cups, till
    p.r = 0.02
    p.box((-1.1, -0.35, 0.94), (-0.35, 0.05, 1.38), steel)
    p.box((-1.1, -0.35, 1.38), (-0.35, 0.05, 1.42), M["black"])
    for x in (-0.9, -0.55):
        p.r = 0.0
        p.cyl((x, 0.05, 1.18), (x, 0.13, 1.18), 0.035, M["chrome"], 12)
        p.cyl((x, 0.13, 1.18), (x, 0.13, 1.12), 0.03, M["chrome"], 12)
        p.cyl((x, 0.13, 1.14), (x + 0.0, 0.32, 1.12), 0.008, M["black"], 6)
        p.cyl((x, 0.12, 0.96), (x, 0.12, 1.02), 0.035, M["white"], 12)
    p.r = 0.004
    p.box((-1.1, 0.0, 0.94), (-0.35, 0.2, 0.96), M["chrome"])
    p.r = 0.0
    p.cyl((-0.25, -0.2, 0.94), (-0.25, -0.2, 1.2), 0.06, M["black"], 12)
    p.cyl((-0.25, -0.2, 1.2), (-0.25, -0.2, 1.38), 0.08, clear_glass(), 12, r2=0.1)
    p.cyl((-0.15, 0.1, 0.94), (-0.15, 0.1, 1.04), 0.07, M["black"], 12)
    for k in range(5):
        p.cyl((-1.12 + k * 0.0, -0.32, 1.42 + k * 0.07), (-1.12, -0.32, 1.48 + k * 0.07), 0.04, M["white"], 12, r2=0.045)
    p.r = 0.006
    p.obox((0.0, 0.0, 1.05), (0.3, 0.03, 0.2), M["black"], pitch=math.radians(-15))
    p.r = 0.0
    p.cyl((0.12, 0.3, 0.94), (0.12, 0.3, 1.08), 0.05, clear_glass(), 12)
    return p.obj("cafe_counter", col)


def cafe_backbar(col, M):
    """The café back bar (2.4 m, the back against the wall at y = 0): base units with a glass-door
    drinks fridge of bottles, a glass washer, a milk fridge, a worktop with a bean hopper, jars of
    pastries and syrups, cups and saucers stacked by size. Front +y."""
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.004)
    p.r = 0.004
    p.box((-1.2, 0.0, 0.0), (1.2, 0.6, 0.88), _paint("Hos_CounterBody", "#2a2c30", 0.6))
    p.box((-1.22, 0.0, 0.88), (1.22, 0.62, 0.92), _paint("Hos_Quartz", "#e8e6e2", 0.15))
    # drinks fridge: lit, glass door, bottles on three shelves
    p.r = 0.0
    p.box((-1.15, 0.05, 0.05), (-0.5, 0.58, 0.84), _plastic("Hos_FridgeLit", "#f6f8fa", 0.3))
    for z in (0.08, 0.33, 0.58):
        for k in range(8):
            c = ("#c8202a", "#2a8a3a", "#f0a020", "#2457a6")[k % 4]
            _bottle(p, -1.1 + k * 0.075, 0.4, z, 0.2, _plastic("Hos_Pop" + c[1:], c, 0.2), M["white"])
    p.box((-1.15, 0.6, 0.05), (-0.5, 0.61, 0.84), clear_glass())
    p.box((-0.55, 0.61, 0.3), (-0.53, 0.63, 0.6), M["chrome"])
    for x in (-0.45, 0.2):
        p.r = 0.004
        p.box((x, 0.6, 0.05), (x + 0.6, 0.62, 0.84), _paint("Hos_Stainless", "#b8bcc0", 0.3))
        p.box((x + 0.25, 0.62, 0.75), (x + 0.35, 0.64, 0.77), M["chrome"])
    p.r = 0.0
    for k in range(4):
        _glass(p, 0.4 + k * 0.1, 0.3, 0.92, 0.25, 0.06)
    for k in range(3):
        for j in range(4):
            p.cyl((-0.3 + k * 0.12, 0.25, 0.92 + j * 0.06), (-0.3 + k * 0.12, 0.25, 0.97 + j * 0.06), 0.045, M["white"], 12, r2=0.05)
    for k in range(6):
        c = ("#6a3a22", "#e8c060", "#c8402a")[k % 3]
        _bottle(p, 0.9 + (k % 3) * 0.08, 0.2 + (k // 3) * 0.12, 0.92, 0.3, _plastic("Hos_Syrup" + c[1:], c, 0.15), M["black"])
    return p.obj("cafe_backbar", col)


def menu_board(col, M):
    """A chalkboard menu over the back bar (2.2 x 0.9 m in an oak frame, 1.6 m up) with the coffee
    list and prices in chalk, and a floating oak shelf of cups and coffee bags under it. The back
    against the wall at y = 0; front +y."""
    cv = Canvas("Hos_Menu", 1024, 512, col="#1c1f1e", rough=0.9)
    b = cv.alloc(2.2, 0.9, 400)
    b.fill(col="#1e2220")
    b.field(lambda X, Y: noise(X.shape, 20, 61, 3) * 0.25, col="#5a605c")
    b.text("COFFEE", 0.1, 0.76, 0.08, weight="bold", col="#f2f2ec")
    b.text("FOOD", 1.2, 0.76, 0.08, weight="bold", col="#f2f2ec")
    for i, (a, v) in enumerate((("Espresso", "2.40"), ("Americano", "2.90"), ("Flat white", "3.30"), ("Latte", "3.40"), ("Cappuccino", "3.40"), ("Hot chocolate", "3.20"))):
        b.text(a, 0.1, 0.64 - i * 0.1, 0.05, col="#e8e8e2")
        b.text(v, 1.0, 0.64 - i * 0.1, 0.05, align="r", col="#e8e8e2")
    for i, (a, v) in enumerate((("Bacon roll", "4.50"), ("Toastie", "5.20"), ("Soup of the day", "4.80"), ("Cake slice", "3.50"), ("Pie and peas", "6.50"))):
        b.text(a, 1.2, 0.64 - i * 0.1, 0.05, col="#e8e8e2")
        b.text(v, 2.1, 0.64 - i * 0.1, 0.05, align="r", col="#f0d070")
    mat = cv.material("Hos_MenuBoard", rough=0.9)
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-1.1, 0.01, 1.6), (1.1, 0.03, 2.5), mat)
    project(p, i0, b, (1.1, 0.03, 1.6), (-1, 0, 0), (0, 0, 1))
    p.r = 0.004
    for (lo, hi) in (((-1.14, 0.0, 1.56), (1.14, 0.05, 1.6)), ((-1.14, 0.0, 2.5), (1.14, 0.05, 2.54)), ((-1.14, 0.0, 1.56), (-1.1, 0.05, 2.54)), ((1.1, 0.0, 1.56), (1.14, 0.05, 2.54))):
        p.box(lo, hi, oak)
    p.box((-1.1, 0.0, 1.3), (1.1, 0.25, 1.34), oak)
    for k in range(8):
        p.r = 0.0
        if k % 3 == 2:
            p.r = 0.006
            p.box((-1.0 + k * 0.26, 0.06, 1.34), (-0.88 + k * 0.26, 0.14, 1.52), _paint("Hos_CoffeeBag", "#5a3a28", 0.7))
        else:
            _mug(p, M, -0.95 + k * 0.26, 0.12, 1.34, "#f0f0ee")
    return p.obj("menu_board", col)


def cafe_table(col, M):
    """A round bistro table (0.7 m, a white marble-effect top on a black cast pedestal) with two
    bentwood chairs, a flat white and a sugar pot. Front +y."""
    wood = _paint("Hos_Bentwood", "#4a3326", 0.5)
    p = PM(0.004)
    p.r = 0.0
    p.cyl((0, 0, 0.0), (0, 0, 0.03), 0.25, M["black"], 20)
    p.cyl((0, 0, 0.03), (0, 0, 0.72), 0.03, M["black"], 12)
    p.cyl((0, 0, 0.72), (0, 0, 0.75), 0.35, _paint("Hos_Marble", "#f0efea", 0.2), 28)
    _mug(p, M, 0.1, -0.08, 0.75)
    p.cyl((-0.1, 0.05, 0.75), (-0.1, 0.05, 0.82), 0.03, M["white"], 10)
    for s in (-1, 1):
        _place(p, _bistro_chair(M, wood), 0.0, s * 0.55, math.pi if s > 0 else 0.0)
    return p.obj("cafe_table", col)


def cafe_table4(col, M):
    """A square café table for four (0.8 m, an oak top on a black four-star base) with four
    bentwood chairs, two cups, a plate with a pastry, a table number. Front +y."""
    wood = _paint("Hos_Bentwood", "#4a3326", 0.5)
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.004)
    p.r = 0.0
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        _tube(p, [(0, 0, 0.05), (0.3 * math.cos(a), 0.3 * math.sin(a), 0.02)], 0.02, M["black"], 6)
    p.cyl((0, 0, 0.05), (0, 0, 0.72), 0.035, M["black"], 12)
    p.r = 0.006
    p.box((-0.4, -0.4, 0.72), (0.4, 0.4, 0.75), oak)
    _mug(p, M, 0.15, 0.12, 0.75)
    _mug(p, M, -0.15, -0.14, 0.75, "#2a2c30")
    _plate(p, M, -0.12, 0.15, 0.75, 0.09)
    p.r = 0.02
    p.box((-0.15, 0.11, 0.77), (-0.07, 0.19, 0.8), _paint("Hos_Pastry", "#d8a050", 0.7))
    p.r = 0.002
    p.box((0.1, -0.15, 0.75), (0.16, -0.14, 0.86), M["black"])
    for k in range(4):
        a = k * math.pi / 2
        _place(p, _bistro_chair(M, wood), 0.62 * math.sin(a), -0.62 * math.cos(a), a)
    return p.obj("cafe_table4", col)


# -- the bar ----------------------------------------------------------------------------------------

def bar_counter(col, M):
    """A pub-style bar counter (3.0 x 0.75 m, 1.1 m high): panelled walnut front with a brass foot
    rail, a dark top with a drip rail, two beer fonts of four taps with their badges, glasses
    upside down on drip mats, four leather bar stools in front. Front +y (the customers' side)."""
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    brass = _paint("Hos_Brass", "#b08a3a", 0.25)
    top = _paint("Hos_BarTop", "#1a1716", 0.2)
    leather = _plastic("Off_Leather", "#1d1e20", 0.45)
    p = PM(0.004)
    p.r = 0.006
    p.box((-1.5, -0.37, 0.0), (1.5, 0.3, 1.06), walnut)
    for k in range(6):
        x = -1.45 + k * 0.5
        p.r = 0.004
        p.box((x + 0.03, 0.3, 0.15), (x + 0.47, 0.32, 0.95), walnut)
    p.r = 0.008
    p.box((-1.53, -0.4, 1.06), (1.53, 0.42, 1.1), top)
    p.r = 0.0
    p.cyl((-1.45, 0.42, 0.18), (1.45, 0.42, 0.18), 0.025, brass, 12)
    for x in (-1.2, 0.0, 1.2):
        p.cyl((x, 0.32, 0.18), (x, 0.42, 0.18), 0.015, brass, 8)
    # beer fonts: a T-bar on a pillar, four taps with coloured badges, drip trays
    for xf in (-0.8, 0.6):
        p.cyl((xf, -0.15, 1.1), (xf, -0.15, 1.45), 0.03, M["chrome"], 12)
        p.cyl((xf - 0.3, -0.15, 1.45), (xf + 0.3, -0.15, 1.45), 0.035, M["chrome"], 12)
        for k in range(4):
            x = xf - 0.24 + k * 0.16
            p.cyl((x, -0.15, 1.45), (x, -0.05, 1.42), 0.012, M["chrome"], 8)
            c = ("#c8202a", "#e3b92a", "#141a28", "#2a8a3a")[k]
            p.r = 0.006
            p.box((x - 0.03, -0.18, 1.5), (x + 0.03, -0.12, 1.66), _paint("Hos_Tap" + c[1:], c, 0.4))
            p.r = 0.0
        p.r = 0.004
        p.box((xf - 0.32, -0.25, 1.1), (xf + 0.32, -0.02, 1.12), M["chrome"])
    for k in range(6):
        _glass(p, -1.3 + k * 0.1, -0.32, 1.1, 0.15, 0.04)
    # stools
    for k in range(4):
        x = -1.1 + k * 0.73
        p.r = 0.0
        p.cyl((x, 0.75, 0.0), (x, 0.75, 0.02), 0.2, M["chrome"], 16)
        p.cyl((x, 0.75, 0.02), (x, 0.75, 0.72), 0.025, M["chrome"], 10)
        _tube(p, [(x + 0.17 * math.cos(t), 0.75 + 0.17 * math.sin(t), 0.3) for t in np.linspace(0, 2 * math.pi, 17)], 0.008, M["chrome"], 6, closed=True)
        p.r = 0.04
        p.cyl((x, 0.75, 0.72), (x, 0.75, 0.8), 0.19, leather, 18)
    return p.obj("bar_counter", col)


def bar_back(col, M):
    """The back bar (3.0 m, the back against the wall at y = 0): glass-door bottle fridges in the
    base under a walnut top, an antiqued mirror behind three glass shelves of spirits and wine,
    optics on a rail, a glass washer, an ice well. Front +y."""
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    mirror = _mirror_glass()
    p = PM(0.004)
    p.r = 0.004
    p.box((-1.5, 0.0, 0.0), (1.5, 0.55, 0.88), _paint("Hos_CounterBody", "#2a2c30", 0.6))
    p.box((-1.52, 0.0, 0.88), (1.52, 0.58, 0.92), walnut)
    rng = np.random.default_rng(71)
    for j, x0 in enumerate((-1.45, -0.45, 0.55)):
        p.r = 0.0
        p.box((x0, 0.05, 0.05), (x0 + 0.9, 0.53, 0.84), _plastic("Hos_FridgeLit", "#f6f8fa", 0.3))
        for z in (0.08, 0.45):
            for k in range(10):
                c = str(rng.choice(["#2a6a2a", "#6a3a1a", "#d8a030", "#2a2a2a"]))
                _bottle(p, x0 + 0.05 + k * 0.085, 0.35, z, 0.27, _plastic("Hos_Bottle" + c[1:], c, 0.1), M["chrome"])
        p.box((x0, 0.55, 0.05), (x0 + 0.9, 0.56, 0.84), clear_glass())
    p.r = 0.0
    p.box((-1.5, 0.0, 0.92), (1.5, 0.02, 2.3), mirror)
    for z in (1.2, 1.55, 1.9):
        p.box((-1.4, 0.02, z), (1.4, 0.3, z + 0.012), clear_glass())
        x = -1.35
        while x < 1.3:
            c = str(rng.choice(["#d8a030", "#6a3a1a", "#e8e8e0", "#2a6a2a", "#7a1a2a", "#2a2a2a"]))
            h = float(rng.uniform(0.24, 0.32))
            _bottle(p, x, 0.15, z + 0.012, h, _plastic("Hos_Bottle" + c[1:], c, 0.1), M["chrome"] if rng.random() < 0.5 else M["black"])
            x += float(rng.uniform(0.08, 0.12))
    for k in range(6):
        x = -0.4 + k * 0.16
        p.r = 0.0
        p.cyl((x, 0.1, 1.0), (x, 0.1, 1.12), 0.01, M["chrome"], 6)
        _bottle(p, x, 0.1, 1.12, 0.28, _plastic("Hos_Optic" + str(k % 3), ("#d8a030", "#e8e8e0", "#6a3a1a")[k % 3], 0.1), M["chrome"])
    p.r = 0.004
    p.box((-1.5, 0.0, 2.3), (1.5, 0.08, 2.4), walnut)
    p.box((0.9, 0.05, 0.92), (1.4, 0.5, 0.95), M["chrome"])
    return p.obj("bar_back", col)


# -- dining and lounge -----------------------------------------------------------------------------

def dining_table(col, M):
    """A restaurant table for four (0.9 m square, a white cloth over an oak table) laid with place
    settings, a candle and a bud vase; four upholstered dining chairs. Front +y."""
    cv, reg = _cushion_canvas("Hos_DiningFabric", 0.5, 0.5, base="#5a6470", seed=91)
    fab = cv.tint_material("TINT_upholstery_dining", "#4a4f58", rough=0.85)
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    cloth = _paint("Hos_Cloth", "#f6f5f0", 0.85)
    p = PM(0.004)
    p.r = 0.0
    for x in (-0.4, 0.4):
        for y in (-0.4, 0.4):
            p.cyl((x, y, 0.0), (x, y, 0.72), 0.025, oak, 8)
    p.r = 0.01
    p.box((-0.48, -0.48, 0.5), (0.48, 0.48, 0.76), cloth)
    for k in range(4):
        a = k * math.pi / 2
        _place_setting(p, M, 0.27 * math.sin(a), -0.27 * math.cos(a), 0.76, a)
        _place(p, _dining_chair(M, fab, oak), 0.66 * math.sin(a), -0.66 * math.cos(a), a)
    p.r = 0.0
    p.cyl((0.05, 0.05, 0.76), (0.05, 0.05, 0.84), 0.03, M["white"], 12)
    p.lathe([(0.0, 0.0), (0.03, 0.0), (0.035, 0.08), (0.01, 0.14), (0.012, 0.18), (0.0, 0.18)], clear_glass(), c=(-0.06, -0.04, 0.76), sides=10)
    return p.obj("dining_table", col)


def banquet_table(col, M):
    """A round banqueting table for ten (1.8 m, a floor-length white cloth): ten chairs in white
    covers with sashes in the club colour, place settings, a tall centrepiece of white flowers and
    the table number on a stand. Front +y."""
    cloth = _paint("Hos_Cloth", "#f6f5f0", 0.85)
    cover = _paint("Hos_ChairCover", "#f2f1ec", 0.85)
    sash = _plastic("TINT_club_sash", "#c8202a", 0.6)
    p = PM(0.004)
    p.r = 0.0
    p.lathe([(0.0, 0.76), (0.9, 0.76), (0.92, 0.74), (0.96, 0.0), (0.0, 0.0)], cloth, sides=40)
    for k in range(10):
        a = 2 * math.pi * k / 10
        _place_setting(p, M, 0.68 * math.sin(a), -0.68 * math.cos(a), 0.76, a)
        c = PM(0.004)
        c.r = 0.03
        c.box((-0.22, -0.22, 0.0), (0.22, 0.24, 0.47), cover)
        c.box((-0.22, -0.28, 0.0), (0.22, -0.2, 0.95), cover)
        c.r = 0.004
        c.box((-0.225, -0.285, 0.62), (0.225, -0.195, 0.7), sash)
        _place(p, c, 1.18 * math.sin(a), -1.18 * math.cos(a), a)
    p.r = 0.0
    p.cyl((0, 0, 0.76), (0, 0, 0.78), 0.12, clear_glass(), 16)
    p.cyl((0, 0, 0.78), (0, 0, 1.25), 0.015, clear_glass(), 8)
    rng = np.random.default_rng(93)
    for k in range(14):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.0, 0.16)
        p.sphere((r * math.cos(a), r * math.sin(a), 1.32 + rng.uniform(-0.05, 0.06)), 0.05, _paint("Hos_Flower", "#f4f2ea", 0.7), 8, 6)
    for k in range(6):
        a = k * math.pi / 3
        _tube(p, [(0, 0, 1.2), (0.25 * math.cos(a), 0.25 * math.sin(a), 1.28)], 0.006, _paint("Hos_Leaf", "#3f6a2e", 0.6), 4)
    p.r = 0.002
    p.box((0.3, 0.25, 0.76), (0.31, 0.26, 0.98), M["chrome"])
    p.box((0.24, 0.25, 0.98), (0.37, 0.26, 1.08), M["white"])
    return p.obj("banquet_table", col)


def armchair(col, M):
    """A lounge tub armchair (0.8 x 0.8 m, a curved upholstered shell, tinted: the club colour by
    default, on four splayed walnut legs) with a seat cushion. Front +y."""
    cv, reg = _cushion_canvas("Hos_TubFabric", 0.8, 0.8, base="#5a6470", seed=95)
    fab = cv.tint_material("TINT_upholstery_tub", "#5a6470", rough=0.9)
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    p = PM(0.01)
    p.r = 0.0
    for x, y in ((-0.28, -0.28), (0.28, -0.28), (-0.28, 0.28), (0.28, 0.28)):
        p.cyl((x * 1.1, y * 1.1, 0.0), (x * 0.9, y * 0.9, 0.2), 0.016, walnut, 8)
    p.r = 0.04
    i0 = len(p.F)
    p.box((-0.36, -0.36, 0.2), (0.36, 0.36, 0.42), fab)
    p.r = 0.0
    shell = [(0.37 * math.cos(t), 0.37 * math.sin(t)) for t in np.linspace(math.pi * 1.1, math.pi * 1.9, 13)]
    for k in range(len(shell) - 1):
        (xa, ya), (xb, yb) = shell[k], shell[k + 1]
        p.r = 0.03
        p.obox(((xa + xb) / 2, (ya + yb) / 2, 0.58), (math.hypot(xb - xa, yb - ya) + 0.02, 0.1, 0.36), fab, yaw=math.atan2(yb - ya, xb - xa))
    p.r = 0.06
    p.box((-0.3, -0.24, 0.42), (0.3, 0.33, 0.52), fab)
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-0.4, -0.4, 0.0), (0.4, 0.4, 0.8))
    return p.obj("armchair", col)


# -- trophies and events ----------------------------------------------------------------------------

def _cup(p, x, y, z, h, metal, ears=True):
    """A trophy cup on its plinth: a turned body, two ear handles."""
    p.r = 0.0
    p.lathe([(0.0, 0.0), (h * 0.22, 0.0), (h * 0.22, h * 0.18), (h * 0.12, h * 0.2), (h * 0.05, h * 0.3), (h * 0.04, h * 0.45),
             (h * 0.12, h * 0.52), (h * 0.24, h * 0.75), (h * 0.27, h * 0.95), (h * 0.25, h), (0.0, h * 0.97)], metal, c=(x, y, z), sides=18)
    if ears:
        for s in (-1, 1):
            _tube(p, [(x + s * h * 0.2, y, z + h * 0.85), (x + s * h * 0.4, y, z + h * 0.8), (x + s * h * 0.38, y, z + h * 0.6), (x + s * h * 0.12, y, z + h * 0.55)],
                  h * 0.02, metal, 6, bend=h * 0.08)


def trophy_cabinet(col, M):
    """A lit trophy cabinet (1.2 x 0.45 x 2.0 m): a walnut base cupboard, a frameless glass case
    with LED-lit glass shelves of cups, a salver, plates on stands, medals on ribbons, pennants. Front +y."""
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    silver = _paint("Hos_Silver", "#d0d4d8", 0.15)
    gold = _paint("Hos_Gold", "#d4aa48", 0.2)
    led = _plastic("Hos_CabinetLed", "#fff6e6", 0.2)
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.6, -0.22, 0.0), (0.6, 0.22, 0.8), walnut)
    p.box((-0.6, -0.22, 1.98), (0.6, 0.22, 2.04), walnut)
    p.r = 0.0
    p.box((-0.6, -0.22, 0.8), (0.6, -0.2, 1.98), _paint("Hos_CabBack", "#141a28", 0.7))
    for z in (1.18, 1.58):
        p.box((-0.58, -0.2, z), (0.58, 0.2, z + 0.01), clear_glass())
    p.box((-0.58, -0.2, 1.97), (0.58, 0.18, 1.975), led)
    p.box((-0.6, 0.2, 0.8), (0.6, 0.22, 1.98), clear_glass())
    for x in (-0.6, 0.58):
        p.box((x, -0.2, 0.8), (x + 0.02, 0.2, 1.98), clear_glass())
    rng = np.random.default_rng(81)
    for z in (0.8, 1.19, 1.59):
        x = -0.48
        while x < 0.45:
            k = int(rng.integers(0, 3))
            m = gold if rng.random() < 0.4 else silver
            if k == 0:
                h = float(rng.uniform(0.22, 0.36))
                p.r = 0.004
                p.box((x - 0.06, -0.06, z), (x + 0.06, 0.06, z + 0.06), M["black"])
                _cup(p, x, 0.0, z + 0.06, h, m)
                x += 0.24
            elif k == 1:
                p.r = 0.0
                p.cyl((x, -0.05, z + 0.17), (x, -0.06, z + 0.17), 0.14, m, 24)
                p.cyl((x, -0.04, z), (x, 0.06, z + 0.12), 0.008, M["black"], 6)
                x += 0.3
            else:
                for j in range(3):
                    p.r = 0.0
                    p.box((x - 0.01 + j * 0.05, -0.18, z + 0.12), (x + 0.01 + j * 0.05, -0.17, z + 0.3), _plastic("TINT_club_ribbon", "#c8202a", 0.5))
                    p.cyl((x + j * 0.05, -0.17, z + 0.1), (x + j * 0.05, -0.16, z + 0.1), 0.03, m, 14)
                x += 0.18
    return p.obj("trophy_cabinet", col)


def trophy_plinth(col, M):
    """The centrepiece of the trophy room: the club's biggest cup (0.6 m, silver) in a glass case
    on a walnut plinth (0.6 x 0.6 x 1.0 m) with a brass plaque and an uplight. Front +y."""
    cv = Canvas("Hos_Plaque", 256, 256, col="#b08a3a", rough=0.3, metal=0.9)
    r = cv.alloc(0.3, 0.12, 800)
    r.fill(col="#b8923e", metal=0.9, rough=0.3)
    r.text("CHAMPIONS", 0.15, 0.07, 0.03, weight="black", align="c", col="#2a2010")
    r.text("1958  1984  2009", 0.15, 0.025, 0.02, weight="bold", align="c", col="#2a2010")
    pm = cv.material("Hos_PlaqueMat")
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    silver = _paint("Hos_Silver", "#d0d4d8", 0.15)
    p = PM(0.004)
    p.r = 0.008
    p.box((-0.3, -0.3, 0.0), (0.3, 0.3, 1.0), walnut)
    p.r = 0.0
    decal(p, (0.0, 0.3, 0.8), (0, 1, 0), (0, 0, 1), 0.3, 0.12, pm, r, lift=0.002)
    p.box((-0.25, -0.25, 1.0), (0.25, 0.25, 1.01), _plastic("Hos_CabinetLed", "#fff6e6", 0.2))
    _cup(p, 0.0, 0.0, 1.01, 0.6, silver)
    for (lo, hi) in (((-0.27, -0.27, 1.0), (0.27, -0.26, 1.75)), ((-0.27, 0.26, 1.0), (0.27, 0.27, 1.75)), ((-0.27, -0.27, 1.0), (-0.26, 0.27, 1.75)),
                     ((0.26, -0.27, 1.0), (0.27, 0.27, 1.75)), ((-0.27, -0.27, 1.74), (0.27, 0.27, 1.75))):
        p.box(lo, hi, clear_glass())
    return p.obj("trophy_plinth", col)


def display_plinth(col, M):
    """A museum display plinth for one trophy (0.75 x 0.75 x 1.0 m, walnut, a brass plaque) under
    an empty glass case (0.7 x 0.7 x 0.8 m) with an uplight: the trophy (a generated prop) stands
    inside, placed 1.01 m up. Front +y."""
    cv = Canvas("Hos_PlaqueHonours", 256, 256, col="#b08a3a", rough=0.3, metal=0.9)
    r = cv.alloc(0.3, 0.12, 800)
    r.fill(col="#b8923e", metal=0.9, rough=0.3)
    r.text("HONOURS", 0.15, 0.07, 0.03, weight="black", align="c", col="#2a2010")
    r.text("WINNERS", 0.15, 0.025, 0.02, weight="bold", align="c", col="#2a2010")
    pm = cv.material("Hos_PlaqueHonoursMat")
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    p = PM(0.004)
    p.r = 0.008
    p.box((-0.375, -0.375, 0.0), (0.375, 0.375, 1.0), walnut)
    p.r = 0.0
    decal(p, (0.0, 0.375, 0.8), (0, 1, 0), (0, 0, 1), 0.3, 0.12, pm, r, lift=0.002)
    p.box((-0.3, -0.3, 1.0), (0.3, 0.3, 1.01), _plastic("Hos_CabinetLed", "#fff6e6", 0.2))
    for (lo, hi) in (((-0.35, -0.35, 1.0), (0.35, -0.34, 1.8)), ((-0.35, 0.34, 1.0), (0.35, 0.35, 1.8)), ((-0.35, -0.35, 1.0), (-0.34, 0.35, 1.8)),
                     ((0.34, -0.35, 1.0), (0.35, 0.35, 1.8)), ((-0.35, -0.35, 1.79), (0.35, 0.35, 1.8))):
        p.box(lo, hi, clear_glass())
    return p.obj("display_plinth", col)


def lectern(col, M):
    """A speaker's lectern (0.6 x 0.5 m, 1.2 m, walnut with a frosted acrylic front carrying the
    club crest, tinted) with a gooseneck microphone and a reading light. Front +y (the audience)."""
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    crest = _plastic("TINT_club_crest", "#c8202a", 0.3)
    p = PM(0.004)
    p.r = 0.008
    p.box((-0.3, -0.25, 0.0), (0.3, 0.2, 1.05), walnut)
    p.obox((0, -0.05, 1.12), (0.62, 0.5, 0.04), walnut, pitch=math.radians(-15))
    p.r = 0.0
    p.box((-0.24, 0.2, 0.3), (0.24, 0.22, 0.95), _plastic("Hos_Acrylic", "#e8eef2", 0.3))
    p.lathe([(0.0, 0.0), (0.1, 0.0), (0.1, 0.06), (0.06, 0.14), (0.0, 0.16)], crest, c=(0, 0.225, 0.55), sides=6)
    _tube(p, [(0.18, 0.05, 1.15), (0.18, 0.05, 1.3), (0.12, 0.12, 1.4)], 0.005, M["black"], 6)
    p.sphere((0.11, 0.13, 1.41), 0.018, M["black"], 10, 6)
    return p.obj("lectern", col)


# -- kitchen and washrooms --------------------------------------------------------------------------

def kitchen_line(col, M):
    """A commercial kitchen cook line (3.0 m, stainless, the back against the wall at y = 0): a
    six-burner range over an oven, a chargrill, a twin-basket fryer, a refrigerated prep counter
    with its gastronorm pans, a stainless splashback, pans on the burners. Front +y."""
    steel = _paint("Hos_Stainless", "#b8bcc0", 0.3)
    p = PM(0.004)
    p.r = 0.004
    p.box((-1.5, 0.0, 0.0), (1.5, 0.05, 1.6), steel)
    # range
    p.box((-1.5, 0.05, 0.1), (-0.6, 0.85, 0.9), steel)
    p.box((-1.45, 0.85, 0.2), (-0.65, 0.86, 0.7), M["black"])
    p.box((-1.2, 0.86, 0.62), (-0.9, 0.88, 0.64), M["chrome"])
    for i in range(3):
        for j in range(2):
            x, y = -1.35 + i * 0.3, 0.25 + j * 0.35
            p.r = 0.0
            p.cyl((x, y, 0.9), (x, y, 0.93), 0.1, M["black"], 16)
            for k in range(4):
                a = k * math.pi / 4
                p.box((x - 0.11 * math.cos(a) - 0.005, y - 0.11 * math.sin(a) - 0.005, 0.93), (x + 0.11 * math.cos(a) + 0.005, y + 0.11 * math.sin(a) + 0.005, 0.945), M["black"])
    for i in range(6):
        p.cyl((-1.42 + i * 0.15, 0.86, 0.82), (-1.42 + i * 0.15, 0.9, 0.82), 0.018, M["black"], 10)
    p.lathe([(0.0, 0.0), (0.13, 0.0), (0.14, 0.12), (0.0, 0.12)], steel, c=(-1.35, 0.25, 0.945), sides=18)
    p.cyl((-1.05, 0.6, 0.945), (-1.05, 0.6, 0.99), 0.14, M["black"], 18)
    _tube(p, [(-1.05, 0.74, 0.97), (-1.05, 0.98, 0.98)], 0.012, M["black"], 6)
    # chargrill and fryer
    p.r = 0.004
    p.box((-0.6, 0.05, 0.1), (0.0, 0.85, 0.9), steel)
    for k in range(10):
        p.box((-0.56 + k * 0.055, 0.15, 0.9), (-0.54 + k * 0.055, 0.75, 0.93), M["black"])
    p.box((0.0, 0.05, 0.1), (0.5, 0.85, 0.9), steel)
    for x in (0.12, 0.38):
        p.box((x - 0.1, 0.2, 0.86), (x + 0.1, 0.65, 0.9), _paint("Hos_Oil", "#c89a3a", 0.1))
        p.r = 0.0
        p.cyl((x, 0.75, 0.95), (x, 0.95, 1.05), 0.008, M["black"], 6)
        p.r = 0.004
    # prep fridge with pans
    p.box((0.5, 0.05, 0.1), (1.5, 0.8, 0.88), steel)
    for x in (0.75, 1.25):
        p.box((x - 0.2, 0.8, 0.15), (x + 0.2, 0.82, 0.8), steel)
        p.box((x - 0.05, 0.82, 0.7), (x + 0.05, 0.84, 0.72), M["chrome"])
    p.box((0.5, 0.05, 0.88), (1.5, 0.8, 0.92), steel)
    for k in range(6):
        c = ("#d84a2a", "#5aa03a", "#e8c040", "#f0f0e8", "#8a3a2a", "#3a8a3a")[k]
        p.box((0.55 + k * 0.155, 0.15, 0.86), (0.69 + k * 0.155, 0.4, 0.93), _paint("Hos_Prep" + c[1:], c, 0.7))
    p.box((-1.5, 0.05, 0.0), (1.5, 0.75, 0.1), M["black"])
    return p.obj("kitchen_line", col)


def extraction_hood(col, M):
    """The kitchen's extraction canopy (3.0 x 1.1 m, brushed stainless, its lower edge 2.0 m up):
    open underneath, a 0.4 m skirt round the front and sides with a grease gutter inside its
    lip, a sloped front face up to the flat top, a V of baffle filters along the back with their
    drip trays, two vapour-proof lights, the riveted panel seams and the duct rising to the
    ceiling with its fire damper. The back against the wall at y = 0; front +y."""
    cv = Canvas("Hos_Brushed", 512, 512, col="#b8bcc0", rough=0.3, metal=0.8)
    r = cv.alloc(1.0, 1.0, 400)
    r.fill(col="#b4b8bc", metal=0.8, rough=0.3)
    r.field(lambda X, Y: noise(X.shape, (4, 200), 3, 1) * 0.12, col="#d8dce0")
    r.field(lambda X, Y: (np.mod(X, 0.5) < 0.004) * 1.0, col="#7a7e82", dh=-0.001)
    brushed = cv.material("Hos_BrushedSteel", metal=0.8)
    baffle = _paint("Hos_Baffle", "#8e9498", 0.35)
    p = PM(0.004)
    W, D, z0 = 1.55, 1.1, 2.0

    def panel(lo, hi):
        i0 = len(p.F)
        p.box(lo, hi, brushed)
        box_uv(p, i0, {"+y": r, "-y": r, "+x": r, "-x": r, "+z": r, "-z": r}, lo, hi)
    p.r = 0.003
    # skirt: front and sides, open underneath
    panel((-W, D - 0.02, z0), (W, D, z0 + 0.4))
    for x in (-W, W - 0.02):
        panel((x, 0.0, z0), (x + 0.02, D, z0 + 0.4))
    # sloped front face and the flat top
    p.r = 0.0
    p.face([(-W, D, z0 + 0.4), (W, D, z0 + 0.4), (W, D - 0.3, z0 + 0.6), (-W, D - 0.3, z0 + 0.6)], brushed)
    panel((-W, 0.0, z0 + 0.6), (W, D - 0.3, z0 + 0.62))
    for s in (-1, 1):
        x = s * W
        p.face([(x, D - 0.3, z0 + 0.6), (x, D, z0 + 0.4), (x, 0.0, z0 + 0.4), (x, 0.0, z0 + 0.6)][:: (1 if s > 0 else -1)], brushed)
    panel((-W, 0.0, z0 + 0.4), (W, 0.02, z0 + 0.6))
    # grease gutter inside the skirt lip
    p.r = 0.002
    p.box((-W + 0.02, D - 0.1, z0), (W - 0.02, D - 0.02, z0 + 0.04), baffle)
    # baffle filters in a V along the back, drip trays under them
    for k in range(10):
        x = -1.35 + k * 0.3
        p.r = 0.002
        p.obox((x, 0.18, z0 + 0.38), (0.28, 0.03, 0.42), baffle, pitch=math.radians(50))
        for j in range(5):
            p.obox((x - 0.1 + j * 0.05, 0.19, z0 + 0.38), (0.012, 0.035, 0.4), _paint("Hos_BaffleFin", "#6e7478", 0.4), pitch=math.radians(50))
    p.box((-W + 0.02, 0.3, z0 + 0.18), (W - 0.02, 0.38, z0 + 0.22), baffle)
    # vapour-proof lights recessed in the top
    for x in (-0.8, 0.8):
        p.r = 0.006
        p.box((x - 0.3, 0.55, z0 + 0.57), (x + 0.3, 0.7, z0 + 0.6), _plastic("Hos_HoodLight", "#fff6e6", 0.2))
    # duct to the ceiling, fire damper access panel
    panel((-0.3, 0.3, z0 + 0.62), (0.3, 0.8, 3.2))
    p.r = 0.004
    p.box((-0.15, 0.8, 2.9), (0.15, 0.805, 3.05), M["red"])
    return p.obj("extraction_hood", col)


def prep_table(col, M):
    """A stainless prep table (1.8 x 0.7 m, an undershelf of gastronorm pans and a stockpot):
    colour-coded chopping boards, a knife block, a stack of plates, a ticket rail. Front +y."""
    steel = _paint("Hos_Stainless", "#b8bcc0", 0.3)
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.9, -0.35, 0.86), (0.9, 0.35, 0.9), steel)
    for x in (-0.86, 0.86):
        for y in (-0.31, 0.31):
            p.r = 0.0
            p.cyl((x, y, 0.0), (x, y, 0.86), 0.02, steel, 8)
    p.r = 0.004
    p.box((-0.86, -0.31, 0.2), (0.86, 0.31, 0.22), steel)
    for k in range(3):
        p.box((-0.8 + k * 0.35, -0.25, 0.22), (-0.5 + k * 0.35, 0.25, 0.32 + k * 0.03), steel)
    p.r = 0.0
    p.cyl((0.55, 0.0, 0.22), (0.55, 0.0, 0.52), 0.18, steel, 18)
    for k, c in enumerate(("#d84a2a", "#2a8a3a", "#2457a6", "#e8c040")):
        p.r = 0.004
        p.box((-0.8 + k * 0.03, -0.25 + k * 0.03, 0.9 + k * 0.012), (-0.35 + k * 0.03, 0.05 + k * 0.03, 0.912 + k * 0.012), _plastic("Hos_Board" + c[1:], c, 0.6))
    p.box((0.1, -0.2, 0.9), (0.22, -0.05, 1.12), M["black"])
    for j in range(8):
        _plate(p, M, 0.55, 0.05, 0.9 + j * 0.022, 0.13)
    return p.obj("prep_table", col)


def wc_cubicles(col, M):
    """A run of three toilet cubicles (2.7 m wide, 1.5 m deep; the back wall at y = 0): compact-
    laminate partitions and doors on stainless feet with indicator bolts, one door ajar, a wall-
    hung pan and roll holder in each. Front +y."""
    panel = _paint("Hos_Laminate", "#5a6470", 0.35)
    ceramic = _paint("Med_Ceramic", "#f6f6f2", 0.12)
    p = PM(0.004)
    for k in range(4):
        x = -1.35 + k * 0.9
        p.r = 0.004
        p.box((x - 0.01, 0.0, 0.15), (x + 0.01, 1.5, 2.0), panel)
        p.r = 0.0
        p.cyl((x, 1.45, 0.0), (x, 1.45, 0.15), 0.014, M["chrome"], 8)
    for k in range(3):
        x = -0.9 + k * 0.9
        p.r = 0.0
        p.lathe([(0.0, 0.25), (0.17, 0.27), (0.19, 0.33), (0.185, 0.4), (0.0, 0.41)], ceramic, c=(x, 0.32, 0.0), sides=18)
        p.r = 0.04
        p.box((x - 0.17, 0.0, 0.28), (x + 0.17, 0.2, 0.41), ceramic)
        p.r = 0.002
        p.box((x - 0.1, 0.0, 0.9), (x + 0.1, 0.03, 1.1), M["chrome"])
        p.r = 0.0
        p.cyl((x + 0.28, 0.6, 0.7), (x + 0.4, 0.6, 0.7), 0.055, M["white"], 12)
        # the door: closed on the outer two, the middle one ajar
        p.r = 0.004
        if k == 1:
            p.obox((x - 0.4 + 0.33 * math.cos(0.8), 1.5 - 0.33 * math.sin(0.8) + 0.0, 1.075), (0.66, 0.02, 1.85), panel, yaw=-0.8)
        else:
            p.box((x - 0.36, 1.49, 0.15), (x + 0.36, 1.51, 2.0), panel)
            p.box((x + 0.25, 1.51, 1.0), (x + 0.3, 1.53, 1.06), _plastic("Hos_Engaged", "#c8202a" if k == 0 else "#2a8a3a", 0.3))
        p.box((x - 0.45, 1.49, 1.98), (x + 0.45, 1.53, 2.02), M["chrome"])
    return p.obj("wc_cubicles", col)


def vanity_basins(col, M):
    """A washroom vanity (2.4 x 0.55 m, a quartz top at 0.85 m on a hidden frame): three inset
    basins with sensor taps, a soap dispenser to each, the plumbing panel below. The back against
    the wall at y = 0; front +y."""
    quartz = _paint("Hos_Quartz", "#e8e6e2", 0.15)
    ceramic = _paint("Med_Ceramic", "#f6f6f2", 0.12)
    p = PM(0.004)
    p.r = 0.006
    p.box((-1.2, 0.0, 0.82), (1.2, 0.55, 0.86), quartz)
    p.box((-1.2, 0.0, 0.3), (1.2, 0.03, 0.82), _paint("Hos_Laminate", "#5a6470", 0.35))
    for k in range(3):
        x = -0.75 + k * 0.75
        p.r = 0.0
        p.lathe([(0.0, 0.7), (0.18, 0.72), (0.2, 0.86), (0.0, 0.86)], ceramic, c=(x, 0.3, 0.0), sides=20)
        p.cyl((x, 0.08, 0.86), (x, 0.08, 1.06), 0.015, M["chrome"], 10)
        p.cyl((x, 0.08, 1.06), (x, 0.2, 1.04), 0.013, M["chrome"], 10)
        p.r = 0.006
        p.box((x + 0.22, 0.05, 0.86), (x + 0.3, 0.13, 1.06), M["chrome"])
    return p.obj("vanity_basins", col)


def washroom_mirror(col, M):
    """The washroom mirror (2.4 x 0.9 m of silvered glass, its lower edge 1.1 m up) on a back-lit
    stand-off with a slim brushed frame, a glass shelf under it, and a stainless jet hand dryer
    beside it with its blue LED and drip tray. The back against the wall at y = 0; front +y."""
    mirror = _mirror_glass()
    frame = _paint("Hos_Stainless", "#b8bcc0", 0.3)
    led = _plastic("Hos_CabinetLed", "#fff6e6", 0.2)
    p = PM(0.004)
    p.r = 0.0
    p.box((-1.17, 0.0, 1.13), (1.17, 0.03, 1.97), led)
    p.r = 0.002
    p.box((-1.2, 0.03, 1.1), (1.2, 0.045, 2.0), frame)
    p.r = 0.0
    p.box((-1.185, 0.045, 1.115), (1.185, 0.05, 1.985), mirror)
    p.r = 0.004
    p.box((-1.0, 0.0, 1.05), (1.0, 0.13, 1.06), clear_glass())
    for x in (-0.9, 0.9):
        p.box((x - 0.02, 0.0, 1.03), (x + 0.02, 0.05, 1.06), frame)
    # jet hand dryer: a tall stainless body, the hand slot, LED, drip tray
    p.r = 0.03
    p.box((1.4, 0.0, 0.95), (1.7, 0.24, 1.65), frame)
    p.r = 0.005
    p.box((1.42, 0.12, 1.1), (1.68, 0.26, 1.3), _plastic("Hos_DryerSlot", "#121314", 0.5))
    p.box((1.52, 0.24, 1.5), (1.58, 0.245, 1.52), _plastic("Hos_DryerLed", "#3aa0ff", 0.2))
    p.box((1.42, 0.0, 0.9), (1.68, 0.2, 0.95), _plastic("Hos_DripTray", "#2a2c2e", 0.5))
    return p.obj("washroom_mirror", col)


BUILDERS = {
    "prop.shirt_rail": shirt_rail, "prop.merch_wall": merch_wall, "prop.gondola": gondola, "prop.shop_mannequin": shop_mannequin,
    "prop.shop_counter": shop_counter, "prop.print_station": print_station, "prop.stock_racking": stock_racking,
    "prop.ticket_desk": ticket_desk, "prop.cafe_counter": cafe_counter, "prop.cafe_backbar": cafe_backbar, "prop.menu_board": menu_board,
    "prop.cafe_table": cafe_table, "prop.cafe_table4": cafe_table4, "prop.bar_counter": bar_counter, "prop.bar_back": bar_back,
    "prop.dining_table": dining_table, "prop.banquet_table": banquet_table, "prop.armchair": armchair,
    "prop.trophy_cabinet": trophy_cabinet, "prop.trophy_plinth": trophy_plinth, "prop.display_plinth": display_plinth, "prop.lectern": lectern,
    "prop.kitchen_line": kitchen_line, "prop.extraction_hood": extraction_hood, "prop.prep_table": prep_table,
    "prop.wc_cubicles": wc_cubicles, "prop.vanity_basins": vanity_basins, "prop.washroom_mirror": washroom_mirror,
}
