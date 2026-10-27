"""Medical, rehab and sports-science equipment for a club's medical centre (kit/site/medical.py):
treatment and examination couches, consulting furniture, clinical hand-wash units, supply
cabinets, trolleys, the defibrillator station, ice machines, privacy curtains; the rehab gym
(treadmill, cycle ergometer, power rack, dumbbells, isokinetic dynamometer, anti-gravity
treadmill); recovery (ice baths, cryotherapy chamber, sauna, pool hoist, sleep pods); imaging and
the lab (ultrasound, X-ray, DEXA, metabolic cart); reception and waiting. Real sizes (manufacturer
classes named in each docstring); origin on the floor at the footprint centre, +Y the front (wall
units: the back against the wall at y = 0, facing +y).
"""
import math

import numpy as np

from .texpaint import Canvas, box_regions, box_uv, decal, noise, project
from .props import PM, rrect, _paint, _plastic, clear_glass, tyre
from .props_amenity import _loft, _tube, _hexbolt


# -- shared bits ----------------------------------------------------------------------------------

def _white(name="Med_White"):
    return _paint(name, "#e9e9e4", 0.38)


def _bag_vinyl(name="Med_BagVinyl"):
    """Clear PVC (an anti-gravity treadmill's pressurised bag): pale, glossy, mostly see-through."""
    import bpy
    from ..core.nodes import material, srgb
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#dfe6ec"), Roughness=0.15, Alpha=0.35, Specular_IOR_Level=0.6))
    return m


def _castor(p, c, M, r=0.05, lock=False):
    """A swivel castor under a frame at c (the top plate's centre): plate, swivel, fork, a grey
    tread wheel; a total-lock pedal if `lock`."""
    x, y, z = c
    rr, p.r = p.r, 0.003
    p.box((x - 0.035, y - 0.035, z - 0.008), (x + 0.035, y + 0.035, z), M["galv"])
    p.cyl((x, y, z - 0.008), (x, y, z - 0.025), 0.022, M["galv"], 12)
    for s in (-1, 1):
        p.box((x + s * 0.026 - 0.003, y - 0.03, z - 0.025 - 2 * r + 0.012), (x + s * 0.026 + 0.003, y + 0.01, z - 0.025), M["galv"])
    wz = z - 0.025 - r
    p.r = 0.0
    tyre(p, (x, y - 0.012, wz), r, 0.036, r * 0.6, M["rubber"], M["grey"], sides=16, axis="x")
    if lock:
        p.r = 0.002
        p.box((x - 0.02, y + 0.012, wz + r * 0.4), (x + 0.02, y + 0.07, wz + r * 0.4 + 0.012), M["red"])
    p.r = rr


def _cushion_canvas(name, w, h, base="#2e5f8f", ppm=300, seed=1):
    """A vinyl upholstery canvas (greys: the tint gives the colour) for a w x h panel: fine grain,
    a welted seam round the edge, the odd scuff."""
    cv = Canvas(name, 512, 512, col="#ffffff", rough=0.55)
    reg = cv.alloc(w, h, ppm)
    reg.fill(col="#ffffff")
    reg.field(lambda X, Y: noise(X.shape, (int(h * 140) + 2, int(w * 140) + 2), seed, 2), dh=0.00015)
    reg.mottle(0.04, 4, seed + 1)
    reg.frame(0.006, 0.006, w - 0.006, h - 0.006, 0.006, r=0.02, col="#d8d8d8", dh=-0.0012)
    reg.field(lambda X, Y: noise(X.shape, 6, seed + 2, 3) ** 4 * 0.5, col="#b8b8b8")
    return cv, reg


def _screen(reg, kind, seed=1):
    """A lit display's contents (emissive): an EMR record, an ultrasound image, a training console."""
    W, H = reg.w, reg.h
    if kind == "emr":
        reg.fill(col="#f2f4f7", emit="#f2f4f7")
        reg.rect(0, H - H * 0.09, W, H, col="#1d4f91", emit="#1d4f91")
        reg.text("PATIENT RECORD", W * 0.03, H - H * 0.07, H * 0.05, weight="bold", col="#ffffff", emit="#ffffff")
        for i in range(9):
            y = H * 0.8 - i * H * 0.075
            reg.rect(W * 0.03, y, W * (0.25 + 0.4 * ((i * 37 + seed) % 7) / 7), y + H * 0.025, col="#9aa3ad", emit="#9aa3ad")
        reg.rect(W * 0.62, H * 0.2, W * 0.97, H * 0.8, col="#dfe6ee", emit="#dfe6ee")
        pts = [(W * (0.64 + 0.32 * k / 11), H * (0.3 + 0.35 * (0.5 + 0.5 * math.sin(k * 0.9 + seed)))) for k in range(12)]
        reg.line(pts, H * 0.012, col="#d0402a", emit="#d0402a")
    elif kind == "ultrasound":
        reg.fill(col="#020203", emit="#020203")
        reg.field(lambda X, Y: np.clip(1 - np.hypot((X - W / 2) / (W * 0.42), (Y - H) / (H * 0.95)), 0, 1)
                  * (np.abs(np.arctan2(X - W / 2, H - Y)) < 0.6) * (0.35 + 0.65 * noise(X.shape, 40, seed, 3)), col="#c8c8c0", emit="#c8c8c0")
        reg.text("MSK  L12-4", W * 0.03, H * 0.93, H * 0.04, col="#7fd0ff", emit="#7fd0ff")
    elif kind == "console":
        reg.fill(col="#0b0f14", emit="#0b0f14")
        for i, (t, v) in enumerate((("SPEED", "14.0"), ("INCLINE", "1.0"), ("HR", "152"))):
            x = W * (0.06 + i * 0.32)
            reg.text(t, x, H * 0.72, H * 0.09, weight="bold", col="#7a8a99", emit="#7a8a99")
            reg.text(v, x, H * 0.3, H * 0.26, weight="black", col="#3fd06a" if i == 2 else "#ffffff", emit="#3fd06a" if i == 2 else "#ffffff")
    elif kind == "graph":
        reg.fill(col="#0d1117", emit="#0d1117")
        for k in range(3):
            pts = [(W * (0.05 + 0.9 * i / 40), H * (0.2 + 0.2 * k + 0.12 * math.sin(i * (0.3 + 0.1 * k) + seed))) for i in range(41)]
            reg.line(pts, H * 0.012, col=("#3fd06a", "#e0b040", "#4aa0f0")[k], emit=("#3fd06a", "#e0b040", "#4aa0f0")[k])
        reg.text("VO2  52.4  ml/kg/min", W * 0.05, H * 0.88, H * 0.06, weight="bold", col="#ffffff", emit="#ffffff")


def _monitor(p, c, yaw, w, h, mat, reg, M, stand=True):
    """A slim flat-panel monitor (screen region `reg`) on a pedestal: c = the base's centre on
    the desk, the screen facing +y turned by yaw."""
    cx, cy, cz = c
    ca, sa = math.cos(yaw), math.sin(yaw)

    def rot(x, y):
        return (cx + x * ca - y * sa, cy + x * sa + y * ca)
    rr, p.r = p.r, 0.004
    if stand:
        p.obox((*rot(0, -0.04), cz + 0.005), (0.22, 0.17, 0.01), M["black"], yaw=yaw)
        p.obox((*rot(0, -0.07), cz + 0.15), (0.05, 0.02, 0.28), M["black"], yaw=yaw)
    z0 = cz + 0.1
    p.obox((*rot(0, -0.045), z0 + h / 2), (w + 0.02, 0.025, h + 0.02), M["black"], yaw=yaw)
    p.r = 0.0
    n = (-sa, ca, 0.0)
    decal(p, (*rot(0, -0.032), z0 + h / 2), n, (0, 0, 1), w, h, mat, reg, lift=0.001)
    p.r = rr


# -- couches and seating --------------------------------------------------------------------------

def med_couch(col, M):
    """An electric three-section treatment couch (Plinth Medical / Akron class, 1.95 x 0.67 m, the
    top 0.62 m up in its mid position): a 60 mm steel base frame on four total-lock castors, a
    grey lifting column, a hi-lo control bar round the base, a raised head section with a face
    hole and plug, seat and leg sections in welted vinyl (tinted: the clinic's colour), a couch
    roll holder at the head and paper drawn half along it, a hand controller on its curly cable.
    Long axis along x, the head at -x; front +y (the physio's side)."""
    L, Wd, ZT = 1.95, 0.67, 0.62
    cv, reg = _cushion_canvas("Med_CouchVinyl", 1.0, 0.7, seed=3)
    up = cv.tint_material("TINT_upholstery_couch", "#2e5f8f", rough=0.55)
    frame = _paint("Med_CouchFrame", "#d9dbdc", 0.4)
    p = PM(0.006)
    # base frame on castors
    for y in (-0.22, 0.22):
        p.box((-0.82, y - 0.03, 0.09), (0.82, y + 0.03, 0.15), frame)
    for x in (-0.82, 0.82):
        p.box((x - 0.03, -0.25, 0.09), (x + 0.03, 0.25, 0.15), frame)
        for y in (-0.22, 0.22):
            _castor(p, (x, y, 0.09), M, 0.04, lock=y > 0)
    # lifting column with its bellows, top frame
    p.r = 0.01
    p.box((-0.16, -0.12, 0.15), (0.16, 0.12, ZT - 0.12), M["grey"])
    p.r = 0.004
    for k in range(6):
        z = 0.18 + k * 0.05
        p.box((-0.165, -0.125, z), (0.165, 0.125, z + 0.012), _plastic("Med_Bellows", "#5b5f63", 0.6))
    p.box((-0.3, -0.18, ZT - 0.12), (0.3, 0.18, ZT - 0.08), frame)
    for y in (-0.27, 0.27):
        p.box((-0.92, y - 0.02, ZT - 0.1), (0.92, y + 0.02, ZT - 0.07), frame)
    # hi-lo bar: a tube loop round the base at ankle height
    p.r = 0.0
    _tube(p, [(-0.9, -0.3, 0.2), (0.9, -0.3, 0.2), (0.9, 0.3, 0.2), (-0.9, 0.3, 0.2)], 0.012, frame, 10, bend=0.06, closed=True)
    for x in (-0.82, 0.82):
        for y in (-0.22, 0.22):
            _tube(p, [(x, y, 0.15), (x * 1.09, y * 1.36, 0.2)], 0.01, frame, 8)
    # cushions: head (raised 12 degrees) with a face hole, seat, legs; welted, 70 mm foam
    p.r = 0.03
    th = 0.07
    i0 = len(p.F)
    p.box((-0.37, -Wd / 2, ZT - 0.07), (0.27, Wd / 2, ZT - 0.07 + th), up)
    p.box((0.29, -Wd / 2, ZT - 0.07), (0.975, Wd / 2, ZT - 0.07 + th), up)
    a = math.radians(10)
    hl_ = 0.58
    zc = ZT - 0.07 + th / 2
    p.obox((-0.39 - hl_ / 2 * math.cos(a), 0.0, zc + hl_ / 2 * math.sin(a)), (hl_, Wd, th), up, roll=a)
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.0, -0.35, 0.5), (1.0, 0.35, 0.8))
    # face hole with its plug, a little proud
    p.r = 0.0
    d_ = 0.42
    hc = (-0.39 - d_ * math.cos(a), 0.0, zc + d_ * math.sin(a) + th / 2 + 0.002)
    p.lathe([(0.0, 0.0), (0.075, 0.0), (0.08, -0.004)], _plastic("Med_FacePlug", "#1d3d5c", 0.5), c=hc, sides=20)
    # couch roll: a holder at the head, the paper drawn halfway down the couch
    p.r = 0.003
    for y in (-0.3, 0.3):
        p.box((-1.0, y - 0.01, ZT - 0.05), (-0.98, y + 0.01, ZT + 0.08), frame)
    p.cyl((-1.02, -0.28, ZT + 0.04), (-1.02, 0.28, ZT + 0.04), 0.06, _white("Med_Paper"), 16)
    paper = _plastic("Med_CouchRoll", "#f4f3ee", 0.85)
    p.r = 0.0
    zt = ZT - 0.07 + th + 0.003
    zh = zt + 0.5 * math.sin(a)
    p.face([(-0.39 - 0.5 * math.cos(a), -0.25, zh), (-0.39 - 0.5 * math.cos(a), 0.25, zh), (-0.39, 0.25, zt), (-0.39, -0.25, zt)], paper)
    p.face([(-0.39, -0.25, zt), (-0.39, 0.25, zt), (0.55, 0.25, zt), (0.55, -0.25, zt)], paper)
    # hand controller on its curly cable from the column
    p.r = 0.004
    p.box((0.18, 0.34, ZT - 0.2), (0.24, 0.36, ZT - 0.08), M["grey"])
    p.r = 0.0
    curl = [(0.16 + 0.012 * math.cos(t * 2.4), 0.12 + 0.02 * t, 0.3 + 0.012 * math.sin(t * 2.4)) for t in np.linspace(0, 12, 60)]
    _tube(p, curl, 0.003, M["black"], 4)
    return p.obj("med_couch", col)


def med_couch_folding(col, M):
    """A portable folding treatment couch (the first-aid room's; 1.85 x 0.61 m, 0.7 m high): two
    vinyl-topped plywood halves on a hinged aluminium frame with cross-braced legs and rubber
    feet, a carry handle on the side, a face hole in the head end. Front +y."""
    L, Wd, ZT = 1.85, 0.61, 0.7
    cv, reg = _cushion_canvas("Med_FoldVinyl", 1.0, 0.65, seed=7)
    up = cv.tint_material("TINT_upholstery_fold", "#2b2b2e", rough=0.55)
    p = PM(0.004)
    for x0, x1 in ((-L / 2, -0.01), (0.01, L / 2)):
        p.box((x0, -Wd / 2, ZT - 0.05), (x1, Wd / 2, ZT - 0.035), M["timber"])
        p.r = 0.025
        i0 = len(p.F)
        p.box((x0, -Wd / 2, ZT - 0.035), (x1, Wd / 2, ZT + 0.015), up)
        box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.0, -0.33, 0.6), (1.0, 0.33, 0.8))
        p.r = 0.004
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.06, 0.0)], M["black"], c=(-0.7, 0.0, ZT + 0.017), sides=18)
    for x in (-0.8, -0.1, 0.1, 0.8):
        legs = [((x, -0.26, ZT - 0.06), (x * 0.98, -0.28, 0.02)), ((x, 0.26, ZT - 0.06), (x * 0.98, 0.28, 0.02))]
        for a_, b_ in legs:
            _tube(p, [a_, b_], 0.012, M["alu"], 8)
            p.cyl((b_[0], b_[1], 0.02), (b_[0], b_[1], 0.0), 0.016, M["rubber"], 10)
        _tube(p, [(x, -0.26, ZT - 0.06), (x * 0.98, 0.28, 0.12)], 0.007, M["alu"], 6)
        _tube(p, [(x, 0.26, ZT - 0.06), (x * 0.98, -0.28, 0.12)], 0.007, M["alu"], 6)
    for y in (-0.27, 0.27):
        _tube(p, [(-0.8, y, 0.12), (-0.1, y, 0.12)], 0.008, M["alu"], 6)
        _tube(p, [(0.1, y, 0.12), (0.8, y, 0.12)], 0.008, M["alu"], 6)
    p.r = 0.006
    p.box((-0.08, Wd / 2, ZT - 0.04), (0.08, Wd / 2 + 0.03, ZT - 0.01), M["black"])
    return p.obj("med_couch_folding", col)


def med_exam_couch(col, M):
    """A fixed-height examination couch (Doherty / Bristol Maid class, 1.83 x 0.61 m, 0.76 m high):
    a white epoxy-coated steel frame with a gas-strut adjustable backrest at 30 degrees, a
    one-piece vinyl mattress, a paper roll holder at the foot with the roll pulled up the couch,
    a two-step stool beside it. Front +y."""
    L, Wd, ZT = 1.83, 0.61, 0.72
    cv, reg = _cushion_canvas("Med_ExamVinyl", 1.0, 0.65, seed=11)
    up = cv.tint_material("TINT_upholstery_exam", "#7d8f9c", rough=0.55)
    frame = _white("Med_ExamFrame")
    p = PM(0.005)
    for x in (-0.82, 0.82):
        for y in (-0.26, 0.26):
            p.box((x - 0.02, y - 0.02, 0.02), (x + 0.02, y + 0.02, ZT - 0.04), frame)
            p.cyl((x, y, 0.02), (x, y, 0.0), 0.022, M["black"], 10)
    for y in (-0.26, 0.26):
        p.box((-0.84, y - 0.02, ZT - 0.08), (0.84, y + 0.02, ZT - 0.04), frame)
        p.box((-0.84, y - 0.015, 0.18), (0.84, y + 0.015, 0.21), frame)
    for x in (-0.82, 0.82):
        p.box((x - 0.015, -0.26, 0.18), (x + 0.015, 0.26, 0.21), frame)
    p.box((-0.8, -0.26, 0.2), (0.8, 0.26, 0.215), M["timber"])
    p.r = 0.03
    i0 = len(p.F)
    p.box((-0.3, -Wd / 2, ZT - 0.04), (L / 2, Wd / 2, ZT + 0.04), up)
    a = math.radians(30)
    p.obox((-0.3 - 0.31 * math.cos(a), 0.0, ZT + 0.31 * math.sin(a)), (0.62, Wd, 0.08), up, roll=a)
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.0, -0.33, 0.6), (1.0, 0.33, 1.0))
    p.r = 0.0
    _tube(p, [(-0.32, -0.24, ZT - 0.05), (-0.62, -0.24, ZT + 0.1)], 0.01, M["chrome"], 8)
    p.r = 0.003
    p.cyl((L / 2 + 0.05, -0.27, ZT - 0.02), (L / 2 + 0.05, 0.27, ZT - 0.02), 0.055, _white("Med_Paper"), 16)
    paper = _plastic("Med_CouchRoll", "#f4f3ee", 0.85)
    p.r = 0.0
    p.face([(L / 2 + 0.05, -0.25, ZT + 0.035), (L / 2 + 0.05, 0.25, ZT + 0.035), (-0.25, 0.25, ZT + 0.045), (-0.25, -0.25, ZT + 0.045)], paper)
    # the step stool, two treads
    p.r = 0.004
    sx, sy = 0.3, 0.55
    for k, z in enumerate((0.2, 0.4)):
        p.box((sx - 0.22, sy - 0.17 + k * 0.12, z - 0.02), (sx + 0.22, sy + 0.05, z), frame)
        p.box((sx - 0.2, sy - 0.15 + k * 0.12, z), (sx + 0.2, sy + 0.03, z + 0.006), M["rubber"])
    for x in (sx - 0.2, sx + 0.2):
        _tube(p, [(x, sy + 0.04, 0.0), (x, sy + 0.04, 0.4)], 0.012, frame, 8)
        _tube(p, [(x, sy - 0.16, 0.0), (x, sy - 0.04, 0.4)], 0.012, frame, 8)
    return p.obj("med_exam_couch", col)


def med_chair_office(col, M):
    """A clinic task chair (mesh back, wipe-clean vinyl seat): a five-star aluminium base on twin
    castors, a gas lift with its shroud, a contoured seat, a mesh backrest on a curved spine, height
    adjustable arms. Front +y."""
    p = PM(0.004)
    mesh = _plastic("Med_ChairMesh", "#2a2c2f", 0.7)
    for k in range(5):
        a = math.pi / 2 + k * 2 * math.pi / 5
        e = (0.33 * math.cos(a), 0.33 * math.sin(a))
        p.r = 0.0
        _tube(p, [(0, 0, 0.1), (e[0], e[1], 0.07)], 0.018, M["alu"], 8)
        p.cyl((e[0], e[1], 0.07), (e[0], e[1], 0.05), 0.012, M["black"], 8)
        tyre(p, (e[0], e[1], 0.03), 0.03, 0.03, 0.015, M["black"], M["black"], sides=12, axis="x")
    p.cyl((0, 0, 0.1), (0, 0, 0.26), 0.03, M["black"], 14)
    p.cyl((0, 0, 0.26), (0, 0, 0.42), 0.014, M["chrome"], 12)
    p.r = 0.04
    p.box((-0.24, -0.24, 0.42), (0.24, 0.22, 0.48), _plastic("Med_ChairSeat", "#33363a", 0.55))
    p.r = 0.006
    p.box((-0.1, -0.27, 0.42), (0.1, -0.24, 0.6), M["black"])
    p.r = 0.02
    p.obox((0, -0.3, 0.8), (0.44, 0.04, 0.45), mesh, pitch=math.radians(-8))
    for s in (-1, 1):
        p.r = 0.005
        p.box((s * 0.25 - 0.015, -0.12, 0.44), (s * 0.25 + 0.015, -0.08, 0.64), M["black"])
        p.box((s * 0.25 - 0.035, -0.18, 0.64), (s * 0.25 + 0.035, 0.08, 0.67), M["black"])
    return p.obj("med_chair_office", col)


def med_chair(col, M):
    """A wipe-clean visitor chair (healthcare upholstery, chrome sled frame with arms). Front +y."""
    cv, reg = _cushion_canvas("Med_ChairVinyl", 0.6, 0.6, seed=13)
    up = cv.tint_material("TINT_upholstery_chair", "#4e6e8a", rough=0.55)
    p = PM(0.004)
    for s in (-1, 1):
        x = s * 0.27
        p.r = 0.0
        _tube(p, [(x, 0.22, 0.01), (x, -0.25, 0.01), (x, -0.25, 0.62), (x, 0.0, 0.62), (x, 0.18, 0.6), (x, 0.22, 0.45), (x, 0.22, 0.01)], 0.012, M["chrome"], 8, bend=0.04)
    p.r = 0.03
    i0 = len(p.F)
    p.box((-0.26, -0.22, 0.42), (0.26, 0.22, 0.49), up)
    p.obox((0, -0.24, 0.72), (0.5, 0.06, 0.42), up, pitch=math.radians(-6))
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-0.3, -0.3, 0.3), (0.3, 0.3, 0.95))
    for s in (-1, 1):
        p.r = 0.01
        p.box((s * 0.27 - 0.03, -0.2, 0.62), (s * 0.27 + 0.03, 0.12, 0.65), M["black"])
    return p.obj("med_chair", col)


# -- consulting, washing, storing -----------------------------------------------------------------

def med_desk(col, M):
    """A consulting desk (1.6 x 0.8 m, white laminate on a steel goalpost frame with a modesty
    panel and a three-drawer pedestal): an electronic patient record on a monitor, keyboard and
    mouse, a desk phone, a sharps-free tray of forms, a stethoscope and a blood-pressure monitor.
    Front +y (the clinician sits on -y, facing +y... the patient's side)."""
    cv = Canvas("Med_Desk", 512, 512, col="#f2f2f0", rough=0.4)
    scr = cv.alloc(0.53, 0.3, 700)
    _screen(scr, "emr", 2)
    bp = cv.alloc(0.12, 0.08, 1500)
    bp.fill(col="#dcdedc")
    bp.rect(0.01, 0.025, 0.07, 0.07, r=0.004, col="#1a2a1a", emit="#1a2a1a")
    bp.text("128", 0.015, 0.05, 0.014, weight="black", col="#9fe0a0", emit="#9fe0a0")
    bp.text("82", 0.015, 0.03, 0.012, weight="black", col="#9fe0a0", emit="#9fe0a0")
    bp.circle(0.095, 0.045, 0.012, col="#3a6ad0")
    scr_m = cv.material("Med_DeskScreen", emission=1.2)
    top = _paint("Med_DeskTop", "#efefec", 0.35)
    frame = _paint("Med_DeskFrame", "#3a3d41", 0.4)
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.8, -0.4, 0.71), (0.8, 0.4, 0.735), top)
    for x in (-0.76, 0.76):
        p.box((x - 0.025, -0.36, 0.0), (x + 0.025, 0.36, 0.03), frame)
        p.box((x - 0.025, -0.02, 0.03), (x + 0.025, 0.02, 0.71), frame)
        p.box((x - 0.025, -0.36, 0.68), (x + 0.025, 0.36, 0.71), frame)
    p.box((-0.73, 0.33, 0.25), (0.73, 0.345, 0.68), top)
    # pedestal with three drawers and pulls
    p.box((0.3, -0.38, 0.02), (0.72, 0.2, 0.69), top)
    for k, z in enumerate((0.05, 0.27, 0.48)):
        p.box((0.31, -0.39, z), (0.71, -0.38, z + 0.19), _paint("Med_Drawer", "#e4e4e0", 0.35))
        p.box((0.46, -0.405, z + 0.15), (0.56, -0.39, z + 0.165), M["alu"])
    _monitor(p, (-0.1, 0.12, 0.735), math.pi, 0.53, 0.3, scr_m, scr, M)
    p.r = 0.003
    p.box((-0.32, -0.2, 0.735), (0.12, -0.06, 0.755), M["black"])
    for i in range(4):
        for j in range(14):
            p.box((-0.31 + j * 0.03, -0.19 + i * 0.033, 0.755), (-0.29 + j * 0.03, -0.165 + i * 0.033, 0.76), _plastic("Med_Key", "#2b2d30", 0.6))
    p.box((0.18, -0.16, 0.735), (0.24, -0.06, 0.765), M["black"])
    # phone, forms tray, stethoscope, BP monitor
    p.box((-0.7, -0.2, 0.735), (-0.5, -0.02, 0.79), _plastic("Med_Phone", "#202224", 0.5))
    p.box((-0.68, -0.15, 0.79), (-0.52, -0.08, 0.83), _plastic("Med_Phone", "#202224", 0.5))
    p.box((0.42, 0.05, 0.735), (0.72, 0.32, 0.76), M["grey"])
    p.box((0.44, 0.07, 0.76), (0.7, 0.3, 0.785), _white("Med_Paper"))
    p.r = 0.0
    loop = [(0.3 + 0.08 * math.cos(t), -0.28 + 0.06 * math.sin(t), 0.74) for t in np.linspace(0, 2 * math.pi, 18)]
    _tube(p, loop, 0.004, M["black"], 5, closed=True)
    p.cyl((0.38, -0.28, 0.735), (0.38, -0.28, 0.745), 0.022, M["chrome"], 14)
    p.r = 0.003
    p.box((-0.72, 0.12, 0.735), (-0.6, 0.2, 0.8), _plastic("Med_BP", "#dcdedc", 0.4))
    decal(p, (-0.66, 0.12, 0.7675), (0, -1, 0), (0, 0, 1), 0.12, 0.065, scr_m, bp, lift=0.001)
    return p.obj("med_desk", col)


def med_sink(col, M):
    """A clinical hand-wash station (HTM 64 type, wall-hung, the back at y = 0): a white IPS panel
    with a ceramic basin without plug or overflow, an elbow-lever mixer, a soap and an alcohol-gel
    dispenser, a paper-towel dispenser, the NHS hand-hygiene poster, a pedal bin for clinical waste
    (yellow bag lid) and one for general waste beneath. Front +y."""
    cv = Canvas("Med_Sink", 1024, 1024, col="#f1f1ee", rough=0.35)
    ips = cv.alloc(1.0, 1.9, 300)
    ips.fill(col="#f2f2ef")
    ips.mottle(0.02, 4, 3)
    for x in (0.0, 1.0):
        ips.rect(max(x - 0.004, 0), 0, min(x + 0.004, 1.0), 1.9, col="#d0d0cc")
    poster = cv.alloc(0.3, 0.42, 900)
    poster.fill(col="#ffffff")
    poster.rect(0, 0.37, 0.3, 0.42, col="#005eb8")
    poster.text("HAND HYGIENE", 0.15, 0.385, 0.022, weight="black", align="c", col="#ffffff")
    for k in range(9):
        cx, cy = 0.055 + (k % 3) * 0.095, 0.3 - (k // 3) * 0.1
        poster.rect(cx - 0.04, cy - 0.04, cx + 0.04, cy + 0.035, r=0.006, col="#e8f0fa")
        poster.poly([(cx - 0.025, cy - 0.02), (cx + 0.02, cy - 0.025), (cx + 0.028, cy + 0.01), (cx - 0.01, cy + 0.025)], col="#f0c8a8")
        poster.text(str(k + 1), cx - 0.035, cy + 0.02, 0.012, weight="bold", col="#005eb8")
    poster.text("40 - 60 seconds", 0.15, 0.012, 0.014, weight="bold", align="c", col="#003087")
    lab = cv.alloc(0.12, 0.05, 1500)
    lab.fill(col="#ffffff")
    lab.text("ALCOHOL GEL", 0.06, 0.02, 0.011, weight="black", align="c", col="#005eb8")
    mat = cv.material("Med_SinkPanel")
    ceramic = _paint("Med_Ceramic", "#f6f6f2", 0.12)
    p = PM(0.004)
    i0 = len(p.F)
    p.box((-0.5, 0.0, 0.0), (0.5, 0.012, 1.9), mat)
    project(p, i0, ips, (-0.5, 0, 0), (1, 0, 0), (0, 0, 1))
    # basin: a lathed bowl on the panel
    p.r = 0.0
    prof = [(0.0, 0.79), (0.15, 0.79), (0.2, 0.83), (0.22, 0.86), (0.235, 0.865), (0.24, 0.85)]
    p.lathe(prof, ceramic, c=(0.0, 0.25, 0.0), sides=28)
    p.r = 0.02
    p.box((-0.25, 0.012, 0.8), (0.25, 0.26, 0.865), ceramic)
    p.r = 0.0
    p.lathe([(0.0, 0.795), (0.02, 0.795)], M["chrome"], c=(0.0, 0.26, 0.0), sides=14)
    # elbow lever mixer on the panel, spout over the bowl
    p.cyl((0.0, 0.012, 1.0), (0.0, 0.06, 1.0), 0.03, M["chrome"], 16)
    _tube(p, [(0.0, 0.05, 1.0), (0.0, 0.17, 0.99), (0.0, 0.2, 0.93)], 0.011, M["chrome"], 10, bend=0.04)
    p.r = 0.003
    p.box((-0.18, 0.04, 0.99), (0.0, 0.055, 1.01), M["chrome"])
    # dispensers: soap left, gel right, towels above
    for x, txt in ((-0.32, "soap"), (0.32, "gel")):
        p.r = 0.01
        p.box((x - 0.06, 0.012, 1.02), (x + 0.06, 0.12, 1.28), _white("Med_Dispenser"))
        p.box((x - 0.03, 0.08, 0.99), (x + 0.03, 0.115, 1.02), M["grey"])
        if txt == "gel":
            decal(p, (x, 0.12, 1.2), (0, 1, 0), (0, 0, 1), 0.11, 0.045, mat, lab, lift=0.001)
    p.r = 0.015
    p.box((-0.16, 0.012, 1.35), (0.16, 0.14, 1.72), _white("Med_Towels"))
    p.r = 0.002
    p.box((-0.11, 0.1, 1.35), (0.11, 0.142, 1.37), _white("Med_Paper"))
    decal(p, (0.36 + 0.05, 0.012, 1.55), (0, 1, 0), (0, 0, 1), 0.3, 0.42, mat, poster, lift=0.002)
    # pedal bins: clinical (yellow lid) and general
    for x, lid in ((-0.32, "#e8c21a"), (-0.02, "#2b2d30")):
        p.r = 0.02
        p.box((x - 0.15, 0.32, 0.0), (x + 0.15, 0.62, 0.62), _white("Med_Bin"))
        p.r = 0.01
        p.box((x - 0.155, 0.31, 0.62), (x + 0.155, 0.63, 0.66), _plastic(f"Med_BinLid{lid[1:]}", lid, 0.45))
        p.box((x - 0.06, 0.62, 0.02), (x + 0.06, 0.7, 0.05), M["grey"])
    return p.obj("med_sink", col)


def med_cabinet(col, M):
    """A medical supply cabinet (1.0 x 0.45 x 2.0 m, white steel): glazed upper doors over shelves
    of labelled supplies (dressings, tape rolls, boxed gloves in three sizes, ice packs), solid
    lower doors with a keyed lock, and a laminated stock list on the side. Front +y."""
    cv = Canvas("Med_Cabinet", 1024, 1024, col="#f0f0ec", rough=0.4)
    inside = cv.alloc(0.96, 1.0, 500)
    inside.fill(col="#e6e6e2")
    rng = np.random.default_rng(5)
    for s, z in enumerate((0.04, 0.37, 0.7)):
        x = 0.02
        while x < 0.92:
            w = float(rng.uniform(0.08, 0.2))
            h = float(rng.uniform(0.12, 0.27))
            c = str(rng.choice(["#3a78c0", "#e8e4dc", "#5a9a52", "#d04a3a", "#8a5ab0", "#f0c030", "#ffffff"]))
            inside.rect(x, z, min(x + w, 0.94), z + h, r=0.004, col=c)
            inside.rect(x + 0.01, z + h * 0.55, min(x + w, 0.94) - 0.01, z + h * 0.75, col="#ffffff")
            inside.text(str(rng.choice(["DRESSINGS", "GLOVES M", "TAPE", "ICE PACK", "GAUZE", "WIPES", "GLOVES L"])), x + 0.012, z + h * 0.6, 0.012, weight="bold", col="#202020", max_w=max(w - 0.02, 0.03))
            x += w + 0.01
        inside.rect(0, z - 0.02, 0.96, z, col="#c8c8c4")
    inside.field(lambda X, Y: np.clip(1 - Y / 1.0, 0, 1) * 0.25, col="#606060")
    sheet = cv.alloc(0.21, 0.3, 900)
    sheet.fill(col="#fbfbf6")
    sheet.text("STOCK CHECK", 0.105, 0.27, 0.016, weight="black", align="c", col="#003087")
    for i in range(14):
        sheet.rect(0.015, 0.24 - i * 0.016, 0.195, 0.2405 - i * 0.016, col="#a8b0bc")
        sheet.text("x", 0.18, 0.243 - i * 0.016, 0.009, col="#1a6a2a")
    mat = cv.material("Med_CabinetInside")
    steel = _white("Med_CabinetSteel")
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.5, -0.22, 0.05), (0.5, 0.22, 2.0), steel)
    p.box((-0.48, -0.2, 0.0), (0.48, 0.2, 0.05), M["black"])
    p.r = 0.0
    decal(p, (0.0, 0.22, 1.49), (0, 1, 0), (0, 0, 1), 0.92, 0.92, mat, inside, lift=0.002)
    for x0, x1 in ((-0.49, -0.005), (0.005, 0.49)):
        p.r = 0.004
        for (a, b, c_, d) in ((x0, 1.0, x0 + 0.03, 1.98), (x1 - 0.03, 1.0, x1, 1.98), (x0, 1.0, x1, 1.03), (x0, 1.95, x1, 1.98)):
            p.box((a, 0.22, b), (c_, 0.24, d), steel)
        p.r = 0.0
        p.face([(x0 + 0.03, 0.235, 1.03), (x1 - 0.03, 0.235, 1.03), (x1 - 0.03, 0.235, 1.95), (x0 + 0.03, 0.235, 1.95)], M["clear"])
        p.r = 0.004
        p.box((x0 + 0.005, 0.22, 0.07), (x1 - 0.005, 0.235, 0.97), steel)
        hx = x1 - 0.06 if x0 < 0 else x0 + 0.06
        p.box((hx - 0.01, 0.235, 0.7), (hx + 0.01, 0.255, 0.9), M["alu"])
    p.cyl((0.0, 0.235, 0.82), (0.0, 0.245, 0.82), 0.012, M["chrome"], 12)
    decal(p, (0.5, 0.0, 1.3), (1, 0, 0), (0, 0, 1), 0.21, 0.3, mat, sheet, lift=0.002)
    return p.obj("med_cabinet", col)


def med_trolley(col, M):
    """A two-tier stainless dressing trolley (0.6 x 0.45 m, 0.9 m high) on four castors, with a
    push handle, a dressing pack, tape, a kidney dish, scissors, an ice pack and a box of gloves on
    top and spare packs below. Front +y."""
    p = PM(0.004)
    for z in (0.3, 0.86):
        p.r = 0.006
        p.box((-0.3, -0.22, z), (0.3, 0.22, z + 0.02), M["alu"])
        p.r = 0.0
        for (a, b) in (((-0.3, -0.22), (0.3, -0.22)), ((0.3, -0.22), (0.3, 0.22)), ((0.3, 0.22), (-0.3, 0.22)), ((-0.3, 0.22), (-0.3, -0.22))):
            _tube(p, [(a[0], a[1], z + 0.03), (b[0], b[1], z + 0.03)], 0.006, M["alu"], 6)
    for x in (-0.28, 0.28):
        for y in (-0.2, 0.2):
            _tube(p, [(x, y, 0.1), (x, y, 0.88)], 0.012, M["alu"], 8)
            _castor(p, (x, y, 0.1), M, 0.04)
    _tube(p, [(-0.28, -0.2, 0.88), (-0.36, -0.2, 0.95), (-0.36, 0.2, 0.95), (-0.28, 0.2, 0.88)], 0.012, M["alu"], 8, bend=0.04)
    p.r = 0.004
    p.box((-0.2, -0.1, 0.88), (0.0, 0.1, 0.93), _white("Med_Pack"))
    p.box((0.05, 0.02, 0.88), (0.25, 0.15, 0.97), _plastic("Med_Gloves", "#3a78c0", 0.5))
    p.r = 0.0
    p.lathe([(0.0, 0.88), (0.06, 0.88), (0.07, 0.9), (0.068, 0.91)], M["chrome"], c=(0.15, -0.12, 0.0), sides=18)
    p.cyl((-0.25, 0.14, 0.885), (-0.25, 0.14, 0.905), 0.035, _white("Med_Tape"), 16)
    p.r = 0.006
    p.box((0.02, -0.18, 0.88), (0.14, -0.1, 0.91), _plastic("Med_IcePack", "#7ab0e0", 0.3))
    p.box((-0.25, -0.15, 0.32), (0.25, 0.15, 0.42), _white("Med_Pack"))
    return p.obj("med_trolley", col)


def med_wall_station(col, M):
    """The wall emergency station (the back at y = 0): a green AED cabinet with the ILCOR heart and
    bolt and a window onto the defibrillator, a first-aid cabinet with the white cross, a yellow
    sharps bin on its bracket, a three-box glove dispenser and an apron roll. Front +y."""
    cv = Canvas("Med_Wall", 1024, 1024, col="#ffffff", rough=0.4)
    aed = cv.alloc(0.42, 0.5, 900)
    aed.fill(col="#00843d")
    aed.rect(0.03, 0.12, 0.39, 0.4, r=0.02, col="#0a3d1e")
    aed.text("AED", 0.21, 0.43, 0.04, weight="black", align="c", col="#ffffff")
    aed.poly([(0.21, 0.045), (0.14, 0.09), (0.15, 0.115), (0.21, 0.09), (0.27, 0.115), (0.28, 0.09)], col="#ffffff")
    aed.text("DEFIBRILLATOR", 0.21, 0.015, 0.022, weight="bold", align="c", col="#ffffff")
    fa = cv.alloc(0.4, 0.3, 900)
    fa.fill(col="#00843d")
    fa.rect(0.17, 0.08, 0.23, 0.26, col="#ffffff")
    fa.rect(0.11, 0.14, 0.29, 0.2, col="#ffffff")
    fa.text("FIRST AID", 0.2, 0.025, 0.03, weight="black", align="c", col="#ffffff")
    sh = cv.alloc(0.2, 0.25, 900)
    sh.fill(col="#f5c400")
    sh.text("SHARPS", 0.1, 0.12, 0.028, weight="black", align="c", col="#1a1a1a")
    sh.rect(0, 0.2, 0.2, 0.25, col="#e8b400")
    mat = cv.material("Med_WallStation")
    green = _paint("Med_AEDGreen", "#00843d", 0.4)
    p = PM(0.006)
    # AED cabinet
    p.box((-0.6, 0.0, 1.2), (-0.18, 0.18, 1.7), green)
    decal(p, (-0.39, 0.18, 1.45), (0, 1, 0), (0, 0, 1), 0.42, 0.5, mat, aed, lift=0.001)
    p.r = 0.0
    p.face([(-0.57, 0.185, 1.33), (-0.21, 0.185, 1.33), (-0.21, 0.185, 1.6), (-0.57, 0.185, 1.6)], M["clear"])
    p.r = 0.01
    p.box((-0.5, 0.06, 1.36), (-0.28, 0.16, 1.58), _plastic("Med_AEDUnit", "#e9b81c", 0.4))
    # first-aid cabinet
    p.r = 0.006
    p.box((-0.1, 0.0, 1.35), (0.3, 0.14, 1.65), green)
    decal(p, (0.1, 0.14, 1.5), (0, 1, 0), (0, 0, 1), 0.4, 0.3, mat, fa, lift=0.001)
    # sharps bin on its bracket
    p.r = 0.0
    p.box((0.4, 0.0, 1.08), (0.56, 0.015, 1.12), M["grey"])
    p.r = 0.015
    p.box((0.38, 0.015, 0.95), (0.58, 0.2, 1.2), _plastic("Med_Sharps", "#f5c400", 0.4))
    decal(p, (0.48, 0.2, 1.08), (0, 1, 0), (0, 0, 1), 0.2, 0.25, mat, sh, lift=0.001)
    # glove dispenser (three boxes) and apron roll
    p.r = 0.004
    p.box((0.68, 0.0, 1.1), (0.98, 0.015, 1.55), M["alu"])
    for k, c in enumerate(("#3a78c0", "#7a5ab0", "#2a9a7a")):
        z = 1.12 + k * 0.145
        p.box((0.7, 0.015, z), (0.96, 0.1, z + 0.13), _plastic(f"Med_GloveBox{k}", c, 0.45))
        p.box((0.68, 0.015, z - 0.01), (0.98, 0.105, z), M["alu"])
        p.box((0.78, 0.1, z + 0.05), (0.88, 0.104, z + 0.09), _white("Med_GloveTissue"))
    p.cyl((0.7, 0.07, 1.7), (0.96, 0.07, 1.7), 0.05, _plastic("Med_Apron", "#f2f4f6", 0.6), 14)
    return p.obj("med_wall_station", col)


def ice_machine(col, M):
    """An under-counter ice maker (Hoshizaki class, 0.63 x 0.6 x 0.86 m, stainless): a hinged bin
    door with a recessed pull, a louvred grille, adjustable feet, and a scoop holder on the side. Front +y."""
    cv = Canvas("Med_Ice", 512, 512, col="#c9cccf", rough=0.3, metal=0.7)
    front = cv.alloc(0.63, 0.86, 500)
    front.fill(col="#c4c7ca", metal=0.8, rough=0.35)
    front.field(lambda X, Y: noise(X.shape, (2, 160), 2, 2), mul=0.96)
    front.rect(0.04, 0.32, 0.59, 0.8, r=0.01, col="#babdc0", dh=0.002, metal=0.8)
    front.rect(0.2, 0.74, 0.43, 0.78, r=0.015, col="#2a2c2e", dh=-0.004)
    for y in np.arange(0.06, 0.26, 0.022):
        front.rect(0.05, y, 0.58, y + 0.01, col="#2a2c2e", dh=-0.003)
    front.text("ICE", 0.315, 0.66, 0.035, weight="black", align="c", col="#2f6fb0", metal=0.0)
    mat = cv.material("Med_IceMachine")
    p = PM(0.006)
    i0 = len(p.F)
    p.box((-0.315, -0.3, 0.05), (0.315, 0.3, 0.86), mat)
    project(p, i0, front, (0.315, 0.3, 0.05), (-1, 0, 0), (0, 0, 1))
    for x in (-0.27, 0.27):
        for y in (-0.25, 0.25):
            p.cyl((x, y, 0.05), (x, y, 0.0), 0.02, M["grey"], 10)
    p.r = 0.003
    p.box((0.315, -0.08, 0.5), (0.335, 0.08, 0.7), M["alu"])
    return p.obj("ice_machine", col)


def med_curtain(col, M):
    """A cubicle privacy curtain (the track at 2.55 m, a 2.4 m run with a 1.6 m return round a
    couch bay): an aluminium ceiling track on drop rods, a disposable antimicrobial curtain in a
    pastel print with a mesh top band, drawn back and bunched in folds at the open end. The bay's
    open side faces +y."""
    cv = Canvas("Med_Curtain", 512, 1024, col="#ffffff", rough=0.8)
    fab = cv.alloc(0.6, 2.2, 400)
    fab.fill(col="#dfe8ef")
    fab.field(lambda X, Y: (np.sin(X * 2 * math.pi / 0.15) > 0.6) * 0.5, col="#b9cbd9")
    fab.field(lambda X, Y: (np.sin(Y * 2 * math.pi / 0.15 + 1.0) > 0.85) * 0.4, col="#9fb8cc")
    fab.rect(0, 1.9, 0.6, 2.2, col="#f4f4f4")
    fab.field(lambda X, Y: (Y > 1.9) * ((np.sin(X * 400) > 0.2) & (np.sin(Y * 400) > 0.2)) * 0.6, col="#c8c8c8")
    fab.field(lambda X, Y: np.clip(1 - Y / 0.6, 0, 1) * 0.12, col="#8a9aa8")
    mat = cv.material("Med_CurtainFabric", normal=0.0)
    p = PM(0.0)
    zt = 2.55
    track = [(1.2, 0.8), (1.2, -0.8), (-1.2, -0.8), (-1.2, 0.8)]
    p.r = 0.002
    for a, b in zip(track[:-1], track[1:]):
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        p.box((x0 - 0.012, y0 - 0.012, zt), (x1 + 0.012, y1 + 0.012, zt + 0.025), M["alu"])
    for x, y in ((1.2, 0.0), (0.0, -0.8), (-1.2, 0.0)):
        p.cyl((x, y, zt + 0.025), (x, y, 2.75), 0.006, M["alu"], 8)
    # the curtain: pleats along the back run, bunched folds at both ends (open front)
    p.r = 0.0

    def pleats(a, b, n, depth, z0=0.3):
        ax, ay = a
        bx_, by_ = b
        L = math.hypot(bx_ - ax, by_ - ay)
        ux, uy = (bx_ - ax) / L, (by_ - ay) / L
        nx, ny = -uy, ux
        pts = [(ax + ux * L * k / n + nx * depth * (1 if k % 2 else -1), ay + uy * L * k / n + ny * depth * (1 if k % 2 else -1)) for k in range(n + 1)]
        for k in range(n):
            (x0, y0), (x1, y1) = pts[k], pts[k + 1]
            u0, u1 = 0.6 * k / n, 0.6 * (k + 1) / n
            p.face([(x0, y0, z0), (x1, y1, z0), (x1, y1, zt - 0.02), (x0, y0, zt - 0.02)], mat,
                   [fab.uv(u0, 0), fab.uv(u1, 0), fab.uv(u1, 2.2), fab.uv(u0, 2.2)])
    pleats((-1.2, -0.8), (1.2, -0.8), 40, 0.035)
    pleats((-1.2, 0.8), (-1.2, 0.35), 12, 0.05)
    pleats((1.2, 0.8), (1.2, 0.45), 10, 0.05)
    pleats((-1.2, 0.35), (-1.2, -0.8), 12, 0.03)
    pleats((1.2, 0.45), (1.2, -0.8), 12, 0.03)
    return p.obj("med_curtain", col)


# -- the rehab gym --------------------------------------------------------------------------------

def treadmill(col, M):
    """A commercial treadmill (Technogym Excite / Woodway class, 2.15 x 0.9 m): a deck in silver
    side covers with aluminium foot rails either side of the black belt, the rear roller in its end
    caps and levelling feet, the motor hood at the front in dark grey, two silver uprights to a
    console with a lit display, speed and incline keys and the emergency-stop magnet on its red
    cord, handrails running back along both sides and a front grab bar. Front +y (the runner
    faces +y)."""
    cv = Canvas("Med_Treadmill", 512, 512, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.34, 0.18, 900)
    _screen(scr, "console", 3)
    keys = cv.alloc(0.62, 0.1, 700)
    keys.fill(col="#1c1e21")
    for i in range(6):
        for s_, c in ((0.04, "#3a7ad0"), (0.5, "#e0a030")):
            keys.rect(s_ + i * 0.014, 0.02, s_ + i * 0.014 + 0.011, 0.08, r=0.002, col=c, emit=c)
    keys.text("SPEED", 0.04, 0.085, 0.01, weight="bold", col="#c8ccd2", emit="#c8ccd2")
    keys.text("INCLINE", 0.5, 0.085, 0.01, weight="bold", col="#c8ccd2", emit="#c8ccd2")
    smat = cv.material("Med_TreadScreen", emission=1.2)
    silver = _plastic("Med_GymSilver", "#b9bdc2", 0.35)
    dark = _plastic("Med_GymBody", "#2b2d31", 0.45)
    belt = _plastic("Med_Belt", "#161718", 0.85)
    p = PM(0.008)
    # deck: silver side covers, foot rails, the belt, rear roller caps, feet
    p.r = 0.01
    p.box((-0.45, -1.05, 0.06), (-0.29, 0.62, 0.2), silver)
    p.box((0.29, -1.05, 0.06), (0.45, 0.62, 0.2), silver)
    p.box((-0.29, -1.0, 0.08), (0.29, 0.6, 0.17), dark)
    p.r = 0.003
    p.box((-0.27, -1.02, 0.17), (0.27, 0.6, 0.19), belt)
    for k in range(32):
        y = -0.98 + k * 0.05
        p.box((-0.27, y, 0.19), (0.27, y + 0.003, 0.191), _plastic("Med_BeltSlat", "#26282a", 0.8))
    for x in (-0.37, 0.37):
        p.box((x - 0.07, -1.0, 0.2), (x + 0.07, 0.6, 0.215), M["alu"])
        for y in np.arange(-0.95, 0.6, 0.08):
            p.box((x - 0.06, y, 0.215), (x + 0.06, y + 0.012, 0.218), _plastic("Med_RailGrip", "#5a5d61", 0.7))
    p.r = 0.0
    for s_ in (-1, 1):
        p.cyl((s_ * 0.45, -1.0, 0.12), (s_ * 0.46, -1.0, 0.12), 0.05, dark, 18)
    p.cyl((-0.27, -1.0, 0.12), (0.27, -1.0, 0.12), 0.045, M["grey"], 18)
    for x, y in ((-0.4, -1.0), (0.4, -1.0), (-0.4, 0.85), (0.4, 0.85)):
        p.cyl((x, y, 0.0), (x, y, 0.06), 0.03, M["rubber"], 12)
    # motor hood
    p.r = 0.04
    p.box((-0.45, 0.6, 0.05), (0.45, 0.95, 0.36), dark)
    p.r = 0.0
    p.box((-0.3, 0.95, 0.12), (0.3, 0.951, 0.3), _plastic("Med_HoodGrille", "#121314", 0.7))
    # uprights, handrails along the sides, the front grab bar
    for x in (-0.38, 0.38):
        _tube(p, [(x, 0.72, 0.34), (x, 0.82, 1.22)], 0.035, silver, 14)
        _tube(p, [(x, 0.8, 1.1), (x * 1.05, 0.55, 1.05), (x * 1.05, 0.15, 1.0), (x, 0.05, 0.98)], 0.017, silver, 10, bend=0.08)
        p.cyl((x * 1.05, 0.15, 1.0), (x * 1.05, 0.5, 1.045), 0.022, _plastic("Med_Grip", "#1a1b1c", 0.8), 12)
    _tube(p, [(-0.32, 0.72, 1.12), (-0.3, 0.68, 1.16), (0.3, 0.68, 1.16), (0.32, 0.72, 1.12)], 0.016, silver, 10, bend=0.04)
    # console: a sloped black panel with the screen and the keys
    p.r = 0.03
    p.obox((0.0, 0.86, 1.33), (0.86, 0.26, 0.14), dark, pitch=math.radians(35))
    p.r = 0.0
    nrm, up = (0, -0.82, 0.57), (0, 0.57, 0.82)
    decal(p, (0.0, 0.79, 1.4), nrm, up, 0.34, 0.18, smat, scr, lift=0.004)
    decal(p, (0.0, 0.75, 1.3), nrm, up, 0.62, 0.1, smat, keys, lift=0.004)
    # the safety key on its cord
    p.r = 0.006
    p.box((0.3, 0.73, 1.32), (0.36, 0.78, 1.35), M["red"])
    _tube(p, [(0.33, 0.74, 1.32), (0.34, 0.66, 1.2), (0.32, 0.62, 1.12)], 0.003, M["red"], 5)
    return p.obj("treadmill", col)


def wattbike(col, M):
    """An air-and-magnet cycle ergometer (Wattbike Pro class, 1.3 x 0.6 m): a T-shaped steel base
    with transport wheels, a fan cage at the front, a crank with clipless pedals, an adjustable
    seat post and saddle, drop handlebars with a monitor. Front +y (the rider faces +y)."""
    cv = Canvas("Med_Wattbike", 256, 256, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.14, 0.09, 1200)
    _screen(scr, "console", 5)
    smat = cv.material("Med_BikeScreen", emission=1.2)
    p = PM(0.006)
    frame = _paint("Med_BikeFrame", "#2a2c30", 0.4)
    p.box((-0.3, 0.38, 0.0), (0.3, 0.46, 0.06), frame)
    p.box((-0.25, -0.52, 0.0), (0.25, -0.44, 0.06), frame)
    p.box((-0.04, -0.5, 0.02), (0.04, 0.42, 0.09), frame)
    p.r = 0.0
    _tube(p, [(0, 0.38, 0.06), (0, 0.32, 0.45)], 0.04, frame, 12)
    _tube(p, [(0, -0.3, 0.06), (0, -0.1, 0.5), (0, 0.05, 0.6)], 0.04, frame, 12)
    _tube(p, [(0, -0.1, 0.5), (0, -0.2, 0.85)], 0.03, M["alu"], 12)
    p.r = 0.03
    p.box((-0.08, -0.32, 0.86), (0.08, -0.06, 0.92), M["black"])
    p.r = 0.0
    _tube(p, [(0, 0.32, 0.45), (0, 0.42, 0.95)], 0.028, M["alu"], 10)
    _tube(p, [(-0.22, 0.48, 0.82), (-0.2, 0.42, 0.98), (0.2, 0.42, 0.98), (0.22, 0.48, 0.82)], 0.014, M["black"], 8, bend=0.05)
    p.r = 0.01
    p.obox((0, 0.45, 1.05), (0.17, 0.035, 0.12), M["black"], pitch=math.radians(-37))
    p.r = 0.0
    decal(p, (0, 0.437, 1.06), (0, -0.6, 0.8), (0, 0.8, 0.6), 0.14, 0.09, smat, scr, lift=0.004)
    # fan cage and crank
    p.r = 0.0
    p.cyl((-0.12, 0.25, 0.32), (0.12, 0.25, 0.32), 0.27, _plastic("Med_FanCage", "#3a3c40", 0.5), 28)
    p.cyl((-0.13, 0.05, 0.3), (0.13, 0.05, 0.3), 0.04, M["alu"], 14)
    for s_, ph in ((-1, 0.0), (1, math.pi)):
        x = s_ * 0.1
        e = (x, 0.05 + 0.17 * math.cos(ph + 0.6), 0.3 + 0.17 * math.sin(ph + 0.6))
        _tube(p, [(x, 0.05, 0.3), e], 0.012, M["alu"], 8)
        p.box((x - 0.05 * (1 if s_ > 0 else 0) - 0.02, e[1] - 0.05, e[2] - 0.015), (x + 0.05 * (1 if s_ < 0 else 0) + 0.02, e[1] + 0.05, e[2] + 0.01), M["black"])
    for x in (-0.24, 0.24):
        tyre(p, (x, 0.5, 0.04), 0.035, 0.03, 0.02, M["black"], M["grey"], sides=12, axis="x")
    return p.obj("wattbike", col)


def squat_rack(col, M):
    """A half power rack (Eleiko class, 1.4 x 1.3 x 2.3 m) on a platform: 75 mm uprights with
    numbered hole rows, J-hooks holding a loaded Olympic bar (two red 25 kg and two blue 20 kg
    bumper plates a side, collars), safety spotter arms, plate storage horns, a pull-up bar, an
    adjustable bench in front. Front +y."""
    cv = Canvas("Med_Rack", 256, 1024, col="#202225", rough=0.45)
    up = cv.alloc(0.075, 2.3, 300)
    up.fill(col="#1e2023")
    for k in range(40):
        y = 0.35 + k * 0.05
        up.circle(0.0375, y, 0.012, col="#050505", dh=-0.004)
        up.text(str(k + 1), 0.0375, y + 0.015, 0.012, weight="bold", align="c", col="#d8d8d8")
    rmat = cv.material("Med_RackUpright")
    p = PM(0.004)
    plat = _plastic("Med_Platform", "#1a1a1a", 0.8)
    p.box((-1.2, -1.0, 0.0), (1.2, 0.8, 0.04), plat)
    p.box((-0.5, -0.9, 0.04), (0.5, 0.7, 0.05), M["timber"])
    steel = _paint("Med_RackSteel", "#1e2023", 0.45)
    for x in (-0.6, 0.6):
        for y in (-0.6, 0.4):
            i0 = len(p.F)
            p.box((x - 0.0375, y - 0.0375, 0.04), (x + 0.0375, y + 0.0375, 2.3), rmat)
            project(p, i0, up, (x - 0.0375, y + 0.0375, 0.04), (1, 0, 0), (0, 0, 1))
        p.box((x - 0.0375, -0.6, 2.25), (x + 0.0375, 0.4, 2.3), steel)
        p.box((x - 0.0375, -0.6, 0.04), (x + 0.0375, 0.4, 0.1), steel)
        # J-hooks and spotter arms
        p.box((x - 0.04, 0.4375, 1.38), (x + 0.04, 0.55, 1.42), steel)
        p.box((x - 0.04, 0.51, 1.42), (x + 0.04, 0.55, 1.48), steel)
        p.box((x - 0.03, 0.44, 0.88), (x + 0.03, 1.1, 0.95), steel)
        # plate horns at the back
        p.r = 0.0
        p.cyl((x, -0.64, 0.5), (x, -0.9, 0.5), 0.025, steel, 12)
        p.r = 0.004
    p.r = 0.0
    p.cyl((-0.6, -0.1, 2.32), (0.6, -0.1, 2.32), 0.016, M["chrome"], 12)
    # the bar and its plates
    bz = 1.47
    p.cyl((-1.1, 0.53, bz), (1.1, 0.53, bz), 0.014, M["chrome"], 12)
    for s_ in (-1, 1):
        p.cyl((s_ * 0.78, 0.53, bz), (s_ * 1.1, 0.53, bz), 0.025, M["chrome"], 14)
        x = s_ * 0.8
        for k, (c, th) in enumerate((("#c4161c", 0.07), ("#c4161c", 0.07), ("#2457a6", 0.06), ("#2457a6", 0.06))):
            x1 = x + s_ * th
            p.cyl((x, 0.53, bz), (x1, 0.53, bz), 0.225, _plastic(f"Med_Bumper{c[1:]}", c, 0.6), 28)
            x = x1
        p.cyl((x, 0.53, bz), (x + s_ * 0.03, 0.53, bz), 0.04, M["black"], 14)
    # spare plates on the horns, the bench in front
    for s_ in (-1, 1):
        p.cyl((s_ * 0.6, -0.66, 0.5), (s_ * 0.6, -0.74, 0.5), 0.225, _plastic("Med_Bumperc4161c", "#c4161c", 0.6), 28)
    p.r = 0.03
    p.box((-0.15, 1.0, 0.36), (0.15, 2.1, 0.44), M["black"])
    p.r = 0.004
    for y in (1.1, 2.0):
        p.box((-0.2, y - 0.04, 0.0), (0.2, y + 0.04, 0.05), steel)
        p.box((-0.03, y - 0.03, 0.05), (0.03, y + 0.03, 0.36), steel)
    return p.obj("squat_rack", col)


def dumbbell_rack(col, M):
    """A two-tier dumbbell rack (1.8 m) holding seven pairs of rubber hex dumbbells from 4 to 16 kg
    and seven heavier pairs to 30 kg, each lying front to back in its cradle: two hex heads on a
    knurled chrome handle, the weight in white on the end caps. Front +y."""
    cv = Canvas("Med_HexLabels", 512, 256, col="#1b1b1c", rough=0.7)
    labs = {}
    for w in range(4, 34, 2):
        r = cv.alloc(0.05, 0.05, 1200)
        r.fill(col="#1b1b1c")
        r.text(str(w), 0.025, 0.012, 0.024, weight="black", align="c", col="#f2f2f0")
        labs[w] = r
    lmat = cv.material("Med_HexLabelMat")
    steel = _paint("Med_RackSteel", "#1e2023", 0.45)
    rub = _plastic("Med_HexRubber", "#1b1b1c", 0.7)
    p = PM(0.004)
    for x in (-0.86, 0.86):
        p.box((x - 0.03, -0.3, 0.0), (x + 0.03, 0.3, 0.04), steel)
        _tube(p, [(x, -0.18, 0.04), (x, -0.18, 0.86)], 0.025, steel, 8)
        _tube(p, [(x, 0.14, 0.04), (x, 0.14, 0.56)], 0.025, steel, 8)
    for yc, z in ((-0.14, 0.82), (0.16, 0.52)):
        p.r = 0.004
        p.box((-0.86, yc - 0.17, z - 0.02), (0.86, yc - 0.15, z + 0.03), steel)
        p.box((-0.86, yc + 0.15, z - 0.02), (0.86, yc + 0.17, z + 0.03), steel)
        p.box((-0.86, yc - 0.17, z - 0.03), (0.86, yc + 0.17, z - 0.02), steel)
    k = 0
    for row, (yc, z) in enumerate(((-0.14, 0.82), (0.16, 0.52))):
        n = 7
        for i in range(n):
            w = 4 + 2 * k
            hr = 0.042 + 0.0016 * w
            length = 0.27 + 0.004 * w
            x = -0.74 + i * (1.48 / (n - 1))
            for pair in (-1, 1):
                xx = x + pair * (hr + 0.006)
                zc = z + hr - 0.01
                for s_ in (-1, 1):
                    y0 = yc + s_ * (length / 2 - 0.045)
                    p.r = 0.004
                    p.cyl((xx, y0 - 0.045, zc), (xx, y0 + 0.045, zc), hr, rub, 6)
                    p.r = 0.0
                    decal(p, (xx, yc + s_ * length / 2 + s_ * 0.001, zc), (0, s_, 0), (0, 0, 1), hr * 1.2, hr * 1.2, lmat, labs[w], lift=0.001)
                p.r = 0.0
                p.cyl((xx, yc - length / 2 + 0.09, zc), (xx, yc + length / 2 - 0.09, zc), 0.016, M["chrome"], 12)
            k += 1
    return p.obj("dumbbell_rack", col)


def isokinetic(col, M):
    """An isokinetic dynamometer (Biodex System 4 class, 2.4 x 1.6 m): a padded chair on a rotating
    base that slides on a floor rail, the dynamometer head on a height-adjustable column with a
    knee attachment, and the control cart with its monitor showing a torque curve. Front +y."""
    cv = Canvas("Med_Biodex", 512, 512, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.5, 0.3, 700)
    _screen(scr, "graph", 7)
    smat = cv.material("Med_BiodexScreen", emission=1.2)
    cv2, reg = _cushion_canvas("Med_BiodexVinyl", 0.6, 0.6, seed=17)
    up = cv2.tint_material("TINT_upholstery_biodex", "#1f3f6a", rough=0.55)
    grey = _plastic("Med_BiodexGrey", "#c9ccd0", 0.4)
    p = PM(0.006)
    p.box((-1.2, -0.12, 0.0), (0.4, 0.12, 0.08), grey)
    p.r = 0.02
    p.box((-0.6, -0.35, 0.08), (0.0, 0.35, 0.45), grey)
    p.r = 0.03
    i0 = len(p.F)
    p.box((-0.55, -0.28, 0.45), (0.0, 0.28, 0.55), up)
    p.obox((-0.6, 0.0, 0.85), (0.1, 0.56, 0.6), up, roll=math.radians(10))
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-0.7, -0.3, 0.4), (0.1, 0.3, 1.2))
    p.r = 0.006
    for y in (-0.3, 0.3):
        p.box((-0.5, y - 0.02, 0.55), (-0.1, y + 0.02, 0.7), grey)
    # the dynamometer on its column
    p.r = 0.02
    p.box((0.4, -0.25, 0.0), (0.75, 0.25, 0.1), grey)
    p.box((0.5, -0.1, 0.1), (0.66, 0.1, 1.0), grey)
    p.r = 0.03
    p.box((0.32, -0.18, 0.62), (0.62, 0.18, 0.86), _plastic("Med_BiodexHead", "#3a3d42", 0.4))
    p.r = 0.0
    p.cyl((0.32, 0.0, 0.74), (0.22, 0.0, 0.74), 0.06, M["chrome"], 16)
    p.box((0.18, -0.03, 0.3), (0.24, 0.03, 0.74), M["alu"])
    p.r = 0.02
    p.box((0.14, -0.08, 0.28), (0.28, 0.08, 0.36), M["black"])
    # control cart
    p.r = 0.006
    p.box((0.9, 0.4, 0.05), (1.4, 0.8, 0.9), grey)
    for x in (0.95, 1.35):
        for y in (0.45, 0.75):
            _castor(p, (x, y, 0.05), M, 0.03)
    _monitor(p, (1.15, 0.55, 0.9), math.radians(200), 0.5, 0.3, smat, scr, M)
    return p.obj("isokinetic", col)


def alter_g(col, M):
    """An anti-gravity treadmill (AlterG Pro class, 2.1 x 0.86 x 1.7 m): a treadmill base whose belt
    shows through the translucent pressurised bag, the bag stiffened by seam ribs and held by a
    hoop at the cockpit, the cockpit ring at waist height with its neoprene skirt, an
    adjustable-height frame with grab bars, and the touchscreen console on its arm. Front +y."""
    cv = Canvas("Med_AlterG", 512, 512, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.3, 0.2, 900)
    _screen(scr, "console", 9)
    smat = cv.material("Med_AlterGScreen", emission=1.2)
    bag = _bag_vinyl()
    body = _plastic("Med_AlterGBody", "#e9eaeb", 0.4)
    belt = _plastic("Med_Belt", "#161718", 0.85)
    frame = _paint("Med_AlterGFrame", "#3a3d42", 0.4)
    p = PM(0.01)
    p.r = 0.04
    p.box((-0.43, -1.05, 0.0), (0.43, 1.05, 0.3), body)
    p.r = 0.003
    p.box((-0.3, -0.95, 0.3), (0.3, 0.95, 0.32), belt)
    # the bag: a long inflated hull from the deck up to the cockpit ring, translucent
    p.r = 0.0
    hull = []
    for y in np.linspace(-0.98, 0.98, 13):
        t = (y + 0.98) / 1.96
        h = 0.62 + 0.25 * math.sin(math.pi * min(t * 1.1, 1.0))
        prof = []
        for a in np.linspace(0, math.pi, 15):
            prof.append((0.41 * math.cos(a), y, 0.31 + h * math.sin(a)))
        hull.append(prof)
    for i in range(len(hull) - 1):
        for j in range(len(hull[0]) - 1):
            p.face([hull[i][j], hull[i + 1][j], hull[i + 1][j + 1], hull[i][j + 1]], bag)
    for prof in (hull[0], hull[-1]):
        p.face(prof + [(0.0, prof[0][1], 0.31)], bag)
    # seam ribs over the bag
    for i in range(1, len(hull) - 1, 2):
        _tube(p, hull[i], 0.006, _plastic("Med_BagSeam", "#9aa6b0", 0.4), 5)
    # cockpit ring with its skirt at waist height over the middle of the deck
    zc = 0.31 + 0.87 * 0.98
    _tube(p, [(0.24 * math.cos(a), 0.15 + 0.2 * math.sin(a), zc) for a in np.linspace(0, 2 * math.pi, 25)], 0.025, M["black"], 10, closed=True)
    p.lathe([(0.0, 0.0), (0.2, 0.0), (0.23, 0.03), (0.24, 0.06)], M["black"], c=(0.0, 0.15, zc - 0.06), sides=24)
    # height-adjustable frame: two columns rising from the deck to a hoop round the cockpit
    for x in (-0.5, 0.5):
        _tube(p, [(x, -0.1, 0.0), (x, -0.1, zc - 0.05), (x * 0.6, 0.15, zc)], 0.032, frame, 12, bend=0.1)
        _tube(p, [(x * 0.62, 0.15, zc + 0.02), (x * 0.9, 0.6, zc - 0.05), (x * 0.8, 0.9, zc - 0.1)], 0.018, M["alu"], 8, bend=0.08)
        p.cyl((x, -0.1, 0.0), (x, -0.1, 0.03), 0.06, frame, 14)
    # console on its arm at the front
    _tube(p, [(0.0, 0.9, 0.3), (0.0, 1.0, 1.2)], 0.03, frame, 12)
    p.r = 0.02
    p.obox((0.0, 1.0, 1.25), (0.5, 0.15, 0.32), body, pitch=math.radians(30))
    decal(p, (0.0, 0.93, 1.26), (0, -0.87, 0.5), (0, 0.5, 0.87), 0.3, 0.2, smat, scr, lift=0.005)
    return p.obj("alter_g", col)


# -- the lab and imaging ----------------------------------------------------------------------------

def vo2_cart(col, M):
    """A metabolic cart (Cosmed Quark CPET class): a four-castor trolley carrying the gas analyser,
    a calibration gas cylinder in its strap, a monitor on an arm showing the VO2 trace, and a face
    mask on its sampling line looped over the arm. Front +y."""
    cv = Canvas("Med_VO2", 512, 512, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.5, 0.3, 700)
    _screen(scr, "graph", 11)
    front = cv.alloc(0.5, 0.25, 700)
    front.fill(col="#e6e8ea")
    front.rect(0.02, 0.15, 0.2, 0.23, r=0.004, col="#0d1117", emit="#0d1117")
    front.text("O2 20.9  CO2 0.04", 0.025, 0.18, 0.012, col="#3fd06a", emit="#3fd06a")
    front.text("Quark", 0.4, 0.04, 0.03, weight="black", align="c", col="#d0402a")
    for k in range(4):
        front.circle(0.28 + k * 0.05, 0.19, 0.012, col="#3a3d42")
    smat = cv.material("Med_VO2Screen", emission=1.2)
    grey = _plastic("Med_CartGrey", "#d6d8da", 0.4)
    p = PM(0.006)
    for z in (0.12, 0.9):
        p.box((-0.32, -0.25, z), (0.32, 0.25, z + 0.025), grey)
    for x in (-0.3, 0.3):
        for y in (-0.23, 0.23):
            _tube(p, [(x, y, 0.12), (x, y, 0.92)], 0.015, M["alu"], 8)
            _castor(p, (x, y, 0.12), M, 0.035, lock=y > 0)
    p.r = 0.01
    i0 = len(p.F)
    p.box((-0.25, -0.2, 0.925), (0.25, 0.2, 1.175), grey)
    decal(p, (0.0, 0.2, 1.05), (0, 1, 0), (0, 0, 1), 0.5, 0.25, smat, front, lift=0.001)
    p.r = 0.0
    p.cyl((0.18, -0.15, 0.15), (0.18, -0.15, 0.85), 0.07, _paint("Med_GasCyl", "#d8d8d8", 0.3), 16)
    p.lathe([(0.0, 0.85), (0.07, 0.85), (0.04, 0.92), (0.02, 0.96)], M["chrome"], c=(0.18, -0.15, 0.0), sides=16)
    _tube(p, [(-0.2, 0.0, 1.17), (-0.2, 0.0, 1.5), (0.0, 0.1, 1.55)], 0.015, M["alu"], 8, bend=0.05)
    _monitor(p, (0.05, 0.18, 1.45), 0.0, 0.5, 0.3, smat, scr, M, stand=False)
    loop = [(0.25, 0.0, 0.95), (0.4, 0.2, 1.1), (0.35, 0.35, 1.3), (0.2, 0.25, 1.48)]
    _tube(p, loop, 0.008, _plastic("Med_SampleLine", "#e8e8e8", 0.3), 6, bend=0.08)
    p.r = 0.02
    p.obox((0.2, 0.25, 1.42), (0.11, 0.08, 0.13), M["black"])
    return p.obj("vo2_cart", col)


def ultrasound(col, M):
    """A diagnostic ultrasound system on its cart (GE LOGIQ class, 0.55 x 0.85 m): the base on
    locking castors, the body with probe holders and three probes on their cables, a touch panel
    and keyboard, the main display on an articulated arm showing a musculoskeletal scan, a gel
    warmer. Front +y."""
    cv = Canvas("Med_US", 512, 512, col="#2a2c2f", rough=0.5)
    scr = cv.alloc(0.48, 0.34, 800)
    _screen(scr, "ultrasound", 13)
    tp = cv.alloc(0.3, 0.2, 900)
    tp.fill(col="#0e1418", emit="#0e1418")
    for i in range(3):
        for j in range(4):
            tp.rect(0.02 + j * 0.07, 0.02 + i * 0.06, 0.08 + j * 0.07, 0.07 + i * 0.06, r=0.006, col="#2b5f8a", emit="#2b5f8a")
    smat = cv.material("Med_USScreen", emission=1.2)
    white = _plastic("Med_USWhite", "#eceeef", 0.35)
    p = PM(0.01)
    p.box((-0.3, -0.42, 0.08), (0.3, 0.42, 0.2), white)
    for x in (-0.25, 0.25):
        for y in (-0.36, 0.36):
            _castor(p, (x, y, 0.08), M, 0.04, lock=y > 0)
    p.r = 0.04
    p.box((-0.24, -0.3, 0.2), (0.24, 0.2, 0.9), white)
    p.box((-0.3, -0.25, 0.9), (0.3, 0.35, 0.98), white)
    p.r = 0.0
    decal(p, (0.0, 0.1, 0.981), (0, 0, 1), (0, 1, 0), 0.3, 0.2, smat, tp, lift=0.001)
    p.r = 0.004
    for k in range(4):
        for j in range(10):
            p.box((-0.25 + j * 0.05, 0.22 + k * 0.03, 0.98), (-0.21 + j * 0.05, 0.245 + k * 0.03, 0.99), M["black"])
    # probes in their holders on the right, cables looping down
    for k in range(3):
        y = -0.15 + k * 0.12
        p.r = 0.006
        p.box((0.3, y - 0.04, 0.85), (0.36, y + 0.04, 0.95), white)
        p.r = 0.008
        p.box((0.31, y - 0.02, 0.9), (0.35, y + 0.02, 1.05), _plastic("Med_Probe", "#3a3d42", 0.4))
        p.r = 0.0
        _tube(p, [(0.33, y, 0.9), (0.4, y, 0.6), (0.33, y + 0.05, 0.4), (0.25, y, 0.5)], 0.007, M["grey"], 6, bend=0.1)
    # the display arm and monitor
    _tube(p, [(0.0, -0.25, 0.98), (0.0, -0.22, 1.3)], 0.03, white, 10)
    _monitor(p, (0.0, -0.2, 1.22), 0.0, 0.48, 0.34, smat, scr, M, stand=False)
    p.r = 0.01
    p.box((-0.3, -0.3, 0.7), (-0.24, -0.15, 0.85), white)
    p.cyl((-0.27, -0.22, 0.85), (-0.27, -0.22, 0.95), 0.025, _plastic("Med_Gel", "#3a8ad0", 0.3), 12)
    return p.obj("ultrasound", col)


def xray_unit(col, M):
    """A digital radiography room set (floor-mounted tube stand class, 3.4 x 2.4 m): a floating-top
    patient table on its base, the tube head on a telescopic column riding floor and wall rails
    with its collimator, and a wall bucky stand with the flat-panel detector; yellow-and-black
    radiation trefoil on the detector. Front +y."""
    cv = Canvas("Med_Xray", 512, 512, col="#e9eaeb", rough=0.4)
    det = cv.alloc(0.45, 0.5, 700)
    det.fill(col="#e9eaeb")
    det.rect(0.05, 0.08, 0.4, 0.43, col="#cfd2d5", dh=-0.002)
    det.circle(0.225, 0.255, 0.04, col="#f2c400")
    for k in range(3):
        a = math.radians(90 + k * 120)
        det.poly([(0.225, 0.255), (0.225 + 0.035 * math.cos(a - 0.5), 0.255 + 0.035 * math.sin(a - 0.5)), (0.225 + 0.035 * math.cos(a + 0.5), 0.255 + 0.035 * math.sin(a + 0.5))], col="#141414")
    det.circle(0.225, 0.255, 0.008, col="#f2c400")
    mat = cv.material("Med_XrayDet")
    white = _plastic("Med_XrayWhite", "#ecedee", 0.35)
    grey = _plastic("Med_XrayGrey", "#9a9da1", 0.4)
    p = PM(0.01)
    # table
    p.r = 0.03
    p.box((-0.6, -0.25, 0.0), (0.6, 0.25, 0.55), white)
    p.r = 0.02
    p.box((-1.1, -0.38, 0.62), (1.1, 0.38, 0.68), white)
    p.box((-1.0, -0.33, 0.68), (1.0, 0.33, 0.7), _plastic("Med_XrayPad", "#3a4a5a", 0.6))
    # floor rail and tube stand
    p.r = 0.004
    p.box((-1.4, -0.95, 0.0), (1.4, -0.85, 0.03), M["alu"])
    p.r = 0.02
    p.box((-0.18, -1.0, 0.03), (0.18, -0.8, 0.2), grey)
    p.box((-0.1, -0.96, 0.2), (0.1, -0.84, 2.2), white)
    p.box((-0.12, -0.96, 1.6), (0.12, -0.2, 1.75), white)
    p.r = 0.04
    p.box((-0.22, -0.4, 1.38), (0.22, -0.05, 1.68), white)
    p.r = 0.01
    p.box((-0.15, -0.3, 1.18), (0.15, -0.08, 1.38), grey)
    p.box((-0.17, -0.35, 1.42), (0.17, -0.05, 1.46), M["black"])
    # wall bucky stand at the +x end
    p.r = 0.01
    p.box((1.55, -0.3, 0.0), (1.85, 0.3, 0.06), grey)
    p.box((1.7, -0.08, 0.06), (1.82, 0.08, 2.0), white)
    i0 = len(p.F)
    p.box((1.6, -0.25, 0.9), (1.7, 0.25, 1.4), white)
    decal(p, (1.6, 0.0, 1.15), (-1, 0, 0), (0, 0, 1), 0.45, 0.5, mat, det, lift=0.002)
    return p.obj("xray_unit", col)


def dexa(col, M):
    """A DEXA bone-densitometry and body-composition scanner (GE Lunar iDXA class, 2.8 x 1.1 m):
    the long table with rounded ends on a recessed plinth, a blue patient pad with a pillow and a
    rolled sheet, the C-arm over it on its carriage (a column at the back edge, the scanning arm
    across the table with the detector window underneath and a blue accent band), the control
    pendant, and the operator's monitor on a pole at the foot. Front +y."""
    cv = Canvas("Med_Dexa", 512, 512, col="#e9eaeb", rough=0.4)
    scr = cv.alloc(0.5, 0.3, 700)
    scr.fill(col="#0b0f14", emit="#0b0f14")
    scr.rect(0.03, 0.03, 0.2, 0.27, col="#1a2028", emit="#1a2028")
    scr.field(lambda X, Y: np.clip(1 - np.hypot((X - 0.115) / 0.05, (Y - 0.15) / 0.12), 0, 1) * 0.8 * (X < 0.2), col="#d0a070", emit="#d0a070")
    scr.text("BODY COMP", 0.25, 0.22, 0.03, weight="bold", col="#ffffff", emit="#ffffff")
    scr.text("Fat 9.8%   Lean 71.2 kg", 0.25, 0.15, 0.022, col="#3fd06a", emit="#3fd06a")
    smat = cv.material("Med_DexaScreen", emission=1.2)
    white = _plastic("Med_DexaWhite", "#eeeff0", 0.35)
    grey = _plastic("Med_DexaGrey", "#b8bcc0", 0.4)
    blue = _plastic("Med_DexaBlue", "#2f6a9a", 0.5)
    p = PM(0.02)
    # plinth and table
    p.r = 0.02
    p.box((-1.3, -0.32, 0.0), (1.3, 0.32, 0.08), grey)
    p.r = 0.04
    p.box((-1.4, -0.42, 0.08), (1.4, 0.42, 0.62), white)
    p.r = 0.03
    p.box((-1.3, -0.33, 0.62), (1.3, 0.33, 0.67), blue)
    p.r = 0.04
    p.box((-1.25, -0.2, 0.67), (-0.95, 0.2, 0.75), _plastic("Med_Pillow", "#f4f4f2", 0.8))
    p.r = 0.0
    p.cyl((1.1, -0.3, 0.71), (1.1, 0.3, 0.71), 0.045, _plastic("Med_Sheet", "#f2f0ea", 0.85), 16)
    # the C-arm: column at the back edge, arm across the table, detector underneath
    x0 = 0.2
    p.r = 0.05
    p.box((x0 - 0.22, -0.62, 0.0), (x0 + 0.22, -0.42, 1.18), white)
    p.box((x0 - 0.26, -0.62, 1.0), (x0 + 0.26, 0.48, 1.22), white)
    p.r = 0.01
    p.box((x0 - 0.265, -0.62, 1.08), (x0 + 0.265, 0.48, 1.12), blue)
    p.box((x0 - 0.18, -0.3, 0.985), (x0 + 0.18, 0.4, 1.0), _plastic("Med_DexaDet", "#2a2d32", 0.3))
    p.r = 0.0
    p.box((x0 - 0.16, -0.25, 0.98), (x0 + 0.16, 0.35, 0.985), _plastic("Med_DexaWin", "#4a5a6a", 0.15))
    # carriage rail along the back of the table
    p.r = 0.006
    p.box((-1.3, -0.46, 0.5), (1.3, -0.42, 0.54), grey)
    # control pendant hanging on its cable
    p.box((x0 + 0.27, 0.36, 0.86), (x0 + 0.33, 0.4, 0.98), _plastic("Med_Pendant", "#e8e8e6", 0.4))
    p.r = 0.0
    _tube(p, [(x0 + 0.26, 0.4, 1.05), (x0 + 0.3, 0.39, 0.98)], 0.004, M["black"], 5)
    # operator's monitor on a pole at the foot
    _tube(p, [(1.6, 0.4, 0.0), (1.6, 0.4, 1.15)], 0.03, M["alu"], 10)
    p.cyl((1.6, 0.4, 0.0), (1.6, 0.4, 0.02), 0.25, M["alu"], 20)
    _monitor(p, (1.6, 0.42, 1.0), math.radians(200), 0.5, 0.3, smat, scr, M, stand=False)
    return p.obj("dexa", col)


# -- recovery ---------------------------------------------------------------------------------------

def ice_bath(col, M):
    """A stainless cold-water immersion tub (iCool / Plunge class, 1.9 x 0.85 m, 0.95 m deep): a
    brushed shell with a rolled rim, water at 6 degrees to 0.75 m, internal step and seat, a chiller
    unit with its grille alongside, a digital temperature readout and a grab rail. Front +y."""
    cv = Canvas("Med_IceBath", 512, 512, col="#c4c7ca", rough=0.3, metal=0.8)
    pan = cv.alloc(0.2, 0.12, 1500)
    pan.fill(col="#14181c", emit="#14181c")
    pan.text("6.0 C", 0.1, 0.04, 0.045, weight="black", align="c", col="#4ab0ff", emit="#4ab0ff")
    pmat = cv.material("Med_IceBathPanel", emission=1.0)
    steel = M["alu"]
    water = _paint("Med_Water", "#2a4a52", 0.05)
    p = PM(0.01)
    L, W, H = 1.9, 0.85, 0.95
    p.r = 0.02
    # shell: outer walls, rim, the water at 0.75 m
    for (a, b) in (((-L / 2, -W / 2), (L / 2, -W / 2 + 0.03)), ((-L / 2, W / 2 - 0.03), (L / 2, W / 2)),
                   ((-L / 2, -W / 2), (-L / 2 + 0.03, W / 2)), ((L / 2 - 0.03, -W / 2), (L / 2, W / 2))):
        p.box((a[0], a[1], 0.0), (b[0], b[1], H), steel)
    p.box((-L / 2 + 0.03, -W / 2 + 0.03, 0.0), (L / 2 - 0.03, W / 2 - 0.03, 0.05), steel)
    p.r = 0.0
    _tube(p, [(-L / 2, -W / 2, H), (L / 2, -W / 2, H), (L / 2, W / 2, H), (-L / 2, W / 2, H)], 0.02, steel, 8, bend=0.04, closed=True)
    p.face([(-L / 2 + 0.03, -W / 2 + 0.03, 0.75), (L / 2 - 0.03, -W / 2 + 0.03, 0.75), (L / 2 - 0.03, W / 2 - 0.03, 0.75), (-L / 2 + 0.03, W / 2 - 0.03, 0.75)], water)
    p.r = 0.01
    p.box((L / 2 - 0.4, -W / 2 + 0.03, 0.05), (L / 2 - 0.03, W / 2 - 0.03, 0.45), steel)
    # chiller alongside, readout, grab rail
    p.box((L / 2 + 0.05, -0.35, 0.0), (L / 2 + 0.6, 0.35, 0.7), _plastic("Med_Chiller", "#3a3d42", 0.45))
    for k in range(10):
        p.box((L / 2 + 0.1, 0.35, 0.08 + k * 0.05), (L / 2 + 0.55, 0.36, 0.1 + k * 0.05), M["black"])
    decal(p, (-0.5, W / 2, H - 0.12), (0, 1, 0), (0, 0, 1), 0.2, 0.12, pmat, pan, lift=0.002)
    p.r = 0.0
    _tube(p, [(L / 2 - 0.2, W / 2 + 0.05, 0.0), (L / 2 - 0.2, W / 2 + 0.05, 1.25), (L / 2 - 0.6, W / 2 + 0.05, 1.25), (L / 2 - 0.6, W / 2 + 0.05, H)], 0.019, M["chrome"], 10, bend=0.06)
    return p.obj("ice_bath", col)


def cryo_chamber(col, M):
    """A whole-body cryotherapy chamber (electric, -110 degrees C; 3.0 x 2.4 x 2.5 m): an insulated
    white panel cabin with an ante-room and the main chamber, frosted porthole doors with chrome
    lever handles, a control station with a touchscreen at the entrance, and a frosty glow through
    the door glass. Front +y."""
    cv = Canvas("Med_Cryo", 1024, 1024, col="#f0f1f2", rough=0.35)
    front = cv.alloc(3.0, 2.5, 300)
    front.fill(col="#f0f1f2")
    for x in np.arange(0.0, 3.0, 1.0):
        front.rect(x, 0, x + 0.006, 2.5, col="#d6d8da")
    front.rect(0, 2.3, 3.0, 2.5, col="#1d4f91")
    front.text("CRYO  -110", 1.5, 2.34, 0.12, weight="black", align="c", col="#ffffff")
    for x0 in (0.3, 1.6):
        front.rect(x0, 0.05, x0 + 0.9, 2.05, r=0.04, col="#e4e6e8", dh=0.004)
        front.circle(x0 + 0.45, 1.45, 0.2, col="#bcd8ea", emit="#9cc8ea")
        front.ring(x0 + 0.45, 1.45, 0.2, 0.23, col="#c0c4c8", dh=0.004)
    scr = cv.alloc(0.3, 0.2, 900)
    scr.fill(col="#0b1a2a", emit="#0b1a2a")
    scr.text("-110 C", 0.15, 0.1, 0.05, weight="black", align="c", col="#7fd0ff", emit="#7fd0ff")
    scr.text("3:00", 0.15, 0.04, 0.035, weight="bold", align="c", col="#ffffff", emit="#ffffff")
    mat = cv.material("Med_CryoPanels", emission=0.8)
    p = PM(0.02)
    i0 = len(p.F)
    p.box((-1.5, -1.2, 0.0), (1.5, 1.2, 2.5), mat)
    project(p, i0, front, (1.5, 1.2, 0.0), (-1, 0, 0), (0, 0, 1))
    p.r = 0.004
    for x in (-1.2 + 0.85, 0.1 + 0.85):
        p.box((x - 0.12, 1.2, 1.0), (x, 1.24, 1.03), M["chrome"])
    # control station
    p.r = 0.01
    p.box((1.55, 0.6, 0.0), (1.85, 1.0, 1.1), _plastic("Med_CryoCtl", "#e9eaeb", 0.35))
    p.r = 0.0
    decal(p, (1.7, 1.0, 1.02), (0, 0.7, 0.7), (0, -0.7, 0.7), 0.3, 0.2, mat, scr, lift=0.003)
    return p.obj("cryo_chamber", col)


def sauna(col, M):
    """A Finnish sauna cabin (2.0 x 2.0 x 2.1 m, spruce): tongue-and-groove walls, a full-height
    bronze glass door with a timber handle, two-tier benches inside, the stove behind its guard
    with stones, a bucket and ladle, a thermometer. Front +y."""
    cv = Canvas("Med_Sauna", 1024, 1024, col="#c9a06a", rough=0.6)
    wall = cv.alloc(2.0, 2.1, 300)
    wall.fill(col="#caa26c")
    for x in np.arange(0.0, 2.0, 0.095):
        wall.rect(x, 0, x + 0.004, 2.1, col="#9a7444", dh=-0.001)
    wall.field(lambda X, Y: noise(X.shape, (6, 120), 3, 3), mul=0.92)
    wall.field(lambda X, Y: noise(X.shape, (40, 4), 4, 2) * 0.2, col="#8a6438")
    mat = cv.material("Med_SaunaWall")
    bronze = _paint("Med_BronzeGlass", "#5a4a32", 0.05)
    p = PM(0.01)
    i0 = len(p.F)
    p.box((-1.0, -1.0, 0.0), (1.0, 1.0, 2.1), mat)
    box_uv(p, i0, {"+y": wall, "-y": wall, "+x": wall, "-x": wall, "+z": wall}, (-1.0, -1.0, 0.0), (1.0, 1.0, 2.1))
    p.r = 0.0
    p.face([(0.1, 1.003, 0.05), (0.8, 1.003, 0.05), (0.8, 1.003, 2.0), (0.1, 1.003, 2.0)], bronze)
    p.r = 0.006
    p.box((0.15, 1.0, 0.9), (0.2, 1.06, 1.3), M["timber"])
    p.r = 0.003
    p.box((-0.6, 1.0, 1.4), (-0.5, 1.02, 1.7), _white("Med_Thermo"))
    return p.obj("sauna", col)


def pool_hoist(col, M):
    """A poolside patient hoist (Aquatic Access class): a stainless mast on a deck-mounted socket, a
    swinging boom with its actuator, and a sling seat hanging from the boom end. Front +y."""
    p = PM(0.006)
    p.r = 0.0
    p.cyl((0, 0, 0.0), (0, 0, 0.08), 0.12, M["alu"], 18)
    _tube(p, [(0, 0, 0.08), (0, 0, 1.6)], 0.045, M["chrome"], 14)
    _tube(p, [(0, 0, 1.55), (0, 0.25, 1.7), (0, 1.2, 1.6)], 0.035, M["chrome"], 12, bend=0.1)
    _tube(p, [(0, 0, 0.9), (0, 0.6, 1.62)], 0.03, _plastic("Med_Actuator", "#e9eaeb", 0.4), 10)
    _tube(p, [(0, 1.2, 1.6), (0, 1.2, 1.1)], 0.006, M["black"], 6)
    p.r = 0.02
    p.box((-0.25, 1.0, 0.95), (0.25, 1.4, 1.0), _plastic("Med_Sling", "#2f6a9a", 0.6))
    p.box((-0.25, 0.98, 1.0), (0.25, 1.03, 1.35), _plastic("Med_Sling", "#2f6a9a", 0.6))
    return p.obj("pool_hoist", col)


def sleep_pod(col, M):
    """A recovery sleep pod (2.1 x 1.1 m): an upholstered platform bed with a hooded shell over the
    head end, a dimmable reading light and controls in the hood, a fitted duvet and pillow. Front +y."""
    cv, reg = _cushion_canvas("Med_PodVinyl", 1.2, 1.0, seed=21)
    up = cv.tint_material("TINT_upholstery_pod", "#26384a", rough=0.6)
    shell = _plastic("Med_PodShell", "#eeeff0", 0.3)
    p = PM(0.03)
    i0 = len(p.F)
    p.box((-1.05, -0.55, 0.0), (1.05, 0.55, 0.4), up)
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.1, -0.6, 0.0), (1.1, 0.6, 0.4))
    p.box((-1.0, -0.5, 0.4), (1.0, 0.5, 0.55), _white("Med_Mattress"))
    p.r = 0.06
    p.box((-0.35, -0.48, 0.55), (0.98, 0.48, 0.63), _plastic("Med_Duvet", "#d9dde2", 0.85))
    p.r = 0.05
    p.box((-0.95, -0.35, 0.55), (-0.65, 0.35, 0.68), _white("Med_Pillow"))
    # the hood: a shell arching over the head end
    p.r = 0.0
    prof = [(0.6 * math.cos(a), 0.55 + 0.75 * math.sin(a)) for a in np.linspace(0, math.pi, 15)]
    for k in range(len(prof) - 1):
        (y0, z0), (y1, z1) = prof[k], prof[k + 1]
        p.face([(-1.1, y0, z0), (-1.1, y1, z1), (-0.3, y1, z1), (-0.3, y0, z0)], shell)
    p.face([(-1.1, y, z) for y, z in prof], shell)
    p.cyl((-0.7, 0.0, 1.25), (-0.7, 0.0, 1.22), 0.06, _paint("Med_PodLight", "#fff3dc", 0.3), 16)
    return p.obj("sleep_pod", col)


# -- front of house ---------------------------------------------------------------------------------

def reception_desk(col, M):
    """A reception counter (3.0 m, a raised transaction top over a lower wheelchair-height section,
    white solid-surface with a club-coloured fascia band): two monitors, a phone, a visitor book
    and a hand-gel stand. Front +y (visitors' side)."""
    cv = Canvas("Med_Reception", 512, 512, col="#f2f2f0", rough=0.4)
    scr = cv.alloc(0.53, 0.3, 700)
    _screen(scr, "emr", 23)
    smat = cv.material("Med_RecScreen", emission=1.2)
    surf = _paint("Med_Solid", "#f3f3f1", 0.25)
    band = _plastic("TINT_club_reception", "#1d4f91", 0.4)
    p = PM(0.01)
    p.r = 0.02
    p.box((-1.5, -0.35, 0.0), (1.5, 0.35, 0.75), surf)
    p.box((-1.5, 0.1, 0.75), (0.6, 0.45, 1.1), surf)
    p.box((-1.55, 0.25, 1.1), (0.65, 0.5, 1.13), surf)
    p.box((0.6, 0.25, 0.73), (1.55, 0.5, 0.77), surf)
    p.r = 0.0
    p.face([(1.5, 0.451, 0.12), (-1.5, 0.451, 0.12), (-1.5, 0.451, 0.32), (1.5, 0.451, 0.32)], band)
    _monitor(p, (-0.9, -0.1, 0.75), math.pi, 0.53, 0.3, smat, scr, M)
    _monitor(p, (-0.2, -0.1, 0.75), math.pi, 0.53, 0.3, smat, scr, M)
    p.r = 0.004
    p.box((0.2, -0.2, 0.75), (0.4, -0.02, 0.8), M["black"])
    p.box((0.9, 0.3, 0.77), (1.2, 0.48, 0.8), _plastic("Med_Book", "#2a2c30", 0.6))
    p.r = 0.0
    p.cyl((1.75, 0.6, 0.0), (1.75, 0.6, 1.05), 0.02, M["alu"], 10)
    p.r = 0.01
    p.box((1.68, 0.53, 1.05), (1.82, 0.67, 1.3), _white("Med_Dispenser"))
    return p.obj("reception_desk", col)


def waiting_seats(col, M):
    """Beam seating (three places, 1.8 m): a steel beam on two splayed legs, wipe-clean upholstered
    seats and backs (tinted), arms between. Front +y."""
    cv, reg = _cushion_canvas("Med_BeamVinyl", 0.6, 0.6, seed=29)
    up = cv.tint_material("TINT_upholstery_beam", "#1d4f91", rough=0.55)
    steel = _paint("Med_BeamSteel", "#3a3d42", 0.4)
    p = PM(0.004)
    p.box((-0.9, -0.05, 0.36), (0.9, 0.05, 0.4), steel)
    for x in (-0.6, 0.6):
        _tube(p, [(x, -0.25, 0.0), (x, 0.0, 0.38)], 0.02, steel, 8)
        _tube(p, [(x, 0.25, 0.0), (x, 0.0, 0.38)], 0.02, steel, 8)
    for k in range(3):
        x = -0.6 + k * 0.6
        p.r = 0.03
        i0 = len(p.F)
        p.box((x - 0.26, -0.22, 0.4), (x + 0.26, 0.24, 0.47), up)
        p.obox((x, -0.25, 0.72), (0.5, 0.06, 0.45), up, pitch=math.radians(-8))
        box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (x - 0.3, -0.3, 0.3), (x + 0.3, 0.3, 0.95))
    for x in (-0.9, -0.3, 0.3, 0.9):
        p.r = 0.01
        p.box((x - 0.025, -0.2, 0.6), (x + 0.025, 0.2, 0.63), M["black"])
        p.box((x - 0.012, -0.05, 0.4), (x + 0.012, 0.05, 0.6), steel)
    return p.obj("waiting_seats", col)


def wheelchair(col, M):
    """A self-propelled wheelchair (folded open): a chrome cross-braced frame, 24 inch rear wheels
    with push rims, castors, a black sling seat and back, push handles, swing-away footplates. Front +y."""
    p = PM(0.004)
    sling = _plastic("Med_Sling", "#1e2023", 0.7)
    for s_ in (-1, 1):
        x = s_ * 0.24
        _tube(p, [(x, 0.3, 0.12), (x, 0.3, 0.5), (x, -0.15, 0.5), (x, -0.2, 0.95), (x, -0.28, 1.0)], 0.012, M["chrome"], 8, bend=0.03)
        _tube(p, [(x, 0.3, 0.5), (x, 0.42, 0.15), (x, 0.36, 0.08)], 0.011, M["chrome"], 8, bend=0.03)
        p.r = 0.0
        tyre(p, (s_ * 0.31, -0.12, 0.3), 0.3, 0.035, 0.27, M["black"], M["grey"], sides=24, axis="x")
        tyre(p, (x, 0.32, 0.075), 0.075, 0.03, 0.04, M["black"], M["grey"], sides=14, axis="x")
        p.r = 0.004
        p.box((x - 0.06, 0.4, 0.08), (x + 0.06, 0.55, 0.095), M["black"])
    p.r = 0.0
    p.face([(-0.24, -0.15, 0.48), (0.24, -0.15, 0.48), (0.24, 0.28, 0.48), (-0.24, 0.28, 0.48)], sling)
    p.face([(-0.24, -0.16, 0.5), (0.24, -0.16, 0.5), (0.24, -0.21, 0.92), (-0.24, -0.21, 0.92)], sling)
    return p.obj("wheelchair", col)


def stretcher(col, M):
    """A hydraulic patient trolley (stretcher, 2.0 x 0.7 m): a radiolucent platform with a 100 mm
    mattress and a folded blanket, a raised backrest, folding side rails, an IV pole, the hydraulic
    column on a shrouded base with central-lock castors. Front +y."""
    cv, reg = _cushion_canvas("Med_TrolleyVinyl", 1.0, 0.7, seed=31)
    up = cv.tint_material("TINT_upholstery_trolley", "#2a5a8a", rough=0.55)
    white = _plastic("Med_TrolleyWhite", "#eceeef", 0.4)
    p = PM(0.01)
    p.r = 0.03
    p.box((-0.8, -0.3, 0.08), (0.8, 0.3, 0.3), white)
    for x in (-0.7, 0.7):
        for y in (-0.25, 0.25):
            _castor(p, (x, y, 0.08), M, 0.06)
    p.box((-0.15, -0.12, 0.3), (0.15, 0.12, 0.6), M["grey"])
    p.r = 0.01
    p.box((-1.0, -0.35, 0.6), (1.0, 0.35, 0.66), white)
    p.r = 0.04
    i0 = len(p.F)
    p.box((-0.4, -0.33, 0.66), (0.98, 0.33, 0.76), up)
    a = math.radians(25)
    p.obox((-0.4 - 0.3 * math.cos(a), 0.0, 0.71 + 0.3 * math.sin(a)), (0.6, 0.66, 0.1), up, roll=a)
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.0, -0.35, 0.6), (1.0, 0.35, 1.0))
    p.r = 0.02
    p.box((0.4, -0.3, 0.76), (0.9, 0.3, 0.83), _plastic("Med_Blanket", "#e6e2d6", 0.85))
    p.r = 0.0
    for s_ in (-1, 1):
        _tube(p, [(-0.5, s_ * 0.37, 0.66), (-0.5, s_ * 0.37, 0.95), (0.5, s_ * 0.37, 0.95), (0.5, s_ * 0.37, 0.66)], 0.012, M["chrome"], 8, bend=0.04)
    _tube(p, [(-0.95, 0.33, 0.66), (-0.95, 0.33, 2.0)], 0.012, M["chrome"], 8)
    for k in range(4):
        a_ = k * math.pi / 2
        _tube(p, [(-0.95, 0.33, 1.95), (-0.95 + 0.1 * math.cos(a_), 0.33 + 0.1 * math.sin(a_), 2.0)], 0.005, M["chrome"], 5)
    return p.obj("stretcher", col)


def med_tv(col, M):
    """A wall-mounted 65 inch information screen (the back at y = 0) showing the club's medical
    team rota and a training clip still. Front +y."""
    cv = Canvas("Med_TV", 512, 512, col="#0b0f14", rough=0.3)
    scr = cv.alloc(1.43, 0.8, 330)
    scr.fill(col="#0d1b2e", emit="#0d1b2e")
    scr.rect(0, 0.72, 1.43, 0.8, col="#1d4f91", emit="#1d4f91")
    scr.text("MEDICAL & PERFORMANCE", 0.04, 0.745, 0.04, weight="black", col="#ffffff", emit="#ffffff")
    scr.rect(0.04, 0.08, 0.84, 0.66, col="#2e6a3a", emit="#2e6a3a")
    scr.field(lambda X, Y: ((X > 0.04) & (X < 0.84) & (Y > 0.08) & (Y < 0.66)) * (np.sin(X * 60) > 0.6) * 0.15, col="#5a9a52", emit="#5a9a52")
    for i, t in enumerate(("Physio 08:00", "Doctor 09:30", "Rehab 10:00", "Pool 14:00", "Ice 15:30")):
        scr.text(t, 0.9, 0.58 - i * 0.1, 0.035, weight="bold", col="#d8e2ee", emit="#d8e2ee")
    mat = cv.material("Med_TVScreen", emission=1.0)
    p = PM(0.006)
    p.box((-0.735, 0.0, 1.15), (0.735, 0.05, 1.99), M["black"])
    decal(p, (0.0, 0.05, 1.57), (0, 1, 0), (0, 0, 1), 1.43, 0.8, mat, scr, lift=0.001)
    return p.obj("med_tv", col)


def indoor_plant(col, M):
    """A floor-standing kentia palm (1.6 m) in a tall white fibreglass planter with a rolled rim
    and a bark-mulch top: eight fronds on slender stems rising and arching out, each with pairs of
    long narrow leaflets that hang from the midrib, longest mid-frond; a new spear unfurling in the
    middle; two greens and the odd browned tip. Spread kept within 0.6 m of the centre (it stands
    near walls). Front +y."""
    p = PM(0.01)
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.19, 0.0), (0.24, 0.53), (0.255, 0.545), (0.25, 0.56), (0.225, 0.56), (0.222, 0.515), (0.0, 0.515)], _white("Med_Pot"), sides=28)
    p.cyl((0, 0, 0.515), (0, 0, 0.525), 0.222, _paint("Med_Bark", "#4a3424", 0.95), 24)
    greens = (_plastic("Med_PalmLeaf", "#3f6a2e", 0.5), _plastic("Med_PalmLeaf2", "#557f38", 0.55))
    stemm = _plastic("Med_PalmStem", "#5a7034", 0.6)
    brown = _plastic("Med_PalmTip", "#8a7440", 0.7)
    rng = np.random.default_rng(3)
    for k in range(8):
        a = k * 2 * math.pi / 8 + rng.uniform(-0.25, 0.25)
        ca, sa = math.cos(a), math.sin(a)
        H = rng.uniform(0.7, 1.0)          # rise of the stem above the pot
        R = rng.uniform(0.26, 0.36)        # reach of its tip from the centre
        leaf = greens[k % 2]
        # the midrib: up and out, arching over at the tip
        pts = []
        # (the stems come up from a clump, not one point)
        for n in range(9):
            t = n / 8.0
            r = 0.035 + (R - 0.035) * t ** 1.4
            z = 0.525 + H * (1.9 * t - 1.05 * t * t)
            pts.append((r * ca, r * sa, z))
        _tube(p, pts, 0.005, stemm, 5, bend=0.05)
        # leaflets in pairs along the outer two-thirds, hanging from the midrib
        for n in range(16):
            t = 0.3 + 0.7 * n / 15.0
            r = 0.035 + (R - 0.035) * t ** 1.4
            z = 0.525 + H * (1.9 * t - 1.05 * t * t)
            L = 0.12 + 0.13 * math.sin(math.pi * (t - 0.3) / 0.7 * 0.9 + 0.15)
            w = 0.011
            for side in (-1.0, 1.0):
                # out to the side of the frond, forward along it, drooping
                sx, sy = -sa * side, ca * side
                d1 = (0.75 * sx + 0.35 * ca, 0.75 * sy + 0.35 * sa, -0.25)
                d2 = (0.5 * sx + 0.3 * ca, 0.5 * sy + 0.3 * sa, -0.8)
                p0 = (r * ca, r * sa, z)
                p1 = (p0[0] + d1[0] * L * 0.5, p0[1] + d1[1] * L * 0.5, p0[2] + d1[2] * L * 0.5)
                p2 = (p1[0] + d2[0] * L * 0.5, p1[1] + d2[1] * L * 0.5, p1[2] + d2[2] * L * 0.5)
                # (the width across the leaflet, horizontal, square to its run)
                nx, ny = ca, sa
                tip = brown if rng.random() < 0.06 else leaf
                p.face([(p0[0] - nx * w * 0.5, p0[1] - ny * w * 0.5, p0[2]), (p0[0] + nx * w * 0.5, p0[1] + ny * w * 0.5, p0[2]),
                        (p1[0] + nx * w, p1[1] + ny * w, p1[2]), (p1[0] - nx * w, p1[1] - ny * w, p1[2])], leaf)
                p.face([(p1[0] - nx * w, p1[1] - ny * w, p1[2]), (p1[0] + nx * w, p1[1] + ny * w, p1[2]), p2], tip)
    # the new spear, still furled
    p.cyl((0.0, 0.0, 0.52), (0.03, 0.02, 1.45), 0.012, greens[1], 8, r2=0.002)
    return p.obj("indoor_plant", col)


# -- washrooms and changing -------------------------------------------------------------------------

def wc_accessible(col, M):
    """An accessible WC fit-out (Doc M pack, the back wall at y = 0, 2.2 m along it): a wall-hung
    pan with its seat and lid on a concealed-cistern duct panel with a flush plate, a fold-down
    grab rail on the open side and a fixed rail on the wall, a small hand basin with a lever tap,
    a roll holder, a soap dispenser, a paper-towel unit, the pull-cord alarm and a lidded bin.
    Front +y (the pan faces +y)."""
    cv = Canvas("Med_WC", 512, 512, col="#ffffff", rough=0.3)
    plate = cv.alloc(0.25, 0.17, 1200)
    plate.fill(col="#e9ebec", metal=0.6, rough=0.3)
    plate.rect(0.02, 0.02, 0.115, 0.15, r=0.01, col="#d4d7d9", dh=0.002, metal=0.6)
    plate.rect(0.135, 0.02, 0.23, 0.15, r=0.01, col="#d4d7d9", dh=0.002, metal=0.6)
    sign = cv.alloc(0.15, 0.15, 1200)
    sign.fill(col="#c81e22")
    sign.text("ALARM", 0.075, 0.06, 0.03, weight="black", align="c", col="#ffffff")
    mat = cv.material("Med_WCPlate")
    ceramic = _paint("Med_Ceramic", "#f6f6f2", 0.12)
    duct = _paint("Med_Duct", "#eceeec", 0.35)
    p = PM(0.004)
    p.r = 0.01
    p.box((-0.45, 0.0, 0.0), (0.45, 0.2, 1.05), duct)
    p.box((-0.47, 0.0, 1.05), (0.47, 0.22, 1.08), duct)
    decal(p, (0.0, 0.2, 0.85), (0, 1, 0), (0, 0, 1), 0.25, 0.17, mat, plate, lift=0.002)
    # wall-hung pan, seat and lid (lid raised against the duct)
    p.r = 0.0
    p.lathe([(0.0, 0.25), (0.17, 0.27), (0.19, 0.33), (0.185, 0.4), (0.0, 0.41)], ceramic, c=(0.0, 0.48, 0.0), sides=24)
    p.r = 0.05
    p.box((-0.18, 0.2, 0.28), (0.18, 0.42, 0.41), ceramic)
    p.r = 0.0
    seat = [(0.17 * math.cos(t), 0.47 + 0.22 * math.sin(t), 0.415) for t in np.linspace(0, 2 * math.pi, 25)[:-1]]
    _tube(p, seat, 0.02, _white("Med_Seat"), 8, closed=True)
    p.r = 0.02
    p.obox((0.0, 0.25, 0.65), (0.36, 0.03, 0.45), _white("Med_Seat"), pitch=math.radians(-8))
    # rails: hinged drop-down on the open side, fixed on the wall side
    p.r = 0.0
    _tube(p, [(0.45, 0.2, 0.68), (0.45, 0.85, 0.68), (0.45, 0.85, 0.82), (0.45, 0.2, 0.82)], 0.016, _paint("Med_Rail", "#e3e3df", 0.3), 10, bend=0.04)
    _tube(p, [(-0.7, 0.04, 0.8), (-0.7, 0.04, 1.4)], 0.016, _paint("Med_Rail", "#e3e3df", 0.3), 10)
    # basin and tap, beside the pan within reach
    p.r = 0.0
    p.lathe([(0.0, 0.7), (0.15, 0.72), (0.2, 0.78), (0.21, 0.8), (0.0, 0.8)], ceramic, c=(-0.95, 0.2, 0.0), sides=20)
    p.cyl((-0.95, 0.03, 0.88), (-0.95, 0.14, 0.88), 0.015, M["chrome"], 10)
    p.r = 0.004
    p.box((-0.98, 0.1, 0.86), (-0.92, 0.16, 0.9), M["chrome"])
    p.box((-1.08, 0.0, 1.0), (-0.98, 0.1, 1.25), _white("Med_Dispenser"))
    p.box((-1.1, 0.0, 1.35), (-0.8, 0.12, 1.7), _white("Med_Towels"))
    # roll holder, alarm cord and its sign, bin
    p.r = 0.0
    p.cyl((0.6, 0.04, 0.75), (0.72, 0.04, 0.75), 0.055, _white("Med_Paper"), 14)
    p.r = 0.004
    p.box((0.58, 0.0, 0.8), (0.74, 0.03, 0.83), M["chrome"])
    p.box((0.75, 0.0, 2.05), (0.83, 0.03, 2.13), _white("Med_Ceiling"))
    p.r = 0.0
    _tube(p, [(0.79, 0.02, 2.05), (0.79, 0.02, 0.1)], 0.003, M["red"], 5)
    p.cyl((0.79, 0.02, 0.15), (0.79, 0.02, 0.1), 0.02, M["red"], 10)
    decal(p, (0.79, 0.0, 1.4), (0, 1, 0), (0, 0, 1), 0.15, 0.15, mat, sign, lift=0.002)
    p.r = 0.02
    p.box((0.75, 0.1, 0.0), (0.98, 0.33, 0.45), _white("Med_Bin"))
    return p.obj("wc_accessible", col)


def lockers(col, M):
    """A bank of four full-height steel lockers (1.2 x 0.45 x 1.8 m) on a 100 mm plinth, a sloping
    top: louvred doors in a tinted finish (the club colour by default), number plates, cam locks,
    a hasp with a padlock on one; a slatted timber changing bench in front. Front +y."""
    cv = Canvas("Med_Lockers", 512, 512, col="#ffffff", rough=0.4)
    door = cv.alloc(0.29, 1.68, 300)
    door.fill(col="#ffffff")
    door.mottle(0.03, 4, 2)
    for y in (1.45, 1.5, 1.55, 0.15, 0.2, 0.25):
        door.rect(0.08, y, 0.21, y + 0.02, r=0.005, col="#4a4a4a", dh=-0.003)
    door.rect(0.09, 1.25, 0.2, 1.33, r=0.004, col="#d8d8d8", dh=0.001)
    door.frame(0.005, 0.005, 0.285, 1.675, 0.008, col="#d0d0d0", dh=0.001)
    tint = cv.tint_material("TINT_locker_door", "#1d4f91", rough=0.4)
    body = _paint("Med_LockerBody", "#d9dadb", 0.4)
    p = PM(0.004)
    p.box((-0.6, -0.22, 0.0), (0.6, 0.22, 0.1), M["black"])
    p.box((-0.6, -0.22, 0.1), (0.6, 0.22, 1.8), body)
    p.r = 0.0
    p.face([(-0.6, 0.22, 1.8), (0.6, 0.22, 1.8), (0.6, 0.0, 1.95), (-0.6, 0.0, 1.95)], body)
    for k in range(4):
        x = -0.45 + k * 0.3
        p.r = 0.003
        i0 = len(p.F)
        p.box((x - 0.145, 0.22, 0.12), (x + 0.145, 0.235, 1.79), tint)
        project(p, i0, door, (x + 0.145, 0.235, 0.12), (-1, 0, 0), (0, 0, 1))
        p.cyl((x + 0.1, 0.235, 1.05), (x + 0.1, 0.25, 1.05), 0.012, M["chrome"], 10)
        if k == 2:
            p.box((x + 0.08, 0.235, 1.12), (x + 0.12, 0.25, 1.15), M["chrome"])
            p.r = 0.0
            p.lathe([(0.0, 1.06), (0.02, 1.06), (0.02, 1.1), (0.0, 1.1)], M["chrome"], c=(x + 0.1, 0.26, 0.0), sides=10)
            p.r = 0.003
    # changing bench
    for x in (-0.5, 0.5):
        p.box((x - 0.025, 0.55, 0.0), (x + 0.025, 0.85, 0.42), _paint("Med_BenchFrame", "#3a3d42", 0.4))
    for k in range(3):
        p.box((-0.6, 0.56 + k * 0.1, 0.42), (0.6, 0.64 + k * 0.1, 0.45), M["timber"])
    return p.obj("lockers", col)


def showers(col, M):
    """A row of three team showers (the back wall at y = 0, 2.85 m along it, 1.2 m deep): 13 mm
    compact-laminate partitions held 150 mm off the floor on stainless feet with a top rail, a
    chrome riser in each bay with a timed push-button control and a vandal-resistant fixed head,
    a wall soap dispenser and a towel hook, and a stainless linear drain along the open front.
    Front +y."""
    cv = Canvas("Med_Showers", 512, 512, col="#ffffff", rough=0.3)
    drain = cv.alloc(2.85, 0.08, 160)
    drain.fill(col="#b8bcc0", metal=0.8, rough=0.35)
    for x in np.arange(0.02, 2.85, 0.03):
        drain.rect(x, 0.015, x + 0.012, 0.065, col="#2a2c2e", dh=-0.003)
    mat = cv.material("Med_ShowerDrain")
    panel = _paint("Med_Laminate", "#7d868f", 0.35)
    steel = M["chrome"]
    p = PM(0.004)
    # partitions with feet and a top rail tying their front edges together
    for k in range(4):
        x = -1.425 + k * 0.95
        p.r = 0.004
        p.box((x - 0.01, 0.0, 0.15), (x + 0.01, 1.2, 2.0), panel)
        p.r = 0.0
        p.cyl((x, 1.1, 0.0), (x, 1.1, 0.15), 0.014, steel, 10)
        p.cyl((x, 0.3, 0.0), (x, 0.3, 0.15), 0.014, steel, 10)
    _tube(p, [(-1.425, 1.2, 2.0), (1.425, 1.2, 2.0)], 0.012, steel, 10)
    for k in range(3):
        x = -0.95 + k * 0.95
        # riser, control and head
        p.r = 0.0
        p.cyl((x, 0.04, 1.0), (x, 0.04, 2.05), 0.011, steel, 10)
        p.cyl((x, 0.0, 2.05), (x, 0.04, 2.05), 0.012, steel, 10)
        p.r = 0.008
        p.box((x - 0.06, 0.0, 1.0), (x + 0.06, 0.05, 1.18), steel)
        p.r = 0.0
        p.cyl((x, 0.05, 1.1), (x, 0.065, 1.1), 0.03, _paint("Med_ShowerBtn", "#d8dadc", 0.25), 16)
        p.cyl((x, 0.0, 2.1), (x, 0.12, 2.0), 0.015, steel, 10)
        p.lathe([(0.0, 0.0), (0.05, 0.0), (0.045, 0.03), (0.0, 0.035)], steel, c=(x, 0.14, 1.97), sides=16)
        # soap dispenser on the wall and a towel hook on the partition outside the spray
        p.r = 0.008
        p.box((x + 0.2, 0.0, 1.3), (x + 0.3, 0.08, 1.48), _white("Med_Dispenser"))
        p.r = 0.0
        p.cyl((x - 0.465, 1.05, 1.75), (x - 0.43, 1.05, 1.75), 0.008, steel, 8)
    decal(p, (0.0, 1.27, 0.0), (0, 0, 1), (0, 1, 0), 2.85, 0.08, mat, drain, lift=0.003)
    return p.obj("showers", col)


BUILDERS = {
    "prop.wc_accessible": wc_accessible, "prop.lockers": lockers, "prop.showers": showers,
    "prop.treadmill": treadmill, "prop.wattbike": wattbike, "prop.squat_rack": squat_rack,
    "prop.dumbbell_rack": dumbbell_rack, "prop.isokinetic": isokinetic, "prop.alter_g": alter_g,
    "prop.vo2_cart": vo2_cart, "prop.ultrasound": ultrasound, "prop.xray_unit": xray_unit, "prop.dexa": dexa,
    "prop.ice_bath": ice_bath, "prop.cryo_chamber": cryo_chamber, "prop.sauna": sauna, "prop.pool_hoist": pool_hoist,
    "prop.sleep_pod": sleep_pod, "prop.reception_desk": reception_desk, "prop.waiting_seats": waiting_seats,
    "prop.wheelchair": wheelchair, "prop.stretcher": stretcher, "prop.med_tv": med_tv, "prop.indoor_plant": indoor_plant,
    "prop.med_couch": med_couch, "prop.med_couch_folding": med_couch_folding, "prop.med_exam_couch": med_exam_couch,
    "prop.med_chair_office": med_chair_office, "prop.med_chair": med_chair, "prop.med_desk": med_desk,
    "prop.med_sink": med_sink, "prop.med_cabinet": med_cabinet, "prop.med_trolley": med_trolley,
    "prop.med_wall_station": med_wall_station, "prop.ice_machine": ice_machine, "prop.med_curtain": med_curtain,
}
