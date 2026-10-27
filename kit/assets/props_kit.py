"""Training kit, gym gear and matchday odds built in code, replacing text-to-3D props that came out
disfigured (docs/qa/asset_audit.md): goalkeeper gloves, a Swiss ball, soft plyo boxes, a kit
hamper, a medals stand, a ball bag, a water-bottle carrier, foam rollers, resistance bands on their
rack, a city bicycle, a kit bag, a tactics easel, a rack of flags, matchday litter, a stack of
training bibs. The manifest (catalog/generated_props.json) names each as its generated prop's
fallback, so placements of genprop.<id> draw these. Real sizes; origin on the floor at the
footprint centre, +Y the front (wall units: the back at y = 0).
"""
import math

import numpy as np

from .texpaint import Canvas, box_uv, decal, noise, project
from .props import PM, _paint, _plastic, clear_glass, tyre
from .props_amenity import _loft, _tube
from .props_training import _ball, _ball_mat, _net_mat, _rot


def _ell(cx, cy, z, a, b, n=16, a0=0.0):
    return [(cx + a * math.cos(a0 + 2 * math.pi * k / n), cy + b * math.sin(a0 + 2 * math.pi * k / n), z) for k in range(n)]


def _rrect_ring(cx, cy, z, w, d, r, n=4):
    """A rounded rectangle (w x d, corner radius r) as a closed ring of points at height z."""
    out = []
    for qx, qy, a0 in ((1, 1, 0.0), (-1, 1, 0.5), (-1, -1, 1.0), (1, -1, 1.5)):
        ox, oy = cx + qx * (w / 2 - r), cy + qy * (d / 2 - r)
        for k in range(n + 1):
            a = math.pi * (a0 + 0.5 * k / n)
            out.append((ox + r * math.cos(a), oy + r * math.sin(a), z))
    return out


def _revolve(p, prof, mat, c=(0, 0, 0), sides=32, cell=None):
    """A surface of revolution with UVs in cells of `cell` metres (for tiling net textures)."""
    V, uv = [], []
    for r, z in prof:
        for k in range(sides + 1):
            a = 2 * math.pi * k / sides
            V.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2] + z))
            uv.append((a * 0.33 / cell, z / cell))
    F = []
    m = sides + 1
    for j in range(len(prof) - 1):
        for k in range(sides):
            F.append((j * m + k, j * m + k + 1, (j + 1) * m + k + 1, (j + 1) * m + k))
    p.faces(V, F, mat, uvs=uv)


# -- training kit ------------------------------------------------------------------------------------

def gk_gloves(col, M):
    """A pair of goalkeeper gloves (0.28 m long) lying on the grass, one palm-up on top of the other
    at an angle: a pale grey latex palm, a backhand in black with neon green panels, four
    slightly curled fingers and a thumb, a wide neoprene cuff with its wrap strap. Front +y."""
    latex = _plastic("Kit_GloveLatex", "#c9cbc6", 0.75)
    back = _plastic("Kit_GloveBack", "#141516", 0.6)
    neon = _plastic("Kit_GloveNeon", "#58e01c", 0.5)
    strap = _plastic("Kit_GloveStrap", "#1d1e20", 0.7)
    p = PM(0.004)

    def glove(x0, y0, yaw, z0, palm_up, s):
        ca, sa = math.cos(yaw), math.sin(yaw)

        def at(u, v, w):
            return (x0 + u * ca - v * sa * s, y0 + u * sa + v * ca * s, z0 + w)
        top, bot = (latex, back) if palm_up else (back, latex)
        # the hand: a flattened rounded slab from the cuff (u = 0) to the knuckles (u = 0.13)
        rings = []
        for u, w_, h_ in ((0.0, 0.05, 0.024), (0.04, 0.052, 0.026), (0.09, 0.054, 0.026), (0.13, 0.05, 0.022)):
            rings.append([at(u, w_ * math.cos(t), h_ * (0.5 + 0.5 * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 17)[:-1]])
        _loft(p, rings, top)
        p.r = 0.0
        p.obox(at(0.07, 0.0, 0.0015), (0.13, 0.1, 0.003), bot, yaw=yaw)
        # fingers: four tapering, slightly curled capsules, the outer ones shorter
        for k, (v, L) in enumerate(((-0.036, 0.075), (-0.012, 0.09), (0.012, 0.088), (0.036, 0.07))):
            pts = [at(0.125, v, 0.012), at(0.125 + L * 0.5, v * 1.08, 0.013), at(0.125 + L, v * 1.12, 0.008)]
            _tube(p, pts, 0.0115, top, 10, bend=0.03)
            p.sphere(pts[-1], 0.0115, top, 10, 6)
            if not palm_up:
                p.r = 0.0
                p.obox(at(0.125 + L * 0.45, v * 1.08, 0.022), (L * 0.6, 0.012, 0.004), neon, yaw=yaw)
        # thumb out to the side
        pts = [at(0.05, 0.05, 0.012), at(0.08, 0.075, 0.012), at(0.11, 0.085, 0.009)]
        _tube(p, pts, 0.012, top, 10, bend=0.025)
        p.sphere(pts[-1], 0.012, top, 10, 6)
        # neoprene cuff and its wrap strap
        rings = [[at(u, w_ * math.cos(t), h_ * (0.5 + 0.5 * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 17)[:-1]]
                 for u, w_, h_ in ((-0.07, 0.044, 0.04), (-0.02, 0.048, 0.034), (0.0, 0.05, 0.026))]
        _loft(p, rings, back)
        p.r = 0.002
        p.obox(at(-0.035, 0.0, 0.042 if not palm_up else 0.039), (0.035, 0.1, 0.006), neon if not palm_up else strap, yaw=yaw)
    glove(-0.04, -0.02, 0.15, 0.0, False, 1.0)
    glove(-0.02, 0.03, -0.35, 0.024, True, -1.0)
    return p.obj("gk_gloves", col)


def swiss_ball(col, M):
    """A 65 cm gym (Swiss) ball in matt anti-burst PVC, sitting slightly flattened where it meets
    the floor, its moulding seam round the middle and the valve plug. Front +y."""
    pvc = _plastic("Kit_SwissBall", "#2f6fc4", 0.55)
    p = PM(0.0)
    R = 0.325
    rings = []
    for j in range(25):
        t = -math.pi / 2 + math.pi * j / 24
        z = R + R * math.sin(t)
        r = R * math.cos(t)
        # (flattened on the floor: the bottom pushed up into a small contact patch)
        if z < 0.03:
            z = 0.03 * (z / 0.03) ** 0.5 if z > 0 else 0.0
        rings.append([(r * math.cos(a), r * math.sin(a), z * 0.985) for a in np.linspace(0, 2 * math.pi, 49)[:-1]])
    rings[0] = [(0.0, 0.0, 0.0)] * 48
    p.loft(rings[1:], pvc, cap0=False, cap1=False)
    p.cyl((0, 0, 0.0), (0, 0, 0.004), np.hypot(*rings[1][0][:2]), pvc, 48)
    _tube(p, [(R * 0.985 * math.cos(a), R * 0.985 * math.sin(a), R * 0.985) for a in np.linspace(0, 2 * math.pi, 65)], 0.0018,
          _plastic("Kit_SwissSeam", "#2a62ae", 0.5), 5, closed=True)
    p.cyl((0.0, -R * 0.4, R * 0.985 + R * 0.9 * 0.985 - R * 0.98), (0.0, -R * 0.4, R * 1.94), 0.012, _plastic("Kit_Valve", "#e8e8e4", 0.4), 12)
    return p.obj("swiss_ball", col)


def plyo_boxes(col, M):
    """A set of three soft plyometric boxes stacked largest first (0.76 m in all): high-density foam
    in black PVC-coated nylon with contrasting piped edges, webbing carry handles on the sides and
    a white height label on the front of each. Front +y."""
    cv = Canvas("Kit_Plyo", 512, 256, col="#ffffff", rough=0.5)
    labs = []
    for t in ("30", "25", "21"):
        r = cv.alloc(0.12, 0.06, 1500)
        r.fill(col="#f2f2f0")
        r.text(t + " cm", 0.06, 0.017, 0.03, weight="black", align="c", col="#141414")
        labs.append(r)
    lmat = cv.material("Kit_PlyoLabels")
    vinyl = _plastic("Kit_PlyoVinyl", "#1b1c1e", 0.55)
    pipe = _plastic("Kit_PlyoPipe", "#3a3c40", 0.45)
    web = _plastic("Kit_PlyoWebbing", "#2a2b2e", 0.8)
    p = PM(0.03)
    z = 0.0
    for k, (w, d, h) in enumerate(((1.0, 0.75, 0.3), (0.86, 0.66, 0.25), (0.72, 0.56, 0.21))):
        p.r = 0.03
        p.box((-w / 2, -d / 2, z), (w / 2, d / 2, z + h), vinyl)
        p.r = 0.0
        for zz in (z + 0.012, z + h - 0.012):
            _tube(p, _rrect_ring(0, 0, zz, w + 0.002, d + 0.002, 0.03), 0.006, pipe, 6, closed=True)
        for s in (-1, 1):
            _tube(p, [(s * (w / 2 + 0.002), -0.09, z + h * 0.62), (s * (w / 2 + 0.03), -0.06, z + h * 0.66),
                      (s * (w / 2 + 0.03), 0.06, z + h * 0.66), (s * (w / 2 + 0.002), 0.09, z + h * 0.62)], 0.008, web, 6, bend=0.02)
        decal(p, (0.0, d / 2, z + h * 0.5), (0, 1, 0), (0, 0, 1), 0.12, 0.06, lmat, labs[k], lift=0.003)
        z += h
    return p.obj("plyo_boxes", col)


def kit_hamper(col, M):
    """A wicker kit hamper on castors (0.9 x 0.6 m, 0.8 m to the top of the kit): a woven willow
    basket with a rolled rim and leather corner straps, its lid propped open against the back,
    piled with the match kit (shirts in the club colour, white shorts, socks), one shirt hanging
    over the front. Front +y."""
    cv = Canvas("Kit_Wicker", 1024, 512, col="#a07a4a", rough=0.8)
    r = cv.alloc(1.0, 0.5, 800)
    r.fill(col="#8c6838")
    for y in np.arange(0.0, 0.5, 0.022):
        for x in np.arange(0.0, 1.0, 0.03):
            off = 0.015 if int(y / 0.022) % 2 else 0.0
            r.rect(x + off, y + 0.002, x + off + 0.026, y + 0.02, r=0.006, col="#b8925c", dh=0.002)
    r.field(lambda X, Y: noise(X.shape, 40, 3, 2) * 0.25, col="#6a4a24")
    wick = cv.material("Kit_WickerWeave")
    kcv = Canvas("Kit_HamperCloth", 256, 256, col="#ffffff", rough=0.85)
    kr = kcv.alloc(0.5, 0.5, 300)
    kr.fill(col="#ffffff")
    kr.field(lambda X, Y: noise(X.shape, 25, 7, 2) * 0.18, col="#9a9a9a")
    shirt = kcv.tint_material("TINT_club_hamper", "#c8202a", rough=0.85)
    white = _paint("Kit_HamperWhite", "#ecebe6", 0.85)
    leather = _paint("Kit_HamperLeather", "#4a2c18", 0.55)
    p = PM(0.004)
    W, D, H = 0.9, 0.6, 0.56
    z0 = 0.08
    p.r = 0.0
    i0 = len(p.F)
    p.box((-W / 2, -D / 2, z0), (W / 2, D / 2, z0 + H), wick)
    box_uv(p, i0, {"+y": r, "-y": r, "+x": r, "-x": r, "-z": r}, (-W / 2, -D / 2, z0), (W / 2, D / 2, z0 + H))
    _tube(p, _rrect_ring(0, 0, z0 + H, W + 0.01, D + 0.01, 0.04), 0.014, leather, 8, closed=True)
    for x in (-W / 2, W / 2):
        for y in (-D / 2, D / 2):
            p.r = 0.002
            p.box((x - 0.012, y - 0.012, z0 + 0.04), (x + 0.012, y + 0.012, z0 + H - 0.02), leather)
            tyre(p, (x * 0.85, y * 0.85, 0.035), 0.035, 0.025, 0.018, M["rubber"], M["grey"], sides=12, axis="y")
    # the lid, propped open against the back
    p.r = 0.0
    i0 = len(p.F)
    p.obox((0.0, -D / 2 - 0.06, z0 + H + 0.27), (W + 0.02, 0.03, 0.56), wick, pitch=math.radians(-12))
    box_uv(p, i0, {"+y": r, "-y": r}, (-W / 2, -D / 2 - 0.15, z0 + H), (W / 2, -D / 2, z0 + H + 0.56))
    # the kit: crumpled shirts and shorts heaped proud of the rim
    rng = np.random.default_rng(17)
    ztop = z0 + H
    for k in range(9):
        x, y = rng.uniform(-0.3, 0.3), rng.uniform(-0.18, 0.18)
        m = white if k % 3 == 1 else shirt
        p.r = 0.03
        p.obox((x, y, ztop - 0.02 + rng.uniform(0.0, 0.1)), (rng.uniform(0.26, 0.4), rng.uniform(0.16, 0.26), 0.07), m,
               yaw=rng.uniform(0, math.pi), pitch=rng.uniform(-0.25, 0.25), roll=rng.uniform(-0.25, 0.25))
    # one shirt hanging over the front edge, a sock over the side
    p.r = 0.01
    p.obox((0.08, D / 2 + 0.015, ztop - 0.12), (0.34, 0.012, 0.26), shirt, pitch=math.radians(4))
    p.obox((0.08, D / 2 - 0.05, ztop + 0.02), (0.34, 0.12, 0.03), shirt)
    _tube(p, [(-W / 2 + 0.1, -0.1, ztop + 0.02), (-W / 2 - 0.01, -0.1, ztop - 0.02), (-W / 2 - 0.02, -0.12, ztop - 0.2)], 0.018, shirt, 8, bend=0.04)
    return p.obj("kit_hamper", col)


def medals_stand(col, M):
    """A medal display stand (0.4 m wide, 0.3 m tall, walnut): a weighted base, two uprights and a
    rail with five medals hanging on ribbons in the club colours, gold, silver and bronze. Front +y."""
    walnut = _paint("Off_Walnut", "#5a3b26", 0.35)
    gold = _paint("Hos_Gold", "#d4aa48", 0.2)
    silver = _paint("Hos_Silver", "#d0d4d8", 0.15)
    bronze = _paint("Kit_Bronze", "#a8693a", 0.25)
    cv = Canvas("Kit_Ribbon", 128, 256, col="#ffffff", rough=0.4)
    rr = cv.alloc(0.02, 0.12, 2000)
    rr.fill(col="#ffffff")
    rr.rect(0.0, 0.0, 0.006, 0.12, col="#bdbdbd")
    rr.rect(0.014, 0.0, 0.02, 0.12, col="#bdbdbd")
    ribbon = cv.tint_material("TINT_club_ribbon", "#c8202a", rough=0.4)
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.2, -0.06, 0.0), (0.2, 0.06, 0.025), walnut)
    for x in (-0.18, 0.18):
        p.box((x - 0.012, -0.012, 0.025), (x + 0.012, 0.012, 0.3), walnut)
    p.r = 0.0
    p.cyl((-0.19, 0.0, 0.285), (0.19, 0.0, 0.285), 0.008, M["chrome"], 12)
    for k, m in enumerate((gold, silver, bronze, gold, silver)):
        x = -0.13 + k * 0.065
        p.r = 0.0
        for s in (-1, 1):
            i0 = len(p.F)
            p.face([(x + s * 0.004, 0.012, 0.285), (x + s * 0.024, 0.012, 0.285), (x + s * 0.014, 0.012, 0.16), (x + s * 0.002, 0.012, 0.165)][::(1 if s > 0 else -1)], ribbon)
        p.cyl((x, 0.006, 0.13), (x, 0.016, 0.13), 0.03, m, 28)
        p.cyl((x, 0.016, 0.13), (x, 0.018, 0.13), 0.022, m, 28)
        p.cyl((x, 0.012, 0.162), (x, 0.012, 0.172), 0.004, m, 8)
    return p.obj("medals_stand", col)


def ball_bag(col, M):
    """A mesh ball sack (0.9 m standing) holding twelve size 5 balls, slumped into a pear shape by
    their weight, the black netting drawn tight at the neck by its cord with a toggle and the
    loop handle. Front +y."""
    net = _net_mat("Prop_BallBagNet", "#151617", 0.045, knot=1.6, twist=True)
    bm = _ball_mat()
    p = PM(0.0)
    rng = np.random.default_rng(7)
    R = 0.11
    centres = [(0.0, 0.0, R)] + [(0.2 * math.cos(a), 0.2 * math.sin(a), R) for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
    centres += [(0.12 * math.cos(a + 0.5), 0.12 * math.sin(a + 0.5), R + 0.19) for a in np.linspace(0, 2 * math.pi, 4)[:-1]]
    centres += [(0.0, 0.0, R + 0.38), (0.08, -0.05, R + 0.32)]
    for k, c in enumerate(centres):
        _ball(p, bm, c, R=R, sub=3, rot=_rot(rng), design=("classic", "used", "hivis", "navy")[k % 4])
    _revolve(p, [(0.0, 0.0), (0.315, 0.004), (0.335, 0.11), (0.31, 0.26), (0.245, 0.42), (0.15, 0.56), (0.07, 0.62), (0.035, 0.65)],
             net, sides=40, cell=0.045)
    _revolve(p, [(0.035, 0.65), (0.05, 0.68), (0.045, 0.71), (0.025, 0.72)], net, sides=20, cell=0.045)
    p.r = 0.0
    _tube(p, [(0.036 * math.cos(a), 0.036 * math.sin(a), 0.65) for a in np.linspace(0, 2 * math.pi, 17)], 0.005, M["black"], 6, closed=True)
    p.cyl((0.05, 0.08, 0.56), (0.05, 0.08, 0.61), 0.012, M["red"], 10)
    _tube(p, [(0.03, 0.02, 0.65), (0.05, 0.06, 0.63), (0.05, 0.08, 0.61)], 0.003, M["black"], 5)
    _tube(p, [(-0.025, 0.0, 0.72), (-0.06, 0.0, 0.78), (0.0, 0.0, 0.82), (0.06, 0.0, 0.78), (0.025, 0.0, 0.72)], 0.007, M["black"], 6, bend=0.04)
    return p.obj("ball_bag", col, 60, bevel=False)


def water_carrier(col, M):
    """A squeeze-bottle carrier (0.49 x 0.29 m): a black moulded crate of eight open cells with a
    centre carry handle, holding eight 1 litre sports bottles in translucent white with
    push-pull caps, two in the club colour. Front +y."""
    crate = _plastic("Kit_Crate", "#18191a", 0.55)
    bottle = _plastic("Kit_Bottle", "#e6ebec", 0.25)
    cap = _plastic("Kit_BottleCap", "#c8202a", 0.4)
    cap2 = _plastic("Kit_BottleCap2", "#1d4f91", 0.4)
    p = PM(0.004)
    W, D, H = 0.48, 0.26, 0.17
    p.r = 0.006
    p.box((-W / 2, -D / 2, 0.0), (W / 2, D / 2, 0.02), crate)
    for x in np.linspace(-W / 2, W / 2, 5):
        p.box((x - 0.006, -D / 2, 0.02), (x + 0.006, D / 2, H), crate)
    for y in (-D / 2, 0.0, D / 2):
        p.box((-W / 2, y - 0.006, 0.02), (W / 2, y + 0.006, H), crate)
    for y in (-D / 2, D / 2):
        p.box((-W / 2, y - 0.008, H - 0.03), (W / 2, y + 0.008, H), crate)
        for x in np.linspace(-W / 2 + 0.06, W / 2 - 0.06, 4):
            p.r = 0.0
            p.box((x - 0.02, y - 0.009, 0.05), (x + 0.02, y + 0.009, H - 0.05), _plastic("Kit_CrateSlot", "#0c0c0d", 0.7))
            p.r = 0.006
    # the handle up the middle
    p.box((-0.012, -0.006, H), (0.012, 0.006, 0.33), crate)
    p.r = 0.02
    p.box((-0.08, -0.012, 0.33), (0.08, 0.012, 0.38), crate)
    p.r = 0.0
    p.box((-0.06, -0.013, 0.34), (0.06, 0.013, 0.36), _plastic("Kit_CrateSlot", "#0c0c0d", 0.7))
    # the bottles: a grip waist, a shoulder, the push-pull cap
    for i in range(4):
        for j in range(2):
            x, y = -W / 2 + 0.06 + i * 0.12, (j - 0.5) * D / 2
            if abs(x) < 0.03:
                continue
            p.lathe([(0.0, 0.0), (0.036, 0.0), (0.038, 0.03), (0.032, 0.1), (0.038, 0.17), (0.037, 0.21), (0.024, 0.235), (0.02, 0.24), (0.0, 0.24)],
                    bottle, c=(x, y, 0.022), sides=18)
            c_ = cap2 if (i + j) % 3 == 0 else cap
            p.cyl((x, y, 0.262), (x, y, 0.285), 0.022, c_, 16)
            p.cyl((x, y, 0.285), (x, y, 0.305), 0.009, c_, 10, r2=0.006)
    return p.obj("water_carrier", col)


def foam_rollers(col, M):
    """Six foam rollers (0.6 m x 0.15 m Ø) stacked in a pyramid: EVA foam in black, blue and
    orange with a moulded grid texture, their ends showing the foam ring round a hollow core.
    Front +y."""
    cv = Canvas("Kit_Foam", 512, 512, col="#ffffff", rough=0.9)
    r = cv.alloc(0.6, 0.48, 800)
    r.fill(col="#ffffff")
    r.field(lambda X, Y: ((np.mod(X, 0.03) < 0.004) | (np.mod(Y, 0.03) < 0.004)) * 0.35, col="#7a7a7a", dh=-0.002)
    r.field(lambda X, Y: noise(X.shape, 60, 11, 2) * 0.15, col="#8a8a8a")
    cols = ("#1a1b1d", "#2457a6", "#e5641c")
    mats = [cv.tint_material("TINT_foam_" + c[1:], c, rough=0.9) for c in cols]
    core = _plastic("Kit_FoamCore", "#0c0c0d", 0.8)
    p = PM(0.0)
    R, L = 0.075, 0.6
    k = 0
    for layer, n in enumerate((3, 2, 1)):
        z = R + layer * R * math.sqrt(3)
        for i in range(n):
            y = (i - (n - 1) / 2) * 2 * R
            m = mats[k % 3]
            i0 = len(p.F)
            p.cyl((-L / 2, y, z), (L / 2, y, z), R, m, 28, cap=False)
            for s in (-1, 1):
                p.cyl((s * L / 2, y, z), (s * (L / 2 + 0.002), y, z), R, m, 28)
                p.cyl((s * (L / 2 + 0.002), y, z), (s * (L / 2 + 0.003), y, z), R * 0.4, core, 20)
            k += 1
    return p.obj("foam_rollers", col)


def resistance_bands(col, M):
    """A wall rack of resistance bands (1.2 m bar, 0.75 m up): a black steel bar on two brackets
    with five hooks, each holding two looped latex power bands in the strength colours (yellow,
    red, green, blue, black), hanging to near the floor. The back against the wall at y = 0;
    front +y."""
    steel = _paint("Kit_RackSteel", "#202124", 0.4)
    colours = ("#f0c71c", "#c8202a", "#2f9a3a", "#2457a6", "#141414", "#7a3aa8")
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.6, 0.0, 0.73), (0.6, 0.05, 0.78), steel)
    for x in (-0.56, 0.56):
        p.box((x - 0.03, 0.0, 0.7), (x + 0.03, 0.012, 0.81), steel)
    rng = np.random.default_rng(5)
    for h in range(5):
        x = -0.44 + h * 0.22
        p.r = 0.0
        _tube(p, [(x, 0.05, 0.75), (x, 0.11, 0.75), (x, 0.13, 0.78)], 0.007, steel, 8, bend=0.015)
        for b in range(2):
            c = colours[(h + b * 2) % len(colours)]
            m = _plastic("Kit_Band" + c[1:], c, 0.45)
            dx = (b - 0.5) * 0.02
            bottom = 0.1 + rng.uniform(0.0, 0.08)
            pts = [(x + dx, 0.105, 0.755), (x + dx - 0.022, 0.1 + b * 0.008, 0.6), (x + dx - 0.025, 0.1 + b * 0.008, bottom + 0.05),
                   (x + dx, 0.1 + b * 0.008, bottom), (x + dx + 0.025, 0.1 + b * 0.008, bottom + 0.05), (x + dx + 0.022, 0.1 + b * 0.008, 0.6)]
            _tube(p, pts, 0.0045, m, 6, bend=0.03, closed=True)
    return p.obj("resistance_bands", col)


def bicycle(col, M):
    """A black city bicycle (1.75 m) on its side stand: a step-through frame, 700c wheels with
    32 spokes and cream-wall tyres, full mudguards, a chain guard, swept-back bars with brown
    grips, a sprung brown leather saddle, a wicker basket on the front, a bell and pedals.
    Length along x, the front (+x) wheel; front +y is the drive side."""
    frame = _paint("Kit_BikeFrame", "#141516", 0.3)
    chrome = M["chrome"]
    leather = _paint("Kit_Saddle", "#6a3e22", 0.5)
    rubber = M["rubber"]
    wall = _plastic("Kit_TyreWall", "#d8cfb2", 0.6)
    cv = Canvas("Kit_Basket", 512, 256, col="#a07a4a", rough=0.8)
    br = cv.alloc(0.8, 0.3, 600)
    br.fill(col="#8c6838")
    for y in np.arange(0.0, 0.3, 0.018):
        for x in np.arange(0.0, 0.8, 0.024):
            off = 0.012 if int(y / 0.018) % 2 else 0.0
            br.rect(x + off, y + 0.002, x + off + 0.02, y + 0.016, r=0.005, col="#b8925c", dh=0.002)
    wick = cv.material("Kit_BasketWeave")
    p = PM(0.003)
    R = 0.34
    wb = 1.06
    xr, xf = -wb / 2, wb / 2
    zc = R
    for xw in (xr, xf):
        tyre(p, (xw, 0.0, zc), R, 0.04, R - 0.03, rubber, chrome, sides=40, axis="y")
        _tube(p, [(xw + (R - 0.018) * math.cos(a), 0.0, zc + (R - 0.018) * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 49)], 0.0045, wall, 5, closed=True)
        p.cyl((xw, -0.05, zc), (xw, 0.05, zc), 0.022, chrome, 14)
        for k in range(32):
            a = 2 * math.pi * k / 32
            side = 0.03 if k % 2 else -0.03
            p.cyl((xw + 0.02 * math.cos(a), side, zc + 0.02 * math.sin(a)), (xw + (R - 0.035) * math.cos(a + 0.12 * (1 if k % 2 else -1)), 0.0, zc + (R - 0.035) * math.sin(a + 0.12 * (1 if k % 2 else -1))),
                   0.0012, chrome, 4)
        # mudguard: an arc over the top of the wheel
        a0, a1 = (math.radians(20), math.radians(200)) if xw > 0 else (math.radians(-20), math.radians(160))
        pts = [(xw + (R + 0.025) * math.cos(a), 0.0, zc + (R + 0.025) * math.sin(a)) for a in np.linspace(a0, a1, 18)]
        _tube(p, pts, 0.022, frame, 8)
    # step-through frame: down tube sweeping low, seat tube, stays, head tube and fork
    bb = (-0.02, 0.0, 0.29)
    head_lo, head_hi = (0.36, 0.0, 0.66), (0.33, 0.0, 0.8)
    seat_top = (-0.16, 0.0, 0.86)
    _tube(p, [head_lo, (0.2, 0.0, 0.48), (0.05, 0.0, 0.33), bb], 0.022, frame, 12, bend=0.12)
    _tube(p, [(0.33, 0.0, 0.74), (0.1, 0.0, 0.52), (-0.08, 0.0, 0.42), (-0.12, 0.0, 0.6)], 0.016, frame, 10, bend=0.1)
    p.cyl(bb, seat_top, 0.017, frame, 12)
    for s in (-1, 1):
        p.cyl((bb[0], s * 0.035, bb[2]), (xr, s * 0.05, zc), 0.01, frame, 8)
        p.cyl((-0.1, s * 0.02, 0.72), (xr, s * 0.05, zc), 0.009, frame, 8)
        p.cyl((head_lo[0] + 0.02, s * 0.04, head_lo[2] - 0.02), (xf, s * 0.05, zc), 0.011, frame, 8)
    p.cyl(head_lo, head_hi, 0.024, frame, 14)
    # bars, stem, grips, bell
    p.cyl(head_hi, (0.32, 0.0, 0.9), 0.012, chrome, 10)
    _tube(p, [(0.16, -0.27, 0.95), (0.26, -0.24, 0.93), (0.33, -0.12, 0.9), (0.33, 0.12, 0.9), (0.26, 0.24, 0.93), (0.16, 0.27, 0.95)], 0.011, chrome, 10, bend=0.06)
    for s in (-1, 1):
        p.cyl((0.2, s * 0.26, 0.945), (0.12, s * 0.28, 0.955), 0.017, leather, 10)
    p.sphere((0.3, -0.16, 0.925), 0.02, chrome, 10, 6, z_min=0.0)
    # the sprung saddle
    p.cyl(seat_top, (-0.18, 0.0, 0.94), 0.013, chrome, 10)
    for s in (-1, 1):
        _tube(p, [(-0.27, s * 0.06, 0.93), (-0.27, s * 0.06, 0.96)], 0.012, chrome, 8)
    rings = [_ell(x, 0.0, 0.975, 0.02, w, 14) for x, w in ((-0.3, 0.1), (-0.23, 0.095), (-0.14, 0.04), (-0.08, 0.02))]
    rings = [[(q[0] + 0.0, q[1], q[2] + 0.015 * math.sin(math.pi * j / 13)) for j, q in enumerate(r_)] for r_ in rings]
    _loft(p, rings, leather)
    # chainring, chain, chain guard, cranks, pedals
    p.cyl((bb[0], 0.055, bb[2]), (bb[0], 0.06, bb[2]), 0.1, chrome, 32)
    _tube(p, [(bb[0] + 0.1 * math.cos(a), 0.058, bb[2] + 0.1 * math.sin(a)) for a in np.linspace(math.pi / 2, 3 * math.pi / 2, 9)] +
          [(xr + 0.04 * math.cos(a), 0.058, zc + 0.04 * math.sin(a)) for a in np.linspace(3 * math.pi / 2, 5 * math.pi / 2, 7)], 0.004, M["steel"], 5, closed=True)
    p.r = 0.0
    p.extrude([(bb[0] + 0.12, bb[2] - 0.02), (bb[0] + 0.12, bb[2] + 0.06), (xr + 0.05, zc + 0.07), (xr + 0.02, zc + 0.02)], 0.07, 0.074, frame)
    for s, a in ((1, 0.7), (-1, 0.7 + math.pi)):
        e = (bb[0] + 0.17 * math.cos(a), s * 0.09, bb[2] + 0.17 * math.sin(a))
        p.cyl((bb[0], s * 0.07, bb[2]), e, 0.01, chrome, 8)
        p.r = 0.004
        p.box((e[0] - 0.045, e[1] - 0.04 if s > 0 else e[1] - 0.04, e[2] - 0.01), (e[0] + 0.045, e[1] + 0.05 if s > 0 else e[1] + 0.04, e[2] + 0.01), M["black"])
        p.r = 0.0
    # side stand down to the ground on the -y side
    p.cyl((-0.1, -0.04, 0.3), (-0.22, -0.14, 0.0), 0.009, chrome, 8)
    # the basket over the front wheel
    i0 = len(p.F)
    p.box((xf - 0.05, -0.17, 0.82), (xf + 0.25, 0.17, 1.04), wick)
    box_uv(p, i0, {"+x": br, "-x": br, "+y": br, "-y": br, "-z": br}, (xf - 0.05, -0.17, 0.82), (xf + 0.25, 0.17, 1.04))
    p.box((xf - 0.04, -0.16, 0.83), (xf + 0.24, 0.16, 1.035), _paint("Kit_BasketIn", "#5a4024", 0.9))
    _tube(p, _rrect_ring(xf + 0.1, 0.0, 1.04, 0.31, 0.35, 0.03), 0.009, wick, 6, closed=True)
    return p.obj("bicycle", col)


def kit_bag(col, M):
    """A players' holdall (0.75 x 0.36 x 0.4 m): a ripstop body in the club colour with black end
    panels and base, a two-way zip along the top with pullers, webbing carry handles joined by a
    wrap, a padded shoulder strap lying over the end, a boot compartment vent on one end. Front +y."""
    cv = Canvas("Kit_BagFabric", 512, 512, col="#ffffff", rough=0.8)
    fr = cv.alloc(0.8, 0.8, 400)
    fr.fill(col="#ffffff")
    fr.field(lambda X, Y: ((np.mod(X, 0.006) < 0.0008) | (np.mod(Y, 0.006) < 0.0008)) * 0.18, col="#7a7a7a")
    fr.field(lambda X, Y: noise(X.shape, 20, 5, 2) * 0.12, col="#8a8a8a")
    body = cv.tint_material("TINT_club_bag", "#c8202a", rough=0.75)
    black = _plastic("Kit_BagBlack", "#18191a", 0.7)
    zipm = _plastic("Kit_Zip", "#0e0e0f", 0.4)
    p = PM(0.004)
    L, W, H = 0.72, 0.34, 0.38
    xs = list(np.linspace(-L / 2, L / 2, 11))
    rings = []
    for x in xs:
        pinch = 1.0 - 0.05 * (2 * x / L) ** 2
        ring = []
        for k in range(24):
            t = 2 * math.pi * k / 24
            c, s_ = math.cos(t), math.sin(t)
            # (a squarish holdall: a superellipse section, flat underneath)
            y = W / 2 * pinch * math.copysign(abs(c) ** 0.55, c)
            z = H / 2 + H / 2 * pinch * math.copysign(abs(s_) ** 0.55, s_)
            ring.append((x, y, max(z, 0.0)))
        rings.append(ring)
    i0 = len(p.F)
    n_seg = len(xs) - 1
    _loft(p, rings, black, mats=lambda i_, j_: black if i_ in (0, n_seg - 1) else body)
    box_uv(p, i0, {"+y": fr, "-y": fr, "+z": fr, "-z": fr, "+x": fr, "-x": fr}, (-L / 2, -W / 2, 0.0), (L / 2, W / 2, H))
    p.r = 0.02
    p.box((-L / 2 + 0.03, -W / 2 + 0.03, 0.0), (L / 2 - 0.03, W / 2 - 0.03, 0.02), black)
    # zip track along the top, two pullers
    p.r = 0.0
    p.box((-L / 2 + 0.04, -0.012, H - 0.004), (L / 2 - 0.04, 0.012, H + 0.004), zipm)
    for x in (-0.05, 0.05):
        p.box((x - 0.012, -0.008, H + 0.004), (x + 0.012, 0.008, H + 0.012), M["chrome"])
        p.box((x - 0.004, 0.008, H + 0.002), (x + 0.004, 0.03, H + 0.008), M["black"])
    # carry handles meeting over the top, with a wrap
    for s in (-1, 1):
        _tube(p, [(-0.14, s * 0.13, H * 0.85), (-0.1, s * 0.05, H + 0.1), (0.1, s * 0.05, H + 0.1), (0.14, s * 0.13, H * 0.85)], 0.008, black, 8, bend=0.05)
    p.r = 0.01
    p.box((-0.06, -0.06, H + 0.085), (0.06, 0.06, H + 0.115), black)
    # the shoulder strap clipped to the ends, lying along the top
    for x in (-L / 2 + 0.06, L / 2 - 0.06):
        p.r = 0.004
        p.box((x - 0.02, -0.025, H - 0.01), (x + 0.02, 0.025, H + 0.012), M["chrome"])
    p.r = 0.003
    p.box((-L / 2 + 0.06, 0.04, H + 0.003), (L / 2 - 0.06, 0.08, H + 0.012), black)
    # the boot vent
    p.r = 0.0
    p.box((-L / 2 - 0.004, -0.07, 0.13), (-L / 2 + 0.004, 0.07, 0.25), _plastic("Kit_Vent", "#2a2b2e", 0.8))
    return p.obj("kit_bag", col)


def tactics_easel(col, M):
    """A coaches' magnetic tactics board (0.9 x 0.6 m, a printed pitch in an aluminium frame) on a
    telescopic tripod easel at 1.8 m, red and blue player magnets set out in a 4-3-3 against a
    4-4-2, a marker in the tray. Front +y."""
    cv = Canvas("Kit_Tactics", 1024, 1024, col="#ffffff", rough=0.3)
    b = cv.alloc(0.9, 0.6, 1100)
    b.fill(col="#2f7a34")
    b.field(lambda X, Y: (np.floor(X / 0.075) % 2 == 0) * 0.3, col="#3b8a3e")
    lw = 0.004
    for pts in ([(0.03, 0.03), (0.87, 0.03), (0.87, 0.57), (0.03, 0.57), (0.03, 0.03)], [(0.45, 0.03), (0.45, 0.57)],
                [(0.03, 0.17), (0.16, 0.17), (0.16, 0.43), (0.03, 0.43)], [(0.87, 0.17), (0.74, 0.17), (0.74, 0.43), (0.87, 0.43)],
                [(0.03, 0.24), (0.07, 0.24), (0.07, 0.36), (0.03, 0.36)], [(0.87, 0.24), (0.83, 0.24), (0.83, 0.36), (0.87, 0.36)]):
        b.line(pts, lw, col="#eef2ee")
    b.line([(0.45 + 0.08 * math.cos(t), 0.3 + 0.08 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 33)], lw, col="#eef2ee")
    b.circle(0.45, 0.3, 0.006, col="#eef2ee")
    mat = cv.material("Kit_TacticsFace", rough=0.25)
    alu = M["alu"]
    red = _plastic("Kit_MagRed", "#c8202a", 0.35)
    blue = _plastic("Kit_MagBlue", "#2457a6", 0.35)
    p = PM(0.004)
    tilt = math.radians(10)
    zc, yc = 1.32, 0.02
    n = (0.0, math.cos(tilt), math.sin(tilt))
    up = (0.0, -math.sin(tilt), math.cos(tilt))

    def on(u, v, w=0.0):
        return (u, yc + v * up[1] + w * n[1], zc + v * up[2] + w * n[2])
    p.r = 0.0
    p.obox((0.0, yc - 0.006 * n[1], zc - 0.006 * n[2]), (0.9, 0.012, 0.6), _plastic("Kit_BoardBack", "#e6e6e2", 0.4), pitch=tilt)
    decal(p, on(0, 0), n, up, 0.9, 0.6, mat, b, lift=0.001)
    p.r = 0.004
    for (u0, v0, u1, v1) in ((-0.46, -0.31, 0.46, -0.29), (-0.46, 0.29, 0.46, 0.31), (-0.46, -0.31, -0.44, 0.31), (0.44, -0.31, 0.46, 0.31)):
        p.obox(on((u0 + u1) / 2, (v0 + v1) / 2, 0.004), (u1 - u0, 0.012, v1 - v0), alu, pitch=tilt)
    # tray along the bottom with a marker
    p.obox(on(0.0, -0.32, 0.03), (0.6, 0.06, 0.012), alu, pitch=tilt)
    p.cyl(on(-0.1, -0.31, 0.04), on(0.05, -0.31, 0.04), 0.008, M["black"], 8)
    # magnets: a 4-3-3 (red) and a 4-4-2 (blue)
    home = [(-0.4, 0.0), (-0.28, -0.2), (-0.3, -0.07), (-0.3, 0.07), (-0.28, 0.2), (-0.15, -0.13), (-0.17, 0.0), (-0.15, 0.13), (-0.03, -0.18), (-0.01, 0.0), (-0.03, 0.18)]
    away = [(0.4, 0.0), (0.27, -0.2), (0.29, -0.07), (0.29, 0.07), (0.27, 0.2), (0.14, -0.2), (0.15, -0.07), (0.15, 0.07), (0.14, 0.2), (0.05, -0.06), (0.05, 0.06)]
    for pts, m in ((home, red), (away, blue)):
        for (u, v) in pts:
            p.r = 0.0
            p.cyl(on(u, v, 0.001), on(u, v, 0.009), 0.014, m, 14)
    # the tripod easel: three telescopic legs, a mast, the board clamp
    p.r = 0.0
    top = (0.0, yc - 0.06, 1.8)
    for (fx, fy) in ((-0.5, 0.25), (0.5, 0.25), (0.0, -0.55)):
        p.cyl((fx, fy, 0.0), (top[0] + fx * 0.04, top[1] + fy * 0.04, 1.0), 0.012, alu, 8)
        p.cyl((top[0] + fx * 0.04, top[1] + fy * 0.04, 1.0), (0.0, yc - 0.06, 1.05), 0.01, alu, 8)
        p.cyl((fx, fy, 0.0), (fx, fy, 0.02), 0.016, M["rubber"], 8)
        p.cyl((fx * 0.45, fy * 0.45, 0.48), (0.0, yc - 0.06, 0.62), 0.006, alu, 6)
    p.cyl((0.0, yc - 0.06, 1.0), top, 0.014, alu, 10)
    p.r = 0.006
    p.box((-0.04, yc - 0.09, 1.62), (0.04, yc - 0.03, 1.68), M["black"])
    p.box((-0.04, yc - 0.09, 0.98), (0.04, yc + 0.0, 1.03), M["black"])
    return p.obj("tactics_easel", col)


def flags_tifo(col, M):
    """A seller's rack of hand flags (1.6 m across): a weighted base and a crossbar holding five
    flags on 1.2 m poles, each a 0.9 x 0.6 m flag in the club colours with a white band and a
    simple star, fluttering slightly. Front +y."""
    cv = Canvas("Kit_Flag", 512, 512, col="#ffffff", rough=0.75)
    fr = cv.alloc(0.9, 0.6, 500)
    fr.fill(col="#ffffff")
    fr.rect(0.0, 0.0, 0.9, 0.22, col="#9a9a9a")
    fr.rect(0.0, 0.38, 0.9, 0.6, col="#9a9a9a")
    star = [(0.45 + 0.09 * math.cos(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.42),
             0.3 + 0.09 * math.sin(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.42)) for k in range(10)]
    fr.poly(star, col="#9a9a9a")
    flag = cv.tint_material("TINT_flag_tifo", "#c8202a", rough=0.75)
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.8, -0.2, 0.0), (0.8, 0.2, 0.06), _paint("Kit_RackBase", "#2a2b2e", 0.5))
    p.r = 0.0
    for x in (-0.75, 0.75):
        p.cyl((x, 0.0, 0.06), (x, 0.0, 0.7), 0.014, M["alu"], 10)
    p.cyl((-0.76, 0.0, 0.7), (0.76, 0.0, 0.7), 0.014, M["alu"], 10)
    for k in range(5):
        x0 = -0.6 + k * 0.3
        # (fanned: each flag flies off at its own angle, the middle one straight out to the front)
        phi = math.pi / 2 + (k - 2) * 0.42
        d = (math.cos(phi), math.sin(phi))
        base, tip = (x0, 0.0, 0.06), (x0 + d[0] * 0.05, d[1] * 0.05, 1.22 + 0.04 * (k % 2))
        p.cyl(base, tip, 0.008, _paint("Kit_FlagPole", "#e8e4dc", 0.5), 8)
        nu, nv = 12, 6
        V, uv = [], []
        for i_ in range(nu + 1):
            for j_ in range(nv + 1):
                u, v = i_ / nu, j_ / nv
                along = u * 0.9
                ripple = 0.05 * math.sin(u * 2.6 * math.pi + k) * u
                x = tip[0] + d[0] * along - d[1] * ripple
                y = tip[1] + d[1] * along + d[0] * ripple
                z = tip[2] - 0.02 - (1 - v) * 0.6 - u * 0.15
                V.append((x, y, z))
                uv.append(fr.uv(u * 0.9, v * 0.6))
        F = [(i_ * (nv + 1) + j_, (i_ + 1) * (nv + 1) + j_, (i_ + 1) * (nv + 1) + j_ + 1, i_ * (nv + 1) + j_ + 1) for i_ in range(nu) for j_ in range(nv)]
        p.faces(V, F, flag, uvs=uv)
    return p.obj("flags_tifo", col)


def matchday_litter(col, M):
    """Matchday litter (0.6 m across): a crushed paper coffee cup with its lid off, a flattened
    plastic pint cup, a scrunched crisp packet, a dropped match programme, an open burger box,
    a balled-up napkin, a couple of bottle caps. Front +y."""
    cv = Canvas("Kit_Litter", 512, 512, col="#ffffff", rough=0.7)
    prog = cv.alloc(0.15, 0.21, 1200)
    prog.fill(col="#c8202a")
    prog.rect(0.0, 0.16, 0.15, 0.21, col="#141a28")
    prog.text("MATCHDAY", 0.075, 0.178, 0.018, weight="black", align="c", col="#ffffff")
    prog.text("PROGRAMME", 0.075, 0.025, 0.012, weight="bold", align="c", col="#ffffff")
    prog.rect(0.02, 0.05, 0.13, 0.14, col="#e8b0a0")
    crisp = cv.alloc(0.14, 0.18, 1200)
    crisp.fill(col="#2457a6")
    crisp.rect(0.0, 0.07, 0.14, 0.11, col="#f0c71c")
    crisp.text("CRISPS", 0.07, 0.078, 0.022, weight="black", align="c", col="#c8202a")
    mat = cv.material("Kit_LitterPrint", rough=0.55)
    paper = _paint("Kit_CupPaper", "#f0ece2", 0.7)
    sleeve = _paint("Kit_CupSleeve", "#8a6038", 0.8)
    pet = clear_glass()
    box = _paint("Kit_BurgerBox", "#e6dccb", 0.8)
    p = PM(0.002)
    # the crushed coffee cup lying on its side
    rings = []
    for k, z in enumerate(np.linspace(0.0, 0.11, 6)):
        r = 0.03 + 0.012 * z / 0.11
        squash = 0.55 if 2 <= k <= 3 else 0.85
        rings.append([(-0.15 + z, 0.05 + r * math.cos(a), r * squash * (1 + math.sin(a)) + 0.002) for a in np.linspace(0, 2 * math.pi, 17)[:-1]])
    _loft(p, rings, paper, cap1=False)
    p.r = 0.0
    p.obox((-0.1, 0.05, 0.03), (0.04, 0.083, 0.06), sleeve, roll=0.1)
    p.cyl((-0.02, 0.12, 0.0), (-0.02, 0.12, 0.008), 0.044, _plastic("Kit_Lid", "#f4f4f2", 0.4), 18)
    # the flattened pint cup
    p.r = 0.0
    p.face([(0.05, -0.15, 0.003), (0.2, -0.17, 0.003), (0.22, -0.08, 0.01), (0.06, -0.06, 0.012)], pet)
    p.face([(0.06, -0.06, 0.012), (0.22, -0.08, 0.01), (0.2, -0.04, 0.02), (0.07, -0.03, 0.018)], pet)
    # crisp packet, scrunched (a crumpled quad strip)
    rng = np.random.default_rng(31)
    V, uv, F = [], [], []
    for i in range(5):
        for j in range(4):
            u, v = i / 4, j / 3
            V.append((0.12 + u * 0.13, 0.08 + v * 0.16, 0.004 + abs(rng.normal(0, 0.012))))
            uv.append(crisp.uv(u * 0.14, v * 0.18))
    F = [(i * 4 + j, (i + 1) * 4 + j, (i + 1) * 4 + j + 1, i * 4 + j + 1) for i in range(4) for j in range(3)]
    p.faces(V, F, mat, uvs=uv)
    # the programme, open slightly
    p.r = 0.0
    decal(p, (-0.14, -0.13, 0.004), (0, 0, 1), (-0.3, -0.95, 0), 0.15, 0.21, mat, prog, lift=0.002)
    p.obox((-0.14, -0.13, 0.003), (0.16, 0.22, 0.004), _paint("Kit_ProgPages", "#f2efe6", 0.8), yaw=-0.3)
    # the burger box, open clamshell
    p.r = 0.004
    p.box((0.16, 0.18, 0.0), (0.26, 0.28, 0.03), box)
    p.obox((0.21, 0.31, 0.025), (0.1, 0.1, 0.006), box, pitch=0.5)
    # a balled napkin and bottle caps
    p.sphere((0.02, 0.1, 0.018), 0.022, _paint("Kit_Napkin", "#f6f4ee", 0.9), 10, 7, scale=(1.1, 0.9, 0.8))
    for x, y in ((0.0, -0.2), (0.27, 0.0)):
        p.r = 0.0
        p.cyl((x, y, 0.0), (x, y, 0.006), 0.014, M["red"], 12)
    return p.obj("matchday_litter", col)


def bibs_stack(col, M):
    """A stack of folded mesh training bibs (0.25 m high) in four colours, sorted roughly by
    colour, each folded in half with its binding showing, and one unfolded bib slung over the
    top. Front +y."""
    cv = Canvas("Kit_Bib", 512, 512, col="#ffffff", rough=0.85)
    r = cv.alloc(0.4, 0.4, 800)
    r.fill(col="#ffffff")
    r.field(lambda X, Y: ((np.mod(X, 0.006) < 0.0025) & (np.mod(Y, 0.006) < 0.0025)) * 0.45, col="#5a5a5a", dh=-0.0005)
    cols = ("#c8f01c", "#ff6a1a", "#2457a6", "#2fbf3a")
    mats = [cv.tint_material("TINT_bib_" + c[1:], c, rough=0.85) for c in cols]
    bind = _plastic("Kit_BibBinding", "#202124", 0.7)
    p = PM(0.003)
    rng = np.random.default_rng(23)
    z = 0.0
    for k in range(18):
        m = mats[min(k // 5, 3)]
        h = 0.012
        yaw = rng.uniform(-0.08, 0.08)
        dx, dy = rng.uniform(-0.012, 0.012), rng.uniform(-0.012, 0.012)
        p.r = 0.006
        i0 = len(p.F)
        p.obox((dx, dy, z + h / 2), (0.4, 0.3, h), m, yaw=yaw)
        box_uv(p, i0, {"+z": r, "+y": r, "-y": r, "+x": r, "-x": r}, (-0.2, -0.15, z), (0.2, 0.15, z + h))
        p.r = 0.0
        p.obox((dx, dy + 0.15 * math.cos(yaw), z + h / 2), (0.38, 0.004, h * 0.6), bind, yaw=yaw)
        z += h
    # one bib slung over the top and down the front
    m = mats[0]
    p.r = 0.004
    i0 = len(p.F)
    p.obox((0.02, 0.0, z + 0.004), (0.42, 0.3, 0.008), m, yaw=0.15)
    box_uv(p, i0, {"+z": r, "+y": r, "-y": r}, (-0.22, -0.16, z), (0.22, 0.16, z + 0.01))
    _tube(p, [(-0.12, 0.17, z + 0.006), (-0.1, 0.22, z - 0.02), (-0.08, 0.2, z - 0.08)], 0.006, bind, 6, bend=0.02)
    return p.obj("bibs_stack", col)


BUILDERS = {
    "prop.gk_gloves": gk_gloves, "prop.swiss_ball": swiss_ball, "prop.plyo_boxes": plyo_boxes, "prop.kit_hamper": kit_hamper,
    "prop.medals_stand": medals_stand, "prop.ball_bag": ball_bag, "prop.water_carrier": water_carrier, "prop.foam_rollers": foam_rollers,
    "prop.resistance_bands": resistance_bands, "prop.bicycle": bicycle, "prop.kit_bag": kit_bag, "prop.tactics_easel": tactics_easel,
    "prop.flags_tifo": flags_tifo, "prop.matchday_litter": matchday_litter, "prop.bibs_stack": bibs_stack,
}
