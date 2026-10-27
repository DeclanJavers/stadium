"""Grounds and construction machines at real size: a John Deere 6120M-class utility tractor, an
X350-class ride-on mower, a 13 t (CAT 313-class) crawler excavator, a 6 t front-tip site dumper
(Thwaites MACH2265 class) and a 2.7 t twin-drum roller (Bomag BW 120 class). Brand-neutral.

Bodies are lofted rounded sections; booms and frames are variable-depth box sections; every ram
is a barrel and chromed rod pinned between lugs; wheels carry real tread bars and wheel nuts.
Surfaces are image-textured worn paint, bare steel, dusty rubber and tread plate
(machine_paint), with a decal atlas per machine and mud laid into the vertex colours.
Origin on the ground at the footprint centre, +Y the front.
"""
import math

import numpy as np

from . import machine_paint as MP
from .props import PM, _paint, _plastic, emissive, round_path, rrect, tyre


# -- shared parts ---------------------------------------------------------------------------------

def beam(p, a, b, w, h, mat, up=(0, 0, 1)):
    """A rectangular section w x h swept from a to b (h measured towards `up`)."""
    prof = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    p.sweep([a, b], prof, mat, twist_ref=up)


def ram(p, a, b, r, M, frac=0.55, body=None):
    """A hydraulic cylinder from a to b: barrel (frac of the length), chromed rod, end eyes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a + (b - a) * frac
    p.cyl(a, m, r, body or M["yellow"], 14)
    p.cyl(m - (b - a) * 0.02, b, r * 0.5, M["chrome"], 12)
    for q in (a, b):
        p.sphere(q, r * 0.9, M["steel"], 10, 6)


def xlathe(p, c, prof, mat, sides=32, mats=None):
    """Surface of revolution about the x axis through c of (radius, dx) points."""
    c = np.asarray(c, float)
    G = [[c + np.array([dx, r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides)]) for r, dx in prof]
         for k in range(sides)]
    m = len(prof)
    for k in range(sides):
        k2 = (k + 1) % sides
        for j in range(m - 1):
            p.face([G[k][j], G[k2][j], G[k2][j + 1], G[k][j + 1]], mats(j) if mats else mat)


def xcyl(p, y, z, x0, x1, r, mat, sides=14, r2=None):
    """A cylinder along x (pins, axles, hubs)."""
    p.cyl((x0, y, z), (x1, y, z), r, mat, sides, r2=r2)


def lug(p, y, z, x0, x1, r, to, mat, n=10):
    """A plate between x0 and x1 round a pin at (y, z) of radius r, its neck running to `to`
    (y, z) where it is welded on."""
    to = np.asarray(to, float)
    c = np.array([y, z])
    d = to - c
    L = np.linalg.norm(d)
    d /= L
    nrm = np.array([-d[1], d[0]])
    a0 = math.atan2(nrm[1], nrm[0])
    poly = [tuple(c + np.array([math.cos(a0 + math.pi * k / n), math.sin(a0 + math.pi * k / n)]) * r) for k in range(n + 1)]
    poly += [tuple(to - nrm * r * 1.1), tuple(to + nrm * r * 1.1)]
    p.extrude_yz(poly, x0, x1, mat)


def ram2(p, a, b, r, body, chrome, frac=0.56, eye=None, pin_mat=None):
    """A hydraulic cylinder pinned across x at a (barrel end) and b (rod end): barrel with a
    gland nut and an end boss, chromed rod, eyes round cross pins."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    eye = eye or r * 0.75
    m = a + d * frac
    p.cyl(a + u * eye * 0.6, m, r, body, 16)
    p.cyl(m - u * 0.01, m + u * 0.05, r * 0.8, body, 16)
    p.cyl(m + u * 0.04, b - u * eye * 0.6, r * 0.45, chrome, 12)
    for q in (a, b):
        p.cyl(q - np.array([eye * 0.55, 0, 0]), q + np.array([eye * 0.55, 0, 0]), eye, body, 14)
        p.cyl(q - np.array([eye * 0.75, 0, 0]), q + np.array([eye * 0.75, 0, 0]), eye * 0.42, pin_mat or chrome, 10)


def vbeam(p, pts, x, w, hs, mat, ch=0.035, ws=None):
    """A box section in the yz plane at x: centreline pts [(y, z)], depth hs[i] (and width ws[i])
    at each, corners chamfered by ch."""
    P = np.asarray(pts, float)
    n = len(P)
    rings = []
    for i in range(n):
        t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        nn = np.array([-t[1], t[0]])
        h = hs[i] / 2
        ww = (ws[i] if ws is not None else w) / 2
        c = min(ch, h * 0.4, ww * 0.4)
        ring = []
        for xx, vv in ((ww, -h + c), (ww, h - c), (ww - c, h), (-ww + c, h), (-ww, h - c), (-ww, -h + c), (-ww + c, -h), (ww - c, -h)):
            q = P[i] + nn * vv
            ring.append((x + xx, q[0], q[1]))
        rings.append(ring)
    p.loft(rings, mat)


def xsec(w, z0, z1, rt, rb, y, cx=0.0, n=4):
    """A cross-section ring in the xz plane at y: w wide from z0 to z1, top corners rounded rt,
    bottom corners rb."""
    out = []
    corners = ((1, 1, 0.0, rt), (-1, 1, 90.0, rt), (-1, -1, 180.0, rb), (1, -1, 270.0, rb))
    for qx, qz, a0, r in corners:
        r = max(min(r, w / 2 - 1e-4, (z1 - z0) / 2 - 1e-4), 1e-4)
        ccx = cx + qx * (w / 2 - r)
        ccz = (z1 - r) if qz > 0 else (z0 + r)
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            out.append((ccx + r * math.cos(a), y, ccz + r * math.sin(a)))
    return out


def sbar(p, pts, w, h, mat, bend=0.12, up=(0, 0, 1)):
    """A box-section bar along a polyline with bent corners (ROPS frames, handrails)."""
    path = round_path(pts, bend, n=6) if len(pts) > 2 else pts
    prof = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    p.sweep(path, prof, mat, twist_ref=up)


def lamp(p, c, n, w, h, d, housing, lens):
    """A rectangular work lamp: housing box behind a lens facing n (horizontal)."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    yaw = math.atan2(-n[0], n[1])
    p.obox(tuple(c - n * d / 2), (w, d, h), housing, yaw=yaw)
    r, p.r = p.r, 0.0
    u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    q = c + n * 0.002
    p.face([q - u * w * 0.42 - np.array([0, 0, h * 0.4]), q + u * w * 0.42 - np.array([0, 0, h * 0.4]),
            q + u * w * 0.42 + np.array([0, 0, h * 0.4]), q - u * w * 0.42 + np.array([0, 0, h * 0.4])], lens)
    p.r = r


def beacon(p, c, M, mats):
    """An amber rotating beacon on a black base."""
    x, y, z = c
    p.cyl((x, y, z), (x, y, z + 0.04), 0.075, M["black"], 16, r2=0.068)
    p.cyl((x, y, z + 0.04), (x, y, z + 0.13), 0.062, mats["amber"], 16, r2=0.056)
    p.sphere((x, y, z + 0.13), 0.056, mats["amber"], 16, 8, z_min=0.0)


def seat(p, c, mats, w=0.5, back=0.45, yaw=0.0, arm=False):
    """A moulded suspension seat facing +y (turned by yaw): dished cushion, contoured back."""
    cx, cy, cz = c
    cs, sn = math.cos(yaw), math.sin(yaw)

    def T(x, y, z):
        return (cx + x * cs - y * sn, cy + x * sn + y * cs, cz + z)
    blk = mats["seat"]
    r0 = p.r
    p.r = 0.03
    # cushion: lofted rounded rings, raised bolsters at the sides
    rings = []
    for z, s in ((0.0, 0.9), (0.04, 1.0), (0.09, 0.97)):
        rings.append([T(x * s, y * s, z + (0.015 if abs(x) > w * 0.3 else 0.0)) for x, y, _ in rrect(w, w * 0.92, 0.09)])
    p.loft(rings, blk)
    # backrest: a curved shell
    rings = []
    for k, (z, s) in enumerate(((0.1, 1.0), (0.1 + back * 0.5, 1.02), (0.1 + back, 0.86))):
        ring = []
        for x, y, _ in rrect(w * s, 0.1, 0.045):
            yy = -w * 0.46 - 0.05 - k * 0.04 + y - 0.04 * (x / (w / 2)) ** 2
            ring.append(T(x, yy, z))
        rings.append(ring)
    p.loft(rings, blk)
    if arm:
        for sx in (-1, 1):
            p.obox(T(sx * (w / 2 + 0.03), -0.05, 0.2), (0.06, 0.32, 0.05), blk, yaw=yaw)
    # suspension base (bellows) and the slide rails
    p.r = 0.01
    p.cyl(T(0, 0, -0.12), T(0, 0, 0.0), w * 0.3, mats["rubberblk"], 14)
    p.box((cx - w * 0.38, cy - w * 0.4, cz - 0.14), (cx + w * 0.38, cy + w * 0.4, cz - 0.11), mats["dark"])
    p.r = r0


def track(p, x, L, r, w, M, links=46):
    """A crawler track loop at lateral offset x: length L between the idler and sprocket centres
    (wheel radius r), steel shoes with grousers around a stadium path."""
    path = []
    n_arc = 10
    for k in range(n_arc + 1):
        a = -math.pi / 2 + math.pi * k / n_arc
        path.append((L / 2 + r * math.cos(a), r + r * math.sin(a)))
    for k in range(n_arc + 1):
        a = math.pi / 2 + math.pi * k / n_arc
        path.append((-L / 2 + r * math.cos(a), r + r * math.sin(a)))
    seg = [np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    seg.append(np.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]))
    total = sum(seg)
    pitch = total / links
    pts = []
    acc = 0.0
    i = 0
    for k in range(links):
        t = k * pitch
        while acc + seg[i] < t:
            acc += seg[i]
            i = (i + 1) % len(path)
        f = (t - acc) / seg[i]
        a, b = np.array(path[i]), np.array(path[(i + 1) % len(path)])
        q = a + (b - a) * f
        d = (b - a) / np.linalg.norm(b - a)
        pts.append((q, d))
    shoe = M["track"]
    for q, d in pts:
        nrm = np.array([d[1], -d[0]])           # outward from the loop (the path runs clockwise)
        c = np.array([x, q[0], q[1]])
        ax = np.array([0, d[0], d[1]])
        up = np.array([0, nrm[0], nrm[1]])
        # shoe plate and a grouser bar
        P = []
        for s_ in (-1, 1):
            for t_ in (-1, 1):
                P.append(c + np.array([s_ * w / 2, 0, 0]) + ax * t_ * pitch * 0.46)
        base = [P[0], P[1], P[3], P[2]]
        top = [v + up * 0.03 for v in base]
        p.faces(base + top, [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], shoe)
        g0 = c + up * 0.03
        p.faces([g0 + np.array([-w / 2, 0, 0]) - ax * 0.02, g0 + np.array([w / 2, 0, 0]) - ax * 0.02,
                 g0 + np.array([w / 2, 0, 0]) + ax * 0.02, g0 + np.array([-w / 2, 0, 0]) + ax * 0.02],
                [(0, 1, 2, 3)], shoe)


def _rod(name="Mach_Rod"):
    """Hard-chromed ram rods and bright pins: polished, but bright enough to read as chrome under
    a plain sky (a pure mirror goes black without reflections to show)."""
    import bpy
    from ..core.nodes import material, srgb
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#e4e6e8"), Metallic=0.8, Roughness=0.1))
    return m


def _sharp(X):
    """Materials of small hardware that is left unbevelled."""
    return {X[k] for k in ("nut", "chrome", "track", "black", "rubberblk", "lens", "amber", "tail", "galv", "steel", "red")}


def pan_seat(p, c, w, mat):
    """A one-piece moulded pan seat (lawn tractors): a dished U-section swept from the front of
    the cushion round into a mid-height back."""
    c = np.asarray(c, float)
    side = [(0.2, 0.02), (0.1, 0.0), (-0.1, 0.0), (-0.17, 0.03), (-0.21, 0.12), (-0.24, 0.24), (-0.26, 0.33)]
    path = [c + np.array([0.0, y, z]) for y, z in side]
    h = w / 2
    prof = [(0.07, -h), (0.0, -h + 0.03), (0.0, h - 0.03), (0.07, h), (0.075, h - 0.025), (0.035, h - 0.05),
            (0.035, -h + 0.05), (0.075, -h + 0.025)]
    r, p.r = p.r, 0.012
    p.sweep(round_path(path, 0.06, n=4), prof, mat, twist_ref=(1, 0, 0))
    p.r = r


def machine_mats(M):
    """The machines' materials (made once): textured paints, steel, rubber, glass, lamps."""
    return {
        "yellow": MP.worn_paint("Mach_Yellow", "#e0a10f", 0.48, 1.2, seed=11, wear=0.6, dust=0.15, dust_col="#7a6440"),
        "ryellow": MP.worn_paint("Mach_RollerYellow", "#e2b416", 0.42, 1.2, seed=12, wear=0.3, dust=0.2, mottle=0.5),
        "green": MP.worn_paint("Mach_Green", "#367c2b", 0.25, 1.4, seed=13, wear=0.08, dust=0.08, chip="#6a6a62", mottle=0.3, streak=0.2),
        "engine": MP.worn_paint("Mach_Engine", "#585c5c", 0.5, 1.0, seed=21, wear=0.2, dust=0.6, chip="#5a5650"),
        "perf": MP.perforated(),
        "mgreen": MP.worn_paint("Mach_MowerGreen", "#367c2b", 0.3, 1.0, seed=14, wear=0.0, dust=0.1, mottle=0.3, streak=0.15),
        "jdyellow": MP.worn_paint("Mach_RimYellow", "#f2c200", 0.35, 0.8, seed=15, wear=0.4, dust=0.35, chip="#6b4a2a", mottle=0.5),
        "dark": MP.worn_paint("Mach_Dark", "#2b2c2d", 0.5, 1.0, seed=16, wear=0.3, dust=0.4, chip="#4f4b45", dust_col="#5e574c"),
        "frame": MP.worn_paint("Mach_Frame", "#222324", 0.5, 1.0, seed=17, wear=0.4, dust=0.5, chip="#4e4840", dust_col="#5e574c"),
        "skipsteel": MP.bare_steel("Mach_SkipSteel", "#77736c", 0.55, 1.2, seed=22, rust=0.3, splat=0.7, splat_col="#9a9890", metallic=0.4),
        "steel": MP.bare_steel("Mach_BareSteel", "#78746d", 0.45, 1.2, seed=18, rust=0.35),
        "drum": MP.bare_steel("Mach_Drum", "#c2c1bc", 0.22, 1.0, seed=19, rust=0.05, scrape=1.0, splat=0.8, metallic=0.75),
        "tyre": MP.dirty_rubber(),
        "track": MP.track_steel(),
        "checker": MP.checker_plate(),
        "seat": _plastic("Mach_Seat", "#1d1e1f", 0.6),
        "rubberblk": _plastic("Mach_RubberBlk", "#151515", 0.8),
        "amber": emissive("Mach_Amber", "#ff9b1e", 0.6),
        "glass": M["clear"],
        "lens": M["lens"],
        "tail": M["tail"],
        "chrome": _rod(),
        "black": M["black"],
        "nut": _paint("Mach_Nut", "#3b3a37", 0.45),
        "galv": M["galv"],
        "red": M["red"],
        "blue": M["blue"],
    }


def roller(col, M):
    """A 2.7 t twin-drum tandem roller (BW 120 class: L 2.53, W 1.27, WB 1.75, drums 1.2 m wide,
    0.7 m across): smooth steel drums held in plate frames with scrapers, spray bars, drive motors
    and hubs; a water-tank hood at the front, the engine hood behind with louvres and exhaust; the
    operator's tread-plate deck over the articulation joint (pivot block and steering ram in plain
    view); a suspension seat on the engine hood, a steering column and dash; a folding ROPS with a
    beacon and work lamps."""
    p = PM(0.02)
    X = machine_mats(M)
    Y, D = X["ryellow"], X["frame"]
    A = MP.Atlas("Mach_RollerDecals", 1024)
    A.add("model", MP.plate("TR120 AD", 560, 120, "#141414", None, bold=0.035))
    A.add("brand", MP.plate("VIBROMAC", 560, 110, "#efb80c", "#161616", bold=0.02))
    A.add("haz", MP.chevrons(512, 64, period=60))
    A.add("rear", MP.chevron_v(420, 120))
    A.add("warn", MP.warning(128))
    A.add("louvre", MP.louvre(320, 170, 8))
    A.add("artic", MP.plate("DANGER - ARTICULATION ZONE", 640, 64, "#111111", "#f2c200", border="#111111"))
    A.add("rops", MP.plate("ROPS ISO 3471", 256, 64, "#222222", "#d9d9d4", border="#555555"))
    A.add("dash", MP.gauges(240, 100))
    A.add("hinge", MP.plate(" ", 80, 30, "#1a1a1a", "#2a2a2a"))
    A.add("hingeline", MP.outline(330, 160, 3, handle=(0.9, 0.5)))
    R, HW = 0.285, 0.6
    WB = 1.75
    XO = 0.635
    for i, y0 in enumerate((WB / 2, -WB / 2)):
        fwd = 1 if y0 > 0 else -1
        # drum: rolled steel shell with chamfered edges, dark end plates
        p.r = 0.0
        prof = [(0.0, -HW + 0.02), (R - 0.03, -HW + 0.02), (R - 0.01, -HW), (R, -HW + 0.012),
                (R, HW - 0.012), (R - 0.01, HW), (R - 0.03, HW - 0.02), (0.0, HW - 0.02)]
        xlathe(p, (0, y0, R), prof, X["drum"], 44, mats=lambda j: X["dark"] if j in (0, 6) else X["drum"])
        # frame side plates round the drum ends, a top plate and end crossmember
        p.r = 0.012
        ye, yi = y0 + fwd * 0.39, y0 - fwd * 0.45
        lo_, hi_ = min(ye, yi), max(ye, yi)
        for sx in (-1, 1):
            # a deep top rail over the drum and a bracket down to the drum bearing: most of the
            # drum end stays in view
            poly = [(lo_, 0.6), (y0 - 0.17, 0.6), (y0 - 0.13, 0.42)] + \
                [(y0 + 0.13 * math.cos(math.radians(a)), R + 0.13 * math.sin(math.radians(a))) for a in range(180, 361, 30)] + \
                [(y0 + 0.13, 0.42), (y0 + 0.17, 0.6), (hi_, 0.6), (hi_, 0.84), (lo_, 0.84)]
            p.extrude_yz(poly, sx * XO - 0.025 if sx > 0 else -XO, XO if sx > 0 else -XO + 0.025, Y)
            # yoke plate on the drum end, rubber buffer blocks between it and the frame bracket
            xcyl(p, y0, R, sx * (HW + 0.002), sx * (HW + 0.012), 0.24, Y, 28)
            for k in range(4):
                a = math.radians(45 + 90 * k)
                q = np.array([y0 + 0.19 * math.cos(a), R + 0.19 * math.sin(a)])
                p.box((min(sx * 0.61, sx * 0.665), q[0] - 0.03, q[1] - 0.03), (max(sx * 0.61, sx * 0.665), q[0] + 0.03, q[1] + 0.03), X["rubberblk"])
            # hub / drive motor outside the plate, its bolt ring and hoses
            xo = sx * XO
            if sx < 0:
                xcyl(p, y0, R, xo, xo - 0.05, 0.16, D, 24)
                xcyl(p, y0, R, xo - 0.05, xo - 0.1, 0.11, X["dark"], 20)
                for k in range(8):
                    a = 2 * math.pi * k / 8
                    xcyl(p, y0 + 0.13 * math.cos(a), R + 0.13 * math.sin(a), xo - 0.05, xo - 0.065, 0.012, X["nut"], 6)
                p.r = 0.0
                for dz in (0.04, -0.04):
                    p.tube([(xo - 0.08, y0 - fwd * 0.05, R + 0.08 + dz), (xo - 0.1, y0 - fwd * 0.2, R + 0.25 + dz), (xo - 0.04, y0 - fwd * 0.3, 0.7 + dz)],
                           0.012, X["rubberblk"], 8, bend=0.08)
                p.r = 0.012
            else:
                xcyl(p, y0, R, xo, xo + 0.03, 0.12, D, 20)
                xcyl(p, y0, R, xo + 0.03, xo + 0.05, 0.06, X["nut"], 12)
        p.box((-XO, lo_, 0.78), (XO, hi_, 0.84), D)
        p.box((-XO, ye - 0.04 if fwd > 0 else ye, 0.6), (XO, ye if fwd > 0 else ye + 0.04, 0.84), Y)
        p.box((-XO + 0.02, ye - 0.02 if fwd > 0 else ye - 0.0, 0.56), (XO - 0.02, ye + 0.03 if fwd > 0 else ye + 0.05, 0.62), D)
        # scrapers (steel bar and blade against the drum, both sides) and the water spray bar
        for s in (-1, 1):
            a = math.radians(-12)
            yy = y0 + s * (R + 0.035) * math.cos(a)
            zz = R + (R + 0.035) * math.sin(a)
            p.obox((0, yy, zz), (2 * HW - 0.04, 0.025, 0.11), X["steel"], pitch=s * 0.35)
            p.obox((0, y0 + s * (R + 0.008) * math.cos(a), R + (R + 0.008) * math.sin(a) - 0.02), (2 * HW - 0.06, 0.012, 0.05), X["rubberblk"], pitch=s * 0.35)
            for sx in (-1, 1):
                p.box((sx * (HW - 0.02) - 0.015, min(yy, y0 + s * (R + 0.11)), zz - 0.02), (sx * (HW - 0.02) + 0.015, max(yy, y0 + s * (R + 0.11)), zz + 0.02), X["steel"])
        p.r = 0.0
        ys = y0 + fwd * 0.3
        p.cyl((-HW, ys, 0.64), (HW, ys, 0.64), 0.02, X["galv"], 10)
        for k in range(9):
            xx = -HW + 0.08 + k * (2 * HW - 0.16) / 8
            p.cyl((xx, ys, 0.64), (xx, ys - fwd * 0.03, 0.6), 0.009, X["black"], 6)
        p.r = 0.02
    # front hood (water tank) and the engine hood behind, lofted rounded sections
    p.r = 0.03
    fr = [(0.40, 1.13), (0.95, 1.13), (1.12, 1.09), (1.2, 1.03), (1.255, 0.95)]
    p.loft([xsec(1.24 - (0.04 if y > 1.15 else 0.0) - (0.04 if y > 1.22 else 0.0), 0.80, z1, 0.17, 0.03, y, n=6) for y, z1 in fr], Y)
    rr = [(-1.255, 1.06), (-1.22, 1.15), (-1.12, 1.21), (-0.6, 1.24), (-0.32, 1.24)]
    p.loft([xsec(1.12 - (0.04 if y < -1.2 else 0.0) - (0.03 if y < -1.24 else 0.0), 0.80, z1, 0.16, 0.03, y, n=6) for y, z1 in rr], Y)
    # hinge lines, latch straps, filler caps
    for sx in (-1, 1):
        for yy in (0.48, 1.08):
            A.put(p, "hinge", (sx * 0.621, yy, 0.86), (sx, 0, 0), (0, 0, 1), 0.07, 0.03)
        p.r = 0.004
        p.box((sx * 0.62 - 0.006, 1.1, 0.95), (sx * 0.62 + 0.006, 1.15, 1.03), X["rubberblk"])
        p.box((sx * 0.562 - 0.006, -0.72, 1.0), (sx * 0.562 + 0.006, -0.66, 1.1), X["rubberblk"])
        A.put(p, "hingeline", (sx * 0.563, -0.88, 0.94), (sx, 0, 0), (0, 0, 1), 0.66, 0.32)
    p.r = 0.03
    # dash console on the front hood's back face, steering column and wheel
    p.r = 0.02
    p.loft([xsec(0.5, 0.86, 1.2, 0.05, 0.01, y) for y in (0.33, 0.405)], X["dark"])
    A.put(p, "dash", (0, 0.329, 1.12), (0, -1, 0), (0, 0, 1), 0.2, 0.07)
    p.r = 0.004
    wc = np.array([0.0, -0.1, 1.52])
    p.r = 0.02
    p.loft([xsec(0.22, 0.86, 1.25, 0.04, 0.01, y) for y in (0.06, 0.3)], X["dark"])
    p.r = 0.004
    p.tube([(0, 0.18, 1.2), tuple(wc + (0, 0.05, -0.06))], 0.03, X["black"], 10)
    tilt = math.radians(35)
    rim_ = [tuple(wc + 0.17 * np.array([math.cos(t), math.sin(t) * math.sin(tilt), -math.sin(t) * math.cos(tilt) * 0 + math.sin(t) * math.cos(tilt) * 0.0]) +
                  np.array([0, 0, 0.17 * math.sin(t) * math.cos(tilt)]) * 0) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    rim_ = [tuple(wc + np.array([0.17 * math.cos(t), -0.17 * math.sin(t) * math.sin(tilt), 0.17 * math.sin(t) * math.cos(tilt)])) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    p.tube(rim_, 0.016, X["black"], 8, bend=0.0, closed=True)
    for t in (math.pi / 2, math.pi * 7 / 6, math.pi * 11 / 6):
        q = wc + np.array([0.17 * math.cos(t), -0.17 * math.sin(t) * math.sin(tilt), 0.17 * math.sin(t) * math.cos(tilt)])
        p.tube([tuple(wc), tuple(q)], 0.012, X["black"], 6)
    p.cyl(wc - (0, 0.02, 0.0), wc + (0, 0.03, 0.03), 0.04, X["black"], 12)
    # forward / reverse lever at the right of the seat
    p.cyl((0.32, -0.4, 1.24), (0.32, -0.4, 1.3), 0.05, X["black"], 12)
    p.tube([(0.32, -0.4, 1.28), (0.32, -0.3, 1.48)], 0.01, X["chrome"], 8)
    p.sphere((0.32, -0.3, 1.5), 0.025, X["black"], 10, 6)
    # operator's deck over the articulation joint
    p.r = 0.01
    p.box((-0.6, -0.32, 0.80), (0.6, 0.40, 0.86), X["checker"])
    for sx in (-1, 1):
        A.put(p, "haz", (sx * 0.6, 0.04, 0.83), (sx, 0, 0), (0, 0, 1), 0.7, 0.055)
        # steps and grab handles
        p.box((sx * 0.43, -0.12, 0.40), (sx * 0.6, 0.18, 0.44), X["checker"])
        for yy in (-0.1, 0.16):
            p.box((sx * 0.52 - 0.012, yy - 0.012, 0.44), (sx * 0.52 + 0.012, yy + 0.012, 0.80), D)
        p.r = 0.0
        p.tube([(sx * 0.6, 0.37, 0.95), (sx * 0.66, 0.37, 0.98), (sx * 0.66, 0.37, 1.32), (sx * 0.6, 0.37, 1.35)], 0.016, X["black"], 8, bend=0.05)
        p.r = 0.01
    # articulation: frame tongues, the vertical pivot pin, the steering ram
    p.box((-0.2, 0.05, 0.40), (0.2, 0.45, 0.62), D)
    p.box((-0.28, -0.36, 0.38), (0.28, -0.02, 0.64), D)
    p.box((-0.24, -0.06, 0.60), (0.24, 0.12, 0.66), D)
    p.box((-0.24, -0.06, 0.36), (0.24, 0.12, 0.42), D)
    p.cyl((0, 0.03, 0.26), (0, 0.03, 0.76), 0.045, X["steel"], 16)
    for z0_, z1_ in ((0.3, 0.4), (0.62, 0.72)):
        p.cyl((0, 0.03, z0_), (0, 0.03, z1_), 0.09, D, 18)
    # the operator's deck belongs to the rear frame: closed skirts under it back to the engine hood
    for sx in (-1, 1):
        p.box((sx * 0.6 - 0.02 if sx > 0 else -0.6, -0.32, 0.46), (0.6 if sx > 0 else -0.6 + 0.02, -0.04, 0.8), Y)
    p.r = 0.006
    for sx in (-1, 1):
        ram2(p, (sx * 0.34, -0.32, 0.5), (sx * 0.3, 0.38, 0.5), 0.05, X["dark"], X["chrome"])
        p.r = 0.0
        p.tube([(sx * 0.34, -0.3, 0.56), (sx * 0.38, -0.1, 0.7), (sx * 0.2, 0.05, 0.74), (sx * 0.1, 0.2, 0.7)], 0.012, X["rubberblk"], 8, bend=0.08)
        p.r = 0.006
    for sx in (-1, 1):
        A.put(p, "artic", (sx * 0.611 + sx * 0.026, -0.62 + 0.0, 0.66), (sx, 0, 0), (0, 0, 1), 0.36, 0.036)
    # suspension seat on the engine hood
    seat(p, (0, -0.62, 1.38), X, w=0.45, back=0.42, arm=True)
    p.r = 0.02
    p.box((-0.2, -0.82, 1.24), (0.2, -0.42, 1.31), D)
    # ROPS: two box-section posts on the frame plates, fold hinge, top bar, lamps, beacon
    yr = -0.98
    for sx in (-1, 1):
        xp = sx * 0.6
        p.r = 0.008
        p.box((xp - 0.07, yr - 0.1, 0.84), (xp + 0.07, yr + 0.1, 0.87), D)
        for bx, by in ((-0.045, -0.07), (0.045, -0.07), (-0.045, 0.07), (0.045, 0.07)):
            p.cyl((xp + bx, yr + by, 0.87), (xp + bx, yr + by, 0.89), 0.012, X["nut"], 6)
        p.r = 0.006
        p.box((xp - 0.06, yr - 0.07, 1.48), (xp + 0.06, yr + 0.07, 1.6), D)
        p.cyl((xp - 0.07, yr, 1.54), (xp + 0.07, yr, 1.54), 0.018, X["steel"], 10)
    p.r = 0.0
    sbar(p, [(-0.6, yr, 0.87), (-0.6, yr, 2.4), (0.6, yr, 2.4), (0.6, yr, 0.87)], 0.1, 0.08, D, bend=0.12, up=(0, 1, 0))
    p.r = 0.006
    A.put(p, "rops", (0.6, yr, 1.75), (0, 1, 0), (0, 0, 1), 0.1, 0.025, lift=0.042)
    beacon(p, (0.0, yr, 2.44), M, X)
    for s in (-1, 1):
        for sx in (-1, 1):
            lamp(p, (sx * 0.38, yr + s * 0.09, 2.35), (0, s, 0), 0.12, 0.09, 0.08, X["black"], X["lens"])
    # exhaust, filler caps, lamps, lifting eyes on the hoods
    p.r = 0.004
    p.cyl((0.38, -1.05, 1.22), (0.38, -1.05, 1.52), 0.035, X["steel"], 12)
    p.cyl((0.38, -1.05, 1.52), (0.38, -1.07, 1.56), 0.04, X["black"], 12)
    p.cyl((-0.35, -0.95, 1.2), (-0.35, -0.95, 1.27), 0.05, X["black"], 14)
    p.cyl((0.3, 0.75, 1.12), (0.3, 0.75, 1.18), 0.07, X["black"], 16)
    for sx in (-1, 1):
        lamp(p, (sx * 0.48, 1.18, 1.09), (0, 1, 0), 0.13, 0.08, 0.07, X["black"], X["lens"])
        lamp(p, (sx * 0.45, -1.26, 1.0), (0, -1, 0), 0.1, 0.06, 0.04, X["black"], X["tail"])
        for yy in (0.6, -1.0):
            p.tube([(sx * 0.42, yy - 0.05, 1.12 if yy > 0 else 1.2), (sx * 0.42, yy - 0.05, 1.18 if yy > 0 else 1.26),
                    (sx * 0.42, yy + 0.05, 1.18 if yy > 0 else 1.26), (sx * 0.42, yy + 0.05, 1.12 if yy > 0 else 1.2)], 0.012, D, 8, bend=0.03)
    # decals: model and maker on the hoods, louvres, rear marking, warning labels
    for sx in (-1, 1):
        A.put(p, "model", (sx * 0.62, 0.78, 0.99), (sx, 0, 0), (0, 0, 1), 0.46, 0.1)
        A.put(p, "louvre", (sx * 0.56, -0.9, 0.98), (sx, 0, 0), (0, 0, 1), 0.48, 0.25)
        A.put(p, "brand", (sx * 0.56, -0.5, 1.14), (sx, 0, 0), (0, 0, 1), 0.26, 0.05)
        A.put(p, "warn", (sx * 0.62, 0.48, 0.98), (sx, 0, 0), (0, 0, 1), 0.08, 0.08)
    A.put(p, "louvre", (0, -1.256, 1.0), (0, -1, 0), (0, 0, 1), 0.5, 0.14)
    A.put(p, "rear", (0, -1.256, 0.86), (0, -1, 0), (0, 0, 1), 0.5, 0.1)
    A.put(p, "brand", (0, 1.256, 0.88), (0, 1, 0), (0, 0, 1), 0.4, 0.08)
    return MP.finish(p, "roller", col, 30, sharp=_sharp(X), mud=0.6, mud_h=0.45, seed=4, tint=(0.5, 0.48, 0.46))



def steering_wheel(p, wc, r, tilt, mat, toward=-1):
    """A three-spoke wheel centred at wc, its rim plane tilted `tilt` from horizontal, facing a
    driver on the -y side (toward=-1) or +y side."""
    wc = np.asarray(wc, float)
    pts = [tuple(wc + np.array([r * math.cos(t), toward * r * math.sin(t) * math.cos(tilt), -toward * r * math.sin(t) * math.sin(tilt) * -1]))
           for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    pts = [tuple(wc + np.array([r * math.cos(t), r * math.sin(t) * math.cos(tilt), r * math.sin(t) * math.sin(tilt) * -toward])) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    p.tube(pts, r * 0.09, mat, 8, bend=0.0, closed=True)
    for t in (math.pi / 2 * -toward, math.pi * 7 / 6, math.pi * 11 / 6):
        q = wc + np.array([r * math.cos(t), r * math.sin(t) * math.cos(tilt), r * math.sin(t) * math.sin(tilt) * -toward])
        p.tube([tuple(wc), tuple(q)], r * 0.06, mat, 6)
    p.cyl(tuple(wc - np.array([0, 0, 0.02])), tuple(wc + np.array([0, 0, 0.03])), r * 0.22, mat, 12)


def tractor(col, M):
    """A 120 hp four-wheel-drive utility tractor (John Deere 6120M class: L 4.36, WB 2.40, rear
    600/65R38 1.75 m, front 540/65R24 1.31 m, cab roof 2.82 m): a long bonnet tapering to a narrow
    nose with a mesh grille, headlamps and a weight frame of suitcase weights; a four-post cab with
    a curved one-piece screen, doors bulged above the mudguards, a moulded roof with work lamps, a
    beacon and mirrors on long arms; steps and a handrail, fuel tank, battery box; lugged R-1W
    tyres on yellow rims under mudguards with lamps; a three-point linkage (lower links, top link,
    lift arms and drop rods), PTO with its shield and a pick-up hitch."""
    p = PM(0.02)
    X = machine_mats(M)
    G, D, YR = X["green"], X["frame"], X["jdyellow"]
    A = MP.Atlas("Mach_TractorDecals", 1024)
    A.add("model", MP.plate("6120", 360, 120, "#f0c21a", None, bold=0.03))
    A.add("brand", MP.plate("AGRIMAX", 480, 110, "#f0c21a", None, bold=0.03, italic=0.15))
    A.add("stripe", MP.plate(" ", 512, 16, "#f0c21a", "#f0c21a"))
    A.add("grille", MP.mesh_grille(320, 360, 18))
    A.add("vent", MP.louvre(300, 160, 6, frame="#1a1a1a"))
    A.add("plate", MP.plate("AX19 FRM", 420, 100, "#111111", "#f4c400", border="#222222"))
    A.add("warn", MP.warning(128))
    A.add("pto", MP.plate("540 / 1000", 300, 70, "#111111", "#f2c200"))
    A.add("split", MP._rgba(256, 16, "#163212"))
    Rr, wr, rr_ = 0.873, 0.6, 0.4826
    Rf, wf, rf_ = 0.655, 0.54, 0.305
    yr, yf = -1.18, 1.22
    xr, xf = 0.95, 0.9
    for sx in (-1, 1):
        MP.wheel(p, (sx * xr, yr, Rr), Rr, wr, rr_, X["tyre"], YR, X["nut"], out=sx, lugs=23, lug_h=0.045, lug_w=0.075,
                 kind="bar", sides=32, dish=0.55, nuts=10, hub_mat=YR)
        MP.wheel(p, (sx * xf, yf, Rf), Rf, wf, rf_, X["tyre"], YR, X["nut"], out=sx, lugs=20, lug_h=0.036, lug_w=0.06,
                 kind="bar", sides=30, dish=0.45, nuts=8, hub_mat=D)
    # -- driveline: rear axle housing and trumpets, transmission, engine sump, front axle
    p.r = 0.03
    p.loft([xsec(0.86, 0.48, 1.18, 0.12, 0.08, y) for y in (-1.62, -0.55)], G)
    for sx in (-1, 1):
        xcyl(p, yr, Rr, sx * 0.4, sx * (xr - 0.2), 0.13, G, 18, r2=0.1)
    # engine and frame under the bonnet: block with sump, head, manifold, turbo, filters, hoses
    E = X["engine"]
    p.r = 0.02
    p.loft([xsec(0.5, 0.42, 0.62, 0.05, 0.05, y) for y in (0.1, 1.9)], D)
    p.loft([xsec(0.66, 0.6, 0.98, 0.06, 0.04, y) for y in (-0.55, 1.15)], E)
    p.loft([xsec(0.56, 0.98, 1.05, 0.03, 0.01, y) for y in (-0.5, 1.1)], E)
    p.r = 0.004
    for k in range(6):
        yy = -0.4 + k * 0.28
        for sx in (-1, 1):
            p.box((sx * 0.33 - 0.015, yy - 0.04, 0.72), (sx * 0.33 + 0.015, yy + 0.04, 0.95), E)
    p.r = 0.006
    p.cyl((0.36, -0.2, 0.9), (0.36, 0.9, 0.9), 0.05, X["steel"], 12)
    p.sphere((0.42, 0.95, 0.88), 0.11, X["steel"], 14, 8)
    p.cyl((-0.36, 0.3, 0.7), (-0.46, 0.3, 0.7), 0.06, X["nut"], 12)
    p.cyl((-0.36, 0.6, 0.66), (-0.36, 0.6, 0.86), 0.05, X["galv"], 12)
    xcyl(p, 1.25, 0.85, -0.3, -0.42, 0.07, X["nut"], 14)
    p.box((-0.28, 1.4, 0.55), (0.28, 1.9, 1.0), K := X["black"])
    p.r = 0.0
    for dz, sx in ((0.0, 1), (0.05, -1), (0.1, 1)):
        p.tube([(sx * 0.3, -0.5, 0.7 + dz), (sx * 0.38, 0.0, 0.66 + dz), (sx * 0.36, 0.8, 0.72 + dz), (sx * 0.2, 1.42, 0.85 + dz)], 0.016, X["rubberblk"], 8, bend=0.2)
    p.r = 0.02
    p.box((-0.52, yf - 0.1, Rf - 0.1), (0.52, yf + 0.1, Rf + 0.1), D)
    p.sphere((0, yf, Rf), 0.2, D, 14, 8, scale=(1.1, 1.0, 1.0))
    for sx in (-1, 1):
        xcyl(p, yf, Rf, sx * 0.5, sx * (xf - 0.17), 0.09, D, 14)
        p.cyl((sx * (xf - 0.19), yf, Rf - 0.12), (sx * (xf - 0.19), yf, Rf + 0.14), 0.07, D, 12)
    # -- bonnet: long, tapering to a narrow nose, side vents, grille, headlamps
    p.r = 0.03
    rings = []
    for t in np.linspace(0.0, 1.0, 7):
        y = -0.25 + 2.2 * t
        w = 0.86 - 0.24 * t ** 1.5
        z1 = 1.86 - 0.24 * t ** 1.8
        z0 = 1.0 + 0.04 * t
        rings.append(xsec(w, z0, z1, 0.2 - 0.06 * t, 0.02, y, n=5))
    yn = -0.25 + 2.2
    rings.append(xsec(0.6, 1.07, 1.58, 0.13, 0.02, yn + 0.06, n=5))
    p.loft(rings, G)
    # grille set 30 mm into a raised surround, a split line down the bonnet top
    A.put(p, "grille", (0, yn + 0.061, 1.32), (0, 1, 0), (0, 0, 1), 0.48, 0.46, lift=0.002)
    p.r = 0.006
    for x0_, x1_, z0_, z1_ in ((-0.27, -0.24, 1.08, 1.56), (0.24, 0.27, 1.08, 1.56), (-0.27, 0.27, 1.06, 1.09), (-0.27, 0.27, 1.55, 1.58)):
        p.box((x0_, yn + 0.04, z0_), (x1_, yn + 0.095, z1_), K0 := X["black"])
    for t in (0.12, 0.42, 0.72):
        y_ = -0.25 + 2.2 * t
        z_ = 1.86 - 0.24 * t ** 1.8
        A.put(p, "split", (0, y_, z_ + 0.001), (0, 0, 1), (0, 1, 0), 0.006, 0.66, lift=0.001)
    p.r = 0.03
    for sx in (-1, 1):
        lamp(p, (sx * 0.22, yn + 0.07, 1.5), (0, 1, 0), 0.12, 0.07, 0.04, X["black"], X["lens"])
        # vents and the stripe, model and maker on each side, laid on the tapering flank
        def side(t, z, key, w, h):
            y = -0.25 + 2.2 * t
            f = (0.86 - 0.24 * t ** 1.5) / 2
            df = -0.24 * 1.5 * math.sqrt(max(t, 1e-6)) / 2 / 2.2
            n = np.array([sx, -df, 0.0])
            A.put(p, key, (sx * (f + 0.003), y, z), n / np.linalg.norm(n), (0, 0, 1), w, h)
        side(0.72, 1.25, "vent", 0.42, 0.22)
        for t in (0.17, 0.45, 0.73):
            side(t, 1.56, "stripe", 0.62, 0.025)
        side(0.25, 1.38, "brand", 0.48, 0.11)
        side(0.6, 1.42, "model", 0.3, 0.1)
    # front weight frame and suitcase weights
    p.r = 0.015
    p.box((-0.42, 1.95, 0.55), (0.42, 2.05, 0.95), D)
    p.r = 0.01
    for k in range(8):
        xx = -0.35 + k * 0.1
        p.r = 0.006
        p.box((xx - 0.045, 2.04, 0.58), (xx + 0.045, 2.16, 0.9), D)
        p.r = 0.0
        p.box((xx - 0.012, 2.13, 0.9), (xx + 0.012, 2.17, 0.98), D)
    # -- cab
    zf_, zr = 1.22, 2.7
    cy0, cy1 = -1.75, -0.28
    K = X["black"]
    p.r = 0.03
    p.box((-0.68, cy0, 1.1), (0.68, cy1, zf_), K)
    # posts
    p.r = 0.0
    for sx in (-1, 1):
        p.tube([(sx * 0.68, cy1, zf_), (sx * 0.68, cy1, 1.8), (sx * 0.76, cy1 - 0.03, 2.2), (sx * 0.72, cy1 - 0.05, zr)], 0.04, K, 6, bend=0.2)
        p.tube([(sx * 0.68, cy0 + 0.03, zf_), (sx * 0.68, cy0 + 0.03, 1.8), (sx * 0.76, cy0 + 0.03, 2.2), (sx * 0.72, cy0 + 0.03, zr)], 0.045, K, 6, bend=0.2)
        p.tube([(sx * 0.68, cy0 + 0.6, 1.8), (sx * 0.76, cy0 + 0.6, 2.2), (sx * 0.72, cy0 + 0.6, zr)], 0.025, K, 6, bend=0.2)
        p.tube([(sx * 0.68, cy0 + 0.05, 1.8), (sx * 0.68, cy1 - 0.02, 1.8)], 0.022, K, 6)
    GL = X["glass"]

    def side_x(z):
        if z <= 1.8:
            return 0.68
        if z <= 2.2:
            return 0.68 + 0.08 * (z - 1.8) / 0.4
        return 0.76 - 0.04 * (z - 2.2) / 0.5
    zs = [1.25, 1.8, 2.0, 2.2, 2.45, 2.68]
    for sx in (-1, 1):
        for za, zb in zip(zs[:-1], zs[1:]):
            q = [(sx * side_x(za), cy0 + 0.06, za), (sx * side_x(za), cy1 - 0.04, za), (sx * side_x(zb), cy1 - 0.04, zb), (sx * side_x(zb), cy0 + 0.06, zb)]
            p.face(q if sx > 0 else q[::-1], GL)
    # curved one-piece screen
    n = 8
    for k in range(n):
        xa, xb_ = -0.68 + 1.36 * k / n, -0.68 + 1.36 * (k + 1) / n
        ya = cy1 + 0.07 * (1 - (xa / 0.68) ** 2)
        yb = cy1 + 0.07 * (1 - (xb_ / 0.68) ** 2)
        p.quad((xa, ya, 1.28), (xb_, yb, 1.28), (xb_, yb - 0.03, 2.66), (xa, ya - 0.03, 2.66), GL)
    p.quad((0.66, cy0, 1.3), (-0.66, cy0, 1.3), (-0.66, cy0, 2.66), (0.66, cy0, 2.66), GL)
    # wiper, door handle, handrail, roof
    p.tube([(0.1, cy1 + 0.08, 1.35), (-0.25, cy1 + 0.07, 2.1)], 0.008, K, 5)
    p.tube([(-0.7, cy1 - 0.12, 1.4), (-0.74, cy1 - 0.12, 1.4), (-0.74, cy1 - 0.12, 1.95), (-0.7, cy1 - 0.12, 1.95)], 0.014, K, 6, bend=0.03)
    p.tube([(-0.78, cy1 - 0.45, 1.6), (-0.8, cy1 - 0.47, 1.6), (-0.8, cy1 - 0.57, 1.6), (-0.78, cy1 - 0.59, 1.6)], 0.01, K, 6, bend=0.02)
    p.r = 0.04
    rc = (cy0 + cy1) / 2 + 0.03
    p.loft([rrect(1.62, 1.62, 0.18, 2.66, cy=rc), rrect(1.78, 1.82, 0.22, 2.72, cy=rc), rrect(1.76, 1.8, 0.22, 2.78, cy=rc),
            rrect(1.5, 1.5, 0.3, 2.83, cy=rc)], G)
    p.r = 0.01
    for sx in (-0.55, -0.2, 0.2, 0.55):
        lamp(p, (sx, rc + 0.905, 2.72), (0, 1, 0), 0.13, 0.05, 0.04, K, X["lens"])
    for sx in (-0.5, 0.5):
        lamp(p, (sx, rc - 0.905, 2.72), (0, -1, 0), 0.13, 0.05, 0.04, K, X["lens"])
    beacon(p, (-0.55, rc - 0.6, 2.82), M, X)
    # mirrors on long arms
    p.r = 0.0
    for sx in (-1, 1):
        p.tube([(sx * 0.74, cy1 - 0.05, 2.55), (sx * 1.0, cy1 + 0.0, 2.6), (sx * 1.22, cy1 + 0.05, 2.45)], 0.014, K, 6, bend=0.08)
        p.r = 0.01
        p.obox((sx * 1.25, cy1 + 0.06, 2.3), (0.2, 0.04, 0.32), K, yaw=sx * 0.2)
        p.r = 0.0
    # exhaust stack at the right A-pillar
    p.r = 0.004
    ex = (0.5, cy1 + 0.13)
    p.cyl((*ex, 1.6), (*ex, 3.02), 0.06, X["steel"], 16)
    p.cyl((*ex, 2.0), (*ex, 2.75), 0.078, X["perf"], 18)
    for z in (2.0, 2.75):
        p.cyl((*ex, z - 0.01), (*ex, z + 0.01), 0.082, K, 18)
    p.obox((ex[0], ex[1] + 0.02, 3.04), (0.13, 0.13, 0.008), K, pitch=0.35)
    # inside: seat with armrest console, wheel on its column, dash
    seat(p, (0, -1.25, 1.48), X, w=0.5, back=0.5)
    p.r = 0.02
    p.box((0.3, -1.45, 1.4), (0.42, -0.95, 1.6), D)
    p.cyl((0.36, -1.0, 1.6), (0.36, -1.0, 1.72), 0.012, K, 6)
    p.sphere((0.36, -1.0, 1.74), 0.03, K, 10, 6)
    p.loft([xsec(0.5, zf_, 1.45, 0.06, 0.02, y) for y in (cy1 + 0.0, cy1 - 0.25)], D)
    p.r = 0.004
    p.tube([(0, cy1 - 0.15, 1.45), (0, -0.7, 1.82)], 0.03, K, 10)
    steering_wheel(p, (0, -0.72, 1.86), 0.2, math.radians(35), K)
    # steps and handrail (left, ahead of the rear tyre), fuel tank, battery box
    p.r = 0.01
    for k, z in enumerate((0.48, 0.78, 1.06)):
        p.box((-0.82, -0.25, z), (-0.56, 0.0, z + 0.03), X["checker"])
    p.box((-0.82, -0.25, 0.46), (-0.8, 0.0, 1.1), K)
    p.box((-0.58, -0.25, 0.46), (-0.56, 0.0, 1.1), K)
    p.r = 0.04
    p.box((-0.55, -0.58, 0.55), (-0.31, -0.02, 1.05), K)
    p.cyl((-0.45, -0.1, 1.05), (-0.45, -0.1, 1.1), 0.06, K, 14)
    p.box((0.31, -0.55, 0.6), (0.55, -0.05, 0.95), K)
    # rear mudguards with lamps, front mudguards on their brackets
    for sx in (-1, 1):
        g = []
        for k in range(13):
            a = math.radians(18 + 146 * k / 12)
            g.append((sx * (xr + 0.0), yr + (Rr + 0.09) * math.cos(a), Rr + (Rr + 0.09) * math.sin(a)))
        fender(p, yr, Rr, sx * xr, Rr + 0.09, 18, 164, 0.7, G, lip=0.02)
        p.r = 0.0
        p.tube([(sx * (xr - 0.36), yr - 0.2, Rr + 0.3), (sx * (xr - 0.36), yr + 0.1, Rr + 0.6)], 0.03, D, 8)
        p.r = 0.01
        q = np.array(g[2]) + np.array([sx * 0.18, 0, 0.0])
        lamp(p, (sx * (xr + 0.22), yr - 0.75, Rr + 0.52), (0, -1, 0), 0.1, 0.07, 0.05, K, X["tail"])
        lamp(p, (sx * (xr + 0.22), yr - 0.73, Rr + 0.6), (0, -1, 0), 0.1, 0.05, 0.05, K, X["amber"])
        lamp(p, (sx * (xr + 0.36), yr + 0.3, Rr + 0.85), (sx, 0, 0), 0.08, 0.05, 0.03, K, X["amber"])
        fg = [(sx * xf, yf + (Rf + 0.08) * math.cos(math.radians(a)), Rf + (Rf + 0.08) * math.sin(math.radians(a))) for a in range(40, 141, 10)]
        fender(p, yf, Rf, sx * xf, Rf + 0.09, 30, 150, 0.4, G, lip=0.015)
        p.r = 0.0
        p.tube([(sx * (xf - 0.2), yf, Rf + 0.08), (sx * (xf - 0.3), yf - 0.04, Rf + 0.45), (sx * (xf - 0.24), yf - 0.06, Rf + 0.62),
                (sx * (xf - 0.12), yf - 0.06, Rf + 0.73)], 0.028, K, 8, bend=0.1)
        p.box((sx * (xf - 0.16), yf - 0.12, Rf + 0.72), (sx * (xf - 0.06), yf, Rf + 0.75), K)
    # -- three-point linkage, PTO, hitch, number plate
    p.r = 0.008
    for sx in (-1, 1):
        a, b = np.array([sx * 0.3, -1.55, 0.55]), np.array([sx * 0.46, -2.15, 0.5])
        beam(p, a, b, 0.05, 0.11, D)
        p.sphere(b + np.array([0, -0.03, 0]), 0.055, X["nut"], 12, 8)
        la, lb = np.array([sx * 0.28, -1.5, 1.2]), np.array([sx * 0.33, -2.0, 1.1])
        beam(p, la, lb, 0.05, 0.09, D)
        p.cyl(lb, a + (b - a) * 0.55, 0.025, D, 10)
        p.cyl(a + (b - a) * 0.55 + np.array([0, 0, 0.2]), a + (b - a) * 0.55 + np.array([0, 0, 0.35]), 0.035, X["nut"], 10)
    p.cyl((0, -1.55, 1.12), (0, -1.85, 1.08), 0.04, D, 12)
    p.cyl((0, -1.85, 1.08), (0, -2.12, 1.04), 0.028, X["chrome"], 12)
    p.box((-0.06, -2.17, 0.98), (0.06, -2.1, 1.1), D)
    p.cyl((0, -1.62, 0.78), (0, -1.75, 0.78), 0.04, X["steel"], 14)
    sh = [(0.15 * math.cos(math.radians(a)), -1.66, 0.78 + 0.15 * math.sin(math.radians(a))) for a in range(0, 181, 20)]
    p.r = 0.004
    p.sweep(sh, [(-0.004, -0.12), (0.004, -0.12), (0.004, 0.12), (-0.004, 0.12)], X["yellow"], twist_ref=(0, 1, 0))
    A.put(p, "pto", (0, -1.58, 0.98), (0, -1, 0), (0, 0, 1), 0.2, 0.05, lift=0.03)
    p.r = 0.01
    p.box((-0.12, -1.85, 0.42), (0.12, -1.5, 0.5), D)
    p.cyl((0, -1.78, 0.38), (0, -1.78, 0.56), 0.025, X["chrome"], 10)
    p.box((-0.2, -1.63, 1.22), (0.2, -1.61, 1.34), K)
    A.put(p, "plate", (0, -1.635, 1.28), (0, -1, 0), (0, 0, 1), 0.38, 0.09)
    # hydraulic couplers with coloured dust caps, their hoses up the rear housing
    p.r = 0.0
    for k, cm in enumerate(("red", "red", "blue", "blue")):
        xx = -0.27 + k * 0.18 if k < 2 else 0.09 + (k - 2) * 0.18
        q = np.array([xx, -1.64, 1.05])
        p.cyl(q, q + np.array([0, -0.06, 0.02]), 0.022, X["chrome"], 10)
        p.cyl(q + np.array([0, -0.06, 0.02]), q + np.array([0, -0.085, 0.028]), 0.026, X[cm], 10)
        p.tube([(xx, -1.58, 1.04), (xx * 0.8, -1.5, 0.98), (xx * 0.5, -1.4, 1.1)], 0.012, X["rubberblk"], 8, bend=0.06)
    p.box((-0.32, -1.62, 1.0), (0.32, -1.58, 1.12), X["frame"])
    p.r = 0.03
    return MP.finish(p, "tractor", col, 30, sharp=_sharp(X), mud=0.8, mud_h=0.6, seed=1)


def ride_mower(col, M):
    """A garden lawn tractor (John Deere X350 class: WB 1.25, W 0.96, 20 in rear and 15 in front
    turf tyres, 42 in deck): a sloping hood with a mesh grille, headlamps and side vents, a dash
    tower with key, throttle and fuel cap, a raked steering column and wheel; ribbed footrests;
    a moulded fender pan wrapping the rear wheels, a mid-back seat on its bracket, the lift lever;
    a twin-blade deck slung between the axles with anti-scalp wheels and a side discharge chute;
    a rear hitch plate."""
    p = PM(0.02)
    X = machine_mats(M)
    G, K, D = X["mgreen"], X["black"], X["frame"]
    A = MP.Atlas("Mach_MowerDecals", 1024)
    A.add("grille", MP.mesh_grille(300, 180, 12))
    A.add("vent", MP.louvre(260, 120, 5, frame="#141414"))
    A.add("vent_f", MP.louvre(300, 150, 7, frame="#181818", lip="#4a4a48"))
    A.add("shut", MP._rgba(16, 256, "#14240f"))
    A.add("quad", MP.louvre(80, 140, 6, frame="#1a1a1a", lip="#555555"))
    A.add("brand", MP.plate("AGRIMAX", 420, 100, "#f0c21a", None, bold=0.03, italic=0.15))
    A.add("model", MP.plate("X42", 220, 100, "#f0c21a", None, bold=0.03))
    A.add("stripe", MP.plate(" ", 512, 16, "#f0c21a", "#f0c21a"))
    A.add("deck", MP.plate("42 IN", 200, 80, "#e8e8e8", None, bold=0.02))
    A.add("warn", MP.warning(128))
    A.add("blade", MP.plate("DANGER  ROTATING BLADES", 520, 64, "#111111", "#f2c200", border="#111111"))
    A.add("key", MP.plate("0  I  II", 200, 60, "#e8e8e8", "#141414"))
    Rr, Rf = 0.254, 0.19
    yr, yf = -0.5, 0.75
    for sx in (-1, 1):
        # turf-saver tyres: 20x10-8 rear, 15x6-6 front, shallow chevrons on a smooth carcass
        MP.wheel(p, (sx * 0.353, yr, Rr), Rr, 0.254, 0.1016, X["tyre"], X["jdyellow"], X["nut"], out=sx, lugs=36, lug_h=0.007,
                 lug_w=0.016, kind="turf", sides=36, dish=0.25, nuts=5, hub_mat=K, rim_detail=False)
        MP.wheel(p, (sx * 0.36, yf, Rf), Rf, 0.152, 0.076, X["tyre"], X["jdyellow"], X["nut"], out=sx, lugs=30, lug_h=0.0035,
                 lug_w=0.013, kind="turf", sides=30, dish=0.25, nuts=4, hub_mat=K, rim_detail=False)
    # frame, axles
    p.r = 0.01
    for sx in (-1, 1):
        p.box((sx * 0.2 - 0.03, -0.85, 0.14), (sx * 0.2 + 0.03, 0.95, 0.26), D)
    p.box((-0.3, yf - 0.04, Rf - 0.05), (0.3, yf + 0.04, Rf + 0.04), D)
    xcyl(p, yr, Rr, -0.3, 0.3, 0.05, D, 12)
    p.box((-0.18, yr - 0.15, Rr - 0.08), (0.18, yr + 0.12, Rr + 0.1), D)
    # hood: a moulded tilting shroud, rounded and tapering to a nose about 60 % of its rear width;
    # a recessed grille insert with slats, headlamps above it, a shut line where it tilts
    p.r = 0.01
    hood = [(0.235, 0.54, 0.76, 0.12), (0.5, 0.53, 0.75, 0.13), (0.72, 0.49, 0.71, 0.15), (0.88, 0.43, 0.65, 0.16),
            (0.97, 0.37, 0.58, 0.15), (1.02, 0.33, 0.51, 0.13)]
    p.loft([xsec(w, 0.3, z1, rt, 0.02, y, n=6) for y, w, z1, rt in hood], G)
    p.box((-0.24, 0.222, 0.32), (0.24, 0.236, 0.74), K)
    A.put(p, "shut", (0, 0.3, 0.7564), (0, 0, 1), (0, 1, 0), 0.32, 0.008, lift=0.0015)
    p.r = 0.004
    p.box((-0.135, 1.0, 0.31), (0.135, 1.03, 0.43), K)
    A.put(p, "vent_f", (0, 1.031, 0.37), (0, 1, 0), (0, 0, 1), 0.24, 0.1)
    for sx in (-1, 1):
        p.box((sx * 0.095 - 0.04, 0.99, 0.445), (sx * 0.095 + 0.04, 1.025, 0.485), K)
        r_, p.r = p.r, 0.0
        p.quad((sx * 0.095 - 0.032, 1.026, 0.45), (sx * 0.095 + 0.032, 1.026, 0.45), (sx * 0.095 + 0.032, 1.026, 0.48), (sx * 0.095 - 0.032, 1.026, 0.48), X["lens"])
        p.r = r_
        A.put(p, "vent", (sx * 0.252, 0.66, 0.5), (sx, -0.09, 0), (0, 0, 1), 0.2, 0.09)
        A.put(p, "brand", (sx * 0.271, 0.4, 0.56), (sx, -0.02, 0), (0, 0, 1), 0.22, 0.052)
        A.put(p, "model", (sx * 0.271, 0.4, 0.46), (sx, -0.02, 0), (0, 0, 1), 0.1, 0.045)
        A.put(p, "stripe", (sx * 0.268, 0.55, 0.34), (sx, -0.05, 0), (0, 0, 1), 0.55, 0.012)
    # front bumper
    p.r = 0.0
    p.tube([(-0.2, 1.0, 0.24), (-0.2, 1.08, 0.24), (0.2, 1.08, 0.24), (0.2, 1.0, 0.24)], 0.02, K, 8, bend=0.04)
    # dash tower, steering, key, throttle, fuel cap
    p.r = 0.02
    p.loft([xsec(0.44, 0.24, 0.62, 0.04, 0.02, y) for y in (0.08, 0.22)], K)
    p.loft([rrect(0.44, 0.14, 0.04, 0.6, cy=0.15), rrect(0.34, 0.12, 0.05, 0.8, cy=0.16)], K)
    A.put(p, "key", (0.12, 0.079, 0.74), (0, -1, 0), (0, 0, 1), 0.1, 0.03)
    p.r = 0.004
    p.cyl((0.12, 0.08, 0.72), (0.12, 0.05, 0.72), 0.012, X["chrome"], 8)
    p.tube([(-0.14, 0.12, 0.8), (-0.14, 0.06, 0.86)], 0.007, X["chrome"], 6)
    p.sphere((-0.14, 0.055, 0.865), 0.015, X["red"], 8, 5)
    p.cyl((-0.15, 0.17, 0.8), (-0.15, 0.17, 0.83), 0.035, K, 12)
    wc = np.array([0.0, -0.02, 0.98])
    p.cyl((0, 0.16, 0.78), tuple(wc + (0, 0.03, -0.04)), 0.04, K, 14, r2=0.026)
    steering_wheel(p, wc, 0.175, math.radians(32), K)
    # footrests: ribbed boards each side
    p.r = 0.01
    for sx in (-1, 1):
        p.box((sx * 0.18, -0.22, 0.22), (sx * 0.47, 0.22, 0.26), X["rubberblk"])
        # pedals: brake on the left, forward / reverse on the right
        p.obox((sx * 0.28, 0.17, 0.32), (0.07 if sx < 0 else 0.1, 0.02, 0.08), X["nut"], pitch=-0.6)
        p.box((sx * 0.28 - 0.01, 0.19, 0.24), (sx * 0.28 + 0.01, 0.22, 0.3), X["nut"])
        for k in range(9):
            yy = -0.19 + k * 0.045
            p.box((sx * 0.2, yy - 0.008, 0.26), (sx * 0.45, yy + 0.008, 0.268), X["rubberblk"])
    # rear body: a narrow seat tower between the wheels, fenders that wrap each tyre with an
    # inner skirt down to the tower, a small flat top for the seat
    p.r = 0.025
    p.loft([rrect(0.44, 0.62, 0.08, z, cy=-0.5) for z in (0.26, 0.6)], G)
    p.loft([rrect(0.44, 0.5, 0.1, z, cy=-0.5) for z in (0.6, 0.62)], G)
    for sx in (-1, 1):
        rad = Rr + 0.05
        fender(p, yr, Rr, sx * 0.353, rad, 0, 180, 0.3, G, lip=0.012, depth=0.008)
        poly = [(yr + rad * math.cos(math.radians(a)), Rr + rad * math.sin(math.radians(a))) for a in range(0, 181, 15)]
        xi = sx * (0.353 - 0.15)
        p.r = 0.004
        p.extrude_yz(poly, min(xi, xi + sx * 0.012), max(xi, xi + sx * 0.012), G)
    # seat on its bracket, lift lever, hitch
    p.r = 0.01
    p.box((-0.15, -0.62, 0.6), (0.15, -0.36, 0.63), D)
    pan_seat(p, (0, -0.48, 0.68), 0.44, X["seat"])
    p.r = 0.006
    p.box((-0.17, -0.64, 0.62), (0.17, -0.3, 0.645), D)
    for sx in (-1, 1):
        p.box((sx * 0.12 - 0.015, -0.66, 0.6), (sx * 0.12 + 0.015, -0.28, 0.625), X["nut"])
    xcyl(p, -0.29, 0.625, -0.15, 0.15, 0.015, X["nut"], 8)
    p.box((-0.06, -0.55, 0.6), (0.06, -0.45, 0.62), K)
    # deck-lift lever in its notched quadrant on the left fender
    p.r = 0.004
    p.box((-0.42, -0.52, 0.55), (-0.3, -0.36, 0.575), K)
    A.put(p, "quad", (-0.36, -0.44, 0.576), (0, 0, 1), (0, 1, 0), 0.08, 0.14)
    p.r = 0.0
    p.tube([(-0.36, -0.45, 0.57), (-0.37, -0.38, 0.78)], 0.01, X["chrome"], 6)
    p.sphere((-0.37, -0.38, 0.8), 0.025, K, 10, 6)
    p.cyl((0, -0.42, 0.5), (0, -0.42, 0.6), 0.035, X["rubberblk"], 10)
    p.r = 0.006
    p.box((-0.08, -0.9, 0.18), (0.08, -0.82, 0.24), D)
    # mower deck: shell, belt cover, chute, anti-scalp wheels
    p.r = 0.03
    p.loft([rrect(1.02, 0.62, 0.27, 0.07, cy=0.13), rrect(1.02, 0.62, 0.27, 0.15, cy=0.13), rrect(0.94, 0.56, 0.24, 0.2, cy=0.13)], X["dark"])
    p.loft([rrect(0.5, 0.18, 0.06, z, cy=0.38) for z in (0.2, 0.24)], X["dark"])
    A.put(p, "deck", (-0.3, 0.13, 0.201), (0, 0, 1), (0, 1, 0), 0.14, 0.055)
    for sx in (-1, 1):
        p.cyl((sx * 0.24, 0.12, 0.2), (sx * 0.24, 0.12, 0.235), 0.045, X["nut"], 14)
    A.put(p, "blade", (0.0, 0.442, 0.13), (0, 1, 0), (0, 0, 1), 0.36, 0.045)
    p.r = 0.015
    p.loft([[(0.5, 0.0, 0.06), (0.62, 0.02, 0.06), (0.62, 0.3, 0.06), (0.5, 0.32, 0.06)],
            [(0.5, 0.04, 0.17), (0.6, 0.06, 0.15), (0.6, 0.28, 0.15), (0.5, 0.3, 0.17)]], K)
    p.r = 0.004
    for sx in (-1, 1):
        xcyl(p, 0.44, 0.045, sx * 0.45, sx * 0.47, 0.04, K, 12)
        p.box((sx * 0.44 - 0.01, 0.38, 0.06), (sx * 0.44 + 0.01, 0.46, 0.1), D)
    return MP.finish(p, "ride_mower", col, 30, sharp=_sharp(X), mud=0.45, mud_h=0.3, seed=6, tint=(0.62, 0.66, 0.45))


def fbox(p, c, ex, et, en, size, mat):
    """A box centred on c with axes ex, et, en (unit vectors) and sizes along them."""
    c = np.asarray(c, float)
    V = []
    for k in (-1, 1):
        for j in (-1, 1):
            for i in (-1, 1):
                V.append(c + ex * i * size[0] / 2 + et * j * size[1] / 2 + en * k * size[2] / 2)
    F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    p.faces(V, F, mat)


def crawler(p, x, yi, ys, r, w, X, links=46):
    """A crawler track at lateral offset x: the shoe loop round a front idler (yi) and a rear
    sprocket (ys) of loop radius r, shoes with grousers outside, link rails inside, bottom and
    carrier rollers, the track frame, guards, idler and toothed sprocket."""
    zc = 0.045 + r
    L = yi - ys
    total = 2 * L + 2 * math.pi * r
    pitch = total / links
    ex = np.array([1.0, 0, 0])
    r0 = p.r
    p.r = 0.0

    def at(s):
        s %= total
        if s < L:                                   # bottom run, rear to front
            return np.array([ys + s, zc - r]), np.array([1.0, 0]), np.array([0, -1.0])
        s -= L
        if s < math.pi * r:                         # round the idler
            a = -math.pi / 2 + s / r
            n = np.array([math.cos(a), math.sin(a)])
            return np.array([yi, zc]) + n * r, np.array([-n[1], n[0]]), n
        s -= math.pi * r
        if s < L:                                   # top run, front to rear
            return np.array([yi - s, zc + r]), np.array([-1.0, 0]), np.array([0, 1.0])
        s -= L
        a = math.pi / 2 + s / r
        n = np.array([math.cos(a), math.sin(a)])
        return np.array([ys, zc]) + n * r, np.array([-n[1], n[0]]), n
    T = X["track"]
    for k in range(links):
        q, t, n = at(k * pitch + pitch / 2)
        c = np.array([x, q[0], q[1]])
        et = np.array([0, t[0], t[1]])
        en = np.array([0, n[0], n[1]])
        fbox(p, c + en * 0.01, ex, et, en, (w, pitch * 0.93, 0.02), T)
        for g_ in (-0.3, 0.02, 0.34):
            fbox(p, c + en * 0.03 + et * pitch * g_, ex, et, en, (w, 0.022, 0.022), T)
        for sx in (-1, 1):
            fbox(p, c - en * 0.035 + ex * sx * 0.1, ex, et, en, (0.045, pitch * 1.0, 0.07), T)
        fbox(p, c - en * 0.05 + et * pitch * 0.5, ex, et, en, (0.25, 0.05, 0.05), T)
    p.r = 0.015
    D = X["frame"]
    # track frame: box section between the tumblers, sloped at the ends
    p.loft([[(x - 0.17, yy, z0), (x + 0.17, yy, z0), (x + 0.17, yy, z1), (x - 0.17, yy, z1)] for yy, z0, z1 in
            ((ys + 0.25, 0.3, 0.52), (ys + 0.55, 0.22, 0.6), (yi - 0.55, 0.22, 0.6), (yi - 0.2, 0.3, 0.52))], D)
    # idler and sprocket
    p.r = 0.0
    for yy in (yi, ys):
        prof = [(0.0, -0.13), (0.24, -0.13), (0.27, -0.1), (0.27, -0.04), (0.24, -0.035), (0.24, 0.035), (0.27, 0.04),
                (0.27, 0.1), (0.24, 0.13), (0.0, 0.13)]
        if yy == yi:
            xlathe(p, (x, yy, zc), prof, D, 28)
        else:
            xlathe(p, (x, yy, zc), [(0.0, -0.12), (0.24, -0.12), (0.24, 0.12), (0.0, 0.12)], D, 24)
            for k in range(13):
                a = 2 * math.pi * k / 13
                tq = np.array([0, math.cos(a), math.sin(a)])
                tt = np.array([0, -math.sin(a), math.cos(a)])
                for sx in (-1, 1):
                    fbox(p, np.array([x + sx * 0.1, yy, zc]) + tq * 0.27, ex, tt, tq, (0.04, 0.07, 0.08), X["steel"])
            # final drive housing on the inner side
            xi = x - np.sign(x) * 0.12
            xcyl(p, yy, zc, xi, xi - np.sign(x) * 0.2, 0.28, D, 24)
            for k in range(10):
                a = 2 * math.pi * k / 10
                xcyl(p, yy + 0.22 * math.cos(a), zc + 0.22 * math.sin(a), xi - np.sign(x) * 0.2, xi - np.sign(x) * 0.215, 0.018, X["nut"], 6)
        xcyl(p, yy, zc, x + 0.13, x + 0.15, 0.09, X["nut"], 12)
        xcyl(p, yy, zc, x - 0.15, x - 0.13, 0.09, X["nut"], 12)
    # bottom rollers, carrier rollers, guards with ribs
    p.r = 0.0
    zb = 0.045 + 0.07 + 0.085
    for k in range(7):
        yy = ys + 0.45 + k * (L - 0.9) / 6
        xlathe(p, (x, yy, zb), [(0.0, -0.13), (0.085, -0.13), (0.085, -0.06), (0.07, -0.05), (0.07, 0.05), (0.085, 0.06), (0.085, 0.13), (0.0, 0.13)], X["steel"], 14)
    zt = zc + r - 0.07 - 0.065
    for f in (0.32, 0.68):
        yc = ys + (yi - ys) * f
        xlathe(p, (x, yc, zt), [(0.0, -0.1), (0.065, -0.1), (0.065, 0.1), (0.0, 0.1)], X["steel"], 14)
        p.box((x - 0.06, yc - 0.06, zt), (x + 0.06, yc + 0.06, 0.6), D)
    for sx in (-1, 1):
        p.box((x + sx * 0.17 - 0.01, ys + 0.35, 0.14), (x + sx * 0.17 + 0.01, yi - 0.35, 0.3), D)
        for k in range(7):
            yk = ys + 0.4 + k * (L - 0.8) / 6
            p.box((x + sx * 0.18 - 0.012, yk - 0.025, 0.15), (x + sx * 0.18 + 0.012, yk + 0.025, 0.3), D)
    p.r = r0
    return zc


def excavator(col, M):
    """A 13 t crawler excavator (CAT 313 class: W 2.49 on 500 mm shoes, cab top 2.83, tail swing
    2.15 m, boom 4.6 m, stick 2.6 m), parked with the boom up and the bucket curled on the ground
    in front. Track loops with grousers round a toothed sprocket and an idler, rollers and guards;
    a slewing house with a rounded counterweight, engine hood with louvres, a tank and tool deck
    with handrails, and a framed ROPS cab glazed all round (seat, consoles and levers inside); a
    banana boom and box-section stick; twin boom rams, stick and bucket rams pinned in lugs, the
    H-link and dog-bone at a bucket with teeth and side cutters; hoses, lamps, mirrors, decals."""
    p = PM(0.02)
    X = machine_mats(M)
    Y, D = X["yellow"], X["frame"]
    A = MP.Atlas("Mach_ExcavatorDecals", 1024)
    A.add("model", MP.plate("EX135", 560, 150, "#141414", None, bold=0.03))
    A.add("brand", MP.plate("TERRADIG", 560, 130, "#141414", None, bold=0.03))
    A.add("haz", MP.chevrons(420, 140, period=110))
    A.add("louvre", MP.louvre(320, 220, 10))
    A.add("louvre_v", MP.louvre(240, 320, 12, frame=None, lip="#b98a12"))
    A.add("door", MP.outline(300, 300, 4, handle=(0.85, 0.5)))
    A.add("hatch", MP.outline(260, 160, 4, handle=(0.5, 0.8)))
    A.add("warn", MP.warning(128))
    A.add("swing", MP.plate("KEEP CLEAR - SWING AREA", 640, 70, "#111111", "#f2c200", border="#111111"))
    A.add("lift", MP.plate("LIFT POINT", 220, 60, "#111111", "#f2c200"))
    A.add("panel", MP.plate("13.3 t", 200, 70, "#e8e8e8", "#1c1c1c"))
    A.add("soot", MP.soot(160))
    # -- undercarriage
    yi, ys, rr = 1.44, -1.44, 0.34
    for sx in (-1, 1):
        crawler(p, sx * 0.995, yi, ys, rr, 0.5, X)
    p.r = 0.02
    p.box((-0.82, -0.75, 0.42), (0.82, 0.75, 0.76), D)
    p.cyl((0, 0, 0.76), (0, 0, 0.94), 0.86, D, 40)
    # -- house
    tail = 2.15
    ay = -math.sqrt(tail ** 2 - 1.24 ** 2)

    def plan(z, inset=0.0, front=1.12):
        pts = [(1.24 - inset, front - inset), (-1.24 + inset, front - inset)]
        a0 = math.atan2(ay, -1.24)
        a1 = math.atan2(ay, 1.24)
        if a0 > 0:
            a0 -= 2 * math.pi
        for k in range(13):
            a = a0 + (a1 - a0) * k / 12
            rad = tail - inset
            pts.append((rad * math.cos(a) * (1.24 - inset) / 1.24, rad * math.sin(a), z))
        return [(q[0], q[1], z) for q in pts]
    p.r = 0.03
    p.loft([plan(0.95, 0.03), plan(1.2)], Y)
    # counterweight: rounded casting, top edge radiused
    def cw(z, inset):
        ring = []
        a0 = math.atan2(ay, -1.24)
        a1 = math.atan2(ay, 1.24)
        for k in range(17):
            a = a0 + (a1 - a0) * k / 16
            ring.append(((tail - inset) * math.cos(a) * (1.24 - inset) / 1.24, (tail - inset) * math.sin(a), z))
        ring += [(1.24 - inset, -1.38, z), (-1.24 + inset, -1.38, z)]
        return ring
    p.loft([cw(0.98, 0.04), cw(1.2, 0.0), cw(1.9, 0.0), cw(2.02, 0.05), cw(2.06, 0.14)], Y)
    # engine compartment: rounded hood
    p.r = 0.03
    p.loft([xsec(2.44, 1.2, z1, 0.18, 0.02, y) for y, z1 in ((-1.4, 2.2), (-0.5, 2.2))], Y)
    p.loft([xsec(2.3, 2.15, 2.24, 0.06, 0.01, y) for y in (-1.3, -0.65)], Y)
    # engine hood grille: a recessed dark frame with angled slats
    p.r = 0.004
    p.box((0.0, -1.24, 2.225), (0.7, -0.72, 2.245), X["black"])
    p.r = 0.0
    for k in range(9):
        yy = -1.2 + k * 0.055
        p.obox((0.35, yy, 2.245), (0.66, 0.045, 0.008), Y, pitch=0.6)
    for sx in (-1, 1):
        A.put(p, "door", (sx * 1.221, -0.95, 1.7), (sx, 0, 0), (0, 0, 1), 0.8, 0.8)
        p.r = 0.0
        for hz in (1.45, 1.95):
            p.cyl((sx * 1.228, -0.56, hz - 0.05), (sx * 1.228, -0.56, hz + 0.05), 0.014, X["nut"], 8)
            p.box((sx * 1.222 - 0.004, -0.6, hz - 0.03), (sx * 1.222 + 0.004, -0.56, hz + 0.03), X["nut"])
        A.put(p, "louvre_v", (sx * 1.222, -0.95, 1.72), (sx, 0, 0), (0, 0, 1), 0.35, 0.5)
    # right front: tank and tool deck, handrails, filler caps
    p.r = 0.03
    p.loft([xsec(0.8, 1.2, 1.88, 0.05, 0.01, y, cx=0.84) for y in (-0.5, 1.0)] + [xsec(0.8, 1.2, 1.7, 0.05, 0.01, 1.12, cx=0.84)], Y)
    A.put(p, "hatch", (1.241, 0.45, 1.5), (1, 0, 0), (0, 0, 1), 0.85, 0.5)
    A.put(p, "warn", (1.242, 0.95, 1.6), (1, 0, 0), (0, 0, 1), 0.12, 0.12)
    p.r = 0.004
    p.cyl((0.7, 0.1, 1.88), (0.7, 0.1, 1.95), 0.08, X["black"], 16)
    p.cyl((0.95, -0.25, 1.88), (0.95, -0.25, 1.94), 0.07, X["black"], 16)
    p.r = 0.0
    hr = [(1.16, -0.4, 1.88), (1.16, -0.4, 2.3), (1.16, 0.95, 2.3), (1.16, 0.95, 1.88)]
    p.tube(hr, 0.017, Y, 8, bend=0.08)
    p.tube([(1.16, 0.3, 1.88), (1.16, 0.3, 2.3)], 0.017, Y, 8)
    p.tube([(1.24, -0.3, 1.25), (1.32, -0.3, 1.3), (1.32, -0.3, 2.1), (1.24, -0.3, 2.15)], 0.015, Y, 8, bend=0.05)
    # mirror on the handrail, steps on the track frame and house side
    p.tube([(1.16, 0.95, 2.3), (1.25, 1.05, 2.55)], 0.012, X["black"], 6)
    p.r = 0.01
    p.obox((1.27, 1.07, 2.62), (0.2, 0.03, 0.26), X["black"], yaw=0.5)
    p.box((1.0, 0.55, 0.62), (1.2, 0.85, 0.66), X["checker"])
    p.box((-1.2, 0.55, 0.62), (-1.0, 0.85, 0.66), X["checker"])
    p.box((1.24, 0.2, 1.24), (1.32, 0.5, 1.27), X["checker"])
    # exhaust stack and air intake
    p.r = 0.004
    p.cyl((0.7, -1.0, 2.2), (0.7, -1.0, 2.62), 0.055, X["steel"], 14)
    A.put(p, "soot", (0.78, -1.0, 2.2405), (0, 0, 1), (0, 1, 0), 0.45, 0.55, lift=0.004)
    p.r = 0.0
    p.box((0.5, -0.35, 1.88), (1.18, 0.95, 1.886), X["checker"])
    p.r = 0.004
    p.cyl((0.7, -1.0, 2.62), (0.7, -1.05, 2.66), 0.06, X["black"], 14)
    p.cyl((-0.2, -1.1, 2.24), (-0.2, -1.1, 2.4), 0.09, X["black"], 16)
    p.cyl((-0.2, -1.1, 2.4), (-0.2, -1.1, 2.44), 0.11, X["black"], 16)
    # counterweight rear: lamps in recesses, chevrons, lifting eyes, swing warning
    p.r = 0.01
    for sx in (-1, 1):
        a = -math.pi / 2 + sx * 0.32
        cx_, cy_ = tail * math.cos(a), tail * math.sin(a)
        lamp(p, (cx_ * 0.985, cy_ * 0.985, 1.72), (math.cos(a), math.sin(a), 0), 0.16, 0.1, 0.06, X["black"], X["lens"])
        A.put(p, "haz", (cx_ * 1.0 + 0.0, cy_ * 1.0, 1.35), (math.cos(a), math.sin(a), 0), (0, 0, 1), 0.42, 0.14, lift=0.006)
        p.r = 0.0
        p.tube([(sx * 0.55, -1.95, 2.05), (sx * 0.55, -1.95, 2.13), (sx * 0.55, -1.8, 2.13), (sx * 0.55, -1.8, 2.05)], 0.018, X["dark"], 8, bend=0.04)
        p.r = 0.01
    A.put(p, "brand", (0, -tail - 0.002, 1.55), (0, -1, 0), (0, 0, 1), 0.85, 0.2)
    A.put(p, "swing", (-1.241, -1.1, 1.3), (-1, 0, 0), (0, 0, 1), 0.7, 0.077)
    # -- cab (left front): ROPS frame, glass all round, roof, wiper; seat and controls inside
    x0, x1, cy0, cy1, z0, z1 = -1.24, -0.3, -0.55, 1.08, 1.2, 2.83
    lean = 0.12
    K = X["black"]
    p.r = 0.02
    p.box((x0, cy0, z0), (x1, cy1, 1.34), Y)
    p.box((x0, cy0, 1.34), (x1, cy0 + 0.06, 2.72), Y)
    p.r = 0.012
    for xx in (x0 + 0.035, x1 - 0.035):
        p.tube([(xx, cy1 - 0.035, 1.34), (xx, cy1 - 0.035 - lean, 2.74)], 0.035, K, 4)
        p.tube([(xx, cy0 + 0.05, 1.34), (xx, cy0 + 0.05, 2.74)], 0.035, K, 4)
    p.tube([(x0 + 0.035, 0.22, 1.34), (x0 + 0.035, 0.22, 2.74)], 0.03, K, 4)
    p.tube([(x1 - 0.035, 0.1, 1.34), (x1 - 0.035, 0.1, 2.74)], 0.025, K, 4)
    p.tube([(x0 + 0.035, 0.24, 1.76), (x0 + 0.035, cy1 - 0.08, 1.76)], 0.02, K, 4)
    p.tube([(x0 + 0.04, cy1 - 0.035 - lean * 0.3, 1.76), (x1 - 0.04, cy1 - 0.035 - lean * 0.3, 1.76)], 0.022, K, 4)
    p.r = 0.05
    p.loft([rrect(x1 - x0 + 0.04, cy1 - cy0 - lean + 0.06, 0.08, z, cx=(x0 + x1) / 2, cy=(cy0 + cy1 - lean) / 2) for z in (2.72, 2.8)] +
           [rrect(x1 - x0 - 0.04, cy1 - cy0 - lean - 0.04, 0.06, 2.84, cx=(x0 + x1) / 2, cy=(cy0 + cy1 - lean) / 2)], Y)
    p.r = 0.0
    G = X["glass"]

    def fy(z):
        return cy1 - 0.03 - lean * (z - 1.34) / 1.4
    for za, zb in ((1.36, 1.74), (1.78, 2.71)):
        p.quad((x0 + 0.07, fy(za), za), (x1 - 0.07, fy(za), za), (x1 - 0.07, fy(zb), zb), (x0 + 0.07, fy(zb), zb), G)
    for xx, sgn in ((x0 + 0.005, -1), (x1 - 0.005, 1)):
        zlo = 1.36 if sgn < 0 else 1.5
        q = [(xx, cy0 + 0.09, zlo), (xx, fy(zlo) - 0.04, zlo), (xx, fy(2.71) - 0.04, 2.71), (xx, cy0 + 0.09, 2.71)]
        p.face(q if sgn > 0 else q[::-1], G)
    p.quad((x0 + 0.1, cy0 - 0.001, 2.05), (x1 - 0.1, cy0 - 0.001, 2.05), (x1 - 0.1, cy0 - 0.001, 2.6), (x0 + 0.1, cy0 - 0.001, 2.6), G)
    p.r = 0.02
    p.box((x1 - 0.02, cy0 + 0.08, 1.34), (x1 + 0.0, cy1 - 0.1, 1.5), Y)
    # door handle, wiper, roof lamp and beacon, left mirror
    p.r = 0.0
    p.tube([(x0 - 0.005, 0.3, 1.55), (x0 - 0.04, 0.32, 1.55), (x0 - 0.04, 0.42, 1.55), (x0 - 0.005, 0.44, 1.55)], 0.01, K, 6, bend=0.02)
    pv = np.array([(x0 + x1) / 2 + 0.25, fy(1.8) + 0.02, 1.8])
    p.tube([tuple(pv), tuple(pv + np.array([-0.35, -lean * 0.42, 0.62]))], 0.008, K, 5)
    p.tube([tuple(pv + np.array([-0.31, -lean * 0.38, 0.55])), tuple(pv + np.array([-0.39, -lean * 0.46, 0.69]))], 0.005, K, 4)
    p.cyl(tuple(pv), tuple(pv + np.array([0, 0.03, 0])), 0.025, K, 10)
    p.r = 0.01
    lamp(p, ((x0 + x1) / 2 + 0.25, cy1 - lean - 0.02, 2.88), (0, 1, 0), 0.16, 0.1, 0.09, K, X["lens"])
    beacon(p, (x0 + 0.2, cy0 + 0.2, 2.84), M, X)
    p.r = 0.0
    p.tube([(x0 + 0.03, cy1 - 0.1, 2.5), (x0 - 0.2, cy1 - 0.05, 2.55)], 0.012, K, 6)
    p.r = 0.01
    p.obox((x0 - 0.25, cy1 - 0.04, 2.55), (0.18, 0.03, 0.24), K, yaw=-0.3)
    # inside: seat, joystick consoles, travel levers, monitor
    seat(p, ((x0 + x1) / 2, 0.0, 1.62), X, w=0.5, back=0.55)
    for sx in (-1, 1):
        cxx = (x0 + x1) / 2 + sx * 0.36
        p.box((cxx - 0.08, -0.15, 1.34), (cxx + 0.08, 0.35, 1.68), D)
        p.cyl((cxx, 0.25, 1.68), (cxx, 0.25, 1.8), 0.012, K, 6)
        p.sphere((cxx, 0.25, 1.82), 0.03, K, 10, 6, scale=(1, 1, 1.4))
    for dx in (-0.1, 0.1):
        p.cyl(((x0 + x1) / 2 + dx, 0.75, 1.34), ((x0 + x1) / 2 + dx, 0.62, 1.85), 0.012, K, 6)
    p.obox((x1 - 0.14, cy1 - 0.25, 1.85), (0.2, 0.04, 0.15), K, yaw=-0.5, pitch=0.3)
    # -- boom, stick, bucket (in the yz plane at x = bx)
    bx = 0.1
    F = np.array([0.75, 1.55])
    B = np.array([4.05, 0.9])
    sd = np.array([math.sin(math.radians(25)), math.cos(math.radians(25))])      # up the stick, raked back 25 deg
    front = np.array([sd[1], -sd[0]])
    N = B + sd * 2.6
    Kn = np.array([2.75, 4.05])
    path = round_path([F, Kn, N], 0.35, n=8)
    seg = np.r_[0, np.cumsum([np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)])]
    tt = seg / seg[-1]
    hs = [0.42 + 0.24 * (t / 0.56) if t < 0.56 else 0.66 - 0.28 * (t - 0.56) / 0.44 for t in tt]
    p.r = 0.02
    vbeam(p, path, bx, 0.44, hs, Y, ch=0.04)

    def boom_at(t, off):
        """Point on the boom at fraction t, offset along its upward normal by off (+ = top)."""
        i = int(np.searchsorted(tt, t).clip(1, len(tt) - 1))
        f = (t - tt[i - 1]) / max(tt[i] - tt[i - 1], 1e-9)
        q = path[i - 1] + (path[i] - path[i - 1]) * f
        d = path[i] - path[i - 1]
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        return q + n * off
    # boom foot boss and the bracket on the house
    xcyl(p, F[0], F[1], bx - 0.3, bx + 0.3, 0.2, Y, 20)
    for sx in (-1, 1):
        lug(p, F[0], F[1], bx + sx * 0.3 - (0.04 if sx > 0 else 0.0), bx + sx * 0.3 + (0.0 if sx > 0 else 0.04), 0.24, (F[0] - 0.1, 1.15), Y)
    xcyl(p, F[0], F[1], bx - 0.36, bx + 0.36, 0.07, X["chrome"], 12)
    # stick: box section from its top lever (behind N) down to the bucket pin
    T = N + sd * 0.7 - front * 0.3
    p.r = 0.02
    vbeam(p, [T, N + np.array([0.04, 0.0]), N - sd * 1.3, B + sd * 0.12], bx, 0.34, [0.36, 0.52, 0.42, 0.3], Y, ch=0.035)
    for sx in (-1, 1):
        lug(p, N[0], N[1], bx + sx * 0.23 - 0.03, bx + sx * 0.23 + 0.03, 0.17, boom_at(0.94, 0.0), Y)
    xcyl(p, N[0], N[1], bx - 0.3, bx + 0.3, 0.06, X["chrome"], 12)
    xcyl(p, B[0], B[1], bx - 0.24, bx + 0.24, 0.12, Y, 16)
    # boom rams (pair, either side), stick ram on top, with lugs and pins
    p.r = 0.006
    base = np.array([1.12, 1.3])
    bu = boom_at(0.36, -hs[int(len(hs) * 0.36)] / 2 - 0.08)
    for sx in (-1, 1):
        xr = bx + sx * 0.33
        lug(p, base[0], base[1], xr - 0.07, xr - 0.04, 0.1, (base[0] + 0.05, 1.18), Y)
        lug(p, base[0], base[1], xr + 0.04, xr + 0.07, 0.1, (base[0] + 0.05, 1.18), Y)
        ram2(p, (xr, base[0], base[1]), (xr, bu[0], bu[1]), 0.085, Y, X["chrome"], frac=0.58)
        lug(p, bu[0], bu[1], bx + sx * 0.22 - 0.02, bx + sx * 0.22 + 0.02, 0.09, boom_at(0.36, 0.0), Y)
    xcyl(p, bu[0], bu[1], bx - 0.42, bx + 0.42, 0.04, X["chrome"], 10)
    # arm (stick) cylinder on top of the boom: from a tall bracket on the knee to the stick heel
    st0 = boom_at(0.56, max(hs) / 2 + 0.24)
    for sx in (-1, 1):
        lug(p, st0[0], st0[1], bx + sx * 0.11 - 0.025, bx + sx * 0.11 + 0.025, 0.1, boom_at(0.56, 0.0), Y)
        lug(p, T[0], T[1], bx + sx * 0.12 - 0.025, bx + sx * 0.12 + 0.025, 0.11, N + sd * 0.25, Y)
    ram2(p, (bx, st0[0], st0[1]), (bx, T[0], T[1]), 0.075, Y, X["chrome"], frac=0.58)
    xcyl(p, st0[0], st0[1], bx - 0.15, bx + 0.15, 0.035, X["chrome"], 10)
    xcyl(p, T[0], T[1], bx - 0.16, bx + 0.16, 0.035, X["chrome"], 10)
    # bucket (GP, 0.9 m wide), resting flat on its floor, teeth towards the machine: flat D-shaped
    # side plates, a wrap plate painted outside and bare steel inside, a straight cutting edge
    # with four teeth on adapters, side cutters, wear strips underneath, ears and pin bosses
    al = math.radians(11.3)
    rot = np.array([[math.cos(al), -math.sin(al)], [math.sin(al), math.cos(al)]])

    def bk(u, v):
        return B + rot @ np.array([u, v])
    prof = [(-0.12, -0.05), (0.40, -0.06), (0.47, -0.20), (0.48, -0.38), (0.43, -0.56), (0.32, -0.72), (0.16, -0.82),
            (-0.02, -0.86), (-0.62, -0.74)]
    sh = [bk(*q) for q in prof]
    cen2 = bk(0.05, -0.45)
    bw = 0.45
    p.r = 0.0
    for i in range(len(sh) - 1):
        (ya, za), (yb, zb) = sh[i], sh[i + 1]
        mat = X["steel"] if i == len(sh) - 2 else Y
        p.quad((bx - bw, ya, za), (bx + bw, ya, za), (bx + bw, yb, zb), (bx - bw, yb, zb), mat)
        ina = sh[i] + (cen2 - sh[i]) * 0.03
        inb = sh[i + 1] + (cen2 - sh[i + 1]) * 0.03
        p.quad((bx - bw + 0.02, ina[0], ina[1]), (bx - bw + 0.02, inb[0], inb[1]), (bx + bw - 0.02, inb[0], inb[1]), (bx + bw - 0.02, ina[0], ina[1]), X["steel"])
    p.r = 0.006
    poly = [tuple(q) for q in sh]
    p.extrude_yz(poly, bx - bw - 0.022, bx - bw, Y)
    p.extrude_yz(poly, bx + bw, bx + bw + 0.022, Y)
    E = bk(0.32, 0.04)
    for sx in (-1, 1):
        for q, rad in ((B, 0.12), (E, 0.09)):
            lug(p, q[0], q[1], bx + sx * 0.19 - 0.025, bx + sx * 0.19 + 0.025, rad, bk(0.14, -0.07), Y)
        xcyl(p, B[0], B[1], bx + sx * 0.165 - 0.0, bx + sx * 0.25, 0.075, X["steel"], 14)
    xcyl(p, E[0], E[1], bx - 0.24, bx + 0.24, 0.045, X["chrome"], 10)
    # cutting edge plate, teeth on adapters, side cutters, wear strips
    tip, heel = sh[-1], sh[-2]
    td = (tip - heel) / np.linalg.norm(tip - heel)
    tn = np.array([-td[1], td[0]])
    if (cen2 - tip) @ tn < 0:
        tn = -tn
    ex = np.array([1.0, 0, 0])
    et = np.array([0, td[0], td[1]])
    en = np.array([0, tn[0], tn[1]])
    fbox(p, np.array([bx, *(tip - td * 0.11)]) + en * 0.018, ex, et, en, (2 * bw + 0.04, 0.22, 0.04), X["steel"])
    for k in range(4):
        xx = bx - bw + 0.13 + k * (2 * bw - 0.26) / 3
        c0 = np.array([xx, *tip]) + en * 0.02
        fbox(p, c0 - et * 0.03, ex, et, en, (0.1, 0.16, 0.08), X["steel"])
        rings = []
        for f, sw_, sh_ in ((0.04, 1.0, 1.0), (0.14, 0.85, 0.8), (0.26, 0.55, 0.35), (0.3, 0.3, 0.15)):
            c = c0 + et * f
            rings.append([tuple(c + ex * a * 0.045 * sw_ + en * b * 0.038 * sh_) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        p.loft(rings, X["steel"])
    md = rot @ np.array([0.5, 0.69])
    md /= np.linalg.norm(md)
    q = tip + md * 0.24
    emd = np.array([0, *md])
    for sx in (-1, 1):
        fbox(p, np.array([bx + sx * (bw + 0.034), *q]), ex, emd, np.cross(ex, emd), (0.026, 0.42, 0.1), X["steel"])
    for xx in (-0.3, 0.0, 0.3):
        p.tube([(bx + xx, *(bk(u, v) + (bk(u, v) - cen2) / np.linalg.norm(bk(u, v) - cen2) * 0.012)) for u, v in prof[3:8]], 0.016, X["steel"], 6, bend=0.05)
    # H-link (pair of side plates) from the stick, twin dog-bones to the bucket, bucket ram
    L1 = B + sd * 0.42
    dLJ, dJE = 0.46, 0.42
    d = np.linalg.norm(E - L1)
    aa = (dLJ ** 2 - dJE ** 2 + d ** 2) / (2 * d)
    hh = math.sqrt(max(dLJ ** 2 - aa ** 2, 1e-6))
    mid = L1 + (E - L1) * aa / d
    perp = np.array([-(E - L1)[1], (E - L1)[0]]) / d
    J = mid + perp * hh if (mid + perp * hh)[0] > (mid - perp * hh)[0] else mid - perp * hh
    p.r = 0.006
    for sx in (-1, 1):
        beam(p, (bx + sx * 0.2, L1[0], L1[1]), (bx + sx * 0.2, J[0], J[1]), 0.035, 0.13, D, up=(0, 0, 1))
        beam(p, (bx + sx * 0.13, J[0], J[1]), (bx + sx * 0.13, E[0], E[1]), 0.04, 0.14, Y, up=(0, 1, 0))
        for q in (L1, J, E):
            xcyl(p, q[0], q[1], bx + sx * 0.1, bx + sx * 0.24, 0.07, D, 14)
    xcyl(p, L1[0], L1[1], bx - 0.27, bx + 0.27, 0.045, X["chrome"], 10)
    xcyl(p, J[0], J[1], bx - 0.27, bx + 0.27, 0.045, X["chrome"], 10)
    S0 = N + front * 0.26 - sd * 0.3
    for sx in (-1, 1):
        lug(p, S0[0], S0[1], bx + sx * 0.1 - 0.02, bx + sx * 0.1 + 0.02, 0.09, N + front * 0.1 - sd * 0.45, Y)
    ram2(p, (bx, S0[0], S0[1]), (bx, J[0], J[1]), 0.07, Y, X["chrome"], frac=0.58)
    # hoses along the boom and down the stick, work lamp on the boom
    p.r = 0.0
    for dx in (-0.08, 0.08):
        hp = [np.array([bx + dx, *boom_at(t, hs[min(int(len(hs) * t), len(hs) - 1)] / 2 + 0.04)]) for t in np.linspace(0.08, 0.92, 9)]
        hp = [np.array([bx + dx, 1.25, 1.75])] + hp + [np.array([bx + dx, *(N + sd * 0.1 + front * 0.32)]), np.array([bx + dx, *(S0 + front * 0.08 + sd * 0.08)])]
        p.tube([tuple(q) for q in hp], 0.022, K, 8, bend=0.12)
    p.r = 0.01
    wl = boom_at(0.28, 0.0)
    for sx in (-1, 1):
        lamp(p, (bx + sx * 0.27, wl[0], wl[1]), (0, 0.6, 0.8), 0.14, 0.1, 0.08, K, X["lens"])
        p.box((bx + sx * 0.225 - 0.0, wl[0] - 0.03, wl[1] - 0.06), (bx + sx * 0.25, wl[0] + 0.03, wl[1] - 0.02), K)
    # boom decals
    for sx in (-1, 1):
        q = boom_at(0.62, 0.0)
        d = boom_at(0.7, 0.0) - boom_at(0.54, 0.0)
        d /= np.linalg.norm(d)
        A.put(p, "model", (bx + sx * 0.222, q[0], q[1]), (sx, 0, 0), (0, -d[1] * sx * 0 + -d[1], d[0]) if False else (0, -d[1], d[0]), 0.9, 0.24)
        q2 = boom_at(0.2, 0.0)
        A.put(p, "warn", (bx + sx * 0.222, q2[0], q2[1]), (sx, 0, 0), (0, 0, 1), 0.14, 0.14)
    return MP.finish(p, "excavator", col, 30, sharp=_sharp(X), mud=0.8, mud_h=0.9, seed=2)


def _quad_offset(p, q, mat, inward, d):
    """A quad and a copy pushed `d` along `inward` (the inside skin of a plate) in another mat."""
    q = [np.asarray(v, float) for v in q]
    p.face(q, mat[0])
    p.face([v + inward * d for v in q[::-1]], mat[1])


def fender(p, y0, z0, x, rad, a0, a1, w, mat, lip=0.014, n=14, depth=0.006):
    """A mudguard arch round a wheel turning about x (centre y0, z0): a sheet `w` wide across x
    at radius `rad`, from angle a0 to a1 (degrees, 0 = front at hub height), with rolled edges."""
    arc = [(x, y0 + rad * math.cos(math.radians(a)), z0 + rad * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
    r, p.r = p.r, 0.006
    p.sweep(arc, [(-depth, -w / 2), (depth, -w / 2), (depth, w / 2), (-depth, w / 2)], mat, twist_ref=(1, 0, 0))
    p.r = 0.0
    if lip:
        for s in (-1, 1):
            p.tube([(q[0] + s * w / 2, q[1], q[2]) for q in arc], lip, mat, 8, bend=0.0)
    p.r = r
    return arc


def dumper(col, M):
    """A 6 t front-tip site dumper (Thwaites MACH2265 class: L 4.475, W 2.24, H 3.43 to the ROPS
    top, 500/60-22.5 tyres about 1.2 m across): a deep fabricated front chassis carrying the skip
    pivots and two tip rams, the articulation joint (oscillating ring, king pin, twin steering
    rams), a rear engine frame under a louvred hood; a 3.3 m3 skip wider than the track, 1 m deep,
    overhanging the front axle, with a rolled rim, ribs, a reinforced front lip and bare steel
    inside; arched mudguards following the rear tyres; the operator's seat, dash and wheel behind
    the skip; a fold-down twin-post ROPS with its hinge knuckles and locking pins, beacon and work
    lamps; lamps, grilles, chevrons, plate and load-rating decals; caked mud low down."""
    p = PM(0.02)
    X = machine_mats(M)
    Y, D = X["yellow"], X["frame"]
    A = MP.Atlas("Mach_DumperDecals", 1024)
    A.add("swl", MP.plate("SWL 6000 kg", 420, 100, "#111111", "#f2f2ee", border="#111111"))
    A.add("brand", MP.plate("TERRADIG", 560, 130, "#141414", None, bold=0.03))
    A.add("model", MP.plate("6T FRONT TIP", 560, 100, "#141414", None, bold=0.02))
    A.add("rear", MP.chevron_v(420, 140))
    A.add("haz", MP.chevrons(420, 90, period=80))
    A.add("louvre", MP.louvre(360, 200, 9))
    A.add("door", MP.outline(300, 200, 4, handle=(0.88, 0.5)))
    A.add("warn", MP.warning(128))
    A.add("belt", MP.plate("WEAR SEAT BELT", 420, 70, "#ffffff", "#1d5fb0"))
    A.add("dash", MP.gauges(240, 100))
    A.add("plate", MP.plate("DT21 KVM", 420, 100, "#111111", "#f4c400", border="#222222"))
    A.add("crush", MP.plate("DANGER - CRUSH ZONE", 520, 64, "#111111", "#f2c200", border="#111111"))
    A.add("artic", MP.chevrons(420, 50, period=46))
    R, TW = 0.6, 0.48
    xw = 0.8
    yf, yr = 0.9, -1.35
    for yy in (yf, yr):
        for sx in (-1, 1):
            MP.wheel(p, (sx * xw, yy, R), R, TW, 0.29, X["tyre"], Y, X["nut"], out=sx, lugs=19, lug_h=0.045, lug_w=0.1,
                     kind="bar", sides=38, dish=0.3, nuts=10, hub_mat=D)
    # -- front chassis: a deep fabricated box, the axle and diff, skip pivots, tip rams
    p.r = 0.03
    p.loft([xsec(1.0, 0.42, 0.98, 0.06, 0.04, y) for y in (-0.02, 1.6)] + [xsec(1.0, 0.62, 0.98, 0.06, 0.04, 1.9)], Y)
    p.r = 0.015
    p.box((-0.62, yf - 0.13, R - 0.12), (0.62, yf + 0.13, R + 0.12), D)
    p.sphere((0, yf, R - 0.05), 0.24, D, 16, 10, scale=(1.0, 0.8, 0.9))
    for sx in (-1, 1):
        xcyl(p, yf, R, sx * 0.6, sx * (xw - 0.16), 0.11, D, 16, r2=0.09)
        A.put(p, "crush", (sx * 0.502, 0.25, 0.86), (sx, 0, 0), (0, 0, 1), 0.5, 0.062)
    piv = np.array([1.8, 1.16])
    for sx in (-1, 1):
        lug(p, piv[0], piv[1], sx * 0.45 - 0.04, sx * 0.45 + 0.04, 0.11, (1.75, 0.95), D)
        lug(p, piv[0], piv[1], sx * 0.6 - 0.035, sx * 0.6 + 0.035, 0.11, (1.8, 1.3), Y)
    for sx in (-1, 1):
        xcyl(p, piv[0], piv[1], sx * 0.4, sx * 0.67, 0.045, X["nut"], 12)
        xcyl(p, piv[0], piv[1], sx * 0.67, sx * 0.69, 0.07, X["nut"], 12)
    p.r = 0.006
    for sx in (-1, 1):
        a, b = np.array([sx * 0.3, -0.12, 0.92]), np.array([sx * 0.3, 0.8, 1.25])
        lug(p, a[1], a[2], sx * 0.3 - 0.11, sx * 0.3 - 0.07, 0.08, (-0.12, 0.8), D)
        lug(p, a[1], a[2], sx * 0.3 + 0.07, sx * 0.3 + 0.11, 0.08, (-0.12, 0.8), D)
        ram2(p, a, b, 0.09, D, X["chrome"], frac=0.6)
    # -- skip
    yb0, yb1, yl, zf = 0.15, 1.55, 2.24, 1.24
    xb, xt = 0.92, 1.12
    zr_top, zl = 2.32, 1.95
    P = {
        "rb": [(-xb, yb0, zf), (xb, yb0, zf)],
        "rt": [(-xt, -0.08, zr_top), (xt, -0.08, zr_top)],
        "fb": [(-xb, yb1, zf), (xb, yb1, zf)],
        "fl": [(-xt + 0.02, yl, zl), (xt - 0.02, yl, zl)],
    }
    SK = X["skipsteel"]
    S = (Y, SK)
    cen = np.array([0.0, 1.0, 1.85])

    def skin(q):
        q = [np.asarray(v, float) for v in q]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        if (cen - sum(q) / len(q)) @ n < 0:
            n = -n
        _quad_offset(p, q, S, n, 0.012)
    p.r = 0.0
    skin([P["rb"][0], P["rb"][1], P["rt"][1], P["rt"][0]])
    skin([P["rb"][1], P["rb"][0], P["fb"][0], P["fb"][1]])
    skin([P["fb"][0], P["fl"][0], P["fl"][1], P["fb"][1]])
    for k in (0, 1):
        q = [np.asarray(P[n_][k], float) for n_ in ("rb", "fb", "fl", "rt")]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        if (cen - sum(q) / 4) @ n < 0:
            n = -n
        p.face(q, Y)
        p.face([v + n * 0.012 for v in q[::-1]], SK)
    rim = [P["rt"][0], P["fl"][0], P["fl"][1], P["rt"][1]]
    p.tube([tuple(v) for v in rim], 0.05, Y, 10, bend=0.12, closed=True)
    p.r = 0.008
    fl0, fl1 = np.asarray(P["fl"][0], float), np.asarray(P["fl"][1], float)
    ch = np.asarray(P["fb"][0], float)
    dch = (fl0 - ch) / np.linalg.norm(fl0 - ch)
    nch = np.cross(fl1 - fl0, ch - fl0)
    nch /= np.linalg.norm(nch)
    if nch[1] < 0:
        nch = -nch
    fbox(p, (fl0 + fl1) / 2 - dch * 0.08 + nch * 0.014, np.array([1.0, 0, 0]), dch, nch, (np.linalg.norm(fl1 - fl0) - 0.04, 0.14, 0.025), X["steel"])
    for k in (0, 1):
        sgn = 1 if k else -1
        rb, fb, fl, rt = (np.asarray(P[n_][k], float) for n_ in ("rb", "fb", "fl", "rt"))
        for f in (0.2, 0.45, 0.7):
            bot = rb + (fb - rb) * f
            topp = rt + (fl - rt) * f
            beam(p, tuple(bot + np.array([sgn * 0.04, 0, 0.02])), tuple(topp + np.array([sgn * 0.04, 0, -0.06])), 0.08, 0.07, Y, up=(1, 0, 0))
        beam(p, tuple(rb + np.array([sgn * 0.03, 0.0, -0.05])), tuple(fb + np.array([sgn * 0.03, 0.0, -0.05])), 0.08, 0.1, Y, up=(1, 0, 0))
    for f in (0.25, 0.5, 0.75):
        xx = -xb + 2 * xb * f
        beam(p, (xx, yb0 + 0.05, zf - 0.05), (xx, yb1, zf - 0.05), 0.1, 0.1, Y)
    xt_ = np.asarray(P["rt"][1], float)
    beam(p, (-xt + 0.1, -0.1, zr_top - 0.25), (xt - 0.1, -0.1, zr_top - 0.25), 0.08, 0.1, Y)
    for sx in (-1, 1):
        p.box((sx * 0.3 - 0.06, 0.72, zf - 0.12), (sx * 0.3 + 0.06, 0.9, zf), Y)
    for k, sx in ((0, -1), (1, 1)):
        rb, fb, fl, rt = (np.asarray(P[n_][k], float) for n_ in ("rb", "fb", "fl", "rt"))
        q = [rb, fb, fl, rt]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        if n[0] * sx < 0:
            n = -n
        c = rb + (fb - rb) * 0.33 + (rt - rb) * 0.45
        up = np.cross(n, (fb - rb) / np.linalg.norm(fb - rb)) * (1 if sx < 0 else -1)
        if up[2] < 0:
            up = -up
        A.put(p, "swl", tuple(c), tuple(n), tuple(up), 0.46, 0.11, lift=0.085)
        A.put(p, "warn", tuple(c + (fb - rb) * 0.3), tuple(n), tuple(up), 0.14, 0.14, lift=0.085)
    # -- articulation at mid-wheelbase: the front frame's upper and lower tongues, the rear frame's
    # yoke between them, two king-pin housings (top and bottom) on one vertical axis, an
    # oscillation ring, and a steering ram either side
    p.r = 0.012
    yj = -0.35
    p.cyl((0, -0.02, 0.7), (0, -0.1, 0.7), 0.34, Y, 28)
    for z0_, z1_ in ((0.46, 0.56), (0.84, 0.94)):
        p.box((-0.26, yj, z0_), (0.26, -0.08, z1_), Y)
    p.box((-0.24, -0.85, 0.6), (0.24, yj + 0.1, 0.8), D)
    for z0_, z1_ in ((0.4, 0.6), (0.8, 1.0)):
        p.cyl((0, yj, z0_), (0, yj, z1_), 0.1, D, 18)
    p.cyl((0, yj, 0.36), (0, yj, 1.04), 0.045, X["steel"], 12)
    p.r = 0.006
    for sx in (-1, 1):
        ram2(p, (sx * 0.42, -0.85, 0.7), (sx * 0.4, 0.12, 0.68), 0.06, D, X["chrome"])
        lug(p, 0.12, 0.68, sx * 0.4 - (0.09 if sx > 0 else -0.06), sx * 0.4 - (0.06 if sx > 0 else -0.09), 0.07, (0.12, 0.5), Y)
        A.put(p, "artic", (sx * 0.501, 0.55, 0.62), (sx, 0, 0), (0, 0, 1), 0.5, 0.06)
        # step plates ahead of the rear wheels, seat-belt straps
        p.r = 0.004
        p.box((sx * 0.5, -0.74, 0.5), (sx * 0.9, -0.56, 0.54), X["checker"])
        p.box((sx * 0.52, -0.73, 0.54), (sx * 0.54, -0.7, 0.98), D)
    # -- rear frame and engine hood, arched mudguards
    p.r = 0.04
    hood = [(-2.24, 0.5, 1.4), (-2.18, 0.45, 1.5), (-1.5, 0.45, 1.52), (-0.6, 0.45, 1.52)]
    p.loft([xsec(1.16 if y > -2.2 else 1.1, z0, z1, 0.15, 0.03, y) for y, z0, z1 in hood], Y)
    for sx in (-1, 1):
        fender(p, yr, R, sx * xw, R + 0.085, 0, 180, 0.54, Y, lip=0.016)
        p.r = 0.02
        p.box((sx * 0.57, -1.95, 1.3), (sx * (xw - 0.25), -0.75, 1.36), Y)
        A.put(p, "brand", (sx * 0.581, -1.72, 1.2), (sx, 0, 0), (0, 0, 1), 0.48, 0.11)
        A.put(p, "door", (sx * 0.581, -1.2, 0.95), (sx, 0, 0), (0, 0, 1), 0.75, 0.5)
        A.put(p, "louvre", (sx * 0.582, -1.2, 0.95), (sx, 0, 0), (0, 0, 1), 0.42, 0.24)
    # operator: footwell over the joint, seat, dash pod, wheel, tip and drive levers
    p.r = 0.02
    p.box((-0.6, -0.85, 0.98), (0.6, -0.22, 1.05), X["checker"])
    p.box((-0.6, -0.3, 1.05), (0.6, -0.2, 1.25), D)
    seat(p, (0, -1.05, 1.62), X, w=0.48, back=0.42)
    p.r = 0.0
    for sx in (-1, 1):
        p.tube([(sx * 0.26, -1.2, 1.64), (sx * 0.27, -0.98, 1.7), (sx * 0.1, -0.86, 1.72)], 0.008, X["black"], 4, bend=0.05)
    p.box((-0.04, -0.88, 1.7), (0.04, -0.85, 1.74), X["chrome"])
    p.r = 0.03
    p.loft([xsec(0.46, 1.05, 1.5, 0.08, 0.02, y) for y in (-0.36, -0.22)], D)
    A.put(p, "dash", (0, -0.365, 1.42), (0, -1, 0), (0, 0, 1), 0.2, 0.08)
    p.r = 0.004
    wc = np.array([0.0, -0.55, 1.74])
    p.tube([(0, -0.34, 1.48), tuple(wc + (0, 0.03, -0.03))], 0.03, X["black"], 10)
    steering_wheel(p, wc, 0.19, math.radians(38), X["black"])
    for dx, ln, mat in ((0.38, 0.3, X["red"]), (0.45, 0.25, X["black"])):
        p.tube([(dx, -0.85, 1.42), (dx, -0.78, 1.42 + ln)], 0.01, X["chrome"], 6)
        p.sphere((dx, -0.78, 1.44 + ln), 0.025, mat, 10, 6)
    A.put(p, "belt", (0, -0.199, 1.15), (0, 1, 0), (0, 0, 1), 0.3, 0.05)
    # ROPS: two 120 x 100 uprights on the rear frame, hinge knuckles with locking pins, top beam
    yro = -1.55
    for sx in (-1, 1):
        xp = sx * 0.64
        p.r = 0.008
        p.box((xp - 0.09, yro - 0.12, 1.32), (xp + 0.09, yro + 0.12, 1.38), D)
        for bx, by in ((-0.06, -0.08), (0.06, -0.08), (-0.06, 0.08), (0.06, 0.08)):
            p.cyl((xp + bx, yro + by, 1.38), (xp + bx, yro + by, 1.4), 0.014, X["nut"], 6)
        p.box((xp - 0.085, yro - 0.075, 2.02), (xp + 0.085, yro + 0.075, 2.22), D)
        p.cyl((xp - 0.1, yro + 0.09, 2.12), (xp + 0.1, yro + 0.09, 2.12), 0.04, D, 14)
        p.cyl((xp - 0.13, yro + 0.09, 2.12), (xp + 0.13, yro + 0.09, 2.12), 0.016, X["chrome"], 8)
        p.r = 0.0
        p.tube([(xp + sx * 0.13, yro + 0.09, 2.12), (xp + sx * 0.16, yro + 0.09, 2.06), (xp + sx * 0.16, yro + 0.0, 2.0)], 0.005, X["chrome"], 5, bend=0.02)
        p.r = 0.006
        p.box((xp - sx * 0.06 - 0.02, yro + 0.06, 1.62), (xp - sx * 0.06 + 0.02, yro + 0.1, 1.7), D)
    p.r = 0.0
    sbar(p, [(-0.64, yro, 1.38), (-0.64, yro, 3.3), (0.64, yro, 3.3), (0.64, yro, 1.38)], 0.12, 0.1, Y, bend=0.16, up=(0, 1, 0))
    p.r = 0.006
    beacon(p, (0.0, yro, 3.35), M, X)
    for s in (-1, 1):
        lamp(p, (0.38 * s, yro + 0.1, 3.24), (0, 1, 0), 0.14, 0.1, 0.08, X["black"], X["lens"])
        lamp(p, (0.38 * s, yro - 0.1, 3.24), (0, -1, 0), 0.14, 0.1, 0.08, X["black"], X["lens"])
    # rear: grille, lamps, chevrons, plate, exhaust, fuel and hydraulic caps, tow pin
    A.put(p, "louvre", (0, -2.241, 1.12), (0, -1, 0), (0, 0, 1), 0.75, 0.34)
    A.put(p, "rear", (0, -2.241, 0.78), (0, -1, 0), (0, 0, 1), 0.8, 0.18)
    p.r = 0.006
    p.box((-0.24, -2.3, 0.5), (0.24, -2.24, 0.64), D)
    A.put(p, "plate", (0, -2.301, 0.57), (0, -1, 0), (0, 0, 1), 0.42, 0.1)
    for sx in (-1, 1):
        lamp(p, (sx * 0.45, -2.245, 1.32), (0, -1, 0), 0.12, 0.08, 0.04, X["black"], X["tail"])
        lamp(p, (sx * 0.45, -2.245, 1.22), (0, -1, 0), 0.12, 0.06, 0.04, X["black"], X["amber"])
        lamp(p, (sx * 0.5, -0.2, 1.36), (0, 1, 0), 0.14, 0.1, 0.07, X["black"], X["lens"])
    p.r = 0.004
    p.cyl((-0.4, -1.95, 1.5), (-0.4, -1.95, 2.05), 0.045, X["steel"], 12)
    p.cyl((-0.4, -1.95, 2.05), (-0.4, -1.99, 2.1), 0.05, X["black"], 12)
    p.cyl((0.35, -1.85, 1.52), (0.35, -1.85, 1.58), 0.06, X["black"], 14)
    p.cyl((0.15, -2.05, 1.5), (0.15, -2.05, 1.55), 0.05, X["black"], 14)
    p.box((-0.12, -2.38, 0.42), (0.12, -2.24, 0.5), D)
    p.cyl((0, -2.32, 0.38), (0, -2.32, 0.56), 0.025, X["chrome"], 10)
    return MP.finish(p, "dumper", col, 30, sharp=_sharp(X), mud=0.85, mud_h=0.95, seed=3)


BUILDERS = {
    "prop.tractor": tractor, "prop.ride_mower": ride_mower, "prop.excavator": excavator,
    "prop.dumper": dumper, "prop.roller": roller,
}
