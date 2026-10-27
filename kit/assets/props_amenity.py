"""Site amenities and services for training grounds: the entrance sign, drinking fountain,
planter, traffic cone, pay machine, CCTV column, fire point, EV charger, gatehouse kiosk, parasol,
flagpole, satellite dish and fuel tank. Real sizes; origin on the ground at the footprint centre,
+Y the front.
"""
import math

import numpy as np
from mathutils import Matrix, Vector

from . import site_materials, stand
from .texpaint import Canvas, box_regions, box_uv, decal, lin, noise, project, weather, wrap_uv
from .texpaint import label as sign_label
from .props import PM, label, rrect, _paint, _plastic, clear_glass, tyre


# -- shared bits ----------------------------------------------------------------------------------

def _flip(p, i0, i1):
    for fi in range(i0, i1):
        p.F[fi] = tuple(p.F[fi][::-1])
        if p.uv[fi] is not None:
            p.uv[fi] = list(p.uv[fi])[::-1]


def _loft(p, rings, mat, cap0=True, cap1=True, mats=None):
    """PM.loft with its side faces wound outwards (PM.loft winds them inwards, which bakes the
    ambient occlusion black on lofted bodies and exports inverted normals)."""
    i0 = len(p.F)
    p.loft(rings, mat, cap0=cap0, cap1=cap1, mats=mats)
    _flip(p, i0, i0 + (len(rings) - 1) * len(rings[0]))


def _tube(p, pts, r, mat, sides=8, bend=None, cap=True, closed=False):
    """PM.tube with its sides wound outwards (PM.sweep winds them inwards)."""
    i0 = len(p.F)
    p.tube(pts, r, mat, sides, bend=bend, cap=cap, closed=closed)
    n_caps = 2 if (cap and not closed) else 0
    _flip(p, i0, len(p.F) - n_caps)


def _hexbolt(p, c, r, h, mat, axis=(0, 0, 1), washer=True):
    """A hex bolt head (with a washer) standing on a face at c along `axis`."""
    c, ax = np.asarray(c, float), np.asarray(axis, float)
    rr, p.r = p.r, 0.0015
    if washer:
        p.cyl(c, c + ax * 0.003, r * 1.45, mat, 12)
        c = c + ax * 0.003
    p.cyl(c, c + ax * h, r, mat, 6)
    p.r = rr


def _spline(pts, n=8):
    """A Catmull-Rom curve through pts, n samples per span (ends kept)."""
    P = [np.asarray(q, float) for q in pts]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2])
    return out


def _newell_(P):
    from .texpaint import _newell
    return _newell(P)


def _lathe_axis(p, prof, mat, origin, axis, sides=16):
    """A lathe of (r, t) about an arbitrary axis through origin (t along the axis)."""
    q = PM(0.0)
    q.lathe(prof, mat, sides=sides)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    rot = Vector(ax).to_track_quat("Z", "Y").to_matrix()
    o = np.asarray(origin, float)
    for f in q.F:
        p.face([o + np.array(rot @ Vector(q.V[i])) for i in f], mat)


def _tube_bezier(P0, P1, P2, P3, n=24):
    P0, P1, P2, P3 = (np.asarray(x, float) for x in (P0, P1, P2, P3))
    t = np.linspace(0, 1, n)[:, None]
    return list((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)


def _emit(name, color, strength):
    from .props import emissive
    return emissive(name, color, strength)


# -- traffic cone ---------------------------------------------------------------------------------

def _cone_atlas():
    cv = Canvas("Amen_Cone", 1024, 1024, col="#e65a1e", rough=0.5)
    body = cv.region(0, 0, 640, 800, 0.64, 0.8)              # one turn x height (z)
    bands = cv.region(650, 0, 370, 800, 0.37, 0.8)           # the sleeve, same height scale
    base = cv.region(0, 806, 210, 210, 0.42, 0.42)            # the base moulding, top view
    rng = np.random.default_rng(7)
    # PVC body: sun-faded towards the top, scuffed and tar-stained towards the bottom
    body.mottle(0.05, 8, 3, rough=0.06)
    body.field(lambda X, Y: np.clip((Y - 0.3) / 0.45, 0, 1) ** 1.5 * 0.35, col="#f0a070")
    body.field(lambda X, Y: np.clip(1 - (Y - 0.06) / 0.12, 0, 1) * 0.5, col="#8a7a68")
    body.grime(0.26, "#4a3a2c", 0.85, 11, y_ground=0.0)
    body.grime(0.12, "#1a1715", 0.85, 12, y_ground=0.0)
    body.field(lambda X, Y: np.clip(1 - Y / 0.3, 0, 1) * np.clip((noise(X.shape, (10, 30), 13, 3) - 0.55) * 4, 0, 1) * 0.7, col="#151312", rough=0.3)
    for k in range(36):
        x, y = rng.uniform(0, 0.64), rng.uniform(0.05, 0.5) ** 1.3
        ang = rng.normal(0, 0.35)
        L = rng.uniform(0.01, 0.06)
        dark = rng.random() < 0.6
        body.line([(x, y), (x + L * math.cos(ang), y + L * math.sin(ang))], rng.uniform(0.001, 0.003),
                  col="#2b2622" if dark else "#ee9a70", k=rng.uniform(0.12, 0.35), rough=0.7)
    # a mould seam up the side
    body.line([(0.32, 0.0), (0.32, 0.8)], 0.0015, dh=0.0004, mul=0.9)
    # retroreflective sleeves: silver-white microprisms in a diamond lattice, overlap seam, edge scuffs
    bands.fill(col="#c4c6c6", rough=0.3, metal=0.15)
    rs, cs, X, Y = bands.grid()
    q = 0.004
    u, v = (X + Y) / q, (X - Y) / q
    d = np.minimum(np.abs(u - np.round(u)), np.abs(v - np.round(v)))
    cv.ht[rs, cs] += 0.00025 * np.clip(d * 2, 0, 1)
    cv.col[rs, cs] *= (0.92 + 0.08 * np.clip(d * 2, 0, 1))[..., None]
    bands.field(lambda X, Y: np.clip(1 - np.abs(X - 0.1) / 0.0015, 0, 1), col="#a9aaa6", dh=0.0005)
    bands.field(lambda X, Y: (X > 0.1) * (X < 0.112) * 0.25, mul=0.85)
    bands.mottle(0.08, 6, 5, rough=0.05)
    for k in range(40):
        x, y = rng.uniform(0, 0.37), rng.choice([0.3, 0.4, 0.465, 0.6]) + rng.normal(0, 0.004)
        bands.line([(x, y), (x + rng.uniform(0.005, 0.03), y + rng.normal(0, 0.003))], 0.002, col="#6a6560", k=0.5)
    bands.field(lambda X, Y: np.clip(1 - (Y - 0.27) / 0.06, 0, 1) * 0.45, col="#57524a")
    bands.field(lambda X, Y: noise(X.shape, (16, 12), 14, 4) * 0.35, col="#8a8478")
    # base: recycled black rubber, embossed lettering round the collar, dust
    base.fill(col="#1b1b1c", rough=0.82)
    base.mottle(0.12, 10, 9, rough=0.05)
    for sx, sy, t in ((0.21 + 0.135, 0.21 - 0.162, "BS EN 13422"), (0.21 - 0.135, 0.21 + 0.156, "ROADLINE 750"),
                      (0.21 - 0.135, 0.21 - 0.162, "RECYCLED"), (0.21 + 0.135, 0.21 + 0.156, "CLASS R2")):
        base.text(t, sx, sy, 0.0075, weight="bold", align="c", dh=0.0006, mul=0.75)
    base.field(lambda X, Y: noise(X.shape, 12, 4, 4) * 0.45, col="#6d6457")
    rs, cs, X, Y = base.grid()
    sp = np.random.default_rng(31).random(X.shape)
    cv.col[rs, cs] += (np.clip((sp - 0.9) * 10, 0, 1) * 0.05)[..., None]
    base.text("CHAPTER 8", 0.21, 0.385, 0.012, weight="black", align="c", dh=0.0008, mul=0.7)
    # worn, lighter corners of the rim
    for cx_, cy_ in ((0.025, 0.025), (0.395, 0.025), (0.025, 0.395), (0.395, 0.395)):
        base.field(lambda X, Y, cx_=cx_, cy_=cy_: np.clip(1 - np.hypot(X - cx_, Y - cy_) / 0.05, 0, 1) * noise(X.shape, 20, 5, 3) * 0.8,
                   col="#4a4744", rough=0.95)
    # a peeled corner of the upper sleeve at its overlap seam
    bands.text("BS EN 13422  R2  CLASS 1", 0.2, 0.47, 0.006, weight="bold", align="c", col="#7a7c7e")
    bands.text("BS EN 13422  R2  CLASS 1", 0.2, 0.305, 0.006, weight="bold", align="c", col="#7a7c7e")
    # contact shadow on the ground
    sh = Canvas("Amen_ConeShadow", 128, 128, col="#0d0c0a", rough=0.9, alpha=0.0)
    g = sh.region(0, 0, 128, 128, 0.6, 0.6)
    g.field(lambda X, Y: np.clip(1 - np.maximum(np.abs(X - 0.3), np.abs(Y - 0.3)) / 0.3, 0, 1) ** 1.2 * 0.75, a=1.0)
    cv._shadow = (sh, g)
    return cv


def traffic_cone(col, M):
    """A 750 mm UK road cone (BS EN 13422): a recycled-rubber base moulding with a raised rim, a
    recessed deck and a collar; a 40-sided PVC cone with a rolled lip; two retroreflective
    sleeves standing proud with an overlap seam; scuffs, tar and sun fade."""
    cv = _cone_atlas()
    mat = cv.material("Amen_Cone", normal=1.0)
    body = cv.region(0, 0, 640, 800, 0.64, 0.8)
    bands = cv.region(650, 0, 370, 800, 0.37, 0.8)
    base = cv.region(0, 806, 210, 210, 0.42, 0.42)
    p = PM(0.004)

    def circ(r, z, m=40):
        return [(r * math.cos(math.pi * 2 * k / m + math.pi / 4), r * math.sin(math.pi * 2 * k / m + math.pi / 4), z) for k in range(m)]

    def sq(w, rad, z, m=40):
        # a rounded square with m points at the same angles as circ (so rings loft cleanly)
        out = []
        for k in range(m):
            a = math.pi * 2 * k / m + math.pi / 4
            d = np.array([math.cos(a), math.sin(a)])
            # march out to the rounded square's edge
            lo_, hi_ = 0.0, w
            for _ in range(30):
                t = (lo_ + hi_) / 2
                q = np.abs(d * t) - (w / 2 - rad)
                inside = np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1]), 0) - rad < 0
                lo_, hi_ = (t, hi_) if inside else (lo_, t)
            out.append((d[0] * lo_, d[1] * lo_, z))
        return out
    i0 = len(p.F)
    _loft(p, [sq(0.39, 0.05, 0.0), sq(0.39, 0.05, 0.044), sq(0.378, 0.046, 0.052), sq(0.358, 0.041, 0.05),
            sq(0.348, 0.04, 0.04), circ(0.146, 0.04), circ(0.141, 0.047), circ(0.137, 0.066)], mat, cap1=False)
    project(p, i0, base, (-0.21, -0.21, 0), (1, 0, 0), (0, 1, 0))
    p.r = 0.0
    i0 = len(p.F)
    prof = [(0.137, 0.066), (0.132, 0.078), (0.13, 0.09)]
    for k in range(1, 23):
        z = 0.09 + (0.725 - 0.09) * k / 22
        r_ = 0.13 + (0.0305 - 0.13) * (z - 0.09) / (0.725 - 0.09)
        if z > 0.62:
            r_ += 0.004 * (z - 0.62) / 0.105            # the taper eases below the tip
        prof.append((r_, z))
    prof += [(0.0365, 0.735), (0.037, 0.743), (0.033, 0.75), (0.026, 0.75), (0.022, 0.742), (0.021, 0.69)]
    p.lathe(prof, mat, sides=40)
    wrap_uv(p, i0, body, z0=0.0)

    def rad(z):
        return 0.13 + (0.0305 - 0.13) * (z - 0.09) / (0.725 - 0.09)
    for z0, z1 in ((0.3, 0.4), (0.465, 0.6)):
        i0 = len(p.F)
        zs = np.linspace(z0, z1, 6)
        pr = [(rad(z0) + 0.0003, z0 - 0.002), (rad(z0) + 0.0028, z0), (rad(z0) + 0.0018, z0 + 0.004)] + \
             [(rad(z) + 0.0016, z) for z in zs[1:-1]] + [(rad(z1) + 0.0018, z1 - 0.004), (rad(z1) + 0.0028, z1), (rad(z1) + 0.0003, z1 + 0.002)]
        p.lathe(pr, mat, sides=40)
        wrap_uv(p, i0, bands, z0=0.0, i1=len(p.F))
        # the sleeve uses the band region, scaled so one turn is the region width
    sh, g = cv._shadow
    decal(p, (0, 0, 0.0), (0, 0, 1), (0, 1, 0), 0.6, 0.6, sh.material("Amen_ConeShadow", rough=0.9, alpha=True, blend=True, normal=0), g, lift=0.002)
    return p.obj("traffic_cone", col, 50)


CLUB = "#9c1426"


def _crest(reg, cx, cy, h, field=CLUB, trim="#ffffff", dark="#13233f", cut=False):
    """A club crest: a shield with a sash, a ball and a star (centre cx, cy, height h). cut: on an
    alpha overlay, leave the field transparent so the tinted (club colour) surface shows through."""
    A = {"a": 1.0} if cut else {}

    def shield(k):
        w, t = 0.42 * h * k, 0.5 * h * k
        pts = [(cx - w, cy + t), (cx + w, cy + t), (cx + w, cy + t * 0.05)]
        pts += [(cx + w * math.cos(a), cy + t * 0.05 - t * 1.05 * math.sin(a)) for a in np.linspace(0.15, math.pi - 0.15, 12)]
        pts += [(cx - w, cy + t * 0.05)]
        return pts
    reg.poly(shield(1.0), col=trim, dh=0.0004, **A)
    reg.poly(shield(0.88), col=dark, **A)
    if cut:
        reg.poly(shield(0.8), a=0.0)
    else:
        reg.poly(shield(0.8), col=field)
    reg.poly([(cx - 0.34 * h, cy + 0.3 * h), (cx - 0.2 * h, cy + 0.4 * h), (cx + 0.34 * h, cy - 0.25 * h), (cx + 0.2 * h, cy - 0.35 * h)], col=trim, **A)
    reg.circle(cx - 0.12 * h, cy - 0.12 * h, 0.13 * h, col=trim, **A)
    reg.poly([(cx - 0.12 * h + 0.06 * h * math.cos(a), cy - 0.12 * h + 0.06 * h * math.sin(a)) for a in np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 6)[:-1]], col=dark, **A)
    st = [(cx + 0.14 * h + (0.11 if k % 2 == 0 else 0.045) * h * math.cos(math.pi / 2 + k * math.pi / 5),
           cy + 0.17 * h + (0.11 if k % 2 == 0 else 0.045) * h * math.sin(math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    reg.poly(st, col="#f0c419", **A)


def _tint_skin(reg, seed=1, fade=0.08, grime=0.15, streaks=0.18, prism=0.0):
    """Paint a TINT_ detail region in greys: white = the slot colour; mottle, UV fade towards the
    top, grime on the lowest `grime` of the height, rain streaks, an optional microprism relief."""
    reg.fill(col="#ffffff", rough=0.35)
    reg.mottle(0.04, 6, seed)
    if fade:
        reg.field(lambda X, Y: np.clip((Y / reg.h - 0.5) * 2, 0, 1) * fade / 0.25, mul=1.25)
    if grime:
        rs, cs, X, Y = reg.grid()
        n = noise(X.shape, (max(2, int(reg.h * 12)), max(2, int(reg.w * 12))), seed + 1, 4)
        g = np.clip(1 - Y / (reg.h * grime) + (n - 0.5) * 0.8, 0, 1) * 0.55
        reg.apply(rs, cs, g, mul=0.45)
    if streaks:
        rs, cs, X, Y = reg.grid()
        cols = np.clip((noise((1, X.shape[1]), (1, max(4, int(reg.w * 30))), seed + 2, 3, wrap=False)[0] - 0.45) * 3, 0, 1)
        lens = 0.4 * reg.h * (0.4 + 0.6 * noise((1, X.shape[1]), (1, max(3, int(reg.w * 8))), seed + 3, 2, wrap=False)[0])
        reg.apply(rs, cs, np.clip(1 - (reg.h - Y) / lens, 0, 1) * cols * streaks, mul=0.6)
    if prism:
        rs, cs, X, Y = reg.grid()
        q = prism
        u, v = (X + Y * 0.577) / q, (Y * 1.155) / q
        d = np.minimum(np.abs(u - np.round(u)), np.abs(v - np.round(v)))
        reg.cv.ht[rs, cs] += 0.00012 * np.clip(d * 2, 0, 1)


def sign_post(col, M):
    """The club's entrance sign: a folded aluminium tray sign (1.9 x 1.1 m, 25 mm returns) whose
    face is the club colour (TINT_club_sign: recoloured per club) with retroreflective microprism
    relief, fade, streaks and grime; printed over it the white border, crest, WELCOME TO THE
    TRAINING GROUND and a directions strip with arrows; dome rivets on the face at the rails;
    bare brushed-aluminium back with two 40 x 20 channel rails and 80 mm banded clips round two
    76 mm satin-black posts with domed caps, set in concrete collars. Front +y."""
    lo, hi = (-0.95, 0.005, 0.75), (0.95, 0.03, 1.85)
    # the tinted face
    tc = Canvas("Amen_SignFace", 1024, 512, col="#ffffff", rough=0.35)
    face = tc.region(0, 0, 950, 500, 1.9, 1.1)
    _tint_skin(face, 3, fade=0.1, grime=0.1, streaks=0.2, prism=0.004)
    face.field(lambda X, Y: np.clip(1 - Y / 0.09, 0, 1) * np.clip((noise(X.shape, (6, 40), 8, 3) - 0.3) * 2, 0, 1) * 0.5, mul=0.55)
    tint = tc.tint_material("TINT_club_sign", "#941829", rough=0.38, normal=1.0)
    # the print: transparent except the white and dark artwork
    oc = Canvas("Amen_SignPrint", 1024, 1024, col="#ffffff", rough=0.35, alpha=0.0)
    f = oc.alloc(1.9, 1.1, 526)
    f.frame(0.018, 0.018, 1.882, 1.082, 0.008, r=0.02, col="#ffffff", a=1.0)
    f.rect(0.035, 0.05, 1.865, 0.29, r=0.012, col="#f2f2ee", a=1.0)
    f.text("←", 0.07, 0.115, 0.1, weight="bold", col="#1a1a1a")
    f.text("Reception · Car park", 0.22, 0.135, 0.06, weight="bold", col="#1a1a1a")
    f.text("Pitches 1-4", 1.68, 0.135, 0.06, weight="bold", align="r", col="#1a1a1a")
    f.text("→", 1.83, 0.115, 0.1, weight="bold", align="r", col="#1a1a1a")
    f.rect(1.147, 0.07, 1.153, 0.27, col="#3a3a38")
    f.field(lambda X, Y: np.clip(1 - (Y - 0.05) / 0.08, 0, 1) * noise(X.shape, (6, 40), 4, 3) * 0.45, 0.035, 0.05, 1.865, 0.29, col="#6e6a5e")
    _crest(f, 0.27, 0.69, 0.56, cut=True)
    f.text("WELCOME TO THE", 0.53, 0.86, 0.06, weight="bold", col="#ffffff", track=0.12, a=1.0)
    f.text("TRAINING GROUND", 0.53, 0.6, 0.17, weight="black", col="#ffffff", max_w=1.31, a=1.0)
    f.rect(0.53, 0.53, 1.84, 0.536, col="#ffffff", a=1.0)
    f.text("First Team · Academy · Women's", 0.53, 0.4, 0.062, weight="medium", col="#ffffff", max_w=1.31, a=1.0)
    # the white print weathers too: streaks from the top edge, grime at the foot
    f.streaks(None, 0.45, "#7a7468", 0.3, 4, density=30)
    f.field(lambda X, Y: np.clip(1 - Y / 0.18, 0, 1) * noise(X.shape, 10, 6, 4) * 0.4, col="#6a6458")
    # the back and returns: brushed aluminium
    back = oc.alloc(1.9, 1.1, 180)
    R = {"-y": back, "+x": oc.alloc(0.025, 1.1, 180), "-x": oc.alloc(0.025, 1.1, 180), "+z": oc.alloc(1.9, 0.025, 180)}
    R["-z"] = R["+z"]
    for k in ("-y", "+x", "-x", "+z"):
        reg = R[k]
        reg.fill(col="#c4c7c8", rough=0.42, metal=0.55, a=1.0)
        rs, cs, X, Y = reg.grid()
        n = noise((X.shape[0], X.shape[1]), (max(2, int(reg.h * 120)), 2), 11, 3)
        oc.col[rs, cs] *= (0.9 + 0.18 * n)[..., None]
        reg.field(lambda X, Y: noise(X.shape, 4, 12, 4) * 0.3, col="#8c8a80")
    back.text("PART No. TS-1911  ·  ACM 3mm  ·  RA1", 0.3, 0.2, 0.025, weight="bold", col="#303030", metal=0.2)
    alu = oc.material("Amen_SignBack")
    prt = oc.material("Amen_SignPrint", rough=0.35, alpha=True, normal=0)
    # trodden soil round the post feet: a soft-edged ground decal
    gc = Canvas("Amen_SignSoil", 512, 128, col="#4a3c2c", rough=0.95, alpha=0.0)
    g = gc.region(0, 0, 512, 128, 2.2, 0.55)
    g.mottle(0.25, 10, 5)
    for x in (0.3, 1.9):
        g.field(lambda X, Y, x=x: np.clip(1 - np.hypot(X - x, (Y - 0.275) * 1.4) / 0.24 + (noise(X.shape, 12, 7, 4) - 0.5) * 0.8, 0, 1) ** 1.3 * 0.9, a=1.0)
    soil = gc.material("Amen_SignSoil", rough=0.95, alpha=True, blend=True, normal=0)

    p = PM(0.004)
    yp = -0.054
    # 76 mm galvanised posts, mottled spangle, a grime band at the foot
    gc_ = Canvas("Amen_SignPost", 256, 512, col="#a5aaab", rough=0.45, metal=0.8)
    gp = gc_.region(0, 0, 256, 512, 0.24, 2.0)
    _galv(gp, 21, 0.0, streaks=True)
    post = gc_.material("Amen_SignPost")
    zt = 1.95
    for x in (-0.8, 0.8):
        p.r = 0.008
        p.lathe([(0.0, -0.02), (0.1, -0.02), (0.1, 0.03), (0.085, 0.055), (0.05, 0.06), (0.0, 0.06)], M["concrete"], (x, yp, 0), sides=20)
        p.r = 0.0
        i0 = len(p.F)
        p.lathe([(0.038, 0.05), (0.038, zt), (0.0385, zt)], post, (x, yp, 0), sides=20)
        wrap_uv(p, i0, gp, z0=0.0, centre=(x, yp))
        p.r = 0.003
        p.lathe([(0.0405, zt - 0.006), (0.0405, zt + 0.012), (0.036, zt + 0.016), (0.0, zt + 0.016)], M["black"], (x, yp, 0), sides=20)
        for z in (1.0, 1.6):
            p.r = 0.003
            p.lathe([(0.0415, z - 0.04), (0.0415, z + 0.04)], M["chrome"], (x, yp, 0), sides=20)
            p.box((x - 0.03, yp + 0.03, z - 0.04), (x + 0.03, -0.015, z + 0.04), M["galv"])
            _hexbolt(p, (x, yp - 0.041, z), 0.009, 0.008, M["galv"], axis=(0, -1, 0))
    p.r = 0.003
    for z in (1.0, 1.6):
        p.box((-0.92, -0.015, z - 0.02), (0.92, 0.005, z + 0.02), M["alu"])
    # the tray panel: 30 mm corner radius, a visible aluminium edge return
    p.r = 0.003
    i0 = len(p.F)
    ring = [(x, z) for x, z, _ in rrect(1.9, 1.1, 0.03, 0.0, n=4)]
    _loft(p, [[(x, y, z + 1.3) for x, z in ring] for y in (0.005, 0.03)], alu)
    for fi in range(i0, len(p.F)):
        P_ = np.array([p.V[i] for i in p.F[fi]])
        n_ = _newell_(P_)
        if abs(n_[1]) > 0.7:
            reg = R["-y"] if P_[:, 1].mean() < 0.02 else None
            if reg is None:
                p.uv[fi] = [face.uv(min(1.9, max(0, 0.95 - q[0])), min(1.1, max(0, q[2] - 0.75))) for q in P_]
                p.mi[fi] = p.m(tint)
            else:
                p.uv[fi] = [reg.uv(min(1.9, max(0, q[0] + 0.95)), min(1.1, max(0, q[2] - 0.75))) for q in P_]
        else:
            p.uv[fi] = [R["+z"].uv(min(1.9, max(0, q[0] + 0.95 + q[2] * 0.0)), min(0.025, max(0, q[1] - 0.005))) for q in P_]
    # the print follows the face, with its own rounded corners
    p.r = 0.0
    nq = 8
    for i in range(nq):
        for j in range(nq // 2):
            x0, x1 = -0.95 + 1.9 * i / nq, -0.95 + 1.9 * (i + 1) / nq
            z0, z1 = 0.75 + 1.1 * j / (nq // 2), 0.75 + 1.1 * (j + 1) / (nq // 2)
            q = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
            p.face([(x, 0.0316, z) for x, z in q][::-1], prt, [f.uv(0.95 - x, z - 0.75) for x, z in q][::-1])
    # dome rivets through the face into the rails
    p.r = 0.0
    for z, xs in ((1.0, (-0.8, -0.3, 0.3, 0.8)), (1.6, (-0.865, 0.865))):
        for x in xs:
            p.cyl((x, 0.031, z), (x, 0.0345, z), 0.007, M["alu"], 10, r2=0.0045)
    # rail end caps
    for z in (1.0, 1.6):
        for sx in (-1, 1):
            p.r = 0.002
            p.box((sx * 0.92 - 0.004, -0.017, z - 0.022), (sx * 0.92 + 0.004, 0.006, z + 0.022), M["black"])
    decal(p, (0, -0.03, 0.0), (0, 0, 1), (0, 1, 0), 2.2, 0.55, soil, g, lift=0.004)
    return p.obj("sign_post", col)


def _brushed(reg, seed=1, col="#c3c6c8", rough=0.3, radial=None):
    """Brushed stainless: fine directional streaks (radial about `radial` if given)."""
    reg.fill(col=col, rough=rough, metal=0.9)
    rs, cs, X, Y = reg.grid()
    if radial is not None:
        cx, cy = radial
        ang = np.arctan2(Y - cy, X - cx)
        rad = np.hypot(X - cx, Y - cy)
        n = noise((256, 64), (2, 64), seed, 3)
        ii = np.clip((rad / max(reg.w, reg.h) * 2 * 255).astype(int), 0, 255)
        jj = ((ang + math.pi) / (2 * math.pi) * 63).astype(int)
        v = n[ii, jj]
    else:
        v = noise(X.shape, (max(2, int(reg.h * 200)), 3), seed, 3)
    reg.cv.col[rs, cs] *= (0.88 + 0.2 * v)[..., None]
    reg.cv.rough[rs, cs] += (v - 0.5) * 0.1


def water_fountain(col, M):
    """An outdoor drinking fountain and bottle filler (powder-coated steel, 1.0 m): a 200 mm
    square pedestal with 12 mm rounded corners on a bolted flange plate and a concrete pad; an
    access panel with security screws, an enamel DRINKING WATER plate and a 50 mm stainless dome
    button in a bezel on the front; a real 70 mm deep stainless bottle-filler recess on the right
    with a spout, a sensor window and a drip grille; a brushed-stainless bowl 80 mm deep with a
    raised rolled rim, a perforated drain disc, limescale ring, and a chrome bubbler under its
    guard; chipped edges and a scuffed kick zone; wet ground in front. Front +y."""
    Z0 = 0.062
    cv = Canvas("Amen_Fountain", 1024, 1024, col="#1e462d", rough=0.45)
    lo, hi = (-0.1, -0.1, Z0), (0.1, 0.1, 0.86)
    R = box_regions(cv, lo, hi, 600, ("+y", "-y", "+x", "-x", "+z"))
    niche = cv.alloc(0.25, 0.3, 600)
    bowl = cv.alloc(0.34, 0.34, 900)
    rim = cv.alloc(1.0, 0.12, 300)
    pad = cv.alloc(0.46, 0.46, 300)
    green = "#1e462d"
    for i, (k, reg) in enumerate(R.items()):
        reg.fill(col=green, rough=0.45)
        reg.field(lambda X, Y: noise(X.shape, (int(reg.h * 150) + 1, int(reg.w * 150) + 1), 20 + i, 2), dh=0.00012)
        weather(reg, z0=Z0, seed=4 * i + 3, mottle=0.04, grime_h=0.18, grime="#4f4636", grime_k=0.5, streak_k=0.0)
        # irregular chips to dark grey primer (some to steel), grime and splash in the lowest 150 mm
        reg.edge_wear(0, 0, reg.w, reg.h, 0.014, "#3c3f3e", 0.85, 30 + i)
        if k != "+z":
            rng = np.random.default_rng(50 + i)
            for _ in range(14):
                x, y = rng.uniform(0.01, reg.w - 0.01), rng.uniform(0.0, 0.3)
                reg.poly(_ellipse(x, y, rng.uniform(0.003, 0.008), rng.uniform(0.002, 0.005), rng.uniform(0, math.pi), 7), col="#3c3f3e", rough=0.6)
            reg.grime(0.15, "#3e3a2e", 0.65, 60 + i, y_ground=-Z0)
    f = R["+y"]
    f.frame(0.03, 0.08, 0.17, 0.46, 0.003, r=0.004, col="#0c1a12", dh=-0.0012)
    for y in (0.11, 0.43):
        f.circle(0.1, y, 0.006, col="#9ea2a2", dh=0.0008, metal=0.8, rough=0.35)
        f.line([(0.096, y), (0.104, y)], 0.0012, col="#2a2a2a")
    f.rect(0.018, 0.53, 0.182, 0.63, r=0.005, col="#ebebe4", metal=0.0, rough=0.25, dh=0.0005)
    for x_, y_ in ((0.026, 0.538), (0.174, 0.538), (0.026, 0.622), (0.174, 0.622)):
        f.circle(x_, y_, 0.003, col="#9a9c9a", dh=0.0006, metal=0.8)
    f.frame(0.022, 0.534, 0.178, 0.626, 0.002, r=0.004, col="#14306a")
    f.text("DRINKING", 0.1, 0.588, 0.022, weight="black", align="c", col="#14306a", metal=0.0)
    f.text("WATER", 0.1, 0.553, 0.022, weight="black", align="c", col="#14306a", metal=0.0)
    f.field(lambda X, Y: np.clip(1 - np.abs(X - 0.1) / 0.07, 0, 1) * np.clip((Y - 0.55) / 0.25, 0, 1) * np.clip((noise(X.shape, (4, 30), 5, 3) - 0.4) * 3, 0, 1) * 0.35, col="#8a9488")
    f.field(lambda X, Y: np.clip(1 - np.abs(X - 0.1) / 0.06, 0, 1) * np.clip((Y - 0.45) / 0.3, 0, 1) * 0.0, col="#000000")
    f.field(lambda X, Y: noise(X.shape, 20, 7, 3) * 0.3, 0.018, 0.53, 0.182, 0.63, col="#a8a89c")
    R["+x"].text("BOTTLE REFILL", 0.1, 0.682, 0.012, weight="bold", align="c", col="#e8ece8")
    # the recess liner: stainless, a sensor window, scale and drips at the bottom
    _brushed(niche, 6, rough=0.3)
    niche.rect(0.09, 0.24, 0.16, 0.265, r=0.004, col="#101214", rough=0.05, metal=0.0)
    niche.circle(0.15, 0.2525, 0.003, col="#d02020", emit="#ff3030", metal=0.0)
    niche.field(lambda X, Y: np.clip(1 - Y / 0.08, 0, 1) * noise(X.shape, 12, 7, 3) * 0.7, col="#d7d2bd", metal=0.3)
    _brushed(bowl, 3, radial=(0.17, 0.17))
    bowl.ring(0.17, 0.17, 0.09, 0.105, col="#e2dfcf", k=0.5, rough=0.6, metal=0.3)
    bowl.field(lambda X, Y: np.clip(1 - np.hypot(X - 0.17, Y - 0.17) / 0.09, 0, 1) * noise(X.shape, 14, 6, 4) * 0.45, col="#8a8270", metal=0.4)
    bowl.circle(0.17, 0.17, 0.022, col="#b4b7b9", dh=0.0008, metal=0.9)
    for rr_, nn in ((0.0, 1), (0.007, 6), (0.014, 12)):
        for k in range(nn):
            a_ = 2 * math.pi * k / nn
            bowl.circle(0.17 + rr_ * math.cos(a_), 0.17 + rr_ * math.sin(a_), 0.0022, col="#0c0c0c", dh=-0.001, metal=0.0)
    _brushed(rim, 4)
    rim.field(lambda X, Y: np.clip(1 - Y / 0.04, 0, 1) * noise(X.shape, (4, 30), 8, 3) * 0.4, col="#7d776a")
    _concrete(pad, 9)
    pad.field(lambda X, Y: np.clip(1 - np.maximum(np.abs(X - 0.23), np.abs(Y - 0.23)) / 0.2, 0, 1) * 0.0 + noise(X.shape, 6, 9, 4) * 0.35, col="#4a4a3a")
    mat = cv.material("Amen_Fountain")
    gc = Canvas("Amen_FountainWet", 256, 256, col="#3a3626", rough=0.25, alpha=0.0)
    g = gc.region(0, 0, 256, 256, 0.7, 0.7)
    g.field(lambda X, Y: np.clip(1 - np.hypot(X - 0.35, (Y - 0.3) * 1.3) / 0.3 + (noise(X.shape, 10, 3, 4) - 0.5) * 0.7, 0, 1) ** 1.5 * 0.55, a=1.0)
    wet = gc.material("Amen_FountainWet", rough=0.2, alpha=True, blend=True, normal=0)

    p = PM(0.004)
    # concrete pad and the bolted flange plate
    p.r = 0.012
    i0 = len(p.F)
    p.box((-0.23, -0.23, -0.03), (0.23, 0.23, 0.05), mat)
    project(p, i0, pad, (-0.23, -0.23, 0), (1, 0, 0), (0, 1, 0))
    p.r = 0.003
    p.box((-0.15, -0.15, 0.05), (0.15, 0.15, Z0), M["galv"])
    for x in (-0.12, 0.12):
        for y in (-0.12, 0.12):
            _hexbolt(p, (x, y, Z0), 0.011, 0.009, M["galv"])
    # the pedestal: below, through and above the bottle-filler recess (a notch in its plan)
    p.r = 0.002
    zn0, zn1, dn = 0.43, 0.72, 0.09
    sq = rrect(0.2, 0.2, 0.02, 0.0)
    notch = [(x, y) for x, y, _ in sq] + [(0.1, -0.055), (0.1 - dn, -0.055), (0.1 - dn, 0.055), (0.1, 0.055)]
    i0 = len(p.F)
    _loft(p, [[(x, y, z) for x, y, _ in sq] for z in (Z0, zn0)], mat, cap0=False)
    _loft(p, [[(x, y, z) for x, y, _ in sq] for z in (zn1, 0.86)], mat, cap1=False)
    i1 = len(p.F)
    _loft(p, [[(x, y, z) for x, y in notch] for z in (zn0, zn1)], mat, cap0=False, cap1=False)
    i2 = len(p.F)
    for fi in range(i1, i2):
        P_ = np.array([p.V[i] for i in p.F[fi]])
        c_ = P_.mean(0)
        if c_[0] < 0.099 and c_[0] > 0.0 and abs(c_[1]) < 0.0565:
            n_ = np.cross(P_[1] - P_[0], P_[2] - P_[0])
            if abs(n_[0]) > abs(n_[1]):
                p.uv[fi] = [niche.uv(0.07 + (q[1] + 0.055), q[2] - zn0) for q in P_]
            else:
                p.uv[fi] = [niche.uv(0.07 * (q[0] - (0.1 - dn)) / dn if q[1] < 0 else 0.18 + 0.07 * (q[0] - (0.1 - dn)) / dn, q[2] - zn0) for q in P_]
    # recess floor and roof (stainless)
    p.r = 0.0
    p.face([(0.1 - dn, -0.055, zn0), (0.1, -0.055, zn0), (0.1, 0.055, zn0), (0.1 - dn, 0.055, zn0)], mat,
           [niche.uv(0.07, 0.0), niche.uv(0.07, 0.07), niche.uv(0.18, 0.07), niche.uv(0.18, 0.0)])
    p.face([(0.1 - dn, 0.055, zn1), (0.1, 0.055, zn1), (0.1, -0.055, zn1), (0.1 - dn, -0.055, zn1)], mat,
           [niche.uv(0.18, 0.29), niche.uv(0.18, 0.22), niche.uv(0.07, 0.22), niche.uv(0.07, 0.29)])
    box_uv(p, i0, R, lo, hi)
    # spout and drip grille in the recess
    p.r = 0.002
    xb = 0.1 - dn
    p.cyl((xb, 0.0, zn1 - 0.03), (xb + 0.045, 0.0, zn1 - 0.03), 0.009, M["chrome"], 12)
    p.cyl((xb + 0.045, 0.0, zn1 - 0.022), (xb + 0.045, 0.0, zn1 - 0.055), 0.008, M["chrome"], 12, r2=0.006)
    p.box((xb + 0.002, -0.052, zn0), (0.098, 0.052, zn0 + 0.014), M["chrome"])
    for i in range(6):
        for j in range(8):
            decal(p, (xb + 0.012 + i * 0.012, -0.042 + j * 0.012, zn0 + 0.014), (0, 0, 1), (1, 0, 0), 0.006, 0.006, M["rubber"], lift=0.0006)
    # the bowl: neck, outer skin with a raised rolled rim, 80 mm deep dish
    p.r = 0.0
    # cap plate on the column, a short tapered neck, a shallow pressed dish with a rolled rim
    p.r = 0.003
    p.box((-0.11, -0.11, 0.86), (0.11, 0.11, 0.872), M["galv"])
    p.r = 0.0
    p.lathe([(0.06, 0.872), (0.045, 0.88), (0.04, 0.886)], M["chrome"], sides=24)
    i0 = len(p.F)
    p.lathe([(0.04, 0.886), (0.07, 0.888), (0.11, 0.9), (0.14, 0.922), (0.158, 0.948), (0.165, 0.961), (0.168, 0.968), (0.166, 0.974), (0.16, 0.976), (0.154, 0.972)], mat, sides=36)
    wrap_uv(p, i0, rim, z0=0.86)
    i0 = len(p.F)
    p.lathe([(0.154, 0.972), (0.148, 0.962), (0.13, 0.938), (0.1, 0.915), (0.06, 0.901), (0.025, 0.896), (0.0, 0.896)], mat, sides=36)
    project(p, i0, bowl, (-0.17, -0.17, 0), (1, 0, 0), (0, 1, 0))
    # centre strainer
    p.lathe([(0.022, 0.8965), (0.022, 0.8985), (0.0, 0.8995)], M["chrome"], sides=20)
    # push button: a 35 mm chrome button in a 70 mm square stainless escutcheon, front, under the bowl
    p.r = 0.002
    p.box((-0.035, 0.1, 0.755), (0.035, 0.104, 0.825), M["chrome"])
    for x_ in (-0.028, 0.028):
        for z_ in (0.762, 0.818):
            p.cyl((x_, 0.104, z_), (x_, 0.106, z_), 0.0035, M["galv"], 6)
    p.cyl((0, 0.104, 0.79), (0, 0.112, 0.79), 0.0175, M["chrome"], 20, r2=0.016)
    # bubbler (chrome, mushroom head) under its guard hood
    p.r = 0.002
    bz = 0.899
    p.cyl((0, -0.07, bz - 0.004), (0, -0.07, bz + 0.03), 0.014, M["chrome"], 14)
    p.cyl((0, -0.07, bz + 0.03), (0, -0.07, bz + 0.042), 0.018, M["chrome"], 14, r2=0.012)
    p.cyl((0, -0.065, bz + 0.038), (0, -0.04, bz + 0.058), 0.007, M["chrome"], 10, r2=0.005)
    p.r = 0.0
    hood = [(0.032 * math.cos(t), -0.07 - 0.006, bz + 0.005 + 0.07 * math.sin(t)) for t in np.linspace(0, math.pi, 11)]
    _tube(p, hood, 0.0045, M["chrome"], 6, bend=0.0)
    _tube(p, [(0.032 * math.cos(t), -0.07 + 0.035 * math.sin(t), bz + 0.005 + 0.04 * math.sin(t)) for t in np.linspace(0, math.pi, 11)], 0.004, M["chrome"], 6, bend=0.0)
    decal(p, (0, 0.24, 0.0), (0, 0, 1), (0, 1, 0), 0.7, 0.7, wet, g, lift=0.003)
    return p.obj("water_fountain", col, 40)


def _ellipse(cx, cy, a, b, ang, n=10):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(cx + a * math.cos(t) * ca - b * math.sin(t) * sa, cy + a * math.cos(t) * sa + b * math.sin(t) * ca)
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def _box_leaf_mats():
    """Clipped box (Buxus): an opaque tiling leaf face and an alpha sprig card."""
    import bpy
    if bpy.data.materials.get("Amen_BoxFace"):
        return bpy.data.materials["Amen_BoxFace"], bpy.data.materials["Amen_BoxSprig"]
    rng = np.random.default_rng(12)
    cv = Canvas("Amen_BoxFace", 512, 512, col="#16260f", rough=0.6)
    r = cv.region(0, 0, 512, 512, 0.4, 0.4)
    greens = ["#2a4a1c", "#33561f", "#3d6224", "#476b2a", "#2f5020", "#55782f", "#263f18"]
    for _ in range(5200):
        x, y = rng.uniform(0, 0.4), rng.uniform(0, 0.4)
        a_ = rng.uniform(0.0055, 0.0085)
        c = greens[rng.integers(len(greens))]
        ang = rng.uniform(0, math.pi)
        for dx in (-0.4, 0.0, 0.4):
            for dy in (-0.4, 0.0, 0.4):
                if 0 <= x + dx + a_ * 1.2 and x + dx - a_ * 1.2 <= 0.4 and 0 <= y + dy + a_ * 1.2 and y + dy - a_ * 1.2 <= 0.4:
                    r.poly(_ellipse(x + dx, y + dy, a_, a_ * 0.55, ang), col=c, h=rng.uniform(0.0005, 0.0015))
    r.field(lambda X, Y: noise(X.shape, 4, 3, 3) * 0.3, mul=0.7)
    face = cv.material("Amen_BoxFace", rough=0.68, repeat=True, normal=1.0, spec=0.3)
    sc = Canvas("Amen_BoxSprig", 256, 256, col="#3b6626", rough=0.6, alpha=0.0)
    q = sc.region(0, 0, 256, 256, 0.1, 0.1)
    for k in range(4):
        ox, oy = (k % 2) * 0.05, (k // 2) * 0.05
        base, tip = (ox + 0.025, oy + 0.003), (ox + 0.025 + rng.normal(0, 0.006), oy + 0.047)
        q.line([base, tip], 0.0012, col="#2f5020", a=1.0)
        for j in range(11):
            t = 0.1 + 0.85 * j / 11
            px, py = base[0] + (tip[0] - base[0]) * t, base[1] + (tip[1] - base[1]) * t
            for sgn in (-1, 1):
                ang = math.pi / 2 + sgn * rng.uniform(0.6, 1.1)
                L = 0.0055 * (1.1 - 0.4 * t)
                q.poly(_ellipse(px + math.cos(ang) * L * 0.8, py + math.sin(ang) * L * 0.8, L, L * 0.55, ang),
                       col=greens[rng.integers(len(greens))], a=1.0)
    sprig = sc.material("Amen_BoxSprig", rough=0.68, alpha=True, normal=0, spec=0.3)
    return face, sprig


def _grass_mat():
    import bpy
    if bpy.data.materials.get("Amen_GrassCard"):
        return bpy.data.materials["Amen_GrassCard"]
    rng = np.random.default_rng(21)
    cv = Canvas("Amen_GrassCard", 512, 512, col="#7a8a4a", rough=0.6, alpha=0.0)
    r = cv.region(0, 0, 512, 512, 0.5, 0.5)
    cols = ["#5f7432", "#768748", "#93955a", "#4f6229", "#a99c62", "#6c7f3c", "#56702e"]
    for _ in range(110):
        x0 = rng.normal(0.25, 0.04)
        lean = rng.normal(0, 0.12)
        hgt = rng.uniform(0.25, 0.48)
        pts = [(x0 + lean * (t ** 1.6) * hgt * 2, 0.005 + hgt * t) for t in np.linspace(0, 1, 9)]
        w0 = rng.uniform(0.003, 0.005)
        c = cols[rng.integers(len(cols))]
        for (x_, y_), (x2, y2), t in zip(pts, pts[1:], np.linspace(0, 1, 9)):
            r.line([(x_, y_), (x2, y2)], w0 * (1 - 0.8 * t) + 0.0008, col=c, a=1.0)
    r.field(lambda X, Y: np.clip(1 - Y / 0.12, 0, 1) * 0.6, mul=0.55)
    return cv.material("Amen_GrassCard", rough=0.6, alpha=True, normal=0)


def _icosphere(sub=3):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(sub):
        cache, F2 = {}, []

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                m = V[i] + V[j]
                V.append(m / np.linalg.norm(m))
                cache[k] = len(V) - 1
            return cache[k]
        for a_, b_, c_ in F:
            ab, bc, ca = mid(a_, b_), mid(b_, c_), mid(c_, a_)
            F2 += [(a_, ab, ca), (b_, bc, ab), (c_, ca, bc), (ab, bc, ca)]
        F = F2
    return np.array(V), F


def _box_ball(p, c, rad, face, sprig, rng, n_cards=50, flat=0.85):
    """A clipped box ball: an icosphere body (gently lumpy, flattened a little on its seat) under a
    tiling leaf face mapped by its dominant axis (no stretching, no pole), plus small leaf sprigs
    lying on the surface to break the silhouette."""
    c = np.asarray(c, float)
    V, F = _icosphere(3)
    ph = rng.uniform(0, 2 * math.pi, 6)

    def disp(d):
        k = 1 + 0.06 * math.sin(3.1 * d[0] + ph[0]) * math.sin(2.7 * d[1] + ph[1]) * math.sin(2.3 * d[2] + ph[2])
        k += 0.03 * math.sin(7.3 * d[0] + 5.1 * d[2] + ph[3]) * math.sin(6.7 * d[1] + ph[4])
        k += 0.012 * math.sin(17.0 * d[0] + 13.0 * d[1] + 11.0 * d[2] + ph[5])
        return k
    P = np.array([c + d * rad * disp(d) * np.array([1.0, 1.0, flat]) for d in V])
    P[:, 2] = np.maximum(P[:, 2], c[2] - rad * flat * 0.8)
    tile = 0.4
    p.r = 0.0
    for f in F:
        q = P[list(f)]
        n_ = V[list(f)].mean(0)
        ax = int(np.argmax(np.abs(n_)))
        keep = [i for i in range(3) if i != ax]
        p.face(q, face, [((x[keep[0]] + 7.3 * ax) / tile, (x[keep[1]] + 3.1 * ax) / tile) for x in q])
    idx = rng.choice(len(V), n_cards, replace=False)
    for i in idx:
        d = V[i]
        if d[2] < -0.45:
            continue
        q = P[i]
        t = np.cross(d, rng.normal(0, 1, 3))
        t /= np.linalg.norm(t) + 1e-9
        D = t * 0.9 + d * 0.3
        D /= np.linalg.norm(D)
        W = np.cross(d, D)
        W /= np.linalg.norm(W) + 1e-9
        sz = rng.uniform(0.035, 0.055)
        A = q - D * sz * 0.5 - d * 0.004
        u0, v0 = rng.integers(2) * 0.5, rng.integers(2) * 0.5
        p.face([A - W * sz / 2, A + W * sz / 2, A + W * sz / 2 + D * sz, A - W * sz / 2 + D * sz], sprig,
               [(u0, v0), (u0 + 0.5, v0), (u0 + 0.5, v0 + 0.5), (u0, v0 + 0.5)])


def _grass_blades(p, base, rng, mat, reg, n=44):
    """An ornamental grass clump (fountain habit): blades that rise from the crown and arch out,
    the outer ones lower and further over, tips curving down; base green to straw tips."""
    base = np.asarray(base, float)
    for k in range(n):
        az = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0, 0.05)
        o = base + np.array([rr * math.cos(az), rr * math.sin(az), 0.0])
        out = rr / 0.05
        tilt = 0.15 + 0.9 * out ** 0.8 + rng.normal(0, 0.12)
        L = rng.uniform(0.4, 0.62) * (1.0 - 0.25 * out)
        d0 = np.array([math.cos(az), math.sin(az), 0.0])
        side = np.array([-math.sin(az), math.cos(az), 0.0])
        w0 = rng.uniform(0.006, 0.009)
        col_ = rng.integers(4)
        segs = 7
        # integrate the bending direction
        P = [o]
        for i in range(segs):
            t = (i + 0.5) / segs
            ang = min(tilt * (0.3 + 1.7 * t ** 1.3), 2.3)
            P.append(P[-1] + (d0 * math.sin(ang) + np.array([0, 0, math.cos(ang)])) * L / segs)
        if k % 6 == 0:
            # a seed head on some of the taller blades
            tip = P[-1]
            sh = tip + (P[-1] - P[-2]) * 0.3
            p.cyl(tuple(P[-1]), tuple(sh + (P[-1] - P[-2]) * 0.6), 0.009, mat, 6, r2=0.002)
        tw = rng.normal(0, 0.4)
        for i in range(segs):
            t0, t1 = i / segs, (i + 1) / segs
            wa, wb = w0 * (1 - 0.85 * t0), w0 * (1 - 0.85 * t1)
            sa = side * math.cos(tw * t0) + np.array([0, 0, 1.0]) * 0.0
            sb = side * math.cos(tw * t1)
            q = [P[i] - sa * wa, P[i] + sa * wa, P[i + 1] + sb * wb, P[i + 1] - sb * wb]
            u0 = col_ * 0.25 * reg.w
            p.face(q, mat, [reg.uv(u0 + 0.02 * reg.w, t0 * reg.h), reg.uv(u0 + 0.23 * reg.w, t0 * reg.h),
                            reg.uv(u0 + 0.23 * reg.w, t1 * reg.h), reg.uv(u0 + 0.02 * reg.w, t1 * reg.h)])


def planter(col, M):
    """A cast-concrete street planter (1.5 x 0.7 x 0.55 m): a recessed plinth, bevelled edges, a
    70 mm rim with the inner wall showing down to bark mulch, aggregate speckle, form-tie marks,
    water streaks below the lip and soil splash at the foot; planted with two clipped box balls
    and a clump of ornamental grass. Front +y."""
    rng = np.random.default_rng(4)
    cv = Canvas("Amen_Planter", 1024, 1024, col="#9a968c", rough=0.88)
    lo, hi = (-0.75, -0.35, 0.04), (0.75, 0.35, 0.55)
    R = box_regions(cv, lo, hi, 400, ("+y", "-y", "+x", "-x", "+z"))
    inner = cv.alloc(1.36 * 2 + 0.56 * 2, 0.07, 250)
    mulch = cv.alloc(1.36, 0.56, 380)
    for i, (k, reg) in enumerate(R.items()):
        _concrete(reg, 10 + i, col="#9c988e", grime=False)
        if k != "+z":
            # form-tie marks, water streaks from the lip, soil splash at the foot
            for x in np.arange(0.15, reg.w, 0.3):
                reg.circle(x, 0.25, 0.006, col="#7d7a72", dh=-0.0008)
            reg.streaks(None, 0.35, "#6a675c", 0.45, 20 + i, density=25)
            reg.grime(0.18, "#4c4334", 0.75, 30 + i, y_ground=-0.04)
            reg.grime(0.1, "#3f4a2a", 0.5, 35 + i, y_ground=-0.04)
            reg.mottle(0.06, 3, 38 + i)
            reg.edge_wear(0, 0, reg.w, reg.h, 0.008, "#c4c0b5", 0.5, 40 + i)
        else:
            reg.field(lambda X, Y: noise(X.shape, 8, 5, 4) * 0.55, col="#6a665a")
            reg.field(lambda X, Y: np.clip((noise(X.shape, 6, 6, 4) - 0.5) * 3, 0, 1) * 0.5, col="#4e5e30")
            reg.field(lambda X, Y: np.clip(1 - np.minimum.reduce([X - 0.07, 1.43 - X, Y - 0.07, 0.63 - Y]) / 0.02, 0, 1) * 0.5, col="#4d4637")
            reg.edge_wear(0, 0, reg.w, reg.h, 0.012, "#c9c5ba", 0.8, 47)
    _concrete(inner, 50, col="#8d897e")
    inner.field(lambda X, Y: np.clip(1 - Y / 0.05, 0, 1) * 0.8, col="#3f372a")
    mulch.fill(col="#3b2a1c", rough=0.9)
    bark = ["#4a3322", "#5a3e28", "#3b2a1c", "#634631", "#33251a", "#6b4c35"]
    for _ in range(2600):
        x, y = rng.uniform(0, 1.36), rng.uniform(0, 0.56)
        w_, h_ = rng.uniform(0.008, 0.026), rng.uniform(0.005, 0.011)
        ang = rng.uniform(0, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [(x + ca * dx - sa * dy, y + sa * dx + ca * dy) for dx, dy in ((-w_ / 2, -h_ / 2), (w_ / 2, -h_ / 2), (w_ / 2, h_ / 2), (-w_ / 2, h_ / 2))]
        mulch.poly(pts, col=bark[rng.integers(len(bark))], h=rng.uniform(0.002, 0.006))
    for _ in range(8):
        x, y = rng.uniform(0.05, 1.3), rng.uniform(0.05, 0.5)
        mulch.poly(_ellipse(x, y, 0.02, 0.011, rng.uniform(0, math.pi)), col=str(rng.choice(["#5e4424", "#6a5230", "#4f3a20"])), h=0.007)
    mulch.field(lambda X, Y: noise(X.shape, 12, 70, 4) * 0.4, mul=0.7)
    mat = cv.material("Amen_Planter")
    face, sprig = _box_leaf_mats()
    gc = Canvas("Amen_GrassBlade", 256, 256, col="#6a7a3a", rough=0.6)
    gblade = gc.region(0, 0, 256, 256, 1.0, 1.0)
    for k, (c0, c1) in enumerate((("#46642d", "#b4aa6e"), ("#4e6c30", "#a8a868"), ("#3e5c28", "#8f9a58"), ("#56743a", "#c0b480"))):
        gblade.field(lambda X, Y, c0=c0: np.ones_like(X), k * 0.25, 0, k * 0.25 + 0.25, 1.0, col=c0)
        gblade.field(lambda X, Y: np.clip((Y - 0.45) / 0.5, 0, 1), k * 0.25, 0, k * 0.25 + 0.25, 1.0, col=c1)
        gblade.field(lambda X, Y: np.clip(1 - Y / 0.15, 0, 1) * 0.5, k * 0.25, 0, k * 0.25 + 0.25, 1.0, mul=0.5)
    grass = gc.material("Amen_GrassBlade", rough=0.6, normal=0, spec=0.3)

    p = PM(0.015)
    p.r = 0.006
    p.box((-0.725, -0.325, 0.0), (0.725, 0.325, 0.045), _paint("Amen_PlanterPlinth", "#5d5a53", 0.8))
    p.r = 0.015
    i0 = len(p.F)
    _loft(p, [rrect(1.5, 0.7, 0.02, z) for z in (0.04, 0.55)], mat, cap0=True, cap1=False)
    _loft(p, [rrect(1.5, 0.7, 0.02, 0.55), rrect(1.36, 0.56, 0.012, 0.55)], mat, cap0=False, cap1=False)
    box_uv(p, i0, R, lo, hi)
    i0 = len(p.F)
    _loft(p, [rrect(1.36, 0.56, 0.012, z) for z in (0.55, 0.48)], mat, cap0=False, cap1=False)
    # the inner wall unrolled along one strip
    for fi in range(i0, len(p.F)):
        P_ = np.array([p.V[i] for i in p.F[fi]])
        uvs = []
        for q in P_:
            x, y = q[0], q[1]
            if abs(y) > 0.27:
                s_ = (x + 0.68) if y > 0 else 1.36 + 0.56 + (0.68 - x)
            else:
                s_ = 1.36 + (0.28 - y) if x > 0 else 2 * 1.36 + 0.56 + (y + 0.28)
            uvs.append(inner.uv(min(inner.w, max(0.0, s_)), q[2] - 0.48))
        p.uv[fi] = uvs
    # bark mulch mounded 60 mm in the middle
    p.r = 0.0

    def mz(x, y):
        return 0.48 + 0.025 * max(0.0, 1 - (x / 0.68) ** 2) * max(0.0, 1 - (y / 0.28) ** 2)
    nxm, nym = 14, 6
    for i in range(nxm):
        for j in range(nym):
            xs = [-0.675 + 1.35 * (i + a_) / nxm for a_ in (0, 1)]
            ys = [-0.275 + 0.55 * (j + b_) / nym for b_ in (0, 1)]
            q = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])]
            p.face([(x, y, mz(x, y)) for x, y in q], mat, [mulch.uv(x + 0.68, y + 0.28) for x, y in q])
    stem = _paint("Amen_BoxStem", "#4a3a28", 0.8)
    for (bx, by, br) in ((-0.33, 0.0, 0.235), (0.14, 0.03, 0.19)):
        bz = mz(bx, by) + 0.05 + br * 0.85
        p.r = 0.0
        p.cyl((bx, by, mz(bx, by) - 0.01), (bx, by, bz - br * 0.5), 0.018, stem, 8, r2=0.012)
        _box_ball(p, (bx, by, bz), br, face, sprig, rng, 220)
    _grass_blades(p, (0.48, -0.02, mz(0.48, -0.02) - 0.01), rng, grass, gblade, 60)
    return p.obj("planter", col, 40)


def _sticker(reg, x0, y0, x1, y1, bg, lines, fg="#111111", weight="bold", r=0.004, wear=0.0, seed=0):
    """A printed vinyl sticker: lines [(text, size)] centred, top down."""
    reg.rect(x0, y0, x1, y1, r=r, col=bg, dh=0.0002, rough=0.3)
    y = y1 - (y1 - y0) * 0.12
    for t, h in lines:
        y -= h
        reg.text(t, (x0 + x1) / 2, y, h, weight=weight, align="c", col=fg, max_w=(x1 - x0) * 0.9)
        y -= h * 0.5
    if wear:
        reg.field(lambda X, Y: np.clip((noise(X.shape, 8, seed, 3) - 0.62) * 6, 0, 1) * wear, x0, y0, x1, y1, mul=0.75)


def _edges(reg, wear=0.6, seed=0, col="#8f9294"):
    reg.edge_wear(0, 0, reg.w, reg.h, 0.007, col, wear * 0.7, seed)


def ticket_machine(col, M):
    """A pay-and-display machine (Strada class, 1675 x 475 x 380 mm): navy powder-coated body on a
    plinth with bolt covers, one moulded canopy with a solar panel overhanging the front, a
    recessed control panel (tariff board, screen behind glass, keypad, coin and card slots,
    contactless reader, ticket and cancel buttons), a ticket cup with a flap, stickers, door
    seam and lock, edge wear and grime. Front +y."""
    cv = Canvas("Amen_Ticket", 1024, 1024, col="#1f2c48", rough=0.5)
    lo, hi = (-0.2375, -0.18, 0.04), (0.2375, 0.18, 1.52)
    R = box_regions(cv, lo, hi, 500, ("+y", "-y", "+x", "-x"))

    def top(y):
        return 1.64 - (y + 0.19) * 0.07 / 0.44
    clo, chi = (-0.2575, -0.19, 1.52), (0.2575, 0.25, 1.64)
    C = box_regions(cv, clo, chi, 400, ("+y", "-y", "+x", "-x", "+z"))
    sol = cv.alloc(0.42, 0.34, 420)
    scr = cv.alloc(0.12, 0.09, 1200)
    navy = "#1e378c"
    for i, (k, reg) in enumerate(R.items()):
        reg.fill(col=navy, rough=0.45)
        reg.field(lambda X, Y: noise(X.shape, (int(reg.h * 150), int(reg.w * 150)), 40 + i, 2), dh=0.00015)
        weather(reg, z0=0.04, seed=3 * i + 1, mottle=0.03, grime_h=0.25, grime="#2e2c28", grime_k=0.55,
                streak_len=0.5, streak="#15203f", streak_k=0.3)
        reg.field(lambda X, Y: np.clip((Y - 1.0) / 0.5, 0, 1) * 0.12, col="#7f90c0")
        reg.field(lambda X, Y: np.clip(1 - (Y - 0.0) / 0.25, 0, 1) * 0.5, rough=0.8)
        rng = np.random.default_rng(90 + i)
        for _ in range(10):
            x, y = rng.uniform(0.01, reg.w - 0.01), rng.uniform(0.02, 0.3) ** 1.2
            reg.circle(x, y, rng.uniform(0.0015, 0.004), col="#585d62", rough=0.5, k=0.7)
    f = R["+y"]
    # door seam with hinge marks, header band, recessed control panel ground
    f.frame(0.012, 0.025, 0.463, 1.465, 0.004, r=0.006, col="#0a0f1c", dh=-0.003)
    for y in (0.25, 1.2):
        f.rect(0.004, y, 0.014, y + 0.06, r=0.002, col="#3c4352", dh=0.0008)
    f.rect(0.035, 1.345, 0.44, 1.445, r=0.006, col="#1d4fa0", dh=0.0004)
    f.rect(0.045, 1.355, 0.125, 1.435, r=0.008, col="#ffffff")
    f.rect(0.05, 1.36, 0.12, 1.43, r=0.006, col="#1d4fa0")
    f.text("P", 0.085, 1.373, 0.045, weight="black", align="c", col="#ffffff")
    f.text("PAY & DISPLAY", 0.285, 1.398, 0.03, weight="black", align="c", col="#ffffff", max_w=0.29)
    f.text("Display ticket clearly inside windscreen", 0.285, 1.365, 0.0105, weight="medium", align="c", col="#dfe8f6", max_w=0.29)
    f.rect(0.035, 0.76, 0.44, 1.33, r=0.004, col="#2a2f37", rough=0.45)
    # tariff board
    tb = (0.05, 0.965, 0.3, 1.315)
    f.rect(*tb, r=0.004, col="#f3f3ef", dh=0.0003, rough=0.25)
    f.rect(tb[0], 1.275, tb[2], tb[3], col="#1d4fa0")
    f.text("TARIFF", 0.175, 1.285, 0.02, weight="black", align="c", col="#ffffff")
    rows = (("Up to 1 hour", "£1.20"), ("Up to 2 hours", "£2.40"), ("Up to 4 hours", "£4.00"), ("All day", "£6.50"))
    for i, (a_, b_) in enumerate(rows):
        y = 1.245 - i * 0.033
        f.text(a_, 0.062, y, 0.0135, weight="medium", col="#1a1a1a")
        f.text(b_, 0.288, y, 0.0135, weight="bold", align="r", col="#1a1a1a")
        f.rect(0.06, y - 0.009, 0.29, y - 0.0082, col="#b9bcc2")
    f.text("Charges apply Mon-Sat 8am-6pm", 0.175, 1.095, 0.0095, weight="medium", align="c", col="#333333", max_w=0.23)
    f.text("Blue Badge holders free · Max stay 10h", 0.175, 1.077, 0.0085, weight="regular", align="c", col="#333333", max_w=0.23)
    f.rect(0.062, 0.99, 0.288, 1.06, r=0.003, col="#fde9a8")
    f.text("1 Insert coins or tap card", 0.07, 1.042, 0.0095, weight="medium", col="#222222", max_w=0.21)
    f.text("2 Press green button", 0.07, 1.024, 0.0095, weight="medium", col="#222222", max_w=0.21)
    f.text("3 Take ticket below", 0.07, 1.006, 0.0095, weight="medium", col="#222222", max_w=0.21)
    # slot and reader labels, keypad surround
    f.text("COINS", 0.1375, 0.812, 0.0085, weight="bold", align="c", col="#d7dbe2")
    f.text("CARD", 0.2375, 0.812, 0.0085, weight="bold", align="c", col="#d7dbe2")
    f.text("TICKET", 0.35, 0.794, 0.0085, weight="bold", align="r", col="#d7dbe2")
    f.rect(0.315, 0.83, 0.43, 0.91, r=0.006, col="#16181c")
    _contactless(f, 0.37, 0.872, 0.042)
    # finger polish and grime where hands go: round the keypad, slots, reader and the ticket cup
    for cx_, cy_, rx_, ry_ in ((0.37, 1.07, 0.06, 0.08), (0.24, 0.89, 0.12, 0.05), (0.2375, 0.65, 0.1, 0.07)):
        f.field(lambda X, Y, cx_=cx_, cy_=cy_, rx_=rx_, ry_=ry_: np.clip(1 - np.hypot((X - cx_) / rx_, (Y - cy_) / ry_), 0, 1) * noise(X.shape, 30, 77, 3), col="#6c7480", rough=0.2, k=0.6)
    f.rect(0.31, 0.97, 0.43, 1.145, r=0.006, col="#1b1e23")
    # help intercom above the screen
    f.rect(0.315, 1.24, 0.425, 1.315, r=0.006, col="#3a3f47", dh=0.0006)
    for i in range(5):
        for j in range(3):
            f.circle(0.33 + i * 0.008, 1.258 + j * 0.008, 0.0022, col="#0c0d0f", dh=-0.0008)
    f.circle(0.4, 1.27, 0.012, col="#e8b400", dh=0.001)
    f.text("HELP", 0.4, 1.29, 0.008, weight="black", align="c", col="#ffffff")
    # ticket cup surround and labels
    f.rect(0.157, 0.57, 0.318, 0.7, r=0.006, col="#0b0c0e")
    f.field(lambda X, Y: np.clip((Y - 0.57) / 0.12, 0, 1) * 0.8, 0.165, 0.575, 0.31, 0.695, col="#000000")
    f.text("TAKE TICKET", 0.2375, 0.715, 0.013, weight="bold", align="c", col="#e8ecf2")
    # stickers, ID plate, lock surround
    _sticker(f, 0.04, 0.47, 0.17, 0.53, "#f2c200", [("NO CHANGE", 0.017), ("GIVEN", 0.017)], wear=0.5, seed=3)
    _sticker(f, 0.25, 0.43, 0.44, 0.53, "#ffffff", [("PAY BY PHONE", 0.016), ("Location 4417", 0.014), ("or use the app", 0.01)],
             fg="#1d4fa0", wear=0.4, seed=4)
    f.rect(0.31, 0.3, 0.43, 0.34, r=0.003, col="#a9adb1", dh=0.0004, rough=0.3, metal=0.8)
    f.text("MACHINE 0147", 0.37, 0.313, 0.0105, weight="bold", align="c", col="#2a2a2a", metal=0.0)
    _bolt_dots(f, [(0.316, 0.32), (0.424, 0.32)], 0.003)
    f.circle(0.43, 0.39, 0.02, col="#14192a", dh=0.0005)
    # sides: a faded tag on one, a council asset sticker on the other
    sl = R["-x"]
    sl.text("ZEKO", 0.18, 0.6, 0.06, weight="black", align="c", col="#b8b8b8", k=0.22, cond=1.2)
    sl.line([(0.08, 0.575), (0.3, 0.585)], 0.005, col="#b8b8b8", k=0.2)
    _sticker(R["+x"], 0.12, 1.25, 0.24, 1.31, "#ffffff", [("ASSET P-0147", 0.012), ("PARKING SERVICES", 0.008)], fg="#222222", wear=0.5, seed=5)
    for i, (k, reg) in enumerate(C.items()):
        reg.fill(col="#202d49", rough=0.45)
        reg.mottle(0.05, 4, 60 + i, rough=0.04)
        _edges(reg, 0.45, 70 + i)
    C["+y"].text("PAY HERE", C["+y"].w / 2, 0.018, 0.022, weight="black", align="c", col="#ffffff")
    # solar panel: cells in an aluminium frame
    sol.fill(col="#a8acb0", rough=0.3, metal=0.9)
    sol.rect(0.012, 0.012, 0.408, 0.328, col="#0d1424", rough=0.12, metal=0.0, dh=-0.001)
    for i in range(6):
        for j in range(4):
            x0, y0 = 0.016 + i * 0.0653, 0.016 + j * 0.0778
            sol.rect(x0 + 0.002, y0 + 0.002, x0 + 0.0633, y0 + 0.0758, r=0.004, col="#16223f")
            for k in range(1, 3):
                sol.rect(x0 + 0.002, y0 + k * 0.025, x0 + 0.0633, y0 + k * 0.025 + 0.0008, col="#6c7486")
    sol.field(lambda X, Y: noise(X.shape, 4, 77, 4) * 0.4, col="#7c7a70")
    # screen UI
    scr.fill(col="#0e1a24", rough=0.08)
    scr.rect(0, 0.072, 0.12, 0.09, col="#1d4fa0")
    scr.text("Car Park 2", 0.006, 0.077, 0.008, weight="bold", col="#ffffff")
    scr.text("Insert coins or", 0.06, 0.054, 0.0078, weight="medium", align="c", col="#e3ecf5")
    scr.text("present card", 0.06, 0.043, 0.0078, weight="medium", align="c", col="#e3ecf5")
    scr.text("£0.00", 0.06, 0.016, 0.016, weight="bold", align="c", col="#ffe066")
    rs, cs, X, Y = scr.grid()
    cv.emit[rs, cs] = cv.col[rs, cs]
    bk = R["-y"]
    bk.frame(0.02, 0.06, 0.455, 1.44, 0.004, r=0.006, col="#0a0f1c", dh=-0.003)
    bk.rect(0.455, 0.06, 0.47, 1.44, col="#1a2a55", dh=0.0012)
    bk.rect(0.12, 1.1, 0.36, 1.22, r=0.004, col="#d9dbdf", dh=0.0005, metal=0.6, rough=0.35)
    for i, t in enumerate(("PARKING SERVICES", "Machine 0147  ·  Car Park 2", "Faults: 0300 555 0147")):
        bk.text(t, 0.24, 1.19 - i * 0.032, 0.016 if i == 0 else 0.012, weight="bold" if i == 0 else "medium", align="c", col="#1a2a4a", metal=0.0)
    sv = R["+x"]
    for j in range(8):
        y = 1.12 + j * 0.03
        sv.rect(0.08, y, 0.28, y + 0.016, r=0.003, col="#0d1530", dh=-0.003, bevel=0.004)
    body = cv.material("Amen_TicketBody")
    screen = cv.material("Amen_TicketScreen", rough=0.08, emission=1.2, normal=0)
    bezel = _paint("Amen_TicketBezel", "#2b2f36", 0.4)

    p = PM(0.006)
    p.r = 0.006
    _loft(p, [rrect(0.52, 0.41, 0.02, z) for z in (0.0, 0.04)], M["steel"])
    for x in (-0.23, 0.23):
        for y in (-0.175, 0.175):
            p.sphere((x, y, 0.04), 0.016, M["black"], 10, 6, z_min=0.0)
    p.r = 0.012
    i0 = len(p.F)
    _loft(p, [rrect(0.475, 0.36, 0.022, z) for z in (0.04, 1.52)], body, cap0=False)
    box_uv(p, i0, R, lo, hi)
    # canopy: one moulding sloping ~9 degrees forward, overhanging the front by 70 mm
    i0 = len(p.F)
    ring = rrect(0.515, 0.44, 0.03, 0.0, cy=0.03)
    inner = rrect(0.495, 0.42, 0.02, 0.0, cy=0.03)
    p.r = 0.01
    lip_in = rrect(0.509, 0.434, 0.027, 0.0, cy=0.03)
    _loft(p, [[(x, y, top(y) - 0.035) for x, y, _ in lip_in], [(x, y, top(y) - 0.035) for x, y, _ in ring], [(x, y, top(y) - 0.004) for x, y, _ in ring],
              [(x, y, top(y)) for x, y, _ in inner]], body, cap0=False)
    p.face([(x, y, top(y) - 0.032) for x, y, _ in lip_in][::-1], body)
    # the body rises inside the hood
    _loft(p, [[(x, y, 1.52) for x, y, _ in rrect(0.475, 0.36, 0.014, 0.0)], [(x, y, top(y) - 0.033) for x, y, _ in rrect(0.475, 0.36, 0.014, 0.0)]], body, cap0=False, cap1=False)
    box_uv(p, i0, C, clo, chi)
    nrm = np.array([0, 0.159, 1.0]) / math.hypot(0.159, 1.0)
    up = np.array([0, -1.0, 0.159]) / math.hypot(1.0, 0.159)
    decal(p, (0, 0.03, top(0.03)), nrm, up, 0.42, 0.34, body, sol, lift=0.004)
    # control panel bezel (the panel sits 15 mm back inside it)
    yf = 0.18
    p.r = 0.004
    for (x0, z0, x1, z1) in ((-0.2025, 0.8, -0.18, 1.37), (0.18, 0.8, 0.2025, 1.37), (-0.18, 0.8, 0.18, 0.818), (-0.18, 1.352, 0.18, 1.37)):
        p.box((x0, yf - 0.002, z0), (x1, yf + 0.015, z1), bezel)
    # screen with glass, keypad, buttons, slots, reader
    X = lambda lx: 0.2375 - lx            # front-region x (from the viewer's left) to world x
    p.r = 0.003
    p.box((X(0.435), yf, 1.155), (X(0.305), yf + 0.008, 1.265), M["black"])
    decal(p, (X(0.37), yf + 0.008, 1.21), (0, 1, 0), (0, 0, 1), 0.12, 0.09, screen, scr, lift=0.0006)
    decal(p, (X(0.37), yf + 0.008, 1.21), (0, 1, 0), (0, 0, 1), 0.126, 0.096, clear_glass(), lift=0.0016)
    for i in range(3):
        for j in range(4):
            c = np.array([X(0.335 + i * 0.035), yf, 1.155 - 0.035 - j * 0.032 + 0.0])
            p.cyl(c, c + (0, 0.006, 0), 0.0085, M["chrome"], 10, r2=0.007)
    p.cyl((X(0.3725), yf, 0.838), (X(0.3725), yf + 0.008, 0.838), 0.017, _plastic("Amen_BtnGreen", "#1f9e4a", 0.35), 16, r2=0.015)
    p.cyl((X(0.418), yf, 0.838), (X(0.418), yf + 0.007, 0.838), 0.011, _plastic("Amen_BtnRed", "#c42323", 0.35), 14, r2=0.01)
    p.box((X(0.17), yf, 0.875), (X(0.105), yf + 0.012, 0.915), M["chrome"])
    decal(p, (X(0.1375), yf + 0.012, 0.895), (0, 1, 0), (0, 0, 1), 0.032, 0.0055, M["rubber"], lift=0.0006)
    p.box((X(0.275), yf, 0.875), (X(0.2), yf + 0.016, 0.915), M["black"])
    decal(p, (X(0.2375), yf + 0.016, 0.895), (0, 1, 0), (0, 0, 1), 0.058, 0.004, M["rubber"], lift=0.0006)
    i0 = len(p.F)
    p.box((X(0.43), yf, 0.87), (X(0.315), yf + 0.01, 0.95), body)
    box_uv(p, i0, R, lo, hi)
    # ticket cup: a hood with a smoked flap
    p.r = 0.003
    # recessed ticket cup: a dark pocket set 40 mm into the fascia under a small hood lip
    p.r = 0.0
    cup = _plastic("Amen_TicketCup", "#0c0d0f", 0.5)
    for (a_, b_, c_, d_) in (((-0.075, yf + 0.001, 0.605), (0.075, yf + 0.001, 0.605), (0.075, yf - 0.04, 0.615), (-0.075, yf - 0.04, 0.615)),
                             ((-0.075, yf - 0.04, 0.615), (0.075, yf - 0.04, 0.615), (0.075, yf - 0.04, 0.695), (-0.075, yf - 0.04, 0.695)),
                             ((-0.075, yf - 0.04, 0.695), (0.075, yf - 0.04, 0.695), (0.075, yf + 0.001, 0.695), (-0.075, yf + 0.001, 0.695))):
        p.face([a_, b_, c_, d_], cup)
    for sx in (-1, 1):
        p.face([(sx * 0.075, yf + 0.001, 0.605), (sx * 0.075, yf - 0.04, 0.615), (sx * 0.075, yf - 0.04, 0.695), (sx * 0.075, yf + 0.001, 0.695)], cup)
    decal(p, (0, yf + 0.0015, 0.65), (0, 1, 0), (0, 0, 1), 0.15, 0.09, cup, lift=0.0)
    p.r = 0.003
    p.box((-0.085, yf, 0.695), (0.085, yf + 0.03, 0.705), M["chrome"])
    p.box((-0.085, yf, 0.6), (0.085, yf + 0.012, 0.607), M["chrome"])
    p.cyl((X(0.43), yf, 0.43), (X(0.43), yf + 0.008, 0.43), 0.011, M["chrome"], 12)
    p.cyl((0.17, -0.18, 0.75), (0.17, -0.19, 0.75), 0.012, M["chrome"], 12)
    for z in (0.3, 0.75, 1.2):
        p.cyl((-0.226, -0.188, z - 0.04), (-0.226, -0.188, z + 0.04), 0.008, M["galv"], 8)
    for x in (-0.25, 0.25):
        for y in (-0.195, 0.195):
            _hexbolt(p, (x, y, 0.04), 0.009, 0.008, M["galv"])
    decal(p, (0.17, -0.19, 0.75), (0, -1, 0), (0, 0, 1), 0.003, 0.012, M["rubber"], lift=0.0005)

    return p.obj("ticket_machine", col, 40)


def _galv(reg, seed=1, z0=0.0, streaks=True):
    """Hot-dip galvanised steel: spangle mottle, dull patches, white-rust streaks, grime low down."""
    reg.fill(col="#b4b8ba", rough=0.48, metal=0.65)
    rs, cs, X, Y = reg.grid()
    cv = reg.cv
    n1 = noise(X.shape, (max(2, int(reg.h * 30)), max(2, int(reg.w * 30))), seed, 3)
    n2 = noise(X.shape, (max(2, int(reg.h * 3)), max(2, int(reg.w * 3))), seed + 1, 4)
    cv.col[rs, cs] *= (0.84 + 0.22 * n1 + 0.3 * (n2 - 0.5))[..., None]
    cv.rough[rs, cs] = np.clip(cv.rough[rs, cs] + (n2 - 0.5) * 0.25, 0.2, 0.9)
    if streaks:
        reg.streaks(None, 1.2, "#7c7f7e", 0.25, seed + 2, density=20)
    if z0 < 0.6:
        reg.grime(0.6, "#4c463a", 0.6, seed + 3, y_ground=-z0)


def cctv_pole(col, M):
    """A 6 m galvanised CCTV column: concrete foundation, 300 mm base plate on four studs with
    nuts and washers, a shaft tapering 150 to 76 mm, an inspection door with a triangular-key lock
    at 0.6 m, an asset plate, a CCTV warning sign on bands, a junction box on band clamps with
    its conduit, a cast bracket with a gusset carrying a bullet camera under a sun shield, a
    smoked dome camera on a pendant, and a domed top cap. Front +y."""
    cv = Canvas("Amen_CCTV", 1024, 1024, col="#b4b8ba", rough=0.48, metal=0.65)
    low = cv.alloc(0.45, 2.0, 300)
    high = cv.alloc(0.32, 4.1, 180)
    sign = cv.alloc(0.3, 0.2, 1000)          # painted at 0.3 x 0.2, shown at 0.45 x 0.3
    lens = cv.alloc(0.12, 0.12, 800)
    jlo, jhi = (-0.33, -0.1, 4.9), (-0.21, 0.1, 5.2)
    J = box_regions(cv, jlo, jhi, 500, ("+y", "-y", "+x", "-x", "+z"))
    flo, fhi = (-0.3, -0.3, -0.03), (0.3, 0.3, 0.15)
    FD = box_regions(cv, flo, fhi, 250, ("+z", "+y", "-y", "+x", "-x"))
    _galv(low, 3, 0.0)
    _galv(high, 5, 2.0)
    # inspection door (front, x = 0.75 turn), asset plate and lock
    fx = 0.45 * 0.75
    low.frame(fx - 0.05, 0.4, fx + 0.05, 0.8, 0.004, r=0.008, col="#4d5051", dh=-0.0015)
    low.field(lambda X, Y: (np.abs(X - fx) < 0.05) * (Y > 0.4) * (Y < 0.8) * 0.12, mul=0.9)
    low.rect(fx - 0.03, 1.55, fx + 0.03, 1.62, r=0.003, col="#e8e6dc", dh=0.0004, metal=0.0, rough=0.4)
    low.text("CC", fx, 1.592, 0.014, weight="black", align="c", col="#1a1a1a", metal=0.0)
    low.text("047", fx, 1.563, 0.017, weight="black", align="c", col="#1a1a1a", metal=0.0)
    low.rect(fx - 0.055, 0.82, fx + 0.055, 0.85, col="#ffd200", dh=0.0002, metal=0.0)
    low.text("DANGER 230V", fx, 0.828, 0.012, weight="black", align="c", col="#111111", metal=0.0)
    low.grime(0.5, "#5a5850", 0.55, 21)
    high.field(lambda X, Y: np.clip(1 - (3.62 - Y) / 0.8, 0, 1) * (Y < 3.62) * np.clip((noise(X.shape, (40, 6), 22, 3) - 0.45) * 3, 0, 1) * 0.6, col="#e6e8e4")
    # rust bleeding from the base plate studs
    low.field(lambda X, Y: np.clip(1 - Y / 0.25, 0, 1) * noise(X.shape, (8, 40), 9, 3) * 0.6, col="#7a4a2a")
    # the sign
    sign.fill(col="#ffd200", rough=0.35, metal=0.0)
    sign.frame(0.006, 0.006, 0.294, 0.194, 0.006, r=0.008, col="#111111")
    sign.rect(0.02, 0.07, 0.11, 0.18, r=0.004, col="#111111")
    sign.rect(0.032, 0.11, 0.08, 0.14, r=0.004, col="#ffd200")
    sign.poly([(0.08, 0.115), (0.095, 0.105), (0.095, 0.145), (0.08, 0.135)], col="#ffd200")
    sign.line([(0.05, 0.11), (0.05, 0.09), (0.035, 0.09)], 0.006, col="#ffd200")
    sign.text("CCTV", 0.2, 0.135, 0.034, weight="black", align="c", col="#111111")
    sign.text("in operation", 0.2, 0.105, 0.016, weight="bold", align="c", col="#111111")
    for i, t in enumerate(("Images are recorded for the", "prevention of crime and public", "safety. Operator: Site Security", "Enquiries 0161 496 0017")):
        sign.text(t, 0.15, 0.06 - i * 0.0135, 0.0085, weight="medium", align="c", col="#111111", max_w=0.26)
    sign.streaks(None, 0.12, "#6d6a50", 0.2, 3)
    # the bullet camera's face: lens, IR ring
    lens.fill(col="#0d0e10", rough=0.15, metal=0.0)
    lens.circle(0.06, 0.06, 0.055, col="#1b1c1f")
    for k in range(12):
        a = 2 * math.pi * k / 12
        lens.circle(0.06 + 0.04 * math.cos(a), 0.06 + 0.04 * math.sin(a), 0.006, col="#3a1414", dh=0.0004)
    lens.circle(0.06, 0.06, 0.024, col="#050506", dh=-0.001)
    lens.circle(0.06, 0.06, 0.014, col="#16223a")
    lens.circle(0.054, 0.066, 0.004, col="#c8d4e0")
    lens.field(lambda X, Y: np.clip((np.hypot(X - 0.06, Y - 0.06) - 0.0495) * 2000, 0, 1), a=0.0)
    for i, (k, reg) in enumerate(J.items()):
        reg.fill(col="#8d9092", rough=0.5, metal=0.3)
        weather(reg, z0=4.9, seed=80 + i, grime_k=0.0, streak_len=0.15, streak_k=0.3, mottle=0.06)
    J["-x"].text("JB-047", J["-x"].w / 2, 0.24, 0.02, weight="black", align="c", col="#1a1a1a", metal=0.0)
    _warning(J["-x"], J["-x"].w / 2, 0.15, 0.06)
    for i, (k, reg) in enumerate(FD.items()):
        _concrete(reg, 90 + i)
    FD["+z"].field(lambda X, Y: np.clip(1 - np.maximum(np.abs(X - 0.3), np.abs(Y - 0.3)) / 0.25, 0, 1) * 0.0 + noise(X.shape, 5, 4, 3) * 0.3, col="#5a5448")
    mat = cv.material("Amen_CCTV")
    lensm = cv.material("Amen_CCTVLens", alpha=True, rough=0.12)
    cam = _paint("Amen_CamWhite", "#e4e4df", 0.35)

    p = PM(0.004)
    # 600 x 600 x 150 concrete plinth, a grey plastic shroud over the base plate and its bolts
    p.r = 0.012
    i0 = len(p.F)
    p.box(flo, fhi, mat)
    box_uv(p, i0, FD, flo, fhi)
    # 300 x 300 x 20 flange plate on four M24 anchor bolts with nuts and washers, a welded collar
    p.r = 0.003
    i0 = len(p.F)
    p.box((-0.15, -0.15, fhi[2] + 0.012), (0.15, 0.15, fhi[2] + 0.032), mat)
    project(p, i0, low, (-0.15, -0.15, 0), (1, 0, 0), (0, 1, 0))
    p.box((-0.16, -0.16, fhi[2]), (0.16, 0.16, fhi[2] + 0.012), _paint("Amen_Grout", "#6e6a62", 0.9))
    for x in (-0.115, 0.115):
        for y in (-0.115, 0.115):
            p.cyl((x, y, fhi[2] + 0.032), (x, y, fhi[2] + 0.085), 0.012, M["galv"], 8)
            _hexbolt(p, (x, y, fhi[2] + 0.032), 0.019, 0.02, M["galv"])
    for k in range(4):
        a_ = k * math.pi / 2
        ca, sa = math.cos(a_), math.sin(a_)
        V = [(0.075, -0.004, fhi[2] + 0.032), (0.14, -0.004, fhi[2] + 0.032), (0.075, -0.004, fhi[2] + 0.16),
             (0.075, 0.004, fhi[2] + 0.032), (0.14, 0.004, fhi[2] + 0.032), (0.075, 0.004, fhi[2] + 0.16)]
        V = [(ca * x - sa * y, sa * x + ca * y, z) for x, y, z in V]
        p.faces(V, [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)], M["galv"])
    p.r = 0.0
    p.lathe([(0.09, fhi[2] + 0.032), (0.088, fhi[2] + 0.05), (0.078, fhi[2] + 0.07)], M["galv"], sides=24)
    z0 = fhi[2] + 0.06

    def R(z):
        # stepped column: a 150 mm root to 3 m, a swaged joint, 100 mm tapering to 76 mm at the top
        if z <= 3.0:
            return 0.075 - 0.004 * (z - z0) / (3.0 - z0)
        if z <= 3.12:
            return 0.071 + (0.05 - 0.071) * (z - 3.0) / 0.12
        return 0.05 + (0.038 - 0.05) * (z - 3.12) / (6.0 - 3.12)
    i0 = len(p.F)
    p.lathe([(R(z), z) for z in (z0, 0.6, 1.2, 2.0)], mat, sides=24)
    wrap_uv(p, i0, low, z0=0.0)
    i0 = len(p.F)
    p.lathe([(R(z), z) for z in (2.0, 2.98, 3.0, 3.06, 3.12, 3.6, 4.4, 5.2, 6.0)] + [(R(6.0) + 0.006, 6.0), (R(6.0) + 0.006, 6.02)], mat, sides=24)
    wrap_uv(p, i0, high, z0=2.0)
    p.r = 0.002
    p.sphere((0, 0, 6.02), R(6.0) + 0.004, M["galv"], 16, 8, z_min=0.0)
    # flush inspection door at 0.6-1.05 m: a curved panel standing 2 mm proud, with its lock
    zd0, zd1, ang = 0.6, 1.05, 0.62
    rr_ = R(0.8) + 0.002
    th = np.linspace(math.pi / 2 - ang, math.pi / 2 + ang, 7)
    p.r = 0.0
    door_m = _paint("Amen_ColDoor", "#8c9193", 0.5)
    for t0, t1 in zip(th, th[1:]):
        q = [(rr_ * math.cos(t0), rr_ * math.sin(t0), zd0), (rr_ * math.cos(t1), rr_ * math.sin(t1), zd0),
             (rr_ * math.cos(t1), rr_ * math.sin(t1), zd1), (rr_ * math.cos(t0), rr_ * math.sin(t0), zd1)]
        p.face(q, door_m)
    for zz, sgn in ((zd0, -1), (zd1, 1)):
        for t0, t1 in zip(th, th[1:]):
            q = [(R(0.8) * math.cos(t0), R(0.8) * math.sin(t0), zz), (rr_ * math.cos(t0), rr_ * math.sin(t0), zz),
                 (rr_ * math.cos(t1), rr_ * math.sin(t1), zz), (R(0.8) * math.cos(t1), R(0.8) * math.sin(t1), zz)]
            p.face(q if sgn < 0 else q[::-1], door_m)
    p.r = 0.002
    p.cyl((0, rr_ - 0.002, 0.95), (0, rr_ + 0.006, 0.95), 0.011, M["galv"], 3)
    # sign on two bands at 2.5 m
    for z in (2.4, 2.6):
        p.lathe([(R(z) + 0.003, z - 0.01), (R(z) + 0.003, z + 0.01)], M["galv"], sides=20)
        p.box((-0.015, R(z), z - 0.012), (0.015, R(z) + 0.03, z + 0.012), M["galv"])
    p.r = 0.003
    yb = R(2.5) + 0.03
    p.box((-0.227, yb, 2.348), (0.227, yb + 0.004, 2.652), M["alu"])
    decal(p, (0, yb + 0.004, 2.5), (0, 1, 0), (0, 0, 1), 0.45, 0.3, mat, sign, lift=0.0005)
    # junction box on band clamps, conduit into the pole
    p.r = 0.006
    i0 = len(p.F)
    p.box(jlo, jhi, mat)
    box_uv(p, i0, J, jlo, jhi)
    for z in (4.95, 5.15):
        p.lathe([(R(z) + 0.003, z - 0.012), (R(z) + 0.003, z + 0.012)], M["galv"], sides=20)
        p.box((-0.215, -0.012, z - 0.012), (-R(z), 0.012, z + 0.012), M["galv"])
    p.r = 0.0
    _tube(p, [(-0.27, 0.0, 4.9), (-0.27, 0.0, 4.82), (-R(4.8), 0.0, 4.78)], 0.012, M["black"], 8, bend=0.05)
    # cast bracket arm with gusset, bullet camera under a sun shield
    p.r = 0.006
    p.lathe([(R(5.7) + 0.006, 5.6), (R(5.7) + 0.006, 5.8)], M["galv"], sides=20)
    p.box((-0.05, R(5.7), 5.62), (0.05, R(5.7) + 0.02, 5.8), cam)
    p.obox((0, 0.24, 5.765), (0.06, 0.42, 0.055), cam)
    p.extrude_yz([(R(5.7) + 0.01, 5.74), (R(5.7) + 0.01, 5.63), (0.3, 5.74)], -0.01, 0.01, cam)
    p.cyl((0, 0.42, 5.74), (0, 0.42, 5.68), 0.025, cam, 14)
    pitch = -0.22
    rot = Matrix.Rotation(pitch, 4, "X")
    o = Vector((0, 0.43, 5.62))

    def T(v):
        return tuple(o + rot @ Vector(v))
    p.r = 0.008
    p.cyl(T((0, -0.14, 0)), T((0, 0.16, 0)), 0.055, cam, 20)
    p.r = 0.003
    p.cyl(T((0, 0.16, 0)), T((0, 0.175, 0)), 0.05, M["black"], 20)
    decal(p, T((0, 0.175, 0)), tuple(rot @ Vector((0, 1, 0))), tuple(rot @ Vector((0, 0, 1))), 0.1, 0.1, lensm, lens, lift=0.0006)
    # sun shield: a curved hood over the body, 30 mm beyond the face
    arc = [(0.075 * math.cos(t), 0.075 * math.sin(t)) for t in np.linspace(0.35, math.pi - 0.35, 7)]
    prof = arc + [(x * 0.94, z * 0.94) for x, z in arc[::-1]]
    V = [T((x, y, z)) for y in (-0.13, 0.205) for x, z in prof]
    m = len(prof)
    p.r = 0.002
    for j in range(m):
        j2 = (j + 1) % m
        p.face([V[j], V[j2], V[m + j2], V[m + j]], cam)
    p.face([V[j] for j in range(m)], cam)
    p.face([V[m + j] for j in range(m)][::-1], cam)
    p.cyl(T((0, -0.03, 0.055)), T((0, -0.03, 0.12)), 0.018, cam, 10)
    p.r = 0.0
    _tube(p, [T((0, -0.14, 0)), T((0, -0.2, -0.02)), (0, 0.2, 5.72)], 0.006, M["black"], 6, bend=0.03)
    # bird spikes along the sun shield
    p.r = 0.0
    for k in range(7):
        b0 = T((0, -0.1 + k * 0.045, 0.077))
        for dx_ in (-0.02, 0.02):
            p.cyl(b0, T((dx_, -0.1 + k * 0.045, 0.14)), 0.0012, M["chrome"], 3)
    # dome camera on a swan-neck arm on the right-hand side
    p.r = 0.006
    p.lathe([(R(5.95) + 0.006, 5.88), (R(5.95) + 0.006, 6.0)], cam, sides=20)
    p.r = 0.0
    _tube(p, [(R(5.95), 0, 5.95), (0.12, 0, 6.0), (0.22, 0, 5.98), (0.22, 0, 5.88)], 0.022, cam, 12, bend=0.06)
    # armoured conduits from the junction box up to the cameras
    _tube(p, [(-0.27, 0.03, 5.2), (-0.27, 0.03, 5.28), (-R(5.4) - 0.012, 0.03, 5.4), (-R(5.6) - 0.012, 0.03, 5.58), (-0.02, 0.06, 5.6), (0, 0.12, 5.74)],
          0.009, M["black"], 6, bend=0.04)
    _tube(p, [(-0.24, -0.03, 5.2), (-0.24, -0.03, 5.3), (-R(5.5) - 0.012, -0.03, 5.5), (-R(5.85) - 0.012, -0.03, 5.85), (0.05, -0.03, 5.96), (0.2, -0.02, 5.94)],
          0.009, M["black"], 6, bend=0.04)
    for z in (5.45, 5.75):
        p.lathe([(R(z) + 0.004, z - 0.008), (R(z) + 0.004, z + 0.008)], M["black"], sides=20)
    p.lathe([(0.0, 5.86), (0.08, 5.86), (0.085, 5.83), (0.085, 5.8), (0.075, 5.79)], cam, c=(0.22, 0, 0), sides=24)
    p.r = 0.0
    p.lathe([(0.072 * math.cos(t), 5.79 - 0.068 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 6)], M["glass"], c=(0.22, 0, 0), sides=24)
    return p.obj("cctv_pole", col, 40)


FIRE_RED = "#ad1a1c"
BOARD_RED = "#b3302a"


def _ext_paint(reg, kind):
    """An extinguisher's skin (one turn wide, z up): red body, BS EN 3 colour band, white
    instruction label on the front (x = w/4 faces +y), dust and scuffs."""
    w = reg.w
    reg.fill(col=FIRE_RED, rough=0.3)
    reg.mottle(0.05, 5, 3 if kind == "foam" else 4, rough=0.04)
    band = "#efe4c4" if kind == "foam" else "#141414"
    fx = w * 0.75                                   # angle +90 deg (the +y side) in wrap_uv's turn
    reg.rect(fx - 0.06, 0.39, fx + 0.06, 0.43, r=0.003, col=band, dh=0.0002)
    lx0, lx1, ly0, ly1 = fx - 0.065, fx + 0.065, 0.12, 0.37
    reg.rect(lx0, ly0, lx1, ly1, r=0.004, col="#f4f2ec", dh=0.0002, rough=0.35)
    reg.rect(lx0, ly1 - 0.035, lx1, ly1, r=0.004, col=band if kind == "foam" else "#1a1a1a")
    reg.text("FOAM" if kind == "foam" else "CO2", fx, ly1 - 0.027, 0.02, weight="black", align="c",
             col="#7a1010" if kind == "foam" else "#ffffff")
    reg.text("FIRE EXTINGUISHER", fx, ly1 - 0.05, 0.0085, weight="bold", align="c", col="#a01818", max_w=0.12)
    for i, (t, c) in enumerate((("A", "#f0f0f0"), ("B", "#f0f0f0"), ("E", "#f0f0f0")) if kind == "foam" else (("B", "#f0f0f0"), ("E", "#f0f0f0"))):
        x = lx0 + 0.018 + i * 0.03
        reg.rect(x, 0.255, x + 0.026, 0.281, r=0.002, col="#1a1a1a")
        reg.text(t, x + 0.013, 0.26, 0.016, weight="black", align="c", col=c)
    for i in range(6):
        y = 0.235 - i * 0.017
        reg.rect(lx0 + 0.01, y, lx0 + 0.01 + (0.11 if i % 3 else 0.07), y + 0.005, col="#7d7d7d")
    reg.text("9 L" if kind == "foam" else "2 kg", lx1 - 0.01, 0.135, 0.012, weight="bold", align="r", col="#202020")
    reg.field(lambda X, Y: np.clip((Y - 0.45) / 0.12, 0, 1) * noise(X.shape, 6, 8, 3) * 0.5, col="#8a7f72")
    reg.grime(0.08, "#3a332b", 0.5, 9)
    rng = np.random.default_rng(5)
    for _ in range(14):
        x, y = rng.uniform(0, w), rng.uniform(0.02, 0.15)
        reg.line([(x, y), (x + rng.uniform(0.005, 0.03), y + rng.normal(0, 0.003))], 0.0015, col="#d6d0c8", k=0.4)


def _extinguisher(p, c, kind, mat, reg, M):
    """A UK 9 L foam or 2 kg CO2 extinguisher at c (its foot on the floor)."""
    cx, cy, cz = c
    r = 0.075 if kind == "foam" else 0.064
    top = 0.47 if kind == "foam" else 0.5
    p.r = 0.0
    i0 = len(p.F)
    prof = [(r * 0.9, cz + 0.035), (r, cz + 0.045)]
    prof += [(r, cz + z) for z in np.linspace(0.07, top, 6)]
    prof += [(r * math.cos(t) + 0.0, cz + top + r * 0.75 * math.sin(t)) for t in np.linspace(0.25, math.pi / 2 - 0.25, 4)]
    prof += [(0.024, cz + top + 0.065), (0.022, cz + top + 0.07)]
    p.lathe(prof, mat, c=(cx, cy, 0), sides=24)
    wrap_uv(p, i0, reg, z0=cz, centre=(cx, cy))
    p.r = 0.003
    p.lathe([(r * 0.9, cz + 0.0), (r + 0.006, cz + 0.004), (r + 0.006, cz + 0.05), (r - 0.002, cz + 0.055)], M["black"], c=(cx, cy, 0), sides=24)
    zt = cz + top + 0.07
    p.cyl((cx, cy, zt - 0.005), (cx, cy, zt + 0.022), 0.022, M["chrome"], 14)
    p.r = 0.005
    p.obox((cx, cy, zt + 0.045), (0.042, 0.05, 0.05), M["black"])
    p.r = 0.003
    p.obox((cx + 0.06, cy, zt + 0.05), (0.11, 0.026, 0.012), M["black"])
    p.obox((cx + 0.055, cy, zt + 0.078), (0.12, 0.024, 0.01), M["black"], roll=0.22)
    # pressure gauge facing the front, safety pin, ring and tag
    g = np.array([cx - 0.002, cy + 0.025, zt + 0.045])
    p.cyl(g, g + (0, 0.008, 0), 0.0135, M["chrome"], 14)
    decal(p, g + (0, 0.008, 0), (0, 1, 0), (0, 0, 1), 0.019, 0.019, _gauge_mat(), lift=0.0005)
    p.r = 0.0
    p.cyl((cx + 0.012, cy - 0.032, zt + 0.064), (cx + 0.012, cy + 0.04, zt + 0.064), 0.0025, M["chrome"], 6)
    ring = [(cx + 0.012 + 0.011 * math.cos(t), cy + 0.042, zt + 0.064 - 0.011 + 0.011 * math.sin(t)) for t in np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 13)]
    _tube(p, ring, 0.0016, M["chrome"], 5, bend=0.0, closed=False)
    p.r = 0.001
    p.obox((cx + 0.012, cy + 0.044, zt + 0.03), (0.022, 0.002, 0.032), _plastic("Amen_TagYellow", "#e7c51c", 0.5))
    p.r = 0.0
    if kind == "foam":
        hp = [(cx - 0.02, cy, zt + 0.04), (cx - 0.07, cy + 0.005, zt + 0.02), (cx - r - 0.012, cy + 0.02, zt - 0.08), (cx - r - 0.012, cy + 0.03, cz + 0.2)]
        _tube(p, hp, 0.0085, M["rubber"], 8, bend=0.03)
        p.r = 0.002
        p.cyl((cx - r - 0.012, cy + 0.03, cz + 0.2), (cx - r - 0.012, cy + 0.03, cz + 0.08), 0.015, M["black"], 12, r2=0.012)
        p.obox((cx - r + 0.002, cy + 0.02, cz + 0.3), (0.02, 0.03, 0.02), M["black"])
    else:
        hp = [(cx - 0.02, cy, zt + 0.04), (cx - 0.065, cy + 0.005, zt + 0.0), (cx - r - 0.02, cy + 0.02, zt - 0.06)]
        _tube(p, hp, 0.008, M["rubber"], 8, bend=0.03)
        p.r = 0.002
        a = np.array([cx - r - 0.02, cy + 0.02, zt - 0.06])
        p.cyl(a, a + (0, 0, -0.04), 0.012, M["black"], 12)
        p.cyl(a + (0, 0, -0.04), (cx - r - 0.025, cy + 0.025, cz + 0.2), 0.013, M["black"], 14, r2=0.042)
        p.obox((cx - r + 0.0, cy + 0.02, cz + 0.36), (0.02, 0.03, 0.02), M["black"])


def _gauge_mat():
    import bpy
    if bpy.data.materials.get("Amen_Gauge"):
        return bpy.data.materials["Amen_Gauge"]
    cv = Canvas("Amen_Gauge", 64, 64, col="#f4f4f0", rough=0.2)
    g = cv.region(0, 0, 64, 64, 0.02, 0.02)
    a = np.radians(np.linspace(200, 150, 6))
    g.line([(0.01 + 0.006 * math.cos(t), 0.01 + 0.006 * math.sin(t)) for t in np.radians(np.linspace(210, 120, 8))], 0.0015, col="#c21d1d")
    g.line([(0.01 + 0.006 * math.cos(t), 0.01 + 0.006 * math.sin(t)) for t in np.radians(np.linspace(120, 60, 6))], 0.0015, col="#1f9a3a")
    g.line([(0.01 + 0.006 * math.cos(t), 0.01 + 0.006 * math.sin(t)) for t in np.radians(np.linspace(60, -30, 8))], 0.0015, col="#c21d1d")
    g.line([(0.01, 0.01), (0.01 + 0.0055 * math.cos(1.6), 0.01 + 0.0055 * math.sin(1.6))], 0.0008, col="#111111")
    del a
    return cv.material("Amen_Gauge", rough=0.15, normal=0)


def fire_point(col, M):
    """A rotomoulded site fire point (1.6 m): a red moulded plinth with two seats, a backboard
    with a curved rain hood, a printed FIRE POINT header and fire action notice, extinguisher ID
    signs, a manual call point, two hook brackets holding a 9 L foam and a 2 kg CO2
    extinguisher (valve heads, squeeze levers, gauges, pins and tags, hose and horn). Front +y."""
    cv = Canvas("Amen_Fire", 1024, 1024, col=FIRE_RED, rough=0.4)
    blo, bhi = (-0.35, -0.2, 0.08), (0.35, -0.14, 1.52)
    B = box_regions(cv, blo, bhi, 450, ("+y", "-y", "+x", "-x"))
    efoam = cv.alloc(0.47, 0.62, 450)
    eco2 = cv.alloc(0.41, 0.66, 450)
    plo, phi = (-0.38, -0.22, 0.0), (0.38, 0.24, 0.08)
    PL = box_regions(cv, plo, phi, 450, ("+z", "+y", "-y", "+x", "-x"))
    hood = cv.alloc(0.76, 0.42, 450)
    clo_, chi_ = (-0.42, -0.27, 0.0), (0.42, 0.29, 0.1)
    CP = box_regions(cv, clo_, chi_, 220, ("+z", "+y", "-y", "+x", "-x"))
    cp = cv.alloc(0.09, 0.09, 600)
    for i, (k, reg) in enumerate(list(B.items()) + list(PL.items())):
        reg.fill(col=BOARD_RED, rough=0.42)
        reg.field(lambda X, Y: noise(X.shape, (int(reg.h * 80) + 1, int(reg.w * 80) + 1), 50 + i, 2), dh=0.0002)
        weather(reg, z0=0.08 if k in B and i < 4 else 0.0, seed=7 * i + 2, mottle=0.06, grime_h=0.25, grime="#4a3a30",
                grime_k=0.5, streak_len=0.5, streak="#5a2a24", streak_k=0.2)
    # sun fade on the upper board
    B["+y"].field(lambda X, Y: np.clip((Y - 0.8) / 0.6, 0, 1) * 0.18, col="#d3786e")
    for reg in (B["+y"], B["-y"]):
        reg.frame(0.006, 0.006, 0.694, 1.434, 0.018, r=0.02, dh=0.003, mul=1.04)
    f = B["+y"]
    # header
    f.rect(0.03, 1.21, 0.67, 1.385, r=0.01, col="#f6f4ef", dh=0.0003, rough=0.3)
    f.rect(0.04, 1.22, 0.66, 1.375, r=0.008, col="#c81e22")
    f.text("FIRE POINT", 0.35, 1.26, 0.072, weight="black", align="c", col="#ffffff", max_w=0.58)
    # fire action notice
    nx0, ny0, nx1, ny1 = 0.04, 0.82, 0.5, 1.18
    f.rect(nx0, ny0, nx1, ny1, r=0.006, col="#f6f6f2", dh=0.0003, rough=0.3)
    f.rect(nx0, ny1 - 0.06, nx1, ny1, r=0.006, col="#1f5fae")
    f.text("FIRE ACTION", (nx0 + nx1) / 2, ny1 - 0.047, 0.034, weight="black", align="c", col="#ffffff")
    lines = [("If you discover a fire:", "bold"), ("1  Raise the alarm - break glass", "regular"), ("2  Call 999 and ask for FIRE", "regular"),
             ("3  Tackle only if safe to do so", "regular"), ("On hearing the alarm:", "bold"), ("Leave by the nearest exit", "regular"),
             ("Report to assembly point B", "regular"), ("Do not return until told", "regular")]
    for i, (t, w) in enumerate(lines):
        f.text(t, nx0 + 0.02, ny1 - 0.088 - i * 0.034, 0.016, weight=w, col="#c81e22" if w == "bold" else "#1a1a1a", max_w=0.42)
    # extinguisher ID signs above each cradle
    for x, kind in ((0.18, "foam"), (0.52, "co2")):
        f.rect(x - 0.1, 0.67, x + 0.1, 0.79, r=0.005, col="#c81e22", dh=0.0003)
        f.rect(x - 0.09, 0.68, x + 0.09, 0.715, col="#efe4c4" if kind == "foam" else "#141414")
        f.text("FOAM" if kind == "foam" else "CO2", x, 0.735, 0.03, weight="black", align="c", col="#ffffff")
        f.text("Class A B" if kind == "foam" else "Electrical", x, 0.691, 0.014, weight="bold", align="c",
               col="#6a1010" if kind == "foam" else "#ffffff")
    _sticker(f, 0.56, 0.835, 0.68, 0.905, "#f4f4f0", [("INSPECTED", 0.016), ("06/26  BS 5306", 0.012)], fg="#1a1a1a", wear=0.3, seed=9)
    for k_, reg in PL.items():
        reg.edge_wear(0, 0, reg.w, reg.h, 0.012, "#5a5652", 0.8, 77)
        reg.field(lambda X, Y: noise(X.shape, 10, 78, 4) * 0.4, col="#4a3a30")
    for i, reg in enumerate(CP.values()):
        _concrete(reg, 120 + i)
        reg.edge_wear(0, 0, reg.w, reg.h, 0.01, "#c0bcb2", 0.6, 130 + i)
    CP["+z"].field(lambda X, Y: np.clip(1 - np.maximum(np.abs(X - 0.42) - 0.36, np.abs(Y - 0.27) - 0.22) / 0.04, 0, 1) * 0.5, col="#4a4038")
    B["+y"].grime(0.25, "#4a3a30", 0.55, 91, y_ground=0.0)
    pl = PL["+z"]
    for x in (0.21, 0.55):
        pl.ring(x, 0.24 + 0.0, 0.07, 0.085, col="#6d1214", dh=-0.002)
    pl.field(lambda X, Y: noise(X.shape, 8, 12, 4) * 0.45, col="#7a6a58")
    hood.fill(col=BOARD_RED, rough=0.4)
    hood.mottle(0.06, 5, 33, rough=0.04)
    hood.field(lambda X, Y: np.ones_like(X) * 0.3, col="#e0745a")
    hood.field(lambda X, Y: noise(X.shape, 6, 34, 4) * 0.35, col="#8f8170")
    # call point: red box, white glass panel with the house symbol
    cp.fill(col="#c81e22", rough=0.35)
    cp.rect(0.015, 0.015, 0.075, 0.075, r=0.003, col="#f4f4f0", rough=0.1)
    cp.poly([(0.03, 0.035), (0.06, 0.035), (0.06, 0.05), (0.045, 0.062), (0.03, 0.05)], col="#c81e22")
    cp.text("FIRE", 0.045, 0.02, 0.008, weight="black", align="c", col="#c81e22")
    cp.text("PUSH HERE", 0.045, 0.08, 0.0055, weight="bold", align="c", col="#ffffff")
    _ext_paint(efoam, "foam")
    _ext_paint(eco2, "co2")
    mat = cv.material("Amen_FireBoard")

    p = PM(0.01)
    i0 = len(p.F)
    p.box(plo, phi, mat)
    box_uv(p, i0, PL, plo, phi)
    i0 = len(p.F)
    p.box(blo, bhi, mat)
    box_uv(p, i0, B, blo, bhi)
    # the rain hood: a curved moulding across the top, overhanging the front
    i0 = len(p.F)
    Rc, yc_, zc_ = 0.38, -0.21, 1.6 - 0.38
    angs = np.radians(np.linspace(90, 48, 10))
    outer = [(yc_ + Rc * math.cos(a_), zc_ + Rc * math.sin(a_)) for a_ in angs]
    inner = [(yc_ + (Rc - 0.006) * math.cos(a_), zc_ + (Rc - 0.006) * math.sin(a_)) for a_ in angs[::-1]]
    fy, fz = outer[-1]
    poly = outer + [(fy + 0.004, fz - 0.045), (fy - 0.004, fz - 0.045)] + inner
    p.r = 0.002
    p.extrude_yz(poly[::-1], -0.38, 0.38, mat)
    project(p, i0, hood, (-0.38, -0.21, 0), (1, 0, 0), (0, 1, 0))
    # moulded gusset cheeks carrying the hood
    p.r = 0.004
    for x in (-0.38, 0.358):
        i0 = len(p.F)
        p.extrude_yz([(-0.2, 1.3), (0.06, 1.505), (-0.2, 1.565)], x, x + 0.022, mat)
        project(p, i0, hood, (-0.38, -0.21, 0), (1, 0, 0), (0, 1, 0))
    for x in (-0.34, 0.34):
        for y in (-0.19, 0.21):
            _hexbolt(p, (x, y, 0.08), 0.009, 0.008, M["galv"])
    # two 60 x 60 posts behind the board, plinth to two-thirds height, bolted through
    p.r = 0.004
    for x in (-0.22, 0.22):
        p.box((x - 0.03, -0.26, 0.0), (x + 0.03, -0.2, 1.05), M["galv"])
        p.box((x - 0.06, -0.29, 0.0), (x + 0.06, -0.17, 0.008), M["galv"])
        for z in (0.5, 0.95):
            _hexbolt(p, (x, -0.26, z), 0.008, 0.007, M["galv"], axis=(0, -1, 0))
    # folded return lip round the board's face; stiffening channels and bolts on the back
    p.r = 0.003
    i0 = len(p.F)
    for (x0, z0, x1, z1) in ((-0.35, 0.08, -0.325, 1.52), (0.325, 0.08, 0.35, 1.52), (-0.325, 1.495, 0.325, 1.52)):
        p.box((x0, -0.14, z0), (x1, -0.128, z1), mat)
    box_uv(p, i0, B, blo, bhi)
    for z in (0.5, 1.1):
        p.box((-0.31, -0.215, z - 0.025), (0.31, -0.2, z + 0.025), M["galv"])
        for x in (-0.26, 0.0, 0.26):
            _hexbolt(p, (x, -0.215, z), 0.008, 0.007, M["galv"], axis=(0, -1, 0))
    # call point on the board's right
    p.r = 0.004
    p.box((-0.29, -0.14, 1.0), (-0.2, -0.095, 1.09), _paint("Amen_CallPoint", "#c81e22", 0.35))
    decal(p, (-0.245, -0.095, 1.045), (0, 1, 0), (0, 0, 1), 0.09, 0.09, mat, cp, lift=0.0008)
    # cradles: a hook bracket from the board under each valve neck, a rubber seat ring
    zf = 0.22                                    # the extinguishers hang with their feet 140 mm up
    for x, kind, reg in ((0.17, "foam", efoam), (-0.17, "co2", eco2)):
        rr = 0.075 if kind == "foam" else 0.064
        zn = zf + (0.47 if kind == "foam" else 0.5) + 0.045
        # J-hook bracket: a plate on the board, an arm out under the valve neck, a lip turned up
        p.r = 0.003
        p.box((x - 0.03, -0.14, zn - 0.09), (x + 0.03, -0.132, zn + 0.01), M["black"])
        p.box((x - 0.012, -0.14, zn - 0.03), (x + 0.012, -0.0, zn - 0.018), M["black"])
        p.box((x - 0.012, -0.012, zn - 0.03), (x + 0.012, 0.0, zn + 0.0), M["black"])
        # a strap round the cylinder at mid height, anchored to a second plate
        zs = zf + 0.25
        p.box((x - 0.03, -0.14, zs - 0.03), (x + 0.03, -0.132, zs + 0.03), M["black"])
        p.r = 0.0
        p.lathe([(rr + 0.004, zs - 0.012), (rr + 0.004, zs + 0.012)], M["black"], c=(x, 0.0, 0), sides=24)
        p.box((x - 0.012, -0.132, zs - 0.012), (x + 0.012, -rr + 0.01, zs + 0.012), M["black"])
        _extinguisher(p, (x, 0.0, zf), kind, mat, reg, M)
        if kind == "foam":
            # laminated inspection tag on a cable tie round the neck
            p.r = 0.001
            p.obox((x - 0.03, 0.02, zn - 0.02), (0.035, 0.002, 0.05), _plastic("Amen_TagWhite", "#f2f2ee", 0.3), roll=0.2)
            p.r = 0.0
            _tube(p, [(x - 0.02, 0.0, zn + 0.01), (x - 0.03, 0.02, zn + 0.004)], 0.0012, M["black"], 4)
    # everything stands on a 100 mm concrete plinth
    p.V = [(x_, y_, z_ + 0.1) for x_, y_, z_ in p.V]
    p.r = 0.012
    i0 = len(p.F)
    p.box(clo_, chi_, mat)
    box_uv(p, i0, CP, clo_, chi_)
    return p.obj("fire_point", col, 40)


def _concrete(reg, seed=1, col="#a7a39a", grime=True):
    """Cast concrete: aggregate speckle, blotchy tone, a few pores."""
    reg.fill(col=col, rough=0.88, metal=0.0)
    rs, cs, X, Y = reg.grid()
    rng = np.random.default_rng(seed)
    sp = rng.random(X.shape)
    cv = reg.cv
    cv.col[rs, cs] *= (1 + (sp - 0.5) * 0.16)[..., None]
    pores = (sp > 0.992).astype(float)
    cv.col[rs, cs] *= (1 - 0.45 * pores)[..., None]
    cv.ht[rs, cs] -= 0.0004 * pores
    reg.mottle(0.1, 4, seed + 1, rough=0.04)
    if grime:
        reg.field(lambda X, Y: noise(X.shape, 6, seed + 2, 4) ** 2 * 0.35, col="#6f6a5e")


def _bolt_dots(reg, pts, r, col="#9a9c9a"):
    for x, y in pts:
        reg.circle(x, y, r, col=col, dh=0.001, bevel=r * 0.5)
        reg.circle(x + r * 0.2, y - r * 0.3, r * 1.5, mul=0.8, k=0.3)


def _contactless(reg, x, y, s, col="#ffffff"):
    for k in range(4):
        rr = s * (0.18 + 0.2 * k)
        a = np.radians(np.linspace(-45, 45, 9))
        reg.line([(x - s * 0.3 + rr * math.cos(t), y + rr * math.sin(t)) for t in a], s * 0.07, col=col)


def _qr(reg, x, y, s, seed=3):
    """A QR-code sticker (white label, black modules), lower-left corner at x, y."""
    reg.rect(x, y, x + s, y + s, r=s * 0.04, col="#f4f4f0", dh=0.0002)
    rng = np.random.default_rng(seed)
    n = 21
    q = s * 0.84 / n
    g = rng.random((n, n)) > 0.52
    for cx, cy in ((0, 0), (n - 7, 0), (0, n - 7)):
        g[cy:cy + 7, cx:cx + 7] = False
    ox, oy = x + s * 0.08, y + s * 0.08
    for (i, j) in zip(*np.nonzero(g)):
        reg.rect(ox + j * q, oy + i * q, ox + (j + 1) * q, oy + (i + 1) * q, col="#111111")
    for cx, cy in ((0, 0), (n - 7, 0), (0, n - 7)):
        X0, Y0 = ox + cx * q, oy + cy * q
        reg.rect(X0, Y0, X0 + 7 * q, Y0 + 7 * q, col="#111111")
        reg.rect(X0 + q, Y0 + q, X0 + 6 * q, Y0 + 6 * q, col="#f4f4f0")
        reg.rect(X0 + 2 * q, Y0 + 2 * q, X0 + 5 * q, Y0 + 5 * q, col="#111111")


def _warning(reg, x, y, s, sym="bolt"):
    """A yellow warning triangle (centre x, y; side s) with a symbol."""
    h = s * 0.866
    tri = [(x - s / 2, y - h / 3), (x + s / 2, y - h / 3), (x, y + 2 * h / 3)]
    reg.poly(tri, col="#111111", dh=0.0002)
    k = 0.8
    reg.poly([(x + (px - x) * k, y + (py - y) * k) for px, py in tri], col="#f2c200")
    if sym == "bolt":
        reg.poly([(x + s * 0.02, y + h * 0.42), (x - s * 0.12, y - h * 0.02), (x + s * 0.0, y + h * 0.02),
                  (x - s * 0.04, y - h * 0.24), (x + s * 0.12, y + h * 0.12), (x + s * 0.01, y + h * 0.08)], col="#111111")
    else:
        reg.line([(x, y + h * 0.35), (x, y - h * 0.02)], s * 0.07, col="#111111")
        reg.circle(x, y - h * 0.14, s * 0.045, col="#111111")


def ev_charger(col, M):
    """A twin-socket 22 kW pedestal charger on its concrete pad (1.44 m): a moulded body with a
    front door (shadow-line seams, lock), recessed screen behind glass, a vertical LED status bar,
    a contactless reader, vent grille, badges, QR sticker and bay numbers; a darker moulded cap;
    two holsters, each with a Type 2 connector seated nose-down and its 28 mm tethered cable
    hanging in a loop back to a gland low on the side. Front +y."""
    cv = Canvas("Amen_EV", 1024, 1024, col="#e0e0dc", rough=0.35)
    lo, hi = (-0.15, -0.10, 0.07), (0.15, 0.10, 1.33)
    R = box_regions(cv, lo, hi, 640, ("+y", "-y", "+x", "-x"))
    clo, chi = (-0.159, -0.109, 1.33), (0.159, 0.109, 1.44)
    C = box_regions(cv, clo, chi, 640, ("+y", "-y", "+x", "-x", "+z"))
    plo, phi = (-0.3, -0.24, -0.03), (0.3, 0.24, 0.05)
    P = box_regions(cv, plo, phi, 300, ("+y", "-y", "+x", "-x", "+z"))
    scr = cv.alloc(0.14, 0.11, 1400)
    green = "#1d8f45"
    for k, reg in R.items():
        reg.fill(col="#dcdcd7", rough=0.35)
        weather(reg, z0=0.07, seed="+y-y+x-x".index(k), mottle=0.035, grime_h=0.32, grime="#5a5244", grime_k=0.5,
                streak_len=0.5, streak_k=0.16)
        # scuffs low down (bumpers, shoes, cable plugs)
        rng = np.random.default_rng(len(k) + ord(k[1]))
        for _ in range(18):
            x, y = rng.uniform(0.01, reg.w - 0.01), rng.uniform(0.02, 0.3)
            reg.line([(x, y), (x + rng.uniform(0.005, 0.035), y + rng.normal(0, 0.004))], 0.0015, col="#8a877e", k=0.45)
    f = R["+y"]
    # front door: shadow-line seams, vent grille, lock surround
    f.frame(0.022, 0.05, 0.278, 0.985, 0.005, r=0.006, col="#2e2e2c", dh=-0.003)
    f.rect(0.0, 0.993, 0.3, 0.997, col="#555552", dh=-0.001)
    for k in range(9):
        y = 0.085 + k * 0.011
        f.rect(0.06, y, 0.24, y + 0.005, r=0.0025, col="#1e1e1d", dh=-0.002, bevel=0.002)
    f.circle(0.255, 0.45, 0.016, col="#c9c9c4", dh=0.0005)
    # LED channel, logos, badges
    f.rect(0.132, 0.535, 0.168, 0.965, r=0.006, col="#2a2c2b", dh=-0.001)
    f.text("PLUGPOST", 0.15, 1.192, 0.019, weight="black", align="c", col=green)
    f.rect(0.035, 0.81, 0.11, 0.875, r=0.008, col=green)
    f.text("22", 0.0725, 0.84, 0.024, weight="black", align="c", col="#ffffff")
    f.text("kW AC", 0.0725, 0.818, 0.011, weight="bold", align="c", col="#ffffff")
    _qr(f, 0.035, 0.66, 0.075)
    f.text("Scan to pay", 0.0725, 0.645, 0.0085, weight="medium", align="c", col="#333333")
    f.rect(0.19, 0.76, 0.265, 0.875, r=0.01, col="#2b2d2c", dh=0.0008, bevel=0.002)
    _contactless(f, 0.23, 0.835, 0.04)
    f.text("TAP CARD", 0.2275, 0.775, 0.009, weight="bold", align="c", col="#ffffff")
    _warning(f, 0.225, 0.25, 0.05)
    f.text("400V AC", 0.225, 0.205, 0.009, weight="bold", align="c", col="#333333")
    for k, reg in ((1, R["-x"]), (2, R["+x"])):
        reg.circle(0.1, 1.15, 0.035, col=green)
        reg.text(str(k), 0.1, 1.135, 0.034, weight="black", align="c", col="#ffffff")
        reg.text("Type 2 · 22kW", 0.1, 0.78, 0.011, weight="bold", align="c", col="#3a3a3a")
        reg.frame(0.015, 0.05, 0.185, 1.24, 0.003, r=0.005, col="#4a4a48", dh=-0.001)
        reg.circle(0.13, 0.33, 0.03, mul=0.75, k=0.6)
    b = R["-y"]
    b.frame(0.022, 0.05, 0.278, 1.18, 0.004, r=0.006, col="#3a3a38", dh=-0.0015)
    b.rect(0.06, 0.95, 0.24, 1.07, r=0.004, col="#f2f2ee", dh=0.0002)
    for i, (t, w) in enumerate((("CHARGER ID  PP-2047", "bold"), ("Fault? 24h line on screen", "regular"),
                                ("Isolate at feeder pillar FP-3", "regular"), ("Max 32A per socket", "regular"))):
        b.text(t, 0.07, 1.045 - i * 0.026, 0.0115, weight=w, col="#202020", max_w=0.16)
    _warning(b, 0.15, 0.3, 0.06)
    for k, reg in C.items():
        reg.fill(col=green, rough=0.42)
        reg.mottle(0.06, 4, 21 + len(k), rough=0.04)
        if k == "+z":
            reg.field(lambda X, Y: noise(X.shape, 5, 9, 4) * 0.35, col="#8f8a78")
    for k, reg in P.items():
        _concrete(reg, 30 + len(k))
    P["+z"].field(lambda X, Y: np.clip(1 - np.maximum(np.abs(X - 0.3) - 0.17, np.abs(Y - 0.24) - 0.12) / 0.05, 0, 1) * 0.5, col="#4e4a43")
    # the screen: dim UI behind glass
    scr.fill(col="#071410", rough=0.1)
    scr.rect(0, 0.088, 0.14, 0.11, col="#1c7a3f")
    scr.text("PLUGPOST", 0.008, 0.094, 0.009, weight="bold", col="#e9f5ec")
    scr.text("12:41", 0.132, 0.094, 0.008, weight="medium", align="r", col="#e9f5ec")
    scr.text("Tap card or scan", 0.07, 0.068, 0.0085, weight="medium", align="c", col="#d8e6dc")
    scr.text("to start charging", 0.07, 0.056, 0.0085, weight="medium", align="c", col="#d8e6dc")
    scr.rect(0.008, 0.008, 0.066, 0.044, r=0.003, col="#13361f")
    scr.rect(0.074, 0.008, 0.132, 0.044, r=0.003, col="#0f2d4a")
    scr.text("1", 0.016, 0.03, 0.01, weight="black", col="#6fe08f")
    scr.text("Available", 0.016, 0.014, 0.0065, weight="medium", col="#c9e8d2")
    scr.text("2", 0.082, 0.03, 0.01, weight="black", col="#8cc4ff")
    scr.text("7.2 kW", 0.082, 0.014, 0.0065, weight="medium", col="#c9dcf0")
    rs, cs, X, Y = scr.grid()
    cv.emit[rs, cs] = cv.col[rs, cs] * 0.9
    # wear: finger grime round the screen and reader, sticker residue, rougher towards the base
    f.field(lambda X, Y: np.clip(1 - np.hypot((X - 0.15) / 0.12, (Y - 1.1) / 0.12), 0, 1) * noise(X.shape, 30, 61, 3) * 0.35, col="#8f8a7c")
    f.field(lambda X, Y: np.clip(1 - np.hypot((X - 0.228) / 0.06, (Y - 0.82) / 0.08), 0, 1) * noise(X.shape, 30, 62, 3) * 0.45, col="#7d786c")
    f.rect(0.04, 0.4, 0.1, 0.44, r=0.004, col="#cfcbbf", rough=0.7, k=0.6)
    f.field(lambda X, Y: noise(X.shape, 40, 63, 2) * 0.5, 0.04, 0.4, 0.1, 0.44, mul=0.85)
    for reg in R.values():
        reg.field(lambda X, Y: np.clip(1 - Y / 0.35, 0, 1) * 0.25, rough=0.65)
    C["+z"].field(lambda X, Y: np.ones_like(X) * 0.35, col="#7fb98f")
    body = cv.material("Amen_EVBody")
    screen = cv.material("Amen_EVScreen", rough=0.08, emission=1.5, normal=0)

    p = PM(0.004)
    i0 = len(p.F)
    p.r = 0.015
    p.box(plo, phi, body)
    box_uv(p, i0, P, plo, phi)
    p.r = 0.003
    p.box((-0.17, -0.12, 0.05), (0.17, 0.12, 0.07), M["steel"])
    for x in (-0.145, 0.145):
        for y in (-0.095, 0.095):
            _hexbolt(p, (x, y, 0.07), 0.008, 0.007, M["galv"])
    for x in (-0.25, 0.25):
        for y in (-0.19, 0.19):
            _hexbolt(p, (x, y, 0.05), 0.014, 0.012, M["galv"])
    p.r = 0.009
    i0 = len(p.F)
    _loft(p, [rrect(0.3, 0.2, 0.03, z) for z in (0.07, 1.325)], body, cap0=False, cap1=False)
    box_uv(p, i0, R, lo, hi)
    # rubber gasket under a moulded cap with a 20 mm overhang and drip edge
    p.r = 0.003
    _loft(p, [rrect(0.294, 0.194, 0.028, z) for z in (1.32, 1.34)], M["rubber"], cap0=False)
    p.r = 0.006
    i0 = len(p.F)
    _loft(p, [rrect(0.326, 0.226, 0.04, 1.335), rrect(0.342, 0.242, 0.045, 1.345), rrect(0.342, 0.242, 0.045, 1.42), rrect(0.322, 0.222, 0.04, 1.452)], body)
    box_uv(p, i0, C, clo, chi)
    # screen bezel, screen and glass
    p.r = 0.004
    p.box((-0.088, 0.1, 1.095), (0.088, 0.114, 1.245), M["black"])
    decal(p, (0, 0.114, 1.17), (0, 1, 0), (0, 0, 1), 0.14, 0.11, screen, scr, lift=0.0006)
    decal(p, (0, 0.114, 1.17), (0, 1, 0), (0, 0, 1), 0.15, 0.12, clear_glass(), lift=0.0016)
    p.r = 0.003
    p.box((-0.011, 0.098, 0.62), (0.011, 0.106, 1.02), _emit("Amen_EVLed", "#38e07c", 3.0))
    p.cyl((-0.105, 0.099, 0.52), (-0.105, 0.11, 0.52), 0.0115, M["chrome"], 14)
    decal(p, (-0.105, 0.11, 0.52), (0, 1, 0), (0, 0, 1), 0.003, 0.012, M["rubber"], lift=0.0005)
    grey = _plastic("Amen_EVPlug", "#3a3c3e", 0.45)
    # status light ring under the cap
    p.r = 0.0
    _loft(p, [rrect(0.304, 0.204, 0.03, z) for z in (1.312, 1.322)], _emit("Amen_EVRing", "#38e07c", 4.0), cap0=False, cap1=False)
    for s_ in (-1, 1):
        # a recessed holster on the side face; the plug sits in it nose-down, its body tucked close
        # to the panel and leaning out a little (about 20 degrees)
        th = s_ * math.radians(20)
        d = np.array([math.sin(th), 0.0, math.cos(th)])          # from the nose towards the cable
        perp = np.array([math.cos(th), 0.0, -math.sin(th)]) * s_
        p.r = 0.006
        p.box((s_ * 0.148 if s_ > 0 else -0.215, -0.06, 0.94), (0.215 if s_ > 0 else -0.148, 0.06, 1.04), M["black"])
        decal(p, (s_ * 0.19, 0.0, 1.04), (0, 0, 1), (0, 1, 0), 0.055, 0.075, M["rubber"], lift=0.0008)
        B = np.array([s_ * 0.19, 0.0, 1.03])
        _lathe_axis(p, [(0.0, -0.035), (0.03, -0.035), (0.031, 0.02), (0.036, 0.03), (0.038, 0.09), (0.034, 0.14),
                        (0.024, 0.17), (0.018, 0.2)], grey, B, d, 18)
        grip = [B + d * 0.035 + perp * 0.03, B + d * 0.06 + perp * 0.07, B + d * 0.13 + perp * 0.078, B + d * 0.17 + perp * 0.028]
        p.r = 0.0
        _tube(p, grip, 0.013, grey, 10, bend=0.04)
        # strain-relief boot, tapering into the cable
        E0 = B + d * 0.2
        _lathe_axis(p, [(0.019, 0.0), (0.017, 0.04), (0.014, 0.08)], M["rubber"], E0, d, 12)
        E = E0 + d * 0.08
        # cable hook: a curved bracket under the holster
        hz = 0.74
        hk = [(s_ * 0.15, 0.0, hz + 0.02), (s_ * 0.27, 0.0, hz), (s_ * 0.31, 0.0, hz + 0.02), (s_ * 0.32, 0.0, hz + 0.08)]
        _tube(p, hk, 0.009, M["black"], 8, bend=0.03)
        p.r = 0.004
        p.box((s_ * 0.15 - (0.01 if s_ > 0 else -0.01), -0.03, hz - 0.03), (s_ * 0.15 + (0.0 if s_ > 0 else 0.0), 0.03, hz + 0.05), M["black"])
        # gland 0.5 m up with its own boot
        G = np.array([s_ * 0.15, 0.04, 0.5])
        p.r = 0.003
        p.cyl(G, G + np.array([s_ * 0.02, 0, 0]), 0.022, M["grey"], 14)
        p.cyl(G + np.array([s_ * 0.02, 0, 0]), G + np.array([s_ * 0.07, 0, 0]), 0.019, M["rubber"], 12, r2=0.0135)
        G1 = G + np.array([s_ * 0.07, 0, 0])
        # the cable: over from the boot, down onto the hook, one loop hanging clear of the plinth,
        # back up into the gland
        pts = [E, E + d * 0.05 + perp * 0.03, np.array([s_ * 0.33, -0.01, 1.2]), np.array([s_ * 0.335, 0.0, 0.95]),
               np.array([s_ * 0.3, 0.0, hz + 0.035]), np.array([s_ * 0.33, 0.04, 0.5]), np.array([s_ * 0.31, 0.08, 0.27]),
               np.array([s_ * 0.27, 0.07, 0.25]), np.array([s_ * 0.24, 0.05, 0.36]), G1 + np.array([s_ * 0.05, 0, -0.04]), G1]
        p.r = 0.0
        _tube(p, _spline(pts, 6), 0.013, M["rubber"], 12, bend=0.0)
    return p.obj("ev_charger", col, 40)


def _interior(reg, kind, seed=1):
    """A kiosk interior seen through glass: back wall, ceiling light, desk and monitor glow (front),
    or lockers, a chair and a coat (side); blinds part-drawn; darker towards the edges."""
    rng = np.random.default_rng(seed)
    w, h = reg.w, reg.h
    reg.fill(col="#7d7a72", rough=0.6)
    reg.field(lambda X, Y: 0.35 + 0.25 * (Y / h), col="#a9a69c")
    reg.rect(w * 0.3, h * 0.9, w * 0.7, h * 0.97, col="#f4f6f0", emit="#c8ccc4")
    if kind == "desk":
        reg.rect(0, 0, w, h * 0.16, col="#4c4a46")
        reg.rect(0, h * 0.16, w, h * 0.19, col="#b9b6ad")
        reg.rect(w * 0.12, h * 0.19, w * 0.42, h * 0.48, r=0.01, col="#16181b")
        reg.rect(w * 0.135, h * 0.215, w * 0.405, h * 0.465, col="#26425e", emit="#3d6a96")
        for k in range(4):
            reg.rect(w * (0.15 + 0.06 * k), h * 0.3, w * (0.2 + 0.06 * k), h * 0.38, col="#4f7aa0", emit="#5a86ac")
        reg.rect(w * 0.25, h * 0.19, w * 0.29, h * 0.21, col="#2a2a2a")
        reg.rect(w * 0.55, h * 0.19, w * 0.72, h * 0.205, col="#efefe8")
        reg.rect(w * 0.78, h * 0.19, w * 0.82, h * 0.26, r=0.01, col="#c03a2a")
        reg.rect(w * 0.6, h * 0.0, w * 0.82, h * 0.33, r=0.04, col="#1d1e20")
        reg.rect(w * 0.88, h * 0.19, w * 0.96, h * 0.32, r=0.008, col="#2b2b2b")
        reg.rect(w * 0.05, h * 0.62, w * 0.35, h * 0.82, col="#e9e6dc")
        for k in range(5):
            reg.rect(w * 0.07, h * (0.65 + k * 0.03), w * (0.2 + rng.uniform(0, 0.12)), h * (0.66 + k * 0.03), col="#5a5a5a")
    else:
        for k in range(3):
            x0 = w * (0.05 + 0.15 * k)
            reg.rect(x0, 0, x0 + w * 0.14, h * 0.85, col="#7f8a8f")
            reg.rect(x0 + w * 0.02, h * 0.7, x0 + w * 0.12, h * 0.72, col="#3d4448")
        reg.rect(w * 0.6, 0, w * 0.86, h * 0.45, r=0.05, col="#1f2022")
        reg.poly([(w * 0.88, h * 0.75), (w * 0.98, h * 0.75), (w * 1.0, h * 0.25), (w * 0.86, h * 0.25)], col="#d9c21a")
        reg.rect(w * 0.88, h * 0.4, w * 1.0, h * 0.42, col="#c9c9c0")
    # blinds part-drawn over the top
    top = h * rng.uniform(0.55, 0.75)
    for y in np.arange(top, h, 0.03):
        reg.rect(0, y, w, y + 0.022, col="#d6d4cc")
    reg.rect(0, top - 0.01, w, top + 0.005, col="#a9a79f")
    reg.field(lambda X, Y: 1 - np.clip(np.minimum.reduce([X / (w * 0.2), (w - X) / (w * 0.2), Y / (h * 0.25), (h - Y) / (h * 0.15)]), 0, 1), mul=0.45)
    reg.cv.col[reg.grid()[0], reg.grid()[1]] *= 0.55


def _window(p, c, n, w, h, frame, glass, interior, reg, mullion=True, sill=True):
    """A window on a wall: painted interior behind clear glass, a 45 mm aluminium frame standing
    35 mm proud with a mullion, and a protruding sill. c: centre on the wall face; n: outward."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    v = np.array([0, 0, 1.0])
    u = np.cross(v, n)
    decal(p, c, n, v, w, h, interior, reg, lift=0.004)
    decal(p, c, n, v, w, h, glass, lift=0.014)
    t, d = 0.045, 0.035
    p.r = 0.004

    def bar(cu, cv_, bw, bh):
        q = c + u * cu + v * cv_ + n * d / 2
        ax = np.abs(u) * bw + np.abs(v) * bh + np.abs(n) * d
        p.box(tuple(q - ax / 2), tuple(q + ax / 2), frame)
    bar(0, h / 2 + t / 2, w + 2 * t, t)
    bar(0, -h / 2 - t / 2, w + 2 * t, t)
    bar(-w / 2 - t / 2, 0, t, h)
    bar(w / 2 + t / 2, 0, t, h)
    if mullion:
        bar(0, 0, t * 0.8, h)
    p.r = 0.0
    for (cu, cv_, bw, bh) in ((0, h / 2 - 0.006, w, 0.012), (0, -h / 2 + 0.006, w, 0.012), (-w / 2 + 0.006, 0, 0.012, h), (w / 2 - 0.006, 0, 0.012, h)):
        decal(p, c + u * cu + v * cv_, n, v, bw, bh, _plastic("Amen_Gasket", "#121212", 0.6), lift=0.016)
    if mullion:
        decal(p, c, n, v, 0.05, h, _plastic("Amen_Gasket", "#121212", 0.6), lift=0.016)
    p.r = 0.004
    if sill:
        q2 = c - v * (h / 2 + t + 0.045) + n * 0.058
        ax2 = np.abs(u) * (w + 2 * t + 0.1) + np.abs(v) * 0.03 + np.abs(n) * 0.004
        p.box(tuple(q2 - ax2 / 2), tuple(q2 + ax2 / 2), frame)
        q = c - v * (h / 2 + t + 0.02) + n * 0.03
        ax = np.abs(u) * (w + 2 * t + 0.1) + np.abs(v) * 0.04 + np.abs(n) * 0.06
        p.box(tuple(q - ax / 2), tuple(q + ax / 2), frame)


def security_hut(col, M):
    """A GRP gatehouse kiosk (3.0 x 2.4 x 2.85 m) on a steel base skid: off-white walls with
    vertical panel seams and a GRP skirting, weathered below the roof and windows; anthracite
    aluminium windows (frames, mullions, sills) with an interior behind the glass (desk, monitor
    glow, lockers, blinds); a door with frame, three hinges, lever handle, kick plate and vision
    panel, a step; a fascia roof with drip edge, gutter and downpipe; a raised SECURITY sign with
    screws, a bulkhead light, an AC condenser and a CCTV camera. Front +y."""
    cv = Canvas("Amen_Hut", 1024, 1024, col="#dfdfd8", rough=0.5)
    L, D, Z0, H = 3.0, 2.4, 0.2, 2.65
    lo, hi = (-L / 2, -D / 2, Z0), (L / 2, D / 2, H)
    door = cv.alloc(0.9, 2.05, 215)
    R = box_regions(cv, lo, hi, 92, ("+y", "-y", "+x", "-x"))
    win_f = cv.alloc(1.3, 1.05, 260)
    win_s = cv.alloc(1.3, 1.05, 180)
    flo, fhi = (-L / 2 - 0.12, -D / 2 - 0.12, H), (L / 2 + 0.12, D / 2 + 0.12, H + 0.2)
    FR = box_regions(cv, flo, fhi, 115, ("+z",))
    FR.update(box_regions(cv, flo, fhi, 100, ("+y", "-y", "+x", "-x")))
    sign = cv.alloc(1.4, 0.26, 300)
    ac = cv.alloc(0.8, 0.55, 260)
    plo_, phi_ = (-L / 2 - 0.2, -D / 2 - 0.2, -0.05), (L / 2 + 0.2, D / 2 + 0.2, 0.06)
    PD = box_regions(cv, plo_, phi_, 60, ("+z",))
    PD.update(box_regions(cv, plo_, phi_, 60, ("+y", "-y", "+x", "-x")))
    grate = cv.alloc(0.2, 0.2, 300)
    icom = cv.alloc(0.12, 0.2, 400)
    # walls: panel seams, GRP skirting, dirt under the roof edge and the windows
    for i, (k, reg) in enumerate(R.items()):
        reg.fill(col="#d6d6cf", rough=0.5)
        reg.field(lambda X, Y: noise(X.shape, (int(reg.h * 60), int(reg.w * 60)), 70 + i, 2), dh=0.0003)
        # panel bowing between the 1 m cover strips, the strips and their fixings
        reg.field(lambda X, Y: 0.5 + 0.5 * np.cos((X % 1.0) / 1.0 * 2 * math.pi), mul=0.97)
        for x in np.arange(1.0 - (reg.w % 1.0) / 2, reg.w - 0.05, 1.0):
            reg.rect(x - 0.02, 0, x + 0.02, reg.h, col="#cfcfc8", dh=0.002, bevel=0.004)
            for y in list(np.arange(0.1, reg.h, 0.4)):
                reg.circle(x, y, 0.004, col="#9a9b98", dh=0.001)
        for y in (0.08, reg.h - 0.06):
            for x in np.arange(0.12, reg.w, 0.25):
                reg.circle(x, y, 0.004, col="#9a9b98", dh=0.001)
        reg.rect(0, 0, reg.w, 0.15, col="#9d9e99", dh=0.004, rough=0.6)
        weather(reg, z0=Z0, seed=5 * i + 1, mottle=0.04, grime_h=0.35, grime="#5d5546", grime_k=0.6,
                streak_len=0.8, streak="#6f6b60", streak_k=0.45)
        reg.grime(0.25, "#5a6a3a", 0.45, 80 + i)
        for x in np.arange(1.0 - (reg.w % 1.0) / 2, reg.w - 0.05, 1.0):
            for y in (reg.h - 0.06,):
                reg.field(lambda X, Y, x=x, y=y: np.clip(1 - np.abs(X - x) / 0.006, 0, 1) * np.clip(1 - (y - Y) / 0.35, 0, 1) * (Y < y) * 0.5, col="#7a4a26")
    # under each window, dirt washes down from the sill corners
    for reg, x0, x1, zs in ((R["+y"], 1.6, 2.9, 0.88), (R["-x"], 0.55, 1.85, 0.88), (R["+x"], 0.8, 1.6, 1.18)):
        reg.field(lambda X, Y: np.clip(1 - (zs - Y) / 0.6, 0, 1) * (Y < zs) * (X > x0) * (X < x1) * noise(X.shape, (6, 40), 3, 3) * 0.6, col="#77736a")
    # the door: GRP leaf, vision panel, kick plate, sign plate
    door.fill(col="#d9d9d2", rough=0.45)
    weather(door, z0=Z0, seed=91, mottle=0.04, grime_h=0.3, grime="#5d5546", grime_k=0.5, streak_len=0.4, streak_k=0.2)
    door.rect(0.2, 1.35, 0.7, 1.85, r=0.02, col="#2a2c2e", dh=-0.003, bevel=0.01)
    door.rect(0.23, 1.38, 0.67, 1.82, r=0.01, col="#3a4046", rough=0.08)
    door.field(lambda X, Y: np.clip((X - 0.23 + (Y - 1.38) * 0.6) / 0.6, 0, 1) * 0.25, 0.23, 1.38, 0.67, 1.82, col="#a9b4bc")
    door.rect(0.03, 0.03, 0.87, 0.3, col="#b6b9bb", metal=0.85, rough=0.35, dh=0.001)
    door.field(lambda X, Y: noise(X.shape, (40, 3), 4, 2) * 0.25, 0.03, 0.03, 0.87, 0.3, mul=0.8)
    for x in (0.06, 0.84):
        for y in (0.06, 0.27):
            door.circle(x, y, 0.006, col="#7c7f80", dh=0.001)
    _sticker(door, 0.2, 1.05, 0.7, 1.25, "#ffffff", [("SECURITY OFFICE", 0.04), ("All visitors report here", 0.026)], fg="#1a2a4a")
    _interior(win_f, "desk", 3)
    _interior(win_s, "lockers", 4)
    for i, (k, reg) in enumerate(FR.items()):
        reg.fill(col="#45484b", rough=0.55)
        reg.mottle(0.06, 5, 40 + i, rough=0.05)
        if k == "+z":
            reg.field(lambda X, Y: noise(X.shape, 10, 41, 4) * 0.5, col="#6a6a62")
            reg.field(lambda X, Y: noise(X.shape, 4, 42, 4) ** 3 * 0.8, col="#2f3d22")
            for y in np.arange(0.9, reg.h, 1.0):
                reg.rect(0.05, y, reg.w - 0.05, y + 0.03, col="#3a3c3f", dh=0.0015, bevel=0.01)
                reg.line([(0.05, y + 0.03), (reg.w - 0.05, y + 0.03)], 0.003, col="#26282a")
            # ponding tide marks, moss in the corners, leaf litter, bird droppings
            for cx_, cy_, rr_ in ((reg.w * 0.6, reg.h * 0.4, 0.55), (reg.w * 0.3, reg.h * 0.62, 0.35)):
                def dd(X, Y, cx_=cx_, cy_=cy_, rr_=rr_):
                    return np.hypot((X - cx_) / 1.3, Y - cy_) / rr_ + (noise(X.shape, 6, int(cx_ * 10), 3) - 0.5) * 0.6
                reg.field(lambda X, Y, dd=dd: np.clip(1 - dd(X, Y), 0, 1) * 0.35, col="#2a2b2b", rough=0.3)
            for cx_, cy_ in ((0.12, 0.12), (reg.w - 0.12, 0.15), (0.15, reg.h - 0.12)):
                reg.field(lambda X, Y, cx_=cx_, cy_=cy_: np.clip(1 - np.hypot(X - cx_, Y - cy_) / 0.35 + (noise(X.shape, 12, 9, 4) - 0.5), 0, 1) * 0.8, col="#4a5a2a", rough=0.9)
            rng = np.random.default_rng(44)
            corners = [(0.15, 0.15), (reg.w - 0.15, 0.2), (0.2, reg.h - 0.15), (reg.w * 0.6, 0.12)]
            for _ in range(90):
                cx0_, cy0_ = corners[rng.integers(len(corners))]
                x, y = cx0_ + rng.normal(0, 0.12), cy0_ + rng.normal(0, 0.08)
                if not (0.03 < x < reg.w - 0.03 and 0.03 < y < reg.h - 0.03):
                    continue
                sc_ = rng.choice([0.6, 1.0, 1.5])
                reg.poly(_ellipse(x, y, 0.025 * sc_, 0.012 * sc_, rng.uniform(0, math.pi)),
                         col=str(rng.choice(["#4a3a22", "#6a5028", "#3e3020", "#7a5a30", "#5a5a2a"])), dh=0.0005)
            for _ in range(8):
                x, y = rng.uniform(0.2, reg.w - 0.2), rng.uniform(0.2, reg.h - 0.2)
                reg.circle(x, y, rng.uniform(0.008, 0.015), col="#e6e4dc", k=0.8)
        else:
            reg.streaks(None, 0.2, "#2a2c2d", 0.4, 43 + i)
    sign.fill(col="#1d2d4f", rough=0.35)
    sign.frame(0.01, 0.01, 1.39, 0.25, 0.01, r=0.012, col="#ffffff")
    sign.text("SECURITY", 0.7, 0.07, 0.12, weight="black", align="c", col="#ffffff", track=0.15)
    for x in (0.035, 1.365):
        for y in (0.035, 0.225):
            sign.circle(x, y, 0.008, col="#c9cbcc", dh=0.0015, metal=0.8)
    sign.streaks(None, 0.15, "#4a4a44", 0.25, 6)
    ac.fill(col="#e6e6e1", rough=0.45)
    ac.circle(0.31, 0.275, 0.21, col="#2a2c2e", dh=-0.004)
    for r_ in np.arange(0.03, 0.21, 0.022):
        ac.ring(0.31, 0.275, r_, r_ + 0.004, col="#cfcfca", dh=0.004)
    for k in range(-5, 6):
        ac.rect(0.31 + k * 0.035 - 0.002, 0.075, 0.31 + k * 0.035 + 0.002, 0.475, col="#cfcfca", dh=0.004)
    ac.text("INVERTER", 0.69, 0.48, 0.022, weight="bold", align="c", col="#4a4d50")
    for y in np.arange(0.06, 0.42, 0.025):
        ac.rect(0.58, y, 0.78, y + 0.008, col="#9a9b98", dh=-0.002)
    ac.grime(0.2, "#6c6556", 0.4, 7)
    for i, reg in enumerate(PD.values()):
        _concrete(reg, 140 + i)
    PD["+z"].field(lambda X, Y: np.clip(1 - np.minimum.reduce([X - 0.2, (L + 0.2) - X, Y - 0.2, (D + 0.2) - Y]) / 0.15, 0, 1) * (np.minimum.reduce([X - 0.2, (L + 0.2) - X, Y - 0.2, (D + 0.2) - Y]) > -0.01) * 0.5, col="#4a4434")
    grate.fill(col="#2a2a28", rough=0.6, metal=0.6)
    for k in range(9):
        grate.rect(0.02 + k * 0.019, 0.02, 0.03 + k * 0.019, 0.18, col="#060606", dh=-0.003)
    grate.field(lambda X, Y: noise(X.shape, 8, 141, 3) * 0.5, col="#6a4a2a")
    icom.fill(col="#5a5d60", rough=0.4, metal=0.5)
    for i in range(5):
        for j in range(6):
            icom.circle(0.035 + i * 0.012, 0.12 + j * 0.011, 0.0025, col="#111111", dh=-0.001)
    icom.circle(0.06, 0.06, 0.02, col="#c9cbcc", dh=0.002, metal=0.8)
    icom.circle(0.06, 0.06, 0.014, col="#2a6ad0", emit="#3a7ae0")
    icom.text("PRESS", 0.06, 0.025, 0.012, weight="bold", align="c", col="#ffffff", metal=0.0)
    mat = cv.material("Amen_Hut")
    inner = cv.material("Amen_HutInterior", rough=0.6, emission=0.8, normal=0)
    frame = _paint("Amen_HutFrame", "#3a3d41", 0.4)
    glass = clear_glass()

    p = PM(0.02)
    # concrete pad 200 mm beyond the walls, galvanised base skid with fork pockets and lifting eyes
    p.r = 0.015
    i0 = len(p.F)
    p.box(plo_, phi_, mat)
    box_uv(p, i0, PD, plo_, phi_)
    p.r = 0.006
    p.box((-L / 2 + 0.02, -D / 2 + 0.02, 0.06), (L / 2 - 0.02, D / 2 - 0.02, Z0), M["galv"])
    for x in (-0.8, 0.8):
        for sy in (-1, 1):
            p.r = 0.0
            decal(p, (x, sy * (D / 2 - 0.02), 0.13), (0, sy, 0), (0, 0, 1), 0.3, 0.08, M["rubber"], lift=0.002)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.r = 0.003
            c_ = np.array([sx * (L / 2 - 0.1), sy * (D / 2 - 0.02), 0.13])
            p.box(tuple(c_ - (0.04, 0.006, 0.03)), tuple(c_ + (0.04, 0.006, 0.03)), M["galv"])
            p.r = 0.0
            _tube(p, [c_ + (0.02 * math.cos(t), sy * 0.012, 0.02 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 9)], 0.005, M["galv"], 5, bend=0.0, closed=True)
    p.r = 0.025
    i0 = len(p.F)
    _loft(p, [rrect(L, D, 0.06, z) for z in (Z0, H)], mat, cap0=True, cap1=False)
    box_uv(p, i0, R, lo, hi)
    # roof: fascia with a drip edge, membrane top
    p.r = 0.012
    i0 = len(p.F)
    _loft(p, [rrect(L + 0.24, D + 0.24, 0.1, H), rrect(L + 0.24, D + 0.24, 0.1, H + 0.2)], mat, cap0=True, cap1=True)
    box_uv(p, i0, FR, flo, fhi)
    p.r = 0.003
    _loft(p, [rrect(L + 0.27, D + 0.27, 0.11, H + 0.2), rrect(L + 0.27, D + 0.27, 0.11, H + 0.225), rrect(L + 0.2, D + 0.2, 0.09, H + 0.225)], frame, cap0=False, cap1=False)
    # soil vent pipe with its cowl and a cable gland on the roof
    p.r = 0.003
    p.cyl((0.8, -0.6, H + 0.2), (0.8, -0.6, H + 0.5), 0.05, M["grey"], 14)
    p.lathe([(0.0, H + 0.58), (0.075, H + 0.53), (0.08, H + 0.52), (0.055, H + 0.5)], M["grey"], c=(0.8, -0.6, 0), sides=14)
    p.box((-0.9, 0.4, H + 0.2), (-0.7, 0.55, H + 0.27), M["grey"])
    # gutter along the back, downpipe at the back right corner
    p.r = 0.0
    p.cyl((-L / 2 - 0.14, -D / 2 - 0.19, H + 0.06), (L / 2 + 0.14, -D / 2 - 0.19, H + 0.06), 0.055, M["black"], 12)
    _tube(p, [(L / 2 - 0.05, -D / 2 - 0.19, H + 0.05), (L / 2 - 0.05, -D / 2 - 0.12, H - 0.15), (L / 2 - 0.05, -D / 2 - 0.05, H - 0.3),
            (L / 2 - 0.05, -D / 2 - 0.05, 0.3), (L / 2 - 0.05, -D / 2 - 0.16, 0.13)], 0.034, M["black"], 10, bend=0.08)
    p.cyl((L / 2 - 0.05, -D / 2 - 0.15, 0.135), (L / 2 - 0.05, -D / 2 - 0.19, 0.11), 0.036, M["black"], 10, r2=0.042)
    decal(p, (L / 2 - 0.05, -D / 2 - 0.12, 0.06), (0, 0, 1), (0, 1, 0), 0.18, 0.12, mat, grate, lift=0.002, uv=(0.01, 0.04, 0.19, 0.16))
    for z in (0.6, 1.2, 1.8, 2.35):
        p.box((L / 2 - 0.09, -D / 2 - 0.09, z - 0.015), (L / 2 - 0.01, -D / 2, z + 0.015), M["black"])
    # windows: front (right of the door), left side, and a small one on the right side
    _window(p, (-0.6, D / 2, 1.55), (0, 1, 0), 1.3, 1.05, frame, glass, inner, win_f)
    _window(p, (-L / 2, 0.0, 1.55), (-1, 0, 0), 1.3, 1.05, frame, glass, inner, win_s)
    _window(p, (L / 2, -0.0, 1.7), (1, 0, 0), 0.8, 0.75, frame, glass, inner, win_s, mullion=False)
    # the door: frame, leaf (its own texture), hinges, lever, step
    dx = 0.85
    p.r = 0.004
    decal(p, (dx, D / 2, Z0 + 0.04 + 1.025), (0, 1, 0), (0, 0, 1), 0.9, 2.05, mat, door, lift=0.012)
    for (x0, z0, x1, z1) in ((dx - 0.5, Z0, dx - 0.45, Z0 + 2.14), (dx + 0.45, Z0, dx + 0.5, Z0 + 2.14), (dx - 0.5, Z0 + 2.09, dx + 0.5, Z0 + 2.14)):
        p.box((x0, D / 2 - 0.002, z0), (x1, D / 2 + 0.04, z1), frame)
    for z in (0.45, 1.2, 1.95):
        p.cyl((dx + 0.445, D / 2 + 0.03, Z0 + z - 0.06), (dx + 0.445, D / 2 + 0.03, Z0 + z + 0.06), 0.012, M["chrome"], 10)
    p.r = 0.003
    p.cyl((dx - 0.36, D / 2 + 0.012, Z0 + 1.05), (dx - 0.36, D / 2 + 0.06, Z0 + 1.05), 0.012, M["chrome"], 10)
    p.cyl((dx - 0.36, D / 2 + 0.06, Z0 + 1.05), (dx - 0.22, D / 2 + 0.065, Z0 + 1.05), 0.01, M["chrome"], 10)
    p.box((dx - 0.385, D / 2 + 0.012, Z0 + 0.9), (dx - 0.335, D / 2 + 0.02, Z0 + 1.12), M["chrome"])
    p.r = 0.015
    p.box((dx - 0.6, D / 2, 0.06), (dx + 0.6, D / 2 + 0.35, Z0 - 0.005), M["concrete"])
    # door closer, intercom beside the door
    p.r = 0.004
    p.box((dx - 0.42, D / 2 + 0.01, Z0 + 2.0), (dx - 0.12, D / 2 + 0.06, Z0 + 2.06), M["grey"])
    p.box((dx - 0.15, D / 2 + 0.04, Z0 + 2.08), (dx + 0.15, D / 2 + 0.06, Z0 + 2.1), M["grey"])
    p.box((dx - 0.64, D / 2, Z0 + 1.15), (dx - 0.52, D / 2 + 0.035, Z0 + 1.35), M["grey"])
    decal(p, (dx - 0.58, D / 2 + 0.035, Z0 + 1.25), (0, 1, 0), (0, 0, 1), 0.12, 0.2, mat, icom, lift=0.0008)
    # sign on the fascia, bulkhead light, AC condenser, CCTV on the corner
    p.r = 0.004
    p.box((-1.3, D / 2, H - 0.4), (0.1, D / 2 + 0.015, H - 0.14), M["alu"])
    decal(p, (-0.6, D / 2 + 0.015, H - 0.27), (0, 1, 0), (0, 0, 1), 1.4, 0.26, mat, sign, lift=0.0008)
    p.r = 0.01
    p.box((dx - 0.15, D / 2, Z0 + 2.22), (dx + 0.15, D / 2 + 0.08, Z0 + 2.36), _paint("Amen_Bulkhead", "#2c2e30", 0.4))
    decal(p, (dx, D / 2 + 0.08, Z0 + 2.29), (0, 1, 0), (0, 0, 1), 0.24, 0.09, _emit("Amen_BulkheadLens", "#fff4dc", 0.4), lift=0.001)
    p.r = 0.01
    ax0, ax1 = (-0.7, -D / 2 - 0.36, 0.35), (0.1, -D / 2 - 0.06, 0.9)
    p.box(ax0, ax1, _paint("Amen_ACBody", "#e6e6e1", 0.45))
    decal(p, (-0.3, -D / 2 - 0.36, 0.625), (0, -1, 0), (0, 0, 1), 0.8, 0.55, mat, ac, lift=0.001)
    p.r = 0.004
    for x in (-0.6, 0.0):
        p.box((x - 0.02, -D / 2 - 0.4, 0.3), (x + 0.02, -D / 2, 0.35), M["galv"])
    p.r = 0.0
    _tube(p, [(0.05, -D / 2 - 0.2, 0.9), (0.05, -D / 2 - 0.2, 1.1), (0.05, -D / 2, 1.3)], 0.018, M["white"], 8, bend=0.08)
    p.r = 0.004
    cx_, cy_ = L / 2 + 0.03, D / 2 + 0.03
    p.box((cx_ - 0.03, cy_ - 0.03, H - 0.1), (cx_ + 0.03, cy_ + 0.03, H - 0.02), M["white"])
    p.cyl((cx_, cy_, H - 0.06), (cx_ + 0.12, cy_ + 0.12, H - 0.1), 0.035, M["white"], 14)
    p.cyl((cx_ + 0.12, cy_ + 0.12, H - 0.1), (cx_ + 0.125, cy_ + 0.125, H - 0.102), 0.03, M["glass"], 14)
    return p.obj("security_hut", col, 40)


def _fabric(reg, color, seed=1, weave=0.003, fade=0.2):
    """Woven polyester canvas: weave relief, faint slubs, sun fade towards the top (region y = h)."""
    reg.fill(col=color, rough=0.82)
    rs, cs, X, Y = reg.grid()
    wv = (np.sin(X / weave * math.pi) * np.sin(Y / weave * math.pi))
    reg.cv.ht[rs, cs] += 0.00012 * wv
    reg.cv.col[rs, cs] *= (1 + 0.035 * wv)[..., None]
    reg.mottle(0.05, 6, seed, rough=0.03)
    if fade:
        reg.field(lambda X, Y: np.clip(Y / reg.h, 0, 1) ** 2 * fade, col="#d9a7a0")


def _granite(reg, seed=1):
    reg.fill(col="#8d8b88", rough=0.6)
    rs, cs, X, Y = reg.grid()
    rng = np.random.default_rng(seed)
    r = rng.random(X.shape)
    cv = reg.cv
    cv.col[rs, cs] *= (0.75 + 0.5 * r)[..., None]
    blk = (rng.random(X.shape) > 0.93).astype(float)
    wht = (rng.random(X.shape) > 0.95).astype(float)
    cv.col[rs, cs] = cv.col[rs, cs] * (1 - 0.7 * blk[..., None]) + 0.5 * wht[..., None] * 0.4
    reg.mottle(0.08, 5, seed + 1)


def parasol(col, M):
    """A 2.7 m commercial parasol: canvas canopy in the club colour (TINT_club_parasol, recoloured
    per club) of eight panels sagging between the ribs and pulled up at the tips, sewn seam strips
    along the ribs, reinforced tip pockets with metal caps, a hemmed scalloped valance, a vented
    top tier and a darker underside; the crest printed on four panels (the weave shows through);
    eight ribs and stretchers to a runner and hub; a brushed-aluminium pole with a joint collar
    and crank winder; a 500 mm granite base with a socket boss and T-handle knob. Front +y."""
    R, ZA, ZR = 1.38, 2.5, 2.16                  # rim radius, apex and rim heights
    half = math.pi / 8
    chord = 2 * R * math.sin(half)
    # the canvas (greys: white = the club colour)
    tc = Canvas("Amen_ParasolTint", 1024, 1024, col="#ffffff", rough=0.8)
    panel = tc.alloc(chord, R, 520)
    under = tc.alloc(chord, R, 160)
    val = tc.alloc(chord, 0.2, 520)
    for reg, dark in ((panel, 1.0), (under, 0.62)):
        reg.fill(col="#ffffff")
        rs, cs, X, Y = reg.grid()
        wv = np.sin(X / 0.0025 * math.pi) * np.sin(Y / 0.0025 * math.pi)
        tc.ht[rs, cs] += 0.00012 * wv
        tc.col[rs, cs] *= (dark * (1 + 0.04 * wv))[..., None]
        reg.mottle(0.05, 6, 3)
        if dark == 1.0:
            # sun fade towards the crown, dust
            reg.field(lambda X, Y: np.clip(1 - Y / R, 0, 1) ** 1.5 * 0.6, mul=1.22)
            reg.field(lambda X, Y: noise(X.shape, 5, 10, 4) * 0.25, mul=0.7)
        # sewn seam strips along both ribs, double stitching, darker reinforced tip pockets, hem
        for sgn in (-1, 1):
            edge = lambda y, sgn=sgn: chord / 2 + sgn * (chord / 2) * (y / R)
            reg.field(lambda X, Y, sgn=sgn: (np.abs(X - (chord / 2 + sgn * chord / 2 * Y / R)) < 0.025) * 0.35, mul=0.62)
            for off in (0.006, 0.02):
                reg.line([(chord / 2, 0.0), (chord / 2 + sgn * (chord / 2 - off), R)], 0.0016, mul=0.55, dh=-0.0003)
            ex = edge(R)
            reg.field(lambda X, Y, ex=ex: np.clip(1 - np.hypot(X - ex, (Y - R) * 0.7) / 0.1, 0, 1) ** 0.7 * 0.6, mul=0.68)
        reg.rect(0, R - 0.035, chord, R, mul=0.82)
        reg.line([(0, R - 0.03), (chord, R - 0.03)], 0.0015, mul=1.3)
    val.fill(col="#ffffff")
    val.mottle(0.05, 6, 7)
    val.line([(0, 0.185), (chord, 0.185)], 0.0015, mul=1.3)
    val.line([(0, 0.02), (chord, 0.02)], 0.0015, mul=1.3)
    val.field(lambda X, Y: np.clip(1 - Y / 0.08, 0, 1) * 0.3, mul=0.6)
    fab = tc.tint_material("TINT_club_parasol", "#9c2a30", rough=0.82, normal=1.0, spec=0.3)
    fabu = fab
    # the crest print (transparent elsewhere), mapped rim-down so it reads upright from outside
    pc = Canvas("Amen_ParasolPrint", 1024, 1024, col="#f0ece4", rough=0.8, alpha=0.0)
    crestp = pc.alloc(chord, R, 720)
    _crest(crestp, chord / 2, R * 0.4, 0.34, trim="#f0ece4", dark="#3a1014", cut=True)
    crestp.text("ATHLETIC", chord / 2, R * 0.6, 0.055, weight="black", align="c", col="#f0ece4", a=1.0)
    rs, cs, X, Y = crestp.grid()
    wv = np.sin(X / 0.0025 * math.pi) * np.sin(Y / 0.0025 * math.pi)
    pc.ht[rs, cs] += 0.00012 * wv
    pc.col[rs, cs] *= (1 + 0.06 * wv)[..., None]
    crestp.field(lambda X, Y: noise(X.shape, 5, 10, 4) * 0.3, col="#a99a8c")
    bc = Canvas("Amen_ParasolBase", 512, 512, col="#8d8b88", rough=0.6)
    base = bc.alloc(0.5, 0.5, 600)
    bside = bc.alloc(2.0, 0.08, 220)
    _granite(base, 60)
    _granite(bside, 61)
    bside.field(lambda X, Y: np.clip(1 - Y / 0.025, 0, 1) * 0.6, mul=0.6)
    base.edge_wear(0, 0, 0.5, 0.5, 0.012, "#b4b0aa", 0.6, 62)
    prt = pc.material("Amen_ParasolPrint", rough=0.8, alpha=True, normal=1.0, spec=0.3)
    stone = bc.material("Amen_ParasolBase", rough=0.6, normal=0)
    alu = _paint("Amen_PoleAlu", "#c3c6c8", 0.35)

    p = PM(0.004)
    # granite base, socket boss, T-handle knob
    p.r = 0.012
    i0 = len(p.F)
    p.box((-0.25, -0.25, 0.0), (0.25, 0.25, 0.08), stone)
    for fi in range(i0, len(p.F)):
        P_ = np.array([p.V[i] for i in p.F[fi]])
        nz = abs(np.cross(P_[1] - P_[0], P_[2] - P_[0])[2]) > 1e-9
        if nz:
            p.uv[fi] = [base.uv(q[0] + 0.25, q[1] + 0.25) for q in P_]
        else:
            p.uv[fi] = [bside.uv((q[0] + q[1] + 1.0) * 0.999, q[2]) for q in P_]
    p.r = 0.004
    p.lathe([(0.075, 0.08), (0.075, 0.1), (0.05, 0.12), (0.05, 0.22), (0.035, 0.235), (0.026, 0.235)], M["steel"], sides=20)
    p.cyl((0.05, 0, 0.18), (0.11, 0, 0.18), 0.009, M["steel"], 8)
    p.cyl((0.11, -0.05, 0.18), (0.11, 0.05, 0.18), 0.011, M["black"], 10)
    # pole, joint collar, crank winder housing, runner, hub
    p.r = 0.0
    p.lathe([(0.0225, 0.1), (0.0225, ZA - 0.05)], alu, sides=16)
    p.r = 0.004
    p.lathe([(0.03, 1.15), (0.03, 1.22)], M["steel"], sides=16)
    p.box((-0.038, 0.0, 1.48), (0.038, 0.065, 1.68), M["steel"])
    p.r = 0.002
    p.cyl((0, 0.065, 1.58), (0, 0.08, 1.58), 0.014, M["steel"], 10)
    p.cyl((0, 0.08, 1.58), (0, 0.085, 1.47), 0.006, M["steel"], 8)
    p.cyl((0, 0.085, 1.47), (0, 0.14, 1.47), 0.012, M["black"], 10)
    p.r = 0.004
    zr = 1.92
    p.lathe([(0.03, zr - 0.06), (0.045, zr - 0.045), (0.045, zr + 0.04), (0.03, zr + 0.05)], M["steel"], sides=16)
    p.lathe([(0.03, ZA - 0.09), (0.05, ZA - 0.07), (0.05, ZA - 0.02), (0.02, ZA - 0.005)], M["steel"], sides=16)
    # ribs, stretchers, metal tip caps
    p.r = 0.0
    for k in range(8):
        a = 2 * math.pi * k / 8
        d = np.array([math.cos(a), math.sin(a), 0.0])
        h0 = np.array([0, 0, ZA - 0.06]) + d * 0.045
        tip = d * R * 0.995 + np.array([0, 0, ZR - 0.016])
        rect_r = [(-0.01, -0.005), (0.01, -0.005), (0.01, 0.005), (-0.01, 0.005)]
        rect_s = [(-0.0075, -0.004), (0.0075, -0.004), (0.0075, 0.004), (-0.0075, 0.004)]
        p.sweep([h0, tip], rect_r, alu, twist_ref=(0, 0, 1))
        p.cyl(tip, tip + d * 0.025 + (0, 0, -0.006), 0.009, M["chrome"], 8, r2=0.006)
        mid = h0 + (tip - h0) * 0.48 - np.array([0, 0, 0.012])
        p.sweep([np.array([0, 0, zr + 0.03]) + d * 0.045, mid], rect_s, alu, twist_ref=(0, 0, 1))
        p.box(tuple(mid - 0.012), tuple(mid + 0.012), M["steel"])
    # canopy: drawn tight between the ribs, sagging between them, pulled up at the tips
    nt, ns = 8, 6
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        G = []
        for i in range(nt + 1):
            t = i / nt
            row = []
            for j in range(ns + 1):
                sv = j / ns
                r_ = 0.04 + (R - 0.04) * t
                p0 = np.array([r_ * math.cos(a0), r_ * math.sin(a0)])
                p1 = np.array([r_ * math.cos(a1), r_ * math.sin(a1)])
                xy = p0 * (1 - sv) + p1 * sv
                w2 = 4 * sv * (1 - sv)
                z = ZA - (ZA - ZR) * t ** 1.15 - 0.075 * w2 * math.sin(math.pi * min(1.0, t * 1.05)) ** 0.7
                z += 0.012 * (1 - w2) * t ** 4
                # the hem lifts between the rib tips
                z += 0.05 * w2 * t ** 6
                if i == nt:
                    xy = xy * (1 - 0.02 * w2)
                row.append((np.array([xy[0], xy[1], z]), ((sv - 0.5) * chord * t + chord / 2, t * R)))
            G.append(row)
        for i in range(nt):
            for j in range(ns):
                quad = [G[i][j], G[i][j + 1], G[i + 1][j + 1], G[i + 1][j]]
                # triangles split on the same diagonal in every layer, so no layer pokes through another
                for q in (quad[:3], [quad[0], quad[2], quad[3]]):
                    P = [x[0] for x in q]
                    p.face(P, fab, [panel.uv(min(panel.w, max(0, x[1][0])), min(panel.h, x[1][1])) for x in q])
                    p.face([x - (0, 0, 0.005) for x in P][::-1], fabu, [under.uv(min(under.w, max(0, x[1][0])), min(under.h, x[1][1])) for x in q][::-1])
                    if k % 2 == 1:
                        p.face([x + (0, 0, 0.004) for x in P], prt, [crestp.uv(min(crestp.w, max(0, x[1][0])), R - x[1][1]) for x in q])
        nv = 24
        hem = [G[nt][0][0] + (G[nt][ns][0] - G[nt][0][0]) * (m_ / nv) for m_ in range(nv + 1)]
        for m_ in range(nv + 1):
            sv = m_ / nv
            w2 = 4 * sv * (1 - sv)
            hem[m_] = np.array([hem[m_][0] * (1 - 0.02 * w2), hem[m_][1] * (1 - 0.02 * w2), ZR + 0.012 * (1 - w2) + 0.05 * w2])

        def drop(sv):
            # two semicircular scallops per panel, 150 mm at the points, 105 mm at their crowns
            u_ = (sv * 2) % 1.0
            return 0.105 + 0.045 * (1 - math.sqrt(max(0.0, 1 - (2 * u_ - 1) ** 2)))
        for m_ in range(nv):
            s0, s1 = m_ / nv, (m_ + 1) / nv
            o0, o1 = hem[m_] * np.array([1.003, 1.003, 1]), hem[m_ + 1] * np.array([1.003, 1.003, 1])
            d0, d1 = drop(s0), drop(s1)
            p.face([o0, o1, o1 - (0, 0, d1), o0 - (0, 0, d0)], fab,
                   [val.uv(s0 * chord, 0.2), val.uv(s1 * chord, 0.2), val.uv(s1 * chord, 0.2 - d1), val.uv(s0 * chord, 0.2 - d0)])
    # vented top tier with its own short valance
    rt, zt = 0.3, ZA + 0.07
    for k in range(8):
        a0, a1 = 2 * math.pi * (k + 0.5) / 8, 2 * math.pi * (k + 1.5) / 8
        e0 = np.array([rt * math.cos(a0), rt * math.sin(a0), zt - 0.06])
        e1 = np.array([rt * math.cos(a1), rt * math.sin(a1), zt - 0.06])
        top = np.array([0, 0, zt + 0.04])
        p.face([top, e0, e1], fab, [panel.uv(chord / 2, 0.02), panel.uv(chord * 0.3, 0.3), panel.uv(chord * 0.7, 0.3)])
        p.face([e0, e1, e1 - (0, 0, 0.06), e0 - (0, 0, 0.06)], fab, [val.uv(0, 0.2), val.uv(chord * 0.3, 0.2), val.uv(chord * 0.3, 0.14), val.uv(0, 0.14)])
    p.r = 0.003
    p.cyl((0, 0, ZA - 0.02), (0, 0, zt - 0.02), 0.012, M["steel"], 8)
    p.sphere((0, 0, zt + 0.06), 0.022, M["steel"], 10, 6)
    return p.obj("parasol", col, 50)


def flagpole(col, M):
    """A 7 m tapered GRP flagpole (120 to 65 mm) on a hinged cast root collar and a bevelled
    concrete pad, a revolving truck with its sheave under a gilded ball finial, a two-part halyard
    down to a cast horn cleat at 1.2 m (wrapped round it), snap hooks on the flag's header; the
    1.8 x 1.2 m club flag in a gentle travelling wave: the field is TINT_flag_club (recoloured
    per club), the white chevron, crest and name printed on both faces so they read the right way
    from either side, a canvas header tape and a hemmed fly. Front +y."""
    H = 7.0
    fw, fh = 1.8, 1.2
    # the field: greys (white = the flag colour): weave, fade towards the fly, dirt on the fly
    tc = Canvas("Amen_FlagField", 1024, 512, col="#ffffff", rough=0.75)
    fld = tc.alloc(fw, fh, 420)
    fld.fill(col="#ffffff")
    rs, cs, X, Y = fld.grid()
    wv = np.sin(X / 0.0025 * math.pi) * np.sin(Y / 0.0025 * math.pi)
    tc.ht[rs, cs] += 0.0001 * wv
    tc.col[rs, cs] *= (1 + 0.04 * wv)[..., None]
    fld.mottle(0.05, 6, 5)
    fld.field(lambda X, Y: np.clip((X - 1.0) / 0.8, 0, 1) * 0.5, mul=1.18)
    fld.field(lambda X, Y: np.clip((X - 1.45) / 0.35, 0, 1) * noise(X.shape, 8, 9, 4) * 0.6, mul=0.6)
    fld.rect(1.77, 0.0, 1.8, 1.2, mul=0.8)
    for x in (1.765, 1.775):
        fld.line([(x, 0.0), (x, 1.2)], 0.0012, mul=0.6)
    field = tc.tint_material("TINT_flag_club", "#9c1426", rough=0.75, normal=1.0, spec=0.3)
    # the print, front and back (the back laid out mirrored so it reads correctly from behind)
    pc = Canvas("Amen_FlagPrint", 1024, 1024, col="#f3f1ea", rough=0.75, alpha=0.0)
    regs = (pc.alloc(fw, fh, 280), pc.alloc(fw, fh, 560))

    def art(r, mirror):
        mx = (lambda x: fw - x) if mirror else (lambda x: x)
        A = {"a": 1.0}
        r.poly([(mx(x), y) for x, y in [(0.06, 1.2), (0.42, 1.2), (1.8, 0.6), (0.42, 0.0), (0.06, 0.0), (1.44, 0.6)]], col="#f3f1ea", **A)
        r.poly([(mx(x), y) for x, y in [(0.14, 1.2), (0.3, 1.2), (1.62, 0.6), (0.3, 0.0), (0.14, 0.0), (1.47, 0.6)]], a=0.0)
        r.poly([(mx(x), y) for x, y in [(1.55, 0.6), (1.8, 0.71), (1.8, 0.49)]], col="#13233f", **A)
        _crest(r, mx(0.74), 0.68, 0.46, trim="#f3f1ea", dark="#13233f", cut=True)
        x0, x1 = (fw - 0.05, fw) if mirror else (0.0, 0.05)
        r.rect(x0, 0.0, x1, 1.2, col="#dcd6c8", **A)
        r.line([(mx(0.048), 0.0), (mx(0.048), 1.2)], 0.002, col="#8f8a7c", **A)
        for y in (0.035, 1.165):
            r.circle(mx(0.025), y, 0.012, col="#b8963c", **A)
            r.circle(mx(0.025), y, 0.006, col="#2a2a2a", **A)
        rs, cs, X, Y = r.grid()
        wv = np.sin(X / 0.0025 * math.pi) * np.sin(Y / 0.0025 * math.pi)
        pc.ht[rs, cs] += 0.0001 * wv
        r.field(lambda X, Y: np.clip((mx(X) - 1.45) / 0.35, 0, 1) * noise(X.shape, 8, 9, 4) * 0.5, col="#6a5e52")
    art(regs[0], False)
    art(regs[1], True)
    prt = pc.material("Amen_FlagPrint", rough=0.75, alpha=True, normal=1.0, spec=0.3)
    # pole and pad
    cv = Canvas("Amen_FlagPole", 512, 512, col="#ecece6", rough=0.32)
    pole = cv.alloc(0.38, 7.0, 60)
    pad = cv.alloc(0.6, 0.6, 250)
    pole.fill(col="#ecece6", rough=0.32)
    pole.mottle(0.03, 4, 12)
    pole.grime(0.5, "#5e6a4a", 0.6, 13)
    pole.streaks(None, 3.0, "#a7a79a", 0.25, 14, density=60)
    _concrete(pad, 15)
    pad.ring(0.3, 0.3, 0.11, 0.14, mul=0.7)
    pad.field(lambda X, Y: noise(X.shape, 8, 16, 4) * 0.4, col="#5a5446")
    mat = cv.material("Amen_FlagPole")
    gold = _paint("Amen_Gilt", "#c9a043", 0.3)
    rope = _plastic("Amen_Rope", "#e9e6dc", 0.7)

    p = PM(0.004)
    # a round concrete surround flush with the turf and a 220 mm cast aluminium base collar
    p.r = 0.01
    i0 = len(p.F)
    p.lathe([(0.0, -0.05), (0.28, -0.05), (0.28, 0.025), (0.26, 0.035), (0.0, 0.035)], mat, sides=24)
    project(p, i0, pad, (-0.3, -0.3, 0), (1, 0, 0), (0, 1, 0))
    p.r = 0.004
    p.lathe([(0.12, 0.035), (0.12, 0.06), (0.105, 0.07), (0.095, 0.2), (0.09, 0.24), (0.075, 0.255), (0.07, 0.255)], M["alu"], sides=24)
    for k in range(4):
        a_ = math.pi / 4 + k * math.pi / 2
        _hexbolt(p, (0.112 * math.cos(a_), 0.112 * math.sin(a_), 0.06), 0.01, 0.009, M["galv"])
    for sx in (-1, 1):
        _hexbolt(p, (sx * 0.098, 0, 0.17), 0.016, 0.011, M["galv"], axis=(sx, 0, 0))
    p.r = 0.0

    def rad(z):
        return 0.065 + (0.03 - 0.065) * z / H
    i0 = len(p.F)
    p.lathe([(rad(z), z) for z in np.linspace(0.24, H, 9)], mat, sides=24)
    wrap_uv(p, i0, pole, z0=0.0)
    # truck with its sheave, gilt finial
    p.r = 0.004
    p.lathe([(rad(H), H), (0.06, H + 0.005), (0.06, H + 0.08), (0.04, H + 0.09), (0.022, H + 0.1)], M["alu"], sides=24)
    p.r = 0.002
    p.cyl((0.058, -0.008, H + 0.035), (0.058, 0.008, H + 0.035), 0.03, M["black"], 16)
    p.cyl((0.04, -0.012, H + 0.035), (0.08, -0.012, H + 0.035), 0.004, M["chrome"], 6)
    p.r = 0.0
    p.sphere((0, 0, H + 0.16), 0.065, gold, 18, 10)
    p.cyl((0, 0, H + 0.09), (0, 0, H + 0.1), 0.015, gold, 10)
    # the flag: a gentle travelling wave (about 4 % of its length), growing towards the fly
    x0, ztop = 0.11, H - 0.1
    nx, nz = 24, 10
    Pg = np.zeros((nx + 1, nz + 1, 3))
    for i in range(nx + 1):
        u = i / nx
        for j in range(nz + 1):
            v = j / nz
            amp = 0.15 * u ** 1.2
            ph = 2 * math.pi * (2.4 * u) - 1.3 * v
            y = amp * math.sin(ph) + 0.025 * u * math.sin(2 * math.pi * 4.3 * u + 2.1 * v + 0.7)
            x = x0 + fw * u * (1 - 0.07 * u)
            z = ztop - fh * (1 - v) - 0.16 * u ** 1.6 * (1 - 0.4 * v)
            Pg[i, j] = (x, y, z)
    Nr = np.zeros_like(Pg)
    for i in range(nx + 1):
        for j in range(nz + 1):
            du = Pg[min(i + 1, nx), j] - Pg[max(i - 1, 0), j]
            dv = Pg[i, min(j + 1, nz)] - Pg[i, max(j - 1, 0)]
            n_ = np.cross(dv, du)
            n_ /= np.linalg.norm(n_) + 1e-12
            Nr[i, j] = n_ if n_[1] > 0 else -n_
    for i in range(nx):
        for j in range(nz):
            quad = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            for idx in (quad[:3], [quad[0], quad[2], quad[3]]):
                uvs = [(a_ / nx * fw, b_ / nz * fh) for a_, b_ in idx]
                P = [Pg[a_, b_] for a_, b_ in idx]
                Nn = [Nr[a_, b_] for a_, b_ in idx]
                p.face(P, field, [fld.uv(x, y) for x, y in uvs])
                # seen from +y the fly is on the viewer's left, so that face carries the mirrored layout
                p.face([q + n_ * 0.004 for q, n_ in zip(P, Nn)], prt, [regs[1].uv(fw - x, y) for x, y in uvs])
                p.face([q - n_ * 0.004 for q, n_ in zip(P, Nn)][::-1], prt, [regs[0].uv(x, y) for x, y in uvs][::-1])
    # halyard: the flag's line with a snap hook at each end of the header, and the free return
    hx = 0.085
    p.r = 0.0
    zc = 1.2
    top_hook, bot_hook = ztop + 0.03, ztop - fh - 0.03
    _tube(p, [(0.058, 0.0, H + 0.065), (hx, 0.0, H + 0.03), (hx, 0.0, top_hook + 0.03)], 0.005, rope, 6, bend=0.02)
    _tube(p, [(hx, 0.0, bot_hook - 0.03), (0.085, 0.004, 3.5), (rad(zc) + 0.03, 0.006, zc + 0.11)], 0.005, rope, 6, bend=0.05)
    _tube(p, [(0.058, 0.0, H + 0.005), (0.075, -0.02, H - 1.5), (0.085, -0.03, 4.0), (0.075, -0.024, 2.5), (rad(zc) + 0.03, -0.01, zc + 0.11)],
          0.005, rope, 6, bend=0.3)
    for z in (top_hook, bot_hook):
        p.r = 0.0
        hook = [(hx + 0.012 * math.sin(t), 0.0, z + 0.03 * math.cos(t) * 0.5) for t in np.linspace(0, 2 * math.pi, 10)]
        _tube(p, hook, 0.0028, M["chrome"], 5, bend=0.0, closed=True)
        p.r = 0.002
        p.cyl((hx, 0, z - 0.02), (x0 + 0.01, 0, z - (0.03 if z > 3 else -0.03)), 0.003, M["chrome"], 6)
    # cast horn cleat and the wraps round it
    p.r = 0.004
    cr = rad(zc)
    cleat_m = _paint("Amen_Cleat", "#3a3c3e", 0.45)
    p.box((cr - 0.004, -0.02, zc - 0.04), (cr + 0.028, 0.02, zc + 0.04), cleat_m)
    p.r = 0.0
    p.cyl((cr + 0.028, 0, zc - 0.1), (cr + 0.028, 0, zc + 0.1), 0.014, cleat_m, 10)
    p.cyl((cr + 0.028, 0, zc + 0.1), (cr + 0.028, 0, zc + 0.115), 0.014, cleat_m, 10, r2=0.007)
    p.cyl((cr + 0.028, 0, zc - 0.1), (cr + 0.028, 0, zc - 0.115), 0.014, cleat_m, 10, r2=0.007)
    for k in range(5):
        zz = zc - 0.06 + k * 0.03
        loop = [(cr + 0.028 + 0.022 * math.cos(t), 0.022 * math.sin(t), zz + 0.04 * math.sin(2 * t) * 0.5) for t in np.linspace(0, 2 * math.pi, 14)]
        _tube(p, loop, 0.005, rope, 5, bend=0.0, closed=True)
    _tube(p, [(cr + 0.05, 0.0, zc - 0.1), (cr + 0.08, 0.03, zc - 0.4), (cr + 0.06, 0.02, zc - 0.62)], 0.005, rope, 5, bend=0.05)
    return p.obj("flagpole", col, 60)


def satellite_dish(col, M):
    """A 1.2 m offset satellite dish on a non-penetrating roof mount: a galvanised 41 mm strut
    frame (slotted) with corner trays of stacked 600 mm paving slabs (chipped, algae-stained,
    slightly out of line) on rubber pads; a 76 mm mast with four braces and nuts at their ends;
    an az/el head with U-bolts, an elevation plate with its degree scale and an elevation screw;
    a deep parabolic offset reflector (pressing rings, grime at its foot and rim, rolled rim, back
    ribs and plate, sticker); a 40 mm square feed arm with a clamp collar and a moulded LNB holder
    carrying the LNB and ringed feedhorn; the coax tied to the arm, a drip loop, down the mast.
    Front +y (the way it points)."""
    cv = Canvas("Amen_Dish", 1024, 1024, col="#b4b8ba", rough=0.5)
    dfront = cv.alloc(1.24, 1.34, 300)
    dback = cv.alloc(1.24, 1.34, 200)
    galv = cv.alloc(1.0, 1.0, 200)
    slab = cv.alloc(0.6, 0.66, 300)
    mat_r = cv.alloc(0.5, 0.5, 100)
    scale = cv.alloc(0.12, 0.12, 600)
    for reg, seed in ((dfront, 1), (dback, 2)):
        reg.fill(col="#d0d0ca", rough=0.55)
        reg.mottle(0.04, 5, seed, rough=0.05)
        rng_ = np.random.default_rng(seed + 40)
        for _ in range(3):
            bx_, by_ = rng_.uniform(0.3, 0.95), rng_.uniform(0.25, 1.1)
            reg.poly(_ellipse(bx_, by_, rng_.uniform(0.012, 0.02), rng_.uniform(0.008, 0.014), rng_.uniform(0, math.pi), 9), col="#f2f1ea")
            reg.line([(bx_, by_), (bx_ + rng_.normal(0, 0.003), by_ - rng_.uniform(0.03, 0.08))], 0.004, col="#e4e2d6", k=0.6)
        rs, cs, X, Y = reg.grid()
        rad_ = np.hypot((X - 0.62) / 0.62, (Y - 0.67) / 0.67)
        cv.ht[rs, cs] += 0.0006 * np.clip(np.sin(rad_ * 2 * math.pi * 3.5), -0.2, 1) * (rad_ < 0.97)
        reg.field(lambda X, Y: np.clip(1 - Y / 0.35, 0, 1) * 0.5 * noise(X.shape, 8, seed + 5, 4), col="#7c786c")
        reg.field(lambda X, Y: np.clip(np.hypot((X - 0.62) / 0.62, (Y - 0.67) / 0.67) - 0.85, 0, 1) * 4 * noise(X.shape, 10, seed, 3), col="#6c685c")
        reg.streaks(0.3, 0.25, "#8d8a80", 0.3, seed + 3, density=30)
        reg.field(lambda X, Y: np.clip(1 - np.hypot(X - 0.9, Y - 0.35) / 0.08, 0, 1) * noise(X.shape, 20, 31, 3), col="#9aa06a", k=0.8)
    dfront.text("SKYLINK", 0.62, 0.11, 0.045, weight="black", align="c", col="#4d5a6c")
    dfront.field(lambda X, Y: noise(X.shape, 30, 32, 3) * 0.4, 0.5, 0.1, 0.74, 0.16, col="#dcdcd6")
    _sticker(dback, 0.45, 0.4, 0.79, 0.52, "#f4f4f0", [("SKYLINK 120 OFFSET", 0.022), ("Ku-band · 1.2 m · CE", 0.016)], fg="#1a2a4a", wear=0.5, seed=4)
    _galv(galv, 7, 0.0, streaks=False)
    galv.field(lambda X, Y: noise(X.shape, 6, 9, 4) * 0.25, col="#7d6a52")
    # strut slots along the members (the frame maps through this region)
    for i in range(10):
        for j in range(20):
            galv.rect(0.03 + i * 0.1, 0.02 + j * 0.05, 0.06 + i * 0.1, 0.03 + j * 0.05, r=0.004, col="#3a3c3c", dh=-0.002)
    # paving slab: a top (600 x 600) and an edge strip (50 mm), chipped, algae, staining
    _concrete(slab, 11, col="#9a968c")
    slab.field(lambda X, Y: noise(X.shape, 10, 12, 4) * 0.5, col="#5a5850")
    slab.field(lambda X, Y: np.clip((noise(X.shape, 6, 13, 4) - 0.55) * 4, 0, 1) * 0.6, col="#5a6a3a")
    slab.edge_wear(0, 0, 0.6, 0.6, 0.015, "#c2beb4", 0.8, 14)
    mat_r.fill(col="#1d1d1d", rough=0.9)
    mat_r.mottle(0.2, 8, 13)
    scale.fill(col="#c8ccce", rough=0.35, metal=0.8)
    for k in range(10):
        t = math.radians(200 + 14 * k)
        L_ = 0.012 if k % 5 == 0 else 0.007
        scale.line([(0.06 + 0.045 * math.cos(t), 0.08 + 0.045 * math.sin(t)), (0.06 + (0.045 - L_) * math.cos(t), 0.08 + (0.045 - L_) * math.sin(t))], 0.0012, col="#111111", metal=0.0)
    scale.text("ELEV", 0.06, 0.08, 0.008, weight="bold", align="c", col="#111111", metal=0.0)
    mat = cv.material("Amen_Dish")

    p = PM(0.004)
    S = 0.9
    # rubber pads under the frame's feet
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.r = 0.004
            i0 = len(p.F)
            p.box((sx * S - 0.12, sy * S - 0.12, 0.0), (sx * S + 0.12, sy * S + 0.12, 0.02), mat)
            project(p, i0, mat_r, (sx * S - 0.25, sy * S - 0.25, 0), (1, 0, 0), (0, 1, 0))
    # the strut frame: perimeter, cross members, corner trays
    p.r = 0.002
    i0 = len(p.F)
    z0 = 0.02
    for sgn in (-1, 1):
        p.box((-S, sgn * S - 0.0205, z0), (S, sgn * S + 0.0205, z0 + 0.041), mat)
        p.box((sgn * S - 0.0205, -S, z0), (sgn * S + 0.0205, S, z0 + 0.041), mat)
    p.box((-S, -0.0205, z0), (S, 0.0205, z0 + 0.041), mat)
    p.box((-0.0205, -S, z0), (0.0205, S, z0 + 0.041), mat)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.box((sx * 0.5 - 0.31, sy * 0.5 - 0.31, z0 + 0.041), (sx * 0.5 + 0.31, sy * 0.5 + 0.31, z0 + 0.047), mat)
    project(p, i0, galv, (-1.0, -1.0, 0), (0.5, 0, 0), (0, 0.5, 0.4))
    # paving slabs stacked three high, a little out of line
    rng = np.random.default_rng(5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for lay in range(3):
                zb = z0 + 0.047 + lay * 0.05
                dx_, dy_, rot_ = rng.normal(0, 0.012), rng.normal(0, 0.012), rng.normal(0, 0.03)
                c_ = np.array([sx * 0.5 + dx_, sy * 0.5 + dy_])
                ca, sa = math.cos(rot_), math.sin(rot_)
                ring = [(c_[0] + ca * x - sa * y, c_[1] + sa * x + ca * y) for x, y, _ in rrect(0.6, 0.6, 0.008, 0.0, n=2)]
                p.r = 0.006
                i0 = len(p.F)
                _loft(p, [[(x, y, zb) for x, y in ring], [(x, y, zb + 0.05) for x, y in ring]], mat)
                for fi in range(i0, len(p.F)):
                    P_ = np.array([p.V[i] for i in p.F[fi]])
                    n_ = np.cross(P_[1] - P_[0], P_[2] - P_[0])
                    if abs(n_[2]) > 0.7 * np.linalg.norm(n_):
                        p.uv[fi] = [slab.uv(min(0.6, max(0, q[0] - c_[0] + 0.3)), min(0.6, max(0, q[1] - c_[1] + 0.3))) for q in P_]
                    else:
                        p.uv[fi] = [slab.uv(min(0.6, max(0, (q[0] + q[1] - c_[0] - c_[1]) * 0.5 + 0.3)), 0.6 + min(0.05, max(0, q[2] - zb))) for q in P_]
    # mast with base plate, four braces with nuts at both ends
    mx, my = 0.0, -0.12
    p.r = 0.003
    p.box((mx - 0.12, my - 0.12, z0 + 0.041), (mx + 0.12, my + 0.12, z0 + 0.053), M["galv"])
    for x in (-0.09, 0.09):
        for y in (-0.09, 0.09):
            _hexbolt(p, (mx + x, my + y, z0 + 0.053), 0.009, 0.008, M["galv"])
    p.r = 0.0
    i0 = len(p.F)
    p.lathe([(0.038, z0 + 0.05), (0.038, 1.16)], mat, c=(mx, my, 0), sides=16)
    wrap_uv(p, i0, galv, z0=0.0, centre=(mx, my), ys=0.8)
    p.sphere((mx, my, 1.16), 0.038, M["galv"], 12, 4, z_min=0.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            a0 = np.array([mx + sx * 0.03, my + sy * 0.03, 0.62])
            b0 = np.array([sx * (S - 0.05), sy * (S - 0.05), z0 + 0.041])
            p.cyl(a0, b0, 0.014, M["galv"], 6)
            _hexbolt(p, b0, 0.009, 0.01, M["galv"])
            p.r = 0.002
            p.box(tuple(a0 - 0.025), tuple(a0 + 0.025), M["galv"])
            p.r = 0.0
    # the reflector: an offset cut of a paraboloid (f = 0.55), 1.2 x 1.3 m
    f, ra, rb, yc = 0.55, 0.6, 0.65, 0.72
    el = math.radians(26)
    bdir = np.array([0, math.cos(el), math.sin(el)])
    udir = np.array([0, -math.sin(el), math.cos(el)])
    xdir = np.array([-1.0, 0, 0])
    zc = yc * yc / (4 * f)
    O = np.array([0.0, 0.1, 1.32])

    def T(x, y, z):
        return O + xdir * x + udir * (y - yc) + bdir * (z - zc)
    nr, nt = 8, 40
    pts = {}
    for i in range(nr + 1):
        rho = i / nr
        for j in range(nt):
            th = 2 * math.pi * j / nt
            x, y = ra * rho * math.cos(th), yc + rb * rho * math.sin(th)
            pts[i, j] = (x, y, (x * x + y * y) / (4 * f))
    p.r = 0.0
    for i in range(nr):
        for j in range(nt):
            j2 = (j + 1) % nt
            q = [pts[i, j], pts[i + 1, j2], pts[i + 1, j]] if i == 0 else [pts[i, j], pts[i, j2], pts[i + 1, j2], pts[i + 1, j]]
            fuv = [dfront.uv(0.62 + x, 0.67 + (y - yc)) for x, y, z in q]
            buv = [dback.uv(0.62 - x, 0.67 + (y - yc)) for x, y, z in q]
            p.face([T(*v) for v in q], mat, fuv)
            p.face([T(*v) - bdir * 0.004 for v in q][::-1], mat, buv[::-1])
    rim = [T(*pts[nr, j]) - bdir * 0.002 for j in range(nt)]
    _tube(p, rim + [rim[0]], 0.006, _paint("Amen_DishRim", "#c8c8c2", 0.55), 6, bend=0.0, closed=True)
    # back plate and ribs
    nw = -udir * (yc / (2 * f)) + bdir
    nw /= np.linalg.norm(nw)
    cb = T(0, yc, zc) - nw * 0.035
    p.r = 0.004
    p.obox(cb, (0.26, 0.03, 0.26), M["galv"], pitch=math.atan2(nw[2], nw[1]))
    for k in range(6):
        th = 2 * math.pi * k / 6 + math.pi / 6
        x0_, y0_ = 0.12 * math.cos(th), yc + 0.12 * math.sin(th)
        a_ = T(x0_, y0_, (x0_ * x0_ + y0_ * y0_) / (4 * f)) - bdir * 0.03
        x_, y_ = 0.5 * math.cos(th), yc + 0.55 * math.sin(th)
        b_ = T(x_, y_, (x_ * x_ + y_ * y_) / (4 * f)) - bdir * 0.03
        mid_ = T((x0_ + x_) / 2, (y0_ + y_) / 2, (((x0_ + x_) / 2) ** 2 + ((y0_ + y_) / 2) ** 2) / (4 * f)) - bdir * 0.03
        _tube(p, [a_, mid_, b_], 0.011, M["white"], 4, bend=0.0)
    # az/el head: clamp with U-bolts on the mast, elevation plate with its scale, elevation screw
    p.r = 0.004
    p.box((mx - 0.06, my - 0.06, 1.0), (mx + 0.06, my + 0.06, 1.14), M["galv"])
    p.cyl((mx, my + 0.06, 1.1), cb - nw * 0.02, 0.022, M["galv"], 8)
    p.cyl((mx, my + 0.06, 1.02), T(0, yc - 0.11, (yc - 0.11) ** 2 / (4 * f)) - bdir * 0.05, 0.008, M["chrome"], 6)
    p.r = 0.002
    p.box((mx + 0.062, my - 0.05, 1.02), (mx + 0.068, my + 0.07, 1.14), M["galv"])
    decal(p, (mx + 0.068, my + 0.01, 1.08), (1, 0, 0), (0, 0, 1), 0.12, 0.12, mat, scale, lift=0.0008)
    p.r = 0.0
    for z in (1.02, 1.12):
        ub = [(mx - 0.05, my - 0.06, z), (mx - 0.05, my - 0.045, z)] + \
             [(mx + 0.046 * math.cos(t), my + 0.046 * math.sin(t) - 0.0, z) for t in np.linspace(math.pi, 2 * math.pi, 7)] + [(mx + 0.05, my - 0.045, z), (mx + 0.05, my - 0.06, z)]
        _tube(p, ub, 0.005, M["chrome"], 6, bend=0.0)
        for sx in (-1, 1):
            _hexbolt(p, (mx + sx * 0.05, my - 0.06, z), 0.009, 0.008, M["galv"], axis=(0, -1, 0))
    # 40 mm square feed arm from the lower rim to a moulded LNB holder at the focus
    lowr = T(0, yc - rb, (yc - rb) ** 2 / (4 * f)) - bdir * 0.01
    focus = T(0, 0, f)
    d = focus - lowr
    L = np.linalg.norm(d)
    dn = d / L
    p.r = 0.003
    sq = [(-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02)]
    p.sweep([lowr, lowr + dn * (L - 0.09)], sq, M["galv"], twist_ref=(1, 0, 0))
    p.box(tuple(lowr - 0.035), tuple(lowr + 0.035), M["galv"])
    # feed-arm support brace from the arm's middle back to the dish's lower edge
    p.r = 0.0
    yb_ = yc - rb * 0.55
    p.cyl(lowr + d * 0.45, T(0.28, yb_, (0.28 ** 2 + yb_ ** 2) / (4 * f)) - bdir * 0.012, 0.006, M["galv"], 6)
    p.cyl(lowr + d * 0.45, T(-0.28, yb_, (0.28 ** 2 + yb_ ** 2) / (4 * f)) - bdir * 0.012, 0.006, M["galv"], 6)
    p.r = 0.003
    aim = T(0, yc, zc) - focus
    aim /= np.linalg.norm(aim)
    p.r = 0.006
    hb = focus - aim * 0.085
    p.box(tuple(hb - (0.04, 0.035, 0.035)), tuple(hb + (0.04, 0.035, 0.035)), M["black"])
    p.r = 0.003
    p.cyl(focus - aim * 0.13, focus - aim * 0.02, 0.022, M["white"], 14)
    p.cyl(focus - aim * 0.02, focus + aim * 0.03, 0.022, M["black"], 14, r2=0.03)
    for k in range(3):
        c_ = focus + aim * (0.01 + 0.012 * k)
        p.cyl(c_, c_ + aim * 0.006, 0.045 - 0.006 * k, M["black"], 16)
    # coax: LNB along the arm (three ties), a drip loop, down the mast and off over the frame
    p.r = 0.0
    c0 = focus - aim * 0.13
    up_ = np.cross(dn, np.array([1.0, 0, 0]))
    up_ = up_ if up_[2] > 0 else -up_
    cab = [c0, c0 - aim * 0.03 - up_ * 0.03] + [lowr + d * t - up_ * 0.028 for t in (0.7, 0.35, 0.05)] + \
          [lowr - udir * 0.12 - bdir * 0.05, (mx + 0.05, my + 0.07, 0.95), (mx + 0.045, my + 0.05, 1.0), (mx + 0.045, my + 0.04, 0.9),
           (mx + 0.045, my + 0.04, 0.2), (mx + 0.06, my + 0.04, 0.075), (0.6, my + 0.04, 0.075)]
    _tube(p, _spline(cab, 5), 0.0045, M["rubber"], 6, bend=0.0)
    for t in (0.25, 0.5, 0.75):
        c_ = lowr + d * t - up_ * 0.014
        ring = [c_ + (np.cross(dn, up_) * math.cos(a_) + up_ * math.sin(a_)) * 0.026 for a_ in np.linspace(0, 2 * math.pi, 9)]
        _tube(p, ring, 0.0022, M["black"], 4, bend=0.0, closed=True)
    for z in (0.4, 0.7):
        p.lathe([(0.043, z - 0.005), (0.043, z + 0.005)], M["black"], c=(mx, my, 0), sides=10)
    return p.obj("satellite_dish", col, 45)


def _diamond(reg, x, y, s, sym):
    """A CLP hazard pictogram: red-bordered white diamond (centre x, y; s corner to corner)."""
    h = s / 2
    reg.poly([(x, y + h), (x + h, y), (x, y - h), (x - h, y)], col="#d21f1f", dh=0.0002)
    k = 0.82
    reg.poly([(x, y + h * k), (x + h * k, y), (x, y - h * k), (x - h * k, y)], col="#ffffff")
    if sym == "flame":
        reg.rect(x - h * 0.4, y - h * 0.42, x + h * 0.4, y - h * 0.34, col="#111111")
        reg.poly([(x - h * 0.3, y - h * 0.3), (x + h * 0.3, y - h * 0.3), (x + h * 0.25, y + h * 0.05), (x + h * 0.08, y + h * 0.42),
                  (x + h * 0.05, y + h * 0.12), (x - h * 0.1, y + h * 0.3), (x - h * 0.28, y + h * 0.02)], col="#111111")
    else:
        reg.line([(x - h * 0.45, y - h * 0.3), (x - h * 0.05, y - h * 0.3)], h * 0.05, col="#111111")
        reg.line([(x - h * 0.3, y - h * 0.3), (x - h * 0.28, y + h * 0.3), (x - h * 0.1, y + h * 0.45)], h * 0.05, col="#111111")
        reg.line([(x - h * 0.28, y + h * 0.1), (x - h * 0.45, y + h * 0.3)], h * 0.04, col="#111111")
        reg.poly(_ellipse(x + h * 0.2, y - h * 0.18, h * 0.22, h * 0.09, 0.0), col="#111111")
        reg.poly([(x + h * 0.4, y - h * 0.18), (x + h * 0.5, y - h * 0.08), (x + h * 0.5, y - h * 0.28)], col="#111111")


def fuel_tank(col, M):
    """A 2500 L bunded steel diesel tank (2.2 x 1.3 x 1.6 m): welded shell with 20 mm bevels,
    pressed stiffening ribs and a perimeter weld seam; box-section skids with fork pockets; four
    lifting eyes; a bolted manway, a contents gauge, a mushroom vent and a fill cap on top; an end
    dispensing cabinet with its door swung open on a hasp and padlock, showing the pump and meter
    (register dial), a coiled 25 mm hose on its hook and the nozzle in its holster; DIESEL, CLP
    flammable and environment diamonds, capacity plate, no-smoking sign; chipped edges, rust
    streaks, grime and a fuel stain on the ground. Front +y."""
    cv = Canvas("Amen_Fuel", 1024, 1024, col="#2f5d3a", rough=0.55)
    lo, hi = (-1.1, -0.65, 0.15), (1.1, 0.65, 1.6)
    R = box_regions(cv, lo, hi, 200, ("+y", "-y", "+x", "-x", "+z"))
    door = cv.alloc(0.9, 0.98, 250)
    dial = cv.alloc(0.16, 0.16, 600)
    ribr = cv.alloc(2.2, 0.2, 200)
    ribr.fill(col="#1f4d32", rough=0.45)
    ribr.mottle(0.05, 8, 71)
    ribr.edge_wear(0, 0, 2.2, 0.2, 0.01, "#8a8a84", 0.4, 72)
    ribr.field(lambda X, Y: np.clip((noise(X.shape, (4, 80), 73, 3) - 0.5) * 3, 0, 1) * 0.5, col="#6e3e1e")
    soil = cv.alloc(1.2, 1.0, 120)
    green = "#1f4d32"
    for i, (k, reg) in enumerate(list(R.items()) + [("door", door)]):
        reg.fill(col=green, rough=0.42)
        reg.field(lambda X, Y: noise(X.shape, (int(reg.h * 60) + 1, int(reg.w * 60) + 1), 80 + i, 2), dh=0.0002)
        if k != "+z":
            weather(reg, z0=0.15 if k != "door" else 0.36, seed=6 * i + 1, mottle=0.06, grime_h=0.35, grime="#4a4030",
                    grime_k=0.65, streak_len=0.7, streak="#3a3a2c", streak_k=0.3)
            reg.field(lambda X, Y: np.clip(1 - np.minimum(X, reg.w - X) / 0.02, 0, 1) * np.clip((noise(X.shape, (40, 4), 30 + i, 3) - 0.45) * 4, 0, 1) * 0.8,
                      col="#3b3a36")
            # weld seam along the top edge and rust bleeding from it
            reg.rect(0, reg.h - 0.03, reg.w, reg.h - 0.024, col="#26482d", dh=0.0012)
            reg.streaks(reg.h - 0.03, 0.12, "#6e3e1e", 0.25, 50 + i, density=25)
            reg.streaks(reg.h - 0.03, 0.4, "#183a26", 0.15, 55 + i, density=30)
            reg.grime(0.15, "#5e3a1e", 0.6, 58 + i, y_ground=0.0)
            # rust bloom along the bottom edge
            reg.field(lambda X, Y: np.clip(1 - Y / 0.08, 0, 1) * np.clip((noise(X.shape, (4, 60), 56 + i, 3) - 0.4) * 3, 0, 1) * 0.8, col="#6a3a1c", rough=0.8)
        else:
            reg.field(lambda X, Y: np.ones_like(X) * 0.18, col="#8aa070")
            reg.field(lambda X, Y: noise(X.shape, 8, 60, 4) * 0.45, col="#7a7563")
            # run marks from the vent and the fill cap
            for cx_, cy_ in ((0.2, 0.25), (1.5, 0.3)):
                reg.field(lambda X, Y, cx_=cx_, cy_=cy_: np.clip(1 - np.hypot(X - cx_, Y - cy_) / 0.12, 0, 1) * 0.6, col="#2a2a1e", rough=0.25)
            reg.edge_wear(0, 0, reg.w, reg.h, 0.015, "#8a8a84", 0.6, 61)
    f = R["+y"]
    f.rect(0.25, 0.95, 1.05, 1.2, r=0.01, col="#f2c200", dh=0.0004)
    f.text("DIESEL", 0.65, 1.0, 0.16, weight="black", align="c", col="#111111")
    _diamond(f, 1.25, 1.07, 0.24, "flame")
    _diamond(f, 1.55, 1.07, 0.24, "env")
    f.rect(1.75, 0.95, 2.05, 1.18, r=0.006, col="#c4c7c9", metal=0.8, rough=0.3, dh=0.0006)
    for i, (t, hh) in enumerate((("FUELSAFE", 0.03), ("2500 LITRES", 0.026), ("110% BUNDED", 0.02), ("BS EN 12285-2", 0.016))):
        f.text(t, 1.9, 1.14 - i * 0.045, hh, weight="bold", align="c", col="#222222", metal=0.0)
    f.circle(0.4, 0.5, 0.09, col="#d21f1f")
    f.circle(0.4, 0.5, 0.075, col="#ffffff")
    f.rect(0.33, 0.485, 0.47, 0.515, col="#111111")
    f.line([(0.345, 0.44), (0.455, 0.56)], 0.014, col="#d21f1f")
    f.text("NO SMOKING", 0.4, 0.52, 0.03, weight="black", align="c", col="#ffffff")
    f.text("FUELSAFE", 1.1, 0.25, 0.07, weight="black", align="c", col="#d8e2d8")
    b_ = R["-y"]
    b_.rect(0.6, 0.95, 1.6, 1.2, r=0.01, col="#f2c200", dh=0.0004)
    b_.text("DIESEL", 1.1, 1.0, 0.16, weight="black", align="c", col="#111111")
    # the cabinet end: dark interior behind the open door, fuel stains
    e = R["+x"]
    e.rect(0.2, 0.21, 1.1, 1.19, col="#1c241e", rough=0.4)
    e.field(lambda X, Y: np.clip(1 - (Y - 0.21) / 0.3, 0, 1) * noise(X.shape, 10, 70, 3) * 0.8, 0.2, 0.21, 1.1, 1.19, col="#0c0d0b", rough=0.15)
    door.rect(0.15, 0.55, 0.75, 0.75, r=0.008, col="#f2c200", dh=0.0004)
    door.text("DIESEL", 0.45, 0.6, 0.11, weight="black", align="c", col="#111111")
    door.text("Switch off engine · No smoking", 0.45, 0.47, 0.028, weight="bold", align="c", col="#e6eee6", max_w=0.7)
    _diamond(door, 0.3, 0.25, 0.18, "flame")
    _diamond(door, 0.6, 0.25, 0.18, "env")
    # meter register dial
    dial.fill(col="#f2f0e8", rough=0.2)
    dial.ring(0.08, 0.08, 0.072, 0.08, col="#2a2a2a")
    dial.rect(0.025, 0.085, 0.135, 0.115, col="#111111")
    for i in range(5):
        dial.text(str((3 + 7 * i) % 10), 0.035 + i * 0.021, 0.091, 0.017, weight="bold", col="#f2f0e8")
    dial.text("LITRES", 0.08, 0.06, 0.012, weight="black", align="c", col="#222222")
    for k in range(10):
        t = math.pi * (1.25 - 1.5 * k / 9)
        dial.line([(0.08 + 0.058 * math.cos(t), 0.04 + 0.058 * math.sin(t) * 0.4), (0.08 + 0.066 * math.cos(t), 0.04 + 0.066 * math.sin(t) * 0.4)], 0.002, col="#222222")
    dial.line([(0.08, 0.03), (0.06, 0.055)], 0.003, col="#c21d1d")
    # stain on the ground under the cabinet
    soil.fill(col="#141310", rough=0.15, a=0.0)
    soil.field(lambda X, Y: np.clip(1 - np.hypot((X - 0.5) / 0.45, (Y - 0.5) / 0.38) + (noise(X.shape, 8, 71, 4) - 0.5) * 0.8, 0, 1) ** 1.2 * 0.75, a=1.0)
    mat = cv.material("Amen_Fuel")
    stain = cv.material("Amen_FuelStain", rough=0.15, alpha=True, blend=True, normal=0)

    p = PM(0.02)
    # skids and fork pockets
    p.r = 0.006
    for y in (-0.5, 0.5):
        p.box((-1.12, y - 0.05, 0.0), (1.12, y + 0.05, 0.1), M["steel"])
    for x in (-0.5, 0.5):
        p.box((x - 0.14, -0.66, 0.08), (x + 0.14, 0.66, 0.15), M["steel"])
        for sy in (-1, 1):
            decal(p, (x, sy * 0.66, 0.115), (0, sy, 0), (0, 0, 1), 0.22, 0.05, M["rubber"], lift=0.001)
    # the shell, pressed ribs, weld seam
    p.r = 0.02
    i0 = len(p.F)
    p.box(lo, hi, mat)
    # box-section stiffening ribs, 50 mm proud and 60 mm deep, along both long sides
    ir = len(p.F)
    for z in (0.32, 0.78):
        for sy in (-1, 1):
            p.r = 0.008
            p.box((-1.07, sy * 0.65 - 0.05 * (sy < 0), z - 0.03), (1.07, sy * 0.65 + 0.05 * (sy > 0), z + 0.03), mat)
    project(p, ir, ribr, (-1.1, 0, 0.0), (1.0, 0, 0), (0, 0.15, 0.15))
    box_uv(p, i0, R, lo, hi)
    # 30 mm flanged roof perimeter
    p.r = 0.004
    _loft(p, [rrect(2.28, 1.38, 0.04, z) for z in (1.56, 1.6)], _paint("Amen_FuelSeam", "#1c4630", 0.45))
    p.r = 0.0
    seam = [(-1.104, -0.654, 1.57), (1.104, -0.654, 1.57), (1.104, 0.654, 1.57), (-1.104, 0.654, 1.57)]
    _tube(p, seam + [seam[0]], 0.005, _paint("Amen_FuelSeam", "#28502f", 0.5), 4, bend=0.0, closed=True)
    # lifting eyes at the corners
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.r = 0.004
            c_ = np.array([sx * 0.98, sy * 0.53, 1.6])
            # an 80 mm lifting plate with a 40 mm hole (a ring of four bars round it)
            pm = _paint("Amen_FuelSeam", "#28502f", 0.5)
            p.box(tuple(c_ - (0.04, 0.008, 0.0)), tuple(c_ + (0.04, 0.008, 0.02)), pm)
            p.box(tuple(c_ + (-0.04, -0.008, 0.06)), tuple(c_ + (0.04, 0.008, 0.08)), pm)
            p.box(tuple(c_ + (-0.04, -0.008, 0.02)), tuple(c_ + (-0.02, 0.008, 0.06)), pm)
            p.box(tuple(c_ + (0.02, -0.008, 0.02)), tuple(c_ + (0.04, 0.008, 0.06)), pm)
    # top fittings: manway with bolts, gauge, vent, fill cap
    p.r = 0.006
    i0 = len(p.F)
    p.lathe([(0.25, 1.6), (0.25, 1.64), (0.23, 1.655), (0.0, 1.655)], mat, c=(-0.4, 0.0, 0), sides=28)
    project(p, i0, R["+z"], (-0.7, -0.3, 0), (1, 0, 0), (0, 1, 0))
    for k in range(8):
        a_ = 2 * math.pi * k / 8
        _hexbolt(p, (-0.4 + 0.21 * math.cos(a_), 0.21 * math.sin(a_), 1.645), 0.011, 0.01, M["galv"])
    p.r = 0.004
    p.cyl((0.25, 0.35, 1.6), (0.25, 0.35, 1.66), 0.05, M["galv"], 16)
    p.cyl((0.25, 0.35, 1.66), (0.25, 0.35, 1.68), 0.06, M["chrome"], 16)
    decal(p, (0.25, 0.35, 1.68), (0, 0, 1), (0, 1, 0), 0.1, 0.1, _gauge_mat(), lift=0.001)
    p.cyl((-0.9, -0.4, 1.6), (-0.9, -0.4, 1.96), 0.03, M["galv"], 12)
    p.lathe([(0.03, 1.94), (0.09, 1.95), (0.095, 1.97), (0.06, 2.01), (0.0, 2.02)], M["galv"], c=(-0.9, -0.4, 0), sides=16)
    p.cyl((0.4, -0.35, 1.6), (0.4, -0.35, 1.68), 0.04, M["galv"], 14)
    p.cyl((0.4, -0.35, 1.68), (0.4, -0.35, 1.72), 0.05, M["black"], 14)
    # the dispensing cabinet on the +x end: walls, the door swung open, pump, meter, hose, nozzle
    cx0, cx1, cy, cz0, cz1 = 1.1, 1.45, 0.45, 0.36, 1.34
    p.r = 0.006
    for (a0, a1) in (((cx0, -cy, cz1), (cx1, cy, cz1 + 0.02)), ((cx0, -cy, cz0 - 0.02), (cx1, cy, cz0)),
                     ((cx0, -cy - 0.02, cz0 - 0.02), (cx1, -cy, cz1 + 0.02)), ((cx0, cy, cz0 - 0.02), (cx1, cy + 0.02, cz1 + 0.02))):
        p.box(a0, a1, _paint("Amen_FuelCab", "#2f5d3a", 0.55))
    th = math.radians(105)
    hinge = np.array([cx1, cy + 0.02, 0])
    dvec = np.array([math.sin(th), -math.cos(th), 0.0])     # from the hinge across the door
    nvec = np.array([math.cos(th), math.sin(th), 0.0])
    dc = hinge + dvec * 0.46 + np.array([0, 0, (cz0 + cz1) / 2])
    p.r = 0.006
    p.obox(dc - nvec * 0.0145, (0.92, 0.025, 0.98), _paint("Amen_FuelDoorIn", "#2a5434", 0.55), yaw=math.atan2(dvec[1], dvec[0]))
    # T-handle lock on the door, drip lip over the opening
    tl = dc + dvec * 0.36 + nvec * 0.002
    p.r = 0.003
    p.cyl(tl, tl + nvec * 0.03, 0.012, M["chrome"], 10)
    p.cyl(tl + nvec * 0.03 - np.array([0, 0, 0.04]), tl + nvec * 0.03 + np.array([0, 0, 0.04]), 0.008, M["chrome"], 8)
    p.box((cx1 - 0.01, -cy - 0.03, cz1 + 0.02), (cx1 + 0.04, cy + 0.03, cz1 + 0.03), _paint("Amen_FuelCab", "#2f5d3a", 0.55))
    # earth bonding cable from the cabinet to a spike in the ground, crocodile clip on the spike
    p.r = 0.0
    _tube(p, _spline([(cx0 + 0.1, -cy + 0.05, cz0), (cx1 + 0.05, -cy - 0.05, 0.2), (cx1 + 0.25, -cy - 0.15, 0.04), (cx1 + 0.35, -cy - 0.25, 0.12)], 6),
          0.005, _plastic("Amen_EarthLead", "#3f8a3a", 0.5), 6, bend=0.0)
    p.r = 0.002
    p.cyl((cx1 + 0.36, -cy - 0.26, -0.05), (cx1 + 0.36, -cy - 0.26, 0.15), 0.008, M["chrome"], 8)
    p.obox((cx1 + 0.35, -cy - 0.255, 0.12), (0.016, 0.012, 0.05), M["red"], roll=0.3)
    decal(p, dc + nvec * 0.0, nvec, (0, 0, 1), 0.9, 0.98, mat, door, lift=0.002)
    for z in (cz0 + 0.12, cz1 - 0.12):
        p.cyl(hinge + (0, 0, z - 0.05), hinge + (0, 0, z + 0.05), 0.012, M["galv"], 8)
    hp = hinge + dvec * 0.88 + np.array([0, 0, 0.85])
    p.r = 0.002
    p.box(tuple(hp - (0.01, 0.01, 0.03)), tuple(hp + (0.01, 0.01, 0.03)), M["galv"])
    pl = hp + nvec * 0.03 - np.array([0, 0, 0.05])
    p.box(tuple(pl - (0.022, 0.012, 0.025)), tuple(pl + (0.022, 0.012, 0.025)), _paint("Amen_Padlock", "#b8963c", 0.3))
    _tube(p, [pl + (-0.014, 0, 0.022), pl + (-0.014, 0, 0.05), pl + (0.014, 0, 0.05), pl + (0.014, 0, 0.035)], 0.004, M["chrome"], 6, bend=0.012)
    p.box((cx1 - 0.015, -cy - 0.02, cz1 - 0.2), (cx1 + 0.01, -cy + 0.01, cz1 - 0.14), M["galv"])
    # pump and meter on the back wall of the cabinet
    p.r = 0.008
    p.box((cx0 + 0.005, -0.38, 0.9), (cx0 + 0.16, -0.1, 1.18), _paint("Amen_Meter", "#d8d8d2", 0.4))
    decal(p, (cx0 + 0.16, -0.24, 1.04), (1, 0, 0), (0, 0, 1), 0.16, 0.16, mat, dial, lift=0.001)
    p.cyl((cx0, -0.24, 0.6), (cx0 + 0.2, -0.24, 0.6), 0.075, M["grey"], 16)
    p.r = 0.0
    _tube(p, [(cx0 + 0.2, -0.24, 0.6), (cx0 + 0.08, -0.24, 0.75), (cx0 + 0.08, -0.24, 0.9)], 0.018, M["black"], 8, bend=0.05)
    # hose reel drum on the back wall (the coils sit on it), door stay arm
    p.r = 0.004
    p.cyl((cx0 + 0.02, 0.2, 0.96), (cx0 + 0.2, 0.2, 0.96), 0.1, M["grey"], 24)
    p.cyl((cx0 + 0.02, 0.2, 0.96), (cx0 + 0.03, 0.2, 0.96), 0.2, M["red"], 24)
    p.r = 0.0
    p.cyl((cx1, cy, cz1 - 0.05), tuple(hinge + dvec * 0.35 + np.array([0, 0, cz1 - 0.05]) + nvec * 0.03), 0.006, M["chrome"], 6)
    # the hose: 25 mm, coiled three times on a hook, out to the nozzle holster
    p.r = 0.004
    p.box((cx0, 0.18, 1.12), (cx0 + 0.18, 0.22, 1.15), M["galv"])
    coil = []
    for t in np.linspace(0, 3 * 2 * math.pi, 48):
        rr = 0.17 + 0.012 * math.sin(t * 0.5)
        coil.append((cx0 + 0.13 + 0.015 * t / (2 * math.pi), 0.2 + rr * math.sin(t) * 0.55, 0.96 - rr * math.cos(t) + 0.0))
    p.r = 0.0
    _tube(p, [(cx0 + 0.16, -0.1, 1.0), (cx0 + 0.13, 0.05, 1.12)] + coil, 0.0135, M["rubber"], 8, bend=0.03)
    p.r = 0.003
    p.box((cx0 + 0.005, 0.28, 0.5), (cx0 + 0.12, 0.38, 0.56), M["galv"])
    nz = np.array([cx0 + 0.09, 0.33, 0.62])
    p.r = 0.006
    p.obox(nz, (0.04, 0.05, 0.16), M["black"], roll=0.3)
    p.r = 0.002
    p.cyl(nz + (0.02, 0, -0.07), nz + (0.08, 0, -0.17), 0.011, M["chrome"], 10)
    _tube(p, [nz + (-0.02, 0, 0.08), nz + (-0.04, -0.05, 0.25), coil[-1]], 0.0135, M["rubber"], 8, bend=0.06)
    decal(p, (1.2, 0.0, 0.0), (0, 0, 1), (0, 1, 0), 1.2, 1.0, stain, soil, lift=0.003)
    return p.obj("fuel_tank", col, 40)


BUILDERS = {
    "prop.sign_post": sign_post, "prop.water_fountain": water_fountain, "prop.planter": planter,
    "prop.traffic_cone": traffic_cone, "prop.ticket_machine": ticket_machine, "prop.cctv_pole": cctv_pole,
    "prop.fire_point": fire_point, "prop.ev_charger": ev_charger, "prop.security_hut": security_hut,
    "prop.parasol": parasol, "prop.flagpole": flagpole, "prop.satellite_dish": satellite_dish,
    "prop.fuel_tank": fuel_tank,
}
