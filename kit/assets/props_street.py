"""Street furniture for training grounds: bins, benches, tables, bollards, the car-park barrier,
lighting and cycle stands (amenities and services are in props_amenity).

Each is modelled on the real item at its real size (a Glasdon Jubilee-style litter bin, a Nexus
140-style recycling trio, a 240 L wheelie bin, an 1100 L Eurobin, a cast-iron park bench, an 8 m
stepped LED column, Sheffield cycle stands ...). Origin on the ground at the footprint centre, +Y
the front. Finishes are painted textures laid out by height (street_paint.skin): soil and splash at
the foot, scuffs at knee and bumper height, rain streaks under rims, sun fading higher up.
"""
import math

import numpy as np

from . import street_paint as SP
from mathutils import Matrix

from .props import PM, emissive, round_path, rrect, tyre

TAU = 2 * math.pi


# -- shared finishes ------------------------------------------------------------------------------

def _tape(name, col, dirt=0.0):
    """Retro-reflective sheeting: micro-prism lattice (3 mm), a little sparkle, dirt low down."""
    def draw(c):
        H, W = c.H, c.W
        X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        k = TAU / (0.003 * c.k)
        prism = np.abs(np.sin(X * k * 0.5 + Y * k * 0.29)) + np.abs(np.sin(X * k * 0.5 - Y * k * 0.29))
        n = SP.noise((H, W), (8, 8), c.r, 3)
        c.alb = c.alb * (0.9 + 0.12 * (prism / 2)[..., None]) * (0.95 + 0.1 * n[..., None])
        c.hgt = prism * 0.6
        if dirt:
            c.alb = SP.mixc(c.alb, SP.lin("#6f6858"), SP.sstep(0.45, 0.8, SP.noise((H, W), (4, 4), c.r, 4)) * dirt)
        c.rough = 0.28 + 0.1 * n
    m = SP.card(name, 0.1, 0.1, draw, ppm=2560, bg=col)
    m["uv_cell"] = (0.1, 0.1)
    return m


def _plain(name, col, rough=0.5, metal=0.0, H=1.2, tile=1.0, **kw):
    """A skin with no height features (lids, tops, small fittings)."""
    kw.setdefault("grime", 0.0)
    kw.setdefault("px", (256, 256))
    return SP.skin(name, col, H, tile=tile, rough=rough, metal=metal, **kw)


def _void():
    return _plain("Street_Void", "#0b0b0b", 0.9, mottle=0.0, peel=0.0)


def _concrete(name="Street_Bedding"):
    return SP.skin(name, "#8e8a82", 0.2, tile=0.5, px=(256, 256), rough=0.85, grime=0.03, grime_amt=0.3, mottle=0.12,
                   peel=0.4, normal_strength=1.0, splash=0.6)


def _key_lock(name="Street_TriLock"):
    """End of a lock cylinder: a ring with a triangular key spindle."""
    def draw(c):
        w = c.W / c.k
        c.poly(SP.circle(w / 2, w / 2, w * 0.48), "#55544f")
        c.poly(SP.circle(w / 2, w / 2, w * 0.38), "#262624")
        t = [(w / 2 + w * 0.2 * math.cos(math.radians(90 + 120 * i)), w / 2 + w * 0.2 * math.sin(math.radians(90 + 120 * i))) for i in range(3)]
        c.poly(t, "#5c5b57")
        c.rough = None
    return SP.card(name, 0.03, 0.03, draw, ppm=8000, bg="#3c3b38", metal=0.6, rough=0.35, alpha=True)


# -- bins -----------------------------------------------------------------------------------------

def bin_litter(col, M):
    """A Jubilee-style 110 L litter bin (1158 x 598 x 553): oval fluted body on a plinth, raised
    gold band, a hooded lid with two posting apertures (front and back) under the dome, a
    galvanised liner and black sack inside with litter showing, triangular-key locks."""
    p = PM(0.0)
    R, SY = 0.292, 0.925
    tile = math.pi * 0.29
    body = SP.skin("TINT_bin_litter", "#1f2023", 1.2, tile=tile, rough=0.45, grime=0.17, grime_amt=0.6, splash=0.6,
                   scuff=(0.2, 0.55, 0.5), streaks=[(0.862, 0.35, 0.18), (0.895, 0.55, 0.25), (1.045, 0.3, 0.12)], chalk=0.2,
                   peel=0.05, gum=5, top_fade=(1.07, 0.5))
    gold = _tape("Street_TapeGold", "#c9a227", 0.2)
    prof = [(0.300, 0.0), (0.302, 0.010), (0.300, 0.045), (0.278, 0.055), (0.272, 0.065), (0.282, 0.42), (0.290, 0.76),
            (R, 0.785), (R, 0.800), (0.2965, 0.803), (0.2965, 0.837), (R, 0.840), (R, 0.862), (0.300, 0.866), (0.302, 0.876),
            (0.296, 0.886), (R, 0.890), (R, 0.895), (R, 1.040), (R, 1.045), (0.312, 1.047), (0.313, 1.060), (0.300, 1.080),
            (0.270, 1.108), (0.215, 1.135), (0.140, 1.152), (0.060, 1.159), (0.0, 1.160)]
    depth = [0.06 if 4 <= j <= 6 else 0.0 for j in range(len(prof))]
    n = 88
    ap = set(range(15, 29)) | set(range(59, 73))
    J_AP = 17
    SP.lathe(p, prof, body, sides=n, flutes=22, depth=depth, sy=SY, uv_r=0.29,
             mats=lambda j, k: gold if j in (8, 9, 10) else body, skip=lambda j, k: j == J_AP and k in ap)
    # the lid shell inside: inner wall, ceiling, jambs round each aperture
    RI = R - 0.035
    SP.lathe(p, [(RI, 1.045), (RI, 1.040), (RI, 0.895), (RI, 0.862), (0.248, 0.862)], body, sides=n, sy=SY, uv_r=0.29,
             skip=lambda j, k: j == 1 and k in ap)
    SP.lathe(p, [(0.0, 1.045), (RI, 1.045)], _void(), sides=n, sy=SY)
    zb, zt = 0.895, 1.040
    O = lambda a, z: SP.ring_pt(R, a, z, sy=SY)
    I = lambda a, z: SP.ring_pt(RI, a, z, sy=SY)
    for k0, k1 in ((15, 29), (59, 73)):
        A = [TAU * k / n for k in range(k0, k1 + 1)]
        for a, b in zip(A, A[1:]):
            p.face([O(a, zb), O(b, zb), I(b, zb), I(a, zb)], body)
            p.face([O(a, zt), I(a, zt), I(b, zt), O(b, zt)], body)
        p.face([O(A[0], zb), I(A[0], zb), I(A[0], zt), O(A[0], zt)], body)
        p.face([O(A[-1], zb), O(A[-1], zt), I(A[-1], zt), I(A[-1], zb)], body)
    # a moulded lip round each mouth: a sill under it and cheeks either side
    p.r = 0.002
    for k0, k1 in ((15, 29), (59, 73)):
        SP.lathe(p, [(R, 0.884), (R + 0.009, 0.884), (R + 0.009, 0.895), (R, 0.897)], body, sides=n, sy=SY, uv_r=0.29,
                 skip=lambda j, k, k0=k0, k1=k1: not (k0 - 1 <= k <= k1))
        SP.lathe(p, [(R, 0.884), (R + 0.007, 0.886), (R + 0.007, 1.044), (R, 1.045)], body, sides=n, sy=SY, uv_r=0.29,
                 skip=lambda j, k, k0=k0, k1=k1: k not in (k0 - 1, k1))
    # lid hinge knuckles low on the back, under the rear mouth
    p.r = 0.0
    yb = -R * SY - 0.008
    for x in (-0.09, 0.09):
        p.cyl((x - 0.03, yb, 0.873), (x + 0.03, yb, 0.873), 0.009, M["galv"], 10)
        p.box((x - 0.022, yb + 0.002, 0.866), (x + 0.022, yb + 0.01, 0.88), M["galv"])
    # liner rim, sack and litter seen through the mouths
    p.r = 0.0
    sack = SP.skin("Street_Sack", "#151617", 1.2, tile=tile, px=(256, 512), rough=0.3, grime=0.0, mottle=0.25, peel=0.3)
    rim = [SP.ring_pt(0.249, TAU * k / 48, 0.866, sy=SY) for k in range(48)]
    SP.sweep(p, rim, [(0.006 * math.cos(TAU * j / 6), 0.006 * math.sin(TAU * j / 6)) for j in range(6)], M["galv"], closed=True)
    bag = []
    for zz, rr in ((0.872, 0.252), (0.86, 0.246), (0.83, 0.236), (0.80, 0.232)):
        bag.append([SP.ring_pt(rr * (1 + 0.025 * math.sin(7 * TAU * k / 48 + zz * 40)), TAU * k / 48, zz, sy=SY) for k in range(48)])
    p.loft(bag, sack, cap0=False, cap1=False)
    SP.lathe(p, [(0.232, 0.80), (0.16, 0.79), (0.0, 0.775)], sack, sides=24, sy=SY)
    p.r = 0.003
    p.cyl((-0.08, 0.13, 0.800), (0.03, 0.17, 0.795), 0.033, M["red"], 14)                    # a drinks can on its side
    p.cyl((0.06, -0.16, 0.79), (0.12, -0.08, 0.84), 0.03, M["white"], 12, r2=0.042)       # a coffee cup
    p.sphere((0.09, 0.14, 0.805), 0.045, M["white"], 10, 6, scale=(1.2, 0.9, 0.7))        # crumpled paper
    p.sphere((-0.12, -0.12, 0.80), 0.04, _plain("Street_Wrapper", "#c9a227", 0.4), 10, 6, scale=(1.3, 0.8, 0.6))
    # triangular-key locks either side of the lid, bolts on the plinth
    lock = _key_lock()
    for s in (-1, 1):
        p.r = 0.002
        p.cyl((s * R, 0, 0.965), (s * (R + 0.006), 0, 0.965), 0.015, M["steel"], 16)
        p.decal((s * (R + 0.007), 0, 0.965), (s, 0, 0), (0, 0, 1), 0.026, 0.026, lock, lift=0.0005)
    for a in (math.radians(30), math.radians(150), math.radians(270)):
        x, y, _ = SP.ring_pt(0.289, a, 0, sy=SY)
        SP.bolt_head(p, (x, y, 0.0475), (0, 0, 1), 0.008, M["galv"], hexa=True)
    # gold lettering and crest painted on the front
    def draw(c):
        w, h = c.W / c.k, c.H / c.k
        c.text("LITTER", w / 2, 0.012, 0.042, "#c9a24a")
        cx, cy = w / 2, 0.095
        shield = [(cx - 0.032, cy + 0.035), (cx + 0.032, cy + 0.035), (cx + 0.032, cy - 0.005), (cx + 0.02, cy - 0.028),
                  (cx, cy - 0.04), (cx - 0.02, cy - 0.028), (cx - 0.032, cy - 0.005)]
        c.poly(shield, "#c9a24a")
        inner = [(x * 0.82 + cx * 0.18, y * 0.82 + (cy - 0.002) * 0.18) for x, y in shield]
        c.poly(inner, "#1d1f20")
        c.poly(SP.rect(cx - 0.004, cy - 0.03, cx + 0.004, cy + 0.028), "#c9a24a")
        c.poly(SP.rect(cx - 0.024, cy + 0.006, cx + 0.024, cy + 0.014), "#c9a24a")
        c.age(0.18, 0.0)
        sc = SP.sstep(0.9, 0.97, SP.noise((c.H, c.W), (60, 3), c.r, 2)) * SP.sstep(0.5, 0.7, SP.noise((c.H, c.W), (5, 5), c.r, 3))
        c.alpha = c.alpha * (1 - sc * 0.9)
    def draw2(c):
        draw(c)
        c.hgt = SP.blur(c.alpha * 3.0, 2)
    plate = SP.card("Street_LitterLettering", 0.26, 0.14, draw2, ppm=2400, bg="#c9a24a", metal=0.5, rough=0.32, alpha=True)
    SP.wrap_decal(p, 0.2876, 0.60, 0.74, math.pi / 2, 0.26, plate, sides=20, lift=0.0025, flutes=22, depth=0.06, sy=SY)
    SP.top_uv(p)
    return p.obj("bin_litter", col, 40)


def _recycle_label(stream):
    title, sub, col, icon = {
        "glass": ("GLASS", "Bottles & jars only", "#2f7d3e", "bottle"),
        "paper": ("PAPER & CARD", "Newspapers, magazines, card", "#1f5aa6", "paper"),
        "general": ("GENERAL WASTE", "Non-recyclable items", "#4a4e52", "bag"),
    }[stream]

    def draw(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.006, 0.006, w - 0.006, h - 0.006, 0.018), "#f4f4f0")
        c.poly(SP.rect(0.012, 0.012, w - 0.012, h - 0.012, 0.014), col)
        c.text(title, w / 2, h - 0.06, 0.036, "#ffffff", max_w=w - 0.05)
        c.poly(SP.rect(0.03, h - 0.075, w - 0.03, h - 0.071), "#ffffff", 0.8)
        cx, cy, s = w / 2, h * 0.43, 0.13
        if icon == "bottle":
            c.poly(SP.picto_bottle(cx - 0.03, cy, s), "#ffffff")
            c.poly(SP.picto_bottle(cx + 0.045, cy - 0.012, s * 0.8), "#ffffff")
        elif icon == "paper":
            back, front = SP.picto_paper(cx, cy, s)
            c.poly(back, "#ffffff", 0.75)
            c.poly(front, "#ffffff")
        else:
            for part in SP.picto_bag(cx, cy, s):
                c.poly(part, "#ffffff")
        if stream != "general":
            for part in SP.picto_recycle(w - 0.045, 0.06, 0.05):
                c.poly(part, "#ffffff")
        c.text(sub, w / 2 - (0.02 if stream != "general" else 0), 0.03, 0.014, "#ffffff", bold=False, max_w=w - 0.11)
        c.age(0.2, 0.15)
        if stream == "glass":                                   # a corner lifting away, catching light, dirt under it
            corner = [(w - 0.006, h - 0.006), (w - 0.006, h - 0.07), (w - 0.07, h - 0.006)]
            c.poly([(x, y - 0.004) for x, y in corner], "#2a2a28")
            c.poly(corner, "#e9e7df")
    return SP.card(f"Street_RecycleLabel_{stream}", 0.36, 0.28, draw, ppm=1800, bg="#f4f4f0")


def _round_port(p, o, eu, ev, c, half, rad, rubber, void, depth=0.07, K=32):
    """A round port in a square opening (centre c = (u, v), half-width `half`): a rubber plate
    from the square to a circle of radius rad, and a rubber collar going down into the dark."""
    o, eu, ev = (np.asarray(x, float) for x in (o, eu, ev))
    n = np.cross(eu, ev)
    P = lambda u, v, d: o + eu * u + ev * v - n * d
    outer, inner = [], []
    for k in range(K + 1):
        a = TAU * k / K
        ca, sa = math.cos(a), math.sin(a)
        t = half / max(abs(ca), abs(sa))
        outer.append((c[0] + t * ca, c[1] + t * sa))
        inner.append((c[0] + rad * ca, c[1] + rad * sa))
    d0 = 0.004
    for k in range(K):
        p.face([P(*outer[k], d0), P(*outer[k + 1], d0), P(*inner[k + 1], d0), P(*inner[k], d0)], rubber)
        p.face([P(*inner[k], d0), P(*inner[k + 1], d0), P(*inner[k + 1], depth), P(*inner[k], depth)], rubber)
    p.face([P(*inner[k], depth) for k in range(K)], void)


def _bristles(name="Street_Brush"):
    def draw(c):
        X = (np.arange(c.W) + 0.5)[None, :] / c.k * np.ones((c.H, 1))
        Y = (np.arange(c.H) + 0.5)[:, None] / c.k * np.ones((1, c.W))
        lines = SP.sstep(0.35, 0.6, np.abs(np.sin(X * TAU / 0.0016 + SP.noise((c.H, c.W), (40, 4), c.r, 2) * 3)))
        tips = 1 - SP.sstep(0.85, 1.0, Y / (c.H / c.k) + SP.noise((c.H, c.W), (60, 2), c.r, 2) * 0.15)
        c.alpha = np.clip(lines * tips, 0, 1)
        c.rough = 0.6
    return SP.card(name, 0.1, 0.03, draw, ppm=6000, bg="#151515", alpha=True)


def bin_recycling(col, M):
    """A three-stream recycling station: three Nexus 140-style units (595 x 606 x 1195) on a
    shared galvanised base rail, linked by brackets: each a tapered moulded body on a recessed,
    bolted plinth with a hinged front door (hinges, a triangular-key lock), panelled sides and
    back with a handle recess, a raked hood whose stream-coloured front carries the aperture (a
    round rubber-collared port for glass, a brushed slot for paper, an open mouth for general
    waste), and a printed stream label with pictograms, sun-faded and scratched."""
    p = PM(0.02)
    body = SP.skin("TINT_bin_recycling", "#3c3e41", 1.25, tile=1.0, rough=0.6, grime=0.17, grime_amt=0.6, splash=0.5,
                   scuff=(0.12, 0.55, 0.6), streaks=[(0.92, 0.45, 0.2)], chalk=0.12, peel=0.05, stickers=1)
    void, lock = _void(), _key_lock()
    W, D = 0.595, 0.606
    th = math.atan(0.013 / 0.85)
    yf = lambda z: 0.29 + (z - 0.07) * 0.013 / 0.85
    xh = lambda z: 0.2825 + (z - 0.07) * 0.015 / 0.85
    ths = math.atan(0.015 / 0.85)
    for i, (stream, hexc, hole) in enumerate((("glass", "#2f7d3e", (0.2, 0.2)), ("paper", "#1f5aa6", (0.30, 0.055)),
                                              ("general", "#6b6f73", (0.30, 0.14)))):
        cx = (i - 1) * (W + 0.012)
        hu0, hv0 = 0.3 - hole[0] / 2, 0.06

        def lip_dirt(alb, rgh, hgt, X, Z, r, h0=(hu0, hv0, hu0 + hole[0], hv0 + hole[1])):
            dx = np.maximum(np.maximum(h0[0] - X, X - h0[2]), 0)
            dz = np.maximum(np.maximum(h0[1] - Z, Z - h0[3]), 0)
            d = np.hypot(dx, dz)
            m = (d > 0) * (1 - SP.sstep(0.0, 0.035, d)) * (0.5 + 0.5 * SP.noise(X.shape, (12, 12), r, 3))
            alb[...] = SP.mixc(alb, alb * 0.55, m * 0.8)
            rgh += 0.15 * m
        hood = SP.skin(f"Street_RecycleHood_{stream}", hexc, 0.5, tile=1.0, px=(512, 256), rough=0.55, grime=0.0, mottle=0.06,
                       scuff=(0.0, 0.5, 0.5), chalk=0.25, paint=lip_dirt)
        hood_body = SP.skin(f"Street_RecycleHoodBody_{stream}", hexc, 1.25, tile=1.0, px=(256, 512), rough=0.58, grime=0.0, mottle=0.06,
                            chalk=0.2, streaks=[(1.0, 0.3, 0.05)], peel=0.05)
        p.r = 0.01
        p.loft([rrect(0.555, 0.57, 0.045, z, cx=cx) for z in (-0.01, 0.072)], M["black"])
        for sx in (-1, 1):
            for yy in (-0.15, 0.15):
                SP.bolt_head(p, (cx + sx * 0.2775, yy, 0.04), (sx, 0, 0), 0.008, M["galv"], hexa=True)
        p.r = 0.02
        p.loft([rrect(0.565, 0.58, 0.05, 0.07, cx=cx), rrect(W, D, 0.055, 0.92, cx=cx)], body, cap0=False)
        # door: a panel standing 4 mm proud, hinges at its left, lock at its right
        p.r = 0.003
        p.obox((cx, yf(0.495) - 0.001, 0.495), (0.49, 0.01, 0.73), body, pitch=-th)
        for z in (0.24, 0.74):
            p.cyl((cx - 0.252, yf(z) + 0.01, z - 0.035), (cx - 0.252, yf(z) + 0.01, z + 0.035), 0.009, M["black"], 10)
        p.cyl((cx + 0.205, yf(0.78), 0.78), (cx + 0.205, yf(0.78) + 0.012, 0.78), 0.013, M["galv"], 14)
        p.decal((cx + 0.205, yf(0.78) + 0.012, 0.78), (0, 1, 0), (0, 0, 1), 0.024, 0.024, lock, lift=0.0005)
        nrm, up = (0, math.cos(th), -math.sin(th)), (0, math.sin(th), math.cos(th))
        p.decal((cx, yf(0.62) + 0.0045, 0.62), nrm, up, 0.36, 0.28, _recycle_label(stream), lift=0.0012)
        # moulded panels on the sides and back, a handle recess on the back
        for sx in (-1, 1):
            p.obox((cx + sx * (xh(0.5) - 0.002), -0.005, 0.5), (0.01, 0.5, 0.7), body, roll=sx * ths)
        p.obox((cx, -yf(0.47) + 0.001, 0.47), (0.49, 0.01, 0.66), body, pitch=th)
        for sx in (-1, 1):
            SP.bolt_head(p, (cx + sx * 0.2, -yf(0.82) - 0.004, 0.82), (0, -1, 0), 0.007, M["galv"], hexa=True)
        # hood: back, top, front lip and a raked front with the aperture
        p.r = 0.018
        x0, x1 = cx - 0.3, cx + 0.3
        yb, yfr, zb, zf, zt, yk = -0.306, 0.306, 0.92, 1.0, 1.195, 0.06
        ev = np.array([0, yk - yfr, zt - zf])
        L = float(np.linalg.norm(ev))
        ev /= L
        hole_r = (hu0, hv0, hu0 + hole[0], hv0 + hole[1])
        us = [0.0, hole_r[0], hole_r[2], 0.6]
        vs = [0.0, hole_r[1], hole_r[3], L]
        rk = lambda v: (0.0, yfr + ev[1] * v, zf + ev[2] * v)
        side_r = [(x1, yb, zb), (x1, yfr, zb), (x1, yfr, zf)] + [(x1, rk(v)[1], rk(v)[2]) for v in vs[1:3]] + [(x1, yk, zt), (x1, yb, zt)]
        p.face(side_r, hood_body)
        p.face([(x0, y, z) for _, y, z in side_r][::-1], hood_body)
        p.face([(x1, yb, zb), (x1, yb, zt), (x0, yb, zt), (x0, yb, zb)], hood_body)
        p.face([(x0, yb, zt), (x1, yb, zt), (x1, yk, zt), (x1 - us[1], yk, zt), (x1 - us[2], yk, zt), (x0, yk, zt)], hood_body)
        p.face([(x0, yfr, zb), (x1, yfr, zb), (x1, yfr, zf), (x1 - us[1], yfr, zf), (x1 - us[2], yfr, zf), (x0, yfr, zf)][::-1], hood_body)
        SP.holed_panel(p, (x1, yfr, zf), (-1, 0, 0), ev, 0.6, L, hole_r, 0.08, hood, M["black"], void)
        p.r = 0.0
        o = np.array((x1, yfr, zf))
        if stream == "glass":
            _round_port(p, o, (-1, 0, 0), ev, (0.3, hv0 + 0.1), 0.1, 0.08, M["rubber"], void)
        elif stream == "paper":
            n = np.cross((-1, 0, 0), ev)
            P = lambda u, v, d: tuple(o + np.array((-1, 0, 0)) * u + ev * v - n * d)
            for va, vb in ((hv0, hv0 + 0.0265), (hv0 + 0.055, hv0 + 0.0285)):
                p.face([P(hu0 + 0.002, va, 0.006), P(hu0 + 0.298, va, 0.006), P(hu0 + 0.298, vb, 0.022), P(hu0 + 0.002, vb, 0.022)],
                       M["rubber"])
    # a slim galvanised ganging bar behind the plinths
    p.r = 0.002
    p.box((-0.9, -0.3, 0.0), (0.9, -0.285, 0.04), M["galv"])
    SP.top_uv(p)
    return p.obj("bin_recycling", col, 35)


def bin_wheelie(col, M):
    """A 240 L wheelie bin (1070 x 580 x 740): tapered body with stiffening ribs, a thick rim, the
    front lifting comb and foot pads, a lid with ribs, front grip and a knuckle hinge, the rear
    handle, a moulded axle skirt with wheel arches, a steel axle and 200 mm solid wheels outside the
    body. Embossed capacity mark, council sticker, RFID tag, painted house number. Front +Y."""
    p = PM(0.03)
    base = "#37573f"
    body = SP.skin("TINT_bin_wheelie", base, 1.1, tile=1.0, rough=0.66, grime=0.17, grime_amt=0.55, splash=0.5,
                   scuff=(0.05, 0.45, 0.8), streaks=[(0.95, 0.55, 0.3)], chalk=0.35, peel=0.05, stickers=1, rubber=(0.03, 0.22, 0.5))
    lid = SP.skin("TINT_bin_wheelie_lid", base, 1.1, tile=1.0, px=(512, 512), rough=0.65, grime=0.0, chalk=0.2, peel=0.05,
                  top_fade=(0.98, 0.7), scuff=(0.97, 1.03, 0.6))
    Zb = 0.04
    ring = lambda t, dz=0.0: rrect(0.46 + 0.10 * t, 0.50 + 0.16 * t, 0.06 + 0.01 * t, Zb + 0.93 * t + dz, cy=-0.05 + 0.07 * t)
    p.loft([ring(0.0), ring(0.5), ring(1.0)], body, cap1=False)
    # corner ribs (two at each corner, 15 mm proud) and a framed panel on each side
    p.r = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            for a in (0.35, 1.2):
                pts = []
                for t in (0.05, 0.95):
                    w2, d2, rc, cy = (0.46 + 0.10 * t) / 2, (0.50 + 0.16 * t) / 2, 0.06 + 0.01 * t, -0.05 + 0.07 * t
                    ccx, ccy = sx * (w2 - rc), cy + sy * (d2 - rc)
                    pts.append((ccx + sx * (rc + 0.004) * math.cos(a), ccy + sy * (rc + 0.004) * math.sin(a), Zb + 0.93 * t))
                p.tube(pts, 0.009, body, 6)
        xs = lambda z: sx * (0.23 + (z - Zb) * 0.05 / 0.93 + 0.003)
        frame = [(xs(0.22), -0.17, 0.22), (xs(0.22), 0.11, 0.22), (xs(0.8), 0.11, 0.8), (xs(0.8), -0.17, 0.8)]
        p.tube(frame, 0.006, body, 6, bend=0.03, closed=True)
    p.r = 0.012
    p.loft([rrect(0.60, 0.71, 0.06, z, cy=0.02) for z in (0.95, 0.99)], body)                     # rim lip
    # stacking ledge all round, 10 cm under the rim
    p.r = 0.004
    t0, t1 = (0.85 - Zb) / 0.93, (0.875 - Zb) / 0.93
    p.loft([rrect(0.46 + 0.10 * t0 - 0.002, 0.50 + 0.16 * t0 - 0.002, 0.06, 0.85, cy=-0.05 + 0.07 * t0),
            rrect(0.46 + 0.10 * t0 + 0.036, 0.50 + 0.16 * t0 + 0.036, 0.075, 0.862, cy=-0.05 + 0.07 * t0),
            rrect(0.46 + 0.10 * t1 + 0.036, 0.50 + 0.16 * t1 + 0.036, 0.075, 0.875, cy=-0.05 + 0.07 * t1)], body, cap0=False)
    # lifting comb on the back, under the handle: a full-width lip on vertical webs
    p.box((-0.25, -0.395, 0.875), (0.25, -0.3, 0.888), body)
    p.box((-0.25, -0.405, 0.835), (0.25, -0.392, 0.888), body)
    for x in (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25):
        p.box((x - 0.006, -0.395, 0.84), (x + 0.006, -0.3, 0.877), body)
    # lid: 35 mm edge, a shallow dome, a front grab lip turned down, hinge knuckles at the back
    p.r = 0.01
    p.loft([rrect(0.625, 0.735, 0.05, 0.988, cy=0.02), rrect(0.625, 0.735, 0.05, 1.022, cy=0.02),
            rrect(0.58, 0.69, 0.07, 1.03, cy=0.02), rrect(0.42, 0.52, 0.09, 1.036, cy=0.02)], lid)
    p.r = 0.006
    p.box((-0.2, 0.36, 0.99), (0.2, 0.42, 1.022), lid)
    p.box((-0.2, 0.408, 0.968), (0.2, 0.42, 1.022), lid)
    p.r = 0.003
    for k in range(6):
        x = -0.25 + k * 0.1
        p.cyl((x - 0.035, -0.352, 1.0), (x + 0.035, -0.352, 1.0), 0.019, lid if k % 2 else body, 12)
    # handle across the back on two moulded stays
    p.tube([(-0.25, -0.405, 0.985), (0.25, -0.405, 0.985)], 0.017, body, 10)
    p.r = 0.008
    for s in (-1, 1):
        p.obox((s * 0.25, -0.375, 0.965), (0.045, 0.07, 0.06), body, pitch=0.3)
    # axle tube moulded into the back, arches over the wheels, steel axle, treaded wheels and hubs
    p.r = 0.006
    p.cyl((-0.215, -0.31, 0.1), (0.215, -0.31, 0.1), 0.045, body, 16)
    p.box((-0.2, -0.335, 0.1), (0.2, -0.27, 0.24), body)
    for s in (-1, 1):
        arc = [(s * 0.258, -0.31 + 0.125 * math.cos(a), 0.10 + 0.125 * math.sin(a)) for a in np.linspace(math.radians(10), math.radians(170), 9)]
        p.r = 0.003
        SP.sweep(p, arc, SP._rr2(0.075, 0.012, 0.004), body, normal=(1, 0, 0))
    p.r = 0.0
    p.cyl((-0.3, -0.31, 0.1), (0.3, -0.31, 0.1), 0.01, M["galv"], 10)
    for s in (-1, 1):
        tyre(p, (s * 0.26, -0.31, 0.1), 0.1, 0.052, 0.062, M["rubber"], M["grey"], sides=20, lugs=14, lug_h=0.003)
        p.cyl((s * 0.287, -0.31, 0.1), (s * 0.296, -0.31, 0.1), 0.02, M["grey"], 10)
        p.cyl((s * 0.296, -0.31, 0.1), (s * 0.3, -0.31, 0.1), 0.012, M["galv"], 8)
    # front feet: the bin stands on them and the wheels
    p.r = 0.008
    for s in (-1, 1):
        p.box((s * 0.17 - 0.05, 0.1, 0.0), (s * 0.17 + 0.05, 0.2, 0.048), body)
    # the front face (tilted back as it rises): embossed mark, sticker, RFID tag, house number
    th = math.atan(0.15 / 0.93)
    yf = lambda z: 0.20 + (z - 0.04) * 0.15 / 0.93
    nrm, up = (0, math.cos(th), -math.sin(th)), (0, math.sin(th), math.cos(th))

    def emboss(c):
        w, h = c.W / c.k, c.H / c.k
        m = c.text("240L", w / 2, h * 0.52, 0.032, base, a=1.0)
        m2 = c.text("EN 840-1", w / 2, h * 0.18, 0.016, base, bold=False)
        c.hgt = SP.blur((m + m2) * 3.0, 2)
        c.alb = SP.mixc(c.alb, SP.lin("#3d4a3d"), np.clip(m + m2, 0, 1) * 0.35)
    p.decal((0.0, yf(0.30), 0.30), nrm, up, 0.16, 0.08, SP.card("Street_WheelieEmboss", 0.16, 0.08, emboss, ppm=3000, bg=base, rough=0.55),
            lift=0.0008)

    def sticker(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0, h - 0.035, w, h), "#1b5e3b")
        c.text("NORTHFIELD BOROUGH COUNCIL", w / 2, h - 0.025, 0.011, "#ffffff", max_w=w - 0.02)
        c.text("PROPERTY OF THE COUNCIL", w / 2, h - 0.06, 0.012, "#222222", max_w=w - 0.02)
        c.text("General waste only", w / 2, h - 0.085, 0.01, "#222222", bold=False)
        c.text("Collected fortnightly - lid closed", w / 2, h - 0.104, 0.008, "#444444", bold=False, max_w=w - 0.02)
        c.age(0.25, 0.3)
        c.grime(0.25)
    p.decal((0.0, yf(0.62), 0.62), nrm, up, 0.2, 0.12, SP.card("Street_WheelieSticker", 0.2, 0.12, sticker, ppm=3500), lift=0.001)

    def rfid(c):
        w, h = c.W / c.k, c.H / c.k
        for k in range(28):
            x = 0.006 + k * (w - 0.012) / 28
            if (k * 7) % 5 != 1:
                c.poly(SP.rect(x, 0.008, x + (0.0015 if k % 3 else 0.0028), h - 0.012), "#111111")
        c.text("0047 2291 18", w / 2, 0.0015, 0.004, "#111111", bold=False)
        c.age(0.2, 0.2)
    p.decal((0.17, yf(0.82), 0.82), nrm, up, 0.07, 0.035, SP.card("Street_WheelieTag", 0.07, 0.035, rfid, ppm=6000), lift=0.001)

    def number(c):
        w, h = c.W / c.k, c.H / c.k
        m = c.text("27", w / 2, h * 0.12, h * 0.7, "#e9e7df", a=0.0)
        brush = SP.sstep(0.3, 0.6, SP.noise((c.H, c.W), (30, 4), c.r, 3))
        a = np.clip(m * (0.75 + 0.25 * brush), 0, 1)
        c.fill(a, "#e9e7df")
    numm = SP.card("Street_WheelieNumber", 0.16, 0.14, number, ppm=2500, bg="#e9e7df", alpha=True, rough=0.6)
    # on the left side face (tilted outwards as it rises)
    sth = math.atan(0.05 / 0.93)
    p.decal((-(0.23 + (0.55 - 0.04) * 0.05 / 0.93) - 0.001, -0.03, 0.55), (-math.cos(sth), 0, -math.sin(sth)), (-math.sin(sth), 0, math.cos(sth)),
            0.16, 0.14, numm, lift=0.001)

    def moulded(c):
        w, h = c.W / c.k, c.H / c.k
        m = np.zeros((c.H, c.W))
        for part in SP.picto_recycle(w / 2, h * 0.62, 0.11):
            m = np.maximum(m, SP.poly_mask(c.H, c.W, [[(x * c.k, y * c.k) for x, y in part]]))
        m = np.maximum(m, SP.text_mask(c.H, c.W, "240L", w / 2 * c.k, 0.02 * c.k, 0.03 * c.k))
        c.hgt = SP.blur(m * 4.0, 2)
        c.alb = SP.mixc(c.alb, c.alb * 1.15, m * 0.5)
        c.alb *= (0.95 + 0.1 * SP.noise((c.H, c.W), (8, 8), c.r, 3))[..., None]
    p.decal(((0.23 + (0.5 - 0.04) * 0.05 / 0.93) + 0.001, -0.03, 0.5), (math.cos(sth), 0, -math.sin(sth)), (math.sin(sth), 0, math.cos(sth)),
            0.15, 0.17, SP.card("Street_WheelieMoulded", 0.15, 0.17, moulded, ppm=2500, bg=base, rough=0.62), lift=0.0008)
    SP.top_uv(p)
    return p.obj("bin_wheelie", col, 30)


def bin_container(col, M):
    """An 1100 L galvanised Eurobin (1260 x 990 x 1370): chamfer-cornered tapered body with two
    swaged ribs and a rolled rim, a domed HDPE lid with a down-turned lip, front grips and a hinge
    bar, welded angle base frame on four 200 mm swivel castors (two braked), lifting trunnions
    on backing plates on the short sides, a drain plug, a stencilled number. Front +Y (long side)."""
    p = PM(0.012)

    galv = SP.skin("Street_EurobinGalv", "#9a9c9f", 1.4, tile=1.0, rough=0.48, metal=0.4, grime=0.28, grime_col="#5d564a",
                   grime_amt=0.6, splash=0.5, scuff=(0.28, 0.6, 0.5), spangle=0.45, mottle=0.0,
                   peel=0.04, normal_strength=0.8, dents=3, streaks=[(1.2, 0.45, 0.4), (1.19, 0.35, 0.6)],
                   rust_runs=[(1.19, 0.35)])
    lid = SP.skin("TINT_bin_container", "#2f5e3a", 1.4, tile=1.0, px=(512, 512), rough=0.66, grime=0.0, chalk=0.3,
                  streaks=[(1.36, 0.3, 0.06)], peel=0.05, scuff=(1.17, 1.4, 0.7), top_fade=(1.27, 0.65))
    zb, zt = 0.28, 1.235
    ring = lambda w, d, z, ch=0.07: rrect(w, d, ch, z, n=1)
    p.loft([ring(1.06, 0.80, zb), ring(1.22, 0.95, zt - 0.03)], galv)
    # pressed ribs: three beads 25 mm proud round all four faces
    p.r = 0.008
    wd = lambda z: (1.06 + 0.16 * (z - zb) / (zt - 0.03 - zb), 0.80 + 0.15 * (z - zb) / (zt - 0.03 - zb))
    for z in (0.5, 0.75, 1.0):
        (w0, d0), (w1, d1) = wd(z - 0.03), wd(z + 0.03)
        p.loft([ring(w0 - 0.002, d0 - 0.002, z - 0.03), ring(w0 + 0.05, d0 + 0.05, z - 0.014), ring(w1 + 0.05, d1 + 0.05, z + 0.014),
                ring(w1 - 0.002, d1 - 0.002, z + 0.03)], galv, cap0=False, cap1=False)
    # rolled rim, 40 mm
    p.r = 0.014
    p.loft([ring(1.215, 0.945, zt - 0.045), ring(1.27, 1.0, zt - 0.032), ring(1.27, 1.0, zt - 0.006), ring(1.225, 0.955, zt)], galv,
           cap0=False)
    # push handles on both short ends, a chassis plate under the body
    p.r = 0.0
    for sx in (-1, 1):
        xe = wd(1.1)[0] / 2
        p.tube([(sx * (xe - 0.01), -0.24, 1.1), (sx * (xe + 0.075), -0.24, 1.1), (sx * (xe + 0.075), 0.24, 1.1), (sx * (xe - 0.01), 0.24, 1.1)],
               0.015, M["galv"], 12, bend=0.045)
    p.r = 0.003
    p.box((-0.5, -0.37, zb - 0.006), (0.5, 0.37, zb), M["galv"])
    # lid: down-turned lip and a shallow dome
    p.r = 0.02
    p.loft([rrect(1.30, 1.03, 0.06, zt - 0.06), rrect(1.30, 1.03, 0.06, zt + 0.03), rrect(1.27, 1.0, 0.07, zt + 0.05),
            rrect(1.12, 0.86, 0.12, zt + 0.095), rrect(0.8, 0.55, 0.15, zt + 0.125)], lid)
    # moulded hand grips under the front lip, shallow strengthening ribs and an embossed logo on top
    p.r = 0.006
    for s in (-1, 1):
        p.box((s * 0.33 - 0.1, 0.5, zt - 0.075), (s * 0.33 + 0.1, 0.545, zt - 0.055), lid)
        for dx in (-0.1, 0.1):
            p.box((s * 0.33 + dx - 0.012, 0.5, zt - 0.075), (s * 0.33 + dx + 0.012, 0.545, zt - 0.03), lid)
    p.r = 0.004
    for y in (-0.21, -0.07, 0.07, 0.21):
        p.box((-0.36, y - 0.012, zt + 0.118), (0.36, y + 0.012, zt + 0.132), lid)

    def logo(c):
        w = c.W / c.k
        m = np.zeros((c.H, c.W))
        for part in SP.picto_recycle(w / 2, w / 2, w * 0.9):
            m = np.maximum(m, SP.poly_mask(c.H, c.W, [[(x * c.k, y * c.k) for x, y in part]]))
        c.hgt = SP.blur(m * 4.0, 2)
        c.alb = SP.mixc(c.alb, c.alb * 1.12, m * 0.6)
    p.decal((0.47, 0.0, zt + 0.1255), (0, 0, 1), (0, 1, 0), 0.14, 0.14,
            SP.card("Street_EurobinLogo", 0.14, 0.14, logo, ppm=2500, bg="#3a6a45", rough=0.66), lift=0.001)
    p.r = 0.0
    p.cyl((-0.6, -0.525, zt - 0.005), (0.6, -0.525, zt - 0.005), 0.0125, M["galv"], 10)
    for x in (-0.48, -0.16, 0.16, 0.48):
        p.cyl((x - 0.04, -0.525, zt - 0.005), (x + 0.04, -0.525, zt - 0.005), 0.022, galv if x in (-0.48, 0.16) else lid, 12)
    # trunnions on the short sides at two-thirds height
    zt2 = 0.96
    t = (zt2 - zb) / (zt - 0.03 - zb)
    xs = (1.06 + 0.16 * t) / 2
    p.r = 0.004
    for s in (-1, 1):
        p.box((s * xs - 0.006, -0.075, zt2 - 0.075), (s * xs + 0.006, 0.075, zt2 + 0.075), galv)
        p.cyl((s * (xs + 0.006), 0, zt2), (s * (xs + 0.1), 0, zt2), 0.025, galv, 16)
        p.cyl((s * (xs + 0.09), 0, zt2), (s * (xs + 0.105), 0, zt2), 0.045, galv, 16)
        for dy in (-0.055, 0.055):
            for dz in (-0.055, 0.055):
                SP.bolt_head(p, (s * (xs + 0.006), dy, zt2 + dz), (s, 0, 0), 0.008, M["galv"], hexa=True)
    # drain plug low on the front
    yfront = lambda z: 0.40 + (z - zb) * 0.075 / (zt - 0.03 - zb)
    SP.bolt_head(p, (0.35, yfront(0.36), 0.36), (0, 1, 0), 0.022, galv, hexa=True)
    # base frame (welded angle) and four castors
    p.r = 0.003
    for y in (-0.37, 0.37):
        p.box((-0.52, y - 0.02, zb - 0.04), (0.52, y + 0.02, zb), galv)
    for x in (-0.5, 0.5):
        p.box((x - 0.02, -0.37, zb - 0.04), (x + 0.02, 0.37, zb), galv)
    for x in (-0.42, 0.42):
        for y in (-0.31, 0.31):
            p.box((x - 0.065, y - 0.065, zb - 0.05), (x + 0.065, y + 0.065, zb - 0.04), M["galv"])
            p.cyl((x, y, zb - 0.075), (x, y, zb - 0.05), 0.03, M["galv"], 12)
            p.r = 0.0
            yo = y - 0.04 * (1 if y > 0 else -1)
            ya, yb2 = min(y, yo) - 0.035, max(y, yo) + 0.035
            for sx in (-1, 1):
                p.box((x + sx * 0.032 - 0.003, ya, 0.085), (x + sx * 0.032 + 0.003, yb2, zb - 0.075), M["galv"])
            p.box((x - 0.035, ya, zb - 0.085), (x + 0.035, yb2, zb - 0.075), M["galv"])
            tyre(p, (x, yo, 0.1), 0.1, 0.045, 0.055, M["rubber"], M["grey"], sides=14)
            p.cyl((x - 0.035, yo, 0.1), (x + 0.035, yo, 0.1), 0.008, M["galv"], 8)
            if y > 0:
                p.box((x - 0.03, y + 0.03, zb - 0.1), (x + 0.03, y + 0.075, zb - 0.088), M["red"])
    # stencilled number (sprayed, soft-edged) on the front
    def stencil(c):
        w, h = c.W / c.k, c.H / c.k
        m = c.text("B12", w / 2, 0.02, h * 0.62, "#efefea", a=0.0)
        soft = SP.blur(m, 2)
        spray = np.clip(soft * (0.8 + 0.35 * SP.noise((c.H, c.W), (40, 20), c.r, 2)) - 0.05, 0, 1)
        c.fill(spray * (0.65 + 0.35 * SP.noise((c.H, c.W), (5, 3), c.r, 3)), "#efefea")
    th = math.atan(0.075 / (zt - 0.03 - zb))

    def hot_ash(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0, 0, w, h, 0.004), "#c8102e")
        c.poly(SP.rect(0.004, 0.004, w - 0.004, h - 0.004, 0.003), "#f3f1ea")
        c.text("NO HOT ASH", w / 2, h * 0.55, h * 0.2, "#c8102e", max_w=w - 0.012)
        c.text("Fire risk - let ashes cool", w / 2, h * 0.2, h * 0.1, "#333333", bold=False, max_w=w - 0.012)
        c.age(0.3, 0.25)
        c.grime(0.25)

    def collect(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0, h * 0.7, w, h), "#1b5e3b")
        c.text("COLLECTION", w / 2, h * 0.77, h * 0.13, "#ffffff", max_w=w - 0.01)
        c.text("TUESDAY", w / 2, h * 0.38, h * 0.18, "#222222", max_w=w - 0.01)
        c.text("Northfield Borough Council", w / 2, h * 0.1, h * 0.08, "#444444", bold=False, max_w=w - 0.01)
        c.age(0.35, 0.3)
        c.grime(0.3)
    for x, z, w_, h_, nm, fn in ((0.22, 0.62, 0.16, 0.09, "Street_NoHotAsh", hot_ash), (0.42, 0.9, 0.1, 0.1, "Street_Collection", collect)):
        p.decal((x, yfront(z), z), (0, math.cos(th), -math.sin(th)), (0, math.sin(th), math.cos(th)), w_, h_,
                SP.card(nm, w_, h_, fn, ppm=3500), lift=0.001)

    def lid_stencil(c):
        w, h = c.W / c.k, c.H / c.k
        m = c.text("NORTHFIELD BC", w / 2, h * 0.22, h * 0.55, "#efefea", a=0.0, max_w=w - 0.02)
        spray = np.clip(SP.blur(m, 2) * (0.75 + 0.35 * SP.noise((c.H, c.W), (40, 8), c.r, 2)) - 0.08, 0, 1)
        c.fill(spray * (0.5 + 0.3 * SP.noise((c.H, c.W), (6, 2), c.r, 3)), "#efefea")
    p.decal((0, 0.05, zt + 0.1255), (0, 0, 1), (0, 1, 0), 0.5, 0.08,
            SP.card("Street_EurobinLidStencil", 0.5, 0.08, lid_stencil, ppm=2000, bg="#efefea", alpha=True, rough=0.65), lift=0.001)
    p.decal((-0.18, yfront(0.87), 0.87), (0, math.cos(th), -math.sin(th)), (0, math.sin(th), math.cos(th)), 0.32, 0.16,
            SP.card("Street_EurobinStencil", 0.32, 0.16, stencil, ppm=2000, bg="#efefea", alpha=True, rough=0.6), lift=0.001)
    SP.top_uv(p)
    return p.obj("bin_container", col, 30)


# -- seating and tables ---------------------------------------------------------------------------

def _hardwood():
    return (SP.timber("Street_Hardwood", base="#8a6e53", grey="#8f8b82", weather=0.35, green=0.2, even=True, vary=0.08, knots=(0, 2)),
            SP.end_grain("Street_HardwoodEnd", "#6e4e33"))


def _strap(p, x, pts, w, t, mat, n=3, bead=False):
    """A flat cast member: a centre line in the (y, z) plane, `w` wide in the plane (a number or
    (start, end)), `t` thick across x, extruded as one flat section."""
    C = SP.spline([(0.0, y, z) for y, z in pts], n)
    C = [np.array((c[1], c[2])) for c in C]
    w0, w1 = (w, w) if not isinstance(w, tuple) else w
    left, right = [], []
    for k, c in enumerate(C):
        d = C[min(k + 1, len(C) - 1)] - C[max(k - 1, 0)]
        d /= np.linalg.norm(d)
        nrm = np.array((-d[1], d[0]))
        hw = (w0 + (w1 - w0) * k / (len(C) - 1)) / 2
        left.append(tuple(c + nrm * hw))
        right.append(tuple(c - nrm * hw))
    poly = left + right[::-1]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
    if area < 0:
        poly = poly[::-1]
    p.extrude_yz(poly, x - t / 2, x + t / 2, mat)
    if bead:                                                    # a raised rib down the middle of both faces
        saved, p.r = p.r, 0.0
        _strap(p, x, pts, (w0 * 0.32, w1 * 0.32), t + 0.012, mat, n)
        p.r = saved


def _bench_end(p, x, frame, arms=True):
    """One cast-iron end frame, 25 mm thick: splayed legs on paw feet, the back leg sweeping up
    into the reclined back support with a scroll at its head, a seat rail, a stretcher, corner
    webs, and (on the ends) an armrest with a wide top and a scrolled front."""
    rail = lambda y: 0.425 + (y - 0.255) * 0.02 / 0.445
    T = 0.025
    p.r = 0.003
    _strap(p, x, [(0.31, 0.03), (0.287, 0.2), (0.263, 0.41), (0.268, 0.53), (0.29, 0.612)] if arms else
           [(0.31, 0.03), (0.287, 0.2), (0.263, 0.41)], (0.056, 0.042), T, frame, bead=True)
    _strap(p, x, [(-0.31, 0.03), (-0.272, 0.2), (-0.214, 0.39), (-0.226, 0.5), (-0.2725, 0.68), (-0.315, 0.84), (-0.322, 0.878)],
           (0.056, 0.04), T, frame, bead=True)
    _strap(p, x, [(0.295, rail(0.295) - 0.022), (0.03, rail(0.03) - 0.027), (-0.235, rail(-0.235) - 0.022)], 0.044, T, frame, bead=True)
    _strap(p, x, [(0.295, 0.13), (0.0, 0.172), (-0.28, 0.13)], 0.034, T * 0.8, frame)
    for poly in ([(0.262, 0.385), (0.16, 0.385), (0.262, 0.28)], [(-0.214, 0.37), (-0.11, 0.375), (-0.232, 0.27)],
                 [(0.29, 0.13), (0.2, 0.152), (0.285, 0.205)], [(-0.27, 0.205), (-0.19, 0.155), (-0.28, 0.13)]):
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
        p.extrude_yz(poly if area > 0 else poly[::-1], x - T * 0.3, x + T * 0.3, frame)
    # head scroll of the back support
    p.cyl((x - T / 2, -0.318, 0.892), (x + T / 2, -0.318, 0.892), 0.03, frame, 20)
    p.cyl((x - T / 2 - 0.004, -0.318, 0.892), (x + T / 2 + 0.004, -0.318, 0.892), 0.014, frame, 14)
    # paw feet
    for y0, sg in ((0.31, 1), (-0.31, -1)):
        foot = [(y0 - 0.055, 0.0), (y0 + 0.06, 0.0), (y0 + 0.05, 0.022), (y0 + 0.025, 0.04), (y0 - 0.02, 0.045), (y0 - 0.048, 0.025)]
        if sg < 0:
            foot = [(2 * y0 - y, z) for y, z in foot][::-1]
        p.extrude_yz(foot, x - 0.03, x + 0.03, frame)
    if arms:
        _strap(p, x, [(-0.235, 0.655), (0.0, 0.665), (0.2, 0.658), (0.285, 0.642)], 0.048, T, frame)
        _strap(p, x, [(-0.225, 0.678), (0.0, 0.688), (0.2, 0.68), (0.28, 0.664)], 0.016, 0.062, frame)      # wide arm top
        p.cyl((x - T / 2, 0.3, 0.618), (x + T / 2, 0.3, 0.618), 0.036, frame, 22)
        p.cyl((x - T / 2 - 0.005, 0.3, 0.618), (x + T / 2 + 0.005, 0.3, 0.618), 0.017, frame, 14)


def bench(col, M):
    """A 1.8 m park bench facing +Y on three cast-iron frames (flat 25 mm sections: paw-footed
    splayed legs, the back leg sweeping up into a scrolled back support, corner webs, scrolled
    armrests with wide tops on the two ends), five hardwood seat slats (the front one with a
    rounded nosing) and four back slats, all chamfered and carriage-bolted, and a brass memorial
    plaque tarnished at its edges."""
    p = PM(0.004)
    frame = SP.skin("Street_BenchFrame", "#1c2721", 1.0, tile=1.0, rough=0.42, grime=0.08, grime_amt=0.4, rust=0.15, mottle=0.03,
                    chalk=0.2, peel=0.12, streaks=[(0.42, 0.3, 0.12)], primer=(0.6, 0.72, 0.6), rust_runs=[(0.62, 0.35)],
                    wear_bands=[(0.0, 0.05, 0.35, "#6e3c22")])
    wood, end = _hardwood()
    r = SP.rng("bench")
    rail = lambda y: 0.425 + (y - 0.255) * 0.02 / 0.445         # seat rail top, front higher
    for x in (-0.8, 0.8):
        _bench_end(p, x, frame)
    _bench_end(p, 0.0, frame, arms=False)
    # seat slats
    for i in range(5):
        y = 0.245 - i * 0.084
        z = rail(y) + 0.016
        up = (0, -0.045, 1.0)
        if i == 0:                                              # waterfall front slat
            y, z, up = y + 0.004, z - 0.012, (0, 0.38, 1.0)
        p.r = 0.014 if i == 0 else 0.006
        SP.board(p, (-0.9, y, z), (0.9, y, z), up, 0.068, 0.032, wood, end, strip=i, flip=bool(r.integers(2)))
        for x in (-0.8, 0.0, 0.8):
            SP.bolt_head(p, (x, y, z + 0.016), up, 0.009, M["galv"], sides=8)
    # back slats on the reclined back supports
    p.r = 0.005
    P0, P1 = np.array([-0.226, 0.5]), np.array([-0.315, 0.84])
    tb = (P1 - P0) / np.linalg.norm(P1 - P0)
    nb = np.array([tb[1], -tb[0]])
    if nb[0] < 0:
        nb = -nb
    for i in range(4):
        s = 0.045 + i * 0.087
        c = P0 + tb * s + nb * (0.02 + 0.013)
        up = (0, nb[0], nb[1])
        SP.board(p, (-0.9, c[0], c[1]), (0.9, c[0], c[1]), up, 0.07, 0.026, wood, end, strip=5 + i % 3, flip=bool(r.integers(2)))
        for x in (-0.8, 0.0, 0.8):
            SP.bolt_head(p, (x, c[0] + nb[0] * 0.013, c[1] + nb[1] * 0.013), up, 0.009, M["galv"], sides=8)
    # memorial plaque on the top back slat
    s = 0.045 + 3 * 0.087
    c = P0 + tb * s + nb * (0.033 + 0.013)

    def plaque(cv):
        w, h = cv.W / cv.k, cv.H / cv.k
        cv.alb *= (0.85 + 0.3 * SP.noise((cv.H, cv.W), (12, 4), cv.r, 3))[..., None]
        cv.poly(SP.rect(0.004, 0.004, w - 0.004, h - 0.004, 0.004), "#6f5520", 0.0)
        cv.text("IN LOVING MEMORY OF", w / 2, h - 0.017, 0.0072, "#1e180a", bold=False)
        cv.text("MARGARET & TOM ELLIS", w / 2, h - 0.036, 0.0115, "#1e180a")
        cv.text("WHO LOVED THIS PLACE", w / 2, 0.01, 0.0072, "#1e180a", bold=False)
        X = (np.arange(cv.W) + 0.5)[None, :] / cv.k * np.ones((cv.H, 1))
        Y = (np.arange(cv.H) + 0.5)[:, None] / cv.k * np.ones((1, cv.W))
        edge = np.minimum(np.minimum(X, w - X), np.minimum(Y, h - Y))
        tarnish = (1 - SP.sstep(0.0, 0.009, edge)) * (0.6 + 0.4 * SP.noise((cv.H, cv.W), (20, 6), cv.r, 3))
        cv.alb = SP.mixc(cv.alb, SP.lin("#4f5a3a"), tarnish * 0.75)
        for sx in (0.008, w - 0.008):
            cv.poly(SP.circle(sx, h / 2, 0.003, 12), "#8a7a52")
        cv.grime(0.2, "#3b2f18", 5)
        dark = np.clip(SP.lin("#a88a47")[1] - cv.alb[..., 1], 0, 1)
        cv.hgt = -SP.blur(dark * 6.0, 1)
        cv.rough = 0.42 + 0.25 * tarnish + 0.1 * SP.noise((cv.H, cv.W), (8, 3), cv.r, 3)
    p.r = 0.001
    up = np.array([0, nb[0], nb[1]])
    v = np.array([0, tb[0], tb[1]])
    cc = np.array([0, c[0], c[1]])
    p.box((-0.1, 0, 0), (0.1, 0.003, 0.06), M["alu"], xf=Matrix(
        [[1, 0, 0, 0], [0, up[1], v[1], cc[1] - up[1] * 0.0015 - v[1] * 0.03], [0, up[2], v[2], cc[2] - up[2] * 0.0015 - v[2] * 0.03], [0, 0, 0, 1]]))
    p.decal(tuple(cc + up * 0.0015), tuple(up), tuple(v), 0.2, 0.06,
            SP.card("Street_BenchPlaque", 0.2, 0.06, plaque, ppm=5000, bg="#a88a47", metal=0.9, rough=0.45), lift=0.0006)
    SP.top_uv(p)
    return p.obj("bench", col, 40)


def picnic_table(col, M):
    """A 1.8 m pub-garden picnic table in tanalised pine: five top boards and two seat boards a
    side with 7 mm gaps, A-frame splayed legs, top and seat bearers, diagonal braces, coach bolts
    with domed heads outside and nuts and washers inside, screws in the boards."""
    p = PM(0.006)
    wood = SP.timber("Street_Pine", base="#7a6748", grey="#858074", weather=0.4, green=0.55, vary=0.13, knots=(2, 4), ground_strips=(6, 7))
    end = SP.end_grain("Street_PineEnd", "#8a7650")
    r = SP.rng("picnic")
    galv = M["galv"]
    L = 1.8
    k = 0
    # top: 5 x 145 x 34, 7 mm gaps
    for i in range(5):
        y = (i - 2) * 0.155
        dz, dx = r.uniform(-0.0015, 0.0015), r.uniform(-0.008, 0.008)
        z = 0.723 + dz
        SP.board(p, (-L / 2 + dx, y, z), (L / 2 + dx + r.uniform(-0.006, 0.006), y, z), (0, r.uniform(-0.01, 0.01), 1), 0.145, 0.034,
                 wood, end, strip=k % 6, flip=bool(r.integers(2)))
        k += 1
        for x in (-0.62, 0.62):
            for sy in (-0.04, 0.04):
                p.r = 0.0
                p.cyl((x, y + sy, z + 0.016), (x, y + sy, z + 0.0185), 0.0045, galv, 8)
                p.r = 0.006
    # seats: 2 x 145 a side at 440
    for s in (-1, 1):
        for j in range(2):
            y = s * (0.535 + j * 0.153)
            z = 0.423 + r.uniform(-0.0015, 0.0015)
            dx = r.uniform(-0.01, 0.01)
            SP.board(p, (-L / 2 + dx, y, z), (L / 2 + dx, y, z), (0, r.uniform(-0.01, 0.01), 1), 0.145, 0.034, wood, end, strip=k % 6,
                     flip=bool(r.integers(2)))
            k += 1
            for x in (-0.62, 0.62):
                for sy in (-0.04, 0.04):
                    p.r = 0.0
                    p.cyl((x, y + sy, z + 0.016), (x, y + sy, z + 0.0185), 0.0045, galv, 8)
                    p.r = 0.006
    for x in (-0.62, 0.62):
        sg = 1 if x > 0 else -1
        xb = x - sg * 0.045                      # bearers inboard of the legs
        SP.board(p, (xb, -0.36, 0.658), (xb, 0.36, 0.658), (1, 0, 0), 0.095, 0.045, wood, end, strip=k % 6)
        SP.board(p, (xb, -0.79, 0.358), (xb, 0.79, 0.358), (1, 0, 0), 0.095, 0.045, wood, end, strip=(k + 1) % 6)
        # A-frame legs: from under the top to the ground, splayed sideways, feet cut below grade
        for s in (-1, 1):
            a = np.array([x, s * 0.27, 0.70])
            b = np.array([x, s * 0.80, 0.0])
            d = (b - a) / np.linalg.norm(b - a)
            SP.board(p, tuple(a), tuple(b), (1, 0, 0), 0.095, 0.045, wood, end, strip=6 + (s > 0), floor=0.0)
            # coach bolts through leg and bearers: dome outside, nut and washer inside
            for zz in (0.658, 0.358):
                t = (0.70 - zz) / 0.70
                yl = s * (0.27 + 0.53 * t)
                SP.bolt_head(p, (x + sg * 0.0225, yl, zz), (sg, 0, 0), 0.012, galv)
                p.r = 0.0
                p.box((xb - sg * 0.0225 - 0.002, yl - 0.017, zz - 0.017), (xb - sg * 0.0225 + 0.002, yl + 0.017, zz + 0.017), galv)
                SP.bolt_head(p, (xb - sg * 0.0245, yl, zz), (-sg, 0, 0), 0.0095, galv, hexa=True)
                p.r = 0.006
        k += 4
        # brace from the seat bearer up to the centre batten under the top
        a = np.array([xb - sg * 0.02, 0.0, 0.40])
        b = np.array([x - sg * 0.47, 0.0, 0.685])
        SP.board(p, tuple(a), tuple(b), (0, 1, 0), 0.07, 0.045, wood, end, strip=k % 6)
        for q in (a + (b - a) * 0.12, a + (b - a) * 0.93):
            for sy in (-1, 1):
                SP.bolt_head(p, (q[0], sy * 0.0225, q[2]), (0, sy, 0), 0.011, galv, sides=8)
        k += 1
    # a centre batten under the top tying the braces
    SP.board(p, (-0.24, 0, 0.687), (0.24, 0, 0.687), (0, 0, 1), 0.07, 0.035, wood, end, strip=k % 6)
    SP.top_uv(p)
    return p.obj("picnic_table", col, 40)


# -- traffic and access ---------------------------------------------------------------------------

def bollard(col, M):
    """A 1 m steel street bollard, 140 mm, core-set in a mortar collar 15 mm proud with a dished
    fillet: a cap with a 5 mm casting lip, a weld bead and a cast top disc, two 75 mm micro-prism
    reflective bands 50 mm apart, powder coat with one diagonal bumper scrape at 450-600 mm (bare
    steel, a smear of the other car's paint, short rust tears), damp dirt at the foot."""
    p = PM(0.004)
    R, Z0 = 0.07, 0.015

    def scrape(alb, rgh, hgt, X, Z, r):
        u = X - 0.12
        t = (Z - 0.47) - u * 0.35                       # rising across the front
        band = (np.abs(t) < 0.022 + 0.01 * SP.noise(X.shape, (6, 30), r, 3)) * (np.abs(u) < 0.11) * (Z > 0.42) * (Z < 0.64)
        streaky = SP.sstep(0.35, 0.6, SP.noise(X.shape, (60, 6), r, 3))
        bare = np.clip(band * (0.5 + 0.5 * streaky), 0, 1)
        alb[...] = SP.mixc(alb, SP.lin("#8c8e8c"), bare)
        smear = band * SP.sstep(0.55, 0.75, SP.noise(X.shape, (30, 6), r, 3)) * (t > 0)
        alb[...] = SP.mixc(alb, SP.lin("#c9cbc8"), smear * 0.8)
        rgh[...] = rgh - 0.15 * bare
        hgt -= bare * 1.5
        for x0 in (0.06, 0.12, 0.17):
            w = 0.004
            z_top = 0.47 + (x0 - 0.12) * 0.35 - 0.015
            run = (np.abs(X - x0) < w) * (Z < z_top) * np.exp(-np.clip(z_top - Z, 0, None) / 0.05) * (1 - np.abs(X - x0) / w)
            alb[...] = SP.mixc(alb, SP.lin("#7a5236"), np.clip(run * 0.8, 0, 1))
    paint = SP.skin("Street_Bollard", "#36383b", 1.05, tile=math.pi * R, px=(256, 1024), rough=0.5, grime=0.1, grime_amt=0.7,
                    grime_col="#4f493d", splash=0.5, streaks=[(0.93, 0.35, 0.12)], rust_runs=[(0.93, 0.45)], chalk=0.25, peel=0.04,
                    rust=0.15, wear_bands=[(0.955, 1.01, 0.4, "#6f7072")], paint=scrape)
    tape = _tape("Street_TapeWhite85", "#ededE6".lower(), 0.1)
    tape_lo = _tape("Street_TapeWhite85Dirty", "#e2e0d6", 0.35)
    mortar = SP.concrete_top("Street_MortarTop", 0.5, base="#8a867d")
    p.r = 0.004
    SP.lathe(p, [(0.11, -0.01), (0.11, 0.006), (0.1, 0.012), (0.08, 0.014), (0.0745, Z0)], mortar, sides=40, uv_xy=True)
    p.r = 0.003
    SP.lathe(p, [(0.0745, Z0 - 0.002), (0.0745, Z0 + 0.012), (R, Z0 + 0.018), (R, 0.925), (0.0712, 0.928), (0.0712, 0.932), (R + 0.005, 0.934),
                 (R + 0.005, 0.95), (0.0715, 0.956), (0.066, 0.975), (0.05, 0.992), (0.026, 1.0), (0.0, 1.002)], paint, sides=40, uv_r=R)

    def mark(c):
        w = c.W / c.k
        ring = c.poly(SP.circle(w / 2, w / 2, w * 0.47), "#36383b")
        hole = c.poly(SP.circle(w / 2, w / 2, w * 0.41), "#36383b")
        t = c.text("BS 7263", w / 2, w * 0.4, w * 0.12, "#36383b")
        c.hgt = SP.blur((ring - hole + t) * 2.0, 1)
        c.alb *= (0.9 + 0.2 * SP.noise((c.H, c.W), (6, 6), c.r, 3))[..., None]
    p.decal((0, 0, 1.0025), (0, 0, 1), (0, 1, 0), 0.05, 0.05, SP.card("Street_BollardMark", 0.05, 0.05, mark, ppm=8000, bg="#36383b",
                                                                     rough=0.45), lift=0.0)
    for z0, tm in ((0.755, tape), (0.63, tape_lo)):
        p.r = 0.0003
        SP.lathe(p, [(R, z0), (R + 0.0006, z0 + 0.0004), (R + 0.0006, z0 + 0.0746), (R, z0 + 0.075)], tm, sides=40, uv_r=R)
        SP.lathe(p, [(R + 0.0006, z0), (R + 0.0012, z0 + 0.0004), (R + 0.0012, z0 + 0.0746), (R + 0.0006, z0 + 0.075)], tm, sides=40,
                 uv_r=R, skip=lambda j, k: k not in (0, 1))
    SP.top_uv(p)
    return p.obj("bollard", col, 40)


def barrier_arm(col, M):
    """A rising-arm car-park barrier (FAAC 620 class): a 350 x 260 x 1080 cabinet bolted to a
    concrete plinth, a domed overhanging cap with an amber LED beacon, a front inspection door
    (shut lines, lock barrel, vent slots, maker plate), a side louvre, a warning label; a 120 mm hub
    boss carrying a cast boom holder that wraps the boom end on four bolts; a 4 m extruded
    aluminium boom (100 x 55, rubber anti-impact strip, red retro-reflective bands, end cap); a
    galvanised rest post on its own pad with a rubber-lined Y-fork. Arm along +x."""
    p = PM(0.012)
    yel = SP.skin("Street_BarrierCab", "#dba92a", 1.2, tile=1.0, rough=0.5, grime=0.35, grime_amt=0.45, grime_col="#8a7d63",
                  splash=0.5, scuff=(0.1, 1.1, 0.5), streaks=[(1.075, 0.45, 0.3)], chalk=0.35, peel=0.05, primer=(0.12, 0.4, 0.25))
    steel = SP.skin("Street_BarrierSteel", "#5b5f62", 1.2, tile=1.0, rough=0.45, metal=0.6, grime=0.15, grime_amt=0.6, rust=0.4,
                    mottle=0.1, spangle=0.6)
    holder = SP.skin("Street_BoomHolder", "#4b4e52", 1.2, tile=1.0, px=(256, 256), rough=0.42, metal=0.6, grime=0.0, mottle=0.08,
                     scuff=(0.85, 1.1, 0.5))
    white = SP.skin("Street_BoomWhite", "#e9e8e2", 1.2, tile=1.0, px=(512, 512), rough=0.35, grime=0.0, mottle=0.04, chalk=0.0,
                    streaks=[(0.95, 0.35, 0.03)])
    red = _tape("Street_TapeRed", "#b3121b", 0.1)
    conc_top, conc = SP.concrete_top("Street_PlinthTop", 0.8), _concrete("Street_BarrierPlinth")
    SP.pad(p, -0.27, -0.2, 0.27, 0.2, 0.1, conc_top, conc, kerb=0.02, kerb_mat=conc)
    p.r = 0.003
    p.box((-0.2, -0.155, 0.1), (0.2, 0.155, 0.112), steel)
    for sx in (-1, 1):
        for sy in (-1, 1):
            SP.bolt_head(p, (sx * 0.175, sy * 0.13, 0.112), (0, 0, 1), 0.01, M["galv"], hexa=True)
    p.r = 0.02
    p.loft([rrect(0.35, 0.26, 0.025, z) for z in (0.112, 1.075)], yel, cap0=False)
    p.r = 0.008
    p.loft([rrect(0.39, 0.30, 0.03, 1.07), rrect(0.39, 0.30, 0.03, 1.095), rrect(0.36, 0.27, 0.04, 1.115), rrect(0.27, 0.18, 0.05, 1.128)],
           yel)
    # amber LED beacon on the cap
    p.r = 0.002
    p.cyl((0.08, -0.04, 1.125), (0.08, -0.04, 1.15), 0.038, M["black"], 20)
    SP.lathe(p, [(0.033, 1.15), (0.033, 1.18), (0.026, 1.198), (0.012, 1.206), (0.0, 1.208)], emissive("Street_Amber", "#ff9c1a", 0.35),
             sides=20, c=(0.08, -0.04, 0))
    # front inspection door: a panel 3 mm proud, lock barrel, vent slots, maker plate
    p.r = 0.003
    p.box((-0.135, 0.126, 0.18), (0.135, 0.134, 0.78), yel)
    p.cyl((0.1, 0.134, 0.72), (0.1, 0.143, 0.72), 0.0125, M["galv"], 14)
    p.decal((0.1, 0.143, 0.72), (0, 1, 0), (0, 0, 1), 0.022, 0.022, _key_lock(), lift=0.0005)
    for k in range(5):
        z = 0.24 + k * 0.03
        p.decal((0.0, 0.134, z), (0, 1, 0), (0, 0, 1), 0.16, 0.01, _void(), lift=0.0006)

    def maker(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.002, 0.002, w - 0.002, h - 0.002, 0.003), "#d6d7d3")
        c.text("AUTOMATIC BARRIER  620", w / 2, h - 0.016, 0.006, "#202020", max_w=w - 0.01)
        for i, t in enumerate(("230V ~ 50Hz   IP44", "Boom max 5 m", "S/N 0623 1187")):
            c.text(t, 0.008, h - 0.03 - i * 0.009, 0.0045, "#303030", align="l", bold=False)
        c.age(0.2, 0.2)
    p.decal((-0.06, 0.134, 0.66), (0, 1, 0), (0, 0, 1), 0.08, 0.06, SP.card("Street_BarrierMaker", 0.08, 0.06, maker, ppm=6000, bg="#b9bab6",
                                                                           metal=0.7, rough=0.35), lift=0.001)
    # louvre on the -x side: a dark recess with angled slats
    p.r = 0.0
    p.quad((-0.176, 0.07, 0.72), (-0.176, -0.07, 0.72), (-0.176, -0.07, 0.92), (-0.176, 0.07, 0.92), _void())
    p.r = 0.002
    for k in range(7):
        z = 0.735 + k * 0.026
        p.obox((-0.18, 0, z + 0.004), (0.004, 0.14, 0.022), yel, roll=0.7)
    p.box((-0.182, -0.078, 0.712), (-0.172, 0.078, 0.722), yel)
    p.box((-0.182, -0.078, 0.918), (-0.172, 0.078, 0.928), yel)

    def warn(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.002, 0.002, w - 0.002, h - 0.002, 0.004), "#111111")
        c.poly(SP.rect(0.005, 0.005, w - 0.005, h - 0.005, 0.003), "#f5c400")
        tri = [(w / 2, h - 0.012), (w / 2 - 0.035, h - 0.072), (w / 2 + 0.035, h - 0.072)]
        c.poly(tri, "#111111")
        c.poly([(w / 2, h - 0.022), (w / 2 - 0.027, h - 0.067), (w / 2 + 0.027, h - 0.067)], "#f5c400")
        c.text("!", w / 2, h - 0.064, 0.026, "#111111")
        c.text("CAUTION", w / 2, 0.034, 0.011, "#111111")
        c.text("AUTOMATIC BARRIER", w / 2, 0.016, 0.0065, "#111111", max_w=w - 0.012)
        c.age(0.25, 0.1)
    p.decal((0.176, 0.0, 0.55), (1, 0, 0), (0, 0, 1), 0.1, 0.12, SP.card("Street_BarrierWarn", 0.1, 0.12, warn, ppm=5000, bg="#f5c400"),
            lift=0.001)
    # hub boss and the cast boom holder wrapping the boom end
    zp = 0.97
    BW, BH = 0.052, 0.092
    yb = 0.2 + BW / 2
    p.r = 0.006
    p.cyl((0.0, 0.13, zp), (0.0, 0.19, zp), 0.06, holder, 28)
    p.r = 0.004
    p.box((-0.13, 0.19, zp - BH / 2 - 0.012), (0.13, 0.2, zp + BH / 2 + 0.012), holder)
    p.box((-0.13, 0.19, zp + BH / 2), (0.13, yb + BW / 2 + 0.012, zp + BH / 2 + 0.012), holder)
    p.box((-0.13, 0.19, zp - BH / 2 - 0.012), (0.13, yb + BW / 2 + 0.012, zp - BH / 2), holder)
    p.box((-0.13, yb + BW / 2, zp - BH / 2 - 0.012), (0.13, yb + BW / 2 + 0.012, zp + BH / 2 + 0.012), holder)
    for dx in (-0.09, 0.09):
        for dz in (-0.03, 0.03):
            SP.bolt_head(p, (dx, yb + BW / 2 + 0.012, zp + dz), (0, 1, 0), 0.009, M["galv"], hexa=True)
    # boom: rounded rectangular section, white with red bands, rubber strip under it, end cap
    se = lambda c: math.copysign(abs(c) ** 0.6, c)              # a rounded, slightly squared oval
    prof = [(BH / 2 * se(math.cos(TAU * k / 20)), BW / 2 * se(math.sin(TAU * k / 20))) for k in range(20)]
    prof_band = [(y * 1.012, z * 1.02) for y, z in prof]
    L = 4.0
    p.r = 0.0
    xs = [-0.12] + [0.35 + 0.5 * k for k in range(8)] + [L]
    for k in range(len(xs) - 1):
        band = k % 2 == 1
        SP.sweep(p, [(xs[k], yb, zp), (xs[k + 1], yb, zp)], prof_band if band else prof, red if band else white, normal=(0, 0, 1))
    p.r = 0.004
    SP.sweep(p, [(L, yb, zp), (L + 0.03, yb, zp)], prof, M["black"], normal=(0, 0, 1))
    p.r = 0.002
    p.box((0.14, yb - 0.0125, zp - BH / 2 - 0.028), (L + 0.02, yb + 0.0125, zp - BH / 2 + 0.001), M["rubber"])
    # rest post on its own pad: galvanised post, red bands, rubber-lined Y-fork round the boom
    xr = L - 0.15
    SP.pad(p, xr - 0.16, yb - 0.16, xr + 0.16, yb + 0.16, 0.04, conc_top, conc, kerb=0.02, kerb_mat=conc)
    p.r = 0.003
    p.box((xr - 0.1, yb - 0.1, 0.04), (xr + 0.1, yb + 0.1, 0.05), steel)
    for sx in (-1, 1):
        for sy in (-1, 1):
            SP.bolt_head(p, (xr + sx * 0.075, yb + sy * 0.075, 0.05), (0, 0, 1), 0.009, M["galv"], hexa=True)
    zf = zp - BH / 2 - 0.028
    p.r = 0.0
    SP.lathe(p, [(0.03, 0.05), (0.03, zf - 0.03), (0.0, zf - 0.03)], M["galv"], sides=20, c=(xr, yb, 0), uv_r=0.03)
    for z0 in (0.55, 0.72):
        SP.lathe(p, [(0.0306, z0), (0.0306, z0 + 0.075)], red, sides=20, c=(xr, yb, 0), uv_r=0.03)
    # U-cradle, 120 mm wide and rubber-lined, the boom sitting in it
    p.r = 0.003
    p.box((xr - 0.04, yb - 0.065, zf - 0.03), (xr + 0.04, yb + 0.065, zf - 0.016), M["galv"])
    p.box((xr - 0.035, yb - 0.045, zf - 0.016), (xr + 0.035, yb + 0.045, zf), M["rubber"])
    for sy in (-1, 1):
        p.box((xr - 0.04, yb + sy * 0.065 - 0.008, zf - 0.03), (xr + 0.04, yb + sy * 0.065 + 0.008, zf + 0.085), M["galv"])
        ya, yb2 = sorted((yb + sy * 0.045, yb + sy * 0.057))
        p.box((xr - 0.035, ya, zf), (xr + 0.035, yb2, zf + 0.08), M["rubber"])
    SP.top_uv(p)
    return p.obj("barrier_arm", col, 35)


# -- lighting -------------------------------------------------------------------------------------

def _led_panel(name="Street_LEDOptic"):
    """The lantern's underside: a dark optic plate with 4 x 6 lens modules, each a grid of LEDs in
    silver reflector cups (emissive: lit at night)."""
    def draw(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.0, 0.0, w, h), "#202224")
        for i in range(4):
            for j in range(6):
                x0, y0 = 0.012 + i * (w - 0.024) / 4, 0.012 + j * (h - 0.024) / 6
                mw, mh = (w - 0.024) / 4 - 0.006, (h - 0.024) / 6 - 0.006
                c.poly(SP.rect(x0, y0, x0 + mw, y0 + mh, 0.004), "#3a3d40")
                for a in range(3):
                    for b in range(3):
                        cx, cy = x0 + mw * (a + 0.5) / 3, y0 + mh * (b + 0.5) / 3
                        c.poly(SP.circle(cx, cy, 0.0075, 20), "#9da2a6")
                        c.poly(SP.circle(cx, cy, 0.0045, 16), "#e8ecef")
                        c.poly(SP.circle(cx, cy, 0.0022, 12), "#fff6e0")
        c.rough = None
    return SP.card(name, 0.24, 0.5, draw, ppm=2000, bg="#202224", rough=0.25, emission=0.6)


def _opal(name="Street_Opal"):
    """Opal polycarbonate diffuser: warm white, faint horizontal ribs, softly lit."""
    px = 128
    Y = (np.arange(px) + 0.5)[:, None] / px * np.ones((1, px))
    ribs = np.sin(Y * TAU * 8)
    alb = np.broadcast_to(SP.lin("#ebe8dc"), (px, px, 3)) * (0.97 + 0.03 * ribs[..., None])
    return SP.image_material(name, alb, 0.2, SP.normal_map(ribs * 1.5, 1.0), cell=(0.032, 0.032), emission=0.12)


def lamp_post(col, M):
    """An 8 m galvanised street-lighting column on a flange plate (four bolt covers) and a cast
    concrete pad: a flared root and one continuous taper from 168 to 90 mm, a base door (groove,
    hinge bumps, two triangular-key locks), a yellow column number plate and a no-fly-posting
    sticker, a swept bracket into the side entry of a die-cast LED lantern (finned top, NEMA
    photocell, rounded nose, tilt knuckle, a bezelled LED optic below)."""
    p = PM(0.004)
    R0, R1, ZS, ZT = 0.08, 0.0445, 1.2, 7.25
    galv = SP.skin("Street_ColumnGalv", "#9b9f9f", 8.2, tile=math.pi * R0, px=(256, 1024), rough=0.5, metal=0.5, grime=0.3,
                   grime_amt=0.6, grime_col="#55503f", splash=0.6, streaks=[(ZT, 0.5, 1.5), (ZS, 0.3, 0.4), (0.82, 0.3, 0.3)],
                   mottle=0.08, chalk=0.0, peel=0.05, rust=0.2)
    # planted root: a mortar collar round the column where it leaves the ground
    p.r = 0.004
    SP.lathe(p, [(0.125, -0.01), (0.125, 0.012), (0.115, 0.024), (0.095, 0.03), (0.0, 0.03)],
             SP.concrete_top("Street_MortarTop2", 0.5, base="#8a867d"), sides=32, uv_xy=True)
    prof = [(R0, -0.05), (R0, ZS), (R0 + 0.004, ZS + 0.004), (R0 + 0.004, ZS + 0.018), (R0, ZS + 0.022), (R1, ZS + 0.058),
            (0.043, 4.5), (0.042, ZT), (0.0, ZT + 0.002)]
    SP.lathe(p, prof, galv, sides=32, uv_r=R0)
    p.r = 0.003
    p.cyl((0, 0, ZT - 0.12), (0, 0, ZT + 0.03), 0.052, galv, 24)                       # bracket spigot sleeve
    for a in (0.3, 2.4, 4.5):
        SP.bolt_head(p, (0.052 * math.cos(a), 0.052 * math.sin(a), ZT - 0.05), (math.cos(a), math.sin(a), 0), 0.007, M["galv"], hexa=True)
    # flush door in the base section: a groove, hinge bumps, two triangular-key locks
    SP.curved_panel(p, R0, math.pi / 2, 0.1, 0.5, 0.8, 0.0008, galv, sides=6)

    def groove(c):
        w, h = c.W / c.k, c.H / c.k
        outer = c.poly(SP.rect(0.0, 0.0, w, h, 0.012), "#151515")
        c.alpha = outer - SP.poly_mask(c.H, c.W, [[(x * c.k, y * c.k) for x, y in SP.rect(0.003, 0.003, w - 0.003, h - 0.003, 0.01)]])
    SP.wrap_decal(p, R0, 0.495, 0.805, math.pi / 2, 0.108, SP.card("Street_DoorGroove", 0.108, 0.31, groove, ppm=2500, bg="#151515",
                                                                   alpha=True), sides=8, lift=0.0005)
    for z in (0.54, 0.76):
        x, y, _ = SP.ring_pt(R0 + 0.0008, math.pi / 2 + 0.66, z)
        p.cyl((x, y, z - 0.018), (x, y, z + 0.018), 0.006, galv, 8)
        x, y, _ = SP.ring_pt(R0 + 0.0008, math.pi / 2 - 0.45, z)
        p.cyl((x, y, z), (x + 0.003, y + 0.006, z), 0.009, M["steel"], 12)
        p.decal((x + 0.003, y + 0.006, z), (0.43, 0.9, 0), (0, 0, 1), 0.016, 0.016, _key_lock(), lift=0.0005)

    def plate(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.002, 0.002, w - 0.002, h - 0.002, 0.004), "#111111")
        c.poly(SP.rect(0.0045, 0.0045, w - 0.0045, h - 0.0045, 0.003), "#f2f1ea")
        c.text("L", w / 2, h * 0.56, h * 0.28, "#111111")
        c.text("27", w / 2, h * 0.14, h * 0.28, "#111111")
        c.age(0.2, 0.2)
    SP.wrap_decal(p, R1 - 0.0004, 2.64, 2.76, math.pi / 2, 0.07, SP.card("Street_ColumnPlateW", 0.07, 0.12, plate, ppm=5000), sides=8,
                  lift=0.0015)

    def sticker(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0.002, 0.002, w - 0.002, h - 0.002), "#c8102e")
        c.text("NO FLY", w / 2, h * 0.55, h * 0.24, "#ffffff")
        c.text("POSTING", w / 2, h * 0.15, h * 0.24, "#ffffff", max_w=w - 0.008)
        c.age(0.35, 0.1)
        c.grime(0.2)
    SP.wrap_decal(p, R0, 1.0, 1.08, math.pi / 2 + 0.6, 0.09, SP.card("Street_NoFlyPosting", 0.09, 0.08, sticker, ppm=4000), sides=8,
                  lift=0.001)
    # bracket: a 0.6 m radius swept up and over to a 1.2 m outreach
    p.r = 0.0
    br = SP.spline([(0, 0, ZT - 0.05), (0, 0, ZT + 0.2), (0, 0.12, ZT + 0.55), (0, 0.45, ZT + 0.7), (0, 0.9, ZT + 0.72), (0, 1.22, ZT + 0.72)], 5)
    SP.sweep(p, br, [(0.03 * math.cos(TAU * j / 16), 0.03 * math.sin(TAU * j / 16)) for j in range(16)], galv)
    # lantern: die-cast body lofted along y, tilted up 5 degrees at the nose
    body = SP.skin("Street_Lantern", "#5d6266", 8.2, tile=1.0, px=(256, 256), rough=0.42, grime=0.0, mottle=0.05, streaks=[(8.0, 0.3, 0.05)])
    tilt = math.radians(5)
    secs = []
    Y0, L, ZL = 1.26, 0.72, ZT + 0.72
    for t in np.linspace(0, 1, 11):
        y = Y0 + t * L
        hw = 0.07 + 0.09 * math.sin(min(1.0, t * 1.6) * math.pi / 2) - 0.06 * (max(0.0, t - 0.8) / 0.2) ** 2 * 2.2
        hw = max(hw, 0.04)
        th = 0.085 - 0.03 * t
        ring = [(x * hw, z * th) for x, z in SP._rr2(2.0, 1.0, 0.3, 4)]
        z0 = ZL + (y - Y0) * math.tan(tilt)
        secs.append([(x, y, z0 + z) for x, z in ring])
    p.r = 0.006
    p.loft(secs[::-1], body)
    zt = lambda y: ZL + (y - Y0) * math.tan(tilt) + (0.085 - 0.03 * (y - Y0) / L) / 2
    zb = lambda y: ZL + (y - Y0) * math.tan(tilt) - (0.085 - 0.03 * (y - Y0) / L) / 2
    p.r = 0.002
    for k in range(7):
        x = -0.09 + k * 0.03
        y0, y1 = Y0 + 0.1, Y0 + 0.42
        p.box((x - 0.003, y0, zt(y0) - 0.006), (x + 0.003, y1, zt(y1) + 0.018), body)
    # NEMA photocell on the top
    yc = Y0 + 0.52
    p.r = 0.003
    p.cyl((0, yc, zt(yc) - 0.004), (0, yc, zt(yc) + 0.022), 0.042, M["black"], 24)
    SP.lathe(p, [(0.036, zt(yc) + 0.022), (0.034, zt(yc) + 0.035), (0.022, zt(yc) + 0.046), (0.0, zt(yc) + 0.05)], M["black"], sides=24,
             c=(0, yc, 0))
    # side-entry clamp with tilt knuckle
    p.r = 0.004
    p.cyl((0, Y0 - 0.07, ZL), (0, Y0 + 0.03, ZL), 0.045, body, 18)
    p.cyl((-0.065, Y0 - 0.02, ZL), (0.065, Y0 - 0.02, ZL), 0.032, body, 16)
    for s in (-1, 1):
        SP.bolt_head(p, (s * 0.065, Y0 - 0.02, ZL), (s, 0, 0), 0.011, M["galv"], hexa=True)
    # optic underneath inside a raised bezel
    p.r = 0.0
    yb0, yb1 = Y0 + 0.2, Y0 + 0.62
    led = _led_panel()
    zz = lambda y: zb(y) - 0.002
    p.face([(0.085, yb0, zz(yb0)), (-0.085, yb0, zz(yb0)), (-0.085, yb1, zz(yb1)), (0.085, yb1, zz(yb1))], led, [(1, 0), (0, 0), (0, 1), (1, 1)])
    p.r = 0.002
    for xa, xb, ya, yb_ in ((-0.1, 0.1, yb0 - 0.015, yb0), (-0.1, 0.1, yb1, yb1 + 0.015), (-0.1, -0.085, yb0, yb1), (0.085, 0.1, yb0, yb1)):
        p.box((xa, ya, zb((ya + yb_) / 2) - 0.009), (xb, yb_, zb((ya + yb_) / 2) + 0.002), body)
    SP.top_uv(p)
    return p.obj("lamp_post", col, 40)


def path_light(col, M):
    """A 600 mm LED bollard light, 110 mm, on a cast concrete footing: base flange with four
    bolts, powder-coated body with a grooved access door (two tamper screws) and a rating label,
    a head of four louvre rings with real gaps over a ribbed opal diffuser, a thin spun cap."""
    p = PM(0.003)
    R, Z0 = 0.055, 0.015
    body = SP.skin("Street_PathLight", "#2f3235", 0.7, tile=math.pi * R, px=(256, 512), rough=0.4, grime=0.15, grime_amt=0.6,
                   grime_col="#4e483c", splash=0.6, scuff=(Z0 + 0.12, Z0 + 0.3, 0.5), streaks=[(Z0 + 0.44, 0.5, 0.12)], peel=0.03,
                   mottle=0.04)
    flange = _plain("Street_PathLightFlange", "#3d4043", 0.45, mottle=0.06, scuff=(0.0, 0.1, 0.5))
    cap = _plain("Street_PathLightCap", "#383b3f", 0.3, mottle=0.03, top_fade=(0.6, 0.35))
    p.r = 0.004
    SP.lathe(p, [(0.088, -0.01), (0.088, 0.006), (0.083, 0.012), (0.0, Z0)], SP.concrete_top("Street_FootingTop", 0.5, base="#86827a"),
             sides=32, uv_xy=True)
    p.r = 0.003
    SP.lathe(p, [(0.08, Z0 - 0.001), (0.08, Z0 + 0.008), (0.075, Z0 + 0.012), (0.06, Z0 + 0.016), (0.0, Z0 + 0.016)], flange, sides=48, uv_r=R)
    SP.lathe(p, [(0.058, Z0 + 0.012), (R, Z0 + 0.03), (R, Z0 + 0.43),
                 (0.06, Z0 + 0.434), (0.06, Z0 + 0.44), (0.0, Z0 + 0.44)], body, sides=48, uv_r=R)
    for k in range(4):
        a = TAU * (k + 0.5) / 4
        SP.bolt_head(p, (0.069 * math.cos(a), 0.069 * math.sin(a), Z0 + 0.010), (0, 0, 1), 0.0065, M["galv"], hexa=True)
    SP.curved_panel(p, R, math.pi / 2, 0.05, Z0 + 0.12, Z0 + 0.30, 0.0008, body, sides=5)

    def groove(c):
        w, h = c.W / c.k, c.H / c.k
        outer = c.poly(SP.rect(0.0, 0.0, w, h, 0.006), "#121212")
        c.alpha = outer - SP.poly_mask(c.H, c.W, [[(x * c.k, y * c.k) for x, y in SP.rect(0.0022, 0.0022, w - 0.0022, h - 0.0022, 0.005)]])
    SP.wrap_decal(p, R, Z0 + 0.116, Z0 + 0.304, math.pi / 2, 0.056, SP.card("Street_HatchGroove", 0.056, 0.188, groove, ppm=4000,
                                                                          bg="#121212", alpha=True), sides=6, lift=0.0004)
    p.r = 0.0
    for z in (Z0 + 0.135, Z0 + 0.285):
        x, y, _ = SP.ring_pt(R + 0.0008, math.pi / 2, z)
        p.cyl((x, y, z), (x, y + 0.0015, z), 0.0035, M["galv"], 8)

    def label(c):
        w, h = c.W / c.k, c.H / c.k
        c.poly(SP.rect(0, 0, w, h, 0.002), "#c9cbcc")
        c.text("LED 12W  IP65", 0.004, h - 0.009, 0.004, "#1a1a1a", align="l")
        c.text("220-240V ~ 50/60Hz", 0.004, h - 0.016, 0.003, "#1a1a1a", align="l", bold=False)
        c.text("CE", w - 0.005, 0.004, 0.006, "#1a1a1a", align="r")
        c.age(0.2, 0.3)
    SP.wrap_decal(p, R, Z0 + 0.37, Z0 + 0.395, math.pi / 2 + 0.9, 0.05, SP.card("Street_PathLightLabel", 0.05, 0.025, label, ppm=8000,
                                                                              metal=0.5), sides=6, lift=0.0008)
    SP.lathe(p, [(0.045, Z0 + 0.44), (0.046, Z0 + 0.445), (0.046, Z0 + 0.555), (0.045, Z0 + 0.56)], _opal(), sides=48)
    for k in range(4):
        z = Z0 + 0.455 + k * 0.027
        SP.lathe(p, [(0.049, z - 0.002), (0.0625, z - 0.0042), (0.0625, z - 0.0018), (0.049, z + 0.002)], cap, sides=48, uv_r=R)
        SP.lathe(p, [(0.049, z + 0.002), (0.049, z - 0.002)], cap, sides=48, uv_r=R)
    for k in range(3):
        a = TAU * k / 3 + 0.3
        c = (0.058 * math.cos(a), 0.058 * math.sin(a))
        p.box((c[0] - 0.003, c[1] - 0.003, Z0 + 0.44), (c[0] + 0.003, c[1] + 0.003, Z0 + 0.562), cap)
    p.r = 0.0015
    SP.lathe(p, [(0.0, Z0 + 0.558), (0.063, Z0 + 0.558), (0.0665, Z0 + 0.553), (0.068, Z0 + 0.556), (0.0685, Z0 + 0.563), (0.066, Z0 + 0.567),
                 (0.058, Z0 + 0.572), (0.04, Z0 + 0.579), (0.0, Z0 + 0.583)], cap, sides=48, uv_r=R)
    SP.top_uv(p)
    return p.obj("path_light", col, 45)


def bike_rack(col, M):
    """Five Sheffield cycle stands at 1 m centres on a block-paved pad with a concrete edging
    course: 48 mm tube bent into a 750 x 750 inverted U (200 mm bends), satin black powder coat
    chipped through to primer and steel where bikes lean, two white reflective bands a leg (150 and
    600 mm), each leg welded to a 150 mm plate held by four tamper-proof bolts."""
    p = PM(0.003)
    PH = 0.045                                                # pad above the lawn
    coat = SP.skin("Street_StandCoat", "#232425", 0.9, tile=1.0, px=(512, 1024), rough=0.4, mottle=0.02, grime=PH + 0.05,
                   grime_col="#5b5345", grime_amt=0.45, splash=0.3, rust=0.25, primer=(PH + 0.4, PH + 0.62, 0.45),
                   scuff=(PH + 0.35, PH + 0.75, 0.6), chalk=0.15, peel=0.04, wear_bands=[(PH + 0.6, PH + 0.8, 0.55, "#8b8d89")])
    tape = _tape("Street_TapeWhiteStand", "#d6d6cf", 0.2)
    SP.pad(p, -2.6, -1.1, 2.6, 1.1, PH, SP.block_paving(), _concrete("Street_PadEdge"))
    prof = [(0.024 * math.cos(TAU * j / 16), 0.024 * math.sin(TAU * j / 16)) for j in range(16)]
    z0 = PH + 0.008
    for k in range(5):
        x = -2.0 + k * 1.0
        pts = [(x, -0.375, z0), (x, -0.375, PH + 0.75 - 0.024), (x, 0.375, PH + 0.75 - 0.024), (x, 0.375, z0)]
        p.r = 0.0
        SP.sweep(p, round_path(pts, 0.165, n=7), prof, coat, normal=(1, 0, 0))
        for y in (-0.375, 0.375):
            p.r = 0.002
            p.box((x - 0.075, y - 0.075, PH), (x + 0.075, y + 0.075, z0), coat)
            SP.lathe(p, [(0.031, z0), (0.027, z0 + 0.007), (0.0241, z0 + 0.012)], coat, c=(x, y, 0), sides=16)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    SP.bolt_head(p, (x + sx * 0.055, y + sy * 0.055, z0), (0, 0, 1), 0.009, M["galv"], sides=8)
            p.r = 0.0
            for zb in (PH + 0.10, PH + 0.55):
                SP.lathe(p, [(0.0242, zb), (0.0247, zb + 0.001), (0.0247, zb + 0.099), (0.0242, zb + 0.1)], tape, c=(x, y, 0),
                         sides=16, uv_r=0.024)
    SP.top_uv(p)
    return p.obj("bike_rack", col, 40)


BUILDERS = {
    "prop.bin_litter": bin_litter, "prop.bin_recycling": bin_recycling, "prop.bin_wheelie": bin_wheelie,
    "prop.bin_container": bin_container, "prop.bench": bench, "prop.picnic_table": picnic_table,
    "prop.bollard": bollard, "prop.barrier_arm": barrier_arm, "prop.lamp_post": lamp_post,
    "prop.path_light": path_light, "prop.bike_rack": bike_rack,
}
