"""Training-gear textures, painted with numpy alone: worn plastic and powder coat (tileable detail
for recolourable surfaces), the football's panel atlas (seams, stitching, pebble grain, prints,
scuffs), knotted netting and welded wire (alpha), nylon webbing, and the free-kick mannequin's
front map (moulded relief, ball marks, mud).

Arrays are float, linear colour, row 0 = the bottom of the image (Blender's convention). Normal
maps follow the OpenGL convention (+y up the image).
"""
import math

import numpy as np

from .foliage_paint import _normal_from_height, fbm, srgb_lin


def grid(w, h=None):
    h = h or w
    return np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)


def _wrap(d):
    return (d + 0.5) % 1.0 - 0.5


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def seg_dist(U, V, p0, p1, wrap=True, aspect=1.0):
    """Distance (in u units) from every pixel to the segment p0-p1; periodic when wrap."""
    dx, dy = U - p0[0], (V - p0[1]) * aspect
    if wrap:
        dx, dy = _wrap(dx), _wrap(dy / aspect) * aspect
    ex, ey = p1[0] - p0[0], (p1[1] - p0[1]) * aspect
    t = np.clip((dx * ex + dy * ey) / (ex * ex + ey * ey + 1e-12), 0, 1)
    return np.hypot(dx - t * ex, dy - t * ey)


def _blob(U, V, c, rx, ry, ang, wrap=True):
    dx, dy = U - c[0], V - c[1]
    if wrap:
        dx, dy = _wrap(dx), _wrap(dy)
    ca, sa = math.cos(ang), math.sin(ang)
    x, y = dx * ca + dy * sa, -dx * sa + dy * ca
    return np.clip(1 - np.hypot(x / rx, y / ry), 0, 1)


# -- worn surfaces --------------------------------------------------------------------------------

def wear(px=512, seed=0, mottle=0.05, scratches=60, scuffs=10, chips=0, tile=0.5):
    """Tileable wear masks over `tile` metres: (lum 0-1 mottling, scratch, scuff, chip, height)."""
    rng = np.random.default_rng(seed)
    S = (px, px)
    U, V = grid(px)
    lum = fbm(S, (3, 3), rng, 5)
    lum = (lum - lum.min()) / (np.ptp(lum) + 1e-9)
    scr = np.zeros(S)
    for _ in range(scratches):
        p0 = rng.random(2)
        ang = rng.normal(0, 0.3) + (math.pi / 2 if rng.random() < 0.25 else 0.0)
        ln = rng.uniform(0.006, 0.06) / tile
        p1 = p0 + ln * np.array([math.cos(ang), math.sin(ang)])
        w = rng.uniform(0.5, 1.4) / px
        d = seg_dist(U, V, p0, p1)
        scr = np.maximum(scr, np.clip(1 - d / w, 0, 1) * rng.uniform(0.35, 1.0))
    scf = np.zeros(S)
    for _ in range(scuffs):
        c = rng.random(2)
        r = rng.uniform(0.01, 0.035) / tile
        b = _blob(U, V, c, r * rng.uniform(1.5, 3.5), r, rng.normal(0, 0.4))
        scf = np.maximum(scf, b ** 1.5 * rng.uniform(0.4, 1.0))
    scf *= 0.6 + 0.8 * fbm(S, (40, 40), rng, 3)
    chp = np.zeros(S)
    if chips:
        brk = fbm(S, (64, 64), rng, 3)
        for _ in range(chips):
            c = rng.random(2)
            r = rng.uniform(0.0015, 0.006) / tile
            b = _blob(U, V, c, r * rng.uniform(1, 2.2), r, rng.uniform(0, math.pi))
            chp = np.maximum(chp, (b + 0.35 * (brk - 0.5) > 0.25).astype(float))
    peel = fbm(S, (96, 96), rng, 2)
    height = 0.25 * peel - 0.8 * scr - 0.6 * chp
    return dict(lum=lum, scratch=np.clip(scr, 0, 1), scuff=np.clip(scf, 0, 1), chip=chp, height=height)


def tint_detail(w, mottle=0.05, scuff=0.35, scratch=0.04, chip=0.3, rough=0.4, rough_var=0.08):
    """A greyscale detail map (max 1) for a recolourable surface, its mean, roughness and normal."""
    det = (1 - mottle * w["lum"]) * (1 - scuff * w["scuff"]) * (1 - chip * w["chip"])
    det = det * (1 - scratch * w["scratch"])
    r = rough + rough_var * (w["lum"] - 0.5) + 0.25 * w["scratch"] + 0.2 * w["scuff"] + 0.15 * w["chip"]
    nrm = _normal_from_height(w["height"], 1.5)
    return np.clip(det, 0, 1), float(det.mean()), np.clip(r, 0.05, 1), nrm


def coloured(w, color, chip_color=None, scuff_color="#5a5148", mottle=0.06, scuff=0.45, chip=1.0, rough=0.5,
             scratch_color=None):
    """Full-colour albedo for a fixed-colour surface (black powder coat chipped to steel)."""
    base = srgb_lin(color)
    alb = base * (1 - mottle * w["lum"])[..., None]
    sc = srgb_lin(scuff_color)
    alb = alb + (sc - alb) * (scuff * w["scuff"])[..., None]
    if scratch_color:
        alb = alb + (srgb_lin(scratch_color) - alb) * (0.7 * w["scratch"])[..., None]
    if chip_color:
        alb = alb + (srgb_lin(chip_color) - alb) * (chip * w["chip"])[..., None]
    r = rough + 0.08 * (w["lum"] - 0.5) + 0.2 * w["scuff"] - 0.15 * w["chip"] + 0.15 * w["scratch"]
    return np.clip(alb, 0, 1), np.clip(r, 0.05, 1), _normal_from_height(w["height"], 1.5)


# -- the football ---------------------------------------------------------------------------------

BALL_CELLS = 3          # 3 x 3 atlas cells
BALL_RHO = 0.145        # a hexagon's circumradius in atlas units (a pentagon's is 0.8507 of it)
BALL_HEX_MM = 44.4      # a size 5 ball's hexagon circumradius, mm


def ball_cell(i):
    """Atlas cell i (0-8) centre in UV."""
    return ((i % BALL_CELLS + 0.5) / BALL_CELLS, (i // BALL_CELLS + 0.5) / BALL_CELLS)


# cell contents: (sides, colour, print)
BALL_LAYOUT = {0: (6, "#d9d9d3", None), 1: (6, "#d9d9d3", "print"), 2: (6, "#d2d1c9", "scuff"),
               3: (5, "#262628", None), 4: (6, "#e0d41c", None), 5: (6, "#ee6a1e", None),
               6: (5, "#2a2b2e", "scuff"), 7: (5, "#1c2a55", None), 8: (6, "#d9d9d3", "print2")}


def _stamp(c, X, Y, text, cx, cy, cap_mm, color, inside):
    """Print `text` (cap height cap_mm) centred at (cx, cy) mm in a panel's plane."""
    from .lettering import text_mask
    tm = text_mask(text, 64, weight=1.0)
    th, tw = tm.shape
    hmm = cap_mm * 8 / 6                       # the mask's height spans 8 units, caps 6
    wmm = hmm * tw / th
    gx = ((X - cx + wmm / 2) / wmm * tw).astype(int)
    gy = ((Y - cy + hmm / 2) / hmm * th).astype(int)
    ok = (gx >= 0) & (gx < tw) & (gy >= 0) & (gy < th)
    m = np.where(ok, tm[np.clip(gy, 0, th - 1), np.clip(gx, 0, tw - 1)], 0.0) * inside
    return c + (srgb_lin(color) - c) * m[..., None]


def ball_atlas(px=1024, seed=5):
    """The 32-panel ball's panels, one per atlas cell (see BALL_LAYOUT): returns (albedo RGBA,
    normal, roughness). Each panel is a regular polygon, a vertex on +u, circumradius BALL_RHO
    (hexagon) in UV; stitched seams sunk at its edges, pillowed towards the middle."""
    rng = np.random.default_rng(seed)
    U, V = grid(px)
    mm = BALL_RHO * px / BALL_HEX_MM            # pixels per mm
    alb = np.zeros((px, px, 3))
    hgt = np.zeros((px, px))
    rgh = np.zeros((px, px))
    grain = fbm((px, px), (180, 180), rng, 2)
    mott = fbm((px, px), (12, 12), rng, 4)
    for i, (n, col, extra) in BALL_LAYOUT.items():
        cu, cv = ball_cell(i)
        x0, x1 = int(round((i % 3) * px / 3)), int(round((i % 3 + 1) * px / 3))
        y0, y1 = int(round((i // 3) * px / 3)), int(round((i // 3 + 1) * px / 3))
        sl = (slice(y0, y1), slice(x0, x1))
        X = (U[sl] - cu) * px / mm                  # mm in the panel's plane
        Y = (V[sl] - cv) * px / mm
        rho = BALL_HEX_MM * (1.0 if n == 6 else 0.8507)
        apo = rho * math.cos(math.pi / n)
        th = [(k + 0.5) * 2 * math.pi / n for k in range(n)]
        proj = np.stack([X * math.cos(t) + Y * math.sin(t) for t in th])
        k = np.argmax(proj, 0)
        d = apo - proj.max(0)                       # mm to the nearest edge (inside > 0)
        along = np.choose(k, [-X * math.sin(t) + Y * math.cos(t) for t in th])
        de = np.clip(d, 0, None)
        c = np.broadcast_to(srgb_lin(col), X.shape + (3,)).copy()
        c *= (0.97 + 0.05 * mott[sl])[..., None]
        # seam: a sunk groove at the edge, the panel rolling down into it over ~5 mm
        groove = 1 - _smooth(0.0, 1.35, de)
        h = 0.9 * _smooth(0.0, 6.0, de) ** 0.7 - 1.0 * groove
        # stitches crossing the groove every 5 mm (thread a little lighter than a dark seam)
        st = ((along / 5.0) % 1.0 < 0.38) & (de < 1.4) & (d > -0.4)
        h = h + 0.25 * st
        r = (0.38 if n == 6 else 0.44) + 0.05 * grain[sl] + 0.3 * groove
        if extra in ("print", "print2"):
            # a printed graphic: sweeping bands and a ring, kept inside the panel
            inside = _smooth(2.5, 4.0, de)
            ox, oy = (-78.0, 58.0) if extra == "print" else (55.0, 50.0)
            dist = np.hypot(X - ox, Y - oy)
            bands = [(64.0, 2.6, "#c4161c"), (70.0, 4.2, "#1c2a55"), (77.0, 1.4, "#c4161c")]
            if extra == "print2":
                bands = [(60.0, 5.0, "#1c2a55"), (67.5, 1.6, "#e0a019")]
            for rad, w, bc in bands:
                m = np.clip((w - np.abs(dist - rad)) * mm * 0.5, 0, 1) * inside
                c = c + (srgb_lin(bc) - c) * m[..., None]
                r = r - 0.05 * m
            if extra == "print":
                # a generic wordmark across the panel
                c = _stamp(c, X, Y, "PRO MATCH", 2.0, -6.0, 6.4, "#1c2a55", inside)
                c = _stamp(c, X, Y, "SIZE 5", 2.0, -17.0, 4.0, "#1c2a55", inside)
            else:
                # inflation valve: a grey rubber boss with a dark pinhole, and a quality roundel
                dv = np.hypot(X - 0.0, Y + 16.0)
                boss = _smooth(4.8, 4.0, dv)
                c = c + (srgb_lin("#8a8a86") - c) * (0.8 * boss)[..., None]
                c = c * (1 - 0.85 * _smooth(1.6, 1.0, dv))[..., None]
                h = h - 0.35 * boss - 0.6 * _smooth(1.6, 1.0, dv)
                r = r + 0.2 * boss
                dq = np.hypot(X + 14.0, Y - 8.0)
                ring = np.clip((0.55 - np.abs(dq - 8.0)) * mm * 0.5, 0, 1) + np.clip((0.45 - np.abs(dq - 6.2)) * mm * 0.5, 0, 1)
                c = c + (srgb_lin("#1c2a55") - c) * (np.clip(ring, 0, 1) * inside)[..., None]
                c = _stamp(c, X, Y, "QC", -14.0, 8.0, 4.4, "#1c2a55", inside)
        if extra == "scuff" or n == 6:
            # handling and turf: grey-brown rubs and faint grass stains (lighter on clean panels)
            amt = 1.0 if extra == "scuff" else 0.3
            s = np.zeros(X.shape)
            for _ in range(5 if extra == "scuff" else 2):
                cx, cy = rng.uniform(-30, 30, 2)
                b = np.clip(1 - np.hypot((X - cx) / rng.uniform(6, 16), (Y - cy) / rng.uniform(3, 8)), 0, 1)
                s = np.maximum(s, b ** 1.2 * rng.uniform(0.4, 1))
            s *= 0.5 + fbm(X.shape, (12, 12), rng, 3)
            tone = srgb_lin("#6b6a4a") if n == 6 else srgb_lin("#3c3b38")
            c = c + (tone - c) * (np.clip(s, 0, 1) * (0.35 if n == 6 else 0.5) * amt)[..., None]
            r = r + 0.15 * s * amt
        # seam shadow: the groove's floor is darker; stitch thread pale grey
        c = c * (1 - 0.72 * groove)[..., None]
        thread = srgb_lin("#9c9c96") if n == 6 else srgb_lin("#3a3a3c")
        c = np.where((st & (groove > 0.3))[..., None], c * 0.7 + thread * 0.3, c)
        alb[sl] = c
        hgt[sl] = h + 0.06 * grain[sl]
        rgh[sl] = r
    nrm = _normal_from_height(hgt, mm * 0.8)
    rgba = np.concatenate([np.clip(alb, 0, 1), np.ones((px, px, 1))], -1)
    return rgba, nrm, np.clip(rgh, 0.05, 1)


# -- netting, wire mesh, webbing ------------------------------------------------------------------

def netting(px=256, cord=0.035, knot=1.7, color="#f2f2ec", twist=True, seed=1, weld=False, shade=0.35):
    """One square cell of knotted cord (or welded wire): (RGBA, normal). cord: diameter as a
    fraction of the cell; the cell's corners are the knots, which sit at the tile's corners."""
    rng = np.random.default_rng(seed)
    U, V = grid(px)
    du, dv = np.abs(_wrap(U)), np.abs(_wrap(V))      # distance to the cell lines through 0
    rc = cord / 2
    dk = np.hypot(du, dv)
    rk = rc * knot
    # cords along u (|dv| < rc) and v (|du| < rc), and a knot at the crossing
    a_h = np.clip((rc - dv) * px + 0.5, 0, 1)
    a_v = np.clip((rc - du) * px + 0.5, 0, 1)
    a_k = np.clip((rk - dk) * px + 0.5, 0, 1)
    alpha = np.maximum(np.maximum(a_h, a_v), a_k)
    # round cords: a cylinder's height across each, a dome on the knot
    h_h = np.sqrt(np.clip(1 - (dv / rc) ** 2, 0, 1)) * rc
    h_v = np.sqrt(np.clip(1 - (du / rc) ** 2, 0, 1)) * rc
    h_k = np.sqrt(np.clip(1 - (dk / rk) ** 2, 0, 1)) * rk * 1.15
    h = np.maximum(np.maximum(h_h, h_v), h_k)
    lit = np.where(a_k > 0.5, 0.82, 1.0)
    if twist:
        # three-strand lay: diagonal ridges along each cord
        tw_h = 0.5 + 0.5 * np.sin((U / (rc * 1.6) + dv / rc * 1.2) * math.pi * 2)
        tw_v = 0.5 + 0.5 * np.sin((V / (rc * 1.6) + du / rc * 1.2) * math.pi * 2)
        tw = np.where(dv < du, tw_h, tw_v)
        h = h * (1 - 0.18 * tw)
        lit = lit * (1 - 0.12 * tw)
    if weld:
        lit = np.where(a_k > 0.5, 0.9, 1.0)
    cyl = np.clip(h / (rk * 1.15), 0, 1)
    lit = lit * ((1 - shade) + shade * cyl)
    c = srgb_lin(color)
    rgb = c * lit[..., None] * (0.88 + 0.2 * fbm((px, px), (8, 8), rng, 3))[..., None]
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1)
    nrm = _normal_from_height(h * px, 1.0)
    return rgba, nrm


def webbing(px=256, color="#1d1e20", seed=2):
    """Tileable woven nylon webbing (a 2/2 twill, ~1 mm yarns at 256 px over 40 mm), with
    light fuzz: (albedo RGBA, normal, roughness)."""
    rng = np.random.default_rng(seed)
    U, V = grid(px)
    n = 40                                           # yarns across the tile
    x, y = U * n, V * n * 2
    ix, iy = np.floor(x), np.floor(y)
    over = ((ix + iy) % 4) < 2                       # twill: warp over two, under two
    fx, fy = x - ix, y - iy
    warp = np.sin(math.pi * fx) ** 0.6 * (0.6 + 0.4 * np.sin(math.pi * fy))
    weft = np.sin(math.pi * fy) ** 0.6 * (0.6 + 0.4 * np.sin(math.pi * fx))
    h = np.where(over, warp, weft * 0.85)
    fuzz = fbm((px, px), (64, 64), rng, 3)
    h = h + 0.15 * fuzz
    c = srgb_lin(color) * (0.75 + 0.45 * h)[..., None] * (0.9 + 0.2 * fbm((px, px), (6, 6), rng, 3))[..., None]
    rgba = np.concatenate([np.clip(c, 0, 1), np.ones((px, px, 1))], -1)
    return rgba, _normal_from_height(h, 2.5), np.clip(0.82 + 0.08 * fuzz, 0, 1)


# -- the free-kick mannequin's front --------------------------------------------------------------

MANNEQUIN_BOX = (-0.32, 0.32, 0.02, 1.85)    # x0, x1, z0, z1 covered by the front map


def mannequin_map(px=(512, 1024), seed=11, outline=None):
    """Greyscale detail (max 1), roughness and normal for the mannequin, projected from the front
    over MANNEQUIN_BOX: soft ball marks at chest height, boot scuffs at the shins, a soft mud
    wash on the feet with streaky drips, sun-faded shoulders, a moulded number on the chest.
    outline(z) -> half width at height z (metres)."""
    from .lettering import text_mask
    W, H = px
    rng = np.random.default_rng(seed)
    U, V = grid(W, H)
    x0, x1, z0, z1 = MANNEQUIN_BOX
    X = x0 + U * (x1 - x0)
    Z = z0 + V * (z1 - z0)
    det = 1 - 0.04 * fbm((H, W), (4, 8), rng, 5)
    h = 0.03 * fbm((H, W), (60, 120), rng, 2)
    r = 0.38 + 0.05 * fbm((H, W), (6, 12), rng, 3)
    # ball marks: soft round smudges, 1.0 - 1.6 m
    for _ in range(6):
        cx, cz = rng.normal(0, 0.08), rng.uniform(1.0, 1.6)
        rad = rng.uniform(0.04, 0.08)
        d = np.hypot(X - cx, (Z - cz) * rng.uniform(0.9, 1.2)) / rad
        m = np.clip(1 - d, 0, 1) ** 1.2 * (0.6 + 0.6 * fbm((H, W), (24, 48), rng, 3))
        det *= 1 - rng.uniform(0.08, 0.16) * np.clip(m, 0, 1)
        r += 0.12 * np.clip(m, 0, 1)
    # boot scuffs: streaks at shin height
    for _ in range(12):
        cx, cz = rng.uniform(-0.15, 0.15), rng.uniform(0.25, 0.55)
        ang = rng.normal(0.25, 0.3)
        dx, dz = X - cx, Z - cz
        a_ = dx * math.cos(ang) + dz * math.sin(ang)
        b_ = -dx * math.sin(ang) + dz * math.cos(ang)
        m = np.clip(1 - np.hypot(a_ / rng.uniform(0.02, 0.05), b_ / rng.uniform(0.004, 0.009)), 0, 1)
        det *= 1 - 0.25 * m
        r += 0.15 * m
    # mud: a soft wash at the feet, fading up with streaks and drips
    wash = 1 - _smooth(0.08, 0.36, Z + 0.06 * (fbm((H, W), (16, 8), rng, 4) - 0.5))
    streak = fbm((H, W), (90, 6), rng, 3)
    drips = np.clip((streak - 0.55) * 4, 0, 1) * (1 - _smooth(0.2, 0.5, Z))
    mud = np.clip(0.75 * wash + 0.5 * drips, 0, 1) * (0.75 + 0.25 * fbm((H, W), (40, 80), rng, 3))
    det *= 1 - 0.35 * mud
    r += 0.25 * mud
    h += 0.15 * mud
    # sun fade: the top of the head and shoulders a touch paler (the rest a touch dirtier)
    det *= 0.95 + 0.05 * _smooth(1.35, 1.8, Z)
    # moulded number panel on the chest: a raised rounded rectangle with a sunk number
    pw, ph, pz = 0.1, 0.125, 1.25
    q = np.maximum(np.abs(X) - (pw - 0.02), 0) ** 2 + np.maximum(np.abs(Z - pz) - (ph - 0.02), 0) ** 2
    plate = _smooth(0.022, 0.016, np.sqrt(q))
    h += 0.8 * plate
    tm = text_mask("7", 96, weight=1.2)
    th, tw = tm.shape
    gh = 0.17                                   # glyph box height on the body (m)
    gx = ((-X + gh * tw / th / 2) / (gh * tw / th) * tw).astype(int)      # read from the front (-x is right)
    gz = ((Z - (pz - gh / 2)) / gh * th).astype(int)
    ok = (gx >= 0) & (gx < tw) & (gz >= 0) & (gz < th)
    num = np.where(ok, tm[np.clip(gz, 0, th - 1), np.clip(gx, 0, tw - 1)], 0.0)
    h -= 0.7 * num
    det *= 1 - 0.05 * num
    if outline is not None:
        hw = outline(Z)
        e = np.abs(hw - np.abs(X))
        det *= 1 - 0.04 * np.exp(-(e / 0.01) ** 2)
    nrm = _normal_from_height(h, 12.0)
    return np.clip(det, 0, 1), float(np.median(det)), np.clip(r, 0.05, 1), nrm


POLE_LEN = 1.7


def pole_map(px=(128, 1024), seed=21):
    """Greyscale detail (max 1), roughness and normal for a 25 mm slalom pole: u round the tube,
    v up its length (POLE_LEN): the extrusion seam, fine lengthwise draw lines, boot scuffs and grass
    stain thickening into the bottom 250 mm."""
    W, H = px
    rng = np.random.default_rng(seed)
    U, V = grid(W, H)
    Z = V * POLE_LEN
    det = 1 - 0.03 * fbm((H, W), (2, 24), rng, 4)
    h = 0.1 * fbm((H, W), (8, 200), rng, 3)                    # draw lines along the extrusion
    seam = np.exp(-((U - 0.5) / 0.008) ** 2)
    h += 0.6 * seam
    det *= 1 - 0.06 * seam
    r = 0.36 + 0.04 * fbm((H, W), (4, 40), rng, 3)
    for _ in range(30):                                          # boot and ball scuffs, mostly low
        cu, cz = rng.random(), min(abs(rng.normal(0, 0.35)), 1.5)
        du = _wrap(U - cu)
        m = np.clip(1 - np.hypot(du / rng.uniform(0.05, 0.2), (Z - cz) / rng.uniform(0.004, 0.015)), 0, 1)
        det *= 1 - rng.uniform(0.08, 0.25) * m
        r += 0.2 * m
    # the maker's name printed along the pole, black, reading upward
    from .lettering import text_mask
    tm = text_mask("SLALOM PRO", 64, weight=1.2)
    th, tw = tm.shape
    span = 0.2 * tw / th * 0.078 / 1.0                  # text length (m) for a 0.2-turn-tall line
    row = ((U - 0.4) / 0.2 * th).astype(int)
    col = ((Z - 1.22) / span * tw).astype(int)
    ok = (row >= 0) & (row < th) & (col >= 0) & (col < tw)
    txt = np.where(ok, tm[np.clip(th - 1 - row, 0, th - 1), np.clip(col, 0, tw - 1)], 0.0)
    det = det * (1 - 0.9 * txt)
    stain = (1 - _smooth(0.02, 0.28, Z + 0.05 * (fbm((H, W), (4, 16), rng, 3) - 0.5))) * \
        (0.7 + 0.3 * fbm((H, W), (8, 64), rng, 3))
    det *= 1 - 0.35 * stain
    r += 0.2 * stain
    return np.clip(det, 0, 1), float(np.median(det)), np.clip(r, 0.05, 1), _normal_from_height(h, 3.0)


# -- the team shelter -----------------------------------------------------------------------------

def shelter_poly(px=(256, 512), seed=31):
    """Clear polycarbonate sheet (u across 2 m, v from the foot of the back sheet over the roof to
    the front edge): a faint blue-grey film whose opacity rises with dust, water streaks running
    down the back and a grimy drip line at the front edge. RGBA, alpha is coverage."""
    W, H = px
    rng = np.random.default_rng(seed)
    U, V = grid(W, H)
    streak = fbm((H, W), (40, 3), rng, 4)
    dust = fbm((H, W), (6, 12), rng, 4)
    low = (1 - _smooth(0.0, 0.35, V)) * np.clip((streak - 0.35) * 1.6, 0, 1)
    drip = _smooth(0.93, 1.0, V) * (0.6 + 0.4 * streak)
    a = 0.13 + 0.06 * dust + 0.22 * low + 0.25 * drip
    clean = srgb_lin("#e4ecef")
    dirt = srgb_lin("#8a8775")
    m = np.clip(1.6 * low + 1.2 * drip, 0, 1)[..., None]
    rgb = clean * (1 - m) + dirt * m
    return np.concatenate([rgb, np.clip(a, 0, 1)[..., None]], -1)


def turf_mat(px=512, seed=33):
    """Tileable artificial-turf mat (0.5 m): dense green fibre tufts with darker gaps, a little
    wear. (albedo RGBA, normal, roughness)."""
    rng = np.random.default_rng(seed)
    S = (px, px)
    tufts = fbm(S, (120, 120), rng, 2)
    blades = fbm(S, (240, 60), rng, 2)
    h = 0.6 * tufts + 0.4 * blades
    g0, g1 = srgb_lin("#1f4a22"), srgb_lin("#3f7a35")
    rgb = g0 + (g1 - g0) * np.clip(h, 0, 1)[..., None] ** 1.3
    wear = np.clip((fbm(S, (4, 4), rng, 4) - 0.55) * 3, 0, 1)
    rgb = rgb * (1 - 0.25 * wear)[..., None] + srgb_lin("#6b6a52") * (0.25 * wear)[..., None]
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((px, px, 1))], -1)
    return rgba, _normal_from_height(h, 4.0), np.clip(0.8 + 0.1 * h, 0, 1)


def grime_strip(px=(512, 128), seed=35):
    """Scuffs and splash on the foot of a panel (u along 2 m, v up 0.35 m): dark RGBA, alpha the
    amount of dirt, densest at the bottom."""
    W, H = px
    rng = np.random.default_rng(seed)
    U, V = grid(W, H)
    splash = fbm((H, W), (40, 10), rng, 4)
    a = np.clip((splash - 0.3 - 0.9 * V) * 1.4, 0, 0.55)
    for _ in range(18):                                   # boot scuffs
        cu, cv = rng.random(), rng.uniform(0.1, 0.7)
        d = np.hypot(_wrap(U - cu) / rng.uniform(0.01, 0.04), (V - cv) / rng.uniform(0.02, 0.06))
        a = np.maximum(a, np.clip(1 - d, 0, 1) * rng.uniform(0.2, 0.45))
    rgb = np.broadcast_to(srgb_lin("#3d3428"), (H, W, 3))
    return np.concatenate([rgb, a[..., None]], -1)


def rivets(px=(64, 32), color="#c9ccce"):
    """One rivet head per tile (u along the row): RGBA with alpha round the domed head."""
    W, H = px
    U, V = grid(W, H)
    d = np.hypot((U - 0.5) * W / H, V - 0.5) / 0.32
    a = np.clip((1 - d) * 6, 0, 1)
    shade = 0.65 + 0.45 * np.clip(1 - np.hypot((U - 0.42) * W / H, V - 0.6) / 0.3, 0, 1)
    rgb = srgb_lin(color) * shade[..., None]
    return np.concatenate([rgb, a[..., None]], -1)


def digits(px_h=64, color="#f2f2ee"):
    """'1'-'9' side by side in equal cells (cell i of 9 holds digit i + 1): RGBA."""
    from .lettering import text_mask
    cells = []
    for dgt in "123456789":
        m = text_mask(dgt, px_h, weight=1.1)
        w = int(px_h * 0.75)
        out = np.zeros((m.shape[0], w))
        x0 = max(0, (w - m.shape[1]) // 2)
        out[:, x0:x0 + min(w, m.shape[1])] = m[:, :min(w, m.shape[1])]
        cells.append(out)
    m = np.hstack(cells)
    rgb = np.broadcast_to(srgb_lin(color), m.shape + (3,))
    return np.concatenate([rgb, m[..., None]], -1)


def crest(px=(256, 320)):
    """A generic club crest (shield, chevron, ball roundel, 'FC'): RGBA, alpha outside the shield 0."""
    from .lettering import text_mask
    W, H = px
    U, V = grid(W, H)
    x, y = (U - 0.5) * 2, V                                  # x -1..1, y 0 (bottom) .. 1
    def inside(ax, ytop, base):
        """A shield: straight sides, flat top, a rounded point at the bottom."""
        bottom = base - base * np.sqrt(np.clip(1 - (x / ax) ** 2, 0, 1)) + 0.02
        return (np.abs(x) < ax) & (y < ytop) & (y > bottom)
    shield = inside(0.92, 0.96, 0.45)
    edge = shield & ~inside(0.82, 0.9, 0.42)
    rgb = np.broadcast_to(srgb_lin("#c8102e"), (H, W, 3)).copy()
    chev = np.abs(y - (0.62 - 0.35 * np.abs(x))) < 0.06
    rgb[chev] = srgb_lin("#f4f4f0")
    ring = np.abs(np.hypot(x * W / 2, (y - 0.3) * H) - 0.15 * W) < 0.028 * W
    rgb[ring] = srgb_lin("#f4f4f0")
    tm = text_mask("FC", 64, weight=1.4)
    th, tw = tm.shape
    gx = ((U - 0.5) / 0.4 * tw + tw / 2).astype(int)
    gy = ((V - 0.7) / 0.22 * th).astype(int)
    ok = (gx >= 0) & (gx < tw) & (gy >= 0) & (gy < th)
    t = np.where(ok, tm[np.clip(gy, 0, th - 1), np.clip(gx, 0, tw - 1)], 0.0) > 0.5
    rgb[t] = srgb_lin("#f4f4f0")
    rgb[edge] = srgb_lin("#1c2a55")
    return np.concatenate([rgb, shield[..., None].astype(float)], -1)
