"""Image materials for the construction-site props (props_site), painted in numpy with site_paint.

Every function returns a Blender material (made once, by name). Tileable sets carry `uv_cell`
(metres per repeat) and expect faces with metric UVs (props_site gives every face u along the
surface, read the right way round from outside, and v = height).
"""
import math

import numpy as np

from . import site_paint as SP
from .site_paint import fbm, lin, smooth, streaks, vnoise


def _get(name):
    import bpy
    return bpy.data.materials.get(name)


def _rust_rgb(H, W, seed, dark="#3e2416", mid="#7a3c1e", hi="#a65a2a"):
    n = fbm(H, W, 24, 24, seed, 4)
    n2 = fbm(H, W, 90, 90, seed + 3, 2)
    c = lin(dark)[None, None] * (1 - smooth(0.3, 0.6, n))[..., None] + lin(mid)[None, None] * smooth(0.3, 0.6, n)[..., None]
    return c * (1 - 0.6 * smooth(0.55, 0.8, n2))[..., None] + lin(hi)[None, None] * (0.6 * smooth(0.55, 0.8, n2))[..., None]


# -- shipping containers ------------------------------------------------------------------------------

CONT_CELL = 6.4          # metres per repeat of the container paint / wear sets
CONT_ROOF_V0 = 2.8       # the roof band starts here (v, metres); walls use v = 0..2.6


def container_paint(tint="#2c5a8a"):
    """TINT_container: greyscale paint shade (dirt toward the foot, dark run marks under the top
    rail, sun-faded roof with pooled stains), a dent normal map and roughness."""
    name = "TINT_container"
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / CONT_CELL
    S = np.ones((N, N))
    v = (N - 0.5 - np.arange(N)) / ppm           # metres up from the bottom of the image, per row
    V = np.repeat(v[:, None], N, 1)
    tone = fbm(N, N, 8, 8, 11, 4)
    S *= 0.9 + 0.2 * tone
    wall = V < 2.62
    roof = (V >= CONT_ROOF_V0) & (V < CONT_ROOF_V0 + 2.45)
    # repainted patches in a slightly different shade, a sun-faded upper half
    rng = np.random.default_rng(17)
    for _ in range(3):                                   # whole repaired panels, welded in
        u0 = rng.uniform(0, N)
        w = rng.uniform(0.55, 1.4) * ppm
        r0, r1 = int(N - 2.49 * ppm), int(N - 0.18 * ppm)
        m = SP.rect_mask(N, N, u0, r0, u0 + w, r1)
        m = np.maximum(m, SP.rect_mask(N, N, u0 - N, r0, u0 + w - N, r1))
        S *= 1 + m * wall * rng.choice([-0.1, 0.09])
    S *= np.where(wall, 1 + 0.13 * smooth(0.6, 2.5, V) * (0.7 + 0.6 * fbm(N, N, 3, 2, 18, 3)), 1)
    # walls: grime rising from the foot, splashes, dark runs from the top rail
    z = V
    splash = fbm(N, N, 40, 10, 12, 3)
    foot = 1 - smooth(0.0, 0.35 + 0.45 * splash, z)
    S = np.where(wall, S * (1 - 0.5 * foot ** 1.3), S)
    run = SP.blur(streaks(N, N, 13, density=0.25, length=0.4, width_px=7), 2)
    run_rows = np.clip((2.62 - z) / 2.62, 0, 1)
    st = np.zeros_like(S)
    top_row = int(round(N - 2.62 * ppm))
    st[top_row:] = run[:N - top_row]
    S = np.where(wall, S * (1 - 0.45 * st), S)
    # roof: sun-faded, with dark pooled stains and rust rings
    pool = fbm(N, N, 6, 6, 14, 4)
    S = np.where(roof, S * 1.16 * (1 - 0.3 * smooth(0.55, 0.72, pool)), S)
    S = np.clip(S, 0.2, 1.3)
    rough = np.clip(0.52 + 0.25 * foot * wall + 0.1 * tone, 0, 1)
    # dents: soft low hollows on the walls, a sagging roof
    dent = fbm(N, N, 10, 3, 15, 3)
    h = np.where(wall, -0.004 * smooth(0.55, 0.85, dent), -0.003 * fbm(N, N, 5, 5, 16, 3))
    nrm = SP.normal_from_height(SP.blur(h, 3), 1 / ppm, 1.0)
    return SP.pbr(name, np.repeat(S[..., None], 3, -1), rough=rough, metal=0.0, cell=(CONT_CELL, CONT_CELL),
                  normal=nrm, tint=tint)


def container_wear():
    """PS_ContainerWear: rust cut out over the paint (alpha-tested), on the container's metric
    UVs: bleeding from the top rail, eating along the foot, chips, scrapes along the lower panels,
    rust rings on the roof."""
    name = "PS_ContainerWear"
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / CONT_CELL
    v = (N - 0.5 - np.arange(N)) / ppm
    V = np.repeat(v[:, None], N, 1)
    wall = V < 2.62
    roof = (V >= CONT_ROOF_V0) & (V < CONT_ROOF_V0 + 2.45)
    a = np.zeros((N, N))
    # bleed from the top rail
    run = streaks(N, N, 21, density=0.22, length=0.3, width_px=3) * smooth(0.35, 0.6, fbm(N, N, 5, 2, 28, 2))
    top_row = int(round(N - 2.62 * ppm))
    st = np.zeros((N, N))
    st[top_row:] = run[:N - top_row]
    a = np.maximum(a, wall * smooth(0.25, 0.45, st))
    # the foot: rust eating up from the bottom rail
    edge = fbm(N, N, 60, 12, 22, 4)
    patchy = smooth(0.4, 0.65, fbm(N, N, 8, 2, 29, 3))
    a = np.maximum(a, wall * (V < (0.02 + 0.2 * edge ** 2) * patchy))
    # chips and spots
    spots = fbm(N, N, 70, 70, 23, 3) * fbm(N, N, 6, 6, 24, 2)
    a = np.maximum(a, wall * (spots * smooth(0.45, 0.7, fbm(N, N, 4, 3, 30, 2)) > 0.36))
    # scrapes: thin near-horizontal scratches on the lower panels
    rng = np.random.default_rng(25)
    sc = np.zeros((N, N))
    for _ in range(9):
        z0 = rng.uniform(0.25, 1.4)
        x0 = rng.uniform(0, N)
        L = rng.uniform(0.2, 1.1) * ppm
        ang = rng.uniform(-0.12, 0.12)
        xs = np.arange(int(L))
        ys = (N - z0 * ppm + xs * ang).astype(int)
        xx = ((x0 + xs) % N).astype(int)
        ok = (ys >= 0) & (ys < N)
        for d in (0, 1):
            sc[np.clip(ys[ok] + d, 0, N - 1), xx[ok]] = 1
    a = np.maximum(a, sc * wall)
    # roof: rust rings and blotches where water stands
    pool = fbm(N, N, 6, 6, 14, 4)
    a = np.maximum(a, roof * (fbm(N, N, 60, 60, 26, 3) * smooth(0.6, 0.78, pool) > 0.55))
    rgb = _rust_rgb(N, N, 27, "#33201a", "#58301f", "#7c4223")
    rgba = np.concatenate([rgb, a[..., None]], -1)
    return SP.pbr(name, rgba, rough=0.85, metal=0.0, cell=(CONT_CELL, CONT_CELL), clip=True)


def container_casting():
    """PS_ContainerCasting: a cast corner fitting face (UV 0-1 per face): oval aperture, rust."""
    name = "PS_ContainerCasting"
    if _get(name):
        return _get(name)
    N = 128
    img = np.zeros((N, N, 3))
    img[:] = _rust_rgb(N, N, 31, "#2a2522", "#4a3b30", "#6d4a32")
    img *= (0.8 + 0.3 * fbm(N, N, 4, 4, 32, 3))[..., None]
    hole = SP.poly_mask(N, N, SP.circle_pts(N / 2, N / 2, N * 0.3, 40, ry=N * 0.17))
    lip = SP.poly_mask(N, N, SP.circle_pts(N / 2, N / 2, N * 0.36, 40, ry=N * 0.23))
    SP.put(img, lip, lin("#1c1a18"), 0.5)
    SP.put(img, hole, lin("#060606"))
    return SP.pbr(name, img, rough=0.8, metal=0.0)


def stencil(name, lines, px=(512, 128), color="#ece9e1", kind="bold", align="l", wear=0.25, seed=0, bg=None):
    """White stencil paint (or a coloured sign when bg is given): lines [(text, cap fraction of
    height, y fraction of the top)] cut out with alpha, a little worn."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 4))
    if bg is not None:
        img[..., :3] = lin(bg)
        img[..., 3] = 1
    col = lin(color)
    for item in lines:
        s, cap, y = item[:3]
        k = item[3] if len(item) > 3 else kind
        x = {"l": 6, "c": W / 2, "r": W - 6}[align]
        SP.text(img, s, x, y * H, cap * H, col, k, align, max_w=W - 12)
    if wear > 0:
        # scuffs through the paint: fine speckle, a little heavier toward the bottom, never whole letters
        n = fbm(H, W, max(4, W // 3), max(4, H // 3), 41 + seed, 2)
        big = fbm(H, W, max(2, W // 64), max(2, H // 16), 43 + seed, 3)
        tilt = np.linspace(-0.04, 0.06, H)[:, None]
        if bg is None:
            img[..., 3] *= (n - 0.12 * smooth(0.55, 0.8, big) > 0.2 + 0.25 * wear + tilt * wear * 2).astype(float)
        else:
            img[..., :3] *= (0.85 + 0.15 * n)[..., None]
    return SP.pbr(name, img, rough=0.6, clip=bg is None)


def csc_plate():
    """PS_CSCPlate: the riveted aluminium safety-approval plate on the left door."""
    name = "PS_CSCPlate"
    if _get(name):
        return _get(name)
    W, H = 256, 160
    img = np.zeros((H, W, 3))
    img[:] = lin("#a9aca9")
    img *= (0.85 + 0.2 * fbm(H, W, 6, 4, 51, 3))[..., None]
    dark = lin("#2b2c2c")
    SP.text(img, "CSC SAFETY APPROVAL", W / 2, 10, 13, dark, "bold", "c")
    for k, s in enumerate(["GB-LR 2187 / 03-1", "DATE MANUFACTURED  06 2014", "IDENTIFICATION NO  KSTU 482913 5",
                           "MAX GROSS WEIGHT  30480 KG", "ALLOWABLE STACK  192000 KG", "RACKING TEST LOAD  15240 KG"]):
        SP.text(img, s, 12, 36 + k * 19, 9, dark, "regular", max_w=W - 24)
    for x in (8, W - 8):
        for y in (8, H - 8):
            SP.put(img, SP.poly_mask(H, W, SP.circle_pts(x, y, 4, 12)), lin("#d8dad8"))
    return SP.pbr(name, img, rough=0.4, metal=0.8)


# -- generic painted steel: greyscale paint detail (TINT_) and a rust overlay ---------------------------

def paint_detail(name, tint, cell=6.4, wall_h=2.62, seed=0, foot=0.5, runs=0.45, patches=6, rough=0.5, dents=0.003,
                 px=1024, fade=0.06):
    """A TINT_ paint shade over walls in metric UVs (u along, v = height): tone, repainted
    patches, dirt rising from the foot, dark runs below the top edge (wall_h), soft dents."""
    if _get(name):
        return _get(name)
    N = px
    ppm = N / cell
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    wall = V < wall_h
    S = 0.9 + 0.2 * fbm(N, N, 8, 8, seed + 1, 4)
    rng = np.random.default_rng(seed + 2)
    for _ in range(patches):
        u0, z0 = rng.uniform(0, N), rng.uniform(0.2, wall_h - 0.5)
        w, h = rng.uniform(0.3, 1.2) * ppm, rng.uniform(0.2, 0.7) * ppm
        r0 = N - (z0 * ppm + h)
        m = np.maximum(SP.rect_mask(N, N, u0, r0, u0 + w, r0 + h), SP.rect_mask(N, N, u0 - N, r0, u0 + w - N, r0 + h))
        S *= 1 + m * rng.choice([-0.1, 0.08])
    S *= 1 + fade * smooth(0.8, wall_h, V)
    splash = fbm(N, N, 40, 10, seed + 3, 3)
    f = 1 - smooth(0.0, 0.35 + 0.45 * splash, V)
    S *= 1 - foot * f ** 1.3
    run = SP.blur(streaks(N, N, seed + 4, density=0.25, length=0.4, width_px=7), 2)
    st = np.zeros_like(S)
    top = int(round(N - wall_h * ppm))
    st[top:] = run[:N - top]
    S *= 1 - runs * st
    S = np.where(wall, S, 0.95 + 0.1 * fbm(N, N, 6, 6, seed + 5, 3))
    r = np.clip(rough + 0.25 * f + 0.08 * fbm(N, N, 6, 6, seed + 6, 2), 0, 1)
    r = np.where(wall, r, 0.7)                             # tops and lids: dull, dusty paint
    h = -dents * smooth(0.55, 0.85, fbm(N, N, 10, 3, seed + 7, 3))
    nrm = SP.normal_from_height(SP.blur(h, 3), 1 / ppm, 1.0)
    return SP.pbr(name, np.clip(S, 0.15, 1.4), rough=r, cell=(cell, cell), normal=nrm, tint=tint)


def wear_overlay(name, cell=6.4, wall_h=2.62, seed=0, runs=0.22, foot=0.2, spots=0.36, scrapes=9, fixings=(),
                 px=1024, rust=("#33201a", "#58301f", "#7c4223"), bands=()):
    """Rust cut out over painted steel (alpha-tested), metric UVs: bleed below the top edge, rust
    along the foot, chips, scrapes, fixings (rows (height, pitch)) with drips under some."""
    if _get(name):
        return _get(name)
    N = px
    ppm = N / cell
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    wall = V < wall_h
    a = np.zeros((N, N))
    run = streaks(N, N, seed + 1, density=runs, length=0.3, width_px=3) * smooth(0.35, 0.6, fbm(N, N, 5, 2, seed + 2, 2))
    top = int(round(N - wall_h * ppm))
    st = np.zeros((N, N))
    st[top:] = run[:N - top]
    a = np.maximum(a, smooth(0.25, 0.45, st))
    edge = fbm(N, N, 60, 12, seed + 3, 4)
    patchy = smooth(0.4, 0.65, fbm(N, N, 8, 2, seed + 4, 3))
    a = np.maximum(a, V < (0.02 + foot * edge ** 2) * patchy)
    if spots:
        sp = fbm(N, N, 70, 70, seed + 5, 3) * smooth(0.45, 0.7, fbm(N, N, 4, 3, seed + 6, 2))
        a = np.maximum(a, sp > spots)
    rng = np.random.default_rng(seed + 7)
    for _ in range(scrapes):
        z0, x0 = rng.uniform(0.25, 1.4), rng.uniform(0, N)
        L = int(rng.uniform(0.2, 1.1) * ppm)
        ang = rng.uniform(-0.12, 0.12)
        xs = np.arange(L)
        ys = (N - z0 * ppm + xs * ang).astype(int)
        ok = (ys >= 0) & (ys < N - 1)
        a[ys[ok], ((x0 + xs) % N).astype(int)[ok]] = 1
        a[ys[ok] + 1, ((x0 + xs) % N).astype(int)[ok]] = 1
    for z, pitch in fixings:
        n = max(1, int(round(cell / pitch)))
        for k in range(n):
            cx, cy = (k + 0.5) * N / n, N - z * ppm
            a = np.maximum(a, SP.poly_mask(N, N, SP.circle_pts(cx, cy, 0.008 * ppm, 10)) > 0.5)
            if rng.random() < 0.35:
                Lr = rng.uniform(0.05, 0.35) * ppm
                a = np.maximum(a, SP.poly_mask(N, N, [(cx - 0.004 * ppm, cy), (cx + 0.004 * ppm, cy), (cx + 0.001 * ppm, cy + Lr),
                                                      (cx - 0.001 * ppm, cy + Lr)]) > 0.4)
    for z, hw_ in bands:                                    # chipped through on rolled edges and hoops
        chip = fbm(N, N, 50, 4, seed + 9 + int(z * 100), 3)
        a = np.maximum(a, (np.abs(V - z) < hw_ * (0.6 + 0.8 * chip)) & (chip > 0.42))
    a *= wall
    rgb = _rust_rgb(N, N, seed + 8, *rust)
    return SP.pbr(name, np.concatenate([rgb, a[..., None]], -1), rough=0.85, cell=(cell, cell), clip=True)


def flat_steel(name, color, cell=4.0, seed=0, dirt=0.3, rough=0.55, metal=0.0, px=512, pools=True):
    """A non-tinted painted or galvanised steel set seen from above or on small parts: tone,
    pooled dirt, fine scratches (metric UVs, tileable)."""
    if _get(name):
        return _get(name)
    N = px
    c = lin(color)
    t = fbm(N, N, 6, 6, seed + 1, 4)
    S = 0.9 + 0.2 * t
    if pools:
        pool = fbm(N, N, 4, 4, seed + 2, 4)
        S *= 1 - dirt * smooth(0.5, 0.75, pool)
    S *= 1 - 0.15 * (fbm(N, N, 120, 8, seed + 3, 2) > 0.72)
    img = c[None, None] * S[..., None]
    return SP.pbr(name, img, rough=np.clip(rough + 0.15 * (1 - S), 0, 1), metal=metal, cell=(cell, cell))


def grating():
    """PS_Grating: galvanised open-mesh step treads (dark slots), 30 x 100 mm pitch."""
    name = "PS_Grating"
    if _get(name):
        return _get(name)
    N = 256
    cell = 0.3
    img = np.zeros((N, N, 3))
    img[:] = lin("#9a9d9c")
    img *= (0.85 + 0.25 * fbm(N, N, 4, 4, 61, 3))[..., None]
    ppm = N / cell
    for k in range(10):
        x = k * 0.03 * ppm
        SP.put(img, SP.rect_mask(N, N, x + 0.004 * ppm, 0, x + 0.03 * ppm, N), lin("#141414"))
    for k in range(3):
        y = k * 0.1 * ppm
        SP.put(img, SP.rect_mask(N, N, 0, y, N, y + 0.006 * ppm), lin("#8f9291"))
    return SP.pbr(name, img, rough=0.5, metal=0.7, cell=(cell, cell))


def ppe_sign(name="PS_Label_PPE", lines=("SAFETY HELMETS, HI-VIS", "AND SAFETY BOOTS", "MUST BE WORN"), kinds=("hat", "vest", "boot"),
             px=(384, 512)):
    """A white site sign: blue mandatory discs over a blue text panel (UV 0-1)."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 3))
    img[:] = lin("#f4f4f0")
    n = len(kinds)
    r = min(W / (n * 2.4), H * 0.17)
    for k, kind in enumerate(kinds):
        SP.sign_mandatory(img, W * (k + 0.5) / n, H * 0.24, r, kind)
    SP.put(img, SP.rect_mask(H=H, W=W, x0=W * 0.05, y0=H * 0.5, x1=W * 0.95, y1=H * 0.95), lin("#1f5aa6"))
    for k, s in enumerate(lines):
        SP.text(img, s, W / 2, H * (0.56 + k * 0.13), H * 0.065, lin("#ffffff"), "bold", "c", max_w=W * 0.86)
    img *= (0.92 + 0.08 * fbm(H, W, 4, 4, 71, 3))[..., None]
    return SP.pbr(name, img, rough=0.45)


def sign_panel(name, draw, px=(512, 256), rough=0.45, grime=0.1, seed=0):
    """A printed sign (UV 0-1): draw(img) paints on a white canvas; light grime over it."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 3))
    img[:] = lin("#f4f4f0")
    draw(img)
    img *= (1 - grime + grime * fbm(H, W, 5, 3, 81 + seed, 3))[..., None]
    return SP.pbr(name, img, rough=rough)


def rubble_img(name="PS_RubbleFill2", cell=2.0, seed=0):
    """Mixed demolition fill seen from above: grey crushed concrete and mortar dust, brick-red
    fragments, timber splinters, dark voids between lumps (tileable, metric)."""
    if _get(name):
        return _get(name)
    N = 512
    rng = np.random.default_rng(seed + 90)
    img = np.zeros((N, N, 3))
    base = fbm(N, N, 16, 16, seed + 91, 4)
    img[:] = lin("#8c877d")[None, None] * (0.75 + 0.45 * base)[..., None]
    hgt = base * 0.02
    ppm = N / cell
    for k in range(900):
        x, y = rng.uniform(0, N, 2)
        r = rng.uniform(0.01, 0.06) * ppm
        kind = rng.random()
        c = lin("#8a3f2c") if kind < 0.25 else lin("#958f85") if kind < 0.65 else lin("#66625a") if kind < 0.9 else lin("#8b6c48")
        c = c * rng.uniform(0.75, 1.15)
        n = rng.integers(5, 8)
        pts = [(x + r * rng.uniform(0.6, 1.2) * math.cos(2 * math.pi * j / n + rng.uniform(-0.3, 0.3)),
                y + r * rng.uniform(0.6, 1.2) * math.sin(2 * math.pi * j / n + rng.uniform(-0.3, 0.3))) for j in range(n)]
        m = SP.poly_mask(N, N, pts, 2)
        # shadow side first (offset down-right), then the lump
        SP.put(img, np.roll(np.roll(m, 2, 0), 2, 1), lin("#2e2b27"), 0.7)
        SP.put(img, m, c)
        hgt = np.maximum(hgt, m * (0.02 + r / ppm * 0.4))
    dust = 0.25 + 0.45 * smooth(0.4, 0.75, fbm(N, N, 6, 6, seed + 92, 3))
    img = img * (1 - dust[..., None]) + lin("#7a7367")[None, None] * dust[..., None] * (0.8 + 0.3 * base[..., None])
    nrm = SP.normal_from_height(SP.blur(hgt, 1), 1 / ppm, 1.0)
    return SP.pbr(name, img, rough=0.92, cell=(cell, cell), normal=nrm)


def rusty_steel(name, cell=2.0, seed=0, paint="#7a6a3a", amount=0.6, px=512):
    """Worn steel inside a skip or on a bearer: scraped paint, brown rust, dark grime (metric)."""
    if _get(name):
        return _get(name)
    N = px
    n = fbm(N, N, 10, 10, seed + 1, 5)
    rust = smooth(0.62 - 0.3 * amount, 0.7 - 0.25 * amount, n)
    p = lin(paint)[None, None] * (0.7 + 0.3 * fbm(N, N, 4, 4, seed + 2, 3))[..., None]
    img = p * (1 - rust[..., None]) + _rust_rgb(N, N, seed + 3, "#2c1d15", "#4e2c1c", "#71401f") * rust[..., None]
    sc = fbm(N, N, 4, 80, seed + 4, 3) > 0.62
    img[sc] = img[sc] * 0.6 + lin("#5a5650") * 0.4
    return SP.pbr(name, img, rough=np.clip(0.55 + 0.35 * rust, 0, 1), cell=(cell, cell))


def builders_bag(name="PS_BulkBagBlack"):
    if _get(name):
        return _get(name)
    N = 256
    n = fbm(N, N, 12, 12, 95, 4)
    w = fbm(N, N, 3, 30, 96, 3)
    img = lin("#2a2b2c")[None, None] * (0.75 + 0.5 * n)[..., None]
    img = img * (1 - 0.4 * smooth(0.55, 0.7, w))[..., None] + lin("#8a857a") * 0.4 * smooth(0.6, 0.75, fbm(N, N, 5, 5, 97, 3))[..., None]
    h = SP.blur(w, 2) * 0.01
    h = h + fbm(N, N, 30, 30, 98, 3) * 0.004
    return SP.pbr(name, img * 0.8, rough=0.45, cell=(1.0, 1.0), normal=SP.normal_from_height(h, 1 / N, 1.0))


def moulded_plastic(name, color, cell=2.4, seed=0, wall_h=2.3, grime=0.45, fade=0.12, scuffs=7, rough=0.62, px=512,
                    grime_col="#4a4234", emission=None, pools=0.0):
    """Rotomoulded polyethylene (toilet cabins, barriers): sun fade toward the top, grime and
    splashes up from the foot, scuffs and handling marks, faint mould texture (metric UVs)."""
    if _get(name):
        return _get(name)
    N = px
    ppm = N / cell
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    c = lin(color)
    img = np.repeat(c[None, None], N, 0).repeat(N, 1).copy()
    t = fbm(N, N, 8, 8, seed + 1, 4)
    img *= (0.92 + 0.14 * t)[..., None]
    f = fade * smooth(0.3, wall_h, V) * (0.7 + 0.6 * fbm(N, N, 4, 4, seed + 2, 3))
    img = img * (1 - f[..., None]) + lin("#c9ccd0") * f[..., None] * 0.6 + img * f[..., None] * 0.4
    splash = fbm(N, N, 30, 8, seed + 3, 4)
    g = (1 - smooth(0.0, 0.25 + 0.3 * splash, V)) * grime
    img = img * (1 - g[..., None]) + lin(grime_col) * g[..., None]
    rng = np.random.default_rng(seed + 4)
    for _ in range(scuffs):
        x, z = rng.uniform(0, N), rng.uniform(0.3, 1.5) * ppm
        L = rng.uniform(0.05, 0.3) * ppm
        a = rng.uniform(-0.6, 0.6)
        m = SP.poly_mask(N, N, [(x, N - z), (x + L * math.cos(a), N - z - L * math.sin(a)), (x + L * math.cos(a), N - z - L * math.sin(a) + 1.5),
                                (x, N - z + 1.5)])
        SP.put(img, m, lin("#9fa4ab"), 0.25)
    run = SP.blur(streaks(N, N, seed + 5, density=0.15, length=0.3, width_px=5), 2)
    top = int(round(N - wall_h * ppm))
    st = np.zeros((N, N))
    st[max(top, 0):] = run[:N - max(top, 0)]
    img *= (1 - 0.25 * st)[..., None]
    h = fbm(N, N, 40, 40, seed + 6, 2) * 0.0008
    if pools:
        pl = smooth(0.5, 0.75, fbm(N, N, 5, 5, seed + 7, 4)) * pools
        img = img * (1 - pl[..., None]) + lin("#8a8270") * pl[..., None]
        sp = np.random.default_rng(seed + 8).random((N, N)) > 0.997
        img[sp] *= 0.4
    return SP.pbr(name, img, rough=np.clip(rough + 0.2 * g, 0, 1), cell=(cell, cell),
                  normal=SP.normal_from_height(h, 1 / ppm, 1.0), emission=emission)


def ladder_card(name="PS_LadderCard", w=0.45, pitch=0.28):
    """PS_LadderCard: a steel ladder (stiles and rungs) cut out with alpha; one rung per repeat
    (cell w x pitch metres, u across, v up)."""
    if _get(name):
        return _get(name)
    W, H = 128, 80
    img = np.zeros((H, W, 4))
    c = lin("#c9a400")
    for x0, x1 in ((0, 0.06), (0.94, 1.0)):
        SP.put(img, SP.rect_mask(H, W, x0 * W, 0, x1 * W, H), c)
    SP.put(img, SP.rect_mask(H, W, 0, H * 0.45, W, H * 0.58), c * 0.9)
    return SP.pbr(name, img, rough=0.6, cell=(w, pitch), clip=True)


def banner(name, draw, px=(1024, 192)):
    """A printed PVC mesh banner (UV 0-1), slightly faded."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 3))
    draw(img)
    img = img * 0.9 + 0.1 * lin("#d8d8d8")
    img *= (0.9 + 0.1 * fbm(H, W, 8, 2, 151, 3))[..., None]
    return SP.pbr(name, img, rough=0.7)


def concrete_block(name="PS_Ballast", color="#8f8b83", cell=2.0, seed=0, stains=0.5, chips=False):
    """Precast concrete: pores, tone, run-off stains and rust weeping from lifting eyes (metric)."""
    if _get(name):
        return _get(name)
    N = 512
    img = lin(color)[None, None] * (0.85 + 0.25 * fbm(N, N, 10, 10, seed + 1, 5))[..., None]
    pores = fbm(N, N, 160, 160, seed + 2, 2) > 0.7
    img[pores] *= 0.7
    run = SP.blur(streaks(N, N, seed + 3, density=0.3, length=0.6, width_px=8), 3)
    img *= (1 - stains * 0.4 * run)[..., None]
    img = img * (1 - 0.3 * run[..., None] * stains) + lin("#6a4a30") * 0.3 * run[..., None] * stains * smooth(0.5, 0.7, fbm(N, N, 6, 6, seed + 4, 2))[..., None]
    h = fbm(N, N, 60, 60, seed + 5, 3) * 0.002 - pores * 0.001
    if chips:
        # moss and dirt at the foot, a few pale chips
        V = (N - 0.5 - np.arange(N))[:, None] / N * cell
        foot = (1 - smooth(0.0, 0.12 + 0.1 * fbm(N, N, 20, 6, seed + 6, 3), V))
        img = img * (1 - 0.6 * foot[..., None]) + lin("#3e4524") * 0.6 * foot[..., None]
        ch = fbm(N, N, 40, 40, seed + 7, 3) > 0.68
        img[ch] = img[ch] * 0.5 + lin("#b9b5ab") * 0.5
    return SP.pbr(name, img, rough=0.9, cell=(cell, cell), normal=SP.normal_from_height(h, cell / N, 1.0))


def galvanised_img(name="PS_GalvTube", cell=(1.0, 4.0), seed=0, mud=0.3, base="#a7abab", rough_base=0.42, metal_base=0.75):
    """Hot-dip galvanised steel: spangle cells, dull zinc bloom and white rust, darker patches,
    mud splashed on the bottom `mud` metres (metric UVs: v = height on vertical faces; faces
    looking up or down should be mapped above v = 1)."""
    if _get(name):
        return _get(name)
    W, H = 256, 1024
    ppm = H / cell[1]
    rng = np.random.default_rng(seed + 160)
    pts = rng.random((260, 2)) * (W, H / 4)
    Y, X = np.mgrid[0:H, 0:W]
    # spangle: flat-shaded cells of slightly different brightness (on a quarter tile, repeated)
    q = H // 4
    Yq, Xq = np.mgrid[0:q, 0:W]
    d = np.full((q, W), 1e9)
    idx = np.zeros((q, W), int)
    for i, (px, py) in enumerate(pts):
        dx = np.minimum(np.abs(Xq - px), W - np.abs(Xq - px))
        dy = np.minimum(np.abs(Yq - py), q - np.abs(Yq - py))
        dd = dx * dx + dy * dy
        m = dd < d
        d[m] = dd[m]
        idx[m] = i
    shade = rng.uniform(0.85, 1.12, len(pts))[idx]
    spangle = np.tile(shade, (4, 1))
    S = spangle * (0.88 + 0.22 * fbm(H, W, 3, 12, seed + 161, 4))
    img = lin(base)[None, None] * S[..., None]
    bloom = smooth(0.6, 0.8, fbm(H, W, 6, 24, seed + 162, 4))
    img = img * (1 - 0.5 * bloom[..., None]) + lin("#d6d8d4") * 0.5 * bloom[..., None]
    dark = smooth(0.62, 0.8, fbm(H, W, 4, 16, seed + 163, 3))
    img *= (1 - 0.35 * dark)[..., None]
    V = (H - 0.5 - Y) / ppm
    if mud > 0:
        m = (1 - smooth(0.0, mud * (0.5 + fbm(H, W, 16, 64, seed + 164, 3)), V)) * (V < 1.0)
        img = img * (1 - 0.8 * m[..., None]) + lin("#5b4a36") * 0.8 * m[..., None]
    else:
        m = np.zeros((H, W))
    metal = np.clip(metal_base - 0.6 * bloom - 0.7 * m, 0, 1)
    rough = np.clip(rough_base + 0.3 * bloom + 0.4 * m + 0.1 * dark, 0, 1)
    return SP.pbr(name, img, rough=rough, metal=metal, cell=cell)


def recycled_rubber(name="PS_RecycledRubber", cell=1.0, seed=0, color="#1e1f20"):
    """Moulded recycled rubber (fence feet, speed bumps): black with coloured crumb speckle, dust."""
    if _get(name):
        return _get(name)
    N = 512
    rng = np.random.default_rng(seed + 170)
    img = lin(color)[None, None] * (0.8 + 0.4 * fbm(N, N, 20, 20, seed + 171, 4))[..., None]
    sp = rng.random((N, N))
    for thr, c in ((0.995, "#5a5a58"), (0.997, "#3d4f6e"), (0.998, "#6e3a30")):
        m = sp > thr
        img[m] = lin(c)
        sp = np.where(m, 0, sp)
    dust = smooth(0.55, 0.8, fbm(N, N, 5, 5, seed + 172, 3))
    img = img * (1 - 0.4 * dust[..., None]) + lin("#77705f") * 0.4 * dust[..., None]
    h = fbm(N, N, 80, 80, seed + 173, 2) * 0.0015
    return SP.pbr(name, img, rough=0.88, cell=(cell, cell), normal=SP.normal_from_height(h, cell / N, 1.0))


def weld_mesh(name, cell_u=0.075, cell_v=0.3, wire=0.004, color="#b2b6b5", px=(64, 256), blend=False):
    """Welded wire mesh cut out with alpha: one cell per repeat, round-looking wires (lighter crown,
    darker edges) and weld nodes at the crossings; uv_cell in metres."""
    if _get(name):
        return _get(name)
    W, H = px
    xs = (np.arange(W) + 0.5) / W * cell_u
    ys = (np.arange(H) + 0.5) / H * cell_v
    X, Y = np.meshgrid(xs, ys)
    dx = np.abs(X - cell_u / 2)
    dy = np.abs(Y - cell_v / 2)
    hw = wire / 2
    a_v = np.clip((hw - dx) / (cell_u / W) + 0.5, 0, 1)
    a_h = np.clip((hw - dy) / (cell_v / H) + 0.5, 0, 1)
    a = np.maximum(a_v, a_h)
    node = np.clip((hw * 1.6 - np.hypot(dx, dy * 1.0)) / (cell_u / W) + 0.5, 0, 1)
    a = np.maximum(a, node)
    crown = np.where(a_v >= a_h, 1 - (dx / hw) ** 2, 1 - (dy / hw) ** 2)
    shade = 0.65 + 0.45 * np.clip(crown, 0, 1)
    img = np.concatenate([lin(color)[None, None] * shade[..., None], a[..., None]], -1)
    return SP.pbr(name, img, rough=0.45, metal=0.6, cell=(cell_u, cell_v), clip=not blend, blend=blend)


def timber(name, color="#8a7356", cell=1.0, along="v", seed=0, weathered=0.5, rough=0.8):
    """Sawn, treated softwood: grain along u or v, knots, end-of-board darkening, greying,
    green algae and ground-line dirt left to the material that uses it (metric, tileable)."""
    if _get(name):
        return _get(name)
    N = 512
    rng = np.random.default_rng(seed + 180)
    # grain along v: stretch the noise vertically
    g = fbm(N, N, 40, 3, seed + 181, 5)
    fine = fbm(N, N, 160, 6, seed + 182, 3)
    rings = np.sin((g * 30 + fine * 4) * math.pi) * 0.5 + 0.5
    c = lin(color)
    img = c[None, None] * (0.78 + 0.22 * rings[..., None] + 0.12 * (fine[..., None] - 0.5))
    for _ in range(4):
        kx, ky = rng.uniform(0, N, 2)
        Y, X = np.mgrid[0:N, 0:N]
        d = np.hypot(np.minimum(np.abs(X - kx), N - np.abs(X - kx)) * 1.0, np.minimum(np.abs(Y - ky), N - np.abs(Y - ky)) * 0.35)
        k = np.clip(1 - d / rng.uniform(6, 12), 0, 1)
        img *= (1 - 0.55 * k)[..., None]
    grey = weathered * smooth(0.3, 0.8, fbm(N, N, 6, 2, seed + 183, 3))
    img = img * (1 - 0.6 * grey[..., None]) + lin("#8c8984") * 0.6 * grey[..., None] * (0.8 + 0.3 * rings[..., None])
    h = (rings * 0.4 + fine * 0.6) * 0.0012
    if along == "u":
        img = np.transpose(img, (1, 0, 2))
        h = h.T
    return SP.pbr(name, img, rough=rough, cell=(cell, cell), normal=SP.normal_from_height(h, cell / N, 1.0))


def plywood_back(name="PS_PlyBack", cell=2.44, seed=0):
    """The unpainted back of hoarding sheets: rotary-cut veneer figure, patches, stencilled grade
    and sheet numbers, weathered grey and dirty toward the ground (metric, one sheet pair per
    repeat, v = height)."""
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / cell
    g = fbm(N, N, 6, 24, seed + 190, 5)
    fig = np.sin(g * 40) * 0.5 + 0.5
    img = lin("#b39c78")[None, None] * (0.82 + 0.18 * fig[..., None] + 0.1 * (fbm(N, N, 4, 4, seed + 191, 3)[..., None] - 0.5))
    rng = np.random.default_rng(seed + 192)
    for _ in range(10):
        x, y = rng.uniform(0, N, 2)
        w, h = rng.uniform(0.04, 0.08) * ppm, rng.uniform(0.15, 0.25) * ppm
        SP.put(img, SP.poly_mask(N, N, SP.circle_pts(x, y, w, 16, ry=h)), lin("#c2a77d"), 0.8)
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    grey = smooth(0.3, 0.8, fbm(N, N, 5, 3, seed + 193, 3)) * 0.45
    img = img * (1 - grey[..., None]) + lin("#8e8a82") * grey[..., None]
    foot = 1 - smooth(0.0, 0.3 + 0.3 * fbm(N, N, 30, 8, seed + 194, 3), V)
    img = img * (1 - 0.5 * foot[..., None]) + lin("#4c3f2f") * 0.5 * foot[..., None]
    run = SP.blur(streaks(N, N, seed + 195, density=0.2, length=0.5, width_px=6), 2)
    img *= (1 - 0.25 * run)[..., None]
    for x0 in (0.2, 1.42):
        SP.text(img, "WBP 18MM  EN 636-2", x0 * ppm, N - 1.9 * ppm, 0.035 * ppm, lin("#2a2a2a"), "narrow", a=0.8)
        SP.text(img, f"No {int(rng.integers(100, 999))}", x0 * ppm, N - 1.82 * ppm, 0.05 * ppm, lin("#1e3f8a"), "bold", a=0.8)
    return SP.pbr(name, img, rough=0.82, cell=(cell, cell))


def hoarding_shade(name="TINT_hoarding", tint="#a8121f", cell=2.44, seed=0):
    """TINT_hoarding: the painted/printed face's greyscale shade: print texture, fade, rain runs
    from the capping, mud splashed up the bottom (metric, v = height)."""
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / cell
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    S = 0.95 + 0.1 * fbm(N, N, 8, 8, seed + 200, 4)
    S *= 1 + 0.08 * smooth(0.5, 2.4, V) * fbm(N, N, 3, 3, seed + 201, 3)
    run = SP.blur(streaks(N, N, seed + 202, density=0.35, length=0.55, width_px=6), 2)
    S *= 1 - 0.3 * run
    f = 1 - smooth(0.0, 0.2 + 0.3 * fbm(N, N, 40, 10, seed + 203, 3), V)
    S *= 1 - 0.55 * f
    return SP.pbr(name, np.repeat(S[..., None], 3, -1), rough=np.clip(0.5 + 0.3 * f, 0, 1), cell=(cell, cell), tint=tint)


def _cgi_render(h, w, seed=0):
    """An architect's render of the new training centre (RGB, linear): graded sky with soft
    cloud, a glazed two-storey pavilion under a cantilevered roof, timber-clad wing, reflections
    in the glass, soft contact shadows, pitches in perspective with mowing stripes, shaded trees,
    figures; finished with a slight blur and grain so it reads as a photographic CGI."""
    rng = np.random.default_rng(seed + 250)
    Y, X = np.mgrid[0:h, 0:w]
    hz = int(h * 0.6)
    t = (Y / hz)[..., None]
    pic = lin("#5d8fc8") * (1 - t) + lin("#d9e6f0") * t
    cl = smooth(0.5, 0.78, fbm(h, w, 5, 3, seed + 251, 5)) * (1 - Y / hz)
    pic = pic * (1 - 0.7 * cl[..., None]) + lin("#f6f8fa") * 0.7 * cl[..., None]
    # distant trees along the horizon, shaded
    for k in range(26):
        cx, r = rng.uniform(0, w), rng.uniform(0.04, 0.09) * h
        cy = hz - r * 0.5
        m = SP.poly_mask(h, w, SP.circle_pts(cx, cy, r * rng.uniform(1.0, 1.6), 24, ry=r))
        shade = np.clip((X - cx) / (r * 2) + 0.5, 0, 1)
        col = lin("#3d5f34") * (1 - shade[..., None]) + lin("#24391f") * shade[..., None]
        pic = pic * (1 - m[..., None]) + col * m[..., None]
    # ground: pitches in perspective with mowing stripes, darker toward the viewer
    gy = (Y - hz) / max(1, h - hz)
    grass = lin("#6e9e48") * (1 - gy[..., None]) + lin("#3f6f2c") * gy[..., None]
    xx = (X - w * 0.5) / (0.2 + gy * 1.8 + 1e-3)
    stripe = (np.floor(xx / (w * 0.08)) % 2)
    grass *= (0.92 + 0.12 * stripe)[..., None]
    pic = np.where((Y >= hz)[..., None], grass, pic)
    # building: shadow, glazed ground floor, clad upper storey, cantilevered roof, timber wing
    bx0, bx1 = w * 0.16, w * 0.82
    by0, by1 = hz - h * 0.36, hz + h * 0.05
    sh = SP.blur(SP.rect_mask(h, w, bx0 - w * 0.02, by1 - h * 0.01, bx1 + w * 0.06, by1 + h * 0.06), 4)
    pic *= (1 - 0.45 * sh)[..., None]
    mid = by0 + (by1 - by0) * 0.45
    up = SP.rect_mask(h, w, bx0, by0, bx1, mid)
    pic = pic * (1 - up[..., None]) + lin("#dcdbd5") * up[..., None] * (0.9 + 0.1 * ((X // 18) % 2))[..., None]
    gl = SP.rect_mask(h, w, bx0, mid, bx1, by1)
    gt = np.clip((Y - mid) / (by1 - mid), 0, 1)[..., None]
    glass = lin("#233746") * (1 - gt) + lin("#4f6d80") * gt
    refl = smooth(0.55, 0.8, fbm(h, w, 6, 2, seed + 252, 3))
    glass = glass * (1 - 0.35 * refl[..., None]) + lin("#a9c3d6") * 0.35 * refl[..., None]
    pic = pic * (1 - gl[..., None]) + glass * gl[..., None]
    for k in range(20):
        xm = bx0 + (bx1 - bx0) * k / 20
        pic *= (1 - 0.85 * SP.rect_mask(h, w, xm, mid, xm + 2, by1))[..., None]
    for k in range(9):
        xm = bx0 + (bx1 - bx0) * k / 9
        pic = pic * (1 - SP.rect_mask(h, w, xm, by0 + 4, xm + (bx1 - bx0) / 9 * 0.55, mid - 6)[..., None]) + \
            lin("#2b3c48") * SP.rect_mask(h, w, xm, by0 + 4, xm + (bx1 - bx0) / 9 * 0.55, mid - 6)[..., None]
    wing = SP.rect_mask(h, w, bx0 - w * 0.12, by0 + h * 0.12, bx0, by1)
    boards = 0.85 + 0.15 * ((X // 4) % 2)
    pic = pic * (1 - wing[..., None]) + lin("#a77b4f") * (boards * (0.9 + 0.2 * fbm(h, w, 30, 3, seed + 253, 2)))[..., None] * wing[..., None]
    roof = SP.rect_mask(h, w, bx0 - w * 0.04, by0 - h * 0.035, bx1 + w * 0.05, by0 + h * 0.005)
    pic = pic * (1 - roof[..., None]) + lin("#303236") * roof[..., None]
    soff = SP.blur(SP.rect_mask(h, w, bx0, by0, bx1, by0 + h * 0.06), 3)
    pic *= (1 - 0.35 * soff * (up + gl))[..., None]
    # trees in front, shaded, and figures
    for cx, cr in ((w * 0.07, 0.13), (w * 0.9, 0.12), (w * 0.96, 0.09)):
        r = cr * h
        cy = hz - r * 0.2
        m = SP.poly_mask(h, w, SP.circle_pts(cx, cy, r, 30))
        shade = np.clip((X - cx + Y - cy) / (r * 2.5) + 0.5, 0, 1)
        col = (lin("#5a8a3c") * (1 - shade[..., None]) + lin("#2a4a24") * shade[..., None]) * (0.85 + 0.3 * fbm(h, w, 40, 40, seed + 254, 2))[..., None]
        pic = pic * (1 - m[..., None]) + col * m[..., None]
        pic *= (1 - 0.6 * SP.rect_mask(h, w, cx - 2, cy + r * 0.8, cx + 2, cy + r * 1.5))[..., None]
    for k in range(9):
        fx, fy = rng.uniform(w * 0.1, w * 0.9), hz + rng.uniform(0.12, 0.3) * (h - hz)
        sc = 0.5 + (fy - hz) / (h - hz)
        m = SP.rect_mask(h, w, fx, fy - 9 * sc, fx + 2.5 * sc, fy)
        pic = pic * (1 - m[..., None]) + lin(rng.choice(["#c8102e", "#1d2c4a", "#e8e8e8"])) * m[..., None]
    for ch in range(3):
        pic[..., ch] = SP.blur(pic[..., ch], 1)
    pic *= (0.97 + 0.06 * np.random.default_rng(seed + 255).random((h, w)))[..., None]
    return np.clip(pic, 0, 1)


def hoarding_print(name="PS_HoardingPrint3", cell=2.44, seed=0):
    """The printed graphic over the hoarding's club colour (alpha cut out): headline, a rendered
    view of the new building, the project strip, contractor and enquiry lines, a site warning
    sign, splashes of mud along the bottom (metric, v = height, x from -1.22)."""
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / cell
    img = np.zeros((N, N, 4))
    X = lambda m: m * ppm
    Y = lambda z: N - z * ppm
    wh = lin("#ffffff")
    SP.text(img, "BUILDING THE FUTURE", X(1.22), Y(2.32), 0.15 * ppm, wh, "black", "c", max_w=X(2.2))
    SP.text(img, "OUR NEW FIRST TEAM TRAINING CENTRE", X(1.22), Y(2.06), 0.055 * ppm, wh, "bold", "c", max_w=X(2.2))
    # the render: sky, the building, pitches and trees, in a white keyline
    x0, x1, z0, z1 = 0.12, 2.32, 0.78, 1.9
    r0, r1, c0, c1 = int(Y(z1)), int(Y(z0)), int(X(x0)), int(X(x1))
    h, w = r1 - r0, c1 - c0
    pic = _cgi_render(h, w, seed)
    sub = img[r0:r1, c0:c1]
    sub[..., :3] = pic
    sub[..., 3] = 1
    k = 6
    for a, b in ((r0 - k, r0), (r1, r1 + k)):
        img[a:b, c0 - k:c1 + k, :3] = wh
        img[a:b, c0 - k:c1 + k, 3] = 1
    for a, b in ((c0 - k, c0), (c1, c1 + k)):
        img[r0 - k:r1 + k, a:b, :3] = wh
        img[r0 - k:r1 + k, a:b, 3] = 1
    # project strip
    s0, s1 = Y(0.68), Y(0.42)
    img[int(s0):int(s1), :, :3] = wh
    img[int(s0):int(s1), :, 3] = 1
    SP.text(img, "KERR & MASON", X(0.12), Y(0.64), 0.07 * ppm, lin("#123a5e"), "black", max_w=X(0.9))
    SP.text(img, "Site enquiries  01632 960190", X(0.12), Y(0.525), 0.035 * ppm, lin("#333333"), "bold", max_w=X(0.9))
    SP.text(img, "Considerate Constructors Scheme", X(1.34), Y(0.64), 0.03 * ppm, lin("#333333"), "bold", max_w=X(0.75))
    SP.text(img, "registered site", X(1.34), Y(0.585), 0.03 * ppm, lin("#333333"), "regular", max_w=X(0.75))
    sgn = np.zeros((int(0.3 * ppm), int(0.22 * ppm), 3))
    sgn[:] = lin("#f5c400")
    SP.sign_warning(sgn, sgn.shape[1] / 2, sgn.shape[0] * 0.35, sgn.shape[1] * 0.32)
    SP.text(sgn, "DANGER", sgn.shape[1] / 2, sgn.shape[0] * 0.66, sgn.shape[0] * 0.08, lin("#111111"), "black", "c", max_w=sgn.shape[1] * 0.9)
    SP.text(sgn, "KEEP OUT", sgn.shape[1] / 2, sgn.shape[0] * 0.8, sgn.shape[0] * 0.07, lin("#111111"), "bold", "c", max_w=sgn.shape[1] * 0.9)
    rs, cs = int(Y(0.67)), int(X(2.16))
    img[rs:rs + sgn.shape[0], cs:cs + sgn.shape[1], :3] = sgn
    img[rs:rs + sgn.shape[0], cs:cs + sgn.shape[1], 3] = 1
    # fade the print a little (more toward the top), splash mud up the bottom
    img[..., :3] *= (0.9 + 0.1 * fbm(N, N, 6, 6, seed + 211, 3))[..., None]
    g = SP.lum(img[..., :3])[..., None]
    fade = 0.12 + 0.1 * np.linspace(1, 0, N)[:, None, None]
    img[..., :3] = img[..., :3] * (1 - fade) + (g * 0.9 + 0.08) * fade
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    mud = (fbm(N, N, 90, 90, seed + 212, 3) * (1 - smooth(0.0, 0.35, V)) > 0.42)
    img[mud, :3] = lin("#5a4936") * (0.8 + 0.3 * fbm(N, N, 30, 30, seed + 213, 2)[mud][..., None])
    img[mud, 3] = 1
    return SP.pbr(name, img, rough=0.5, cell=(cell, cell), clip=True)


def woven_pp(name="PS_WovenPP", color="#dcd8cc", cell=0.5, seed=0, dirt=0.12, wall_h=1.2):
    """Woven polypropylene (bulk bags, sacks): a tape weave, slight sheen variation, dust and
    dirt rising from the foot (metric, v = height)."""
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / cell
    tape = 0.0028 * ppm
    Y, X = np.mgrid[0:N, 0:N]
    u = (X / tape) % 2
    v = (Y / tape) % 2
    warp = ((X // tape) + (Y // tape)) % 2
    sh = np.where(warp == 0, 0.92 + 0.1 * np.sin(np.pi * (Y % tape) / tape), 1.0 - 0.06 * np.sin(np.pi * (X % tape) / tape))
    img = lin(color)[None, None] * (sh * (0.92 + 0.12 * fbm(N, N, 6, 6, seed + 230, 3)))[..., None]
    d = dirt * smooth(0.45, 0.8, fbm(N, N, 4, 4, seed + 231, 4))
    img = img * (1 - d[..., None]) + lin("#9a8a6c") * d[..., None]
    h = (sh - 0.95) * 0.0006
    return SP.pbr(name, img, rough=0.72, cell=(cell, cell), normal=SP.normal_from_height(h, cell / N, 1.0))


def sand_img(name="PS_SandHeap", cell=1.0, seed=0):
    if _get(name):
        return _get(name)
    N = 512
    n = fbm(N, N, 30, 30, seed + 240, 5)
    img = lin("#b8915a")[None, None] * (0.8 + 0.35 * n)[..., None]
    g = np.random.default_rng(seed + 241).random((N, N))
    img[g > 0.985] *= 0.6
    img[g < 0.01] *= 1.3
    wet = smooth(0.5, 0.7, fbm(N, N, 4, 4, seed + 242, 3))
    img *= (1 - 0.25 * wet)[..., None]
    h = fbm(N, N, 60, 60, seed + 243, 3) * 0.002
    return SP.pbr(name, img, rough=0.95, cell=(cell, cell), normal=SP.normal_from_height(h, cell / N, 1.0))


def upvc(name="PS_uPVC2", color="#b9582b", cell=1.0, seed=0, mud=0.0):
    """Orange uPVC drainage pipe: semi-gloss, faint extrusion lines along the length (v), dust."""
    if _get(name):
        return _get(name)
    N = 512
    lines = fbm(N, N, 2, 120, seed + 270, 3)
    img = lin(color)[None, None] * (0.94 + 0.1 * lines + 0.06 * fbm(N, N, 6, 6, seed + 271, 3))[..., None]
    dust = smooth(0.5, 0.85, fbm(N, N, 3, 3, seed + 272, 4)) * 0.18
    img = img * (1 - dust[..., None]) + lin("#b8a88c") * dust[..., None]
    if mud:
        m = smooth(0.42, 0.75, fbm(N, N, 24, 24, seed + 273, 4)) * mud
        img = img * (1 - m[..., None]) + lin("#8a7658") * m[..., None]
        dust = np.maximum(dust, m)
    return SP.pbr(name, img, rough=np.clip(0.55 + 0.3 * dust, 0, 1), cell=(cell, cell))


def pipe_bore(name="PS_PipeBore2", depth=0.45):
    """The inside of a pipe seen from its end: the same orange, a little glossier, darkening a
    couple of diameters in but never black (UV v = 1 at the mouth, 0 at `depth`)."""
    if _get(name):
        return _get(name)
    H, W = 64, 8
    t = np.linspace(0, 1, H)[:, None, None]           # 0 at the mouth (row 0), 1 deep
    img = lin("#c4602c") * (1 - smooth(0.0, 0.85, t)) + lin("#3d1d0e") * smooth(0.0, 0.85, t)
    img = np.broadcast_to(img, (H, W, 3)).copy()[::-1]
    return SP.pbr(name, img, rough=0.35)


def pipe_print(name="PS_Label_PipePrint", px=(1024, 32)):
    """The black ink line along a drain pipe (UV 0-1 over one metre)."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 4))
    s = "KM PIPES  BS EN 1401-1  160  SN4  UD  PVC-U  12/03/25"
    SP.text(img, s, 10, H * 0.15, H * 0.7, lin("#101010"), "bold", max_w=W - 20)
    img[..., 3] *= fbm(H, W, 40, 2, 281, 2) > 0.15
    return SP.pbr(name, img, rough=0.4, clip=True)


def soil(name="PS_Soil", cell=0.6, seed=0, color="#5d4a36"):
    """Freshly disturbed topsoil: clods, small stones, damp patches (metric)."""
    if _get(name):
        return _get(name)
    N = 512
    rng = np.random.default_rng(seed + 300)
    n = fbm(N, N, 14, 14, seed + 301, 5)
    img = lin(color)[None, None] * (0.7 + 0.5 * n)[..., None]
    h = n * 0.01
    for _ in range(140):
        x, y = rng.uniform(0, N, 2)
        r = rng.uniform(2, 9)
        m = SP.poly_mask(N, N, SP.circle_pts(x, y, r, 10, ry=r * rng.uniform(0.6, 1.0)))
        c = lin("#8a8274") if rng.random() < 0.3 else lin(color) * rng.uniform(0.6, 1.1)
        SP.put(img, np.roll(m, 2, 0), lin("#211a12"), 0.6)
        SP.put(img, m, c)
        h = np.maximum(h, m * 0.012)
    damp = smooth(0.5, 0.7, fbm(N, N, 4, 4, seed + 302, 3))
    img *= (1 - 0.3 * damp)[..., None]
    return SP.pbr(name, img, rough=np.clip(0.95 - 0.25 * damp, 0, 1), cell=(cell, cell),
                  normal=SP.normal_from_height(SP.blur(h, 1), cell / N, 1.0))


def peg_wood(name="PS_PegWood2", paint="#ff4a14", top=0.28, seed=0):
    """A sawn softwood peg (metric, v = height, 0.4 m per repeat in v): pale yellow-brown with
    straight vertical grain streaks and a knot, weathering grey toward the ground, marking paint
    on the top 70 mm with a hard edge, a little overspray and two short runs, mud at the ground."""
    if _get(name):
        return _get(name)
    W, H = 128, 512
    cell = (0.1, 0.4)
    ppm = H / cell[1]
    rng = np.random.default_rng(seed + 309)
    # grain: thin streaks running straight up (noise stretched hard along v), early/late wood
    g = fbm(H, W, 24, 1, seed + 310, 4)
    late = smooth(0.48, 0.62, g)
    img = lin("#c8aa78")[None, None] * (1.0 - 0.22 * late[..., None]) * (0.95 + 0.08 * fbm(H, W, 2, 6, seed + 311, 3))[..., None]
    fibre = fbm(H, W, 64, 2, seed + 312, 2)
    img *= (0.94 + 0.1 * fibre)[..., None]
    kx, ky = W * 0.3, H * 0.55
    Y, X = np.mgrid[0:H, 0:W]
    k = np.clip(1 - np.hypot(X - kx, (Y - ky) * 0.4) / 6, 0, 1)
    img *= (1 - 0.5 * k)[..., None]
    V = np.repeat(((H - 0.5 - np.arange(H)) / ppm)[:, None], W, 1)
    grey = (1 - smooth(0.0, 0.15, V)) * 0.35
    img = img * (1 - grey[..., None]) + lin("#7d7466") * grey[..., None]
    edge = top - 0.07 + 0.003 * (fbm(H, W, 12, 2, seed + 313, 2) - 0.5)
    pm = smooth(edge - 0.002, edge + 0.002, V)
    over = (rng.random((H, W)) < 0.12 * smooth(edge - 0.012, edge, V)) * (V < edge)
    runs = np.zeros((H, W))
    for x in rng.uniform(0, W, 2):
        L = rng.uniform(0.012, 0.03)
        m = (np.abs(X - x) < 1.6) & (V < edge) & (V > edge - L)
        runs = np.maximum(runs, m)
    m = np.clip(pm + 0.8 * over + runs, 0, 1)
    img = img * (1 - m[..., None]) + lin(paint) * (0.9 + 0.1 * fbm(H, W, 4, 8, seed + 314, 2))[..., None] * m[..., None]
    mud = 1 - smooth(0.0, 0.04 + 0.03 * fbm(H, W, 10, 4, seed + 315, 2), V)
    img = img * (1 - 0.85 * mud[..., None]) + lin("#3b2e21") * 0.85 * mud[..., None]
    h = late * 0.0004 + fibre * 0.0002
    return SP.pbr(name, img, rough=np.clip(0.82 - 0.35 * m, 0, 1), cell=cell, normal=SP.normal_from_height(h, cell[1] / H, 1.0))


def end_grain(name="PS_PegTop", paint="#ff4a14"):
    """The hammered top of a painted peg (UV 0-1): paint over sawn end grain with growth rings
    showing through and bruising round the edge."""
    if _get(name):
        return _get(name)
    N = 128
    Y, X = np.mgrid[0:N, 0:N]
    r = np.hypot(X - N * 0.15, Y - N * 1.1) / N
    rings = 0.5 + 0.5 * np.sin(r * 70 + fbm(N, N, 6, 6, 316, 3) * 4)
    img = lin(paint)[None, None] * (0.8 + 0.25 * rings[..., None])
    bruise = smooth(0.55, 0.8, fbm(N, N, 8, 8, 317, 3))
    img = img * (1 - 0.4 * bruise[..., None]) + lin("#8a5a32") * 0.4 * bruise[..., None]
    return SP.pbr(name, img, rough=0.6)


def flag_tape(name="PS_FlagTape3"):
    """Fluorescent pink flagging tape: bright, slightly translucent-looking, faint sheen."""
    if _get(name):
        return _get(name)
    N = 64
    img = lin("#ee5596")[None, None] * (0.85 + 0.2 * fbm(N, N, 4, 16, 318, 2))[..., None]
    return SP.pbr(name, np.broadcast_to(img, (N, N, 3)).copy(), rough=0.5, cell=(0.3, 0.3))


def treated_post(name="PS_PostTreated", color="#5e5238", cell=(0.5, 2.0), seed=0, algae=0.4):
    """Tanalised/weathered post (metric, v = height up to 2 m): vertical grain, checks (splits),
    greying, green algae patches, darker and damp toward the ground."""
    if _get(name):
        return _get(name)
    W, H = 256, 1024
    ppm = H / cell[1]
    g = fbm(H, W, 6, 60, seed + 320, 5)
    rings = np.sin(g * 30) * 0.5 + 0.5
    img = lin(color)[None, None] * (0.75 + 0.3 * rings[..., None])
    grey = smooth(0.35, 0.75, fbm(H, W, 3, 6, seed + 321, 3)) * 0.3
    img = img * (1 - grey[..., None]) + lin("#8a877e") * grey[..., None] * (0.8 + 0.3 * rings[..., None])
    rng = np.random.default_rng(seed + 322)
    for _ in range(9):
        x = rng.uniform(0, W)
        y0, L = rng.uniform(0, H), rng.uniform(0.1, 0.5) * ppm
        pts = [(x - 1.2, y0), (x + 1.2, y0), (x + 0.3, y0 + L), (x - 0.3, y0 + L)]
        SP.put(img, SP.poly_mask(H, W, pts), lin("#1e1a14"), 0.85)
    al = algae * smooth(0.55, 0.75, fbm(H, W, 4, 10, seed + 323, 4))
    img = img * (1 - al[..., None]) + lin("#4c5a2a") * al[..., None]
    V = np.repeat(((H - 0.5 - np.arange(H)) / ppm)[:, None], W, 1)
    foot = 1 - smooth(0.0, 0.3 + 0.2 * fbm(H, W, 10, 20, seed + 324, 3), V)
    img = img * (1 - 0.6 * foot[..., None]) + lin("#2e2619") * 0.6 * foot[..., None]
    h = rings * 0.001
    return SP.pbr(name, img, rough=0.85, cell=cell, normal=SP.normal_from_height(h, cell[1] / H, 1.0))


def site_board_face(name="PS_Label_SiteBoard", px=(1024, 400)):
    """The contractor's board below its club-coloured header: parties to the contract, contacts,
    the contractor's mark, a row of mandatory PPE signs and a green safety strip (UV 0-1)."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 3))
    img[:] = lin("#ebebe4")
    navy, dark = lin("#123a5e"), lin("#262626")
    SP.put(img, SP.rect_mask(H, W, W * 0.03, H * 0.06, W * 0.17, H * 0.42), navy)
    SP.text(img, "KM", W * 0.1, H * 0.13, H * 0.17, lin("#ffffff"), "black", "c", max_w=W * 0.12)
    SP.put(img, SP.rect_mask(H, W, W * 0.03, H * 0.36, W * 0.17, H * 0.42), lin("#f2c200"))
    rows = [("CLIENT", "First Team Training Centre Ltd"), ("PRINCIPAL CONTRACTOR", "Kerr & Mason Construction Ltd"),
            ("ARCHITECT", "Hale Partnership"), ("SITE MANAGER", "01632 960190"), ("OUT OF HOURS", "01632 960555")]
    for i, (k, v) in enumerate(rows):
        y = H * (0.06 + i * 0.075)
        SP.text(img, k, W * 0.21, y, H * 0.03, lin("#6a6a6a"), "bold", max_w=W * 0.24)
        SP.text(img, v, W * 0.46, y - H * 0.004, H * 0.04, dark, "bold", max_w=W * 0.5)
    for i, kind in enumerate(("hat", "vest", "boot", "gloves", "eyes")):
        SP.sign_mandatory(img, W * (0.1 + i * 0.12), H * 0.62, H * 0.085, kind)
    SP.text(img, "ALL VISITORS MUST", W * 0.8, H * 0.55, H * 0.04, dark, "black", "c", max_w=W * 0.3)
    SP.text(img, "REPORT TO SITE OFFICE", W * 0.8, H * 0.62, H * 0.04, dark, "black", "c", max_w=W * 0.3)
    SP.put(img, SP.rect_mask(H, W, 0, H * 0.78, W, H), lin("#1d7a3e"))
    SP.text(img, "SAFETY FIRST  -  THINK BEFORE YOU ACT", W / 2, H * 0.845, H * 0.06, lin("#ffffff"), "black", "c", max_w=W * 0.9)
    V = np.repeat(((H - 0.5 - np.arange(H)) / H)[:, None], W, 1)
    run = SP.blur(streaks(H, W, 331, density=0.2, length=0.6, width_px=5), 2)
    img *= (1 - 0.18 * run)[..., None]
    mud = (1 - smooth(0.0, 0.12, V)) * (fbm(H, W, 60, 20, 332, 3) > 0.45)
    img = img * (1 - 0.7 * mud[..., None]) + lin("#6b5a44") * 0.7 * mud[..., None]
    img *= (0.94 + 0.06 * fbm(H, W, 6, 3, 333, 3))[..., None]
    # grime weeping from the bolt heads (at x = 0.125 / 0.875 of the width, 1/6 up)
    Y, X = np.mgrid[0:H, 0:W]
    for bx in (0.125, 0.875):
        for by in (1 - 0.15 / 0.9, 1 - 0.6 / 0.9):
            dr = np.clip(1 - np.abs(X - bx * W) / 4, 0, 1) * (Y > by * H) * np.clip(1 - (Y - by * H) / (H * 0.25), 0, 1)
            img *= (1 - 0.35 * dr)[..., None]
    return SP.pbr(name, img, rough=0.45)


def hazard_diamond(name="PS_Label_Flammable", px=128):
    """A class 3 flammable-liquid diamond (UV 0-1, square), cut out with alpha."""
    if _get(name):
        return _get(name)
    N = px
    img = np.zeros((N, N, 4))
    c = N / 2
    SP.put(img, SP.poly_mask(N, N, [(c, 1), (N - 1, c), (c, N - 1), (1, c)]), lin("#ffffff"))
    SP.put(img, SP.poly_mask(N, N, [(c, 5), (N - 5, c), (c, N - 5), (5, c)]), lin("#d2101c"))
    fl = [(c, N * 0.18), (c + N * 0.12, N * 0.4), (c + N * 0.06, N * 0.38), (c + N * 0.1, N * 0.52), (c - N * 0.1, N * 0.52),
          (c - N * 0.06, N * 0.36), (c - N * 0.1, N * 0.4)]
    SP.put(img, SP.poly_mask(N, N, fl), lin("#ffffff"))
    SP.text(img, "3", c, N * 0.66, N * 0.15, lin("#ffffff"), "bold", "c")
    img[..., 3] *= fbm(N, N, 10, 10, 401, 3) > 0.2
    return SP.pbr(name, img, rough=0.5, clip=True)


def paper_label(name="PS_Label_DrumPaper", lines=("HYDRAULIC OIL", "ISO VG 46", "205 LITRES", "Batch 2291-07"), px=(192, 256)):
    """A weathered, torn paper label (UV 0-1), alpha-cut round its ragged edge."""
    if _get(name):
        return _get(name)
    W, H = px
    img = np.zeros((H, W, 4))
    img[..., :3] = lin("#e2dccb") * (0.85 + 0.2 * fbm(H, W, 6, 8, 411, 3))[..., None]
    img[..., 3] = 1
    SP.put(img, SP.rect_mask(H, W, 0, 0, W, H * 0.22), lin("#2b4f8a"))
    SP.text(img, lines[0], W / 2, H * 0.06, H * 0.09, lin("#ffffff"), "black", "c", max_w=W * 0.9)
    for i, s_ in enumerate(lines[1:]):
        SP.text(img, s_, W / 2, H * (0.32 + i * 0.14), H * 0.07, lin("#2a2a2a"), "bold", "c", max_w=W * 0.86)
    stain = smooth(0.5, 0.75, fbm(H, W, 4, 5, 412, 4))
    img[..., :3] = img[..., :3] * (1 - 0.35 * stain[..., None]) + lin("#8a7350") * 0.35 * stain[..., None]
    Y, X = np.mgrid[0:H, 0:W]
    edge = np.minimum(np.minimum(X, W - 1 - X) / W, np.minimum(Y, H - 1 - Y) / H)
    torn = fbm(H, W, 12, 12, 413, 3)
    img[..., 3] = (edge > 0.03 + 0.08 * torn ** 2) & ~((X > W * 0.6) & (Y > H * 0.75) & (torn > 0.45))
    return SP.pbr(name, img, rough=0.8, clip=True)


TYRE_V = dict(top=(0.0, 0.25), tread=(0.27, 0.5), bottom=(0.52, 0.77), liner=(0.79, 1.0))


def tyre_img(name="PS_TyreOld", seed=0, mud=False):
    """An old car tyre unwrapped (u once round, v across the section: top sidewall, tread, bottom
    sidewall, inner liner; see TYRE_V): sidewall ribs and moulded lettering, a block tread with
    circumferential grooves and sipes, dust on the upper sidewall, mud at the bottom."""
    if _get(name):
        return _get(name)
    W, H = 1024, 512
    rng = np.random.default_rng(seed + 420)
    img = np.zeros((H, W, 3))
    base = lin("#1c1c1d")
    img[:] = base
    hgt = np.zeros((H, W))
    row = lambda v: int(v * H)          # v measured from the top row (row 0 = v 0)
    t0, t1 = row(TYRE_V["tread"][0]), row(TYRE_V["tread"][1])
    # tread blocks: grooves round the tyre, slanted lateral grooves
    Y, X = np.mgrid[0:H, 0:W]
    tv = (Y - t0) / max(1, t1 - t0)
    tread = (Y >= t0) & (Y < t1)
    circ = np.zeros((H, W), bool)
    for g in (0.22, 0.42, 0.62, 0.8):
        circ |= np.abs(tv - g) < 0.025
    lat = ((X + (tv - 0.5) * 60 * np.sign(tv - 0.5)) % 34) < 6
    lat &= (tv < 0.22) | (tv > 0.8) | ((tv > 0.42) & (tv < 0.62) & (((X + 17) % 34) < 6))
    sipe = ((X + tv * 20) % 11) < 1.2
    groove = tread & (circ | lat)
    hgt[tread] = 1.0
    hgt[groove] = 0.0
    hgt[tread & sipe & ~groove] = 0.7
    worn = tread & ~groove
    img[worn] = lin("#2a2a2a")
    img[groove] = lin("#0d0d0d")
    # sidewalls: concentric ribs, a raised band with the maker's lettering
    for key, letters in (("top", True), ("bottom", False)):
        a, b = row(TYRE_V[key][0]), row(TYRE_V[key][1])
        for k in range(4):
            r = a + int((b - a) * (0.18 + 0.05 * k))
            hgt[r:r + 2] = 0.5
        if letters:
            band = np.zeros((b - a, W, 3))
            for i, (s, cap, y) in enumerate((("MERIDIAN", 0.2, 0.38), ("205/55 R16 91V", 0.12, 0.42), ("TUBELESS", 0.08, 0.45))):
                for rep in range(1):
                    x = W * (0.1 + i * 0.28 + rep * 0.5)
                    SP.text(band, s, x, (b - a) * (y - cap / 2 - 0.1), (b - a) * cap, np.ones(3), "black" if i == 0 else "bold")
            m = band[..., 0][::-1]
            img[a:b] = img[a:b] * (1 - m[..., None]) + lin("#585858") * m[..., None]
            hgt[a:b] = np.maximum(hgt[a:b], m * 0.8)
    # liner: a little shinier, greyer
    a = row(TYRE_V["liner"][0])
    img[a:] = lin("#262626") * (0.9 + 0.2 * fbm(H - a, W, 8, 4, seed + 421, 3))[..., None]
    # dust on the upper sidewall and shoulder, mud on the lower
    a, b = row(TYRE_V["top"][0]), row(TYRE_V["tread"][0]) + 10
    d = smooth(0.45, 0.8, fbm(b - a, W, 12, 6, seed + 422, 4)) * 0.25
    img[a:b] = img[a:b] * (1 - d[..., None]) + lin("#7a7466") * d[..., None]
    if mud:
        a, b = row(TYRE_V["tread"][1]) - 20, row(TYRE_V["bottom"][1])
        d = smooth(0.3, 0.6, fbm(b - a, W, 16, 6, seed + 423, 4)) * 0.8
        img[a:b] = img[a:b] * (1 - 0.6 * d[..., None]) + lin("#2e2a24") * 0.6 * d[..., None]
    img *= (0.9 + 0.2 * fbm(H, W, 30, 10, seed + 424, 3))[..., None]
    rough = np.where(Y >= row(TYRE_V["liner"][0]), 0.6, 0.85)
    nrm = SP.normal_from_height(SP.blur(hgt, 1) * 0.003, 0.002, 1.0)
    return SP.pbr(name, img, rough=rough, normal=nrm)


def wear_soft(name, cell=6.4, wall_h=2.62, seed=0, chips=0.5, foot=0.15, runs=0.4, scrapes=8, top_bleed=0.4,
              fixings=(), bands=(), roof=0.0, roof_v0=2.8, roof_w=2.45, algae=0.0, px=1024, mud=0.0, seam=False,
              scrape_band=None):
    """Layered paint wear blended over painted steel (soft alpha, metric UVs, v = height):
    chips with a primer-grey rim, dark brown rust and an orange heart in the worst; brown runs
    hanging below chips, fixings and the top edge; rust and dirt along the foot; bright scrapes
    through to primer and steel; optional chipped bands (rolled edges, hoops), mud splash,
    algae, and rust pooling on a roof band."""
    if _get(name):
        return _get(name)
    N = px
    ppm = N / cell
    V = np.repeat(((N - 0.5 - np.arange(N)) / ppm)[:, None], N, 1)
    wall = V < wall_h
    rng = np.random.default_rng(seed + 500)
    primer, steel = lin("#8d8a83"), lin("#a9a8a3")
    dark, mid, orange = lin("#3d2418"), lin("#5c3220"), lin("#8c4a22")
    rgb = np.zeros((N, N, 3))
    a = np.zeros((N, N))

    def lay(m, c, alpha=1.0):
        nonlocal rgb, a
        m = np.clip(m, 0, 1) * alpha
        rgb = rgb * (1 - m[..., None]) + np.asarray(c)[None, None] * m[..., None] if np.ndim(c) == 1 else rgb * (1 - m[..., None]) + c * m[..., None]
        a = np.maximum(a, m) if alpha >= 0.99 else a + m * (1 - a)
    # chip field: clustered toward the foot, the top edge and fixings
    cl = fbm(N, N, 5, 3, seed + 501, 3)
    bias = 0.86 + 0.22 * (smooth(wall_h - 0.4, wall_h, V) + (1 - smooth(0.0, 0.5, V)))
    f = fbm(N, N, 60, 60, seed + 502, 4) * (0.6 + 0.5 * cl) * bias
    t = 0.62 - 0.12 * chips
    for z, pitch in fixings:
        n = max(1, int(round(cell / pitch)))
        for k in range(n):
            cx, cy = (k + 0.5) * N / n + rng.uniform(-3, 3), N - z * ppm
            f = np.maximum(f, (t + 0.06) * SP.poly_mask(N, N, SP.circle_pts(cx, cy, 0.012 * ppm * rng.uniform(0.6, 1.4), 12)))
    for z, hw_ in bands:
        g = fbm(N, N, 40, 3, seed + 503 + int(z * 100), 3)
        f = np.maximum(f, (np.abs(V - z) < hw_ * (0.5 + g)) * (t + 0.08 * g) * (g > 0.35))
    rim = smooth(t - 0.035, t - 0.01, f)
    core = smooth(t - 0.005, t + 0.015, f)
    heart = smooth(t + 0.09, t + 0.13, f)
    # runs below chips and the top edge
    src = SP.blur(core, 1)
    run = np.zeros((N, N))
    acc = np.zeros(N)
    for r in range(N):
        acc = np.maximum(acc * (1 - 1 / (0.35 * ppm)), src[r])
        run[r] = acc
    jitter = fbm(N, N, 90, 4, seed + 504, 2)
    run *= smooth(0.3, 0.7, jitter)
    tb = SP.blur(streaks(N, N, seed + 505, density=0.3, length=0.4, width_px=4), 1)
    top = int(round(N - wall_h * ppm))
    tbb = np.zeros((N, N))
    tbb[max(top, 0):] = tb[:N - max(top, 0)]
    run = np.maximum(run * runs, tbb * top_bleed)
    lay(run * 0.75, mid * 0.9 + dark * 0.1, 0.65)
    # foot: rust and dirt eating up from the bottom edge
    e = fbm(N, N, 50, 10, seed + 506, 4)
    fz = (1 - smooth(0.0, 0.03 + foot * (0.4 + e), V)) * smooth(0.35, 0.6, fbm(N, N, 7, 2, seed + 507, 3))
    lay(fz, dark * (0.8 + 0.4 * e[..., None]), 0.9)
    if mud > 0:
        md = (1 - smooth(0.0, mud * (0.5 + e), V)) * smooth(0.45, 0.6, fbm(N, N, 80, 30, seed + 508, 3))
        lay(md, lin("#4f4232"), 0.85)
    if algae > 0:
        al = (1 - smooth(0.0, 0.6, V)) * smooth(0.5, 0.7, fbm(N, N, 4, 2, seed + 509, 4)) * algae
        lay(al * smooth(0.4, 0.7, fbm(N, N, 40, 40, seed + 510, 3)), lin("#4d5a2c"), 0.6)
    lay(rim, primer)
    lay(core, (dark * 0.45 + mid * 0.55) * (0.85 + 0.3 * jitter[..., None]))
    lay(heart, orange * (0.8 + 0.4 * jitter[..., None]))
    # scrapes through to primer and bright steel along the lower panels
    for _ in range(scrapes):
        lo, hi = scrape_band or (0.2, min(1.3, wall_h * 0.6))
        z0, x0 = rng.uniform(lo, hi), rng.uniform(0, N)
        L = rng.uniform(0.15, 0.8) * ppm
        ang = rng.uniform(-0.08, 0.08)
        w = rng.uniform(1.0, 2.2)
        pts = [(x0, N - z0 * ppm), (x0 + L, N - z0 * ppm - L * ang), (x0 + L, N - z0 * ppm - L * ang + w), (x0, N - z0 * ppm + w * 0.6)]
        m = SP.poly_mask(N, N, pts) + SP.poly_mask(N, N, [(x - N, y) for x, y in pts])
        lay(SP.blur(m, 1) * 1.3, steel if rng.random() < 0.5 else primer)
    if seam:                                              # the side seam weld down the drum
        sm = np.clip(1 - np.abs(np.arange(N) - 2.0) / 2.5, 0, 1)[None, :] * np.ones((N, 1))
        lay(sm * 0.7, dark * 0.6 + primer * 0.4)
    a *= wall
    if roof > 0:
        rf = (V >= roof_v0) & (V < roof_v0 + roof_w)
        pool = fbm(N, N, 6, 6, seed + 511, 4)
        st = smooth(0.58, 0.72, pool) * smooth(0.4, 0.7, fbm(N, N, 30, 30, seed + 512, 3)) * roof
        rgb = np.where(rf[..., None], rgb * (1 - st[..., None]) + (dark * 0.6 + mid * 0.4) * st[..., None], rgb)
        a = np.where(rf, np.maximum(a, st * 0.6), a)
    rough = np.clip(0.6 + 0.3 * core - 0.15 * (rgb[..., 0] > 0.3), 0, 1)
    return SP.pbr(name, np.concatenate([rgb, a[..., None]], -1), rough=rough, cell=(cell, cell), blend=True)


def blind(name="PS_Blind"):
    """A white venetian blind seen through glass, one 0.5 m wide repeat across (UV u in repeats,
    v 0-1 down its drop)."""
    if _get(name):
        return _get(name)
    W, H = 64, 256
    img = np.zeros((H, W, 3))
    y = np.arange(H)[:, None]
    slat = ((y % 10) < 7).astype(float)
    img[:] = lin("#d8d6cf") * (0.75 + 0.25 * slat)[..., None] * (0.85 + 0.15 * np.sin(np.pi * (y % 10) / 10))[..., None]
    img *= (0.9 + 0.1 * fbm(H, W, 2, 8, 701, 2))[..., None]
    return SP.pbr(name, img, rough=0.6)


def room(name="PS_Room"):
    """The dim inside of a cabin seen through a window: a dark wall with a little variation."""
    if _get(name):
        return _get(name)
    N = 64
    img = lin("#2b2d2e")[None, None] * (0.8 + 0.4 * fbm(N, N, 3, 3, 702, 3))[..., None]
    return SP.pbr(name, np.broadcast_to(img, (N, N, 3)).copy(), rough=0.9, cell=(1.5, 1.5))


def cabin_roof(name="PS_CabinRoof2", color="#9a9c99", L=9.75, D=3.05, cell=10.0):
    """A cabin's profiled steel roof, mapped once over the whole roof (u = y, v = x in metres,
    wrapping at `cell`): water ponding tide marks across the middle, dirt washed to the edges,
    moss and dark silt packed into the corners, rust round the fixings."""
    if _get(name):
        return _get(name)
    N = 1024
    ppm = N / cell
    u = (np.arange(N) + 0.5) / ppm
    v = (N - 0.5 - np.arange(N)) / ppm
    U, Vv = np.meshgrid(np.where(u > cell / 2, u - cell, u), np.where(v > cell / 2, v - cell, v))
    y, x = U, Vv                                   # roof coordinates
    img = lin(color)[None, None] * (0.88 + 0.2 * fbm(N, N, 10, 10, 711, 4))[..., None]
    ex = np.clip((np.abs(x) - (L / 2 - 0.6)) / 0.6, 0, 1)
    ey = np.clip((np.abs(y) - (D / 2 - 0.45)) / 0.45, 0, 1)
    edge = np.maximum(ex, ey)
    corner = ex * ey
    dirt = smooth(0.2, 1.0, edge) * (0.6 + 0.5 * fbm(N, N, 30, 30, 712, 3))
    img = img * (1 - 0.45 * dirt[..., None]) + lin("#4a4234") * 0.45 * dirt[..., None]
    pond = fbm(N, N, 4, 4, 713, 4) * (1 - edge)
    lvl = smooth(0.5, 0.56, pond)
    tide = np.exp(-((pond - 0.53) / 0.012) ** 2) + 0.6 * np.exp(-((pond - 0.6) / 0.01) ** 2)
    img = img * (1 - 0.25 * lvl[..., None]) + lin("#585a58") * 0.25 * lvl[..., None]
    img = img * (1 - 0.35 * tide[..., None]) + lin("#5e5243") * 0.35 * tide[..., None]
    moss = smooth(0.25, 0.7, corner + 0.25 * smooth(0.6, 1, edge) * fbm(N, N, 20, 20, 714, 3)) * smooth(0.35, 0.6, fbm(N, N, 40, 40, 715, 3))
    img = img * (1 - moss[..., None]) + lin("#3f4a22") * moss[..., None] * (0.7 + 0.5 * fbm(N, N, 80, 80, 716, 2))[..., None]
    rough = np.clip(0.5 + 0.35 * dirt + 0.4 * moss - 0.2 * lvl, 0, 1)
    return SP.pbr(name, img, rough=rough, cell=(cell, cell))


def soil_fine(name="PS_SoilFine", cell=0.5, seed=0):
    """Freshly broken topsoil seen close: dark crumbly brown, small crumbs, a few grass roots."""
    if _get(name):
        return _get(name)
    N = 512
    n = fbm(N, N, 40, 40, seed + 720, 5)
    img = lin("#4a3a2a")[None, None] * (0.7 + 0.5 * n)[..., None]
    crumbs = fbm(N, N, 160, 160, seed + 721, 2)
    img *= (0.85 + 0.3 * smooth(0.55, 0.7, crumbs))[..., None]
    roots = fbm(N, N, 6, 60, seed + 722, 3)
    img = np.where((np.abs(roots - 0.5) < 0.0025)[..., None], lin("#7a6a48"), img)
    return SP.pbr(name, img, rough=0.95, cell=(cell, cell), normal=SP.normal_from_height(n * 0.004 + crumbs * 0.002, cell / N, 1.0))


def tinted_glass(name="PS_TintedGlass"):
    """Tinted cab glazing: dark green-grey, glossy, partly see-through (blended)."""
    if _get(name):
        return _get(name)
    N = 16
    img = np.zeros((N, N, 4))
    img[..., :3] = lin("#2a3833")
    img[..., 3] = 0.62
    return SP.pbr(name, img, rough=0.05, blend=True)


def mud_track(name="PS_MudTrack", cell=1.0):
    """Trodden, rutted mud with grass gone at the middle and creeping back at the edges, cut out
    with soft alpha (metric, tileable)."""
    if _get(name):
        return _get(name)
    N = 256
    n = fbm(N, N, 8, 8, 730, 5)
    img = lin("#4a3b2b")[None, None] * (0.7 + 0.5 * n)[..., None]
    wet = smooth(0.55, 0.7, fbm(N, N, 4, 4, 731, 3))
    img *= (1 - 0.3 * wet)[..., None]
    a = smooth(0.32, 0.5, fbm(N, N, 6, 6, 732, 4))
    return SP.pbr(name, np.concatenate([img, a[..., None]], -1), rough=np.clip(0.95 - 0.4 * wet, 0, 1), cell=(cell, cell), blend=True)


def lid_stain(name="PS_LidStain"):
    """Oil and dust on a drum lid (UV 0-1 over the lid): dark glossy spills round the bungs, grey
    dust settled in the pressed ring, soft alpha."""
    if _get(name):
        return _get(name)
    N = 256
    Y, X = np.mgrid[0:N, 0:N]
    rgb = np.zeros((N, N, 3))
    a = np.zeros((N, N))
    for (bx, by, r) in ((0.5 + 0.17 / 0.5, 0.5 - 0.04 / 0.5 * 0.5, 0.12), (0.5 - 0.18 / 0.5, 0.5 + 0.05 / 0.5 * 0.5, 0.09)):
        d = np.hypot(X / N - (bx * 0.5 + 0.25 * 0), Y / N - by) / r
        n = fbm(N, N, 6, 6, int(bx * 100), 3)
        m = smooth(1.0, 0.4, d + 0.4 * n)
        rgb = rgb * (1 - m[..., None]) + lin("#111210") * m[..., None]
        a = np.maximum(a, m * 0.8)
    dust = smooth(0.5, 0.75, fbm(N, N, 8, 8, 741, 4)) * (np.hypot(X / N - 0.5, Y / N - 0.5) < 0.48)
    rgb = rgb * (1 - dust[..., None]) + lin("#8b857a") * dust[..., None]
    a = np.maximum(a, dust * 0.55)
    return SP.pbr(name, np.concatenate([rgb, a[..., None]], -1), rough=0.4, blend=True)


def bag_dirt(name="PS_BagDirt"):
    """Dirt and sand dust on a bulk bag (soft alpha, metric UVs, v = height from the ground): a
    splash band up from the pallet, finger marks and sand dust higher up."""
    if _get(name):
        return _get(name)
    N = 512
    cell = 1.2
    V = np.repeat(((N - 0.5 - np.arange(N)) / N * cell)[:, None], N, 1)
    n = fbm(N, N, 12, 8, 750, 4)
    band = (1 - smooth(0.12, 0.45 + 0.15 * n, V)) * (0.55 + 0.45 * n)
    dust = smooth(0.6, 0.85, fbm(N, N, 6, 6, 751, 4)) * 0.2
    a = np.clip(np.maximum(band * 0.8, dust), 0, 1)
    rgb = lin("#6b5a44") * band[..., None] + lin("#b39a6e") * (1 - band[..., None])
    return SP.pbr(name, np.concatenate([rgb, a[..., None]], -1), rough=0.85, cell=(cell, cell), blend=True)


def mud_strip(name="PS_MudStrip"):
    """A worn, muddy strip (UV 0-1: u across, v along): bare trodden mud down the middle with two
    darker wheel/hoof ruts and a puddle tint, fading out through broken turf at the edges."""
    if _get(name):
        return _get(name)
    W, H = 128, 512
    u = (np.arange(W) + 0.5) / W
    U = np.repeat(u[None, :], H, 0)
    n = fbm(H, W, 6, 24, 760, 4)
    across = 1 - np.abs(U - 0.5) * 2
    a = smooth(0.1, 0.6, across + 0.35 * (n - 0.5)) * smooth(0.0, 0.08, np.repeat(np.linspace(0, 1, H)[:, None], W, 1)) \
        * smooth(0.0, 0.08, np.repeat(np.linspace(1, 0, H)[:, None], W, 1))
    a *= 0.6 + 0.5 * smooth(0.3, 0.6, fbm(H, W, 20, 60, 761, 3))
    rgb = lin("#4f3e2c")[None, None] * (0.75 + 0.45 * fbm(H, W, 16, 40, 762, 4))[..., None]
    rut = np.exp(-((U - 0.35) / 0.05) ** 2) + np.exp(-((U - 0.65) / 0.05) ** 2)
    rgb *= (1 - 0.35 * rut)[..., None]
    wet = smooth(0.6, 0.75, fbm(H, W, 3, 8, 763, 3)) * rut
    rgb = rgb * (1 - 0.4 * wet[..., None]) + lin("#3a3b36") * 0.4 * wet[..., None]
    return SP.pbr(name, np.concatenate([rgb, np.clip(a, 0, 1)[..., None]], -1), rough=np.clip(0.95 - 0.5 * wet, 0, 1), blend=True)


def scuff_ring(name="PS_ScuffRing"):
    """Broken soil round a driven peg (UV 0-1): crumbs and torn turf, soft-edged, mostly near the
    centre."""
    if _get(name):
        return _get(name)
    N = 256
    Y, X = np.mgrid[0:N, 0:N]
    r = np.hypot(X / N - 0.5, Y / N - 0.5) * 2
    n = fbm(N, N, 10, 10, 770, 4)
    a = smooth(1.0, 0.35, r + 0.35 * (n - 0.5)) * (0.55 + 0.6 * smooth(0.35, 0.6, fbm(N, N, 40, 40, 771, 3)))
    rgb = lin("#4b3a29")[None, None] * (0.7 + 0.5 * n)[..., None]
    return SP.pbr(name, np.concatenate([rgb, np.clip(a, 0, 1)[..., None]], -1), rough=0.95, blend=True)


def peg_marking(name="PS_PegMarking"):
    """Marker-pen chainage and level on a peg face (UV 0-1), cut out."""
    if _get(name):
        return _get(name)
    W, H = 96, 128
    img = np.zeros((H, W, 4))
    k = lin("#1b1b2a")
    SP.text(img, "CH120", W / 2, H * 0.12, H * 0.2, k, "narrow", "c", max_w=W * 0.95)
    SP.text(img, "+0.35", W / 2, H * 0.52, H * 0.2, k, "narrow", "c", max_w=W * 0.95)
    return SP.pbr(name, img, rough=0.7, clip=True)
