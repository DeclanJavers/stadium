"""Office, recruitment and data-room furniture for a club's scouting building (kit/site/interior.py):
scouts' desks from a single-screen desk to an analyst's triple-screen sit-stand station, bench
desking for an open-plan office, filing cabinets and archive shelving of season reports, a
whiteboard of targets and the magnetic squad-planning board, the match-viewing video wall, a TV
unit with its DVD library, sofas and coffee tables, meeting and boardroom tables with their chairs,
server racks, acoustic phone booths, a kitchenette and framed signed shirts. Real sizes; origin on
the floor at the footprint centre, +Y the front (the user's side; wall units: the back against
the wall at y = 0, facing +y). Desks and tables bring their chairs.
"""
import math

import numpy as np

from .texpaint import Canvas, box_uv, decal, noise, project
from .props import PM, _paint, _plastic, clear_glass, tyre
from .props_amenity import _tube
from .props_medical import _cushion_canvas, _monitor, _white

NAMES = ("Okafor", "Lindqvist", "Moreau", "Santos", "Kowalski", "Brennan", "Haddad", "Ivanov", "Tanaka",
         "Mendes", "Novak", "Duarte", "Keane", "Oyelaran", "Petit", "Ruiz", "Hansen", "Varga", "Bakker",
         "Costa", "Murphy", "Diallo", "Nilsen", "Ferreira")


# -- placing a sub-part ----------------------------------------------------------------------------

def _place(dst, src, x, y, yaw, z=0.0):
    """Append the faces of `src` to `dst`, turned by yaw about z and moved to (x, y, z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    base = len(dst.V)
    dst.V.extend((x + vx * c - vy * s, y + vx * s + vy * c, z + vz) for vx, vy, vz in src.V)
    for f, mi, uv, rad in zip(src.F, src.mi, src.uv, src.rad):
        dst.F.append(tuple(i + base for i in f))
        dst.mi.append(dst.m(src.mats[mi]))
        dst.uv.append(uv)
        dst.rad.append(rad)


def _task_chair(M, seat="#33363a"):
    """A mesh-back task chair facing +y (five-star base on castors, gas lift, adjustable arms)."""
    p = PM(0.004)
    mesh = _plastic("Off_ChairMesh", "#2a2c2f", 0.7)
    for k in range(5):
        a = math.pi / 2 + k * 2 * math.pi / 5
        e = (0.31 * math.cos(a), 0.31 * math.sin(a))
        p.r = 0.0
        _tube(p, [(0, 0, 0.1), (e[0], e[1], 0.07)], 0.017, M["black"], 8)
        p.cyl((e[0], e[1], 0.07), (e[0], e[1], 0.05), 0.011, M["black"], 8)
        tyre(p, (e[0], e[1], 0.03), 0.03, 0.03, 0.015, M["black"], M["black"], sides=12, axis="x")
    p.cyl((0, 0, 0.1), (0, 0, 0.26), 0.03, M["black"], 14)
    p.cyl((0, 0, 0.26), (0, 0, 0.42), 0.014, M["chrome"], 12)
    p.r = 0.04
    p.box((-0.24, -0.24, 0.42), (0.24, 0.22, 0.48), _plastic("Off_ChairSeat_" + seat[1:], seat, 0.6))
    p.r = 0.006
    p.box((-0.1, -0.27, 0.42), (0.1, -0.24, 0.6), M["black"])
    p.r = 0.02
    p.obox((0, -0.3, 0.8), (0.44, 0.04, 0.45), mesh, pitch=math.radians(-8))
    p.r = 0.008
    p.obox((0, -0.29, 0.66), (0.3, 0.05, 0.08), M["black"], pitch=math.radians(-8))
    for s in (-1, 1):
        p.r = 0.005
        p.box((s * 0.25 - 0.015, -0.12, 0.44), (s * 0.25 + 0.015, -0.08, 0.64), M["black"])
        p.box((s * 0.25 - 0.035, -0.18, 0.64), (s * 0.25 + 0.035, 0.08, 0.67), M["black"])
    return p


def _meeting_chair(M, fabric):
    """A cantilever meeting chair facing +y: a chrome sled frame, an upholstered seat and back."""
    p = PM(0.004)
    p.r = 0.0
    for s in (-1, 1):
        x = s * 0.22
        _tube(p, [(x, 0.25, 0.0), (x, -0.25, 0.0), (x, -0.2, 0.45), (x, 0.22, 0.45)], 0.011, M["chrome"], 8, bend=0.05)
        _tube(p, [(x, -0.2, 0.45), (x, -0.25, 0.9)], 0.011, M["chrome"], 8)
    p.r = 0.03
    p.box((-0.24, -0.2, 0.45), (0.24, 0.24, 0.52), fabric)
    p.obox((0, -0.24, 0.73), (0.46, 0.05, 0.38), fabric, pitch=math.radians(-10))
    return p


def _exec_chair(M, leather):
    """A high-back boardroom chair facing +y: polished five-star base, padded leather seat and back."""
    p = PM(0.004)
    for k in range(5):
        a = math.pi / 2 + k * 2 * math.pi / 5
        e = (0.33 * math.cos(a), 0.33 * math.sin(a))
        p.r = 0.0
        _tube(p, [(0, 0, 0.11), (e[0], e[1], 0.07)], 0.02, M["chrome"], 8)
        tyre(p, (e[0], e[1], 0.03), 0.03, 0.03, 0.015, M["black"], M["black"], sides=12, axis="x")
    p.cyl((0, 0, 0.11), (0, 0, 0.44), 0.025, M["chrome"], 14)
    p.r = 0.05
    p.box((-0.26, -0.24, 0.44), (0.26, 0.26, 0.54), leather)
    p.obox((0, -0.3, 0.88), (0.5, 0.09, 0.7), leather, pitch=math.radians(-10))
    for s in (-1, 1):
        p.r = 0.008
        _tube(p, [(s * 0.28, 0.12, 0.5), (s * 0.28, 0.12, 0.66), (s * 0.28, -0.2, 0.66), (s * 0.28, -0.24, 0.52)], 0.012, M["chrome"], 8, bend=0.04)
        p.r = 0.01
        p.box((s * 0.28 - 0.03, -0.18, 0.66), (s * 0.28 + 0.03, 0.12, 0.69), leather)
    return p


# -- screens ---------------------------------------------------------------------------------------

def _broadcast(reg, x0, y0, x1, y1, seed):
    """A match on a broadcast camera: striped turf, touchline and halfway line, the centre circle,
    two teams, the crowd and the ad boards, the score bug."""
    W, H = x1 - x0, y1 - y0
    rng = np.random.default_rng(seed)
    inside = lambda X, Y: (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    reg.rect(x0, y0, x1, y1, col="#3a7a33", emit="#3a7a33")
    sk = 0.25 + 0.2 * rng.random()
    reg.field(lambda X, Y: inside(X, Y) * ((np.floor(((X - x0) + (Y - y0) * sk) / (W * 0.09)) % 2) == 0) * 0.45, col="#5a9d48", emit="#5a9d48")
    yt = y1 - H * 0.22
    reg.rect(x0, yt + H * 0.035, x1, y1, col="#24272c", emit="#24272c")
    reg.field(lambda X, Y: inside(X, Y) * (Y > yt + H * 0.06) * noise(X.shape, 90, seed, 2) * 0.8, col="#9aa0aa", emit="#9aa0aa")
    xs = x0
    for k in range(8):
        w = W * (0.1 + 0.08 * rng.random())
        c = ("#d8dce2", "#1d4f91", "#c8202a", "#f0c030", "#101418")[int(rng.integers(0, 5))]
        reg.rect(xs, yt, min(xs + w, x1), yt + H * 0.035, col=c, emit=c)
        xs += w
    lw = max(H * 0.006, 0.0015)
    reg.line([(x0, yt - H * 0.02), (x1, yt - H * 0.02)], lw, col="#e6ece6", emit="#e6ece6")
    hx = x0 + W * (0.3 + 0.4 * rng.random())
    reg.line([(hx + W * 0.06, yt - H * 0.02), (hx - W * 0.08, y0)], lw, col="#e6ece6", emit="#e6ece6")
    cy = y0 + (yt - y0) * 0.45
    ell = [(hx - W * 0.01 + W * 0.17 * math.cos(t), cy + H * 0.14 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 41)]
    reg.line(ell, lw, col="#e6ece6", emit="#e6ece6")
    for k in range(14):
        px = x0 + W * (0.05 + 0.9 * rng.random())
        py = y0 + (yt - y0) * (0.08 + 0.8 * rng.random())
        s = 1.0 - 0.4 * (py - y0) / max(yt - y0, 1e-6)
        c = "#c8202a" if k % 2 else "#f2f2f2"
        reg.rect(px, py - H * 0.004, px + W * 0.012 * s, py, col="#1f3d1c", emit="#1f3d1c")
        reg.rect(px, py, px + W * 0.008 * s, py + H * 0.05 * s, col=c, emit=c)
    bx, by = x0 + W * (0.3 + 0.4 * rng.random()), y0 + (yt - y0) * 0.4
    reg.circle(bx, by, H * 0.008, col="#ffffff", emit="#ffffff")
    reg.rect(x0 + W * 0.03, y1 - H * 0.1, x0 + W * 0.3, y1 - H * 0.035, col="#111418", emit="#111418")
    reg.text(f"HOM {seed % 3}-{(seed // 3) % 2} AWY  {20 + seed * 7 % 70}:{seed * 13 % 60:02d}", x0 + W * 0.04, y1 - H * 0.085, H * 0.04,
             weight="bold", col="#ffffff", emit="#ffffff", max_w=W * 0.25)


def _pitch_top(reg, x0, y0, x1, y1, seed, heat=True):
    """A top-down pitch with its markings; a player's heat map over it."""
    W, H = x1 - x0, y1 - y0
    reg.rect(x0, y0, x1, y1, col="#2f6a2c", emit="#2f6a2c")
    lw = max(H * 0.01, 0.0015)
    for pts in ([(x0 + W * 0.03, y0 + H * 0.05), (x1 - W * 0.03, y0 + H * 0.05), (x1 - W * 0.03, y1 - H * 0.05), (x0 + W * 0.03, y1 - H * 0.05), (x0 + W * 0.03, y0 + H * 0.05)],
                [(x0 + W / 2, y0 + H * 0.05), (x0 + W / 2, y1 - H * 0.05)],
                [(x0 + W * 0.03, y0 + H * 0.3), (x0 + W * 0.17, y0 + H * 0.3), (x0 + W * 0.17, y1 - H * 0.3), (x0 + W * 0.03, y1 - H * 0.3)],
                [(x1 - W * 0.03, y0 + H * 0.3), (x1 - W * 0.17, y0 + H * 0.3), (x1 - W * 0.17, y1 - H * 0.3), (x1 - W * 0.03, y1 - H * 0.3)]):
        reg.line(pts, lw, col="#d8e4d8", emit="#d8e4d8")
    reg.line([(x0 + W / 2 + H * 0.15 * math.cos(t), y0 + H / 2 + H * 0.15 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 33)], lw, col="#d8e4d8", emit="#d8e4d8")
    if heat:
        rng = np.random.default_rng(seed)
        cx, cy = x0 + W * (0.3 + 0.4 * rng.random()), y0 + H * (0.2 + 0.6 * rng.random())

        def blob(X, Y):
            m = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
            return m * np.exp(-(((X - cx) / (W * 0.16)) ** 2 + ((Y - cy) / (H * 0.22)) ** 2))
        reg.field(lambda X, Y: np.clip(blob(X, Y) * 1.2, 0, 1) * 0.8, col="#e0c030", emit="#e0c030")
        reg.field(lambda X, Y: np.clip(blob(X, Y) * 2.0 - 1.0, 0, 1) * 0.9, col="#d03020", emit="#d03020")


def _display(reg, kind, seed=1):
    """A lit screen's contents: a match feed, an analysis tool, a player dashboard, a shortlist."""
    W, H = reg.w, reg.h
    rng = np.random.default_rng(seed)
    if kind == "match":
        _broadcast(reg, 0, 0, W, H, seed)
    elif kind == "video":
        reg.fill(col="#1b1e23", emit="#1b1e23")
        _broadcast(reg, W * 0.02, H * 0.22, W * 0.72, H * 0.96, seed)
        reg.rect(W * 0.02, H * 0.05, W * 0.72, H * 0.17, col="#2a2e35", emit="#2a2e35")
        for k in range(18):
            x = W * (0.03 + 0.68 * rng.random())
            c = ("#3fd06a", "#e0b040", "#4aa0f0", "#e0503a")[k % 4]
            reg.rect(x, H * (0.06 + 0.03 * (k % 3)), x + W * 0.012, H * (0.085 + 0.03 * (k % 3)), col=c, emit=c)
        reg.rect(W * 0.3, H * 0.05, W * 0.302, H * 0.17, col="#ffffff", emit="#ffffff")
        for i in range(9):
            y = H * (0.88 - i * 0.09)
            reg.rect(W * 0.75, y, W * 0.98, y + H * 0.07, col="#252a31", emit="#252a31")
            reg.text(f"{i + 1:02d}  {('Press', 'Cross', 'Shot', 'Duel', 'Pass', 'Run')[i % 6]} {seed + i * 7 % 90}'", W * 0.765, y + H * 0.02, H * 0.035,
                     col="#c8d0da", emit="#c8d0da", max_w=W * 0.2)
    elif kind == "data":
        reg.fill(col="#10151c", emit="#10151c")
        reg.rect(0, H * 0.9, W, H, col="#1d4f91", emit="#1d4f91")
        reg.text(f"PLAYER PROFILE  {NAMES[seed % len(NAMES)].upper()}", W * 0.02, H * 0.925, H * 0.05, weight="bold", col="#ffffff", emit="#ffffff", max_w=W * 0.9)
        cx, cy, R = W * 0.25, H * 0.45, min(W * 0.2, H * 0.36)
        n = 8
        for k in range(n):
            a = 2 * math.pi * k / n
            reg.line([(cx, cy), (cx + R * math.cos(a), cy + R * math.sin(a))], H * 0.004, col="#3a4452", emit="#3a4452")
        for f in (0.33, 0.66, 1.0):
            reg.line([(cx + R * f * math.cos(2 * math.pi * k / n), cy + R * f * math.sin(2 * math.pi * k / n)) for k in range(n + 1)], H * 0.004, col="#3a4452", emit="#3a4452")
        vals = [0.35 + 0.6 * rng.random() for _ in range(n)]
        pts = [(cx + R * v * math.cos(2 * math.pi * k / n), cy + R * v * math.sin(2 * math.pi * k / n)) for k, v in enumerate(vals)]
        reg.poly(pts, col="#2f8fd8", emit="#2f8fd8", k=0.6)
        reg.line(pts + [pts[0]], H * 0.008, col="#6fc0ff", emit="#6fc0ff")
        for i, lab in enumerate(("xG", "xA", "PROG", "DUELS", "PRESS", "PASS")):
            y = H * (0.75 - i * 0.12)
            v = 0.2 + 0.75 * rng.random()
            reg.text(lab, W * 0.52, y, H * 0.045, weight="bold", col="#8a98a8", emit="#8a98a8")
            reg.rect(W * 0.64, y, W * (0.64 + 0.3 * v), y + H * 0.05, col="#3fd06a" if v > 0.6 else "#e0b040", emit="#3fd06a" if v > 0.6 else "#e0b040")
    elif kind == "heat":
        reg.fill(col="#10151c", emit="#10151c")
        _pitch_top(reg, W * 0.03, H * 0.05, W * 0.97, H * 0.85, seed)
        reg.text("TOUCHES  LAST 10", W * 0.03, H * 0.9, H * 0.06, weight="bold", col="#ffffff", emit="#ffffff")
    elif kind == "edit":
        reg.fill(col="#16181c", emit="#16181c")
        _broadcast(reg, W * 0.5, H * 0.45, W * 0.98, H * 0.95, seed)
        reg.rect(W * 0.02, H * 0.45, W * 0.48, H * 0.95, col="#202328", emit="#202328")
        for i in range(8):
            reg.rect(W * 0.04, H * (0.88 - i * 0.055), W * (0.12 + 0.3 * ((seed + i * 7) % 5) / 5), H * (0.9 - i * 0.055), col="#5a6270", emit="#5a6270")
        for t in range(5):
            y = H * (0.36 - t * 0.065)
            x = W * 0.04
            while x < W * 0.96:
                w = W * (0.05 + 0.12 * rng.random())
                c = ("#3a7ad0", "#5a5ac0", "#3a9a6a", "#3a9a6a", "#c08a3a")[t]
                reg.rect(x, y, min(x + w, W * 0.96), y + H * 0.05, col=c, emit=c)
                x += w + W * 0.004
        reg.field(lambda X, Y: ((Y > H * 0.11) & (Y < H * 0.22)) * (np.abs(np.sin(X * 400 + seed)) * (0.3 + 0.7 * noise(X.shape, 60, seed, 2)) > 0.5) * 0.8, col="#7ad08a", emit="#7ad08a")
        reg.rect(W * 0.43, H * 0.05, W * 0.433, H * 0.42, col="#e0302a", emit="#e0302a")
    elif kind in ("camfeed", "graphics"):
        reg.field(lambda X, Y: np.ones_like(X), col="#0a1530", emit="#0a1530")
        reg.field(lambda X, Y: np.clip(1 - np.hypot((X - W * 0.5) / W, (Y - H * 0.7) / H) * 1.6, 0, 1), col="#1d4f91", emit="#1d4f91")
        for k in range(6):
            x0 = W * (0.1 + 0.15 * k)
            reg.line([(x0, 0.0), (x0 + W * 0.25, H)], H * 0.01, col="#3a7ad0", emit="#3a7ad0", k=0.5)
        if kind == "graphics":
            reg.text("UNITED", W * 0.5, H * 0.42, H * 0.22, weight="black", align="c", col="#ffffff", emit="#ffffff")
            reg.text("MATCHDAY LIVE", W * 0.5, H * 0.3, H * 0.07, weight="bold", align="c", col="#9ac0f0", emit="#9ac0f0")
        else:
            reg.poly([(W * 0.2, 0.0), (W * 0.8, 0.0), (W * 0.74, H * 0.3), (W * 0.26, H * 0.3)], col="#e8eaee", emit="#e8eaee")
            for x in (0.38, 0.62):
                reg.rect(W * (x - 0.07), H * 0.3, W * (x + 0.07), H * 0.58, r=W * 0.03, col="#20242c", emit="#20242c")
                reg.circle(W * x, H * 0.66, H * 0.07, col="#c89a78", emit="#c89a78")
                reg.circle(W * x, H * 0.7, H * 0.05, col="#2a2018", emit="#2a2018")
            reg.rect(W * 0.05, H * 0.06, W * 0.6, H * 0.15, col="#c8202a", emit="#c8202a")
            reg.text("UNITED TV  |  THE BUILD-UP", W * 0.07, H * 0.085, H * 0.05, weight="black", col="#ffffff", emit="#ffffff", max_w=W * 0.5)
    elif kind == "sales":
        reg.fill(col="#f4f6f9", emit="#f4f6f9")
        reg.rect(0, H * 0.9, W, H, col="#1d4f91", emit="#1d4f91")
        reg.text("MATCHDAY SALES", W * 0.02, H * 0.925, H * 0.05, weight="bold", col="#ffffff", emit="#ffffff")
        for i, (lab, v) in enumerate((("TICKETS", "31,204"), ("RETAIL", "£184k"), ("HOSPITALITY", "96%"))):
            x = W * (0.03 + i * 0.32)
            reg.rect(x, H * 0.66, x + W * 0.29, H * 0.86, r=H * 0.01, col="#ffffff", emit="#ffffff")
            reg.text(lab, x + W * 0.02, H * 0.8, H * 0.035, weight="bold", col="#7a8794", emit="#7a8794")
            reg.text(v, x + W * 0.02, H * 0.69, H * 0.08, weight="black", col="#1a2230", emit="#1a2230", max_w=W * 0.25)
        for k in range(12):
            v = 0.25 + 0.7 * rng.random()
            x = W * (0.05 + k * 0.075)
            reg.rect(x, H * 0.08, x + W * 0.05, H * (0.08 + 0.48 * v), col="#2f8fd8" if k < 11 else "#3fbf5a", emit="#2f8fd8" if k < 11 else "#3fbf5a")
        reg.rect(W * 0.03, H * 0.075, W * 0.97, H * 0.078, col="#9aa3ad", emit="#9aa3ad")
    else:   # sheet: a shortlist in a spreadsheet
        reg.fill(col="#f6f7f9", emit="#f6f7f9")
        reg.rect(0, H * 0.9, W, H, col="#217346", emit="#217346")
        reg.text("SHORTLIST", W * 0.02, H * 0.925, H * 0.05, weight="bold", col="#ffffff", emit="#ffffff")
        rows = 12
        for i in range(rows):
            y = H * 0.84 - i * H * 0.07
            reg.rect(0, y - H * 0.002, W, y, col="#c8ccd2", emit="#c8ccd2")
            reg.text(NAMES[(seed + i * 5) % len(NAMES)], W * 0.03, y + H * 0.015, H * 0.04, col="#202428", emit="#202428", max_w=W * 0.3)
            reg.text(f"{18 + (seed + i * 3) % 14}", W * 0.4, y + H * 0.015, H * 0.04, col="#202428", emit="#202428")
            c = ("#3fbf5a", "#e8b030", "#d8443a")[int(rng.integers(0, 3))]
            reg.rect(W * 0.55, y + H * 0.008, W * 0.7, y + H * 0.062, col=c, emit=c)
            reg.rect(W * 0.75, y + H * 0.02, W * (0.75 + 0.2 * rng.random()), y + H * 0.05, col="#7a8794", emit="#7a8794")
        for x in (0.37, 0.52, 0.72):
            reg.rect(W * x, 0, W * x + W * 0.002, H * 0.9, col="#c8ccd2", emit="#c8ccd2")


def _arm_monitors(p, M, kinds, cv, seed, w=0.6, h=0.34, y=-0.25, z=0.745, angle=0.32):
    """Monitors on a desk-clamped arm along x, facing +y (sides angled in): their screens."""
    n = len(kinds)
    mat = None
    regs = []
    for k, kind in enumerate(kinds):
        r = cv.alloc(w, h, 600)
        _display(r, kind, seed + k)
        regs.append(r)
    mat = cv.material(cv.name + "_Screens", emission=1.1)
    p.r = 0.0
    p.cyl((0, y - 0.06, z), (0, y - 0.06, z + 0.5), 0.02, M["alu"], 12)
    p.r = 0.004
    p.box((-0.04, y - 0.12, z - 0.06), (0.04, y - 0.02, z + 0.02), M["alu"])
    span = (n - 1) / 2.0
    for k, r in enumerate(regs):
        off = k - span
        if n == 3 and k != 1:
            yaw = -angle if off < 0 else angle
            cx = off * (w + 0.01) * math.cos(angle)
            cy = y + abs(off) * (w / 2.0) * math.sin(angle) + 0.02
        else:
            yaw, cx, cy = 0.0, off * (w + 0.03), y
        _monitor(p, (cx, cy, z + 0.07), yaw, w, h, mat, r, M, stand=False)
        p.r = 0.004
        p.box((min(cx, 0) - 0.02, y - 0.08, z + 0.27), (max(cx, 0) + 0.02, y - 0.04, z + 0.3), M["alu"])


def _keyboard(p, M, x, y, z, w=0.44):
    p.r = 0.003
    p.box((x - w / 2, y - 0.07, z), (x + w / 2, y + 0.07, z + 0.02), M["black"])
    for i in range(4):
        for j in range(int(w / 0.03)):
            p.box((x - w / 2 + 0.01 + j * 0.03, y - 0.06 + i * 0.032, z + 0.02), (x - w / 2 + 0.032 + j * 0.03, y - 0.035 + i * 0.032, z + 0.025), _plastic("Off_Key", "#2b2d30", 0.6))
    p.r = 0.008
    p.box((x + w / 2 + 0.08, y - 0.05, z), (x + w / 2 + 0.14, y + 0.05, z + 0.03), M["black"])


def _papers(p, x, y, z, n, seed, mat):
    rng = np.random.default_rng(seed)
    p.r = 0.0
    for k in range(n):
        p.obox((x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01), z + 0.002 + k * 0.003), (0.21, 0.297, 0.003), mat, yaw=rng.uniform(-0.12, 0.12))


def _mug(p, M, x, y, z, col="#f0f0ee"):
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.04, 0.0), (0.042, 0.095), (0.036, 0.095), (0.034, 0.01), (0.0, 0.01)], _plastic("Off_Mug" + col[1:], col, 0.2), c=(x, y, z), sides=14)
    _tube(p, [(x + 0.04, y, z + 0.075), (x + 0.065, y, z + 0.06), (x + 0.04, y, z + 0.025)], 0.006, _plastic("Off_Mug" + col[1:], col, 0.2), 6)


def _report_canvas(cv, seed):
    """A typed scout's match report on A4."""
    r = cv.alloc(0.21, 0.297, 700)
    r.fill(col="#fbfbf7")
    r.rect(0.015, 0.265, 0.195, 0.285, col="#1d4f91")
    r.text("SCOUT REPORT", 0.02, 0.269, 0.011, weight="black", col="#ffffff")
    for i in range(22):
        y = 0.25 - i * 0.0105
        r.rect(0.02, y, 0.02 + 0.17 * (0.5 + 0.5 * ((seed * 7 + i * 13) % 10) / 10), y + 0.003, col="#8a9098")
    return r


# -- desks -----------------------------------------------------------------------------------------

def _desk_frame(p, M, w, d, top, frame, pedestal=True, sitstand=False):
    """A desk top (w x d, its working edge at +y) on a frame with a modesty panel at -y."""
    p.r = 0.004
    p.box((-w / 2, -d / 2, 0.72), (w / 2, d / 2, 0.745), top)
    if sitstand:
        for x in (-w / 2 + 0.12, w / 2 - 0.12):
            p.box((x - 0.035, -d / 2 + 0.05, 0.0), (x + 0.035, d / 2 - 0.05, 0.03), frame)
            p.box((x - 0.04, -0.03, 0.03), (x + 0.04, 0.03, 0.45), frame)
            p.box((x - 0.032, -0.024, 0.45), (x + 0.032, 0.024, 0.69), frame)
            p.box((x - 0.03, -d / 2 + 0.06, 0.69), (x + 0.03, d / 2 - 0.06, 0.72), frame)
        p.box((-w / 2 + 0.12, -0.03, 0.66), (w / 2 - 0.12, 0.03, 0.7), frame)
        p.box((w / 2 - 0.25, d / 2 - 0.09, 0.705), (w / 2 - 0.13, d / 2 - 0.02, 0.72), M["black"])
    else:
        for x in (-w / 2 + 0.03, w / 2 - 0.03):
            p.box((x - 0.025, -d / 2 + 0.04, 0.0), (x + 0.025, d / 2 - 0.04, 0.03), frame)
            p.box((x - 0.025, -0.02, 0.03), (x + 0.025, 0.02, 0.69), frame)
            p.box((x - 0.025, -d / 2 + 0.04, 0.69), (x + 0.025, d / 2 - 0.04, 0.72), frame)
    p.box((-w / 2 + 0.07, -d / 2 + 0.03, 0.32), (w / 2 - 0.07, -d / 2 + 0.045, 0.7), frame)
    if pedestal:
        x1 = w / 2 - 0.08
        p.box((x1 - 0.42, -d / 2 + 0.08, 0.02), (x1, d / 2 - 0.12, 0.6), top)
        for z in (0.05, 0.22, 0.4):
            p.box((x1 - 0.41, d / 2 - 0.12, z), (x1 - 0.01, d / 2 - 0.11, z + 0.16), top)
            p.box((x1 - 0.26, d / 2 - 0.11, z + 0.12), (x1 - 0.16, d / 2 - 0.095, z + 0.135), M["alu"])
    # cable tray under the back edge
    p.box((-w / 2 + 0.15, -d / 2 + 0.06, 0.62), (w / 2 - 0.15, -d / 2 + 0.2, 0.63), frame)


def desk_basic(col, M):
    """A scout's desk in a portacabin office (1.4 x 0.7 m, beech-effect laminate on a grey frame
    with a drawer pedestal): one 22 inch monitor on its stand showing the shortlist spreadsheet, a
    PC tower on the floor, keyboard and mouse, piles of typed match reports, a stack of DVDs of
    matches, a ring binder, a mug; a task chair. Front +y (the user's side)."""
    cv = Canvas("Off_DeskBasic", 1024, 512, col="#202020", rough=0.4)
    scr = cv.alloc(0.48, 0.27, 700)
    _display(scr, "sheet", 3)
    rep = _report_canvas(cv, 3)
    smat = cv.material("Off_DeskBasicScreen", emission=1.0)
    rmat = cv.material("Off_DeskBasicPaper")
    top = _paint("Off_Beech", "#c9a473", 0.45)
    frame = _paint("Off_FrameGrey", "#8c9196", 0.4)
    p = PM(0.004)
    _desk_frame(p, M, 1.4, 0.7, top, frame)
    _monitor(p, (-0.15, -0.18, 0.745), 0.0, 0.48, 0.27, smat, scr, M)
    _keyboard(p, M, -0.18, 0.12, 0.745)
    # PC tower on the floor, under the free end
    p.r = 0.006
    p.box((-0.6, -0.25, 0.0), (-0.42, 0.2, 0.44), _plastic("Off_Tower", "#2a2c2f", 0.5))
    p.box((-0.58, 0.2, 0.36), (-0.44, 0.205, 0.4), M["grey"])
    # reports, a ring binder, DVD cases, a mug
    for k, (x, y) in enumerate(((0.42, -0.05), (0.2, 0.18))):
        p.r = 0.0
        for j in range(4 + k * 3):
            i0 = len(p.F)
            p.obox((x, y, 0.746 + j * 0.003), (0.21, 0.297, 0.003), rmat, yaw=0.08 * ((j * 5 + k) % 3 - 1))
            project(p, i0, rep, (x - 0.105, y - 0.1485, 0.0), (1, 0, 0), (0, 1, 0))
    p.r = 0.006
    p.box((0.5, -0.3, 0.745), (0.6, -0.02, 1.06), _plastic("Off_Binder", "#2457a6", 0.5))
    for j in range(7):
        p.r = 0.002
        p.box((-0.62, -0.3, 0.745 + j * 0.015), (-0.43, -0.165, 0.758 + j * 0.015), _plastic("Off_DVD", ("#1a1c1f", "#e8e8e8", "#2a3a6a")[j % 3], 0.4))
    _mug(p, M, 0.05, 0.22, 0.745, "#1d4f91")
    _place(p, _task_chair(M), -0.1, 0.62, math.pi)
    return p.obj("desk_basic", col)


def desk_dual(col, M, kinds=("data", "video"), name="desk_dual"):
    """A scout's or head of recruitment's desk (1.6 x 0.8 m, white laminate, black steel frame,
    drawer pedestal): two 24 inch monitors on a clamp arm (a player dashboard and a match clip),
    a laptop on a riser, keyboard and mouse, a headset, a notebook and pen, a desk phone, a mug;
    a mesh task chair. Front +y (the user's side)."""
    cv = Canvas("Off_DeskDual", 1024, 1024, col="#202020", rough=0.4)
    lap = cv.alloc(0.32, 0.2, 700)
    _display(lap, "sheet", 11)
    rep = _report_canvas(cv, 9)
    top = _paint("Off_WhiteTop", "#eeeeea", 0.35)
    frame = _paint("Off_FrameBlack", "#25272a", 0.4)
    p = PM(0.004)
    _desk_frame(p, M, 1.6, 0.8, top, frame)
    _arm_monitors(p, M, kinds, cv, 5, w=0.53, h=0.3, y=-0.22)
    lmat = cv.material("Off_DeskDualLaptop", emission=1.0)
    # laptop on a riser at the side
    p.r = 0.004
    p.obox((0.62, -0.12, 0.8), (0.26, 0.24, 0.01), M["alu"], pitch=math.radians(12))
    p.box((0.52, -0.2, 0.745), (0.72, -0.18, 0.79), M["alu"])
    p.obox((0.62, -0.24, 0.93), (0.33, 0.012, 0.21), M["alu"], pitch=math.radians(-10))
    decal(p, (0.62, -0.232, 0.93), (0, math.cos(math.radians(10)), math.sin(math.radians(10))), (0, -math.sin(math.radians(10)), math.cos(math.radians(10))), 0.31, 0.19, lmat, lap, lift=0.002)
    _keyboard(p, M, -0.1, 0.14, 0.745)
    # headset, notebook, phone, mug
    p.r = 0.0
    _tube(p, [(-0.62, 0.05, 0.76), (-0.62, 0.05, 0.86), (-0.5, 0.05, 0.86), (-0.5, 0.05, 0.76)], 0.008, M["black"], 6, bend=0.05)
    for x in (-0.62, -0.5):
        p.cyl((x, 0.0, 0.765), (x, 0.1, 0.765), 0.035, M["black"], 12)
    i0 = len(p.F)
    p.r = 0.002
    p.obox((0.32, 0.2, 0.75), (0.21, 0.297, 0.01), cv.material("Off_DeskDualPaper"), yaw=0.15)
    project(p, i0, rep, (0.32 - 0.105, 0.2 - 0.1485, 0.0), (1, 0, 0), (0, 1, 0))
    p.cyl((0.3, 0.12, 0.756), (0.42, 0.2, 0.756), 0.004, M["black"], 6)
    p.r = 0.004
    p.box((-0.75, -0.3, 0.745), (-0.55, -0.1, 0.8), _plastic("Off_Phone", "#202224", 0.5))
    _mug(p, M, 0.5, 0.25, 0.745)
    _place(p, _task_chair(M), -0.05, 0.68, math.pi)
    return p.obj(name, col)


def desk_sales(col, M):
    """The commercial team's desk: as desk_dual, the screens on the ticketing and retail sales
    dashboard and the hospitality bookings sheet. Front +y (the user's side)."""
    return desk_dual(col, M, ("sales", "sheet"), "desk_sales")


def desk_triple(col, M):
    """A video analyst's sit-stand workstation (1.8 x 0.8 m, an oak-effect top on a two-column
    electric frame with its handset): three 27 inch monitors on a clamp arm (the coding software
    with a match clip and tagged timeline, a heat map, a player dashboard), a mechanical keyboard,
    a jog-shuttle controller for the video, headphones on a stand, a notepad; a mesh task chair.
    Front +y (the user's side)."""
    cv = Canvas("Off_DeskTriple", 1024, 1024, col="#202020", rough=0.4)
    top = _paint("Off_Oak", "#a77f55", 0.45)
    frame = _paint("Off_FrameBlack", "#25272a", 0.4)
    p = PM(0.004)
    _desk_frame(p, M, 1.8, 0.8, top, frame, pedestal=False, sitstand=True)
    _arm_monitors(p, M, ("heat", "video", "data"), cv, 17, w=0.6, h=0.34, y=-0.24)
    _keyboard(p, M, -0.12, 0.12, 0.745)
    # jog-shuttle, headphones on their stand, notepad
    p.r = 0.006
    p.box((0.32, 0.04, 0.745), (0.48, 0.18, 0.775), M["black"])
    p.r = 0.0
    p.cyl((0.4, 0.11, 0.775), (0.4, 0.11, 0.79), 0.045, M["grey"], 18)
    p.cyl((0.72, -0.1, 0.745), (0.72, -0.1, 0.99), 0.012, M["alu"], 10)
    p.cyl((0.72, -0.1, 0.745), (0.72, -0.1, 0.755), 0.07, M["alu"], 16)
    _tube(p, [(0.62, -0.1, 0.92), (0.64, -0.1, 1.04), (0.8, -0.1, 1.04), (0.82, -0.1, 0.92)], 0.012, M["black"], 6, bend=0.06)
    for x in (0.62, 0.82):
        p.cyl((x, -0.15, 0.9), (x, -0.05, 0.9), 0.05, M["black"], 14)
    p.r = 0.002
    p.box((-0.75, 0.0, 0.745), (-0.55, 0.28, 0.755), _plastic("Off_Notepad", "#f2e98a", 0.6))
    _mug(p, M, -0.6, -0.15, 0.745, "#2a2c30")
    _place(p, _task_chair(M), -0.05, 0.68, math.pi)
    return p.obj("desk_triple", col)


def bench_desk(col, M):
    """Four-person bench desking for an open-plan office (2.8 x 1.6 m: two desks back to back each
    side of a fabric screen, a shared steel frame and cable spine): a monitor and a docked laptop
    at each place, keyboards, papers and mugs; four task chairs. Long axis along x; places face
    +y and -y."""
    cv = Canvas("Off_Bench", 1024, 1024, col="#202020", rough=0.4)
    regs = []
    for k in range(4):
        r = cv.alloc(0.53, 0.3, 600)
        _display(r, ("data", "sheet", "video", "match")[k], 30 + k)
        regs.append(r)
    smat = cv.material("Off_BenchScreens", emission=1.1)
    rep = _report_canvas(cv, 31)
    pmat = cv.material("Off_BenchPaper")
    scv, sreg = _cushion_canvas("Off_ScreenFabric", 1.4, 0.4, seed=41)
    fabric = scv.tint_material("TINT_upholstery_bench", "#5a6470", rough=0.85)
    top = _paint("Off_WhiteTop", "#eeeeea", 0.35)
    frame = _paint("Off_FrameBlack", "#25272a", 0.4)
    p = PM(0.004)
    p.r = 0.004
    for s in (-1, 1):
        p.box((-1.4, 0.0 if s > 0 else -0.8, 0.72), (1.4, 0.8 if s > 0 else 0.0, 0.745), top)
    for x in (-1.37, 0.0, 1.37):
        p.box((x - 0.025, -0.75, 0.0), (x + 0.025, 0.75, 0.03), frame)
        p.box((x - 0.025, -0.02, 0.03), (x + 0.025, 0.02, 0.72), frame)
        p.box((x - 0.025, -0.75, 0.69), (x + 0.025, 0.75, 0.72), frame)
    p.box((-1.35, -0.08, 0.55), (1.35, 0.08, 0.62), frame)
    # the screen down the middle, in its aluminium edge
    p.r = 0.01
    i0 = len(p.F)
    p.box((-1.38, -0.015, 0.745), (1.38, 0.015, 1.15), fabric)
    box_uv(p, i0, {"+y": sreg, "-y": sreg, "+z": sreg, "+x": sreg, "-x": sreg}, (-1.4, -0.02, 0.745), (1.4, 0.02, 1.15))
    p.r = 0.0
    p.cyl((-1.38, 0.0, 1.15), (1.38, 0.0, 1.15), 0.017, M["alu"], 10)
    for s in (-1, 1):
        for j, x in enumerate((-0.7, 0.7)):
            k = (0 if s > 0 else 2) + j
            yaw = 0.0 if s > 0 else math.pi
            ca, sa = math.cos(yaw), math.sin(yaw)

            def at(u, v):
                return (x + u * ca - v * sa, u * sa + v * ca)
            _monitor(p, (*at(0.0, 0.25), 0.745), yaw, 0.53, 0.3, smat, regs[k], M)
            kx, ky = at(0.0, 0.52)
            _keyboard(p, M, kx - 0.05 * s, ky, 0.745, 0.4)
            lx, ly = at(0.42, 0.42)
            p.r = 0.004
            p.obox((lx, ly, 0.755), (0.32, 0.22, 0.02), M["alu"], yaw=yaw)
            if (k + 1) % 2:
                px, py = at(-0.45, 0.45)
                p.r = 0.0
                for n in range(3):
                    i0 = len(p.F)
                    p.obox((px, py, 0.746 + n * 0.003), (0.21, 0.297, 0.003), pmat, yaw=yaw + 0.1 * (n - 1))
                    project(p, i0, rep, (px - 0.105, py - 0.1485, 0.0), (1, 0, 0), (0, 1, 0))
            mx, my = at(-0.2, 0.3)
            _mug(p, M, mx, my, 0.745, ("#f0f0ee", "#1d4f91", "#c8202a", "#2a2c30")[k])
            cx, cy = at(0.0, 1.12)
            _place(p, _task_chair(M), cx, cy, yaw + math.pi + 0.15 * (j - 0.5))
    return p.obj("bench_desk", col)


# -- storage ---------------------------------------------------------------------------------------

def filing_cabinet(col, M):
    """A four-drawer steel filing cabinet (0.47 x 0.62 x 1.32 m, light grey): recessed pulls, card
    label holders (scouting regions and seasons), the lock at the top; box files and a pile of
    match programmes on top. Front +y."""
    cv = Canvas("Off_Filing", 512, 512, col="#c8cbce", rough=0.4)
    lab = cv.alloc(0.08, 0.04, 2000)
    lab.fill(col="#f7f4e8")
    labels = []
    for k, t in enumerate(("UK  NORTH", "UK  SOUTH", "EUROPE", "ACADEMY")):
        r = cv.alloc(0.08, 0.04, 2000)
        r.fill(col="#f7f4e8")
        r.text(t, 0.04, 0.012, 0.012, weight="bold", align="c", col="#1c2a4a", max_w=0.075)
        labels.append(r)
    lmat = cv.material("Off_FilingLabels")
    steel = _paint("Off_FilingSteel", "#b9bdc1", 0.4)
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.235, -0.31, 0.0), (0.235, 0.31, 1.32), steel)
    for k in range(4):
        z = 0.04 + k * 0.32
        p.r = 0.006
        p.box((-0.225, 0.31, z), (0.225, 0.325, z + 0.3), steel)
        p.r = 0.003
        p.box((-0.09, 0.325, z + 0.2), (0.09, 0.345, z + 0.23), M["alu"])
        p.box((-0.05, 0.325, z + 0.24), (0.05, 0.33, z + 0.29), M["alu"])
        decal(p, (0.0, 0.33, z + 0.265), (0, 1, 0), (0, 0, 1), 0.08, 0.04, lmat, labels[3 - k], lift=0.002)
    p.r = 0.0
    p.cyl((0.17, 0.31, 1.25), (0.17, 0.335, 1.25), 0.012, M["chrome"], 10)
    p.r = 0.006
    for j, c in enumerate(("#1d4f91", "#2a2c30")):
        p.box((-0.2 + j * 0.09, -0.15, 1.32), (-0.12 + j * 0.09, 0.2, 1.6), _plastic("Off_BoxFile" + c[1:], c, 0.5))
    p.r = 0.002
    for j in range(6):
        p.obox((0.11, 0.0, 1.322 + j * 0.006), (0.21, 0.29, 0.005), _paint("Off_Programme" + str(j % 3), ("#c8202a", "#f0f0ea", "#1d4f91")[j % 3], 0.5), yaw=0.05 * (j % 3 - 1))
    return p.obj("filing_cabinet", col)


def archive_shelving(col, M):
    """Steel archive shelving (1.0 x 0.4 x 2.0 m, five shelves): box files labelled by season and
    region, lever-arch binders of match reports, rows of DVD and USB-drive cases of games, the odd
    gap. Front +y."""
    cv = Canvas("Off_Archive", 1024, 1024, col="#d0d0d0", rough=0.5)
    regs = []
    rng = np.random.default_rng(7)
    for k in range(4):
        r = cv.alloc(0.96, 0.33, 500)
        r.fill(col="#2a2c30")
        x = 0.0
        while x < 0.95:
            kind = rng.random()
            if kind < 0.45:
                w = 0.08
                c = str(rng.choice(["#1d4f91", "#2a2c30", "#7a2a2a", "#2d5a3a"]))
                r.rect(x, 0.0, x + w - 0.004, 0.32, col=c)
                r.rect(x + 0.01, 0.18, x + w - 0.014, 0.25, col="#f2efe4")
                r.text(f"{2014 + int(rng.integers(0, 12))}/{int(rng.integers(10, 99))}", x + 0.04, 0.205, 0.016, weight="bold", align="c", col="#1a1a1a", max_w=w - 0.03)
                r.circle(x + 0.04, 0.08, 0.012, col="#1a1a1a")
            elif kind < 0.8:
                w = 0.015
                n = int(rng.integers(6, 16))
                for j in range(n):
                    c = str(rng.choice(["#101214", "#e8e8e4", "#3a5a9a", "#1a1a1a"]))
                    r.rect(x + j * w, 0.0, x + (j + 1) * w - 0.002, 0.19, col=c)
                    r.rect(x + j * w + 0.002, 0.14, x + (j + 1) * w - 0.004, 0.16, col="#d0d0d0")
                w = n * w
            else:
                w = float(rng.uniform(0.05, 0.12))
            x += w + 0.004
        regs.append(r)
    bmat = cv.material("Off_ArchiveContents")
    steel = _paint("Off_ShelfSteel", "#9ea3a8", 0.45)
    p = PM(0.004)
    for x in (-0.49, 0.49):
        for y in (-0.19, 0.19):
            p.r = 0.002
            p.box((x - 0.015, y - 0.015, 0.0), (x + 0.015, y + 0.015, 2.0), steel)
    for k in range(5):
        z = 0.1 + k * 0.45
        p.r = 0.002
        p.box((-0.48, -0.19, z), (0.48, 0.19, z + 0.025), steel)
        if k < 4:
            i0 = len(p.F)
            p.r = 0.002
            p.box((-0.48, -0.15, z + 0.025), (0.48, 0.17, z + 0.355), bmat)
            project(p, i0, regs[k], (0.48, 0.17, z + 0.025), (-1, 0, 0), (0, 0, 1))
    for z in (0.3, 1.2, 1.9):
        p.r = 0.0
        _tube(p, [(-0.49, -0.2, z - 0.25), (-0.49, -0.2, z), (0.49, -0.2, z - 0.25)], 0.004, steel, 4)
    return p.obj("archive_shelving", col)


def server_rack(col, M):
    """A 42U server cabinet (0.6 x 1.0 x 2.0 m, black): a perforated front door with its handle,
    servers, storage and switches with lit status lights behind it, patch cables, the vented roof;
    a UPS at the bottom. Front +y."""
    cv = Canvas("Off_Rack", 512, 1024, col="#151617", rough=0.4)
    r = cv.alloc(0.5, 1.86, 500)
    r.fill(col="#18191b")
    rng = np.random.default_rng(13)
    z = 0.02
    while z < 1.8:
        u = int(rng.choice([1, 1, 2, 2, 4]))
        h = u * 0.0445
        c = str(rng.choice(["#2a2c30", "#3a3d42", "#d8dadc", "#22252a"]))
        r.rect(0.01, z, 0.49, z + h - 0.003, r=0.002, col=c)
        if u == 1 and rng.random() < 0.5:
            for j in range(24):
                r.rect(0.05 + j * 0.017, z + 0.012, 0.062 + j * 0.017, z + 0.03, col="#0a0a0a")
                if rng.random() < 0.7:
                    r.rect(0.05 + j * 0.017, z + 0.03, 0.054 + j * 0.017, z + 0.034, col="#3fd06a", emit="#3fd06a")
        else:
            for j in range(int(rng.integers(2, 7))):
                cc = str(rng.choice(["#3fd06a", "#4aa0f0", "#e0b040"]))
                r.circle(0.4 + j * 0.012, z + h / 2, 0.003, col=cc, emit=cc)
            for j in range(int(rng.integers(0, 8))):
                r.rect(0.04 + j * 0.03, z + 0.008, 0.065 + j * 0.03, z + h - 0.01, col="#101214")
        z += h
    door = cv.alloc(0.56, 1.9, 120)
    door.fill(col="#141516", a=0.3)
    door.rect(0.0, 0.0, 0.56, 0.04, col="#141516", a=1.0)
    door.rect(0.0, 1.86, 0.56, 1.9, col="#141516", a=1.0)
    gear = cv.material("Off_RackGear", emission=1.5)
    perf = cv.material("Off_RackDoor", alpha=True)
    body = _paint("Off_RackBody", "#1b1c1e", 0.45)
    p = PM(0.004)
    p.r = 0.006
    p.box((-0.3, -0.5, 0.0), (0.3, 0.5, 0.06), body)
    p.box((-0.3, -0.5, 1.94), (0.3, 0.5, 2.0), body)
    for x in (-0.3, 0.27):
        p.box((x, -0.5, 0.06), (x + 0.03, 0.5, 1.94), body)
    p.box((-0.27, -0.5, 0.06), (0.27, -0.48, 1.94), body)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-0.25, -0.4, 0.07), (0.25, 0.43, 1.93), gear)
    project(p, i0, r, (0.25, 0.43, 0.07), (-1, 0, 0), (0, 0, 1))
    # patch leads looping across the front of the switches
    for k in range(6):
        zz = 1.4 + k * 0.05
        c = ("#3a8ad0", "#e0b040", "#d04a3a", "#3fbf5a")[k % 4]
        _tube(p, [(-0.2, 0.44, zz), (0.0, 0.47, zz - 0.06), (0.2, 0.44, zz)], 0.003, _plastic("Off_Patch" + c[1:], c, 0.5), 5)
    i0 = len(p.F)
    p.box((-0.28, 0.48, 0.05), (0.28, 0.5, 1.95), perf)
    project(p, i0, door, (0.28, 0.5, 0.05), (-1, 0, 0), (0, 0, 1))
    p.r = 0.004
    p.box((0.2, 0.5, 0.9), (0.23, 0.53, 1.15), M["chrome"])
    return p.obj("server_rack", col)


# -- boards and screens on the wall ----------------------------------------------------------------

def whiteboard(col, M):
    """A wall-mounted whiteboard (1.8 x 1.2 m, aluminium frame, a marker tray with markers and an
    eraser): a back four sketched with arrows, a target list by position in marker, a fixture list.
    The back against the wall at y = 0; front +y."""
    cv = Canvas("Off_Whiteboard", 1024, 1024, col="#f8f8f6", rough=0.15)
    b = cv.alloc(1.76, 1.16, 500)
    b.fill(col="#f6f7f5")
    b.field(lambda X, Y: noise(X.shape, 8, 5, 2) * 0.06, col="#9aa0a8")
    blue, red, black = "#1f3fa8", "#c8202a", "#1a1a1a"
    b.text("TARGETS  -  JANUARY", 0.06, 1.04, 0.06, col=blue)
    for i, (pos, nm) in enumerate((("LB", "Lindqvist / Varga"), ("CM", "Okafor ?"), ("RW", "Duarte  -  watch Sat"), ("ST", "Ruiz / Hansen / Bakker"), ("GK", "loan  -  ask agent"))):
        y = 0.9 - i * 0.12
        b.text(pos, 0.08, y, 0.05, col=red)
        b.text(nm, 0.22, y, 0.045, col=black, max_w=0.7)
    # a back four and a midfield three with arrows
    for x, y in ((1.12, 0.3), (1.32, 0.26), (1.52, 0.26), (1.72, 0.3), (1.22, 0.55), (1.42, 0.6), (1.62, 0.55)):
        b.ring(x - 0.1, y + 0.2, 0.03, 0.038, col=blue)
    for (xa, ya, xb, yb) in ((1.02, 0.5, 1.02, 0.85), (1.62, 0.5, 1.75, 0.85), (1.32, 0.8, 1.45, 0.98)):
        b.line([(xa, ya), (xb, yb)], 0.006, col=red)
        b.line([(xb - 0.03, yb - 0.04), (xb, yb), (xb + 0.025, yb - 0.045)], 0.006, col=red)
    b.text("Sat 3pm  away", 0.08, 0.16, 0.04, col=black)
    b.text("Tue 7.45  U21", 0.08, 0.08, 0.04, col=black)
    mat = cv.material("Off_WhiteboardFace", rough=0.12)
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-0.88, 0.005, 0.9), (0.88, 0.02, 2.06), mat)
    project(p, i0, b, (0.88, 0.02, 0.9), (-1, 0, 0), (0, 0, 1))
    p.r = 0.004
    for (lo, hi) in (((-0.9, 0.0, 0.88), (0.9, 0.03, 0.9)), ((-0.9, 0.0, 2.06), (0.9, 0.03, 2.08)), ((-0.9, 0.0, 0.88), (-0.88, 0.03, 2.08)), ((0.88, 0.0, 0.88), (0.9, 0.03, 2.08))):
        p.box(lo, hi, M["alu"])
    p.box((-0.6, 0.0, 0.86), (0.6, 0.08, 0.88), M["alu"])
    p.box((-0.6, 0.07, 0.88), (0.6, 0.08, 0.9), M["alu"])
    for j, c in enumerate((blue, red, black, "#2a8a3a")):
        p.r = 0.0
        p.cyl((-0.35 + j * 0.06, 0.04, 0.892), (-0.22 + j * 0.06, 0.045, 0.892), 0.008, _plastic("Off_Marker" + c[1:], c, 0.4), 8)
    p.r = 0.006
    p.box((0.25, 0.02, 0.88), (0.4, 0.07, 0.92), _plastic("Off_Eraser", "#2a2c30", 0.7))
    return p.obj("whiteboard", col)


def squad_board(col, M):
    """A magnetic squad-planning board (2.4 x 1.4 m, a printed pitch on steel in an aluminium
    frame): name magnets at every position in a 4-3-3, a depth chart of up to three per position,
    coloured dots for contract status, targets in red, the season in the header. The back against
    the wall at y = 0; front +y."""
    cv = Canvas("Off_SquadBoard", 1024, 1024, col="#ffffff", rough=0.3)
    b = cv.alloc(2.36, 1.36, 400)
    b.fill(col="#1c2430")
    b.text("SQUAD PLANNING  2027/28", 0.06, 1.25, 0.07, weight="black", col="#ffffff")
    b.rect(0.06, 0.06, 1.66, 1.2, col="#2f6a2c")
    b.field(lambda X, Y: ((X > 0.06) & (X < 1.66) & (Y > 0.06) & (Y < 1.2)) * ((np.floor((Y - 0.06) / 0.1) % 2) == 0) * 0.35, col="#3d7e38")
    lw = 0.006
    for pts in ([(0.1, 0.1), (1.62, 0.1), (1.62, 1.16), (0.1, 1.16), (0.1, 0.1)], [(0.1, 0.63), (1.62, 0.63)],
                [(0.5, 0.1), (0.5, 0.32), (1.22, 0.32), (1.22, 0.1)], [(0.5, 1.16), (0.5, 0.94), (1.22, 0.94), (1.22, 1.16)]):
        b.line(pts, lw, col="#e6ece6")
    b.line([(0.86 + 0.16 * math.cos(t), 0.63 + 0.16 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 33)], lw, col="#e6ece6")
    pos = ((0.86, 0.16), (0.28, 0.36), (0.66, 0.3), (1.06, 0.3), (1.44, 0.36), (0.5, 0.62), (0.86, 0.56), (1.22, 0.62), (0.32, 0.92), (0.86, 0.98), (1.4, 0.92))
    rng = np.random.default_rng(21)
    for i, (x, y) in enumerate(pos):
        n = int(rng.integers(2, 4))
        for j in range(n):
            yy = y - j * 0.075
            target = j == n - 1 and rng.random() < 0.5
            b.rect(x - 0.13, yy - 0.03, x + 0.13, yy + 0.03, r=0.006, col="#f8d7d4" if target else "#f6f6f2", dh=0.002)
            b.text(NAMES[(i * 3 + j) % len(NAMES)].upper(), x - 0.1, yy - 0.014, 0.026, weight="bold", col="#c8202a" if target else "#1a1a1a", max_w=0.18)
            c = ("#3fbf5a", "#e8b030", "#d8443a")[int(rng.integers(0, 3))]
            b.circle(x + 0.11, yy, 0.012, col=c)
    b.text("KEY", 1.74, 1.12, 0.04, weight="bold", col="#ffffff")
    for i, (c, t) in enumerate((("#3fbf5a", "2+ yrs"), ("#e8b030", "1 yr"), ("#d8443a", "expiring"), ("#f8d7d4", "target"))):
        b.circle(1.77, 1.02 - i * 0.08, 0.018, col=c)
        b.text(t, 1.81, 1.005 - i * 0.08, 0.035, col="#dfe4ea")
    b.text("OUT ON LOAN", 1.74, 0.6, 0.04, weight="bold", col="#ffffff")
    for i in range(4):
        b.rect(1.74, 0.5 - i * 0.075, 2.28, 0.555 - i * 0.075, r=0.006, col="#f6f6f2", dh=0.002)
        b.text(NAMES[(i * 7 + 2) % len(NAMES)].upper(), 1.77, 0.515 - i * 0.075, 0.026, weight="bold", col="#1a1a1a")
    mat = cv.material("Off_SquadBoardFace", rough=0.3)
    p = PM(0.004)
    p.r = 0.0
    i0 = len(p.F)
    p.box((-1.18, 0.005, 0.8), (1.18, 0.02, 2.16), mat)
    project(p, i0, b, (1.18, 0.02, 0.8), (-1, 0, 0), (0, 0, 1))
    p.r = 0.004
    for (lo, hi) in (((-1.2, 0.0, 0.78), (1.2, 0.03, 0.8)), ((-1.2, 0.0, 2.16), (1.2, 0.03, 2.18)), ((-1.2, 0.0, 0.78), (-1.18, 0.03, 2.18)), ((1.18, 0.0, 0.78), (1.2, 0.03, 2.18))):
        p.box(lo, hi, M["alu"])
    return p.obj("squad_board", col)


def video_wall(col, M):
    """A match-viewing video wall: nine 55 inch narrow-bezel panels in a 3 x 3 grid (3.65 x 2.06 m)
    on a wall-mounted frame, its lower edge 0.7 m up, each showing a different live match. The
    back against the wall at y = 0; front +y."""
    cv = Canvas("Off_VideoWall", 2048, 1024, col="#0a0a0a", rough=0.3)
    regs = []
    for k in range(9):
        r = cv.alloc(1.21, 0.68, 260)
        if k == 4:
            _display(r, "video", 50 + k)
        else:
            _display(r, "match", 50 + k)
        regs.append(r)
    mat = cv.material("Off_VideoWallScreens", emission=1.2)
    frame = _paint("Off_WallFrame", "#2a2c2f", 0.5)
    p = PM(0.004)
    p.r = 0.004
    for x in (-1.5, -0.5, 0.5, 1.5):
        p.box((x - 0.03, 0.0, 0.75), (x + 0.03, 0.06, 2.7), frame)
    for z in (0.95, 2.4):
        p.box((-1.8, 0.0, z), (1.8, 0.05, z + 0.06), frame)
    for i in range(3):
        for j in range(3):
            cx = (j - 1) * 1.218
            cz = 0.7 + 0.34 + i * 0.688
            p.r = 0.002
            p.box((cx - 0.607, 0.06, cz - 0.342), (cx + 0.607, 0.1, cz + 0.342), M["black"])
            p.r = 0.0
            decal(p, (cx, 0.1, cz), (0, 1, 0), (0, 0, 1), 1.205, 0.678, mat, regs[i * 3 + j], lift=0.001)
    return p.obj("video_wall", col)


def wall_tv(col, M, kind="data", name="wall_tv", seed=71):
    """A 75 inch screen on a flat wall bracket (1.68 x 0.97 m, its lower edge 1.1 m up) showing a
    player dashboard, with a soundbar under it and the cables in a white trunking to the floor.
    The back against the wall at y = 0; front +y."""
    cv = Canvas("Off_WallTV", 1024, 1024, col="#0a0a0a", rough=0.3)
    r = cv.alloc(1.65, 0.93, 500)
    _display(r, kind, seed)
    mat = cv.material("Off_WallTVScreen_" + kind, emission=1.1)
    p = PM(0.004)
    p.r = 0.004
    p.box((-0.3, 0.0, 1.35), (0.3, 0.03, 1.75), M["steel"])
    p.box((-0.84, 0.03, 1.1), (0.84, 0.07, 2.07), M["black"])
    p.r = 0.0
    decal(p, (0.0, 0.07, 1.585), (0, 1, 0), (0, 0, 1), 1.65, 0.93, mat, r, lift=0.001)
    p.r = 0.01
    p.box((-0.5, 0.0, 0.98), (0.5, 0.09, 1.05), M["black"])
    p.r = 0.003
    p.box((0.62, 0.0, 0.0), (0.68, 0.025, 1.1), _white("Off_Trunking"))
    return p.obj(name, col)


def wall_tv_match(col, M):
    """As wall_tv, showing the live match (bars, lounges, cafés)."""
    return wall_tv(col, M, "match", "wall_tv_match", 77)


def tv_cabinet(col, M):
    """A TV unit (1.6 x 0.45 m, oak-effect, two open shelves and doors): a 55 inch TV on its feet
    showing a match, a Blu-ray recorder and a satellite box with their lights, a stack of DVDs of
    games, the remotes. Front +y."""
    cv = Canvas("Off_TVUnit", 1024, 512, col="#0a0a0a", rough=0.3)
    r = cv.alloc(1.21, 0.68, 330)
    _display(r, "match", 61)
    mat = cv.material("Off_TVUnitScreen", emission=1.1)
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.006)
    p.r = 0.006
    p.box((-0.8, -0.225, 0.08), (0.8, 0.225, 0.5), oak)
    p.box((-0.8, -0.2, 0.0), (-0.75, 0.2, 0.08), M["black"])
    p.box((0.75, -0.2, 0.0), (0.8, 0.2, 0.08), M["black"])
    p.r = 0.0
    p.box((-0.77, -0.15, 0.12), (0.0, 0.226, 0.46), _paint("Off_Shadow", "#1a1714", 0.8))
    p.box((-0.77, -0.15, 0.285), (0.0, 0.227, 0.3), oak)
    for x in (0.03, 0.4):
        p.r = 0.004
        p.box((x, 0.225, 0.11), (x + 0.36, 0.235, 0.47), oak)
        p.box((x + 0.03, 0.235, 0.27), (x + 0.04, 0.25, 0.33), M["alu"])
    # recorder and satellite box with lights, DVDs
    p.r = 0.004
    p.box((-0.7, -0.12, 0.3), (-0.28, 0.2, 0.36), M["black"])
    p.box((-0.7, -0.12, 0.13), (-0.38, 0.2, 0.19), M["black"])
    for (x, z) in ((-0.32, 0.33), (-0.42, 0.16)):
        p.box((x, 0.2, z), (x + 0.01, 0.205, z + 0.01), _plastic("Off_Led", "#40e060", 0.2))
    for j in range(9):
        p.r = 0.002
        p.box((-0.24, -0.1 + 0.0, 0.135 + j * 0.016), (-0.05, 0.035, 0.148 + j * 0.016), _plastic("Off_DVD", ("#1a1c1f", "#e8e8e8", "#2a3a6a")[j % 3], 0.4))
    # the TV
    p.r = 0.004
    for x in (-0.55, 0.55):
        p.box((x - 0.04, -0.12, 0.5), (x + 0.04, 0.05, 0.52), M["black"])
        p.box((x - 0.015, -0.03, 0.52), (x + 0.015, 0.0, 0.6), M["black"])
    p.box((-0.615, -0.04, 0.58), (0.615, 0.0, 1.27), M["black"])
    p.r = 0.0
    decal(p, (0.0, 0.0, 0.925), (0, 1, 0), (0, 0, 1), 1.205, 0.678, mat, r, lift=0.001)
    p.r = 0.004
    for x in (0.15, 0.3):
        p.obox((x, 0.12, 0.505), (0.05, 0.18, 0.02), M["black"], yaw=0.2)
    return p.obj("tv_cabinet", col)


# -- soft seating and tables -----------------------------------------------------------------------

def sofa(col, M):
    """A three-seat sofa (2.1 x 0.9 m, woven fabric, tinted: the club's colour by default): a
    framed base on tapered oak feet, three seat and three back cushions, rounded arms, two scatter
    cushions. Front +y."""
    cv, reg = _cushion_canvas("Off_SofaFabric", 0.8, 0.8, base="#5a6470", seed=51)
    reg.field(lambda X, Y: ((np.mod(X, 0.004) < 0.002) ^ (np.mod(Y, 0.004) < 0.002)) * 0.06, col="#bdbdbd")
    fab = cv.tint_material("TINT_upholstery_sofa", "#5a6470", rough=0.9)
    p = PM(0.01)
    for x in (-0.95, 0.95):
        for y in (-0.35, 0.35):
            p.r = 0.0
            p.cyl((x, y, 0.0), (x, y, 0.12), 0.018, M["timber"], 10, r2=0.025)
    p.r = 0.03
    i0 = len(p.F)
    p.box((-1.05, -0.45, 0.12), (1.05, 0.45, 0.4), fab)
    for x in (-1.05, 0.87):
        p.r = 0.06
        p.box((x, -0.45, 0.4), (x + 0.18, 0.43, 0.64), fab)
    p.r = 0.05
    p.box((-0.87, -0.45, 0.4), (0.87, -0.25, 0.85), fab)
    for k in range(3):
        x = -0.58 + k * 0.58
        p.r = 0.06
        p.box((x - 0.285, -0.25, 0.4), (x + 0.285, 0.44, 0.52), fab)
        p.obox((x, -0.18, 0.7), (0.56, 0.16, 0.38), fab, pitch=math.radians(-12))
    box_uv(p, i0, {"+z": reg, "+y": reg, "-y": reg, "+x": reg, "-x": reg}, (-1.1, -0.5, 0.0), (1.1, 0.5, 0.9))
    p.r = 0.05
    p.obox((-0.72, -0.05, 0.66), (0.38, 0.12, 0.36), _paint("Off_Scatter", "#d8d4cc", 0.9), yaw=0.2, pitch=math.radians(-20))
    p.obox((0.72, -0.05, 0.66), (0.38, 0.12, 0.36), _paint("Off_Scatter", "#d8d4cc", 0.9), yaw=-0.2, pitch=math.radians(-20))
    return p.obj("sofa", col)


def coffee_table(col, M):
    """A low coffee table (1.2 x 0.6 m, oak top on black steel legs, a shelf under): football
    magazines, a match programme, a remote, two coffee cups, a coaster. Front +y."""
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.004)
    p.r = 0.008
    p.box((-0.6, -0.3, 0.38), (0.6, 0.3, 0.42), oak)
    p.r = 0.003
    for x in (-0.56, 0.56):
        for y in (-0.26, 0.26):
            p.box((x - 0.015, y - 0.015, 0.0), (x + 0.015, y + 0.015, 0.38), M["steel"])
    p.box((-0.56, -0.26, 0.12), (0.56, 0.26, 0.135), M["steel"])
    for j, c in enumerate(("#c8202a", "#1d4f91", "#f0f0ea")):
        p.r = 0.0
        p.obox((-0.3 + 0.02 * j, 0.0 + 0.01 * j, 0.423 + j * 0.004), (0.21, 0.28, 0.004), _paint("Off_Mag" + c[1:], c, 0.5), yaw=0.2 * (j - 1))
    for j in range(4):
        p.obox((0.0, 0.0, 0.137 + j * 0.006), (0.3, 0.22, 0.006), _paint("Off_Mag" + ("c8202a", "1d4f91", "f0f0ea")[j % 3], ("#c8202a", "#1d4f91", "#f0f0ea")[j % 3], 0.5), yaw=0.1 * (j - 1.5))
    p.r = 0.004
    p.obox((0.25, -0.1, 0.43), (0.05, 0.17, 0.02), M["black"], yaw=0.4)
    _mug(p, M, 0.35, 0.12, 0.42)
    _mug(p, M, 0.45, -0.05, 0.42, "#1d4f91")
    return p.obj("coffee_table", col)


def meeting_table(col, M):
    """A meeting table for six (2.4 x 1.1 m, white top on a black twin-pedestal frame, a cable port
    with power and a conference speakerphone): six cantilever chairs in fabric (tinted), a laptop,
    notepads and a jug of water with glasses. Long axis along x."""
    cv, reg = _cushion_canvas("Off_ChairFabric", 0.5, 0.5, base="#5a6470", seed=61)
    fab = cv.tint_material("TINT_upholstery_meeting", "#3a4450", rough=0.85)
    top = _paint("Off_WhiteTop", "#eeeeea", 0.35)
    frame = _paint("Off_FrameBlack", "#25272a", 0.4)
    p = PM(0.004)
    p.r = 0.006
    p.box((-1.2, -0.55, 0.72), (1.2, 0.55, 0.75), top)
    for x in (-0.75, 0.75):
        p.r = 0.004
        p.box((x - 0.04, -0.4, 0.0), (x + 0.04, 0.4, 0.03), frame)
        p.box((x - 0.06, -0.06, 0.03), (x + 0.06, 0.06, 0.72), frame)
    p.box((-0.75, -0.04, 0.6), (0.75, 0.04, 0.68), frame)
    p.r = 0.002
    p.box((-0.15, -0.06, 0.75), (0.15, 0.06, 0.755), M["alu"])
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.12, 0.0), (0.1, 0.03), (0.0, 0.035)], M["black"], c=(0.3, 0.0, 0.75), sides=20)
    p.lathe([(0.0, 0.0), (0.06, 0.0), (0.065, 0.2), (0.05, 0.24), (0.0, 0.24)], clear_glass(), c=(-0.3, 0.05, 0.75), sides=14)
    for k in range(3):
        p.lathe([(0.0, 0.0), (0.03, 0.0), (0.035, 0.1), (0.0, 0.1)], clear_glass(), c=(-0.45 + k * 0.08, -0.1, 0.75), sides=10)
    p.r = 0.004
    p.obox((0.8, 0.25, 0.76), (0.32, 0.22, 0.015), M["alu"], yaw=0.1)
    p.obox((0.79, 0.15, 0.86), (0.32, 0.01, 0.2), M["alu"], yaw=0.1, pitch=math.radians(-10))
    for x in (-0.75, 0.0, 0.75):
        for s in (-1, 1):
            if x == 0.0 and s > 0:
                p.r = 0.001
                p.obox((x, 0.32, 0.752), (0.21, 0.297, 0.004), _paint("Off_Notepad2", "#f6f4ea", 0.6), yaw=0.1)
            _place(p, _meeting_chair(M, fab), x, s * 0.78, math.pi if s > 0 else 0.0)
    return p.obj("meeting_table", col)


def boardroom_table(col, M):
    """The recruitment room's boardroom table (4.8 x 1.4 m, a walnut top with a leather inlay strip
    and flush power ports on two slab legs): ten high-back leather chairs, laptops, player dossiers
    in club folders, bottled water and glasses, a conference microphone. Long axis along x."""
    cv = Canvas("Off_Boardroom", 512, 512, col="#ffffff", rough=0.4)
    dos = cv.alloc(0.22, 0.31, 600)
    dos.fill(col="#ffffff")
    dos.text("PLAYER DOSSIER", 0.11, 0.2, 0.016, weight="black", align="c", col="#888888")
    dos.rect(0.04, 0.12, 0.18, 0.13, col="#9a9a9a")
    tint = cv.tint_material("TINT_club_dossier", "#1d4f91", rough=0.5)
    walnut = _paint("Off_Walnut", "#5a3b26", 0.3)
    leather = _plastic("Off_Leather", "#1d1e20", 0.45)
    p = PM(0.004)
    p.r = 0.01
    p.box((-2.4, -0.7, 0.72), (2.4, 0.7, 0.76), walnut)
    p.r = 0.0
    p.box((-2.0, -0.08, 0.76), (2.0, 0.08, 0.762), leather)
    for x in (-1.6, 1.6):
        p.r = 0.01
        p.box((x - 0.05, -0.45, 0.0), (x + 0.05, 0.45, 0.72), walnut)
    for x in (-1.2, 0.0, 1.2):
        p.r = 0.002
        p.box((x - 0.1, -0.05, 0.762), (x + 0.1, 0.05, 0.766), M["alu"])
        p.r = 0.0
        p.cyl((x, 0.0, 0.766), (x, 0.0, 0.79), 0.02, M["black"], 12)
        _tube(p, [(x, 0.0, 0.79), (x, 0.0, 1.0), (x + 0.05, 0.05, 1.06)], 0.004, M["black"], 6)
    places = [(-1.8 + k * 0.9, s) for k in range(5) for s in (-1, 1)]
    for j, (x, s) in enumerate(places):
        _place(p, _exec_chair(M, leather), x, s * 0.98, math.pi if s > 0 else 0.0)
        y = s * 0.45
        if j % 3 == 0:
            p.r = 0.004
            p.obox((x, y, 0.77), (0.32, 0.22, 0.015), M["alu"])
            p.obox((x, y - s * 0.1, 0.87), (0.32, 0.01, 0.2), M["alu"], pitch=math.radians(10 * s))
        else:
            i0 = len(p.F)
            p.r = 0.001
            p.obox((x + 0.05, y, 0.765), (0.22, 0.31, 0.01), tint, yaw=0.1 * s)
            project(p, i0, dos, (x + 0.05 - 0.11, y - 0.155, 0.0), (1, 0, 0), (0, 1, 0))
        p.r = 0.0
        p.cyl((x - 0.25, y - s * 0.12, 0.76), (x - 0.25, y - s * 0.12, 0.96), 0.03, clear_glass(), 10)
        p.cyl((x - 0.25, y - s * 0.12, 0.96), (x - 0.25, y - s * 0.12, 0.99), 0.014, M["blue"], 8)
    return p.obj("boardroom_table", col)


# -- booths, kitchen, walls --------------------------------------------------------------------------

def phone_booth(col, M):
    """An acoustic phone booth (1.0 x 1.0 x 2.2 m): felt-lined walls (tinted), a full-height glass
    door with its pull, a perch seat and a laptop shelf with power, a ceiling light and the
    ventilation grille on the roof, levelling feet. Front +y."""
    cv, reg = _cushion_canvas("Off_PodFelt", 1.0, 2.2, base="#4a525c", ppm=200, seed=71)
    reg.field(lambda X, Y: noise(X.shape, 80, 72, 2) * 0.12, col="#a0a0a0")
    felt = cv.tint_material("TINT_upholstery_pod", "#4a525c", rough=0.95)
    shell = _paint("Off_PodShell", "#e9e9e6", 0.4)
    oak = _paint("Off_Oak", "#a77f55", 0.45)
    p = PM(0.006)
    p.r = 0.01
    p.box((-0.5, -0.5, 0.04), (0.5, 0.5, 0.1), shell)
    p.box((-0.5, -0.5, 2.12), (0.5, 0.5, 2.2), shell)
    for x in (-0.5, 0.45):
        p.box((x, -0.5, 0.1), (x + 0.05, 0.5, 2.12), shell)
    p.box((-0.45, -0.5, 0.1), (0.45, -0.45, 2.12), shell)
    i0 = len(p.F)
    p.r = 0.0
    p.box((-0.45, -0.45, 0.1), (-0.43, 0.45, 2.12), felt)
    p.box((0.43, -0.45, 0.1), (0.45, 0.45, 2.12), felt)
    p.box((-0.43, -0.45, 0.1), (0.43, -0.43, 2.12), felt)
    box_uv(p, i0, {"+x": reg, "-x": reg, "+y": reg}, (-0.5, -0.5, 0.0), (0.5, 0.5, 2.2))
    for x in (-0.44, 0.44):
        for y in (-0.44, 0.44):
            p.cyl((x, y, 0.0), (x, y, 0.04), 0.02, M["black"], 10)
    # perch seat, shelf, light, roof grille
    p.r = 0.02
    p.box((-0.25, -0.43, 0.68), (0.25, -0.15, 0.74), felt)
    p.r = 0.006
    p.box((-0.43, -0.43, 1.02), (0.43, -0.1, 1.05), oak)
    p.box((0.3, -0.43, 1.05), (0.4, -0.4, 1.09), M["black"])
    p.box((-0.3, -0.3, 2.08), (0.3, 0.3, 2.12), _plastic("Off_PodLight", "#fff6e6", 0.2))
    p.r = 0.002
    for k in range(6):
        p.box((-0.25, -0.3 + k * 0.1, 2.2), (0.25, -0.26 + k * 0.1, 2.215), M["grey"])
    # the glass door and its pull
    p.r = 0.0
    p.box((-0.44, 0.47, 0.11), (0.44, 0.485, 2.11), clear_glass())
    p.box((-0.45, 0.46, 0.1), (-0.43, 0.5, 2.12), M["black"])
    p.box((0.43, 0.46, 0.1), (0.45, 0.5, 2.12), M["black"])
    p.cyl((0.36, 0.49, 0.8), (0.36, 0.49, 1.4), 0.012, M["alu"], 10)
    for z in (0.8, 1.4):
        p.cyl((0.36, 0.485, z), (0.36, 0.52, z), 0.008, M["alu"], 8)
    p.cyl((0.36, 0.52, 0.8), (0.36, 0.52, 1.4), 0.012, M["alu"], 10)
    return p.obj("phone_booth", col)


def kitchenette(col, M):
    """A staff kitchenette run (2.4 x 0.6 m base units, 0.9 m worktop): a stainless sink with a
    swan-neck tap and a drainer, an under-counter fridge and a dishwasher, a bean-to-cup coffee
    machine, a kettle, a microwave, mugs on a tree, a fruit bowl. The back against the wall at
    y = 0; front +y."""
    unit = _paint("Off_KitchenUnit", "#e8e6e0", 0.35)
    work = _paint("Off_Worktop", "#3a3836", 0.35)
    p = PM(0.004)
    p.r = 0.004
    p.box((-1.2, 0.05, 0.0), (1.2, 0.55, 0.1), _paint("Off_Plinth", "#2a2826", 0.6))
    p.box((-1.2, 0.0, 0.1), (1.2, 0.56, 0.86), unit)
    for k in range(4):
        x = -1.2 + k * 0.6
        p.box((x + 0.005, 0.56, 0.11), (x + 0.595, 0.58, 0.85), unit if k not in (1, 2) else _white("Off_Appliance"))
        p.box((x + 0.25, 0.58, 0.78), (x + 0.35, 0.6, 0.8), M["alu"])
    p.r = 0.006
    p.box((-1.22, 0.0, 0.86), (1.22, 0.62, 0.9), work)
    p.r = 0.0
    p.box((-1.2, 0.0, 0.9), (1.2, 0.012, 1.4), _paint("Off_Splash", "#d8d8d4", 0.2))
    # sink and drainer, tap
    p.box((-0.95, 0.12, 0.9), (-0.5, 0.5, 0.902), M["chrome"])
    p.box((-0.93, 0.14, 0.75), (-0.62, 0.48, 0.9), _paint("Off_SinkBowl", "#9aa0a4", 0.25))
    for k in range(6):
        p.box((-0.6 + 0.0, 0.16 + k * 0.05, 0.902), (-0.52, 0.18 + k * 0.05, 0.906), M["chrome"])
    p.cyl((-0.78, 0.06, 0.9), (-0.78, 0.06, 1.2), 0.014, M["chrome"], 10)
    _tube(p, [(-0.78, 0.06, 1.2), (-0.78, 0.12, 1.26), (-0.78, 0.2, 1.2), (-0.78, 0.2, 1.14)], 0.012, M["chrome"], 8, bend=0.04)
    # coffee machine, kettle, microwave, mugs, fruit
    p.r = 0.01
    p.box((-0.3, 0.08, 0.9), (0.05, 0.5, 1.35), M["black"])
    p.box((-0.25, 0.48, 1.1), (0.0, 0.5, 1.25), _plastic("Off_CoffeePanel", "#3a3d42", 0.3))
    p.box((-0.2, 0.38, 0.9), (-0.05, 0.5, 0.94), M["chrome"])
    p.r = 0.0
    p.lathe([(0.0, 0.0), (0.08, 0.0), (0.075, 0.2), (0.05, 0.24), (0.0, 0.24)], _plastic("Off_Kettle", "#e8e8e4", 0.3), c=(0.25, 0.3, 0.9), sides=16)
    p.r = 0.01
    p.box((0.5, 0.06, 0.9), (1.0, 0.44, 1.2), _white("Off_Microwave"))
    p.box((0.52, 0.44, 0.93), (0.85, 0.445, 1.17), M["glass"])
    p.r = 0.0
    p.cyl((1.1, 0.3, 0.9), (1.1, 0.3, 1.2), 0.008, M["chrome"], 8)
    for k in range(4):
        a = k * math.pi / 2
        _mug(p, M, 1.1 + 0.08 * math.cos(a), 0.3 + 0.08 * math.sin(a), 0.92 + 0.07 * (k % 2), ("#f0f0ee", "#1d4f91", "#c8202a", "#2a2c30")[k])
    for k, c in enumerate(("#d84a2a", "#e8c030", "#5aa03a")):
        p.sphere((0.33 + 0.05 * k, 0.15, 0.94), 0.035, _plastic("Off_Fruit" + c[1:], c, 0.4))
    return p.obj("kitchenette", col)


def kitchen_wall_units(col, M):
    """Kitchen wall cupboards (2.4 x 0.33 m, 0.7 m tall, their underside 1.45 m up): four doors
    with bar handles, an under-cabinet LED strip. The back against the wall at y = 0; front +y."""
    unit = _paint("Off_KitchenUnit", "#e8e6e0", 0.35)
    p = PM(0.004)
    p.r = 0.004
    p.box((-1.2, 0.0, 1.45), (1.2, 0.33, 2.15), unit)
    for k in range(4):
        x = -1.2 + k * 0.6
        p.box((x + 0.005, 0.33, 1.455), (x + 0.595, 0.35, 2.145), unit)
        p.box((x + 0.25, 0.35, 1.5), (x + 0.35, 0.37, 1.52), M["alu"])
    p.r = 0.0
    p.box((-1.15, 0.24, 1.445), (1.15, 0.27, 1.45), _plastic("Off_LedStrip", "#fff6e6", 0.2))
    return p.obj("kitchen_wall_units", col)


def framed_shirt(col, M):
    """A framed signed shirt (0.8 x 1.0 m box frame, black): the club shirt (tinted) spread on a
    navy mount with a number and signatures in marker, a brass plaque. The back against the wall
    at y = 0; front +y."""
    cv = Canvas("Off_FramedShirt", 512, 512, col="#14192a", rough=0.8)
    m = cv.alloc(0.76, 0.96, 400)
    m.fill(col="#141a2c")
    m.rect(0.26, 0.04, 0.5, 0.09, col="#b08a3a", metal=0.8, rough=0.3)
    m.text("SIGNED  2026/27", 0.38, 0.055, 0.018, weight="bold", align="c", col="#2a2010")
    mount = cv.material("Off_ShirtMount")
    scv = Canvas("Off_ShirtCloth", 256, 256, col="#ffffff", rough=0.8)
    sr = scv.alloc(0.6, 0.7, 300)
    sr.fill(col="#ffffff")
    sr.field(lambda X, Y: noise(X.shape, 30, 81, 2) * 0.12, col="#a0a0a0")
    shirt = scv.tint_material("TINT_club_shirt", "#c8202a", rough=0.8)
    p = PM(0.004)
    p.r = 0.006
    for (lo, hi) in (((-0.4, 0.0, 0.9), (0.4, 0.08, 0.94)), ((-0.4, 0.0, 1.86), (0.4, 0.08, 1.9)), ((-0.4, 0.0, 0.9), (-0.36, 0.08, 1.9)), ((0.36, 0.0, 0.9), (0.4, 0.08, 1.9))):
        p.box(lo, hi, M["black"])
    p.r = 0.0
    i0 = len(p.F)
    p.box((-0.38, 0.0, 0.92), (0.38, 0.01, 1.88), mount)
    project(p, i0, m, (0.38, 0.01, 0.92), (-1, 0, 0), (0, 0, 1))
    # the shirt: body, sleeves and collar as a shallow solid on the mount
    body = [(-0.2, 1.05), (0.2, 1.05), (0.2, 1.52), (0.32, 1.42), (0.38, 1.52), (0.24, 1.68), (0.07, 1.7), (0.0, 1.66), (-0.07, 1.7), (-0.24, 1.68), (-0.38, 1.52), (-0.32, 1.42), (-0.2, 1.52)]
    p.extrude([(x, z) for x, z in body], 0.012, 0.03, shirt)
    nm = Canvas("Off_ShirtNumber", 256, 256, col="#ffffff", rough=0.6)
    nr = nm.alloc(0.24, 0.24, 500)
    nr.fill(col="#ffffff", a=0.0)
    nr.text("9", 0.12, 0.02, 0.2, weight="black", align="c", col="#ffffff", a=1.0)
    nr.line([(0.02, 0.2), (0.08, 0.22), (0.12, 0.18), (0.2, 0.22)], 0.004, col="#101010", a=1.0)
    nmat = nm.material("Off_ShirtNumberMat", alpha=True)
    decal(p, (0.0, 0.031, 1.32), (0, 1, 0), (0, 0, 1), 0.24, 0.24, nmat, nr, lift=0.001)
    p.box((-0.37, 0.07, 0.92), (0.37, 0.075, 1.88), clear_glass())
    return p.obj("framed_shirt", col)


BUILDERS = {
    "prop.desk_basic": desk_basic, "prop.desk_dual": desk_dual, "prop.desk_triple": desk_triple,
    "prop.bench_desk": bench_desk, "prop.filing_cabinet": filing_cabinet, "prop.archive_shelving": archive_shelving,
    "prop.server_rack": server_rack, "prop.whiteboard": whiteboard, "prop.squad_board": squad_board,
    "prop.video_wall": video_wall, "prop.wall_tv": wall_tv, "prop.wall_tv_match": wall_tv_match, "prop.desk_sales": desk_sales, "prop.tv_cabinet": tv_cabinet, "prop.sofa": sofa, "prop.coffee_table": coffee_table,
    "prop.meeting_table": meeting_table, "prop.boardroom_table": boardroom_table, "prop.phone_booth": phone_booth,
    "prop.kitchenette": kitchenette, "prop.kitchen_wall_units": kitchen_wall_units, "prop.framed_shirt": framed_shirt,
}
