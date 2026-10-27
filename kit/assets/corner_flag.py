"""Corner flagpost: Law 1 post (>= 1.5 m, non-pointed top), spring or rigid base, cloth-draped flag.

Local frame: origin at the base on the corner of the lines, +Y towards the centre mark.
"""
import math

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import srgb
from ..core.scene import collection, empty, link, mesh_object, set_params
from .goal import duplicate

DEFAULTS = {"preset": "pro_spring", "wind": 0.15, "seed": 1}
POLE_R = 0.016


def resolve(params):
    params = dict(params or {})
    p = catalog.merge(DEFAULTS, catalog.preset("corner_flags", params.get("preset", DEFAULTS["preset"])))
    return catalog.merge(p, params)


def _pole_mat(slot, color, preset):
    """Gloss moulded PVC / painted post. One material per preset and band (TINT_pole_a_<preset>,
    _b_<preset>), so each preset carries its own colours; recolourable in the game by that name."""
    name = f"TINT_pole_{slot}_{preset}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    from .seat import flat
    c = np.array(srgb(color)[:3])
    c = np.minimum(c, 0.72)                     # a white post is off-white, never clipping
    if c[0] > 0.4 and c[1] < 0.1:               # club red tape: a touch less saturated
        c = c * 0.85 + c.mean() * 0.15
    if preset == "grassroots":                  # sun-faded cheap PVC
        c = c * 0.85 + c.mean() * 0.15
    hexc = "#" + "".join(f"{int(round(255 * (v / 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055))):02x}" for v in c)
    return flat(name, hexc, rough=0.28)


def _pole(p, col, root):
    """Post (evenly banded, tape seams at the joints), domed top cap, and its base: a zinc spring
    standing in a ground socket collar 30 mm proud of the turf, or a flexible rubber joint over a
    spike collar. A worn soil ring round it; mud, grass stain and boot scuffs in the vertex colours."""
    from . import kit_detail as KD
    from .props import PM
    from .seat import flat
    h = p["height"]
    colors = p["pole_colors"]
    mats = [_pole_mat("ab"[i], c, p["preset"]) for i, c in enumerate(colors[:2])]
    q = PM(0.0)
    spring = p["base"] == "spring"
    z0 = 0.17 if spring else 0.115
    top = h - 0.012
    if p["band_length"] > 0 and len(colors) > 1:
        nb_ = max(1, int(round((top - z0) / min(p["band_length"], 0.25))))
        zs = list(np.linspace(z0, top, nb_ + 1))
    else:
        zs = [z0, top]
    for k, (a, b) in enumerate(zip(zs[:-1], zs[1:])):
        q.cyl((0, 0, a), (0, 0, b), POLE_R, mats[k % len(mats)], sides=18, cap=False)
    for k, z in enumerate(zs[1:-1]):                       # tape overlap seams
        q.cyl((0, 0, z - 0.002), (0, 0, z + 0.003), POLE_R + 0.0006, mats[(k + 1) % len(mats)], sides=18)
    q.r = 0.002
    q.cyl((0, 0, h - 0.035), (0, 0, h - 0.005), POLE_R + 0.0035, mats[0], sides=18)          # top cap
    q.r = 0.0
    q.sphere((0, 0, h - 0.005), POLE_R + 0.0035, mats[0], nu=18, nv=8, scale=(1, 1, 0.55), z_min=0.0)
    rubber = flat("FlagBase", "#141414", rough=0.8)
    socket = flat("FlagSocket", "#1d1e1f", rough=0.6)
    if spring:
        zinc = flat("SpringZinc", "#c4c7c9", rough=0.35, metallic=1.0)
        t = np.linspace(0, 1, 160)
        helix = np.stack([0.0165 * np.cos(2 * np.pi * 9 * t), 0.0165 * np.sin(2 * np.pi * 9 * t), 0.045 + 0.115 * t], 1)
        q.r = 0.0
        q.sweep(helix, [(0.0033 * math.cos(a), 0.0033 * math.sin(a)) for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)],
                zinc, twist_ref=(0, 0, 1))
        q.r = 0.004
        q.cyl((0, 0, 0.15), (0, 0, 0.185), 0.022, rubber, sides=20, r2=POLE_R + 0.002)    # top ferrule
        q.cyl((0, 0, 0.03), (0, 0, 0.165), 0.008, zinc, sides=8)                          # core pin
    else:
        q.r = 0.004
        q.cyl((0, 0, 0.026), (0, 0, 0.12), POLE_R + 0.006, rubber, sides=18)              # flexible joint
        for z in (0.05, 0.075, 0.1):
            q.cyl((0, 0, z - 0.004), (0, 0, z + 0.004), POLE_R + 0.0075, rubber, sides=18)
    if spring:
        q.r = 0.003
        q.cyl((0, 0, -0.01), (0, 0, 0.008), 0.035, socket, sides=24)                      # flush socket cap, 70 mm
        q.cyl((0, 0, 0.008), (0, 0, 0.03), 0.024, socket, sides=24, r2=0.021)            # spring adaptor collar
    else:
        q.r = 0.008                                                                      # weighted rubber base
        q.cyl((0, 0, 0.0), (0, 0, 0.04), 0.15, socket, sides=32, r2=0.13)
        q.r = 0.0
        for k in range(12):                                                              # moulded grip ribs
            a = 2 * np.pi * k / 12
            c, sn = math.cos(a), math.sin(a)
            q.obox((0.1 * c, 0.1 * sn, 0.039), (0.06, 0.008, 0.006), socket, yaw=a)
    o = q.obj("Pole", col, smooth_deg=50)
    o.parent = root
    KD.finish(o, distance=0.2, strength=0.7)
    rng = np.random.default_rng(p["seed"] + 5)
    ph = rng.uniform(0, 6.28, 4)

    def grime(P, N):
        z = P[:, 2]
        g = np.clip(1 - (z - 0.03) / 0.26, 0, 1) ** 1.3
        ang = np.arctan2(P[:, 1], P[:, 0])
        streak = 0.5 + 0.5 * np.sin(ang * 3 + ph[0]) * np.sin(z * 40 + ph[1])
        g = g * (0.65 + 0.35 * streak) * (np.hypot(P[:, 0], P[:, 1]) < 0.03)
        mud = np.array([0.55, 0.52, 0.38])
        out = 1 - g[:, None] * (1 - mud)
        scuff = (np.sin(ang * 2 + ph[2]) * np.sin(z * 17 + ph[3]) > 0.9) & (z > 0.3) & (z < 0.9)
        out[scuff] *= np.array([0.82, 0.8, 0.74])
        return out
    KD.tint_vertices(o, grime)
    return o


def _flag_image(p, n=512):
    """The flag cloth (linear RGB): the pattern, a 12 mm double-stitched hem on the free edges, the
    sleeve seam, a fine woven-polyester thread texture, sun fade towards the fly, a little grime at
    the sleeve (and a faded, grubby grassroots flag)."""
    from . import kit_detail as KD
    cols = [np.minimum(np.array(srgb(c)[:3]), 0.85) for c in p["flag_colors"]]
    grubby = p["base"] != "spring"
    if p["flag_pattern"] == "checker":                 # fluorescent red-orange and off-white
        cols = [np.array([0.95, 0.16, 0.04]) if c[0] > c[1] * 2 else np.minimum(c, 0.8) for c in cols]
    if p["flag_pattern"] == "solid" and cols[0][0] > 0.6 and cols[0][1] > 0.5:   # fluorescent yellow
        cols[0] = np.array([0.85, 0.8, 0.06])
    c0, c1 = cols[0], cols[1 % len(cols)]
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n          # x along the flag (0 at the pole), y up
    pat = p["flag_pattern"]
    if pat == "diagonal":
        m = (x + (1 - y)) > 1.0
    elif pat == "checker":
        m = ((np.floor(x * 4) + np.floor(y * 4)) % 2) == 1
    elif pat == "quartered":
        m = (x < 0.5) ^ (y < 0.5)
    elif pat == "stripes":
        m = (np.floor(y * 4) % 2) == 1
    else:
        m = np.zeros_like(x, bool)
    img = np.where(m[..., None], c1, c0)
    if pat == "checker":                         # sewn squares: a faint seam where they meet
        sx = np.abs(x * 4 - np.round(x * 4)) < 0.8 / n * 4
        sy = np.abs(y * 4 - np.round(y * 4)) < 0.8 / n * 4
        img = img * np.where((sx | sy)[..., None], 0.85, 1.0)
    size = p["flag_size"][0]
    hem = 0.012 / size
    edge = np.minimum.reduce([1 - x, y, 1 - y])
    img = img * np.where((edge < hem)[..., None], 0.9, 1.0)
    dash = (np.floor((x + y) * n / 4) % 2) == 0
    for d in (hem * 0.35, hem * 0.8):
        img = img * np.where(((np.abs(edge - d) < 0.7 / n) & dash)[..., None], 0.72, 1.0)
    img = img * np.where((np.abs(x - 0.07) < 0.9 / n)[..., None], 0.75, 1.0)
    thread = 1 + 0.035 * np.sign(np.sin(np.pi * x * n / 2)) * np.sign(np.sin(np.pi * y * n / 2))
    img = img * thread[..., None]
    img = img * (1 - 0.07 * x[..., None])                            # sun fade towards the fly
    img = img * (1 - 0.12 * np.clip(1 - x / 0.12, 0, 1)[..., None])  # handling grime at the sleeve
    img = img * (0.95 + 0.1 * KD.pnoise(n, n, 14, 14, seed=p["seed"])[..., None])
    if grubby:
        dirt = np.clip(1 - np.hypot(x - 0.9, y - 0.05) / 0.45, 0, 1) * 0.3
        img = img * (1 - dirt[..., None]) + np.array([0.18, 0.15, 0.1]) * dirt[..., None]
    return np.clip(img, 0, 1)


def _flag_material(p):
    """One woven fabric for both faces, with a little emission standing in for the light that
    comes through thin polyester (the shaded face never goes dull and dark)."""
    from . import kit_detail as KD
    name = f"TINT_flag_{p['preset']}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    alb = KD.image(f"FlagPattern_{p['preset']}", _flag_image(p))
    return KD.image_material(name, albedo=alb, rough=0.8, extension="EXTEND", emission=(alb, 0.12))


def _flag(p, col, root, pole_obj_names=None):
    """A polyester flag hanging from a sleeve round the top of the post in a light breeze: gravity
    sag towards the fly, three soft folds rippling the whole length (growing away from the pole),
    a lifted trailing edge. Two faces (front and back) share the one material."""
    from .props import PM
    fw = max(p["flag_size"][0], 0.45)
    fh = max(p["flag_size"][1], 0.45)
    h = p["height"]
    ztop = h - 0.04
    rs = POLE_R + 0.004
    rng = np.random.default_rng(p["seed"])
    phi = rng.uniform(0, 6.28)
    nx, nz = 18, 12
    u = np.linspace(0, 1, nx + 1)
    v = np.linspace(0, 1, nz + 1)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    # pose: limp (grassroots), light breeze (pro), wind (striped). The cloth's fly swings down
    # under gravity along a curve (length kept), lower rows dropping more; folds run diagonally
    # and grow towards the fly, the free edge curls, the free bottom corner hangs low.
    pose = {"grassroots": "limp", "pro_spring": "breeze"}.get(p["preset"], "wind")
    droop, droop_v, A, lam, curl = {"limp": (1.15, 0.35, 0.03, 0.12, 0.02),
                                    "breeze": (0.32, 0.22, 0.034, 0.19, 0.035),
                                    "wind": (0.12, 0.12, 0.045, 0.22, 0.05)}[pose]
    du = fw / nx
    X = np.zeros_like(U)
    Z = np.zeros_like(U)
    for i in range(1, nx + 1):
        th = (droop * U[i - 1] ** 1.2 + droop_v * U[i - 1] * Vv[i - 1]) * (0.9 + 0.1 * np.cos(Vv[i - 1] * 3 + phi))
        X[i] = X[i - 1] + du * np.cos(th)
        Z[i] = Z[i - 1] - du * np.sin(th)
    X = rs + X * (1 - 0.03 * Vv)
    Z = ztop - Vv * fh + Z
    ph = 2 * np.pi * U * fw / lam + phi + 1.6 * Vv                      # diagonal folds
    Y = ((0.012 + A * U ** 1.1) * np.sin(ph) + curl * np.clip((U - 0.75) / 0.25, 0, 1) ** 2 * (0.4 + Vv)
         + 0.02 * U * Vv ** 2)
    yaw = math.radians(rng.uniform(-35, -15))
    ca, sa = math.cos(yaw), math.sin(yaw)
    P = np.stack([X * ca - Y * sa, X * sa + Y * ca, Z], -1)
    dPi = np.gradient(P, axis=0)
    dPj = np.gradient(P, axis=1)
    Nf = np.cross(dPj, dPi)
    Nf /= np.linalg.norm(Nf, axis=-1, keepdims=True)
    mat = _flag_material(p)
    q = PM(0.0)
    sl = 0.07                                  # the sleeve takes the first 7 % of the cloth
    for side, off in ((1, 0.0012), (-1, -0.0012)):
        Q = P + Nf * off
        for i in range(nx):
            for j in range(nz):
                ids = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
                quad = [Q[a, b] for a, b in ids]
                uv = [(sl + (1 - sl) * U[a, b], 1 - Vv[a, b]) for a, b in ids]
                if side < 0:
                    quad, uv = quad[::-1], uv[::-1]
                q.face(quad, mat, uv)
    # the sleeve: the cloth wrapped round the post, stitched to the flag
    k = 14
    for j in range(nz):
        for t in range(k):
            a0, a1 = 2 * np.pi * t / k + yaw, 2 * np.pi * (t + 1) / k + yaw
            za, zb = ztop - v[j] * fh, ztop - v[j + 1] * fh
            pts = [(rs * math.cos(a0), rs * math.sin(a0), za), (rs * math.cos(a0), rs * math.sin(a0), zb),
                   (rs * math.cos(a1), rs * math.sin(a1), zb), (rs * math.cos(a1), rs * math.sin(a1), za)]
            uv = [(sl * t / k, 1 - v[j]), (sl * t / k, 1 - v[j + 1]), (sl * (t + 1) / k, 1 - v[j + 1]), (sl * (t + 1) / k, 1 - v[j])]
            q.face(pts, mat, uv)
    from .seat import flat
    tie = flat("FlagTie", "#1a1a1a", rough=0.6)
    for z in (ztop + 0.004, ztop - fh + 0.006):
        q.cyl((0, 0, z - 0.006), (0, 0, z + 0.006), rs + 0.0025, tie, sides=14)
    o = q.obj("Flag", col, smooth_deg=80, bevel=False)
    o.parent = root
    return o


def build(params=None, col=None, template=None):
    col = col or collection("CornerFlag")
    if template is not None:
        return duplicate(template, col)
    p = resolve(params)
    root = empty("CornerFlag", col, display="ARROWS", size=0.5)
    set_params(root, f"corner_flag.{p['preset']}", "corner_flag", "corner_flag", p)
    pole = _pole(p, col, root)
    flag = _flag(p, col, root, None)
    if p["base"] != "spring":
        # a rigid grassroots post never stands quite straight
        lean = math.radians(1.5)
        for o in (pole, flag):
            o.data.transform(__import__("mathutils").Matrix.Rotation(lean, 4, "X"))
    root["height"] = p["height"]
    return root
