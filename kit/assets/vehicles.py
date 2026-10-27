"""Road vehicles: hatchback (VW Golf Mk8 class), saloon (BMW 3 Series G20), crossover SUV (Nissan
Qashqai J12), minibus (Ford Transit L4 high-roof 17-seat) and an abandoned hatch on blocks. Each type
has body variants (VARIANTS: a supermini, a hot hatch, an estate, a touring, a fastback, a
coupe-SUV, a seven-seater, a medium-roof and a shorter minibus...), each with its own plate.

Bodies are designed the way stylists draw them: side-view profiles (roof, belt, shoulder crease,
sill), a plan (width, roof width) and a section through the car with tumblehome, evaluated as a
smooth surface (monotone cubic along the length, Catmull-Rom around the section; the shoulder and
belt are hard creases). The ends are rounded caps. Windows are cut out of that skin and glazed
with inset tinted glass in black frames, so the cabin (seats, dash, wheel) shows through. Lamps,
grilles, intakes, badges and bumper splits are painted (kit/assets/vehicle_fascia.py) on patches
wrapped around the nose and tail by casting rays onto the skin. Mirrors and handles are
superellipsoids; wheels are lathed tyres (sidewall lettering, tread) with modelled spokes,
brake discs and callipers. Occlusion and road grime are baked into the vertex colours.

Local frame: +Y is the front, +X the right side, origin on the ground at the middle of the car;
metres. Each vehicle is one mesh; body paint is TINT_car_paint (TINT_van_paint for the minibus)
and is recoloured per instance.
"""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import mesh_object
from . import vehicle_paint as VP


# -- materials ------------------------------------------------------------------------------------

def _mat(name, color, rough=0.4, metal=0.0, alpha=None, spec=None):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    kw = dict(Base_Color=srgb(color), Roughness=rough, Metallic=metal)
    if alpha is not None:
        kw["Alpha"] = alpha
    if spec is not None:
        kw["Specular_IOR_Level"] = spec
    nb.output(nb.principled(**kw))
    return m


def car_paint(name="TINT_car_paint", color="#1d3157"):
    """Clear-coated paint: a glossy dielectric (the sky reflection does the work)."""
    return _mat(name, color, 0.14, 0.05)


def mats(paint=None):
    from . import vehicle_fascia as VF
    return {
        "paint": paint or car_paint(),
        "glass": _mat("Prop_GlassClear_CarTint", "#0b1013", 0.03, alpha=0.6, spec=0.9),
        "privacy": _mat("Prop_GlassClear_CarPrivacy", "#06080a", 0.03, alpha=0.84, spec=0.9),
        "glass_lod": _mat("Car_GlassOpaque", "#0f1518", 0.05),
        "trim": _mat("Car_GlossBlack", "#0d0e0f", 0.32),
        "clad": _mat("Car_Plastic", "#18191a", 0.62),
        "frit": _mat("Car_Frit", "#040404", 0.9),
        "seal": _mat("Car_WindowSeal", "#040404", 0.88),
        "chrome": _mat("Car_Chrome", "#d5d8da", 0.1, 1.0),
        "mirror": _mat("Car_MirrorGlass", "#aeb7bd", 0.02, 1.0),
        "alloy": _mat("Car_Alloy", "#c9ccce", 0.28, 0.55),
        "rail": _mat("Car_RoofRail", "#a3a7ab", 0.35, 0.35),
        "alloy_dark": _mat("Car_AlloyDark", "#3a3d40", 0.3, 0.8),
        "steel": _mat("Car_SteelWheel", "#8d9194", 0.35, 0.7),
        "disc": _mat("Car_Disc", "#5d5955", 0.45, 1.0),
        "caliper": _mat("Car_Caliper", "#2a2b2d", 0.4, 0.3),
        "under": _mat("Car_Underbody", "#0d0d0d", 0.85),
        "seat": _mat("Car_Upholstery", "#1c1d1f", 0.85),
        "cabin": _mat("Car_InteriorTrim", "#2a2b2c", 0.7),
        "liner": _mat("Car_Headliner", "#55534f", 0.9),
    }


# -- smooth profiles ------------------------------------------------------------------------------

def pchip(xk, yk, x):
    """Monotone cubic interpolation (no overshoot) through (xk, yk)."""
    xk = np.asarray(xk, float)
    yk = np.asarray(yk, float)
    x = np.clip(np.asarray(x, float), xk[0], xk[-1])
    h = np.diff(xk)
    d = np.diff(yk) / h
    n = len(xk)
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    i = np.clip(np.searchsorted(xk, x) - 1, 0, n - 2)
    t = (x - xk[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * yk[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
            + (-2 * t3 + 3 * t2) * yk[i + 1] + (t3 - t2) * h[i] * m[i + 1])


class Prof:
    def __init__(self, keys):
        k = sorted(keys)
        self.x = [a for a, _ in k]
        self.y = [b for _, b in k]

    def __call__(self, y):
        return float(pchip(self.x, self.y, y))


class Spec:
    def __init__(self, **kw):
        kw.setdefault("sculpt", 1.0)
        self.__dict__.update(kw)
        for k in ("roof", "belt", "shoulder", "sill", "halfw", "roofw"):
            setattr(self, k, Prof(kw[k]))
        self.rail = Prof(kw["rail"]) if kw.get("rail") else None
        self.yF, self.yR = self.L / 2, -self.L / 2

    def rail_z(self, y):
        return self.rail(y) if self.rail else self.roof(y) - self.rail_drop


# -- polygon soup ---------------------------------------------------------------------------------

class Soup:
    def __init__(self):
        self.mats = []
        self.V, self.F, self.mi, self.uv = [], [], [], []
        self.sharp = []

    def m(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def face(self, pts, mat, uv=None):
        b = len(self.V)
        self.V.extend([tuple(map(float, p)) for p in pts])
        self.F.append(tuple(range(b, b + len(pts))))
        self.mi.append(self.m(mat))
        self.uv.extend(uv or [(0.0, 0.0)] * len(pts))

    def grid(self, P, mat, uv=None, closed_u=False, keep=None, flip=False):
        """Quads over a (nu, nv, 3) grid of points with shared vertices; uv (nu, nv, 2);
        keep(i, j) -> bool filters quads."""
        P = np.asarray(P, float)
        nu, nv = P.shape[:2]
        b = len(self.V)
        self.V.extend(map(tuple, P.reshape(-1, 3)))
        mi = self.m(mat) if not callable(mat) else None
        us = range(nu) if closed_u else range(nu - 1)
        for i in us:
            i2 = (i + 1) % nu
            for j in range(nv - 1):
                if keep is not None and not keep(i, j):
                    continue
                f = (b + i * nv + j, b + i2 * nv + j, b + i2 * nv + j + 1, b + i * nv + j + 1)
                c = [(i, j), (i2, j), (i2, j + 1), (i, j + 1)]
                if flip:
                    f, c = f[::-1], c[::-1]
                self.F.append(f)
                self.mi.append(mi if mi is not None else self.m(mat(i, j)))
                if uv is None:
                    self.uv.extend([(0.0, 0.0)] * 4)
                else:
                    self.uv.extend([tuple(uv[a][bb]) for a, bb in c])

    def box(self, c, size, mat, R=None):
        """Box at centre c, size (sx, sy, sz), optional 3x3 rotation."""
        c = np.asarray(c, float)
        R = np.eye(3) if R is None else np.asarray(R)
        h = np.asarray(size) / 2
        V = [c + R @ (h * np.array([x, y, z])) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]
        for f in ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)):
            self.face([V[i] for i in f], mat)

    def blob(self, c, ax, mat, axis=None, e=(0.5, 0.5), nu=12, nv=7, taper=(0.0, 0.0), matf=None, cut=None, flip=None):
        """Superellipsoid: half sizes ax = (long, a, b) along axis frame (3x3 columns: long, a, b);
        e = (exponent along, exponent around); taper = (a, b) shrink towards +long;
        matf(phi, alpha) -> material; cut = (alpha_min) clips the -long end flat."""
        c = np.asarray(c, float)
        Fm = np.eye(3) if axis is None else np.asarray(axis, float)

        def sp(x, p):
            return np.sign(x) * np.abs(x) ** p
        alphas = np.linspace(-math.pi / 2 if cut is None else cut, math.pi / 2, nv)
        phis = np.linspace(0, 2 * math.pi, nu, endpoint=False)
        G = np.zeros((nu, nv, 3))
        for i, ph in enumerate(phis):
            for j, al in enumerate(alphas):
                lo = ax[0] * sp(math.sin(al), e[0])
                rr = sp(math.cos(al), e[0])
                t = (lo / ax[0] + 1) / 2
                a = ax[1] * rr * sp(math.cos(ph), e[1]) * (1 - taper[0] * t)
                b = ax[2] * rr * sp(math.sin(ph), e[1]) * (1 - taper[1] * t)
                G[i, j] = c + Fm @ np.array([lo, a, b])
        mf = (lambda i, j: matf(phis[i], alphas[j])) if matf else mat
        self.grid(G, mf, closed_u=True, flip=np.linalg.det(Fm) < 0)
        if cut is not None:     # close the cut end
            ring = [G[i, 0] for i in range(nu)][::-1]
            self.face(ring, matf(0, cut) if matf else mat)

    def tube(self, pts, normals, w, h, mat):
        """A rectangular rod (w wide, h tall) along a polyline lying on a surface (normals)."""
        pts = [np.asarray(p, float) for p in pts]
        rings = []
        for k, p in enumerate(pts):
            t = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
            t /= np.linalg.norm(t) + 1e-9
            n = np.asarray(normals[k], float)
            sd = np.cross(n, t)
            sd /= np.linalg.norm(sd) + 1e-9
            rings.append([p - sd * w / 2, p + sd * w / 2, p + sd * w / 2 + n * h, p - sd * w / 2 + n * h])
        for k in range(len(rings) - 1):
            for q in range(4):
                q2 = (q + 1) % 4
                self.face([rings[k][q], rings[k + 1][q], rings[k + 1][q2], rings[k][q2]], mat)
        self.face(rings[0][::-1], mat)
        self.face(rings[-1], mat)

    def obj(self, name, col):
        return mesh_object(name, np.array(self.V), self.F, col, materials=self.mats, mat_index=self.mi, uvs=self.uv)


# -- body skin ------------------------------------------------------------------------------------

def _cr(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


SPANS = [1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 2, 2]


def section(s, y, lod):
    """Half section at y: refined (x, z) points from the underside return to the roof centre,
    and the indices of the key points in the refined list."""
    zs = s.sill(y)
    for ax in s.axles:
        dy = y - ax
        if abs(dy) < s.arch:
            zs = max(zs, s.wheel_r + math.sqrt(max(s.arch ** 2 - dy * dy, 0.0)) * 0.985)
    zc, zb, zt = s.shoulder(y), s.belt(y), s.roof(y)
    zb = max(zb, zc + 0.04)
    zs = min(zs, zc - s.rocker - 0.08)
    w = s.halfw(y)
    rw = s.roofw(y)
    zr = s.rail_z(y)
    g = float(np.clip((zt - zb - 0.03) / 0.12, 0, 1))
    if s.rail:
        g = float(np.clip((zr - zb - 0.02) / 0.10, 0, 1))
    zr = max(zr, zb + 0.02)
    fl = sum(s.flare * math.exp(-((y - ax) / (s.arch * 0.9)) ** 2) for ax in s.axles)
    sh = s.sculpt                 # how strongly the side is shaped (tuck, shoulder shelf)
    wl = w - 0.04 * sh + fl
    wm = w - 0.003 * sh + fl * 0.7
    rk = s.rocker
    K = [(wl - 0.085, zs), (wl - 0.012, zs + 0.008), (wl, zs + rk), (wm, zs + rk + 0.45 * (zc - zs - rk)),
         (w + 0.008, zc), (w - 0.024 * sh, zc + 0.55 * (zb - zc)), (w - 0.05 * sh, zb)]
    cab = [(w - 0.05 * sh - 0.012, zb + 0.012), None, (rw, zr), (rw - s.edge[0], zr + s.edge[1]), (rw * 0.55, zt - s.crown), (0.0, zt)]
    cab[1] = (0.5 * (cab[0][0] + rw) + s.tumble_bulge, 0.5 * (cab[0][1] + zr))
    dk = [(w - 0.05 * sh - 0.02, zb + 0.004), (w - 0.11, 0.6 * zb + 0.4 * zt - 0.004), (w * 0.72, 0.25 * zb + 0.75 * zt - 0.002),
          (w * 0.5, zt - 0.001), (w * 0.25, zt), (0.0, zt)]
    if s.rail:     # tall bodies: the side runs up past the rail to the roof edge
        z3 = max(zt - s.edge[1], zr + 0.012)
        cab[3] = (rw - s.edge[0], z3)
        cab[4] = (rw * 0.6, max(zt - s.crown, z3 + 0.004))
        cab[5] = (0.0, max(zt, z3 + 0.006))
    for a, b in zip(cab, dk):
        K.append((a[0] * g + b[0] * (1 - g), a[1] * g + b[1] * (1 - g)))
    K = np.array(K)
    sharp = {4, 6} | ({2} if s.clad else set()) | ({9} if g > 0.85 else set())
    spans = [1] * 12 if lod else list(SPANS)
    if s.rail and not lod:
        spans[9] = 4          # the tall panel above a van's windows: rows so shading stays even
    out = [K[0]]
    keys = [0]
    n = len(K)
    for a in range(n - 1):
        p1, p2 = K[a], K[a + 1]
        p0 = K[a - 1] if a > 0 and a not in sharp else 2 * p1 - p2
        if a + 1 == n - 1:
            p3 = np.array([-p1[0], p1[1]])
        elif a + 1 in sharp:
            p3 = 2 * p2 - p1
        else:
            p3 = K[a + 2]
        for t in np.arange(1, spans[a] + 1) / spans[a]:
            out.append(_cr(p0, p1, p2, p3, t) if t < 1 else p2)
        keys.append(len(out) - 1)
    return np.array(out), keys, g


def ring_of(half):
    return np.concatenate([half, (half[-2::-1] * np.array([-1, 1]))], 0)


class Skin:
    """The body surface: vertices, quads with materials and groups, sharp edges."""

    def __init__(self, s, lod):
        self.s = s
        self.lod = lod
        rcf, rcr = s.cap_r
        y0, y1 = s.yR + rcr + s.face_bulge[1], s.yF - rcf - s.face_bulge[0]
        n = s.sections // (3 if lod else 1)
        ys = list(np.linspace(y0, y1, n))
        na = 6 if lod else 14
        for ax in s.axles:
            ys += [ax + s.arch * math.cos(t) for t in np.linspace(0, math.pi, na)]
            ys += [ax - s.arch - 0.006, ax + s.arch + 0.006]
        if not lod:
            for w in s.windows:
                ys += [w["bot"][0], w["bot"][1], w["top"][0], w["top"][1]]
            for sc in s.screens:
                ys += list(sc["y"])
            ys += list(np.linspace(y1 - 0.25, y1, 5)) + list(np.linspace(y0, y0 + 0.25, 5))
        ys = np.array(sorted(set(np.round(np.clip(ys, y0, y1), 3))))
        keep = [ys[0]]
        for y in ys[1:]:
            if y - keep[-1] > (0.03 if lod else 0.012):
                keep.append(y)
        keep[-1] = y1
        ys = np.array(keep)
        rings, gs = [], []
        for y in ys:
            half, keys, g = section(s, y, lod)
            rings.append(ring_of(half))
            gs.append(g)
        self.keys = keys
        R = np.array(rings)                       # (n, m, 2)
        n, m = R.shape[:2]
        self.m = m
        V = np.zeros((n, m, 3))
        V[..., 0] = R[..., 0]
        V[..., 2] = R[..., 1]
        V[..., 1] = ys[:, None]
        if not lod:
            self._snap(V, ys, keys)
        self.V = [tuple(p) for p in V.reshape(-1, 3)]
        self.quads, self.qmat, self.qgroup = [], [], []
        idx = lambda i, j: i * m + j
        half_n = (m + 1) // 2
        clad_j = keys[2] if s.clad else keys[1]
        for i in range(n - 1):
            for j in range(m - 1):
                jj = j if j < half_n - 1 else m - 2 - j      # mirror index (left side)
                mat = "paint"
                if jj < clad_j:
                    mat = "clad" if s.clad else "under"
                self.quads.append((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)))
                self.qmat.append(mat)
                self.qgroup.append(0)
        # creases
        self.sharp = []
        for k in (keys[4], keys[6]):
            for j in (k, m - 1 - k):
                for i in range(n - 1):
                    self.sharp.append((idx(i, j), idx(i + 1, j)))
        for j in (keys[9], m - 1 - keys[9]):
            for i in range(n - 1):
                if gs[i] > 0.85 and gs[i + 1] > 0.85:
                    self.sharp.append((idx(i, j), idx(i + 1, j)))
        self.n_main = len(self.quads)
        self.cap_ref = {}
        self.ys = ys
        self.gs = gs
        self._cap(V[-1], idx(n - 1, 0), +1, rcf, 1)
        self._cap(V[0], idx(0, 0), -1, rcr, 2)
        self._deform()
        for side, yd, zd, rd, depth in getattr(s, "dents", ()):
            V2 = np.array(self.V)
            d2 = (V2[:, 1] - yd) ** 2 + (V2[:, 2] - zd) ** 2
            w = np.exp(-d2 / (rd * rd)) * (V2[:, 0] * side > 0.5)
            V2[:, 0] -= side * depth * w
            self.V = [tuple(p) for p in V2]

    def _snap(self, V, ys, keys):
        """Move the glass-band vertices nearest each window's front and back edges onto them, so
        the cut follows the slanted pillars instead of stepping along the sections."""
        s = self.s
        n, m = V.shape[:2]
        cache = {}

        def sec(y):
            if y not in cache:
                cache[y] = section(s, y, False)[0]
            return cache[y]
        moved = set()
        for w in s.windows:
            (yb0, yb1), (yt0, yt1) = w["bot"], w["top"]
            db, dt = w.get("db", 0.012), w.get("dt", 0.0)
            for ya, yt_ in ((yb0, yt0), (yb1, yt1)):
                if abs(yt_ - ya) < 0.01:      # an upright edge already lies on a section
                    continue
                zlo = s.belt(ya) + db
                zhi = (w.get("top_z") or s.rail_z(yt_)) - dt
                for j in range(keys[6] + 1, keys[9] + 1):
                    i0 = int(np.argmin(np.abs(ys - ya)))
                    z = V[i0, j, 2]
                    if z < zlo - 0.004 or z > zhi + 0.004:
                        continue
                    tt = float(np.clip((z - zlo) / max(zhi - zlo, 1e-6), 0, 1))
                    ye = ya + tt * (yt_ - ya)
                    i = int(np.argmin(np.abs(ys - ye)))
                    if i in (0, n - 1) or (i, j) in moved:
                        continue
                    sp = 0.5 * (ys[i + 1] - ys[i - 1])
                    if abs(ys[i] - ye) > 0.6 * sp:
                        continue
                    x, zz = sec(round(float(ye), 5))[j]
                    V[i, j] = (x, ye, zz)
                    V[i, m - 1 - j] = (-x, ye, zz)
                    moved.add((i, j))

    def _cap(self, ring, base, sgn, rc, group):
        s = self.s
        m = len(ring)
        P = ring[:, [0, 2]]
        c = P.mean(0)
        c[1] = 0.5 * (P[:, 1].min() + P[:, 1].max())
        N = np.zeros_like(P)
        area = 0.5 * sum(P[j - 1][0] * P[j][1] - P[j][0] * P[j - 1][1] for j in range(m))
        for j in range(m):
            t = P[(j + 1) % m] - P[j - 1]
            nn = np.array([-t[1], t[0]]) if area > 0 else np.array([t[1], -t[0]])     # the inward side
            N[j] = nn / (np.linalg.norm(nn) + 1e-9)
        y_end = ring[0, 1]
        prev = list(range(base, base + m))
        nr = 2 if self.lod else 4
        rings = []
        for k in range(1, nr + 1):
            ph = math.pi / 2 * k / nr
            Q = P + N * rc * (1 - math.cos(ph))
            rings.append((Q, y_end + sgn * rc * math.sin(ph)))
        y_face = rings[-1][1]
        for Q, y in rings:
            b = len(self.V)
            self.V.extend([(q[0], y, q[1]) for q in Q])
            cur = list(range(b, b + m))
            for j in range(m):
                j2 = (j + 1) % m
                self.quads.append((prev[j], cur[j], cur[j2], prev[j2]))
                self.qmat.append("paint")
                self.qgroup.append(group)
            prev = cur
        # the face: a regular grid inside the outline, zipped to it (clean shading, no fan)
        O = rings[-1][0]
        h = (m - 1) // 2
        eps = 0.012
        zb, zt = O[:, 1].min() + eps, O[h, 1] - eps

        def X(z):
            xs = []
            for j in range(m):
                a, b2 = O[j], O[(j + 1) % m]
                if (a[1] - z) * (b2[1] - z) <= 0 and a[1] != b2[1]:
                    t = (z - a[1]) / (b2[1] - a[1])
                    xs.append(a[0] + t * (b2[0] - a[0]))
            return max(xs) - eps if xs else 0.0
        NX, NZ = (4, 3) if self.lod else (16, 9)
        zs = np.linspace(zb, zt, NZ + 1)
        bulge = s.face_bulge[0 if sgn > 0 else 1]
        G = np.zeros((NZ + 1, NX + 1), int)
        for r, z in enumerate(zs):
            xr = X(z)
            for k, u in enumerate(np.linspace(-1, 1, NX + 1)):
                G[r, k] = len(self.V)
                vz = 2 * (z - zb) / (zt - zb) - 1
                self.V.append((u * xr, y_face + sgn * bulge * (1 - u * u) * (1 - vz * vz), z))
        for r in range(NZ):
            for k in range(NX):
                self.quads.append((G[r, k], G[r, k + 1], G[r + 1, k + 1], G[r + 1, k]))
                self.qmat.append("paint")
                self.qgroup.append(group)
        # boundary loop in the outline's order: up the right side, across the top, down the left,
        # back along the bottom
        B = [G[r, NX] for r in range(NZ + 1)] + [G[NZ, k] for k in range(NX - 1, -1, -1)] + \
            [G[r, 0] for r in range(NZ - 1, -1, -1)] + [G[0, k] for k in range(1, NX)]
        Vn = np.array(self.V)

        def params(idx):
            P2 = Vn[idx][:, [0, 2]]
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([P2, P2[:1]]), axis=0), axis=1))]
            return d[:-1] / d[-1]
        # start both loops at the bottom-right corner
        Oi = prev
        tO, tB = params(Oi), params(B)
        nb = len(B)
        i = j = 0
        while i < m or j < nb:
            o0, o1 = Oi[i % m], Oi[(i + 1) % m]
            b0, b1 = B[j % nb], B[(j + 1) % nb]
            ta = tO[(i + 1) % m] if i + 1 < m else 1.0
            tb = tB[(j + 1) % nb] if j + 1 < nb else 1.0
            if j >= nb or (i < m and ta <= tb):
                self.quads.append((o0, o1, b0))
                i += 1
            else:
                self.quads.append((o0, b1, b0))
                j += 1
            self.qmat.append("paint")
            self.qgroup.append(group)
        self.cap_ref[group] = (np.array([0.0, c[1]]), sgn)

    def _deform(self):
        """Sweep the corners back in plan and lean the faces (front and rear)."""
        s = self.s
        V = np.array(self.V)
        hw = s.halfw(0.0)
        for sgn, (sweep, lean, zl, span) in ((1, s.front_shape), (-1, s.rear_shape)):
            yE = s.yF if sgn > 0 else s.yR
            t = np.clip((V[:, 1] - (yE - sgn * span)) / (sgn * span), 0, 1)
            wgt = t * t * (3 - 2 * t)
            dy = sweep * (V[:, 0] / hw) ** 2 + lean * np.maximum(V[:, 2] - zl, 0)
            V[:, 1] -= sgn * dy * wgt
        self.V = [tuple(p) for p in V]

    def orient(self):
        """Majority vote on the loft so faces point out of the body; cap faces one by one."""
        V = np.array(self.V)
        ctr = np.array([0.0, 0.0, 0.75 * self.s.roof(0.0) * 0.6])
        for k, g in enumerate(self.qgroup):
            if g == 0:
                continue
            c2, sgn = self.cap_ref[g]
            p = V[list(self.quads[k])]
            nrm = np.cross(p[1] - p[0], p[2] - p[0])
            cc = p.mean(0)
            ref = np.array([cc[0] - c2[0], sgn * 1.0, cc[2] - c2[1]])
            if np.dot(nrm, ref) < 0:
                self.quads[k] = self.quads[k][::-1]
        for g in (0,):
            ids = [k for k, gg in enumerate(self.qgroup) if gg == g]
            score = 0.0
            for k in ids[::7]:
                q = self.quads[k]
                p = V[list(q)]
                nrm = np.cross(p[1] - p[0], p[2] - p[0])
                c = p.mean(0)
                dvec = c - np.array([0, np.clip(c[1], self.s.yR + 0.8, self.s.yF - 0.8), ctr[2]])
                score += np.sign(np.dot(nrm, dvec))
            if score < 0:
                for k in ids:
                    self.quads[k] = self.quads[k][::-1]

    def bvh(self):
        return BVHTree.FromPolygons([Vector(v) for v in self.V], [list(q) for q in self.quads], all_triangles=False)


# -- casting onto the skin ------------------------------------------------------------------------

class Fitter:
    def __init__(self, bvh, soup):
        self.bvh = bvh
        self.soup = soup

    def hit(self, origin, direction):
        loc, nrm, idx, d = self.bvh.ray_cast(Vector(origin), Vector(direction))
        if loc is None:
            return None, None
        n = np.array(nrm)
        if np.dot(n, direction) > 0:
            n = -n
        return np.array(loc), n

    def cast_grid(self, O, D):
        """O, D: (nu, nv, 3) ray origins / directions -> hit points, normals, ok mask."""
        nu, nv = O.shape[:2]
        H = np.zeros_like(O)
        N = np.zeros_like(O)
        ok = np.zeros((nu, nv), bool)
        for i in range(nu):
            for j in range(nv):
                p, n = self.hit(O[i, j], D[i, j])
                if p is not None:
                    H[i, j], N[i, j], ok[i, j] = p, n, True
        return H, N, ok

    def strip(self, pts, width, direction, mat, lift=0.0015, offset=0.0):
        """A thin band along points cast onto the body (shut lines, trims)."""
        d = np.asarray(direction, float)
        G = []
        for k, p in enumerate(pts):
            p = np.asarray(p, float)
            loc, nrm = self.hit(p - d * 2.0, d)
            if loc is None:
                if len(G) > 1:
                    self.soup.grid(np.array(G), mat)
                G = []
                continue
            if G and np.linalg.norm(loc - G[-1][0]) > 3 * np.linalg.norm(np.asarray(pts[k], float) - np.asarray(pts[k - 1], float)) + 0.03:
                if len(G) > 1:
                    self_grid = np.array(G)
                    self.soup.grid(self_grid, mat)
                G = []
            nxt = np.asarray(pts[min(k + 1, len(pts) - 1)], float) - np.asarray(pts[max(k - 1, 0)], float)
            side = np.cross(nrm, nxt)
            side /= np.linalg.norm(side) + 1e-9
            q = loc + nrm * lift + side * offset
            G.append([q - side * width / 2, q + side * width / 2])
        if len(G) > 1:
            self.soup.grid(np.array(G), mat)


# -- panes (glazing) ------------------------------------------------------------------------------

def _resample(pts, n):
    P = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(P[:1], n, 0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, P[:, k]) for k in range(P.shape[1])], 1)


def _coons(B, T, Lf, Rt, nu, nv):
    B, T = _resample(B, nu + 1), _resample(T, nu + 1)
    Lf, Rt = _resample(Lf, nv + 1), _resample(Rt, nv + 1)
    G = np.zeros((nu + 1, nv + 1, 2))
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            v = j / nv
            G[i, j] = ((1 - v) * B[i] + v * T[i] + (1 - u) * Lf[j] + u * Rt[j]
                       - ((1 - u) * (1 - v) * B[0] + u * (1 - v) * B[-1] + (1 - u) * v * T[0] + u * v * T[-1]))
    return G


def _pip(pts, poly):
    """Points (k, 2) inside polygon (n, 2)."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    P = np.asarray(poly)
    for i in range(len(P)):
        xi, yi = P[i]
        xj, yj = P[i - 1]
        if yi == yj:
            continue
        inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    return inside


class Pane:
    """A window: outline (bottom, right, top, left edges) in the 2D coordinates of a projection
    (axes a1, a2 of x/y/z), cast along -sign on axis c onto the skin."""

    def __init__(self, B, Rt, T, Lf, axes, c, sign, mat, see=True, inset=0.007, frame=0.014, frame_in=0.022,
                 res=0.07, name=""):
        self.B, self.Rt, self.T, self.Lf = [np.asarray(e, float) for e in (B, Rt, T, Lf)]
        self.axes, self.c, self.sign = axes, c, sign
        self.mat, self.see, self.inset, self.frame, self.frame_in, self.res = mat, see, inset, frame, frame_in, res
        self.name = name
        self.poly = np.concatenate([self.B, self.Rt[1:], self.T[::-1][1:], self.Lf[::-1][1:-1]], 0)
        self.bb = (self.poly[:, 0].min() - 0.01, self.poly[:, 0].max() + 0.01, self.poly[:, 1].min() - 0.01, self.poly[:, 1].max() + 0.01)
        P = self.poly
        k = len(P)
        area = 0.5 * sum(P[i - 1][0] * P[i][1] - P[i][0] * P[i - 1][1] for i in range(k))
        X = []
        for i in range(k):
            tt = P[(i + 1) % k] - P[i - 1]
            nn = np.array([tt[1], -tt[0]]) / (np.linalg.norm(tt) + 1e-9)
            X.append(P[i] + (nn if area > 0 else -nn) * 0.005)
        self.poly_x = np.array(X)

    def to3(self, uv, out=6.0):
        uv = np.asarray(uv, float)
        p = np.zeros(uv.shape[:-1] + (3,))
        p[..., self.axes[0]] = uv[..., 0]
        p[..., self.axes[1]] = uv[..., 1]
        p[..., self.c] = self.sign * out
        d = np.zeros(3)
        d[self.c] = -self.sign
        return p, d

    def coords(self, P):
        P = np.asarray(P)
        return np.stack([P[:, self.axes[0]], P[:, self.axes[1]]], 1)

    def facing(self, nrm, P):
        return nrm[self.c] * self.sign > 0.3 and (P[:, self.c] * self.sign > 0).all() if self.c != 2 else nrm[2] > 0.3


def build_pane(fit, soup, pane, M, lod=False):
    """Glass (inset) and its frame, cast onto the skin."""
    lb = np.linalg.norm(np.diff(pane.B, axis=0), axis=1).sum()
    lh = np.linalg.norm(np.diff(pane.Lf, axis=0), axis=1).sum()
    res = pane.res * (2.5 if lod else 1)
    nu, nv = max(2, int(math.ceil(lb / res))), max(1, int(math.ceil(lh / res)))
    G2 = _coons(pane.B, pane.T, pane.Lf, pane.Rt, nu, nv)
    O, d = pane.to3(G2)
    H, N, ok = fit.cast_grid(O, np.broadcast_to(d, O.shape))
    if not ok.all():
        # fill misses from neighbours (degenerate corners)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if not ok[i, j]:
                    ii, jj = np.argwhere(ok)[np.argmin(np.abs(np.argwhere(ok) - [i, j]).sum(1))]
                    H[i, j], N[i, j] = H[ii, jj], N[ii, jj]
    mat = pane.mat if not lod else (M["glass_lod"] if pane.mat in (M["glass"], M["privacy"]) else pane.mat)
    soup.grid(H - N * (pane.inset if not lod else -0.004), mat)
    if pane.frame and not lod:
        # frame: a band around the outline, from frame_in inside to frame outside
        poly = pane.poly
        k = len(poly)
        area = 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(k))
        outer, inner = [], []
        for i in range(k):
            t = poly[(i + 1) % k] - poly[i - 1]
            nn = np.array([t[1], -t[0]]) / (np.linalg.norm(t) + 1e-9)
            if area < 0:
                nn = -nn
            outer.append(poly[i] + nn * pane.frame)
            inner.append(poly[i] - nn * pane.frame_in)
        ring = np.array([inner + inner[:1], outer + outer[:1]]).transpose(1, 0, 2)    # (k+1, 2, 2)
        O, d = pane.to3(ring)
        H2, N2, ok2 = fit.cast_grid(O, np.broadcast_to(d, O.shape))
        good = ok2.all(1) & ((N2 @ -d) > 0.3).all(1)
        for i in range(1, len(good)):
            if np.linalg.norm(H2[i, 1] - H2[i - 1, 1]) > 0.12 or np.linalg.norm(H2[i, 1] - H2[i, 0]) > 2.5 * (pane.frame + pane.frame_in) + 0.02:
                good[i] = False
        soup.grid(H2 + N2 * 0.0025, M["seal"], keep=lambda i, j: good[i] and good[i + 1])


# -- wheels ---------------------------------------------------------------------------------------

def lathe(soup, c, side, prof, mat, n, uv_v=None, u_rep=1.0, a0=0.0):
    """Surface of revolution about the x axis through c; prof = [(radius, axial)] with axial
    measured outwards (towards side); profile ordered so the outside is on its left."""
    c = np.asarray(c, float)
    G = np.zeros((n + 1, len(prof), 3))
    for i in range(n + 1):
        a = a0 + 2 * math.pi * i / n
        for j, (rr, ax) in enumerate(prof):
            G[i, j] = c + np.array([ax * side, rr * math.cos(a), rr * math.sin(a)])
    uv = None
    if uv_v is not None:
        uv = [[((i / n) * u_rep * (-1 if side > 0 else 1), uv_v[j]) for j in range(len(prof))] for i in range(n + 1)]
    soup.grid(G, mat, uv=uv, flip=side < 0)


def wheel(soup, M, s, centre, side, lod=False, inner_dual=False):
    """Tyre, rim (style s.rim), brake disc and calliper. side = +1 right / -1 left."""
    from . import vehicle_fascia as VF
    c = np.asarray(centre, float)
    r, rim_r, tw = s.wheel_r, s.rim_r, s.tyre_w
    w = tw / 2
    sw = r - rim_r
    n = 14 if lod else 28
    # tyre: inner bead -> inner sidewall -> tread -> outer sidewall -> outer bead (outside on the left)
    prof = [(rim_r + 0.012, -w * 0.86), (rim_r + 0.45 * sw, -w * 1.0), (r - 0.03, -w * 0.97), (r - 0.008, -w * 0.88),
            (r, -w * 0.72), (r, w * 0.72), (r - 0.008, w * 0.88), (r - 0.03, w * 0.97), (rim_r + 0.6 * sw, w * 1.02),
            (rim_r + 0.25 * sw, w * 0.98), (rim_r + 0.012, w * 0.86)]
    if lod:
        prof = [prof[0], prof[1], prof[3], prof[6], prof[8], prof[10]]
    if inner_dual:
        prof = [(rr, -ax) for rr, ax in prof[::-1]]
    P = np.array(prof)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    vv = np.r_[0, np.cumsum(seg)]
    vv = VF.TYRE_V0 + (vv / vv[-1]) * (VF.TYRE_V1 - VF.TYRE_V0)
    if inner_dual:
        vv = vv[::-1]
    lathe(soup, c, side, prof, M["tyre"], n, uv_v=list(vv), u_rep=2.0)
    face_x = w * 0.78
    if s.rim == "steel":
        face_x = w * 0.55 if not inner_dual else -w * 0.4
    # rim lip and barrel
    lip = [(rim_r + 0.014, face_x - 0.004), (rim_r + 0.008, face_x + 0.006), (rim_r - 0.004, face_x + 0.004), (rim_r - 0.012, face_x - 0.012)]
    lathe(soup, c, side, lip, M["steel"] if s.rim == "steel" else M["alloy"], n)
    lathe(soup, c, side, [(rim_r - 0.012, face_x - 0.012), (rim_r - 0.012, -w * 0.6)], M["under"], max(8, n // 2))
    if lod or inner_dual:
        lathe(soup, c, side, [(rim_r - 0.012, face_x - 0.02), (0.0, face_x - 0.03)], M["alloy"] if s.rim != "steel" else M["steel"], 10)
        return
    # brake disc, hat and calliper behind the spokes
    dr = rim_r * 0.82
    lathe(soup, c, side, [(dr, face_x - 0.095), (dr, face_x - 0.12)], M["disc"], 16)
    lathe(soup, c, side, [(dr, face_x - 0.095), (rim_r * 0.3, face_x - 0.095), (rim_r * 0.28, face_x - 0.07), (0.0, face_x - 0.07)], M["disc"], 16)
    ca = math.radians(55 if side > 0 else 125)
    cc = c + np.array([side * (face_x - 0.1), dr * 0.92 * math.cos(ca) * (1 if side > 0 else 1), dr * 0.92 * math.sin(ca)])
    soup.blob(cc, (0.03, 0.09, 0.045), M["caliper"], axis=np.array([[1, 0, 0], [0, -math.sin(ca), math.cos(ca)], [0, math.cos(ca), math.sin(ca)]]).T,
              e=(0.4, 0.4), nu=8, nv=4)
    if s.rim == "steel":
        _steel_face(soup, M, s, c, side, face_x, inner_dual)
        return
    # dish behind the spokes (dark), hub, spokes, centre cap, nuts
    lathe(soup, c, side, [(rim_r - 0.012, face_x - 0.012), (rim_r * 0.6, face_x - 0.05), (rim_r * 0.3, face_x - 0.06)], M["alloy_dark"], n)
    lathe(soup, c, side, [(rim_r * 0.3, face_x - 0.03), (rim_r * 0.3, face_x - 0.012), (rim_r * 0.22, face_x - 0.004), (0.0, face_x - 0.004)], M["alloy"], 20)
    nsp = s.spokes
    for k in range(nsp):
        for off in s.spoke_pair:
            a = 2 * math.pi * k / nsp + off
            _spoke(soup, c, side, a, rim_r, face_x, s.spoke_w, M["alloy"])
    for k in range(5):
        a = 2 * math.pi * k / 5 + 0.3
        p = c + np.array([side * (face_x - 0.003), 0.056 * math.cos(a), 0.056 * math.sin(a)])
        lathe(soup, p, side, [(0.0105, -0.006), (0.0105, 0.004), (0.006, 0.008), (0.0, 0.008)], M["chrome"], 6)
    lathe(soup, c + np.array([side * 0.0, 0, 0]), side, [(0.032, face_x - 0.002), (0.03, face_x + 0.004), (0.0, face_x + 0.006)], M["trim"], 16)


def _spoke(soup, c, side, a, rim_r, face_x, sw, mat):
    """One spoke: a tapered bar from the hub to the rim, standing proud of the dish."""
    dvec = np.array([0, math.cos(a), math.sin(a)])
    tvec = np.array([0, -math.sin(a), math.cos(a)])
    xv = np.array([side, 0, 0])
    stations = [(rim_r * 0.24, face_x - 0.004, sw[0]), (rim_r * 0.62, face_x - 0.012, (sw[0] + sw[1]) / 2), (rim_r * 0.97, face_x - 0.004, sw[1])]
    rings = []
    for rr, fx, hw in stations:
        base = c + dvec * rr
        depth = 0.03
        rings.append([base + xv * fx - tvec * hw, base + xv * fx + tvec * hw,
                      base + xv * (fx - depth) + tvec * hw * 0.8, base + xv * (fx - depth) - tvec * hw * 0.8])
    for k in range(len(rings) - 1):
        for q in range(4):
            q2 = (q + 1) % 4
            f = [rings[k][q], rings[k + 1][q], rings[k + 1][q2], rings[k][q2]]
            soup.face(f if side > 0 else f[::-1], mat)


def _steel_face(soup, M, s, c, side, face_x, inner):
    """A van's steel wheel: dished centre with vent holes, six nuts and a centre cap."""
    rim_r = s.rim_r
    fx = face_x
    lathe(soup, c, side, [(rim_r - 0.012, fx - 0.012), (rim_r * 0.82, fx - 0.03), (rim_r * 0.62, fx - 0.03), (rim_r * 0.5, fx - 0.012),
                          (rim_r * 0.36, fx), (0.0, fx)], M["steel"], 24)
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        p = c + np.array([side * (fx - 0.026), rim_r * 0.72 * math.cos(a), rim_r * 0.72 * math.sin(a)])
        soup.blob(p, (0.004, 0.032, 0.02), M["under"], axis=np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]]).T,
                  e=(1.0, 0.7), nu=6, nv=3)
    for k in range(6):
        a = 2 * math.pi * k / 6
        p = c + np.array([side * (fx + 0.001), 0.068 * math.cos(a), 0.068 * math.sin(a)])
        lathe(soup, p, side, [(0.012, -0.004), (0.012, 0.012), (0.007, 0.016), (0.0, 0.016)], M["chrome"], 6)
    lathe(soup, c, side, [(0.045, fx), (0.042, fx + 0.025), (0.0, fx + 0.03)], M["trim"] if not inner else M["steel"], 16)


def arch_liner(soup, M, s, y, side, depth):
    """Black wheel-well liner inside the arch, so the body never looks hollow."""
    hw = s.halfw(y)
    n = 14
    r = s.arch + 0.01
    G = np.zeros((n + 1, 2, 3))
    angs = np.linspace(-0.25, math.pi + 0.25, n + 1)
    for i, a in enumerate(angs):
        for j, x in enumerate((hw - depth, hw - 0.06)):
            G[i, j] = (side * x, y + r * math.cos(a), s.wheel_r + r * math.sin(a))
    soup.grid(G, M["under"], flip=side > 0)
    P = [(side * (hw - depth), y + r * math.cos(a), s.wheel_r + r * math.sin(a)) for a in angs]
    soup.face(P if side > 0 else P[::-1], M["under"])


# -- interior -------------------------------------------------------------------------------------

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def seat(soup, M, x, y, zf, w=0.5, back_h=0.62, rake=0.28, head=True, low=False):
    """A seat facing +y: cushion, backrest and headrest, rounded."""
    nu = 6 if low else 8
    soup.blob((x, y, zf + 0.26), (0.08, w / 2, 0.25), M["seat"], axis=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
              e=(0.35, 0.35), nu=nu, nv=3 if low else 4)
    R = _rot_x(-rake)
    bc = np.array([x, y - 0.24, zf + 0.3]) + R @ np.array([0, 0, back_h / 2])
    soup.blob(bc, (back_h / 2, w / 2 * 0.95, 0.075), M["seat"], axis=R @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
              e=(0.3, 0.4), nu=nu, nv=4 if low else 5, taper=(0.12, 0.0))
    if head:
        hc = np.array([x, y - 0.24, zf + 0.3]) + R @ np.array([0, 0.02, back_h + 0.13])
        soup.blob(hc, (0.1, 0.13, 0.05), M["seat"], axis=R @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T, e=(0.3, 0.3), nu=8, nv=3 if low else 4)


def interior(soup, M, s):
    """Cabin: floor, door cards, dash with steering wheel (right-hand drive), seats, shelf, liner."""
    I = s.cabin
    y0, y1 = I["rear"], I["dash"]
    zf = s.sill(0) + 0.1
    hw = s.halfw(0.0) - 0.075
    zb = s.belt(0.0)
    soup.face([(-hw, y0, zf), (hw, y0, zf), (hw, y1, zf), (-hw, y1, zf)], M["cabin"])
    for sx in (-1, 1):
        P = [(sx * hw, y0, zf), (sx * hw, y1, zf), (sx * hw, y1, zb - 0.03), (sx * hw, y0, zb - 0.03)]
        soup.face(P if sx < 0 else P[::-1], M["cabin"])
    # headliner
    # (following the roof line, so a variant's lower or sloping roof never leaves it standing proud)
    yl0, yl1 = I.get("liner", (y0 + 0.2, y1 - 0.5))
    lys = np.linspace(yl0, yl1, 8)
    for ly0, ly1 in zip(lys[:-1], lys[1:]):
        lz0, lz1 = s.roof(ly0) - 0.07, s.roof(ly1) - 0.07
        lw0, lw1 = min(s.roofw(ly0), s.roofw(0)) - 0.05, min(s.roofw(ly1), s.roofw(0)) - 0.05
        soup.face([(-lw0, ly0, lz0), (lw0, ly0, lz0), (lw1, ly1, lz1), (-lw1, ly1, lz1)], M["liner"])
    # dash: a sloping top from the screen base, a front face and the firewall
    zd = I.get("dash_z", zb + 0.02)
    soup.face([(-hw, y1 + 0.05, zd - 0.02), (hw, y1 + 0.05, zd - 0.02), (hw, y1 - 0.35, zd), (-hw, y1 - 0.35, zd)], M["cabin"])
    soup.face([(-hw, y1 - 0.35, zd), (hw, y1 - 0.35, zd), (hw, y1 - 0.42, zd - 0.3), (-hw, y1 - 0.42, zd - 0.3)], M["cabin"])
    soup.face([(-hw, y1 - 0.42, zd - 0.3), (hw, y1 - 0.42, zd - 0.3), (hw, y1 + 0.05, zf), (-hw, y1 + 0.05, zf)], M["cabin"])
    soup.box((0.0, y1 - 0.55, zf + 0.18), (0.22, 0.5, 0.3), M["cabin"])        # centre console
    # steering wheel (right-hand drive) and binnacle
    xw = I.get("wheel_x", 0.37)
    R = _rot_x(math.radians(-62))
    wc = np.array([xw, y1 - 0.5, zd - 0.06])
    G = np.zeros((16, 5, 3))
    for i in range(16):
        a = 2 * math.pi * i / 16
        for j in range(5):
            b = 2 * math.pi * j / 4
            rr = 0.185 + 0.017 * math.cos(b)
            G[i, j] = wc + R @ np.array([rr * math.cos(a), 0.017 * math.sin(b), rr * math.sin(a)])
    soup.grid(G, M["seat"], closed_u=True)
    soup.blob(wc + R @ np.array([0, -0.03, 0]), (0.03, 0.07, 0.06), M["seat"], axis=R, e=(0.6, 0.5), nu=8, nv=3)
    soup.box(wc + np.array([0, 0.18, 0.02]), (0.05, 0.36, 0.05), M["seat"], R=_rot_x(math.radians(-25)))
    soup.box((xw, y1 - 0.33, zd + 0.04), (0.36, 0.14, 0.08), M["cabin"])
    for row in I["rows"]:
        y, xs, w, bh = row
        for x in xs:
            seat(soup, M, x, y, zf, w=w, back_h=bh, low=I.get("lowpoly", False))
    if I.get("shelf"):
        ys0, ys1, zsh = I["shelf"]
        soup.face([(-hw, ys0, zsh), (hw, ys0, zsh), (hw, ys1, zsh), (-hw, ys1, zsh)], M["cabin"])


# -- assembling a vehicle -------------------------------------------------------------------------

def side_window(s, w, side, M):
    """A side window from its corner positions along the car (y) and the belt / rail lines."""
    (yb0, yb1), (yt0, yt1) = w["bot"], w["top"]
    db, dt = w.get("db", 0.012), w.get("dt", 0.0)
    zbot = lambda y: s.belt(y) + db
    ztop = lambda y: s.rail_z(y) - dt
    B = [(y, zbot(y)) for y in np.linspace(yb0, yb1, 9)]
    T = [(y, ztop(y)) for y in np.linspace(yt0, yt1, 9)]
    if "top_z" in w:
        T = [(y, w["top_z"]) for y in np.linspace(yt0, yt1, 9)]
    Lf = [B[0], T[0]]
    Rt = [B[-1], T[-1]]
    mat = M[w.get("mat", "glass")]
    return Pane(B, Rt, T, Lf, (1, 2), 0, side, mat, see=w.get("see", True), frame=w.get("frame", 0.014), res=w.get("res", 0.07),
                name=w.get("name", "side"))


def screen(s, sc, M):
    """Windscreen / rear screen in plan (x, y), cast from above."""
    yb, yt = sc["y"]                # bottom (cowl / tail) and top (header / roof) y
    # the sides run along the roof-edge row of the skin, the ends along sections: a clean cut
    ex = lambda y: section(s, y, False)[0][section(s, y, False)[1][10]][0] - sc.get("inset_x", 0.0)
    ysd = np.linspace(yb, yt, 10)
    Rt = [(ex(y), y) for y in ysd]
    Lf = [(-ex(y), y) for y in ysd]
    B = [(x, yb) for x in np.linspace(Lf[0][0], Rt[0][0], 11)]
    T = [(x, yt) for x in np.linspace(Lf[-1][0], Rt[-1][0], 11)]
    return Pane(B, Rt, T, Lf, (0, 1), 2, 1, M[sc.get("mat", "glass")], see=sc.get("see", True),
                frame=sc.get("frame", 0.016), frame_in=0.03, inset=0.005, name="screen")


def end_window(s, ew, M):
    """A window on an end face (minibus rear doors): rectangle in (x, z), cast along y."""
    x0, x1, z0, z1 = ew["rect"]
    sg = ew["sign"]
    B = [(x, z0) for x in np.linspace(x0, x1, 5)]
    T = [(x, z1) for x in np.linspace(x0, x1, 5)]
    return Pane(B, [B[-1], T[-1]], T, [B[0], T[0]], (0, 2), 1, sg, M[ew.get("mat", "privacy")], see=False,
                frame=0.035, inset=-0.004, name="end")


def _snap_to_panes(skin, V, status, panes, fit):
    """Paint quads the edge of a see-through window crosses keep their outside corners; the
    corners inside the window slide onto its outline (back on the skin), so the paint ends on the
    glass edge instead of leaving slivers over the glass where the outline cuts quads slantwise
    (the windscreen's lower corners)."""
    moved = {}
    for k, q in enumerate(skin.quads):
        st = status[k]
        if not (st in ("keep", "frit") or isinstance(st, tuple)) or skin.qmat[k] != "paint":
            continue
        P = V[list(q)]
        for pane in panes:
            if not pane.see or not (pane.bb[0] <= P[:, pane.axes[0]].max() and P[:, pane.axes[0]].min() <= pane.bb[1]
                                    and pane.bb[2] <= P[:, pane.axes[1]].max() and P[:, pane.axes[1]].min() <= pane.bb[3]):
                continue
            nrm = np.cross(P[2] - P[0], P[3] - P[1])
            nrm /= np.linalg.norm(nrm) + 1e-9
            if not pane.facing(nrm, P):
                continue
            ins = _pip(pane.coords(P), pane.poly)
            if not ins.any() or ins.all():
                continue
            status[k] = "keep"
            for vi, inside in zip(q, ins):
                if inside and vi not in moved:
                    moved[vi] = pane
    for vi, pane in moved.items():
        uv = pane.coords(V[vi][None, :])[0]
        poly = pane.poly
        best, bd = uv, 1e9
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            ab = b - a
            t = float(np.clip(np.dot(uv - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1))
            c = a + ab * t
            d = np.linalg.norm(uv - c)
            if d < bd:
                best, bd = c, d
        if bd > 0.25:
            continue
        o, d = pane.to3(best)
        p, _ = fit.hit(o, d)
        if p is not None:
            V[vi] = p
            skin.V[vi] = tuple(p)


def vehicle(col, name, s, lod=False, paint=None, scrap=False, plate=0):
    from . import vehicle_fascia as VF
    M = mats(paint)
    M["tyre"] = VF.tyre_material(s.tyre_size)
    if scrap:
        M["paint"] = VF.scrap_paint()
    s.plate_text = PLATES[s.kind][plate % len(PLATES[s.kind])]
    skin = Skin(s, lod)
    skin.orient()
    bvh = skin.bvh()
    soup = Soup()
    fit = Fitter(bvh, soup)
    V = np.array(skin.V)
    # glazing
    panes = []
    for w in s.windows:
        for side in (-1, 1):
            if w.get("only") and w["only"] != side:
                continue
            panes.append(side_window(s, w, side, M))
    for sc in s.screens:
        panes.append(screen(s, sc, M))
    for ew in s.end_windows:
        panes.append(end_window(s, ew, M))
    smashed = set(s.smashed) if scrap else set()
    # holes in the skin (lod: glass painted on the skin instead)
    status = ["keep"] * len(skin.quads)
    for k, q in enumerate(skin.quads):
        if len(q) != 4 or skin.qmat[k] != "paint":
            continue
        P = V[list(q)]
        nrm = np.cross(P[2] - P[0], P[3] - P[1])
        nrm /= np.linalg.norm(nrm) + 1e-9
        for pi_, pane in enumerate(panes):
            if not pane.facing(nrm, P):
                continue
            if not (pane.bb[0] <= P[:, pane.axes[0]].max() and P[:, pane.axes[0]].min() <= pane.bb[1]
                    and pane.bb[2] <= P[:, pane.axes[1]].max() and P[:, pane.axes[1]].min() <= pane.bb[3]):
                continue
            ins = _pip(pane.coords(P), pane.poly_x)
            if ins.all():
                status[k] = "hole" if (pane.see and not lod) else ("glass" if lod else "frit")
                break
            if ins.sum() == 3 and pane.see and not lod and s.kind != "transit":
                # a slanted edge crosses the quad on its diagonal: cut it there, keep the outer half
                out = int(np.argmin(ins))
                status[k] = ("tri", (q[(out - 1) % 4], q[out], q[(out + 1) % 4]))
                break
            if _pip(pane.coords(P.mean(0)[None, :]), pane.poly)[0]:
                status[k] = "frit"
                break
    if not lod:
        _snap_to_panes(skin, V, status, panes, fit)
    for k, st in enumerate(status):
        if st == "frit":
            skin.qmat[k] = "frit"
        elif st == "glass":
            skin.qmat[k] = "glass_lod"
    for pi_, pane in enumerate(panes):
        if pi_ in smashed:
            # an empty frame with a few shards left in it
            _shards(fit, soup, pane, M)
            continue
        if lod:
            continue
        build_pane(fit, soup, pane, M, lod)
    # the skin itself
    body = Soup()
    body.mats = soup.mats
    body.V = list(skin.V)
    for k, q in enumerate(skin.quads):
        if status[k] == "hole":
            continue
        if isinstance(status[k], tuple):
            q = status[k][1]
        body.F.append(tuple(q))
        body.mi.append(body.m(M[skin.qmat[k]]))
        body.uv.extend([(0.0, 0.0)] * len(q))
    nb = len(body.V)
    sharp = list(skin.sharp)
    # everything else goes after the skin's vertices
    for f in soup.F:
        body.F.append(tuple(i + nb for i in f))
    body.V.extend(soup.V)
    body.mi.extend(soup.mi)
    body.uv.extend(soup.uv)
    det = Soup()
    det.mats = body.mats
    fit.soup = det
    VF.details(fit, det, s, M, lod, scrap)
    if not lod:
        interior(det, M, s)
    # underbody, wheel wells, wheels
    hw = s.halfw(0.0)
    zu = s.sill(0.0) + 0.03
    det.face([(-hw + 0.1, s.yR + 0.35, zu), (hw - 0.1, s.yR + 0.35, zu), (hw - 0.1, s.yF - 0.35, zu), (-hw + 0.1, s.yF - 0.35, zu)][::-1], M["under"])
    for ai, ax in enumerate(s.axles):
        for sx in (-1, 1):
            arch_liner(det, M, s, ax, sx, s.liner_depth)
            if scrap:
                continue
            tx = s.track[ai]
            wheel(det, M, s, (sx * tx, ax, s.wheel_r), sx, lod)
            if s.dual and ai == 0:
                wheel(det, M, s, (sx * (tx - s.tyre_w - 0.02), ax, s.wheel_r), sx, lod, inner_dual=True)
    if scrap:
        VF.blocks(det, M, s)
    nb = len(body.V)
    for f in det.F:
        body.F.append(tuple(i + nb for i in f))
    body.V.extend(det.V)
    body.mi.extend(det.mi)
    body.uv.extend(det.uv)
    if s.lift:
        body.V = [(x, y, z + s.lift) for x, y, z in body.V]
    obj = body.obj(name, col)
    me = obj.data
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(48))
    # skin creases
    lookup = {tuple(sorted(e.vertices)): e.index for e in me.edges}
    attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", "BOOLEAN", "EDGE")
    vals = np.zeros(len(me.edges), bool)
    attr.data.foreach_get("value", vals)
    for a, b in sharp:
        e = lookup.get((min(a, b), max(a, b)))
        if e is not None:
            vals[e] = True
    attr.data.foreach_set("value", vals)
    if scrap:
        VF.box_uv_for(obj, M["paint"], tile=2.0)
    if not lod:
        VF.vertex_shading(obj, s, scrap)
    obj.name = name
    return obj


def _shards(fit, soup, pane, M):
    """A smashed side window: a jagged fringe of glass left in the bottom of the frame."""
    B = pane.B
    rng = np.random.default_rng(7)
    edge = [B[0]]
    for k in range(1, len(B)):
        edge.append(B[k])
    top = [(p[0], p[1] + 0.03 + 0.09 * rng.random() ** 2) for p in B]
    pn = Pane(B, [B[-1], top[-1]], top, [B[0], top[0]], pane.axes, pane.c, pane.sign, M["glass"], frame=pane.frame,
              res=0.04)
    build_pane(fit, soup, pn, M)


# -- the fleet ------------------------------------------------------------------------------------


def _vary(v, d):
    """A body variant of a spec (its keyword dict d): v's keys replace d's (a new roof line,
    windows, rear screen, wheels, rails), then sx stretches the whole car along its length
    (overhangs, wheelbase, every station of the profiles, windows, doors and cabin) and hk scales
    the cabin above z_hk (default 0.95 m: the roof, the rails, the end windows)."""
    sx, hk, zb = v.pop("sx", 1.0), v.pop("hk", 1.0), v.pop("z_hk", 0.95)
    d.update(v)

    def y(a):
        return a * sx

    def z(a):
        return zb + (a - zb) * hk if a > zb else a
    d["L"] = d["L"] * sx
    d["axles"] = tuple(y(a) for a in d["axles"])
    for k in ("roof", "rail"):
        if d.get(k):
            d[k] = [(y(a), z(b)) for a, b in d[k]]
    for k in ("belt", "shoulder", "sill", "halfw", "roofw"):
        d[k] = [(y(a), b) for a, b in d[k]]
    d["windows"] = [dict(w, bot=tuple(y(a) for a in w["bot"]), top=tuple(y(a) for a in w["top"])) for w in d["windows"]]
    d["screens"] = [dict(sc, y=tuple(y(a) for a in sc["y"])) for sc in d["screens"]]
    d["end_windows"] = [dict(e, rect=(e["rect"][0], e["rect"][1], z(e["rect"][2]), z(e["rect"][3]))) for e in d["end_windows"]]
    c = dict(d["cabin"])
    for k in ("rear", "dash"):
        c[k] = y(c[k])
    c["rows"] = [(y(r[0]),) + tuple(r[1:]) for r in c["rows"]]
    if "shelf" in c:
        c["shelf"] = (y(c["shelf"][0]), y(c["shelf"][1]), c["shelf"][2])
    c["liner"] = (y(c["liner"][0]), c["liner"][1])
    d["cabin"] = c
    d["doors"] = [tuple(y(a) for a in dr) for dr in d["doors"]]
    d["handles"] = tuple(y(a) for a in d["handles"])
    d["mirror"] = (y(d["mirror"][0]), d["mirror"][1])
    d["fuel"] = (y(d["fuel"][0]), d["fuel"][1])
    if d.get("fin") is not None:
        d["fin"] = y(d["fin"])
    return d

def hatch_spec(**v):
    """VW Golf Mk8 5-door: L 4284, W 1789, H 1456, WB 2636; 225/45 R17."""
    L, fo, wb = 4.284, 0.865, 2.636
    yF = L / 2
    af = yF - fo
    ar = af - wb
    return Spec(**_vary(v, dict(
        kind="golf", tyre_size="225/45 R17 94W", L=L, sections=84, axles=(ar, af), wheel_r=0.317, rim_r=0.216, tyre_w=0.225, track=(0.772, 0.778),
        arch=0.352, flare=0.016, sculpt=1.3, rocker=0.05, clad=False, cap_r=(0.075, 0.06), face_bulge=(0.008, 0.006),
        front_shape=(0.11, 0.0, 0.6, 0.5), rear_shape=(0.08, 0.0, 0.6, 0.45),
        roof=[(yF, 0.80), (2.06, 0.822), (1.8, 0.875), (1.5, 0.94), (1.2, 0.988), (1.13, 1.0), (0.8, 1.165), (0.5, 1.32),
              (0.27, 1.405), (-0.05, 1.448), (-0.6, 1.455), (-1.2, 1.428), (-1.5, 1.392), (-1.62, 1.355), (-1.8, 1.21),
              (-1.93, 1.06), (-2.0, 1.01), (-2.08, 0.985), (-yF, 0.965)],
        belt=[(yF, 0.80), (1.7, 0.885), (1.13, 0.965), (0.4, 0.972), (-0.6, 0.985), (-1.2, 1.0), (-1.45, 1.03), (-1.75, 1.03),
              (-2.0, 1.0), (-yF, 0.97)],
        shoulder=[(yF, 0.74), (1.6, 0.80), (1.0, 0.85), (0.0, 0.872), (-1.2, 0.89), (-1.8, 0.915), (-yF, 0.925)],
        sill=[(yF, 0.255), (1.95, 0.225), (1.65, 0.2), (-1.75, 0.2), (-2.0, 0.27), (-yF, 0.33)],
        halfw=[(yF, 0.77), (2.05, 0.835), (1.9, 0.872), (1.6, 0.89), (1.0, 0.8945), (-1.0, 0.8945), (-1.6, 0.89),
               (-1.95, 0.876), (-2.07, 0.845), (-yF, 0.80)],
        roofw=[(1.13, 0.765), (0.65, 0.71), (0.27, 0.672), (-0.6, 0.672), (-1.4, 0.645), (-1.7, 0.66), (-2.0, 0.72)],
        rail_drop=0.05, edge=(0.03, 0.03), crown=0.012, tumble_bulge=0.008, liner_depth=0.34,
        windows=[dict(bot=(-0.06, 1.06), top=(-0.06, 0.32)), dict(bot=(-1.42, -0.16), top=(-1.18, -0.16))],
        screens=[dict(y=(1.145, 0.285), x=(0.70, 0.62), curve=(0.07, 0.03)),
                 dict(y=(-1.94, -1.60), x=(0.60, 0.555), curve=(0.04, 0.02), mat="privacy")],
        end_windows=[], smashed=[], dual=False, lift=0.0, rim="twin5", spokes=5, spoke_pair=(-0.12, 0.12), spoke_w=(0.016, 0.011),
        cabin=dict(rear=-1.62, dash=1.1, rows=[(0.1, (-0.37, 0.37), 0.5, 0.62), (-0.95, (-0.42, 0.0, 0.42), 0.42, 0.56)],
                   shelf=(-1.95, -1.3, 1.0), liner=(-1.5, 0.18)),
        doors=[(0.97, -0.10), (-0.10, -1.08)], handles=(0.22, -0.84), handle_z=0.895, mirror=(0.98, 1.03),
        fuel=(-1.62, 0.84), spoiler=True, rear_wiper=True, roof_rails=False, fin=-1.2, plates=(0.485, 0.62),
        fascia=(0.22, 0.83, 0.3, 1.0), wrap=0.88,
    )))


def saloon_spec(**v):
    """BMW 3 Series G20: L 4709, W 1827, H 1442, WB 2851; overhangs 790 / 1068; 225/45 R18."""
    L, fo, wb = 4.709, 0.79, 2.851
    yF = L / 2
    af = yF - fo
    ar = af - wb
    return Spec(**_vary(v, dict(
        kind="bmw", tyre_size="225/45 R18 95Y", L=L, sections=82, axles=(ar, af), wheel_r=0.330, rim_r=0.229, tyre_w=0.225, track=(0.79, 0.80),
        arch=0.358, flare=0.018, sculpt=1.35, fascia_dz=(0.0, 0.03), rocker=0.05, clad=False, cap_r=(0.07, 0.06), face_bulge=(0.008, 0.006),
        front_shape=(0.16, 0.12, 0.5, 0.6), rear_shape=(0.08, -0.05, 0.75, 0.5),
        roof=[(yF, 0.705), (2.26, 0.73), (2.0, 0.785), (1.5, 0.862), (1.0, 0.928), (0.80, 0.958), (0.45, 1.15), (0.1, 1.335),
              (-0.12, 1.405), (-0.45, 1.442), (-0.9, 1.432), (-1.22, 1.39), (-1.45, 1.29), (-1.72, 1.10), (-1.86, 1.03),
              (-2.0, 1.0), (-2.25, 0.995), (-yF, 0.975)],
        belt=[(yF, 0.73), (1.6, 0.865), (0.8, 0.952), (0.0, 0.97), (-1.0, 0.99), (-1.6, 1.0), (-2.0, 0.99), (-yF, 0.96)],
        shoulder=[(yF, 0.665), (1.8, 0.755), (1.0, 0.83), (0.0, 0.85), (-1.5, 0.87), (-yF, 0.885)],
        sill=[(yF, 0.225), (2.1, 0.2), (1.85, 0.185), (-1.9, 0.185), (-2.2, 0.26), (-yF, 0.33)],
        halfw=[(yF, 0.78), (2.26, 0.862), (2.0, 0.9), (1.5, 0.912), (-1.5, 0.9135), (-2.1, 0.9), (-2.28, 0.862), (-yF, 0.80)],
        roofw=[(0.8, 0.76), (0.3, 0.70), (-0.12, 0.668), (-0.8, 0.672), (-1.3, 0.645), (-1.8, 0.72)],
        rail_drop=0.05, edge=(0.03, 0.03), crown=0.012, tumble_bulge=0.008, liner_depth=0.34,
        windows=[dict(bot=(-0.28, 0.74), top=(-0.28, -0.04)), dict(bot=(-1.52, -0.38), top=(-1.2, -0.38))],
        screens=[dict(y=(0.805, -0.105), x=(0.71, 0.62), curve=(0.08, 0.03)),
                 dict(y=(-1.83, -1.25), x=(0.62, 0.58), curve=(0.05, 0.02))],
        end_windows=[], smashed=[], dual=False, lift=0.0, rim="vspoke", spokes=5, spoke_pair=(-0.16, 0.16), spoke_w=(0.012, 0.016),
        cabin=dict(rear=-1.7, dash=0.77, rows=[(-0.2, (-0.38, 0.38), 0.5, 0.62), (-1.15, (-0.42, 0.0, 0.42), 0.42, 0.55)],
                   shelf=(-1.85, -1.42, 1.0), liner=(-1.3, 0.1)),
        doors=[(0.79, -0.33), (-0.33, -1.21)], handles=(0.0, -0.98), handle_z=0.87, mirror=(0.66, 1.0),
        fuel=(-1.72, 0.86), spoiler=False, rear_wiper=False, roof_rails=False, fin=-1.05, plates=(0.425, 0.6),
        fascia=(0.21, 0.725, 0.3, 1.03), wrap=0.85,
    )))


def suv_spec(**v):
    """Nissan Qashqai J12: L 4425, W 1835, H 1625, WB 2665; 235/55 R18; 180 mm clearance."""
    L, fo, wb = 4.425, 0.925, 2.665
    yF = L / 2
    af = yF - fo
    ar = af - wb
    return Spec(**_vary(v, dict(
        kind="qashqai", tyre_size="235/55 R18 100V", L=L, sections=68, axles=(ar, af), wheel_r=0.358, rim_r=0.229, tyre_w=0.235, track=(0.79, 0.795),
        arch=0.398, flare=0.022, rocker=0.11, clad=True, cap_r=(0.07, 0.065), face_bulge=(0.01, 0.006), sculpt=1.3,
        fascia_dz=(0.03, 0.0), front_shape=(0.15, 0.0, 0.75, 0.6), rear_shape=(0.09, 0.0, 0.8, 0.45),
        roof=[(yF, 0.90), (2.12, 0.925), (1.9, 0.965), (1.6, 1.03), (1.3, 1.088), (1.18, 1.115), (0.85, 1.29), (0.52, 1.47),
              (0.3, 1.57), (0.05, 1.61), (-0.3, 1.622), (-0.9, 1.618), (-1.5, 1.585), (-1.78, 1.55), (-1.88, 1.47), (-1.98, 1.33),
              (-2.07, 1.18), (-2.14, 1.115), (-yF, 1.09)],
        belt=[(yF, 0.885), (2.0, 0.94), (1.6, 1.035), (1.18, 1.105), (0.0, 1.12), (-1.0, 1.145), (-1.5, 1.17), (-1.8, 1.17), (-2.05, 1.14), (-yF, 1.1)],
        shoulder=[(yF, 0.815), (1.6, 0.93), (1.0, 0.975), (0.0, 0.99), (-1.5, 1.02), (-yF, 1.035)],
        sill=[(yF, 0.37), (2.0, 0.31), (1.7, 0.265), (-1.85, 0.265), (-2.05, 0.33), (-yF, 0.40)],
        halfw=[(yF, 0.80), (2.12, 0.87), (1.9, 0.905), (1.4, 0.9175), (-1.4, 0.9175), (-1.95, 0.905), (-2.12, 0.87), (-yF, 0.82)],
        roofw=[(1.18, 0.78), (0.7, 0.73), (0.3, 0.69), (-0.9, 0.682), (-1.6, 0.65), (-1.9, 0.67), (-2.1, 0.72)],
        rail_drop=0.05, edge=(0.03, 0.03), crown=0.014, tumble_bulge=0.006, liner_depth=0.36,
        windows=[dict(bot=(-0.02, 1.08), top=(-0.02, 0.38)), dict(bot=(-0.94, -0.12), top=(-0.94, -0.12), mat="privacy"),
                 dict(bot=(-1.70, -1.02), top=(-1.56, -1.02), db=0.016, mat="privacy")],
        screens=[dict(y=(1.185, 0.34), x=(0.71, 0.63), curve=(0.07, 0.03)),
                 dict(y=(-2.08, -1.80), x=(0.62, 0.58), curve=(0.04, 0.02), mat="privacy")],
        end_windows=[], smashed=[], dual=False, lift=0.0, rim="multi", spokes=10, spoke_pair=(0.0,), spoke_w=(0.012, 0.014),
        cabin=dict(rear=-1.85, dash=1.14, rows=[(0.1, (-0.38, 0.38), 0.5, 0.64), (-0.9, (-0.43, 0.0, 0.43), 0.43, 0.6)],
                   shelf=(-2.05, -1.35, 1.12), liner=(-1.7, 0.25)),
        doors=[(1.15, -0.07), (-0.07, -1.0)], handles=(0.22, -0.74), handle_z=1.02, mirror=(1.02, 1.17),
        fuel=(-1.6, 0.98), spoiler=True, rear_wiper=True, roof_rails=True, fin=-1.3, plates=(0.57, 0.74),
        fascia=(0.36, 0.96, 0.38, 1.12), wrap=0.9,
    )))


def minibus_spec(**v):
    """Ford Transit L4 H3 17-seat minibus: L 6704, W 2111, H 2780, WB 3750; 235/65 R16C, twin rear wheels."""
    L, fo, wb = 6.704, 1.0, 3.75
    yF = L / 2
    af = yF - fo
    ar = af - wb
    return Spec(**_vary(v, dict(
        kind="transit", tyre_size="235/65 R16C 115R", L=L, sections=48, axles=(ar, af), wheel_r=0.356, rim_r=0.203, tyre_w=0.215, track=(0.92, 0.87),
        arch=0.40, flare=0.012, rocker=0.09, clad=True, cap_r=(0.09, 0.07), face_bulge=(0.008, 0.004),
        front_shape=(0.2, 0.12, 0.7, 0.8), rear_shape=(0.03, 0.0, 3.0, 0.3),
        roof=[(yF, 1.10), (3.25, 1.135), (2.9, 1.235), (2.5, 1.33), (2.36, 1.375), (2.25, 1.47), (2.05, 1.8), (1.86, 2.08),
              (1.74, 2.27), (1.56, 2.55), (1.38, 2.72), (1.15, 2.78), (-3.2, 2.78), (-yF, 2.765)],
        rail=[(yF, 1.05), (3.25, 1.085), (2.9, 1.185), (2.5, 1.28), (2.36, 1.325), (2.25, 1.42), (2.05, 1.75), (1.86, 2.03),
              (1.72, 2.13), (1.5, 2.19), (1.3, 2.22), (-yF, 2.22)],
        belt=[(yF, 1.11), (2.9, 1.24), (2.4, 1.355), (1.9, 1.37), (1.4, 1.36), (-yF, 1.36)],
        shoulder=[(yF, 0.99), (2.8, 1.10), (2.3, 1.17), (0.0, 1.175), (-yF, 1.18)],
        sill=[(yF, 0.43), (3.0, 0.41), (2.7, 0.405), (-3.0, 0.42), (-yF, 0.47)],
        halfw=[(yF, 0.92), (3.25, 0.985), (3.0, 1.03), (2.5, 1.05), (1.5, 1.0555), (-3.2, 1.0555), (-yF, 1.035)],
        roofw=[(2.4, 0.93), (1.86, 0.95), (1.4, 1.0), (-yF, 1.0)],
        rail_drop=0.6, edge=(0.035, 0.06), crown=0.012, tumble_bulge=0.004, liner_depth=0.48,
        windows=[dict(bot=(1.55, 2.28), top=(1.55, 1.95), name="cab"),
                 dict(bot=(0.18, 1.40), top=(0.18, 1.40), mat="privacy", res=0.16),
                 dict(bot=(-1.22, 0.06), top=(-1.22, 0.06), mat="privacy", res=0.16),
                 dict(bot=(-2.62, -1.34), top=(-2.62, -1.34), mat="privacy", res=0.16),
                 dict(bot=(-3.18, -2.74), top=(-3.18, -2.74), mat="privacy", res=0.16)],
        screens=[dict(y=(2.34, 1.88), x=(0.90, 0.86), curve=(0.06, 0.02))],
        end_windows=[dict(rect=(0.08, 0.86, 1.48, 2.22), sign=-1), dict(rect=(-0.86, -0.08, 1.48, 2.22), sign=-1)],
        smashed=[], dual=True, lift=0.0, rim="steel", spokes=0, spoke_pair=(0.0,), spoke_w=(0.01, 0.01),
        cabin=dict(rear=-3.15, dash=2.28, wheel_x=0.5, dash_z=1.42, liner=(-3.1, 1.15), lowpoly=True,
                   rows=[(1.55, (-0.62, -0.1, 0.52), 0.46, 0.72), (0.6, (-0.68, -0.22, 0.3, 0.72), 0.44, 0.72),
                         (-0.25, (-0.68, -0.22, 0.3, 0.72), 0.44, 0.72), (-1.1, (-0.68, -0.22, 0.3, 0.72), 0.44, 0.72),
                         (-1.95, (-0.68, -0.22, 0.3, 0.72), 0.44, 0.72), (-2.75, (-0.68, -0.22, 0.3, 0.72), 0.44, 0.72)]),
        doors=[(2.32, 1.5), (1.45, 0.12)], handles=(1.62, 0.24), handle_z=1.22, mirror=(2.22, 1.5),
        fuel=(1.2, 1.05), spoiler=False, rear_wiper=False, roof_rails=False, fin=None, plates=(0.60, 0.78),
        fascia=(0.40, 1.22, 0.44, 2.70), wrap=1.0,
    )))


def scrap_spec():
    s = hatch_spec()
    s.kind = "scrap"
    s.smashed = [0, 3]            # front-left door glass and the right rear door glass
    s.lift = 0.03
    s.fin = None
    s.spoiler = False
    s.dents = [(-1, 0.45, 0.68, 0.16, 0.028), (1, -1.72, 0.74, 0.13, 0.022), (1, 1.75, 0.6, 0.12, 0.018)]
    return s


PLATES = {
    "golf": ["KD19 RVX", "LB69 FHT", "MA20 WDK", "YC70 PNU", "GN21 LKE"],
    "bmw": ["LR70 HNF", "WP19 SXC", "BK21 YTD", "EO69 LUH", "FV20 RBO"],
    "qashqai": ["YF21 OKC", "SK70 BWV", "DA19 MXE", "HN69 CJU", "NJ71 ZPA"],
    "transit": ["WU68 TBN", "CE19 HXK", "PO20 FMV"],
    "scrap": ["R412 KLM"],
}


def _named(base, lod, k):
    return base + (f"_{k + 1}" if k else "") + ("_lod2" if lod else "")


_MULTI = dict(rim="multi", spokes=10, spoke_pair=(0.0,), spoke_w=(0.012, 0.014))
_VSPOKE = dict(rim="vspoke", spokes=5, spoke_pair=(-0.16, 0.16), spoke_w=(0.012, 0.016))
_TWIN5 = dict(rim="twin5", spokes=5, spoke_pair=(-0.12, 0.12), spoke_w=(0.016, 0.011))

# body variants per type (one per number plate): the base car, then models of the same class with
# their own length, cabin height, roof line, glasshouse and wheels
VARIANTS = {
    "golf": [
        {},
        # a supermini: shorter, a touch lower, smaller wheels with multi-spoke alloys
        dict(sx=0.94, hk=0.97, wheel_r=0.305, rim_r=0.19, tyre_w=0.205, **_MULTI),
        # a hot hatch: lower, bigger wheels, V-spokes
        dict(hk=0.94, wheel_r=0.325, rim_r=0.24, **_VSPOKE),
        # a hatch estate: the roof carried back over a longer tail, a third side window, rails
        dict(sx=1.07, roof_rails=True, spoiler=False,
             roof=[(2.142, 0.80), (2.06, 0.822), (1.8, 0.875), (1.5, 0.94), (1.2, 0.988), (1.13, 1.0), (0.8, 1.165), (0.5, 1.32),
                   (0.27, 1.405), (-0.05, 1.448), (-0.6, 1.455), (-1.2, 1.445), (-1.6, 1.43), (-1.85, 1.41), (-1.96, 1.33),
                   (-2.04, 1.12), (-2.09, 1.0), (-2.142, 0.965)],
             windows=[dict(bot=(-0.06, 1.06), top=(-0.06, 0.32)), dict(bot=(-1.3, -0.16), top=(-1.3, -0.16)),
                      dict(bot=(-1.9, -1.4), top=(-1.84, -1.4), mat="privacy")],
             screens=[dict(y=(1.145, 0.285), x=(0.70, 0.62), curve=(0.07, 0.03)),
                      dict(y=(-2.05, -1.93), x=(0.60, 0.58), curve=(0.03, 0.01), mat="privacy")]),
        # a family hatch a size up
        dict(sx=1.035, hk=1.01, **_MULTI),
    ],
    "bmw": [
        {},
        # a touring estate: the roof run back to a near-vertical tailgate, a third window, rails
        dict(roof_rails=True, rear_wiper=True,
             roof=[(2.3545, 0.705), (2.26, 0.73), (2.0, 0.785), (1.5, 0.862), (1.0, 0.928), (0.80, 0.958), (0.45, 1.15), (0.1, 1.335),
                   (-0.12, 1.405), (-0.45, 1.442), (-0.9, 1.44), (-1.5, 1.425), (-1.95, 1.395), (-2.13, 1.3), (-2.24, 1.1),
                   (-2.3, 1.0), (-2.3545, 0.975)],
             windows=[dict(bot=(-0.28, 0.74), top=(-0.28, -0.04)), dict(bot=(-1.45, -0.38), top=(-1.45, -0.38)),
                      dict(bot=(-2.08, -1.55), top=(-2.0, -1.55), mat="privacy")],
             screens=[dict(y=(0.805, -0.105), x=(0.71, 0.62), curve=(0.08, 0.03)),
                      dict(y=(-2.26, -2.1), x=(0.62, 0.6), curve=(0.03, 0.01), mat="privacy")]),
        # a fastback: one long slope from the roof to the boot lid
        dict(roof=[(2.3545, 0.705), (2.26, 0.73), (2.0, 0.785), (1.5, 0.862), (1.0, 0.928), (0.80, 0.958), (0.45, 1.15), (0.1, 1.335),
                   (-0.12, 1.405), (-0.45, 1.44), (-0.9, 1.405), (-1.3, 1.33), (-1.7, 1.19), (-2.0, 1.065), (-2.22, 1.0),
                   (-2.3545, 0.975)],
             windows=[dict(bot=(-0.28, 0.74), top=(-0.28, -0.04)), dict(bot=(-1.45, -0.38), top=(-1.05, -0.38))],
             screens=[dict(y=(0.805, -0.105), x=(0.71, 0.62), curve=(0.08, 0.03)),
                      dict(y=(-2.02, -1.12), x=(0.62, 0.56), curve=(0.05, 0.02))]),
        # an executive saloon: longer, a little taller, multi-spoke wheels
        dict(sx=1.06, hk=1.01, wheel_r=0.338, rim_r=0.24, **_MULTI),
        # a compact saloon
        dict(sx=0.94, hk=0.99, wheel_r=0.318, rim_r=0.216, **_TWIN5),
    ],
    "qashqai": [
        {},
        # a small crossover
        dict(sx=0.93, hk=0.96, wheel_r=0.345, **_VSPOKE),
        # a seven-seat SUV: longer and taller, roof rails
        dict(sx=1.08, hk=1.05, roof_rails=True, wheel_r=0.365, rim_r=0.24, **_TWIN5),
        # a coupe-SUV: the roof sloping away to a short tail, no rails
        dict(roof_rails=False, spoiler=False, wheel_r=0.362, rim_r=0.24, **_VSPOKE,
             roof=[(2.2125, 0.90), (2.12, 0.925), (1.9, 0.965), (1.6, 1.03), (1.3, 1.088), (1.18, 1.115), (0.85, 1.29), (0.52, 1.47),
                   (0.3, 1.57), (0.05, 1.61), (-0.3, 1.618), (-0.9, 1.575), (-1.4, 1.48), (-1.8, 1.33), (-2.0, 1.2), (-2.12, 1.12),
                   (-2.2125, 1.09)],
             windows=[dict(bot=(-0.02, 1.08), top=(-0.02, 0.38)), dict(bot=(-0.94, -0.12), top=(-0.94, -0.12), mat="privacy"),
                      dict(bot=(-1.6, -1.02), top=(-1.3, -1.02), db=0.016, mat="privacy")],
             screens=[dict(y=(1.185, 0.34), x=(0.71, 0.63), curve=(0.07, 0.03)),
                      dict(y=(-2.03, -1.5), x=(0.62, 0.58), curve=(0.05, 0.02), mat="privacy")]),
        # a squarer, taller SUV
        dict(hk=1.04, roof_rails=True, **_MULTI),
    ],
    "transit": [
        {},
        # medium roof
        dict(hk=0.8, z_hk=1.36),
        # medium wheelbase, high roof
        dict(sx=0.9),
    ],
    "scrap": [{}],
}


def car_hatch(col, lod=False, k=0):
    return vehicle(col, _named("car_hatch", lod, k), hatch_spec(**VARIANTS["golf"][k]), lod, plate=k)


def car_saloon(col, lod=False, k=0):
    return vehicle(col, _named("car_saloon", lod, k), saloon_spec(**VARIANTS["bmw"][k]), lod, plate=k)


def car_suv(col, lod=False, k=0):
    return vehicle(col, _named("car_suv", lod, k), suv_spec(**VARIANTS["qashqai"][k]), lod, plate=k)


def minibus(col, lod=False, k=0):
    return vehicle(col, _named("minibus", lod, k), minibus_spec(**VARIANTS["transit"][k]), lod, paint=car_paint("TINT_van_paint", "#e2e2de"), plate=k)


def scrap_car(col, lod=False, k=0):
    return vehicle(col, _named("scrap_car", lod, k), scrap_spec(), lod, scrap=True)


BUILDERS = {}
LODS = {}
# variants: prop.car_hatch, prop.car_hatch_2 .. _5 (each its own body from VARIANTS and its own registration)
for _k, (_f, _kind) in {"prop.car_hatch": (car_hatch, "golf"), "prop.car_saloon": (car_saloon, "bmw"),
                        "prop.car_suv": (car_suv, "qashqai"), "prop.minibus": (minibus, "transit"),
                        "prop.scrap_car": (scrap_car, "scrap")}.items():
    for _i in range(len(PLATES[_kind])):
        _key = _k + (f"_{_i + 1}" if _i else "")
        BUILDERS[_key] = (lambda f, i: (lambda col, M: f(col, False, i)))(_f, _i)
        LODS[_key] = (lambda f, i: (lambda col, M: f(col, True, i)))(_f, _i)
