"""Trees, hedges, scrub and ground plants for training grounds.

Trees are grown, not assembled. Each species first gets its architecture by hand (a tapering,
buttressed stem; the oak's few heavy, tortuous scaffold limbs; the birch's leader with ascending
limbs that arch over; the willow's leaning bole and arching limbs) and a crown made of separate
clumps at the limb ends; a space-colonisation pass then grows the finer branching into those
clumps (so crowns are lobed with sky between the lobes), branch thickness follows the pipe model
(a parent's cross-section carries its children's), and the skeleton is skinned with continuous
tubes (parallel-transport frames, bark UVs in metres). Foliage is spray cards painted in
foliage_paint.py, with normals bent outwards from the crown (so the crown shades as a volume) and
vertex colours carrying crown occlusion (dark inside each clump and towards the core), clump tint
and a yellower top (glTF COLOR_0, multiplied into the base colour). Weeping species hang strands:
birch twigs as hanging cards, willow whips as twisting ribbons of a tiling leaf strip.

Ground plants are built blade by blade and leaf by leaf (strap and grass blades as tapered
ribbons whose vertex colour runs green to straw, bulrush heads as solid cylinders, weed leaves
as folded cards), and woody plants stand on a litter patch. Every prop has a `.lod2`.

Origins on the ground at the stem; +Z up. Sizes are those of mature trees on an English site
(oak 18 m, birch 17 m, willow 14 m, spruce 23 m); hedges are 2.4 m segments along x (they are
laid at about 2.2 m, so neighbours overlap at their rounded ends).
"""
import math
import zlib

import bpy
import numpy as np
from mathutils.kdtree import KDTree

from ..core import materials as MAT
from ..core.nodes import material
from ..core.scene import mesh_object
from . import foliage_paint as FP

BARK_TILE = 0.6   # metres covered by one bark texture repeat
BIRCH_BASE = 1.5  # height covered by the birch butt texture
UP = np.array([0, 0, 1.0])


def _seed(name):
    return zlib.crc32(name.encode()) % 100000


def _lin(hexs):
    return FP.srgb_lin(hexs)


# -- materials ------------------------------------------------------------------------------------

def _image_material(name, arr, extension="CLIP", alpha=True, rough=0.6):
    m, nb = material(name)
    img = MAT.image_from_array(f"{name}_img", arr)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", image=img, extension=extension)
    nb.connect(uv, tex.inputs["Vector"])
    if alpha:
        clip = nb.math("ROUND", tex.outputs["Alpha"])
        nb.output(nb.principled(Base_Color=tex.outputs["Color"], Roughness=rough, Alpha=clip,
                                Subsurface_Weight=0.15, Subsurface_Radius=(0.05, 0.08, 0.02)))
    else:
        nb.output(nb.principled(Base_Color=tex.outputs["Color"], Roughness=rough))
    return m


def foliage_material(kind):
    """Alpha-tested foliage (Leaf_* materials shade as foliage in the viewer): spray atlases per
    species, conifer shoots, the willow strip, grass/reed blades, seed heads, weeds, litter."""
    name = f"Leaf_{kind}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    sd = _seed(name)
    if kind == "conifer":
        return _image_material(name, FP.conifer_atlas(seed=sd))
    if kind == "willowstrand":
        return _image_material(name, FP.strand_strip(sd), extension="REPEAT")
    if kind == "blade":
        return _image_material(name, FP.blade_texture(seed=sd), extension="REPEAT", alpha=False, rough=0.55)
    if kind == "seedheads":
        return _image_material(name, FP.seedhead_atlas(sd))
    if kind == "weeds":
        return _image_material(name, FP.weed_atlas(sd), rough=0.55)
    if kind == "litter":
        return _image_material(name, FP.litter_atlas(sd), rough=0.85)
    return _image_material(name, FP.spray_atlas(kind, seed=sd))


def leaf_face_material(species):
    """Opaque, tiling leaf face (one repeat per 0.5 m of UV): the clipped faces of a hedge."""
    name = f"LeafFace_{species}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    return _image_material(name, FP.leaf_wall(species, seed=_seed(name)), extension="REPEAT", alpha=False, rough=0.7)


def flat_material(name, rough):
    """White base whose colour comes from the vertex colours (canes, bulrush heads, fungus)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=(1.0, 1.0, 1.0, 1.0), Roughness=rough))
    return m


def _data_image(name, arr):
    """A non-colour image (normal or roughness map) from an (H, W, 3|1) array."""
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate([arr, np.ones((h, w, 1))], -1).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def bark(kind="oak"):
    """Bark from painted maps (albedo, normal, roughness), one repeat per BARK_TILE metres of UV."""
    name = f"Bark_{kind}"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    alb, nrm, rough = FP.bark(kind, seed=_seed(name))
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    ext = "EXTEND" if kind == "birchbase" else "REPEAT"
    t_alb = nb.node("ShaderNodeTexImage", image=MAT.image_from_array(f"{name}_albedo", alb), extension=ext)
    t_nrm = nb.node("ShaderNodeTexImage", image=_data_image(f"{name}_normal", nrm), extension=ext)
    t_rgh = nb.node("ShaderNodeTexImage", image=_data_image(f"{name}_rough", rough), extension=ext)
    for t in (t_alb, t_nrm, t_rgh):
        nb.connect(uv, t.inputs["Vector"])
    nm = nb.node("ShaderNodeNormalMap")
    nb.connect(t_nrm.outputs["Color"], nm.inputs["Color"])
    nb.output(nb.principled(Base_Color=t_alb.outputs["Color"], Roughness=t_rgh.outputs["Color"], Normal=nm.outputs[0]))
    return m


# -- a mesh with custom normals and vertex colours -----------------------------------------------

class TM:
    """Shared-vertex mesh: per-vertex normals (custom) and colours, per-loop UVs, a material per face."""

    def __init__(self):
        self.V, self.N, self.C = [], [], []
        self.F, self.UV, self.mi, self.mats = [], [], [], []

    def mat(self, m):
        if m not in self.mats:
            self.mats.append(m)
        return self.mats.index(m)

    def verts(self, P, N, C):
        base = len(self.V)
        self.V.extend(np.asarray(P, float).tolist())
        self.N.extend(np.asarray(N, float).tolist())
        self.C.extend(np.asarray(C, float).tolist())
        return base

    def face(self, idx, uvs, m):
        self.F.append(tuple(idx))
        self.UV.extend(uvs)
        self.mi.append(m)

    def face_out(self, idx, uvs, m, out):
        """A face wound so its geometric normal points along `out`."""
        a, b, c = (np.array(self.V[i]) for i in idx[:3])
        if np.cross(b - a, c - a) @ np.asarray(out) < 0:
            idx, uvs = list(idx)[::-1], list(uvs)[::-1]
        self.face(idx, uvs, m)

    def obj(self, name, col):
        o = mesh_object(name, np.array(self.V, float), self.F, col, materials=self.mats, mat_index=self.mi, uvs=self.UV)
        me = o.data
        me.shade_smooth()
        N = np.array(self.N, float)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
        me.normals_split_custom_set_from_vertices(N.tolist())
        attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        C = np.concatenate([np.clip(np.array(self.C, float), 0, 1), np.ones((len(self.C), 1))], 1)
        attr.data.foreach_set("color", C.astype(np.float32).ravel())
        me.color_attributes.active_color = attr
        return o


def ribbon(tm, P, W, N, C, mat, u=(0.0, 1.0), v=None):
    """A strip through points P with half-width vectors W; v (per point) along it."""
    P, W = np.asarray(P, float), np.asarray(W, float)
    n = len(P)
    if v is None:
        v = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    L, R = P - W, P + W
    pts = np.empty((2 * n, 3))
    pts[0::2], pts[1::2] = L, R
    N2 = np.repeat(np.asarray(N, float), 2, 0)
    C2 = np.repeat(np.asarray(C, float), 2, 0)
    b = tm.verts(pts, N2, C2)
    m = tm.mat(mat)
    for i in range(n - 1):
        a = b + 2 * i
        tm.face((a, a + 1, a + 3, a + 2), [(u[0], v[i]), (u[1], v[i]), (u[1], v[i + 1]), (u[0], v[i + 1])], m)


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# -- growth ---------------------------------------------------------------------------------------

class Crown:
    """A crown envelope: the union of clumps (ellipsoids); also gives crown depth for shading."""

    def __init__(self, centre, radii, clumps, z_min=-1e9, floor=0.38):
        self.c = np.array(centre, float)
        self.r = np.array(radii, float)
        self.clumps = [(np.asarray(c, float), np.asarray(r, float)) for c, r in clumps]
        self.z_min = z_min
        self.floor = floor

    def inside(self, P):
        ok = np.zeros(len(P), bool)
        for c, r in self.clumps:
            ok |= np.linalg.norm((P - c) / r, axis=1) < 1.0
        return ok & (P[:, 2] > self.z_min)

    def sample(self, n, rng):
        lo = np.min([c - r for c, r in self.clumps], 0)
        hi = np.max([c + r for c, r in self.clumps], 0)
        out = []
        while sum(len(o) for o in out) < n:
            P = rng.uniform(lo, hi, (n * 4, 3))
            out.append(P[self.inside(P)])
        return np.concatenate(out)[:n]

    def depth(self, P):
        """0 at the crown centre, ~1 at the envelope surface (and beyond outside)."""
        return np.linalg.norm((np.atleast_2d(P) - self.c) / self.r, axis=1)

    def local(self, P):
        """How deep inside its own clump a point lies: 0 at a clump centre, 1 at its surface."""
        P = np.atleast_2d(P)
        if len(self.clumps) < 2:
            return np.ones(len(P))
        return np.min([np.linalg.norm((P - c) / r, axis=1) for c, r in self.clumps], 0)

    def radial(self, P):
        R = np.atleast_2d(P) - self.c + np.array([0, 0, self.r[2] * 0.25])
        return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)

    def shade(self, P):
        """Occlusion in the crown: dark at the core and inside each clump, darker underneath."""
        P = np.atleast_2d(P)
        g = self.floor + (1 - self.floor) * _smooth((self.depth(P) - 0.15) / 0.85)
        loc = 0.55 + 0.45 * _smooth(self.local(P))
        under = np.clip((P[:, 2] - (self.c[2] - self.r[2])) / (1.2 * self.r[2]), 0, 1)
        return g * loc * (0.74 + 0.26 * under)

    def sun(self, P, occ):
        """Tint for the sunlit top of the crown: new growth, yellower."""
        P = np.atleast_2d(P)
        k = np.clip((P[:, 2] - self.c[2]) / self.r[2], 0, 1) * np.clip((occ - 0.7) / 0.3, 0, 1)
        return 1 + k[:, None] * np.array([0.1, 0.1, -0.15])


class ConeCrown(Crown):
    """A spruce's narrow cone: depth is the distance from the stem relative to the cone's radius."""

    def __init__(self, base_z, top_z, radius):
        super().__init__((0, 0, (base_z + top_z) / 2), (radius, radius, (top_z - base_z) / 2),
                         [((0, 0, (base_z + top_z) / 2), (radius, radius, (top_z - base_z) / 2))])
        self.z0, self.z1, self.R = base_z, top_z, radius

    def rad_at(self, z):
        return np.maximum(self.R * np.clip((self.z1 - z) / (self.z1 - self.z0), 0, 1) ** 0.95, 0.25)

    def radial(self, P):
        P = np.atleast_2d(P)
        R = np.stack([P[:, 0], P[:, 1], np.full(len(P), 0.0)], 1)
        R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9) + UP * 0.35
        return R / np.linalg.norm(R, axis=1, keepdims=True)

    def shade(self, P):
        P = np.atleast_2d(P)
        rho = np.hypot(P[:, 0], P[:, 1]) / self.rad_at(P[:, 2])
        occ = 0.34 + 0.66 * _smooth((rho - 0.1) / 0.85)
        lo = 0.78 + 0.22 * np.clip((P[:, 2] - self.z0) / (0.35 * (self.z1 - self.z0)), 0, 1)
        return occ * lo

    def sun(self, P, occ):
        P = np.atleast_2d(P)
        k = np.clip((occ - 0.75) / 0.25, 0, 1)
        return 1 + k[:, None] * np.array([0.1, 0.14, -0.12])


def clumps_at(points, sizes, flat=0.7):
    return [(np.asarray(p, float), np.array([s, s, s * flat])) for p, s in zip(points, sizes)]


class Skeleton:
    def __init__(self):
        self.P = []     # positions
        self.par = []   # parent index (-1 for the root)

    def add(self, p, parent):
        self.P.append(np.array(p, float))
        self.par.append(parent)
        return len(self.P) - 1

    def drop(self, roots):
        """Remove the subtrees under `roots` (before finish)."""
        roots = set(roots)
        keep, new = [], {}
        gone = np.zeros(len(self.P), bool)
        for i in range(len(self.P)):
            if i in roots or (self.par[i] >= 0 and gone[self.par[i]]):
                gone[i] = True
                continue
            new[i] = len(keep)
            keep.append(i)
        self.P = [self.P[i] for i in keep]
        self.par = [new[self.par[i]] if self.par[i] >= 0 else -1 for i in keep]
        return new

    def finish(self, exponent=2.0, r_base=0.4, smooth=3, trunk=None, taper=0.0):
        n = len(self.P)
        self.P = np.array(self.P)
        self.par = np.array(self.par)
        self.kids = [[] for _ in range(n)]
        for i in range(1, n):
            self.kids[self.par[i]].append(i)
        # pipe model: a parent carries its children's cross-sections (children come after parents)
        tips = np.zeros(n)
        for i in range(n - 1, -1, -1):
            tips[i] = 1.0 if not self.kids[i] else sum(tips[k] for k in self.kids[i])
        self.tips = tips
        self.r = r_base * (tips / tips[0]) ** (1 / exponent)
        if trunk:
            # the bole tapers from the ground to the crown break
            z0, z1 = self.P[trunk[0]][2], self.P[trunk[-1]][2]
            for i in trunk:
                self.r[i] *= 1 + taper * (1 - (self.P[i][2] - z0) / (z1 - z0 + 1e-9))
        # straighten the zigzag of the growth steps along each limb
        for _ in range(smooth):
            Q = self.P.copy()
            for i in range(1, n):
                if len(self.kids[i]) >= 1 and self.r[i] < r_base * 0.5:
                    m = max(self.kids[i], key=lambda k: self.r[k])
                    Q[i] = 0.5 * self.P[i] + 0.25 * (self.P[self.par[i]] + self.P[m])
            self.P = Q
        return self

    def chains(self):
        """Limbs as node lists: each follows its thickest child; side branches start new chains."""
        out = []
        starts = [0]
        while starts:
            s = starts.pop()
            chain = [s]
            i = s
            while self.kids[i]:
                ks = sorted(self.kids[i], key=lambda k: -self.r[k])
                starts.extend(ks[1:])
                i = ks[0]
                chain.append(i)
            out.append(chain)
        return out


def colonize(sk, attractors, step, influence, kill, rng, iters=200, tropism=(0, 0, 0), jitter=0.12):
    """Grow `sk` towards the attractor cloud (Runions et al. 2007)."""
    A = np.asarray(attractors, float)
    alive = np.ones(len(A), bool)
    trop = np.array(tropism, float)
    for it in range(iters):
        kd = KDTree(len(sk.P))
        for i, p in enumerate(sk.P):
            kd.insert(p, i)
        kd.balance()
        pull = {}
        for ai in np.flatnonzero(alive):
            co, ni, d = kd.find(A[ai])
            if d < kill:
                alive[ai] = False
            elif d < influence:
                pull.setdefault(ni, []).append(ai)
        if not pull:
            break
        for ni, al in pull.items():
            p = sk.P[ni]
            v = A[al] - p
            v = (v / np.linalg.norm(v, axis=1, keepdims=True)).sum(0)
            ln = np.linalg.norm(v)
            if ln < 1e-6:
                v = rng.normal(0, 1, 3)
                ln = np.linalg.norm(v)
            d = v / ln + trop + rng.normal(0, jitter, 3)
            sk.add(p + d / np.linalg.norm(d) * step, ni)
    return sk


def stem(sk, height, rng, lean=0.04, wander=0.05, step=0.45, base=-0.3, kink=None):
    """The bole: rings close together at the foot (for the root flare), a lean and a slow wander.
    Returns the node indices bottom to top."""
    lx, ly = rng.normal(0, lean, 2)
    zs = [base, 0.0, 0.1, 0.22, 0.38, 0.6, 0.9, 1.25]
    while zs[-1] < height - 1e-6:
        zs.append(min(zs[-1] + step, height))
    off = np.zeros(2)
    ids = []
    prev = -1
    for j, z in enumerate(zs):
        if j > 0:
            off = off + rng.normal(0, wander, 2) * (zs[j] - zs[j - 1])
            if kink is not None and zs[j - 1] < kink <= z:
                off = off + rng.normal(0, 0.12, 2)
        zz = max(z, 0.0)
        prev = sk.add((lx * zz + off[0], ly * zz + off[1], z), prev)
        ids.append(prev)
    return ids


def limb(sk, parent, az, elev0, elev1, length, rng, step=0.4, tort=0.12, kinks=1, sag=0.0):
    """A scaffold limb from `parent`: heading `az`, its elevation running from elev0 to elev1 (radians
    above horizontal), wandering (tortuous) with a few sharp kinks; `sag` bows it down mid-length."""
    p = np.array(sk.P[parent], float)
    n = max(2, int(round(length / step)))
    kink_at = set(rng.choice(np.arange(2, n), size=min(kinks, n - 2), replace=False).tolist()) if n > 3 else set()
    ids = []
    idx = parent
    for j in range(n):
        t = j / max(n - 1, 1)
        el = elev0 + (elev1 - elev0) * t - sag * math.sin(math.pi * t)
        az += rng.normal(0, tort)
        if j in kink_at:
            az += rng.choice([-1, 1]) * rng.uniform(0.3, 0.6)
            el += rng.uniform(-0.25, 0.25)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        p = p + d * step
        idx = sk.add(p, idx)
        ids.append(idx)
    return ids


def twig_fans(sk, crown, rng, n, length):
    """Fine twigs the growth pass is too coarse to make: a few short shoots fanning out (and
    towards the light) from every growing tip and from some nodes behind it."""
    if not n:
        return
    kids = np.zeros(len(sk.P), int)
    for p in sk.par[1:]:
        kids[p] += 1
    tips = [i for i in range(1, len(sk.P)) if kids[i] == 0]
    for i in tips:
        chain = [i]
        for _ in range(2):
            if sk.par[chain[-1]] > 0:
                chain.append(sk.par[chain[-1]])
        for j, node in enumerate(chain):
            p = sk.P[node]
            d = p - sk.P[sk.par[node]]
            d /= np.linalg.norm(d) + 1e-9
            out = p - crown.c
            out /= np.linalg.norm(out) + 1e-9
            for k in range(max(1, n - j)):
                v = d * 0.6 + out * 0.5 + rng.normal(0, 0.55, 3) + UP * 0.25
                v /= np.linalg.norm(v)
                a = sk.add(p + v * length * rng.uniform(0.6, 1.1), node)
                if rng.uniform() < 0.5:
                    v2 = v + rng.normal(0, 0.4, 3)
                    sk.add(sk.P[a] + v2 / np.linalg.norm(v2) * length * 0.6, a)


# -- skinning -------------------------------------------------------------------------------------

def _frames(P):
    """Tangents and parallel-transported normals along a polyline."""
    n = len(P)
    T = np.zeros((n, 3))
    for i in range(n):
        a, b = P[max(i - 1, 0)], P[min(i + 1, n - 1)]
        t = b - a
        T[i] = t / (np.linalg.norm(t) + 1e-12)
    Nn = np.zeros((n, 3))
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    v = np.cross(T[0], ref)
    Nn[0] = v / np.linalg.norm(v)
    for i in range(1, n):
        v = Nn[i - 1] - T[i] * (Nn[i - 1] @ T[i])
        ln = np.linalg.norm(v)
        Nn[i] = v / ln if ln > 1e-9 else Nn[i - 1]
    return T, Nn


def tube(tm, P, R, mat, crown, sides, flare=None, cap="cone", ao_floor=0.0, tint=None, seg_mat=None, base=None,
         rng=None, cap_mat=None, colour=None):
    """A tube through points P with radii R. flare(z, theta) scales the radius near the ground;
    tint (n, 3) multiplies the ring colours; seg_mat(i, k) picks a material per face; base =
    (material, height) maps the faces below `height` onto a butt texture that spans it once.
    cap: "cone" (a short point), "point" (a long taper, dead twigs), "jagged" (snapped), None."""
    P = np.asarray(P, float)
    T, Nn = _frames(P)
    B = np.cross(T, Nn)
    circ = 2 * math.pi * R[0]
    tiles = max(1, round(circ / BARK_TILE))
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    m = tm.mat(mat)
    rings = []
    for i in range(len(P)):
        th = np.linspace(0, 2 * math.pi, sides + 1)
        r = np.full(sides + 1, R[i])
        if flare is not None:
            r = r * flare(P[i][2], th)
        dirs = np.cos(th)[:, None] * Nn[i] + np.sin(th)[:, None] * B[i]
        pts = P[i] + dirs * r[:, None]
        if colour is not None:
            C = np.repeat(np.asarray(colour[i], float)[None, :], sides + 1, 0)
        else:
            occ = crown.shade(pts) if crown is not None else np.ones(len(pts))
            occ = np.maximum(occ, ao_floor)
            C = np.repeat(occ[:, None], 3, 1)
        if tint is not None:
            C = C * np.asarray(tint[i], float)[None, :]
        rings.append(tm.verts(pts, dirs, C))
    bm = tm.mat(base[0]) if base is not None else None
    for i in range(len(P) - 1):
        a, b = rings[i], rings[i + 1]
        for k in range(sides):
            u0, u1 = k / sides * tiles, (k + 1) / sides * tiles
            if base is not None and P[i + 1][2] <= base[1] + 1e-6:
                v0, v1 = max(P[i][2], 0) / base[1], max(P[i + 1][2], 0) / base[1]
                tm.face((a + k, a + k + 1, b + k + 1, b + k), [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], bm)
                continue
            v0, v1 = seg[i] / BARK_TILE, seg[i + 1] / BARK_TILE
            mm = tm.mat(seg_mat(i, k)) if seg_mat is not None else m
            tm.face((a + k, a + k + 1, b + k + 1, b + k), [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], mm)
    if cap:
        a = rings[-1]
        c_last = tm.C[a]
        if cap == "jagged":
            jagged_cap(tm, P[-1], T[-1], Nn[-1], B[-1], R[-1], sides, cap_mat or mat, c_last, rng)
            return
        tip = P[-1] + T[-1] * R[-1] * (1.5 if cap == "cone" else 3.5)
        t = tm.verts([tip], [T[-1]], [c_last])
        cm = tm.mat(cap_mat) if cap_mat is not None else m
        for k in range(sides):
            tm.face((a + k, a + k + 1, t), [(k / sides, 0), ((k + 1) / sides, 0), ((k + 0.5) / sides, 0.1)], cm)


def jagged_cap(tm, c, T, N, B, R, sides, mat, col, rng):
    """A snapped end: splinters of bare wood standing up round a sunken, broken centre."""
    rng = rng or np.random.default_rng(0)
    m = tm.mat(mat)
    n = max(sides, 6)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    dirs = np.cos(th)[:, None] * N + np.sin(th)[:, None] * B
    h = rng.uniform(0.0, 1.0, n) ** 2 * R * 2.2
    h[::2] *= 0.3
    rim = c + dirs * R * 0.97 + T * h[:, None]
    ctr = c + T * R * 0.15
    col = np.asarray(col, float)
    i0 = tm.verts(rim, (dirs * 0.4 + T).tolist(), [col * 1.05] * n)
    ic = tm.verts([ctr], [T], [col * 0.8])
    for k in range(n):
        u = 0.5 + 0.45 * np.cos(th[[k, (k + 1) % n]])
        v = 0.5 + 0.45 * np.sin(th[[k, (k + 1) % n]])
        tm.face((i0 + k, i0 + (k + 1) % n, ic), [(u[0], v[0]), (u[1], v[1]), (0.5, 0.5)], m)


def skin(tm, sk, mat, crown, min_r=0.012, flare=0.0, sides_for=None, ao_floor=0.0, tint_fn=None, seg_mat_fn=None,
         base=None, cap_for=None, decimate=0.05, rng=None, collar=0.0, taper_tips=False):
    """Tubes for every limb thicker than min_r (thinner twigs are under the leaf sprays). Thin
    limbs keep every other node (they are nearly straight between growth steps)."""
    sides_for = sides_for or (lambda r: 16 if r > 0.25 else 11 if r > 0.1 else 7 if r > 0.05 else 5 if r > 0.025 else 4)
    rng = rng or np.random.default_rng(len(sk.P))
    for ci, chain in enumerate(sk.chains()):
        nodes = [n for n in chain if sk.r[n] >= min_r]
        if len(nodes) < 1:
            continue
        nodes = [n for j, n in enumerate(nodes) if sk.r[n] > decimate or j % 2 == 0 or j == len(nodes) - 1]
        p = sk.par[nodes[0]]
        idx = ([p] if p >= 0 else []) + nodes
        pts = [sk.P[n] for n in idx]
        rad = [sk.r[nodes[0]]] * (1 if p >= 0 else 0) + [sk.r[n] for n in nodes]
        if len(pts) < 2:
            continue
        if collar and p >= 0:
            # a branch collar: the child swells where it leaves the parent and starts inside it
            d0 = pts[1] - pts[0]
            ln = np.linalg.norm(d0)
            if ln > 1e-6:
                d0 = d0 / ln
                rc = sk.r[nodes[0]]
                along = min(max(collar * 0.6, sk.r[p] * 1.2), ln * 0.6)
                pts = [pts[0] + d0 * min(sk.r[p] * 0.3, ln * 0.2), pts[0] + d0 * along] + pts[1:]
                rad = [min(sk.r[p] * 0.85, rc * 1.9), rc * (1 + collar)] + rad[1:]
                idx = [idx[0], idx[0]] + idx[1:]
        if taper_tips and nodes[-1] == chain[-1] and len(pts) > 2:
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))]
            t = seg / seg[-1]
            rad = [r * (1 - 0.8 * np.clip((tt - 0.7) / 0.3, 0, 1)) for r, tt in zip(rad, t)]
        fl = None
        if flare and nodes[0] == 0:
            ph = rng.uniform(0, 2 * math.pi)
            amp = rng.uniform(0.5, 1.0, 6)

            def fl(z, th, ph=ph, amp=amp):
                k = flare * math.exp(-max(z, 0.0) / 0.55)
                lobes = sum(amp[j] * np.clip(np.cos(th * 1 - (2 * math.pi * j / 5 + ph + 0.3 * amp[j])), 0, 1) ** 6
                            for j in range(5))
                return 1 + k * (0.35 + 0.9 * lobes) + 0.04 * np.cos(3 * th + ph)
        tip_node = chain[-1] if nodes[-1] == chain[-1] else None
        cap = cap_for(tip_node, ci) if cap_for else ("cone" if tip_node is not None else "cone")
        tint = tint_fn(idx) if tint_fn else None
        sm = (lambda i, k, idx=idx, ci=ci: seg_mat_fn(idx, i, k, ci)) if seg_mat_fn else None
        tube(tm, pts, np.array(rad), mat, crown, sides_for(rad[0]), flare=fl, ao_floor=ao_floor if nodes[0] == 0 else 0.0,
             tint=tint, seg_mat=sm, base=base if nodes[0] == 0 else None, cap=cap, rng=rng,
             cap_mat=seg_mat_fn(idx, len(idx) - 2, 0, ci) if seg_mat_fn and cap == "jagged" else None)


# -- foliage cards --------------------------------------------------------------------------------

def quarter(rng, p=None):
    k = rng.choice(4, p=p) if p is not None else rng.integers(4)
    u0, v0 = (k % 2) * 0.5, (k // 2) * 0.5
    return (u0, v0, u0 + 0.5, v0 + 0.5)


def card(tm, A, D, W, size, mat, crown, uv, tint, bend=0.25, aspect=1.0, normal_bias=0.75, out=None, shade=True):
    """A card from A along D (unit) with width axis W (unit, perpendicular), in two segments (the
    outer half bending by `bend` radians about W). uv = (u0, v0, u1, v1): v0 at A. Normals lean
    outwards from the crown (or along `out`)."""
    m = tm.mat(mat)
    h = size
    w = size * aspect
    n0 = np.cross(W, D)
    D2 = D * math.cos(bend) - n0 * math.sin(bend)
    P = np.array([A - W * w / 2, A + W * w / 2,
                  A + D * h * 0.5 + W * w / 2, A + D * h * 0.5 - W * w / 2,
                  A + D * h * 0.5 + D2 * h * 0.5 + W * w / 2, A + D * h * 0.5 + D2 * h * 0.5 - W * w / 2])
    if out is not None:
        radial = np.repeat(np.asarray(out, float)[None, :], len(P), 0)
    else:
        radial = crown.radial(P)
    face_n = n0 if (n0 @ radial.mean(0)) > 0 else -n0
    N = normal_bias * radial + (1 - normal_bias) * face_n
    occ = crown.shade(P) if shade else np.ones(len(P))
    C = occ[:, None] * np.asarray(tint, float)[None, :] * crown.sun(P, occ)
    i = tm.verts(P, N, C)
    u0, v0, u1, v1 = uv
    vm = (v0 + v1) / 2
    us = [(u0, v0), (u1, v0), (u1, vm), (u0, vm), (u1, v1), (u0, v1)]
    tm.face((i, i + 1, i + 2, i + 3), [us[0], us[1], us[2], us[3]], m)
    tm.face((i + 3, i + 2, i + 4, i + 5), [us[3], us[2], us[4], us[5]], m)


def _tint(rng, v=0.1, h=0.05):
    hh = rng.normal(0, h)
    return np.array([1 + hh, 1.0, 1 - 1.5 * hh]) * rng.uniform(1 - v, 1 + v)


def _width_axis(D, rng, roll=0.9):
    """A width axis across D that keeps the card's face mostly up (leaves face the sky)."""
    w = np.cross(UP, D)
    if np.linalg.norm(w) < 1e-6:
        w = np.array([1.0, 0, 0])
    w /= np.linalg.norm(w)
    a = rng.uniform(-roll, roll)
    n = np.cross(D, w)
    w = w * math.cos(a) + n * math.sin(a)
    return w / np.linalg.norm(w)


def _clump_tint(clumps, p, rng, cell=2.5):
    key = tuple(np.floor(p / cell).astype(int))
    if key not in clumps:
        clumps[key] = _tint(rng, 0.1, 0.05)
    return clumps[key]


def twig_ends(sk, twig_tips=3, per_tip=2.0, inner=0.5, rng=None):
    out = []
    for i in range(1, len(sk.P)):
        if sk.tips[i] <= twig_tips:
            n = per_tip if not sk.kids[i] else inner
            cnt = int(n) + (rng.uniform() < n - int(n))
            out.extend([i] * cnt)
    return out


def leaf_sprays(tm, sk, crown, mat, rng, size, per_tip=2.0, inner=0.5, twig_tips=3, droop=0.0, lift=0.15,
                outward=0.5, budget=None, colour=(1.0, 1.0, 1.0), vary=0.07, face_out=False):
    """Spray cards on the twigs: `per_tip` on each growing tip, `inner` on other twig nodes."""
    ends = twig_ends(sk, twig_tips, per_tip, inner, rng)
    if budget and len(ends) > budget:
        ends = [ends[j] for j in sorted(rng.choice(len(ends), budget, replace=False))]
    clumps = {}
    for i in ends:
        p = sk.P[i]
        d = p - sk.P[sk.par[i]]
        d /= np.linalg.norm(d) + 1e-9
        out = p - crown.c
        out /= np.linalg.norm(out) + 1e-9
        base_t = _clump_tint(clumps, p, rng)
        D = d + out * outward + rng.normal(0, 0.35, 3) + UP * (lift - droop)
        D /= np.linalg.norm(D)
        W = _width_axis(D, rng)
        if face_out:
            # the card's face turned to the outside of the crown (never edge-on to a viewer outside)
            r_ = crown.radial(p)[0]
            D = D - r_ * (D @ r_) * 0.85
            D /= np.linalg.norm(D) + 1e-9
            W = np.cross(r_, D)
            W /= np.linalg.norm(W) + 1e-9
        s = size * rng.uniform(0.8, 1.2)
        A = p - D * s * 0.25 + rng.normal(0, size * 0.12, 3)
        card(tm, A, D, W, s, mat, crown, quarter(rng), base_t * _tint(rng, vary, vary * 0.4) * np.asarray(colour),
             bend=rng.uniform(0.1, 0.5) + droop * 0.6)


def hanging_cards(tm, sk, crown, mat, rng, length, aspect, budget, min_depth=0.4, uv_w=(0.08, 0.42), tips=2,
                  spread=(0.75, 1.25), yellow=0.0):
    """Pendulous twigs (birch): cards hanging from twig ends in the outer crown, the twig entering
    at the top of the texture."""
    ends = [i for i in range(1, len(sk.P)) if sk.tips[i] <= tips and crown.depth(sk.P[i])[0] > min_depth]
    if len(ends) > budget:
        ends = [ends[j] for j in sorted(rng.choice(len(ends), budget, replace=False))]
    clumps = {}
    for i in ends:
        p = sk.P[i]
        out = np.array([p[0] - crown.c[0], p[1] - crown.c[1], 0.0])
        out /= np.linalg.norm(out) + 1e-9
        D = -UP + out * rng.uniform(0.1, 0.45) + rng.normal(0, 0.12, 3)
        D /= np.linalg.norm(D)
        yaw = rng.uniform(0, math.pi)
        W = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        W -= D * (W @ D)
        W /= np.linalg.norm(W)
        q = quarter(rng)
        uv = (q[0] + uv_w[0], q[3], q[0] + uv_w[1], q[1])     # top of the quarter at the twig
        s = length * rng.uniform(*spread)
        yt = np.array([1.35, 1.15, 0.55]) if rng.uniform() < yellow else np.ones(3)
        card(tm, p, D, W, s, mat, crown, uv, _clump_tint(clumps, p, rng, 2.0) * _tint(rng, 0.08, 0.04) * yt,
             bend=rng.uniform(-0.15, 0.25), aspect=aspect, normal_bias=0.7)


def lod_cards(tm, sk, crown, mat, rng, size, cell=1.6, per=2, twig_tips=3, hang=False):
    """Coarse foliage: twig nodes binned into cells, a few big crossed cards per occupied cell."""
    bins = {}
    for i in range(1, len(sk.P)):
        if sk.tips[i] <= twig_tips:
            bins.setdefault(tuple(np.floor(sk.P[i] / cell).astype(int)), []).append(sk.P[i])
    for key, pts in bins.items():
        if len(pts) < 2:
            continue
        c = np.mean(pts, 0)
        out = c - crown.c
        out /= np.linalg.norm(out) + 1e-9
        t = _tint(rng, 0.1, 0.04)
        for k in range(per):
            q = quarter(rng)
            if hang:
                D = -UP + rng.normal(0, 0.2, 3)
                D /= np.linalg.norm(D)
                W = _width_axis(D, rng, 3.0)
                card(tm, c + UP * size * 0.4, D, W, size, mat, crown, (q[0] + 0.05, q[3], q[2] - 0.05, q[1]), t, bend=0.1,
                     aspect=0.8)
            else:
                D = out * 0.6 + rng.normal(0, 0.4, 3) + UP * 0.3
                D /= np.linalg.norm(D)
                W = _width_axis(D, rng, 1.3)
                card(tm, c - D * size * 0.45, D, W, size, mat, crown, q, t, bend=0.3)


def litter(tm, centre, rx, ry, q, rng, z=0.012, n=14, shade=0.85):
    """A litter patch on the ground: a ragged disc mapped onto one quarter of the litter atlas."""
    mat = foliage_material("litter")
    m = tm.mat(mat)
    u0, v0 = (q % 2) * 0.5, (q // 2) * 0.5
    c = np.asarray(centre, float)
    rot = rng.uniform(0, 2 * math.pi)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = 1.0 + rng.normal(0, 0.05, n)
    loc = np.stack([np.cos(th) * rx * rr, np.sin(th) * ry * rr], 1)
    rotm = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    xy = loc @ rotm.T
    P = np.c_[c[0] + xy[:, 0], c[1] + xy[:, 1], np.full(n, z)]
    ctr = np.array([c[0], c[1], z + 0.002])
    i = tm.verts(np.vstack([P, ctr]), [UP] * (n + 1), [[shade] * 3] * n + [[shade * 0.9] * 3])
    uvp = [(u0 + 0.25 + 0.245 * math.cos(t) * r, v0 + 0.25 + 0.245 * math.sin(t) * r) for t, r in zip(th, rr)]
    for k in range(n):
        k2 = (k + 1) % n
        tm.face((i + k, i + k2, i + n), [uvp[k], uvp[k2], (u0 + 0.25, v0 + 0.25)], m)


# -- broadleaf species ----------------------------------------------------------------------------

def _scatter_clumps(cands, rng, n, min_gap):
    """Pick up to n clump centres from candidates, keeping them apart (sky between the lobes)."""
    out = []
    for c, s in cands:
        if all(np.linalg.norm((c - o) * np.array([1, 1, 1.4])) > min_gap * (s + so) for o, so in out):
            out.append((c, s))
        if len(out) >= n:
            break
    return out


def _dome_clumps(cands, centre, radii, rng, n_surf, size, n, min_gap, up=0.45):
    """Clump candidates over a dome (upper part of an ellipsoid) as well as the given ones, so the
    crown reads as one broad dome broken into lobes, centred over the stem."""
    c, r = np.asarray(centre, float), np.asarray(radii, float)
    surf = []
    for k in range(n_surf):
        d = rng.normal(0, 1, 3)
        d[2] = abs(d[2]) * (1 - up) + up
        d /= np.linalg.norm(d)
        s = size * rng.uniform(0.85, 1.15)
        surf.append((c + d * (r - s * 0.7), s))
    return _scatter_clumps(list(cands) + surf, rng, n, min_gap)


def oak():
    """Quercus robur, open grown, about 19 m tall and 19 m across: a massive buttressed bole 1.2 m
    across with a 4 m clear stem, then four heavy, sinuous limbs that zig-zag out at 30-55 degrees
    (one low and near horizontal), each forking, and a short leader; the crown is nine or ten
    separate masses at the limb ends with sky and dark limbs showing between them."""
    rng = np.random.default_rng(11)
    sk = Skeleton()
    trunk = stem(sk, 4.0, rng, lean=0.03, wander=0.05, kink=2.6)
    az0 = rng.uniform(0, 2 * math.pi)
    ends = []
    for j in range(4):
        az = az0 + 2 * math.pi * j / 4 + rng.normal(0, 0.25)
        if j == 0:
            ids = limb(sk, trunk[-3], az, math.radians(22), math.radians(2), 8.5, rng, step=0.45, tort=0.16, kinks=3)
        else:
            e0 = math.radians(rng.uniform(38, 55))
            ids = limb(sk, trunk[-1 - (j % 2)], az, e0, e0 - math.radians(rng.uniform(22, 32)), rng.uniform(8.5, 9.8), rng,
                       step=0.45, tort=0.16, kinks=3)
        ends.append(sk.P[ids[-1]])
        for frac in (0.4, 0.65):
            f = limb(sk, ids[int(len(ids) * frac)], az + rng.choice([-1, 1]) * rng.uniform(0.55, 0.9), math.radians(50),
                     math.radians(25), rng.uniform(3.5, 4.8), rng, step=0.45, tort=0.15, kinks=2)
            ends.append(sk.P[f[-1]])
    lead = limb(sk, trunk[-1], az0 + 2.2, math.radians(74), math.radians(55), 7.0, rng, step=0.45, tort=0.1, kinks=2)
    ends.append(sk.P[lead[-1]])
    cands = [(e + UP * 1.5, rng.uniform(3.4, 4.0)) for e in ends]
    cl = _scatter_clumps(cands, rng, 10, 0.62)
    allc = np.array([c for c, _ in cl])
    centre = np.r_[allc[:, :2].mean(0) * 0.5, allc[:, 2].mean()]
    radii = np.maximum(np.abs(allc - centre).max(0) + 3.5, 3.0)
    crown = Crown(centre, radii, clumps_at([c for c, _ in cl], [s for _, s in cl], 0.75), z_min=4.4, floor=0.3)
    colonize(sk, crown.sample(5200, rng), 0.38, 2.2, 0.48, rng, jitter=0.22)
    twig_fans(sk, crown, rng, 2, 0.35)
    sk.finish(exponent=2.3, r_base=0.58, trunk=trunk, taper=0.28)
    return sk, crown, dict(bark="oak", leaf="oak", size=0.7, flare=0.55, per_tip=1.4, inner=0.45, budget=4600,
                           min_r=0.055, trunk=trunk)


def birch():
    """Betula pendula, about 17 m: a white stem with a black fissured butt, clear to 4.5 m; a
    leader running to the top and eight limbs spiralling round it (larger low down) that rise at
    50-60 degrees, fork, then let fine twigs hang in curtains; an ovoid crown about 7 m across."""
    rng = np.random.default_rng(23)
    sk = Skeleton()
    trunk = stem(sk, 4.5, rng, lean=0.03, wander=0.06, step=0.4)
    lead = limb(sk, trunk[-1], rng.uniform(0, 6.3), math.radians(86), math.radians(80), 12.2, rng, step=0.4, tort=0.05,
                kinks=1)
    zl = np.array([sk.P[i][2] for i in lead])
    pts, sz = [], []
    az0 = rng.uniform(0, 2 * math.pi)
    for j in range(8):
        z = 4.7 + 1.0 * j + rng.uniform(-0.15, 0.15)
        node = lead[int(np.argmin(np.abs(zl - z)))] if j else trunk[-1]
        az = az0 + j * 2.4 + rng.normal(0, 0.2)
        e0 = math.radians(rng.uniform(52, 62))
        f_ = j / 7
        L = rng.uniform(4.2, 5.0) * (1.0 - 0.5 * f_ ** 1.3)
        ids = limb(sk, node, az, e0, e0 - math.radians(rng.uniform(18, 28)), L, rng, step=0.35, tort=0.12, kinks=1)
        f = limb(sk, ids[int(len(ids) * 0.55)], az + rng.choice([-1, 1]) * rng.uniform(0.5, 0.8), math.radians(48),
                 math.radians(28), L * rng.uniform(0.4, 0.5), rng, step=0.35, kinks=1)
        pts += [sk.P[ids[-1]], sk.P[f[-1]]]
        sz += [rng.uniform(1.5, 1.9), rng.uniform(1.2, 1.5)]
    pts.append(sk.P[lead[-1]] - UP * 0.4)
    sz.append(1.5)
    c = np.array([0.0, 0.0, 10.8])
    cl = [(c, np.array([3.0, 3.0, 5.6]))] + clumps_at(pts, sz, 1.0)
    crown = Crown(c, (3.6, 3.6, 6.2), cl, z_min=5.2, floor=0.3)
    colonize(sk, crown.sample(2600, rng), 0.3, 1.5, 0.36, rng, tropism=(0, 0, -0.12), jitter=0.16)
    twig_fans(sk, crown, rng, 1, 0.3)
    sk.finish(exponent=2.2, r_base=0.18, trunk=trunk, taper=0.35)
    return sk, crown, dict(bark="birch", leaf="birch", flare=0.3, min_r=0.012, trunk=trunk)


def dead():
    """A dead field tree, about 7 m: a gently flared, leaning bole (3-4 buttress lobes in the bottom
    half metre), the leader snapped off, a limb torn back to a stub, another hanging; branches that
    fork and curve with thinning tips, a third of them broken short; bark sloughing off the upper
    wood in patches."""
    rng = np.random.default_rng(41)
    sk = Skeleton()
    trunk = stem(sk, 2.8, rng, lean=0.07, wander=0.07, kink=1.6)
    az0 = rng.uniform(0, 2 * math.pi)
    lead = limb(sk, trunk[-1], az0, math.radians(78), math.radians(70), 4.4, rng, tort=0.1, kinks=1)
    l1 = limb(sk, trunk[-1], az0 + 1.9, math.radians(48), math.radians(25), 4.4, rng, kinks=2)
    l2 = limb(sk, trunk[-2], az0 + 3.5, math.radians(30), math.radians(-15), 3.8, rng, kinks=1, sag=0.08)
    stub = limb(sk, trunk[-3], az0 + 5.0, math.radians(35), math.radians(30), 0.55, rng, step=0.27, kinks=0)
    l4 = limb(sk, lead[4], az0 - 1.2, math.radians(50), math.radians(30), 3.4, rng, kinks=1)
    snapped = {lead[-1], stub[-1]}
    pts = [sk.P[l1[-1]] + UP * 0.6, sk.P[l2[-1]] + UP * 0.4, sk.P[l4[-1]] + UP * 0.7, sk.P[lead[5]] + UP * 1.2,
           sk.P[l1[len(l1) // 2]] + UP * 1.2]
    crown = Crown(np.mean(pts, 0), (5.5, 5.5, 3.5), clumps_at(pts, [2.4, 2.0, 2.2, 1.6, 1.6], 0.75), z_min=3.0)
    A = crown.sample(1100, rng)
    A = A[np.linalg.norm(A - sk.P[lead[-1]], axis=1) > 1.6]
    colonize(sk, A, 0.26, 2.0, 0.5, rng, jitter=0.32)
    # the snapped ends grow nothing; the finest twigs are the first to go
    kids = {}
    for i in range(1, len(sk.P)):
        kids.setdefault(sk.par[i], []).append(i)
    drop = [k for s in snapped for k in kids.get(s, [])]
    has_kids = set(sk.par[1:])
    leaves = [i for i in range(len(sk.P)) if i not in has_kids and i not in snapped]
    drop += [i for i in leaves if rng.uniform() < 0.3]
    new = sk.drop(drop)
    trunk = [new[i] for i in trunk]
    snapped = {new[i] for i in snapped}
    sk.finish(exponent=2.3, r_base=0.25, trunk=trunk, taper=0.18)
    # a third of the remaining tips are broken off short
    tips = [i for i in range(1, len(sk.P)) if not sk.kids[i]]
    broken = {i for i in tips if rng.uniform() < 0.4}
    return sk, crown, dict(bark="dead", leaf=None, flare=0.28, min_r=0.012, trunk=trunk, snapped=snapped, broken=broken)


SPECIES = dict(oak=oak, birch=birch, dead=dead)


def _crotch_tint(sk, trunk, damp=True):
    """Darker in the forks (wet, shaded) and greener at the damp foot of the bole."""
    def fn(idx):
        out = []
        for n in idx:
            k = 1.0
            kids = sk.kids[n]
            if len(kids) >= 2 and sorted(sk.r[kids])[-2] > 0.35 * sk.r[n]:
                k = 0.62
            elif sk.par[n] >= 0 and len(sk.kids[sk.par[n]]) >= 2 and sk.r[n] > 0.05:
                k = 0.8
            t = np.array([k, k, k])
            z = sk.P[n][2]
            if damp and z < 0.8:
                t = t * (np.array([0.78, 0.86, 0.66]) + np.array([0.22, 0.14, 0.34]) * max(z, 0) / 0.8)
            out.append(t)
        return out
    return fn


def tree(col, kind, lod=False):
    sk, crown, o = SPECIES[kind]()
    rng = np.random.default_rng(len(sk.P) + (7 if lod else 0))
    name = f"tree_{kind}" + ("_lod2" if lod else "")
    bark_m = bark(o["bark"])
    tm = TM()
    tint_fn = _crotch_tint(sk, o["trunk"])
    seg_mat_fn, cap_for, base = None, None, None
    if kind == "birch":
        base = (bark("birchbase"), BIRCH_BASE)
        # thin birch wood is dark purplish brown, only the stem and limbs turn white
        def tint_fn(idx, f=tint_fn):
            return [t * (np.array([0.42, 0.36, 0.34]) + np.array([0.58, 0.64, 0.66]) * float(np.clip((sk.r[n] - 0.012) / 0.03, 0, 1)))
                    for t, n in zip(f(idx), idx)]
    if kind == "dead":
        dw = bark("deadwood")
        snapped = o["snapped"]
        ph = rng.uniform(0, 6.3, 3)

        def seg_mat_fn(idx, i, k, ci):
            n = idx[min(i + 1, len(idx) - 1)]
            z = sk.P[n][2]
            r = sk.r[n]
            if r < 0.03:
                return dw if (ci * 7 + i) % 5 < 3 else bark_m
            near_snap = any(np.linalg.norm(sk.P[n] - sk.P[s]) < 0.9 for s in snapped)
            th = 2 * math.pi * k / 7
            patch = math.sin(z * 1.7 + ph[0] + ci) * math.sin(th * 2 + z * 0.9 + ph[1]) + 0.4 * math.sin(z * 4.1 + ph[2])
            return dw if (near_snap or patch > (0.6 if z > 3.5 else 0.95)) else bark_m

        broken = o["broken"]

        def cap_for(tip, ci):
            return "jagged" if (tip in snapped or tip in broken) else "point"
    if lod:
        skin(tm, sk, bark_m, crown, min_r=0.05, flare=o["flare"], sides_for=lambda r: 8 if r > 0.15 else 5, ao_floor=0.8,
             tint_fn=tint_fn, seg_mat_fn=seg_mat_fn, base=base, cap_for=cap_for, decimate=0.2, rng=rng)
    else:
        skin(tm, sk, bark_m, crown, min_r=o["min_r"], flare=o["flare"], ao_floor=0.8, tint_fn=tint_fn,
             seg_mat_fn=seg_mat_fn, base=base, cap_for=cap_for, rng=rng, collar=0.3, taper_tips=kind == "dead")
    if kind == "oak":
        leaf_m = foliage_material("oak")
        if lod:
            lod_cards(tm, sk, crown, leaf_m, rng, 2.4, cell=1.8)
        else:
            leaf_sprays(tm, sk, crown, leaf_m, rng, o["size"], o["per_tip"], o["inner"], budget=o["budget"],
                        colour=(0.84, 0.8, 0.62))
    elif kind == "birch":
        leaf_m = foliage_material("birch")
        if lod:
            lod_cards(tm, sk, crown, leaf_m, rng, 2.2, cell=1.5, hang=True)
        else:
            hanging_cards(tm, sk, crown, leaf_m, rng, 0.8, 0.6, 3400, min_depth=0.0, spread=(0.4, 1.5), yellow=0.07)
    elif kind == "dead":
        if not lod:
            _brackets(tm, sk, o["trunk"], rng)
            _fallen(tm, rng, bark_m, bark("deadwood"))
    return tm.obj(name, col)


def _brackets(tm, sk, trunk, rng):
    """Bracket fungi on the bole: semicircular shelves, brown and zoned above, cream beneath."""
    m = flat_material("Fungus_Bracket", 0.8)
    zs = [0.9, 1.25, 1.45]
    az = rng.uniform(0, 6.3)
    for j, z in enumerate(zs):
        i = trunk[int(np.argmin([abs(sk.P[n][2] - z) for n in trunk]))]
        c0 = sk.P[i]
        a = az + j * 0.5 + rng.normal(0, 0.2)
        out = np.array([math.cos(a), math.sin(a), 0.0])
        side = np.array([-out[1], out[0], 0.0])
        c = c0 + out * sk.r[i] * 1.12
        R = rng.uniform(0.14, 0.22)
        th = np.linspace(-math.pi / 2, math.pi / 2, 9)
        rim = c + (np.cos(th)[:, None] * out * R + np.sin(th)[:, None] * side * R * 1.25)
        rim[:, 2] -= 0.015
        top = c + UP * 0.02 - out * 0.01
        bot = c - UP * 0.03 - out * 0.01
        cols_rim = [np.array([0.55, 0.42, 0.2])] * 9
        it = tm.verts(np.vstack([rim, [top]]), [out * 0.4 + UP] * 9 + [UP], cols_rim + [np.array([0.22, 0.13, 0.06])])
        ib = tm.verts(np.vstack([rim, [bot]]), [-UP + out * 0.3] * 10, [np.array([0.6, 0.55, 0.45])] * 10)
        mi = tm.mat(m)
        for k in range(8):
            tm.face((it + k + 1, it + k, it + 9), [(0, 0), (1, 0), (0.5, 1)], mi)
            tm.face((ib + k, ib + k + 1, ib + 9), [(0, 0), (1, 0), (0.5, 1)], mi)


def _fallen(tm, rng, bark_m, dw):
    """A few dropped limbs lying in the grass at the foot."""
    for j in range(3):
        a = rng.uniform(0, 6.3)
        d = rng.uniform(1.2, 2.6)
        p0 = np.array([math.cos(a) * d, math.sin(a) * d, 0.0])
        h = a + rng.normal(0, 1.0)
        L = rng.uniform(1.2, 2.4)
        r0 = rng.uniform(0.04, 0.08)
        dirv = np.array([math.cos(h), math.sin(h), 0.0])
        pts = [p0 + dirv * L * t + np.array([0, 0, r0 * (1 - 0.5 * t) * 0.8]) + np.array([-dirv[1], dirv[0], 0]) * 0.08 * math.sin(3 * t + j)
               for t in np.linspace(0, 1, 5)]
        R = np.linspace(r0, r0 * 0.45, 5)
        tube(tm, pts, R, dw if j % 2 else bark_m, None, 7, cap="jagged", rng=rng, cap_mat=dw,
             colour=[np.array([0.8, 0.8, 0.78])] * 5)
        if L > 1.2:
            q = pts[2]
            br = [q + np.array([math.cos(h + 0.7), math.sin(h + 0.7), 0]) * 0.5 * t + UP * (0.02 + 0.05 * t) for t in np.linspace(0, 1, 3)]
            tube(tm, br, np.linspace(r0 * 0.45, r0 * 0.2, 3), dw, None, 4, cap="point", colour=[np.array([0.8, 0.8, 0.78])] * 3)


# -- weeping willow (built directly: arching limbs with strands hanging off them) ----------------

def _bezier(P0, P1, P2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t * t * P2


def weeping_willow(col, lod=False):
    """Salix x sepulcralis, about 15 m tall and 17 m across: a short massive leaning bole (0.75 m
    across, flared) forking at 2.5 m into five heavy limbs that rise and arch over into a dome,
    each carrying secondary arches further out and down; pendulous branchlets leave the outer
    60% of every arch at 20-40 degrees below horizontal and fall under their weight, the outer
    ones to within a metre of the ground (a few trailing on it), the inner ones shorter and
    darker, leaves hanging down them."""
    rng = np.random.default_rng(37 + (5 if lod else 0))
    rs = np.random.default_rng(37)                      # the skeleton is the same at every LOD
    tm = TM()
    b = bark("willow")
    strand_m = foliage_material("willowstrand")
    crown = Crown((0, 0, 8.0), (8.5, 8.5, 6.0), [((0, 0, 8.0), (8.5, 8.5, 6.0))], floor=0.45)
    lean = np.array([rs.normal(0, 0.06), rs.normal(0, 0.06), 1.0])
    zs = np.array([-0.3, 0.0, 0.12, 0.3, 0.55, 0.9, 1.4, 1.9, 2.4, 2.9, 3.3])
    T = np.stack([lean[0] * np.maximum(zs, 0), lean[1] * np.maximum(zs, 0), zs], 1)
    R = 0.38 - 0.03 * np.clip(zs, 0, 2.4) - 0.06 * np.clip(zs - 2.4, 0, 1)
    ph = rs.uniform(0, 6.3)

    def fl(z, th):
        k = 0.6 * math.exp(-max(z, 0.0) / 0.5)
        return 1 + k * (0.4 + 0.8 * np.clip(np.cos(4 * th + ph), 0, 1) ** 3)
    trunk_c = [np.array([0.8, 0.8, 0.78]) * (0.85 if z < 0.5 else 1.0) for z in zs]
    tube(tm, T, R, b, None, 16 if not lod else 9, flare=fl, cap=None, colour=trunk_c)
    branches = []                                         # (points, radii, strand share)
    az0 = rs.uniform(0, 2 * math.pi)
    n_seg = 14 if not lod else 7
    for j in range(10):
        centre_limb = j >= 7                              # three shorter limbs fill the middle of the dome
        fork = T[[-4, -3, -2, -1, -3, -2, -1, -1, -2, -1][j]]
        az = (az0 + 2 * math.pi * j / 7 if not centre_limb else az0 + 0.6 + 2 * math.pi * (j - 7) / 3) + rs.normal(0, 0.2)
        h = np.array([math.cos(az), math.sin(az), 0.0])
        reach = rs.uniform(6.5, 8.5) if not centre_limb else rs.uniform(2.5, 3.5)
        top = rs.uniform(11.5, 13.5) if not centre_limb else rs.uniform(13.5, 14.5)
        P0 = fork + h * 0.12
        P1 = fork + h * reach * 0.35 + UP * (top - fork[2]) * 1.35
        P2 = fork + h * reach + UP * (top - fork[2] - 3.2)
        pts = _bezier(P0, P1, P2, n_seg)
        pts += np.c_[rs.normal(0, 0.12, (n_seg, 2)), np.zeros(n_seg)] * np.linspace(0, 1, n_seg)[:, None]
        rad = np.linspace(0.26 if not centre_limb else 0.18, 0.045, n_seg) ** 1.0
        branches.append((pts, rad, 1.0))
        # secondary arches alternate side to side, fanning out to close the gaps between limbs
        side = rs.choice([-1, 1])
        for frac in ((0.3, 0.42, 0.54, 0.66, 0.78, 0.88) if not centre_limb else (0.45, 0.65, 0.8)):
            i = int(frac * (n_seg - 1))
            side = -side
            a2 = az + side * rs.uniform(0.55, 1.1)
            h2 = np.array([math.cos(a2), math.sin(a2), 0.0])
            q0 = pts[i]
            L2 = rs.uniform(3.5, 5.0) * (1.15 - 0.4 * frac)
            q1 = q0 + h2 * L2 * 0.45 + UP * rs.uniform(1.0, 1.8)
            q2 = q0 + h2 * L2 + UP * rs.uniform(-1.6, -0.6)
            p2 = _bezier(q0, q1, q2, max(4, n_seg // 2))
            branches.append((p2, np.linspace(rad[i] * 0.55, 0.025, len(p2)), 1.0))
    for pts, rad, _ in branches:
        col_ = [np.array([0.62, 0.6, 0.58]) * (0.7 + 0.3 * float(crown.shade(p)[0])) for p in pts]
        tube(tm, pts, rad, b, None, (10 if rad[0] > 0.15 else 6) if not lod else 5, cap="cone", colour=col_)
    # pendulous branchlets
    spacing = 0.3 if not lod else 0.8
    width = 0.42 if not lod else 0.9
    seg = 0.8 if not lod else 1.6
    az_long = rng.uniform(0, 2 * math.pi)
    for pts, rad, _ in branches:
        L_br = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if not lod:
            # leafy shoots lying along the top of the branch, so no bare wood shows from above
            a0 = int(len(pts) * 0.15)
            q = pts[a0:] + UP * 0.08
            tg = np.gradient(q, axis=0)
            sd = np.cross(tg, UP)
            sd /= np.linalg.norm(sd, axis=1, keepdims=True) + 1e-9
            sv = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            ribbon(tm, q, sd * np.linspace(0.6, 0.35, len(q))[:, None], np.repeat(UP[None, :], len(q), 0),
                   [np.array([0.62, 0.66, 0.5])] * len(q), strand_m, v=sv / 1.68)
        # young shoots springing from the top of the arch, rising half a metre and falling back
        # over it: from eye height they hide the wood and give the dome its soft, tufted outline
        for sl in np.arange(L_br[-1] * 0.2, L_br[-1], 0.38 if not lod else 1.0):
            i = min(np.searchsorted(L_br, sl) - 1, len(pts) - 2)
            p = pts[i] + (pts[i + 1] - pts[i]) * ((sl - L_br[i]) / (L_br[i + 1] - L_br[i] + 1e-9))
            p = p + UP * rad[i] * 0.8
            tng = pts[i + 1] - pts[i]
            th_ = np.array([tng[0], tng[1], 0.0])
            th_ /= np.linalg.norm(th_) + 1e-9
            a = rng.normal(0, 0.9)
            h = th_ * math.cos(a) + np.array([-th_[1], th_[0], 0.0]) * math.sin(a)
            L = rng.uniform(1.3, 2.6)
            rise = rng.uniform(0.35, 0.75)
            s = np.r_[0.0, 0.3, 0.6, 0.9, np.arange(1.4, L, 0.6), L] if not lod else np.r_[0.0, 0.6, L]
            s = np.unique(np.clip(s, 0, L))
            reach = 0.7 * np.sin(np.clip(s / 1.2, 0, 1) * math.pi / 2)
            P = p[None, :] + h[None, :] * reach[:, None]
            P[:, 2] = p[2] + rise * np.sin(np.clip(s / 1.2, 0, 1) * math.pi) - np.maximum(s - 1.2, 0)
            ph_ = rng.uniform(0, math.pi) + s * rng.uniform(0.4, 0.8) * rng.choice([-1, 1])
            hs = np.array([-h[1], h[0], 0.0])
            Wd = hs[None, :] * np.cos(ph_)[:, None] + UP[None, :] * np.sin(ph_)[:, None] * 0.5
            Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
            wid = (0.36 if not lod else 0.8) / 2 * (1 - 0.4 * (s / L) ** 2)
            outv = np.array([p[0], p[1], 0.0]) / (np.hypot(p[0], p[1]) + 1e-9)
            N = np.repeat((outv * 0.4 + UP * 0.9)[None, :], len(s), 0)
            t = _tint(rng, 0.06, 0.03) * np.array([1.04, 1.06, 0.9]) * (0.8 + 0.2 * float(crown.shade(p)[0]))
            ribbon(tm, P, Wd * wid[:, None], N, [t] * len(s), strand_m, v=s / 1.68 + rng.uniform(0, 1))
        for sl in np.arange(L_br[-1] * 0.22, L_br[-1], spacing):
            i = min(np.searchsorted(L_br, sl) - 1, len(pts) - 2)
            f = (sl - L_br[i]) / (L_br[i + 1] - L_br[i] + 1e-9)
            p = pts[i] + (pts[i + 1] - pts[i]) * f
            tng = pts[i + 1] - pts[i]
            th_ = np.array([tng[0], tng[1], 0.0])
            th_ /= np.linalg.norm(th_) + 1e-9
            outv = np.array([p[0], p[1], 0.0])
            rad_xy = np.linalg.norm(outv)
            outv /= rad_xy + 1e-9
            for k in range(2 if not lod else 1):
                side = np.array([-th_[1], th_[0], 0.0]) * (1 if k else -1)
                h = th_ * 0.5 + side * 0.6 + outv * 0.5 + rng.normal(0, 0.25, 3) * np.array([1, 1, 0])
                h /= np.linalg.norm(h) + 1e-9
                drop0 = math.radians(rng.uniform(20, 40))
                outer = np.clip((rad_xy - 3.0) / 4.5, 0, 1)
                az = math.atan2(outv[1], outv[0])
                hem = rng.uniform(0.0, 0.9) + 0.6 * (0.5 - 0.5 * math.cos(az - az_long))
                full = max(p[2] - hem, 1.0)
                L = full * (0.35 + 0.65 * outer) * (rng.uniform(0.5, 0.85) if rng.uniform() < 0.25 else 1.0)
                L = float(np.clip(L, 0.8, 11.0))
                s = np.unique(np.r_[0.0, 0.25, 0.6, 1.1, np.arange(1.1 + seg, L, seg), L])
                # leaves the branch heading out and a little down, then falls under its weight
                bend = 0.8
                hx = (bend * (1 - np.exp(-s / bend))) * math.cos(drop0)
                fall = s - hx * math.cos(drop0)
                P = p[None, :] + h[None, :] * hx[:, None] - UP[None, :] * (fall * 0.98 + 0.02 * s)[:, None]
                P[:, 2] = np.maximum(P[:, 2], 0.03)
                ph_ = rng.uniform(0, math.pi) + s * rng.uniform(0.2, 0.45) * rng.choice([-1, 1])
                hs = np.array([-h[1], h[0], 0.0])
                Wd = hs[None, :] * np.cos(ph_)[:, None] + h[None, :] * np.sin(ph_)[:, None] * 0.6
                Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
                wid = width / 2 * (1 - 0.45 * (s / L) ** 2)
                N = np.repeat((outv * 0.55 + UP * 0.6 + h * 0.2)[None, :], len(s), 0)
                depth = float(crown.shade(p)[0])
                t = _tint(rng, 0.06, 0.03) * (0.55 + 0.45 * outer) * (0.7 + 0.3 * depth)
                bright = np.array([1.0, 1.02, 0.86]) * (0.92 + 0.12 * outer)
                C = [t * bright * (1 - 0.3 * (sj / 10.0)) for sj in s]
                ribbon(tm, P, Wd * wid[:, None], N, C, strand_m, v=s / 1.68 + rng.uniform(0, 1))
    return tm.obj("tree_willow" + ("_lod2" if lod else ""), col)


# -- conifer (whorled, so built directly rather than grown) ---------------------------------------

def card_side(tm, A, D, W, size, mat, crown, uvq, tint, droop):
    """A conifer shoot card: the shoot enters at the card's left edge (texture u = 0) and runs
    along D; W spans the card's height. Outer part droops."""
    m = tm.mat(mat)
    h = size
    n0 = np.cross(D, W)
    D2 = D * math.cos(droop) - UP * math.sin(droop) * 0.6
    D2 /= np.linalg.norm(D2)
    o = A - W * h * 0.62
    P = np.array([o, o + D * h * 0.5, o + D * h * 0.5 + D2 * h * 0.5,
                  o + W * h, o + W * h + D * h * 0.5, o + W * h + D * h * 0.5 + D2 * h * 0.5])
    radial = crown.radial(P)
    face_n = n0 if (n0 @ radial.mean(0)) > 0 else -n0
    N = 0.65 * radial + 0.35 * face_n
    occ = crown.shade(P)
    i = tm.verts(P, N, occ[:, None] * np.asarray(tint)[None, :] * crown.sun(P, occ))
    u0, v0 = uvq
    uv = [(u0, v0), (u0 + 0.25, v0), (u0 + 0.5, v0), (u0, v0 + 0.5), (u0 + 0.25, v0 + 0.5), (u0 + 0.5, v0 + 0.5)]
    tm.face((i, i + 1, i + 4, i + 3), [uv[0], uv[1], uv[4], uv[3]], m)
    tm.face((i + 1, i + 2, i + 5, i + 4), [uv[1], uv[2], uv[5], uv[4]], m)


def _spruce_branch(tm, b, leaf_m, crown, rng, z, d, ln, L, lod, dense, tint, sides):
    """One branch: sags in the middle, upturned at the tip, shoot cards along it (one flat, one
    hanging below as the curtain of side shoots a spruce has), paler new growth at the tip."""
    ts = np.linspace(0, 1, 6 if not lod else 4)
    pts = [np.array([0, 0, z]) + d * (0.1 + ln * t) + np.array([0, 0, -ln * 0.28 * math.sin(math.pi * t * 0.8) + ln * 0.14 * t ** 3 - 0.05 * ln * t])
           for t in ts]
    rr = np.maximum(0.055 * (L / 3.4) ** 0.8 * (1 - ts) + 0.006, 0.006)
    if not lod or rr[0] > 0.035:
        # the branch wood stays in the shade under its shoots: dark, so it never reads on top
        tube(tm, pts, rr, b, None, sides, cap="cone",
             colour=[np.full(3, 0.22 + 0.12 * t) for t in ts])
    step = (0.24 if dense else 0.3) if not lod else 0.9
    m = max(1, int(ln / step))
    for j in range(m):
        t = (j + 0.5) / m
        fi = t * (len(pts) - 1)
        ii = min(int(fi), len(pts) - 2)
        q = pts[ii] + (pts[ii + 1] - pts[ii]) * (fi - ii)
        D = pts[ii + 1] - pts[ii]
        D /= np.linalg.norm(D)
        tt = tint * _tint(rng, 0.05, 0.02)
        if t > 0.78:
            tt = tt * np.array([1.1, 1.17, 0.86])          # this year's growth
        s = (1.0 if not lod else 2.0) * (0.72 + 0.3 * (1 - t)) * min(1.0, max(0.75, 0.5 + L / 4))
        for side, (tilt, droop) in enumerate(((0.0, 0.1), (rng.uniform(0.6, 1.0), 0.5))):
            W = np.cross(UP, D)
            W /= np.linalg.norm(W) + 1e-9
            nrm = np.cross(D, W)
            sgn = 1 if (j + side) % 2 else -1
            Wt = W * math.cos(tilt) - nrm * math.sin(tilt) * sgn
            uq = (rng.integers(2) * 0.5, rng.integers(2) * 0.5)
            card_side(tm, q - D * s * 0.15, D, Wt, s, leaf_m, crown, uq, tt, droop)
            if lod:
                break


def conifer(col, lod=False):
    """Picea abies, open grown: 23 m, a narrow cone 7 m across at the base of the live crown,
    whorls of drooping, upswept branches, a fine spire leader, needle litter beneath."""
    rng = np.random.default_rng(53)
    tm = TM()
    H = 23.0
    z_live = 2.3
    crown = ConeCrown(z_live, H, 4.1)
    b = bark("spruce")
    leaf_m = foliage_material("conifer")
    # stem: flared foot, tapering to a thin leader
    zs = np.r_[-0.3, 0.0, 0.1, 0.22, 0.4, 0.65, 1.0, np.linspace(1.5, H, 26 if not lod else 12)]
    P = np.stack([0.04 * np.sin(zs * 0.5), 0.03 * np.cos(zs * 0.37), zs], 1)
    R = 0.37 * (1 - np.clip(zs, 0, H) / H) ** 1.05 + 0.004
    ph = rng.uniform(0, 6.3)

    def fl(z, th):
        k = 0.55 * math.exp(-max(z, 0.0) / 0.45)
        return 1 + k * (0.45 + 0.75 * np.clip(np.cos(5 * th + ph), 0, 1) ** 3)
    stem_c = [np.full(3, 0.85 if z < z_live else 0.5 if z < z_live + 1.5 else 0.2) for z in zs]
    tube(tm, P, R, b, crown, 12 if not lod else 7, flare=fl, cap="cone", colour=stem_c)
    sides = 5 if not lod else 4
    z = 0.7
    whorl = 0
    while z < H - 0.55:
        frac = max(z - z_live, 0) / (H - z_live)
        L = max(4.1 * (1 - frac) ** 0.92 + 0.3, 0.75 * min(1.0, (H - z) / 1.2))   # dense tiers to the tip
        n = rng.integers(6, 8)
        dead = z < z_live
        bt = _tint(rng, 0.07, 0.04) * 0.82
        for k in range(n):
            ang = 2 * math.pi * (k + rng.uniform(-0.25, 0.25)) / n + whorl * 0.7
            d = np.array([math.cos(ang), math.sin(ang), 0.0])
            if dead:
                if lod:
                    continue
                ln = rng.uniform(0.25, 0.9)
                pts = [np.array([0, 0, z]) + d * (0.12 + ln * t) + UP * (-0.15 * ln * t) for t in (0, 0.5, 1)]
                tube(tm, pts, np.array([0.025, 0.016, 0.008]), b, None, 4, cap="point",
                     colour=[np.array([0.55, 0.52, 0.5])] * 3)
                continue
            ln = L * rng.uniform(0.82, 1.1)
            _spruce_branch(tm, b, leaf_m, crown, rng, z, d, ln, L, lod, frac > 0.3, bt * _tint(rng, 0.04, 0.03), sides)
        z += (rng.uniform(0.45, 0.58) * (0.6 + 0.4 * min(1.0, (H - z) / 4.0))) if not dead else rng.uniform(0.3, 0.5)
        whorl += 1
    # the leader: this year's shoot, 0.45 m, with a tuft of short upswept shoots at its tip
    for k in range(3 if not lod else 1):
        a = k * 2.1
        W = np.array([math.cos(a), math.sin(a), 0.0])
        card_side(tm, np.array([0, 0, H - 0.55]), UP, W, 0.5, leaf_m, crown, (0.0, 0.0), np.array([1.15, 1.25, 0.95]), 0.0)
    # a dark inner cone: the shaded heart of the crown, so it reads solid and the stem disappears
    core = flat_material("LeafCore_spruce", 0.9)
    mi = tm.mat(core)
    zc = np.linspace(z_live + 0.6, H - 7.0, 11 if not lod else 6)
    sides_c = 12 if not lod else 7
    th = np.linspace(0, 2 * math.pi, sides_c, endpoint=False)
    rings = []
    for z in zc:
        rr = (0.36 - 0.16 * (z - zc[0]) / (zc[-1] - zc[0])) * crown.rad_at(z) * (1 + 0.12 * np.sin(3 * th + z * 1.3) + 0.08 * np.sin(5 * th - z * 0.7))
        P3 = np.stack([np.cos(th) * rr, np.sin(th) * rr, np.full(sides_c, z)], 1)
        N3 = np.stack([np.cos(th), np.sin(th), np.full(sides_c, 0.4)], 1)
        rings.append(tm.verts(P3, N3, np.repeat(np.array([[0.008, 0.014, 0.009]]), sides_c, 0)))
    for a, b2 in zip(rings[:-1], rings[1:]):
        for k in range(sides_c):
            k2 = (k + 1) % sides_c
            tm.face_out((a + k, a + k2, b2 + k2, b2 + k), [(0, 0), (1, 0), (1, 1), (0, 1)], mi,
                        (math.cos(th[k]), math.sin(th[k]), 0))
    return tm.obj("tree_conifer" + ("_lod2" if lod else ""), col)


# -- hedge ----------------------------------------------------------------------------------------

def _noise3(P, ph, amp, freq):
    P = np.atleast_2d(P)
    out = np.zeros(len(P))
    for j, f in enumerate(freq):
        out += amp / (j + 1) * np.sin(P[:, 0] * f + ph[j, 0]) * np.sin(P[:, 1] * f * 1.3 + ph[j, 1]) * np.sin(P[:, 2] * f * 0.9 + ph[j, 2])
    return out


def hedge(col, lod=False):
    """A clipped field hedge segment 2.4 m long: battered sides (1.1 m at the base, 0.75 m across
    the top), rounded shoulders and ends, a lumpy clipped face with lighter new growth on top and
    a patch of die-back, a thin dark foot where bare stems show, litter along the bottom, and a
    few unclipped shoots standing up."""
    rng = np.random.default_rng(61)
    tm = TM()
    leaf_m = foliage_material("privet")
    face_m = leaf_face_material("privet")
    Lh = 1.2
    crown = Crown((0, 0, 0.95), (1.5, 0.65, 1.0), [((0, 0, 0.95), (1.5, 0.65, 1.0))], floor=0.55)
    # battered: 1.12 m across low down, 0.86 m across the top; the leafy body stops 0.2 m above
    # the ground where the bare stems show; the profile is closed underneath
    prof = np.array([(-0.36, 0.03), (-0.48, 0.14), (-0.56, 0.42), (-0.54, 0.85), (-0.5, 1.25), (-0.46, 1.5), (-0.42, 1.66),
                     (-0.32, 1.78), (-0.12, 1.83), (0.06, 1.835), (0.24, 1.8), (0.36, 1.72), (0.44, 1.58), (0.49, 1.3),
                     (0.53, 0.85), (0.56, 0.42), (0.48, 0.14), (0.36, 0.03), (0.0, 0.02), (-0.36, 0.03)])
    end = 0.32
    xs = np.r_[np.linspace(-Lh, -Lh + end, 5), np.linspace(-Lh + end, Lh - end, 6 if lod else 9)[1:-1], np.linspace(Lh - end, Lh, 5)]
    ph = rng.uniform(0, 6.3, (4, 3))
    G = np.zeros((len(xs), len(prof), 3))
    for a, x in enumerate(xs):
        e = max(abs(x) - (Lh - end), 0) / end           # 0 in the body, 1 at the very end
        s = math.sqrt(max(1 - e * e, 0.0))
        sy = 0.28 + 0.72 * s
        for k, (y, z) in enumerate(prof):
            zz = z * (0.72 + 0.28 * s) if z > 0.2 else z
            G[a, k] = (x + 0.06 * e * np.sign(x) * (1 - abs(y) / 0.56), y * sy, zz)
    def grid_normals(G):
        nx, nk = G.shape[:2]
        Nrm = np.zeros_like(G)
        for a in range(nx - 1):
            for k in range(nk - 1):
                n = np.cross(G[a + 1, k] - G[a, k], G[a, k + 1] - G[a, k])
                for (i, j) in ((a, k), (a + 1, k), (a, k + 1), (a + 1, k + 1)):
                    Nrm[i, j] += n
        Nrm /= np.linalg.norm(Nrm, axis=-1, keepdims=True) + 1e-9
        if Nrm[nx // 2, 3][1] > 0:      # profile point 3 is on the -y face
            Nrm = -Nrm
        return Nrm
    # lumps: displace along the surface normal (low-frequency, so faces never fold)
    Nrm = grid_normals(G)
    bump = _noise3(G.reshape(-1, 3), ph, 0.05, (1.3, 2.6)).reshape(G.shape[:2])
    bump *= np.clip((G[..., 2] - 0.1) / 0.3, 0, 1) * (1 - 0.6 * np.clip((G[..., 2] - 1.35) / 0.3, 0, 1))
    G = G + Nrm * bump[..., None]
    Nrm = grid_normals(G)
    nx, nk = G.shape[:2]
    flat = G.reshape(-1, 3)
    z = flat[:, 2]
    base = 0.28 + 0.72 * _smooth((z - 0.03) / 0.75)
    top = np.clip((z - 1.5) / 0.35, 0, 1)
    die = np.clip((_noise3(flat * 0.6, ph[::-1], 1.0, (1.7,)) - 0.35) * 3, 0, 1)
    C = base[:, None] * (1 + top[:, None] * np.array([0.12, 0.14, -0.1]))
    C = C * (1 - die[:, None] * np.array([-0.18, 0.12, 0.45]))
    C = C * (0.9 + 0.2 * (_noise3(flat, ph, 1.0, (2.3,)) * 0.5 + 0.5))[:, None]
    b0 = tm.verts(flat, Nrm.reshape(-1, 3), C)
    mi = tm.mat(face_m)

    def uv(p, n):
        ax = int(np.argmax(np.abs(n)))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        return (p[a] / 0.5, p[b] / 0.5)
    for a in range(nx - 1):
        for k in range(nk - 1):
            ids = (a * nk + k, a * nk + k + 1, (a + 1) * nk + k + 1, (a + 1) * nk + k)
            n = Nrm[a, k] + Nrm[a + 1, k + 1]
            tm.face_out(tuple(b0 + i for i in ids), [uv(flat[i], n) for i in ids], mi, n)
    # close the ends (tiny, the rounding pulls them in)
    for a, sgn in ((0, -1), (nx - 1, 1)):
        cp = G[a].mean(0) + np.array([sgn * 0.05, 0, 0])
        c = tm.verts([cp], [[sgn, 0, 0.3]], [[0.75, 0.75, 0.75]])
        for k in range(nk):
            i0, i1 = b0 + a * nk + k, b0 + a * nk + (k + 1) % nk
            tm.face_out((i0, i1, c), [(flat[i0 - b0][1] / 0.5, flat[i0 - b0][2] / 0.5), (flat[i1 - b0][1] / 0.5, flat[i1 - b0][2] / 0.5),
                                      (cp[1] / 0.5, cp[2] / 0.5)], mi, (sgn, 0, 0))
    # leaf sprays lying on the faces, standing proud at the shoulders and ends
    quads = [(a, k) for a in range(nx - 1) for k in range(nk - 1)]
    area = np.array([np.linalg.norm(np.cross(G[a + 1, k] - G[a, k], G[a, k + 1] - G[a, k])) for a, k in quads])
    w = area * np.array([1.6 if G[a, k][2] > 1.45 or abs(G[a, k][0]) > Lh - end else 1.0 for a, k in quads])
    w *= np.array([0.0 if G[a, k][2] < 0.06 or G[a + 1, k + 1][2] < 0.06 else 0.6 if G[a, k][2] < 0.4 else 1.0
                   for a, k in quads])
    w /= w.sum()
    clumps = {}
    for j in range(130 if lod else 1150):
        a, k = quads[rng.choice(len(quads), p=w)]
        u, v = rng.uniform(0, 1, 2)
        p = (G[a, k] * (1 - u) * (1 - v) + G[a + 1, k] * u * (1 - v) + G[a, k + 1] * (1 - u) * v + G[a + 1, k + 1] * u * v)
        n = Nrm[a, k] + Nrm[a + 1, k + 1]
        n /= np.linalg.norm(n)
        edge = p[2] > 1.5 or abs(p[0]) > Lh - end
        p = p + n * rng.uniform(-0.015, 0.03)
        t = rng.normal(0, 1, 3) + UP * 0.8
        t -= n * (t @ n)
        t /= np.linalg.norm(t) + 1e-9
        Dd = t + n * (0.15 if edge else 0.08)
        Dd /= np.linalg.norm(Dd)
        W = np.cross(n, Dd)
        W /= np.linalg.norm(W) + 1e-9
        deep = rng.uniform() < 0.15
        if deep:                                      # older leaves set back in the hedge: darker, bluer
            p = p - n * 0.03
        s = (0.26 if not lod else 0.6) * rng.uniform(0.8, 1.2)
        zt = 1 + max(p[2] - 1.5, 0) / 0.35 * np.array([0.14, 0.16, -0.1])
        tint = _clump_tint(clumps, p, rng, 0.6) * _tint(rng, 0.07, 0.03) * zt * (0.45 + 0.55 * _smooth((p[2] - 0.05) / 0.75))
        if deep:
            tint = tint * np.array([0.75, 0.8, 0.84])
        r_ = rng.uniform()
        if r_ < 0.025:
            tint = tint * np.array([1.6, 1.05, 0.45])         # yellowing
        elif r_ < 0.045:
            tint = tint * np.array([1.4, 0.75, 0.4])          # browned
        card(tm, p - Dd * s * 0.3, Dd, W, s, leaf_m, crown, quarter(rng), tint, bend=0.35 if edge else 0.1, normal_bias=0.7, out=n)
    if not lod:
        # unclipped shoots standing up from the top, bare stems in the shadow at the foot
        twig = bark("oak")
        for j in range(36):
            x = rng.uniform(-Lh + 0.15, Lh - 0.15)
            y = rng.uniform(-0.3, 0.3)
            p0 = np.array([x, y, -0.02])
            d = np.array([rng.normal(0, 0.2), y * 0.9 + rng.normal(0, 0.15), 1.0])
            d /= np.linalg.norm(d)
            kink = d + rng.normal(0, 0.15, 3)
            tube(tm, [p0, p0 + d * 0.17, p0 + d * 0.17 + kink * 0.2], np.array([0.014, 0.011, 0.009]) * rng.uniform(0.6, 1.2),
                 twig, None, 4, cap=None, colour=[np.array([0.3, 0.28, 0.26]), np.array([0.36, 0.34, 0.31]), np.array([0.3, 0.3, 0.28])])
        # sprigs of new growth poking 5-15 cm above the cut line and out of the faces
        twig = bark("oak")
        for j in range(55):
            a = rng.integers(nx - 1)
            top_k = [k for k in range(nk - 1) if G[a, k][2] > 1.55]
            side = j % 5 == 0
            if side or not top_k:
                k = int(rng.integers(3, nk - 4))
            else:
                k = top_k[rng.integers(len(top_k))]
            p = G[a, k] + (G[a + 1, k] - G[a, k]) * rng.uniform()
            n = Nrm[a, k]
            Dd = (n * 0.9 + UP * 0.6 if side else UP + n * 0.3) + rng.normal(0, 0.3, 3)
            Dd /= np.linalg.norm(Dd)
            s = rng.uniform(0.16, 0.26)
            card(tm, p - Dd * s * 0.25, Dd, _width_axis(Dd, rng, 1.2), s, leaf_m, crown, quarter(rng),
                 np.array([1.1, 1.15, 0.8]) * _tint(rng, 0.06, 0.03), bend=0.3, normal_bias=0.6)
    return tm.obj("hedge" + ("_lod2" if lod else ""), col)


# -- shrub ----------------------------------------------------------------------------------------

def shrub(col, lod=False, name="shrub", leaf="shrub", radius=0.8, height=1.25, seed=5, z0=0.0):
    """A free-growing shrub, wider than tall: woody stems fanning from one root crown, branching
    out to a lumpy, irregular head of leaf sprays with a few long shoots poking out, lifted off
    the ground on one side where the stems show, on a patch of leaf litter."""
    rng = np.random.default_rng(seed)
    tm = TM()
    leaf_m = foliage_material(leaf)
    s = radius / 0.8
    c = np.array([0.05, 0.0, z0 + height * 0.52])
    r = np.array([radius, radius * 0.8, height * 0.46])
    cl = [(c, r * 0.85)] + [(c + d * r * 0.55, r * rng.uniform(0.45, 0.6)) for d in
                            [np.array([math.cos(a), math.sin(a), rng.uniform(-0.2, 0.5)]) for a in rng.uniform(0, 6.3, 7)]]
    crown = Crown(c, r * 1.1, cl, z_min=z0 + 0.18 * s, floor=0.28)
    sk = Skeleton()
    root = sk.add((0, 0, z0 - 0.03), -1)
    for j in range(7):
        a = 2 * math.pi * j / 7 + rng.normal(0, 0.3)
        limb(sk, root, a, math.radians(rng.uniform(55, 75)), math.radians(rng.uniform(40, 60)), 0.28 * s, rng, step=0.07 * s,
             tort=0.15, kinks=1)
    A = crown.sample(1300, rng)
    sprig = []
    for j in range(5):
        a = rng.uniform(0, 6.3)
        d = np.array([math.cos(a), math.sin(a), rng.uniform(0.1, 0.8)])
        d /= np.linalg.norm(d)
        sprig.append(c + d * r * 1.08 + rng.normal(0, 0.04, (12, 3)))
    A = np.vstack([A] + sprig)
    colonize(sk, A, 0.07 * s, 0.34 * s, 0.07 * s, rng, jitter=0.18)
    sk.finish(exponent=2.2, r_base=0.04 * s)
    skin(tm, sk, bark("oak"), crown, min_r=0.006 if not lod else 0.015, sides_for=lambda rr: 5 if rr > 0.012 else 3,
         decimate=0.02, ao_floor=0.5, tint_fn=lambda idx: [np.array([0.62, 0.58, 0.55])] * len(idx))
    if lod:
        lod_cards(tm, sk, crown, leaf_m, rng, 0.6 * s, cell=0.35 * s)
    else:
        leaf_sprays(tm, sk, crown, leaf_m, rng, 0.3 * s, per_tip=1.3, inner=0.4, budget=640, outward=0.45, vary=0.08,
                    face_out=True)
    # an inner mass of older, shaded leaves, so the middle is never see-through to a black void
    for j in range(70 if not lod else 15):
        q = c + rng.normal(0, 1, 3) * r * np.array([0.3, 0.3, 0.25])
        D = rng.normal(0, 1, 3) + UP * 0.5
        D /= np.linalg.norm(D)
        card(tm, q - D * 0.2 * s, D, _width_axis(D, rng, 1.2), 0.45 * s, leaf_m, crown, quarter(rng),
             np.array([0.75, 0.8, 0.8]) * _tint(rng, 0.06, 0.03), bend=0.3, normal_bias=0.8)
    return tm.obj(name + ("_lod2" if lod else ""), col)


# -- bramble --------------------------------------------------------------------------------------

BRAMBLE_Q = (0.38, 0.38, 0.12, 0.12)   # leaves only, leaves only, flowers, fruit


def bramble(col, lod=False):
    """A bramble patch (Rubus fruticosus): 20 thorny canes, reddish purple going green, arching
    up to 1.2 m and down again 0.8-1.8 m out, several rooting at the tip, wider one way than the
    other; compound leaves along the canes (flowers and blackberries on some), an open, dark,
    stemmy base on dry litter."""
    rng = np.random.default_rng(71)
    tm = TM()
    leaf_m = foliage_material("bramble")
    cane_m = flat_material("Bramble_Cane", 0.55)
    crown = Crown((0, 0, 0.45), (1.6, 1.25, 0.85), [((0, 0, 0.45), (1.6, 1.25, 0.85))], floor=0.28)
    c_old, c_young = np.array([0.11, 0.025, 0.03]), np.array([0.155, 0.09, 0.045])
    berry_m = flat_material("Bramble_Berry", 0.25)
    berries = []
    n_canes = 26 if not lod else 10
    clumps = {}
    for j in range(n_canes):
        root = np.array([rng.normal(0, 0.18), rng.normal(0, 0.14), 0.0])
        a = 2 * math.pi * j / n_canes + rng.normal(0, 0.35)
        d = rng.uniform(1.0, 1.9) if j % 4 else rng.uniform(0.4, 0.8)
        dirv = np.array([math.cos(a) * 1.0, math.sin(a) * 0.78, 0.0])
        # one side of the patch stands taller; a few first-year canes arch high over the mass
        tall = 0.75 + 0.35 * math.cos(a - 0.8)
        h = rng.uniform(0.6, 1.1) * tall * (1.0 if d > 0.6 else 0.7)
        if j % 7 == 3:
            h, d = rng.uniform(1.3, 1.5), rng.uniform(1.4, 1.9)
        z_end = 0.0 if rng.uniform() < 0.4 else rng.uniform(0.12, 0.5)
        side = np.array([-dirv[1], dirv[0], 0]) * rng.normal(0, 0.25)
        P0, P2 = root, root + dirv * d + UP * z_end
        P1 = root + dirv * d * 0.3 + side + UP * h * 1.75
        ts = np.linspace(0, 1, 12 if not lod else 6)
        pts = np.array([(1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t * t * P2 for t in ts])
        pts[:, 2] = np.maximum(pts[:, 2], 0.01)
        rr = np.linspace(rng.uniform(0.0045, 0.006), 0.002, len(ts)) * (1.0 if j % 3 else 1.0)
        if j % 3 == 0:
            rr = np.linspace(0.0055, 0.0022, len(ts))
        cols = [c_old * (1 - t) + c_young * t for t in ts]
        tube(tm, pts, rr * 1.8, cane_m, None, 6 if not lod else 3, cap="cone",
             colour=[cc * (0.55 + 0.45 * float(crown.shade(p)[0])) for cc, p in zip(cols, pts)])
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        total = seg[-1]
        if not lod:
            # thorns: little hooked spikes every 6 cm
            for k, sl in enumerate(np.arange(0.05, total - 0.05, 0.09)):
                i = min(np.searchsorted(seg, sl) - 1, len(pts) - 2)
                f = (sl - seg[i]) / (seg[i + 1] - seg[i] + 1e-9)
                q = pts[i] + (pts[i + 1] - pts[i]) * f
                T = pts[i + 1] - pts[i]
                T /= np.linalg.norm(T)
                th = k * 2.4
                n = np.cross(T, UP if abs(T[2]) < 0.9 else np.array([1.0, 0, 0]))
                n /= np.linalg.norm(n)
                n = n * math.cos(th) + np.cross(T, n) * math.sin(th)
                rq = rr[i]
                base = q + n * rq * 0.8
                i0 = tm.verts([base - T * 0.003, base + T * 0.003, base + n * 0.007 - T * 0.003],
                              [n, n, n], [cols[i] * 1.3] * 3)
                tm.face((i0, i0 + 1, i0 + 2), [(0, 0), (1, 0), (0.5, 1)], tm.mat(cane_m))
        if rng.uniform() < 0.7:
            berries.append(pts[-2] + UP * 0.03)
        if rng.uniform() < 0.5:
            berries.append(pts[len(pts) // 2] + UP * 0.02)
        # leaves along the cane, from 0.25 m up (more in the sunny upper part)
        step = 0.085 if not lod else 0.25
        for sl in np.arange(0.22, total, step):
            i = min(np.searchsorted(seg, sl) - 1, len(pts) - 2)
            f = (sl - seg[i]) / (seg[i + 1] - seg[i] + 1e-9)
            q = pts[i] + (pts[i + 1] - pts[i]) * f
            T = pts[i + 1] - pts[i]
            T /= np.linalg.norm(T)
            sd = np.cross(T, UP)
            sd /= np.linalg.norm(sd) + 1e-9
            D = T * 0.5 + sd * rng.choice([-1, 1]) * rng.uniform(0.4, 1.0) + UP * rng.uniform(0.0, 0.5) + rng.normal(0, 0.2, 3)
            D /= np.linalg.norm(D)
            W = _width_axis(D, rng, 0.6)
            for rep_ in range(2 if (q[2] > 0.6 and not lod and rng.uniform() < 0.4) else 1):
                s = (0.46 if not lod else 0.85) * rng.uniform(0.8, 1.2)
                if rep_:
                    D = D + rng.normal(0, 0.5, 3)
                    D /= np.linalg.norm(D)
                    W = _width_axis(D, rng, 0.6)
                card(tm, q - D * s * 0.15, D, W, s, leaf_m, crown, quarter(rng, BRAMBLE_Q),
                     _clump_tint(clumps, q, rng, 0.7) * _tint(rng, 0.07, 0.03), bend=rng.uniform(0.2, 0.6), normal_bias=0.6)
    # a denser core so the middle is not see-through, kept off the ground
    for j in range(160 if not lod else 20):
        a = rng.uniform(0, 6.3)
        rad = rng.uniform(0, 0.9) ** 0.7
        q = np.array([math.cos(a) * rad * 1.1, math.sin(a) * rad * 0.85, rng.uniform(0.35, 0.95) * (1 - 0.4 * rad)])
        D = np.array([math.cos(a), math.sin(a), 0.6]) + rng.normal(0, 0.4, 3)
        D /= np.linalg.norm(D)
        s = (0.4 if not lod else 0.8) * rng.uniform(0.8, 1.2)
        card(tm, q - D * s * 0.3, D, _width_axis(D, rng, 0.8), s, leaf_m, crown, quarter(rng, BRAMBLE_Q),
             _clump_tint(clumps, q, rng, 0.7) * _tint(rng, 0.07, 0.03) * (0.6 + 0.4 * rad), bend=0.4, normal_bias=0.65)
    if not lod:
        # last year's dead canes inside the thicket: grey, leafless, broken
        for j in range(8):
            a = rng.uniform(0, 6.3)
            dv = np.array([math.cos(a), math.sin(a) * 0.8, 0.0])
            P0 = np.array([rng.normal(0, 0.15), rng.normal(0, 0.12), 0.0])
            P1 = P0 + dv * 0.4 + UP * rng.uniform(0.6, 1.0)
            P2 = P0 + dv * rng.uniform(0.7, 1.1) + UP * rng.uniform(0.1, 0.4)
            pts = _bezier(P0, P1, P2, 6) * np.array([1, 1, 1])
            tube(tm, pts, np.linspace(0.009, 0.004, 6), cane_m, None, 5, cap="jagged", rng=rng,
                 colour=[np.array([0.16, 0.15, 0.13])] * 6)
        # fruit: trusses of 3-8 berries, green, red and glossy black, at cane tips and leaf axils
        mi = tm.mat(berry_m)
        oct_d = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
        oct_f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        for c0 in berries:
            for b_ in range(rng.integers(3, 9)):
                cpos = c0 + rng.normal(0, 0.022, 3)
                r_ = rng.uniform(0.011, 0.014)
                k = rng.uniform()
                bcol = (np.array([0.012, 0.008, 0.014]) if k < 0.5 else np.array([0.22, 0.015, 0.02]) if k < 0.8
                       else np.array([0.1, 0.14, 0.035]))
                i0 = tm.verts(cpos + oct_d * r_ * np.array([1, 1, 1.15]), oct_d, [bcol] * 6)
                for f in oct_f:
                    tm.face(tuple(i0 + v for v in f), [(0, 0), (1, 0), (0.5, 1)], mi)
        # dead grass and old leaves round the base (no bare soil)
        bm = foliage_material("blade")
        for j in range(40):
            a = rng.uniform(0, 6.3)
            p0 = np.array([math.cos(a) * rng.uniform(0.2, 1.1), math.sin(a) * rng.uniform(0.2, 0.9), 0.0])
            dc = np.array([0.26, 0.2, 0.1]) * rng.uniform(0.6, 1.1)
            blade(tm, bm, p0, a + rng.normal(0, 0.6), rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.4), 0.006, rng.uniform(0.8, 1.5),
                  rng.normal(0, 0.5), (dc * 0.5, dc, dc * 0.9), n=3)
    return tm.obj("bramble" + ("_lod2" if lod else ""), col)


# -- ground plants: blades, bulrushes, weeds --------------------------------------------------------

def blade(tm, mat, root, az, elev, length, width, droop, twist, cols, n=5, strap=False, normal_out=None, vfold=0.0):
    """A grass or reed leaf as a tapered ribbon: from `root` heading `az`, rising at `elev` (radians
    above horizontal) and bending over by `droop` along its length, twisting by `twist`; cols =
    colours at base, middle and tip (vertex colours over a near-white blade texture)."""
    h = np.array([math.cos(az), math.sin(az), 0.0])
    side0 = np.array([-h[1], h[0], 0.0])
    p = np.array(root, float)
    P, W, N, C, FN = [], [], [], [], []
    seg = length / n
    for i in range(n + 1):
        t = i / n
        e = elev - droop * t ** 1.5
        d = h * math.cos(e) + UP * math.sin(e)
        if i:
            p = p + d * seg
            if p[2] < 0.008:
                p[2] = 0.008
        tw = twist * t
        s = side0 * math.cos(tw) + np.cross(d, side0) * math.sin(tw)
        wt = width * ((1.0 if t < 0.85 else max((1 - t) / 0.15, 0.0)) if strap else (1 - t ** 1.7) * (0.8 + 0.2 * (1 - t)))
        wt = max(wt, 0.0008)
        fn = np.cross(s, d)
        if fn[2] < 0:
            fn = -fn
        out = normal_out if normal_out is not None else h
        nrm = fn * 0.4 + (out * 0.55 + UP * 0.8) * 0.6
        P.append(p.copy())
        W.append(s * wt / 2)
        N.append(nrm)
        C.append(cols[0] + (cols[1] - cols[0]) * min(t * 2, 1) if t < 0.5 else cols[1] + (cols[2] - cols[1]) * (t - 0.5) * 2)
        if vfold:
            FN.append(fn)
    v = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(P), axis=0), axis=1))] / 0.25
    if not vfold:
        ribbon(tm, P, W, N, C, mat, v=v)
        return
    # a V-folded leaf: the midrib sunk below the two margins (the channel faces up and out)
    P, W, FN = np.array(P), np.array(W), np.array(FN)
    wl = np.linalg.norm(W, axis=1, keepdims=True)
    mid = P - FN * wl * math.tan(vfold)
    pts = np.stack([P - W, mid, P + W], 1).reshape(-1, 3)
    Ns = np.stack([np.array(N) - W * 8, np.array(N), np.array(N) + W * 8], 1).reshape(-1, 3)
    Cs = np.repeat(np.array(C), 3, 0) * np.tile([1.06, 0.94, 1.06], len(P))[:, None]
    b = tm.verts(pts, Ns, Cs)
    m = tm.mat(mat)
    for i in range(len(P) - 1):
        for c in range(2):
            a0, a1 = b + 3 * i + c, b + 3 * (i + 1) + c
            tm.face((a0, a0 + 1, a1 + 1, a1), [(c * 0.5, v[i]), ((c + 1) * 0.5, v[i]), ((c + 1) * 0.5, v[i + 1]), (c * 0.5, v[i + 1])], m)


def _lin3(h, k=1.0):
    return _lin(h) * k


def grass_tuft(col, lod=False):
    """A lopsided tussock (cock's-foot / tufted hair-grass), 0.5 m of leaf and a group of flowering
    culms to 0.8 m: a dense dark crown off-centre, blades 2-6 mm wide and of mixed length (longer
    and flopping on one side, some folded or broken), darker sheaths at the base and yellowing
    tips, about 30% dead straw underneath and outside, and 9 culms of similar height with nodding
    panicles, some purple-brown, some buff."""
    rng = np.random.default_rng(81)
    tm = TM()
    m = foliage_material("blade")
    sh = foliage_material("seedheads")
    crown = Crown((0, 0, 0.25), (0.45, 0.45, 0.35), [((0, 0, 0.25), (0.45, 0.45, 0.35))])
    sheath = _lin("#3c4623") * 0.85
    green = np.array([0.045, 0.095, 0.02])
    tipc = _lin("#96965a") * 0.85
    straw = _lin("#a5916a") * 0.85
    off = np.array([0.03, -0.02, 0.0])
    lop = rng.uniform(0, 2 * math.pi)

    def root(spread=0.09):
        a, r = rng.uniform(0, 6.3), spread * math.sqrt(rng.uniform())
        return off + np.array([math.cos(a) * r, math.sin(a) * r * 0.85, 0.0]), a, r / spread
    k = 1 if not lod else 3
    for j in range(380 // k):
        p, a, rr = root()
        az = a + rng.normal(0, 0.5)
        bias = 0.75 + 0.45 * (0.5 + 0.5 * math.cos(az - lop))           # longer on one side
        L = rng.uniform(0.18, 0.55) * bias * (0.8 + 0.4 * rr)
        elev = math.pi / 2 - rng.uniform(0.05, 0.4) - 0.5 * rr
        droop = rng.uniform(0.2, 0.7) + 1.6 * max(L - 0.35, 0) / 0.3 * rng.uniform(0.6, 1.2) * bias
        if rng.uniform() < 0.12:
            droop += rng.uniform(1.4, 2.2)                                  # folded or broken over
        g = green * rng.uniform(0.8, 1.2) * np.array([1 + rng.normal(0, 0.06), 1, 1])
        t_ = tipc * rng.uniform(0.85, 1.1) if rng.uniform() < 0.22 else g * np.array([1.2, 1.1, 0.9])
        wd = rng.choice([0.002, 0.004, 0.006], p=[0.3, 0.4, 0.3]) * rng.uniform(0.9, 1.15)
        blade(tm, m, p, az, elev, L, wd, droop, rng.normal(0, 0.8), (sheath, g, t_), n=4 if not lod else 2)
    for j in range(115 // k):                                               # dead straw, outside and under
        p, a, rr = root(0.11)
        L = rng.uniform(0.2, 0.45)
        dc = straw * rng.uniform(0.75, 1.15)
        blade(tm, m, p, a + rng.normal(0, 0.3), rng.uniform(0.45, 1.0), L, rng.uniform(0.003, 0.006), rng.uniform(1.4, 2.4),
              rng.normal(0, 1.2), (dc * 0.5, dc, dc * 0.95), n=4 if not lod else 2)
    if not lod:
        for j in range(90):                                                 # dark matted thatch in the crown
            p, a, rr = root(0.08)
            mix = rng.uniform()
            cc = np.array([0.06, 0.045, 0.025]) * (1 - mix) + straw * 0.4 * mix
            blade(tm, m, p, a + rng.normal(0, 0.6), rng.uniform(0.6, 1.3), rng.uniform(0.05, 0.12), 0.006, rng.uniform(0.3, 1.2),
                  rng.normal(0, 1.5), (cc * 0.5, cc, cc * 1.1), n=3)
        for j in range(16):                                                 # straw lying flat round the base
            p, a, rr = root(0.1)
            dc = straw * rng.uniform(0.6, 0.95)
            blade(tm, m, p + UP * 0.005, a, 0.05, rng.uniform(0.15, 0.3), 0.005, 0.1, rng.normal(0, 0.4),
                  (dc * 0.6, dc, dc * 0.85), n=3)
    # flowering culms: a group of similar height, nodding, mixed ages (one card each, crossed)
    cells = [0, 1, 0, 1, 2, 0, 1, 3, 0] if not lod else [0, 1, 0]
    for j, cell in enumerate(cells):
        p, a, rr = root(0.07)
        Hc = rng.uniform(0.72, 0.85)
        lean = 0.1 + abs(rng.normal(0, 0.06))
        D = np.array([math.cos(lop + rng.normal(0, 0.8)) * lean, math.sin(lop + rng.normal(0, 0.8)) * lean, 1.0])
        D /= np.linalg.norm(D)
        for x in range(2 if not lod else 1):
            yaw = rng.uniform(0, math.pi) + x * math.pi / 2
            W = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            W -= D * (W @ D)
            W /= np.linalg.norm(W)
            card(tm, p, D, W, Hc, sh, crown, (cell / 4, 0.0, (cell + 1) / 4, 1.0), np.ones(3), bend=rng.uniform(0.25, 0.45),
                 aspect=0.25, normal_bias=0.6, shade=False)
    return tm.obj("grass_tuft" + ("_lod2" if lod else ""), col)


def reeds(col, lod=False):
    """A stand of bulrush (Typha latifolia), 2 m: thirty shoots in a 0.9 x 0.65 m clump, each a fan
    of 5-7 thick strap leaves 15-20 mm wide, V-folded and slowly twisting, glaucous green with
    yellowing tips, the outer leaves leaning out and a quarter of them dead straw; seed heads as
    blunt dark velvety brown cylinders (26 x 200 mm) on 9 mm stems sheathed at the foot, staggered
    in height, one burst into fluff, the thin male spike above each; broken dead stems lying at
    the foot."""
    rng = np.random.default_rng(83)
    tm = TM()
    m = foliage_material("blade")
    head_m = flat_material("Typha_Head", 0.9)
    shoots = []
    for j in range(30 if not lod else 9):
        a, r = rng.uniform(0, 6.3), math.sqrt(rng.uniform())
        shoots.append(np.array([math.cos(a) * r * 0.45, math.sin(a) * r * 0.32, 0.0]))
    live0 = _lin("#6f8a5a") * 0.6
    live1 = _lin("#86a06a") * 0.6
    tip_y = _lin("#b9a46a") * 0.65
    straw = _lin("#c9b48a") * 0.62
    for sp in shoots:
        plane = rng.uniform(0, math.pi)
        outv = np.array([sp[0], sp[1], 0.0])
        rr = np.linalg.norm(outv)
        outv = outv / (rr + 1e-9) if rr > 1e-6 else np.array([1.0, 0, 0])
        edge = rr / 0.45
        n_l = rng.integers(5, 8) if not lod else 3
        for k in range(n_l):
            az = plane + (math.pi if k % 2 else 0) + rng.normal(0, 0.2)
            inner = k >= n_l - 3
            L = rng.uniform(1.4, 2.1) * (1.0 if inner else 0.85)
            elev = math.pi / 2 - (rng.uniform(0.02, 0.08) if inner else rng.uniform(0.06, 0.2))
            droop = rng.uniform(0.7, 1.3) if rng.uniform() < 0.22 else rng.uniform(0.04, 0.22)
            g = live0 + (live1 - live0) * rng.uniform()
            g = g * rng.uniform(0.88, 1.12)
            tipc = tip_y * rng.uniform(0.85, 1.1) if rng.uniform() < 0.3 else g * np.array([1.08, 1.06, 0.92])
            cols = (g * 0.55, g, tipc)
            if not inner and rng.uniform() < 0.1 + 0.25 * edge:     # dead straw, outside, bending out
                dc = straw * rng.uniform(0.8, 1.1)
                cols = (dc * 0.55, dc, dc * 1.05)
                az = math.atan2(outv[1], outv[0]) + rng.normal(0, 0.5)
                elev, droop = math.pi / 2 - rng.uniform(0.25, 0.5), rng.uniform(0.5, 1.1)
                L *= rng.uniform(0.55, 0.85)
            blade(tm, m, sp + rng.normal(0, 0.01, 3) * np.array([1, 1, 0]), az, elev, L, rng.uniform(0.015, 0.02), droop,
                  rng.uniform(0.5, 1.3) * rng.choice([-1, 1]), cols, n=6 if not lod else 3, strap=True, normal_out=outv,
                  vfold=0.0 if lod else math.radians(25))
    # flower spikes
    heights = np.linspace(1.55, 2.1, 9 if not lod else 3)
    rng.shuffle(heights)
    for j, top in enumerate(heights):
        sp = shoots[(j * 3) % len(shoots)] + rng.normal(0, 0.02, 3) * np.array([1, 1, 0])
        lean = rng.normal(0, 0.03, 2)
        d = np.array([lean[0], lean[1], 1.0])
        d /= np.linalg.norm(d)
        z_head = top - 0.35
        stem_pts = [sp + d * z for z in np.linspace(0, z_head, 4)]
        sc = live0 * 0.9
        tube(tm, stem_pts, np.array([0.0048, 0.0045, 0.0042, 0.004]), m, None, 5 if not lod else 3, cap=None,
             colour=[sc * 0.5, sc, sc, sc])
        if not lod:   # the leaf sheath wrapping the foot of the flowering stem
            tube(tm, [sp, sp + d * 0.15, sp + d * 0.32], np.array([0.013, 0.011, 0.007]), m, None, 6, cap=None,
                 colour=[live0 * 0.5, live0 * 0.85, live0])
        h0 = sp + d * z_head
        hl = rng.uniform(0.17, 0.23)
        burst = j == 4
        hc = (np.array([0.32, 0.3, 0.27]) if burst else _lin("#4a2e1c")) * rng.uniform(0.85, 1.1)
        rmax = 0.019 if burst else 0.013
        prof = [(0.0, 0.004), (0.006, rmax * 0.8), (0.016, rmax), (hl - 0.016, rmax), (hl - 0.006, rmax * 0.8), (hl, 0.004)]
        pts = [h0 + d * z for z, _ in prof]
        tube(tm, pts, np.array([r for _, r in prof]), head_m, None, 8 if not lod else 5, cap="cone",
             colour=[hc * (0.9 + 0.1 * math.sin(i)) for i in range(len(prof))])
        if not lod:
            s0 = h0 + d * (hl + rng.uniform(0.005, 0.03))
            tube(tm, [h0 + d * hl, s0, s0 + d * 0.13], np.array([0.0035, 0.0028, 0.0012]), head_m, None, 4, cap="cone",
                 colour=[sc, np.array([0.3, 0.24, 0.12]), np.array([0.38, 0.3, 0.16])])
    if not lod:
        # last year's stems, broken and lying at the foot
        for j in range(7):
            a = rng.uniform(0, 6.3)
            p0 = np.array([math.cos(a) * rng.uniform(0.2, 0.5), math.sin(a) * rng.uniform(0.15, 0.35), 0.012])
            h = a + rng.normal(0, 0.6)
            dv = np.array([math.cos(h), math.sin(h), 0.0])
            L = rng.uniform(0.4, 0.9)
            tube(tm, [p0, p0 + dv * L * 0.5 + UP * 0.01, p0 + dv * L], np.array([0.005, 0.0045, 0.004]), m, None, 4,
                 cap="jagged", rng=rng, colour=[straw * 0.8] * 3)
    return tm.obj("reeds" + ("_lod2" if lod else ""), col)


WEED_CELL = 0.25


def weed_leaf(tm, mat, crown, base, az, elev, length, cell, col, fold=0.25, curl=0.4, rows=4, normal_up=0.6, flip=False,
              wave=0.0):
    """A leaf as a folded card: three columns (margin, midrib, margin) by `rows`, the margins
    raised by `fold` (a V about the midrib), arching down by `curl` along its length; the leaf
    is painted with its base at the bottom centre of a weed-atlas cell."""
    i, j = cell % 4, cell // 4
    u0, v0 = i * WEED_CELL, j * WEED_CELL
    h = np.array([math.cos(az), math.sin(az), 0.0])
    side = np.array([-h[1], h[0], 0.0])
    p = np.array(base, float)
    rowsP, rowsN = [], []
    for r in range(rows + 1):
        t = r / rows
        e = elev - curl * t
        d = h * math.cos(e) + UP * math.sin(e)
        if r:
            p = p + d * length / rows
        p[2] = max(p[2], 0.006)
        up_n = np.cross(side, d)
        if up_n[2] < 0:
            up_n = -up_n
        w = length / 2
        wv = wave * w * math.sin(t * math.pi * 3.0) * min(t * 4, 1)
        L = p - side * w + up_n * (fold * w + wv)
        R = p + side * w + up_n * (fold * w - wv)
        rowsP.append([L, p.copy(), R])
        nrm = up_n * (1 - normal_up) + UP * normal_up
        rowsN.append([nrm - side * fold * 0.6, nrm, nrm + side * fold * 0.6])
    P = np.array(rowsP).reshape(-1, 3)
    N = np.array(rowsN).reshape(-1, 3)
    occ = 0.75 + 0.25 * np.clip(P[:, 2] / 0.3, 0, 1)
    b = tm.verts(P, N, occ[:, None] * col[None, :])
    mi = tm.mat(mat)
    us = [u0 + 0.004, u0 + WEED_CELL / 2, u0 + WEED_CELL - 0.004]
    if flip:
        us = us[::-1]
    for r in range(rows):
        va, vb = v0 + WEED_CELL * r / rows, v0 + WEED_CELL * (r + 1) / rows
        for c in range(2):
            a0, a1 = b + r * 3 + c, b + (r + 1) * 3 + c
            tm.face((a0, a0 + 1, a1 + 1, a1), [(us[c], va), (us[c + 1], va), (us[c + 1], vb), (us[c], vb)], mi)


def weeds(col, lod=False):
    """A patch of waste-ground weeds, each one plant: a broad-leaved dock rosette (leaves splayed
    20-45 degrees, red midribs, an old holed leaf) with its rust-brown seed spike, two dandelion
    rosettes lying flat with a flower and a seed clock, a clump of nettles (square stems, opposite
    leaf pairs at right angles, smaller up the stem, drooping) and a ribwort plantain in flower,
    on bare soil."""
    rng = np.random.default_rng(85)
    tm = TM()
    m = foliage_material("weeds")
    stem_m = foliage_material("blade")
    crown = Crown((0, 0, 0.3), (0.6, 0.6, 0.4), [((0, 0, 0.3), (0.6, 0.6, 0.4))])
    g = lambda: _tint(rng, 0.08, 0.03)
    lo = lod

    def card_x(base, Hc, cell, tint=np.ones(3), lean=0.05, aspect=1.0):
        i, j = cell % 4, cell // 4
        du = WEED_CELL * (1 - aspect) / 2
        D = np.array([rng.normal(0, lean), rng.normal(0, lean), 1.0])
        D /= np.linalg.norm(D)
        for x in range(1 if lo else 2):
            yaw = rng.uniform(0, math.pi) + x * math.pi / 2
            W = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            card(tm, np.asarray(base, float), D, W, Hc, m, crown, (i * WEED_CELL + du, j * WEED_CELL, (i + 1) * WEED_CELL - du, (j + 1) * WEED_CELL),
                 tint, bend=0.0, normal_bias=0.5, shade=False, aspect=aspect)

    def stalk(p0, p1, r, colr, sides=4):
        tube(tm, [p0, (p0 + p1) / 2 + rng.normal(0, 0.01, 3) * np.array([1, 1, 0]), p1], np.array([r, r * 0.9, r * 0.8]), stem_m, None,
             sides, cap=None, colour=[colr * 0.7, colr, colr])

    # dock
    dock = np.array([0.08, 0.05, 0.0])
    n = 7 if not lo else 4
    for k in range(n):
        az = 2 * math.pi * k / n + rng.normal(0, 0.2)
        old = k == 0
        weed_leaf(tm, m, crown, dock, az, math.radians(rng.uniform(12, 30) if not old else 6), rng.uniform(0.26, 0.36),
                  1 if old else 0, g() * (np.array([1.08, 1.0, 0.82]) if old else 1.0), fold=0.28, curl=rng.uniform(0.25, 0.5),
                  rows=6 if not lo else 2, wave=0.0 if lo else 0.22)
    # the dock's flowering stem: leaves graded up it, then a panicle of whorled seed spikes
    top = dock + np.array([0.03, 0.02, 0.0])
    tip = top + np.array([0.02, -0.01, 0.78])
    stalk(dock, tip, 0.0045, np.array([0.16, 0.07, 0.035]))
    for k in range(3 if not lo else 1):
        q = dock + (tip - dock) * (0.15 + 0.13 * k)
        weed_leaf(tm, m, crown, q, rng.uniform(0, 6.3), math.radians(30), 0.16 - 0.035 * k, 0, g(), fold=0.25, curl=0.5,
                  rows=3 if not lo else 1)
    card_x(tip - UP * 0.24, 0.27, 10, aspect=0.4, lean=0.02)
    for k in range(4 if not lo else 1):
        q0 = dock + (tip - dock) * (0.62 + 0.08 * k)
        a = k * 1.9 + rng.uniform(-0.3, 0.3)
        q1 = q0 + np.array([math.cos(a) * 0.07, math.sin(a) * 0.07, 0.06])
        stalk(q0, q1, 0.0022, np.array([0.16, 0.07, 0.035]), 3)
        card_x(q1 - UP * 0.02, 0.16 - 0.02 * k, 10, aspect=0.4, lean=0.15)
    # dandelions
    for c0 in (np.array([-0.32, 0.22, 0.0]), np.array([-0.12, 0.42, 0.0])):
        for k in range(11 if not lo else 5):
            az = 2 * math.pi * k / 11 + rng.normal(0, 0.15)
            weed_leaf(tm, m, crown, c0, az, math.radians(rng.uniform(4, 16)), rng.uniform(0.15, 0.26), [2, 3, 2, 3, 15][k % 5], g(),
                      fold=0.12, curl=0.1, rows=3 if not lo else 1, flip=k % 2 == 1)
    if not lo:
        # flower heads on hollow stalks 15-30 cm, facing the sky with a side view for low angles,
        # and two seed clocks
        for k, c0 in enumerate([np.array([-0.32, 0.22, 0.0])] * 3 + [np.array([-0.12, 0.42, 0.0])] * 3):
            a = rng.uniform(0, 6.3)
            hgt = rng.uniform(0.15, 0.3)
            q = c0 + np.array([math.cos(a) * 0.06, math.sin(a) * 0.06, hgt])
            stalk(c0, q, 0.0022, np.array([0.11, 0.14, 0.06]))
            if k in (2, 5):
                card_x(q - UP * 0.045, 0.09, 9)
                continue
            card(tm, q - np.array([0.022, 0, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.045, m, crown,
                 (0.0, 0.5, 0.25, 0.75), np.ones(3), bend=0.0, normal_bias=0.3, shade=False)
            card_x(q - UP * 0.005, 0.05, 12)
    # nettles: a clonal clump of square stems 0.6-1 m, opposite pairs of heart-shaped leaves all the
    # way up (each pair at right angles to the last), shrinking towards the tip and drooping, and
    # thin green flower strands hanging from the upper axils
    nets = [np.array([0.3, -0.16, 0.0]) + np.array([rng.normal(0, 0.09), rng.normal(0, 0.07), 0.0]) for _ in range(6 if not lo else 2)]
    for sidx, s0 in enumerate(nets):
        Hs = rng.uniform(0.5, 0.85)
        lean = rng.normal(0, 0.07, 2)
        d = np.array([lean[0], lean[1], 1.0])
        d /= np.linalg.norm(d)
        stalk(s0, s0 + d * Hs, 0.0035, np.array([0.07, 0.095, 0.03]))
        nodes = 9 if not lo else 3
        for k in range(nodes):
            t = 0.08 + 0.87 * k / nodes
            q = s0 + d * Hs * t
            sz = 0.15 * (1.1 - 0.5 * t)
            for side in (0, 1):
                az = (k % 2) * math.pi / 2 + side * math.pi + rng.normal(0, 0.15) + sidx
                weed_leaf(tm, m, crown, q, az, math.radians(rng.uniform(10, 30)), sz * rng.uniform(0.9, 1.1), 4 if k < 6 else 5,
                          g() * (np.array([1.1, 1.04, 0.8]) if k == 0 else 1.0), fold=0.25, curl=rng.uniform(0.3, 0.5),
                          rows=3 if not lo else 1)
                if not lo and t > 0.45:
                    h = np.array([math.cos(az + 0.5), math.sin(az + 0.5), 0.0])
                    a0 = q + h * 0.01
                    tube(tm, [a0, a0 + h * 0.025 - UP * 0.02, a0 + h * 0.035 - UP * 0.075], np.array([0.0012, 0.001, 0.0008]),
                         stem_m, None, 3, cap=None, colour=[np.array([0.2, 0.24, 0.1])] * 3)
        tq = s0 + d * Hs
        for side in (0, 1):     # the young pair at the top, small and held up
            weed_leaf(tm, m, crown, tq, sidx + side * math.pi, math.radians(55), 0.05, 5, g(), fold=0.3, curl=0.3,
                      rows=2 if not lo else 1)
    # ribwort plantain
    pc = np.array([-0.1, -0.3, 0.0])
    for k in range(7 if not lo else 3):
        az = 2 * math.pi * k / 7 + rng.normal(0, 0.2)
        weed_leaf(tm, m, crown, pc, az, math.radians(rng.uniform(30, 55)), rng.uniform(0.12, 0.2), 6 + k % 2, g(), fold=0.25,
                  curl=0.5, rows=3 if not lo else 1)
    for k in range(2 if not lo else 1):
        q = pc + np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), 0.0])
        tip = q + np.array([rng.normal(0, 0.04), rng.normal(0, 0.04), rng.uniform(0.28, 0.36)])
        stalk(q, tip, 0.0018, np.array([0.1, 0.12, 0.05]), 3)
        card_x(tip - UP * 0.04, 0.09, 13)
    if not lo:
        # last year's dead leaves lying flat among the plants
        for j in range(10):
            a = rng.uniform(0, 6.3)
            p0 = np.array([math.cos(a), math.sin(a), 0.0]) * rng.uniform(0.1, 0.5) + UP * 0.006
            weed_leaf(tm, m, crown, p0, rng.uniform(0, 6.3), math.radians(2), rng.uniform(0.1, 0.2), 14, _tint(rng, 0.1, 0.04) * 0.5,
                      fold=0.05, curl=-0.05, rows=1)
        # low filler round the plants: coarse grass blades and a few dead ones, no bare soil disc
        for j in range(70):
            a = rng.uniform(0, 6.3)
            p0 = np.array([math.cos(a), math.sin(a), 0.0]) * rng.uniform(0.05, 0.55)
            dead_ = rng.uniform() < 0.25
            gc = np.array([0.26, 0.2, 0.1]) * rng.uniform(0.7, 1.1) if dead_ else np.array([0.05, 0.1, 0.022]) * rng.uniform(0.8, 1.2)
            blade(tm, stem_m, p0, a + rng.normal(0, 0.8), rng.uniform(0.6, 1.3), rng.uniform(0.08, 0.2), 0.006,
                  rng.uniform(0.3, 1.4), rng.normal(0, 0.6), (gc * 0.45, gc, gc * 1.1), n=3)
    return tm.obj("weeds" + ("_lod2" if lod else ""), col)


# -- registry -------------------------------------------------------------------------------------

def _pair(fn):
    """A builder (col, M) for the prop and one for its .lod2."""
    return (lambda col, M: fn(col, False)), (lambda col, M: fn(col, True))


BUILDERS = {}
LODS = {}
for _key, _fn in {
    "prop.tree_oak": lambda col, lod: tree(col, "oak", lod),
    "prop.tree_birch": lambda col, lod: tree(col, "birch", lod),
    "prop.tree_willow": weeping_willow,
    "prop.tree_dead": lambda col, lod: tree(col, "dead", lod),
    "prop.tree_conifer": conifer,
    "prop.shrub": lambda col, lod: shrub(col, lod),
    "prop.hedge": hedge,
    "prop.bramble": bramble,
    "prop.grass_tuft": grass_tuft,
    "prop.reeds": reeds,
    "prop.weeds": weeds,
}.items():
    BUILDERS[_key], LODS[_key] = _pair(_fn)
