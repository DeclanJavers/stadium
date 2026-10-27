"""Reusable props for training grounds (and stadiums): the mesh toolkit, shared materials and the
registry. The props themselves live by category: props_street (bins, benches, lighting, signs),
props_training (mannequins, cones, goals, balls), props_site (the construction kit, containers,
derelict odds), props_machines (tractor, mower, plant), vehicles (cars, minibus) and vegetation
(trees, hedges, scrub). Each is generated at real size; origin on the ground at its footprint
centre, local +Y its front (where a bench faces, where a car points).

Every prop is one mesh with a few materials, light enough to instance by the hundred, with
ambient occlusion baked into its vertex colours. Names starting with TINT_ are recolourable
(bins, cabins, car paint, containers, hoardings, club colours).

    build("prop.bench", col) -> root          keys(): every prop key (".lod2" coarse levels too)
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import collection, mesh_object
from . import dugout, facade, genprops, seat, site_materials, stand


# -- a mesh accumulator ---------------------------------------------------------------------------

class PM:
    """Polygons with a material per face; UVs box-projected in metres unless given.

    Every face also carries an edge radius (`self.r` when it was added, metres): when the object
    is made, coincident vertices are welded so each part is a closed solid, and every hard edge is
    rounded with its faces' radius (a moulded bin body 25 mm, a steel tube's end 3 mm), with
    hardened normals so flat faces stay flat."""

    def __init__(self, r=0.006):
        self.V, self.F, self.mi, self.uv, self.mats, self.rad = [], [], [], [], [], []
        self.r = r

    def m(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def face(self, pts, mat, uv=None):
        base = len(self.V)
        self.V.extend([tuple(map(float, p)) for p in pts])
        self.F.append(tuple(range(base, base + len(pts))))
        self.mi.append(self.m(mat))
        self.uv.append(uv)
        self.rad.append(self.r)

    def faces(self, V, F, mat, xf=None, uvs=None):
        """Append a vertex list and index faces, optionally transformed (uvs: per vertex)."""
        V = [xf @ Vector(v) if xf is not None else Vector(v) for v in V]
        for f in F:
            self.face([V[i] for i in f], mat, [uvs[i] for i in f] if uvs is not None else None)

    def box(self, lo, hi, mat, xf=None):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        V = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        self.faces(V, F, mat, xf)

    def obox(self, c, size, mat, yaw=0.0, pitch=0.0, roll=0.0):
        """Box of `size` centred on c, turned by yaw (z), pitch (x), roll (y)."""
        sx, sy, sz = (s / 2 for s in size)
        xf = Matrix.Translation(Vector(c)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(pitch, 4, "X") @ Matrix.Rotation(roll, 4, "Y")
        self.box((-sx, -sy, -sz), (sx, sy, sz), mat, xf)

    def cyl(self, a, b, r, mat, sides=16, r2=None, cap=True):
        """Cylinder (or cone, r2) from point a to point b."""
        a, b = Vector(a), Vector(b)
        d = b - a
        ln = d.length
        if ln < 1e-9:
            return
        rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
        xf = Matrix.Translation(a) @ rot
        r2 = r if r2 is None else r2
        ring0 = [(r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides), 0.0) for k in range(sides)]
        ring1 = [(r2 * math.cos(2 * math.pi * k / sides), r2 * math.sin(2 * math.pi * k / sides), ln) for k in range(sides)]
        V = ring0 + ring1
        F = [(k, (k + 1) % sides, sides + (k + 1) % sides, sides + k) for k in range(sides)]
        if cap:
            F.append(tuple(range(sides - 1, -1, -1)))
            if r2 > 1e-6:
                F.append(tuple(range(sides, 2 * sides)))
        self.faces(V, F, mat, xf)

    def sweep(self, path, profile, mat, closed=False, cap=True, twist_ref=(0, 0, 1)):
        """Sweep a closed 2D profile [(u, v)] along a 3D path with parallel-transport frames (the
        profile never flips) and mitred corners. closed: the path is a loop."""
        P = np.asarray(path, float)
        n = len(P)
        if closed:
            T = [P[(i + 1) % n] - P[i - 1] for i in range(n)]
        else:
            T = [P[min(i + 1, n - 1)] - P[max(i - 1, 0)] for i in range(n)]
        T = [t / (np.linalg.norm(t) + 1e-12) for t in T]
        ref = np.array(twist_ref, float)
        if abs(T[0] @ ref) > 0.95:
            ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0])
        N = [np.cross(ref, T[0])]
        N[0] /= np.linalg.norm(N[0])
        for i in range(1, n):
            v = N[-1] - T[i] * (N[-1] @ T[i])
            N.append(v / (np.linalg.norm(v) + 1e-12))
        m = len(profile)
        V = []
        for i in range(n):
            B = np.cross(T[i], N[i])
            V.extend(P[i] + N[i] * u + B * v for u, v in profile)
        F = []
        for i in range(n if closed else n - 1):
            i2 = (i + 1) % n
            for j in range(m):
                j2 = (j + 1) % m
                F.append((i * m + j, i2 * m + j, i2 * m + j2, i * m + j2))
        if cap and not closed:
            F.append(tuple(range(m - 1, -1, -1)))
            F.append(tuple((n - 1) * m + j for j in range(m)))
        self.faces(V, F, mat)

    def tube(self, pts, r, mat, sides=8, bend=None, cap=True, closed=False):
        """A round tube through pts, its corners bent round on radius `bend` (default 3 r)."""
        if len(pts) > 2:
            pts = round_path(pts, bend if bend is not None else 3 * r, closed=closed)
        prof = [(r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides)) for k in range(sides)]
        self.sweep(pts, prof, mat, closed=closed, cap=cap)

    def sphere(self, c, r, mat, nu=16, nv=10, scale=(1, 1, 1), z_min=-1.0):
        V, F = [], []
        rows = [math.pi * (j / nv) - math.pi / 2 for j in range(nv + 1)]
        rows = [t for t in rows if math.sin(t) >= z_min - 1e-9] or rows
        for t in rows:
            for k in range(nu):
                a = 2 * math.pi * k / nu
                V.append((c[0] + r * scale[0] * math.cos(t) * math.cos(a), c[1] + r * scale[1] * math.cos(t) * math.sin(a),
                          c[2] + r * scale[2] * math.sin(t)))
        for j in range(len(rows) - 1):
            for k in range(nu):
                a, b = j * nu + k, j * nu + (k + 1) % nu
                F.append((a, b, b + nu, a + nu))
        self.faces(V, F, mat)

    def lathe(self, prof, mat, c=(0, 0, 0), sides=16, flutes=0, depth=0.0, mats=None, cap_top=False):
        """Surface of revolution of (r, z) points about the vertical through c. flutes and depth:
        vertical ribs; mats(j, k) -> the material of profile segment j, sector k."""
        V, F, M = [], [], []
        for r, z in prof:
            for k in range(sides):
                a = 2 * math.pi * k / sides
                rr = r * (1 - depth * (0.5 - 0.5 * math.cos(flutes * a))) if flutes else r
                V.append(Vector((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a), c[2] + z)))
        for j in range(len(prof) - 1):
            for k in range(sides):
                a, b = j * sides + k, j * sides + (k + 1) % sides
                F.append((a, b, b + sides, a + sides))
                M.append(mats(j, k) if mats else mat)
        if cap_top:
            j = len(prof) - 1
            F.append(tuple(j * sides + k for k in range(sides)))
            M.append(mat)
        for f, mm in zip(F, M):
            self.face([V[i] for i in f], mm)

    def loft(self, rings, mat, cap0=True, cap1=True, mats=None):
        """Skin closed rings (lists of 3D points, all the same length) in order, capping the ends."""
        n, m = len(rings), len(rings[0])
        V = [Vector(p) for r in rings for p in r]
        for i in range(n - 1):
            for j in range(m):
                j2 = (j + 1) % m
                f = (i * m + j, (i + 1) * m + j, (i + 1) * m + j2, i * m + j2)
                self.face([V[k] for k in f], mats(i, j) if mats else mat)
        if cap0:
            self.face([V[j] for j in range(m - 1, -1, -1)], mat)
        if cap1:
            self.face([V[(n - 1) * m + j] for j in range(m)], mat)

    def extrude(self, poly, y0, y1, mat, plane="xz"):
        """A polygon extruded across: "xz" polygons (x, z) counter-clockwise seen from -y, from
        y0 to y1 > y0; "yz" polygons (y, z) counter-clockwise seen from +x, from x0 to x1."""
        n = len(poly)
        if plane == "xz":
            V = [(x, y0, z) for x, z in poly] + [(x, y1, z) for x, z in poly]
            F = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
            F += [(n + i, n + (i + 1) % n, (i + 1) % n, i) for i in range(n)]
        else:
            V = [(y0, a, b) for a, b in poly] + [(y1, a, b) for a, b in poly]
            F = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
            F += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        self.faces(V, F, mat)

    def extrude_yz(self, poly, x0, x1, mat):
        """A polygon in the (y, z) plane (counter-clockwise seen from +x) between x0 and x1."""
        self.extrude(poly, x0, x1, mat, plane="yz")

    def quad(self, a, b, c, d, mat, uv=None):
        self.face([a, b, c, d], mat, uv)

    def card(self, c, w, h, yaw, mat, uv=(0, 0, 1, 1), tilt=0.0, bottom=True):
        """A vertical card w x h standing on (or centred at) c, turned by yaw, leaning by tilt."""
        xf = Matrix.Translation(Vector(c)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(tilt, 4, "X")
        z0 = 0.0 if bottom else -h / 2
        P = [xf @ Vector(p) for p in ((-w / 2, 0, z0), (w / 2, 0, z0), (w / 2, 0, z0 + h), (-w / 2, 0, z0 + h))]
        u0, v0, u1, v1 = uv
        self.face(P, mat, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    def decal(self, c, n, v, w, h, mat, lift=0.002):
        """A flat label w x h centred on c on a face with outward normal n, its top towards v,
        read the right way round from outside, standing `lift` off the face."""
        c, n, v = (np.asarray(x, float) for x in (c, n, v))
        u = np.cross(v, n)
        c = c + n * lift
        P = [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
        r, self.r = self.r, 0.0
        self.face(P, mat, [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.r = r

    def obj(self, name, col, smooth_deg=None, bevel=True):
        """The mesh object: welded, smoothed by angle, hard edges rounded per face radius."""
        if not self.F:
            return None
        V = np.array(self.V, float)
        uvs = []
        for f, uv, mi in zip(self.F, self.uv, self.mi):
            if uv is not None:
                cell = self.mats[mi].get("uv_cell")
                uvs.extend([(u / cell[0], v / cell[1]) for u, v in uv] if cell else uv)
                continue
            P = V[list(f)]
            n = np.cross(P[1] - P[0], P[2] - P[0]) if len(f) >= 3 else np.zeros(3)
            ax = int(np.argmax(np.abs(n)))
            keep = [i for i in range(3) if i != ax]
            uvs.extend([tuple(p[keep]) for p in P])
        o = mesh_object(name, V, self.F, col, materials=self.mats, mat_index=self.mi, uvs=uvs)
        ang = math.radians(smooth_deg or 35.0)
        bm = bmesh.new()
        bm.from_mesh(o.data)
        rl = bm.faces.layers.float.new("edge_radius")
        for f, r in zip(bm.faces, self.rad):
            f[rl] = r
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
        rmax = max(self.rad) if bevel else 0.0
        if rmax > 0:
            # bevel weight per edge: the smaller radius of its two faces, on edges sharper than ang
            bw = bm.edges.layers.float.get("bevel_weight_edge") or bm.edges.layers.float.new("bevel_weight_edge")
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > ang:
                    e[bw] = min(e.link_faces[0][rl], e.link_faces[1][rl]) / rmax
        bm.faces.layers.float.remove(rl)
        bm.to_mesh(o.data)
        bm.free()
        stand.smooth_by_angle(o, smooth_deg or 35.0)
        if rmax > 0:
            mod = o.modifiers.new("Bevel", "BEVEL")
            mod.width = rmax
            mod.segments = 3
            mod.profile = 0.6
            mod.limit_method = "WEIGHT"
            mod.harden_normals = True
            mod.use_clamp_overlap = True
            apply_modifiers(o)
        return o


def round_path(pts, r, n=5, closed=False):
    """A polyline with its corners replaced by circular arcs of radius r (shrunk to fit)."""
    P = [np.asarray(p, float) for p in pts]
    if r <= 0:
        return P
    m = len(P)
    out = []
    for i in range(m):
        if not closed and (i == 0 or i == m - 1):
            out.append(P[i])
            continue
        a, b, c = P[i - 1], P[i], P[(i + 1) % m]
        u, v = a - b, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-9 or lv < 1e-9:
            out.append(b)
            continue
        u, v = u / lu, v / lv
        cosang = float(np.clip(u @ v, -1, 1))
        if cosang < -0.999:
            out.append(b)
            continue
        half = math.acos(cosang) / 2
        t = min(r / math.tan(half), lu * 0.49, lv * 0.49)
        rr = t * math.tan(half)
        p0, p1 = b + u * t, b + v * t
        bis = (u + v) / np.linalg.norm(u + v)
        cen = b + bis * (rr / math.sin(half))
        for k in range(n + 1):
            q = p0 * (1 - k / n) + p1 * (k / n)
            d = q - cen
            out.append(cen + d / np.linalg.norm(d) * rr)
    return out


def rrect(w, d, r, z=0.0, n=4, cx=0.0, cy=0.0):
    """A rounded rectangle (w along x, d along y, corner radius r) as a counter-clockwise ring."""
    r = min(r, w / 2 - 1e-4, d / 2 - 1e-4)
    out = []
    for qx, qy, a0 in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
        ccx, ccy = cx + qx * (w / 2 - r), cy + qy * (d / 2 - r)
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            out.append((ccx + r * math.cos(a), ccy + r * math.sin(a), z))
    return out


def apply_modifiers(o):
    """Bake an object's modifier stack into its mesh."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = o.data
    o.modifiers.clear()
    o.data = me
    bpy.data.meshes.remove(old)


# -- materials ------------------------------------------------------------------------------------

def _plastic(name, color, rough=0.45):
    return MAT.plastic(name, color, rough=rough, dirt=0.2)


def _paint(name, color, rough=0.35):
    return MAT.painted_metal(name, color, rough=rough, dirt=0.3, scuffs=0.25)


def car_paint(name="TINT_car_paint", color="#2b3f66"):
    from . import vehicles
    return vehicles.car_paint(name, color)


def glass_dark(name="Prop_Glass"):
    return facade.dark_glass(name)


def emissive(name, color, strength=2.0):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=0.3, Emission_Color=srgb(color), Emission_Strength=strength))
    return m


def mesh_image(name, cell=0.1, wire=0.004, px=256, color="#d8dad6", cell_v=None):
    """An alpha image of welded mesh or netting: one cell (cell x cell_v metres) per repeat. Faces
    using it give UVs in metres; PM scales them to cells (the scale lives in the geometry, so
    every engine tiles it right)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    cell_v = cell_v or cell
    s = np.linspace(0, 1, px, endpoint=False) + 0.5 / px
    X, Y = np.meshgrid(s, s)
    a = np.maximum(np.abs(X - 0.5) > 0.5 - wire / cell, np.abs(Y - 0.5) > 0.5 - wire / cell_v).astype(float)
    c = np.array(srgb(color)[:3])
    rgba = np.concatenate([np.broadcast_to(c, (px, px, 3)), a[..., None]], -1)
    img = MAT.image_from_array(name + "_img", rgba)
    m, nb = material(name)
    uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
    tex = nb.node("ShaderNodeTexImage", image=img, extension="REPEAT")
    nb.connect(uv, tex.inputs["Vector"])
    clip = nb.math("ROUND", tex.outputs["Alpha"])
    nb.output(nb.principled(Base_Color=srgb(color), Metallic=0.6, Roughness=0.45, Alpha=clip))
    m["uv_cell"] = (cell, cell_v)
    return m


def clear_glass(name="Prop_GlassClear"):
    """Clear glazing (plant cabs, kiosks): faintly tinted, mostly see-through."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#9fb0b6"), Roughness=0.05, Alpha=0.22, Specular_IOR_Level=0.8))
    return m


def label(name, lines, bg, fg, px=(512, 256)):
    """A sign face: lines of text (the kit's 5x7 face, scaled to fit and softened) in fg on bg;
    bg None leaves the background transparent (letters cut out of a clear card)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    from ..stands.patterns import hex_rgb, text_bitmap
    W, H = px
    bmps = [text_bitmap(t) for t in lines]
    cols = max(b.shape[1] for b in bmps)
    rows = 7 * len(bmps) + 3 * (len(bmps) - 1)
    cell = max(1, int(min(W * 0.9 / cols, H * 0.78 / rows)))
    mask = np.zeros((H, W))
    y = (H - rows * cell) // 2
    for b in bmps:
        x = (W - b.shape[1] * cell) // 2
        big = np.kron(b.astype(float), np.ones((cell, cell)))
        mask[y:y + big.shape[0], x:x + big.shape[1]] = big
        y += 10 * cell
    # soften the pixel steps a little (letters read as type, not as blocks)
    k = max(1, cell // 4)
    for ax in (0, 1):
        mask = sum(np.roll(mask, d, ax) for d in range(-k, k + 1)) / (2 * k + 1)
    mask = np.clip((mask - 0.5) * 3 + 0.5, 0, 1)[::-1]
    lin = lambda h: hex_rgb(h) ** 2.2
    f = lin(fg)
    rgba = np.zeros((H, W, 4))
    if bg is None:
        rgba[..., :3] = f
        rgba[..., 3] = mask
    else:
        b = lin(bg)
        rgba[..., :3] = b + (f - b) * mask[..., None]
        rgba[..., 3] = 1.0
    img = MAT.image_from_array(name + "_img", rgba)
    m, nb = material(name)
    tex = nb.node("ShaderNodeTexImage", image=img, extension="CLIP")
    nb.connect(nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0], tex.inputs["Vector"])
    if bg is None:
        nb.output(nb.principled(Base_Color=tex.outputs["Color"], Roughness=0.4, Alpha=nb.math("ROUND", tex.outputs["Alpha"])))
    else:
        nb.output(nb.principled(Base_Color=tex.outputs["Color"], Roughness=0.4))
    return m


def tyre(p, c, r, w, rim_r, rubber, rim, sides=20, lugs=0, lug_h=0.0, axis="x"):
    """A wheel on an axle along x (or y): a tyre with rounded shoulders (and lugs: chevron tread
    blocks for tractors and plant), a dished rim and hub."""
    c = np.asarray(c, float)
    ax = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 1.0, 0])
    e1 = np.array([0, 1.0, 0]) if axis == "x" else np.array([1.0, 0, 0])
    e2 = np.array([0, 0, 1.0])
    hw = w / 2
    sw = r - rim_r

    def ring(profile, mat, n=sides):
        G = [[c + ax * a + (e1 * math.cos(2 * math.pi * k / n) + e2 * math.sin(2 * math.pi * k / n)) * rr for rr, a in profile] for k in range(n)]
        m = len(profile)
        for k in range(n):
            k2 = (k + 1) % n
            for j in range(m - 1):
                q = [G[k][j], G[k2][j], G[k2][j + 1], G[k][j + 1]]
                p.face(q if axis == "x" else q[::-1], mat)
    prof = [(rim_r * 1.02, -hw * 0.85), (rim_r + 0.55 * sw, -hw), (r - 0.18 * sw, -hw * 0.97), (r, -hw * 0.75),
            (r, hw * 0.75), (r - 0.18 * sw, hw * 0.97), (rim_r + 0.55 * sw, hw), (rim_r * 1.02, hw * 0.85)]
    ring(prof, rubber)
    ring([(rim_r * 1.02, hw * 0.85), (rim_r, hw * 0.7), (rim_r * 0.35, hw * 0.55), (rim_r * 0.3, hw * 0.75), (0.0, hw * 0.75)], rim)
    ring([(0.0, -hw * 0.75), (rim_r * 0.3, -hw * 0.75), (rim_r * 0.35, -hw * 0.55), (rim_r, -hw * 0.7), (rim_r * 1.02, -hw * 0.85)], rim)
    if lugs:
        # chevron lugs: angled blocks standing proud of the tread, alternating sides
        for k in range(lugs):
            a = 2 * math.pi * k / lugs
            for s in (-1, 1):
                a2 = a + s * 0.08
                d0 = e1 * math.cos(a) + e2 * math.sin(a)
                d1 = e1 * math.cos(a2 + 0.25) + e2 * math.sin(a2 + 0.25)
                p0 = c + ax * (s * 0.05 * w) + d0 * r
                p1 = c + ax * (s * 0.48 * w) + d1 * r
                t = np.cross(ax, (p1 - p0) / np.linalg.norm(p1 - p0))
                bw = 0.035 * r / 0.5
                q = [p0 - t * bw, p1 - t * bw, p1 + t * bw, p0 + t * bw]
                n0 = (d0 + d1) / 2
                n0 /= np.linalg.norm(n0)
                top = [x + n0 * lug_h for x in q]
                p.faces(q + top, [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], rubber)


def mats():
    """Shared prop materials (made once)."""
    return {
        "galv": seat.galvanised(),
        "steel": _paint("Prop_Steel", "#2c2f33", 0.45),
        "black": _plastic("Prop_Black", "#18191a", 0.5),
        "grey": _plastic("Prop_Grey", "#77797b", 0.5),
        "white": _plastic("Prop_White", "#e8e8e4", 0.4),
        "red": _plastic("Prop_Red", "#c4161c", 0.4),
        "orange": _plastic("Prop_Orange", "#e5641c", 0.45),
        "yellow": _paint("Prop_Yellow", "#e2a91a", 0.38),
        "green": _plastic("Prop_Green", "#2d6a3a", 0.45),
        "blue": _plastic("Prop_Blue", "#2457a6", 0.45),
        "rubber": MAT.rubber("Prop_Rubber", "#141414"),
        "alu": MAT.aluminium("Prop_Alu", 0.3),
        "glass": glass_dark(),
        "timber": site_materials.planks("Prop_Timber", "#7a5a3c", board=0.12),
        "concrete": stand.concrete("Prop_Concrete", "#a39f96"),
        "rust": site_materials.rust(),
        "lens": emissive("Prop_Lamp", "#fff3dc", 1.5),
        "tail": emissive("Prop_Tail", "#b0141a", 0.3),
        "poly": dugout.polycarbonate("#3d4954"),
        "net": mesh_image("Prop_Net", 0.06, 0.003, color="#f2f2ee"),
        "clear": clear_glass(),
        "chrome": _chrome(),
        "track": _paint("Prop_Track", "#2b2a28", 0.6),
    }


def _chrome(name="Prop_Chrome"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb("#c9cdd0"), Metallic=1.0, Roughness=0.12))
    return m


def _registry():
    """Every prop builder by key (category modules import this one, so gather them lazily)."""
    from . import props_amenity, props_hospitality, props_kit, props_machines, props_media, props_medical, props_office, props_site, props_street, props_training, vegetation, vehicles
    out = {}
    for mod in (props_street, props_amenity, props_training, props_site, props_machines, vegetation, vehicles, props_medical, props_office, props_hospitality, props_media, props_kit):
        out.update(mod.BUILDERS)
    return out


def _lods():
    from . import vegetation, vehicles
    return dict(vegetation.LODS, **vehicles.LODS)


def build(key, col=None):
    """Build a prop; returns its root (an empty named PROP:<key> holding the mesh)."""
    from ..core import ao
    from ..core.scene import empty
    col = col or collection("Props")
    if key.startswith(genprops.PREFIX):
        # a text-to-3D prop: its finished file, else its code fallback, else unknown
        root = genprops.build(key, col)
        if root is None:
            fb = genprops.fallback(key)
            if fb is None:
                raise KeyError(f"generated prop {key!r} is not finished and has no fallback")
            return build(fb, col)
        return root
    M = mats()
    fn = _lods().get(key[:-5]) if key.endswith(".lod2") else _registry().get(key)
    if fn is None:
        raise KeyError(f"unknown prop {key!r}")
    objs = fn(col, M)
    objs = [o for o in (objs if isinstance(objs, (list, tuple)) else [objs]) if o is not None]
    if not key.endswith(".lod2"):
        for o in objs:
            # occlusion in the vertex colours (vegetation brings its own crown shading)
            if o.type == "MESH" and "Col" not in o.data.color_attributes:
                ao.bake(o)
    root = empty(f"PROP:{key}", col)
    for o in objs:
        o.parent = root
    root["asset_id"] = key
    root["asset_family"] = "prop"
    return root


def keys():
    """Every prop key, coarse levels (key + ".lod2", drawn beyond the editor's LOD distance) and the
    finished generated props (genprop.<id>) included."""
    k = list(_registry())
    lods = _lods()
    return k + [f"{x}.lod2" for x in k if x in lods] + genprops.keys()
