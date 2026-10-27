"""Seats. Six styles, each with variants, built at real dimensions.

Origin: on the tread at its back edge (the riser line of the row behind), centred on the seat.
+Y points at the pitch, +Z up. Seat colour comes from the per-instance attribute `seat_color`
(set by the stand's colour pattern) and falls back to the TINT_seat colour.
"""
import math

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import material, srgb
from ..core.scene import bake, collection, join, mesh_object, set_params
from . import kit_detail as KD


# -- materials --------------------------------------------------------------------------------

def seat_plastic(default="#953030"):
    """Polypropylene shell: satin (roughness ~0.5), faint spark-eroded mould texture; colour from
    the instancer if present (the game engine replaces it per instance)."""
    name = "TINT_seat"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    col = nb.mix(attr[3], srgb(default), attr[0])  # Alpha is 0 when no instancer colour exists
    pos = nb.geometry()["Position"]
    grain = nb.noise(pos, 900.0, detail=2)
    bsdf = nb.principled(Base_Color=col, Roughness=nb.madd(nb.noise(pos, 6.0, detail=3), 0.08, 0.53),
                         Specular_IOR_Level=0.42, Normal=nb.bump(grain, 0.1, 0.0004))
    nb.output(bsdf)
    return m


def seat_bench(default="#b9bcbf"):
    """Extruded aluminium bench planks (mill finish, slightly dull); per-instance colour too."""
    name = "TINT_seat_bench"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    col = nb.mix(attr[3], srgb(default), attr[0])
    pos = nb.geometry()["Position"]
    brush = nb.noise(nb.vmath("MULTIPLY", pos, (2.0, 300.0, 300.0)), 4.0, detail=2)
    bsdf = nb.principled(Base_Color=col, Metallic=0.55, Roughness=nb.madd(brush, 0.1, 0.38),
                         Normal=nb.bump(brush, 0.05, 0.0005))
    nb.output(bsdf)
    return m


def seat_fabric(default="#22252b"):
    name = "TINT_seat_fabric"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    col = nb.mix(attr[3], srgb(default), attr[0])
    pos = nb.geometry()["Position"]
    weave = nb.noise(pos, 1400.0, detail=1)
    bsdf = nb.principled(Base_Color=col, Roughness=0.88, Sheen_Weight=0.5, Sheen_Roughness=0.45,
                         Specular_IOR_Level=0.3, Normal=nb.bump(weave, 0.2, 0.0005))
    nb.output(bsdf)
    return m


def _wood_texture(px=1024):
    """Weathered iroko, 1 m square, grain along u: irregular grain flow (late-wood lines at varying
    spacing, warped round a few knots), varnish worn back to silver-grey timber in patches,
    fine checks. The same tone everywhere: slats take different parts of the tile."""
    yy, xx = np.mgrid[0:px, 0:px] / px
    warp = (KD.pnoise(px, px, 3, 1, seed=11) - 0.5) * 0.25 + (KD.pnoise(px, px, 10, 2, seed=12) - 0.5) * 0.05
    rng = np.random.default_rng(7)
    knots = np.zeros_like(xx)
    for _ in range(5):
        kx, ky = rng.uniform(0, 1, 2)
        dx = (xx - kx + 0.5) % 1 - 0.5
        dy = (yy - ky + 0.5) % 1 - 0.5
        r2 = (dx / 0.05) ** 2 + (dy / 0.012) ** 2
        warp += 0.012 * np.exp(-r2) * np.sign(dy)
        knots = np.maximum(knots, np.exp(-((dx / 0.008) ** 2 + (dy / 0.005) ** 2)))
    dens = 60 + 50 * KD.pnoise(px, px, 2, 1, seed=13)
    ring = (yy + warp) * dens
    f = ring - np.floor(ring)
    late = np.clip(1 - np.abs(f - 0.7) / 0.18, 0, 1) ** 2
    fine = KD.pnoise(px, px, 300, 8, seed=14)
    patch = KD.pnoise(px, px, 5, 2, seed=15)
    brown, silver, dark = KD.to_lin("#7d6249"), KD.to_lin("#8e877b"), KD.to_lin("#3a2a1d")
    wthr = np.clip((patch - 0.32) * 1.7, 0, 1)[..., None] * 0.7
    col = brown * (1 - wthr) + silver * wthr
    col = col * (1 - 0.22 * late[..., None]) * (0.95 + 0.1 * fine[..., None])
    col = col * (1 - knots[..., None] * 0.7) + dark * knots[..., None] * 0.7
    checks = (KD.pnoise(px, px, 500, 4, seed=16) > 0.84) * 0.4
    col = col * (1 - checks[..., None] * 0.6)
    return np.clip(col, 0, 1)


def wood(name="Seat_Wood"):
    """Weathered hardwood slats: an image texture (kept by the game export), metric UVs."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    img = KD.image(name + "_img", _wood_texture())
    h = KD.pnoise(512, 512, 180, 5, seed=16) + 0.3 * KD.pnoise(512, 512, 40, 2, seed=17)
    nrm = KD.image(name + "_nrm", KD.normal_from_height(h, 1.2), non_color=True)
    return KD.image_material(name, albedo=img, normal=nrm, rough=0.8, normal_strength=0.5)


def flat(name, color, rough=0.5, metallic=0.0):
    """A plain material whose colour survives the game bake exactly (dark plastics, coatings)."""
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=rough, Metallic=metallic,
                            Normal=nb.bump(nb.noise(pos, 300.0, detail=2), 0.05, 0.0005)))
    return m


def steel(name="Seat_Steel", color="#2f3236"):
    """Powder-coated steel or cast aluminium hardware (satin)."""
    return flat(name, color, rough=0.5)


def galvanised(name="Galvanised"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    spangle = nb.voronoi(pos, 60.0, out="Distance")
    tone = nb.noise(pos, 4.0, detail=4)
    col = nb.mix(nb.madd(spangle, 0.5, nb.mul(tone, 0.4)), srgb("#8e9396"), srgb("#c4c8ca"))
    bsdf = nb.principled(Base_Color=col, Metallic=0.9, Roughness=nb.madd(tone, 0.2, 0.35))
    nb.output(bsdf)
    return m


# -- geometry helpers -------------------------------------------------------------------------

def shell(name, prof_yz, widths, dish, thickness, n_across=7, levels=2, col=None, mat=None):
    """Moulded shell: a side-view profile swept across x, dished, thickened and subdivided."""
    P = np.asarray(prof_yz, float)
    widths = np.broadcast_to(np.asarray(widths, float), (len(P),))
    tang = np.gradient(P, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    fwd = np.stack([tang[:, 1], -tang[:, 0]], 1)        # normal in side view
    fwd *= np.sign(fwd[:, :1] + 1e-9)                    # pointing towards +Y (the sitter)
    u = np.linspace(-1, 1, n_across)
    V = []
    for (y, z), w, f in zip(P, widths, fwd):
        for a in u:
            d = -dish * (1 - a * a)                      # dished: centre pushed away from the sitter
            V.append((a * w / 2, y + f[0] * d, z + f[1] * d))
    F = [(i * n_across + j, i * n_across + j + 1, (i + 1) * n_across + j + 1, (i + 1) * n_across + j)
         for i in range(len(P) - 1) for j in range(n_across - 1)]
    o = mesh_object(name, V, F, col, materials=[mat] if mat else ())
    so = o.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = thickness
    so.offset = 0.0
    so.use_even_offset = True
    sub = o.modifiers.new("Subsurf", "SUBSURF")
    sub.levels = sub.render_levels = levels
    bake(o)
    o.data.shade_smooth()
    return o


def side_plate(name, poly_yz, x, thickness, col=None, mat=None, bevel=0.002):
    """A flat profiled plate (seat standard, bracket): side-view polygon extruded in x."""
    P = np.asarray(poly_yz, float)
    K = len(P)
    V = [(x - thickness / 2, y, z) for y, z in P] + [(x + thickness / 2, y, z) for y, z in P]
    F = [tuple(range(K))[::-1], tuple(range(K, 2 * K))]
    F += [(i, (i + 1) % K, K + (i + 1) % K, K + i) for i in range(K)]
    o = mesh_object(name, V, F, col, materials=[mat] if mat else ())
    if bevel:
        bv = o.modifiers.new("Bevel", "BEVEL")
        bv.width = bevel
        bv.segments = 2
        bv.limit_method = "ANGLE"
        bake(o)
    return o


def rounded_box(name, size, centre, radius, col=None, mat=None, tilt_x=0.0, segments=3):
    sx, sy, sz = (s / 2 for s in size)
    V = [(x, y, z) for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]
    F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    o = mesh_object(name, V, F, col, materials=[mat] if mat else ())
    bv = o.modifiers.new("Bevel", "BEVEL")
    bv.width = radius
    bv.segments = segments
    bake(o)
    o.data.shade_smooth()
    o.rotation_euler = (tilt_x, 0, 0)
    o.location = centre
    _apply_transform(o)
    return o


def tube(name, pts, r, col=None, mat=None, sides=12):
    V, F, UV = M.sweep(M.densify(np.asarray(pts, float), 0.2), M.profile_ellipse(2 * r, 2 * r, sides),
                       (0, 0, 1) if abs(pts[-1][2] - pts[0][2]) < 1e-6 else (1, 0, 0))
    return mesh_object(name, V, F, col, smooth=True, materials=[mat] if mat else (), uvs=UV)


def _apply_transform(o):
    mw = o.matrix_basis.copy()
    o.data.transform(mw)
    o.matrix_basis.identity()


def _rotate_about_x(o, pivot, angle):
    from mathutils import Matrix, Vector
    T = Matrix.Translation(Vector(pivot)) @ Matrix.Rotation(angle, 4, "X") @ Matrix.Translation(-Vector(pivot))
    o.data.transform(T)


# -- detail helpers ---------------------------------------------------------------------------

GROUT = 0.032      # seats bolt down through a grout bed / cast boss that stands proud of the floor


def _pm(r=0.004):
    from .props import PM
    return PM(r)


def _ccw(poly):
    """A (y, z) or (x, z) polygon wound counter-clockwise (what PM.extrude expects)."""
    P = np.asarray(poly, float)
    a = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return [tuple(p) for p in (P if a > 0 else P[::-1])]


def _bolt(p, c, n, r, mat, h=0.005, dome=False, washer=None):
    """A hex bolt head (or a domed carriage-bolt head) on a face with outward normal n."""
    r0, p.r = p.r, 0.0
    c, n = np.asarray(c, float), np.asarray(n, float)
    if washer:
        p.cyl(c - n * 0.001, c + n * 0.0018, r * 1.45, washer, sides=10)
        c = c + n * 0.0018
    if dome:
        p.cyl(c - n * 0.001, c + n * h * 0.5, r, mat, sides=10, r2=r * 0.92)
        p.cyl(c + n * h * 0.5, c + n * h, r * 0.92, mat, sides=10, r2=r * 0.45)
    else:
        p.cyl(c - n * 0.001, c + n * h, r, mat, sides=6)
    p.r = r0


def grout():
    return flat("Seat_Grout", "#8d8a84", rough=0.92)


def _base_plate(p, x0, x1, y0, y1, mat, bolt_mat, t=0.01, bolts=4, inset=0.022, bed=True):
    """A steel base plate on a grout bed (the bed stands proud of the floor), anchor bolts with
    nuts and washers actually seated on the plate. Returns the plate's top height."""
    z0 = GROUT if bed else 0.0
    if bed:
        r0, p.r = p.r, 0.006
        p.box((x0 - 0.014, y0 - 0.014, -0.01), (x1 + 0.014, y1 + 0.014, z0), grout())
        p.r = r0
    r0, p.r = p.r, 0.003
    p.box((x0, y0, z0), (x1, y1, z0 + t), mat)
    p.r = r0
    zt = z0 + t
    pts = [(x0 + inset, y0 + inset), (x1 - inset, y0 + inset), (x1 - inset, y1 - inset), (x0 + inset, y1 - inset)]
    if bolts == 2:
        pts = [((x0 + x1) / 2, y0 + inset), ((x0 + x1) / 2, y1 - inset)]
    for bx, by in pts:
        _bolt(p, (bx, by, zt), (0, 0, 1), 0.0095, bolt_mat, h=0.008, washer=bolt_mat)
        r0, p.r = p.r, 0.0
        p.cyl((bx, by, zt + 0.0098), (bx, by, zt + 0.017), 0.0055, bolt_mat, sides=8)   # stud end
        p.r = r0
    return zt


def _rib(p, sh, xn, s, back, h, w, mat):
    """A stiffening rib on the back face of a shell: along the profile (xn fixed) or across it."""
    xn, s = np.broadcast_arrays(np.asarray(xn, float), np.asarray(s, float))
    t = np.linspace(0, 1, 20)
    k = np.linspace(0, 1, len(s))
    xn, s = np.interp(t, k, xn), np.interp(t, k, s)        # follow the curved shell closely
    w = min(w, 0.0045)                                       # a moulded web, 3-4 mm thick
    pts = sh.point(xn, s, back=back + h / 2 - 0.0025)      # sunk into the wall: fused
    T = pts[-1] - pts[0]
    T /= np.linalg.norm(T)
    n = sh.sitter(float(np.mean(s)))
    ref = np.cross(T, n)
    prof = [(-h / 2, -w / 2), (h / 2, -w / 2), (h / 2, w / 2), (-h / 2, w / 2)]
    r0, p.r = p.r, 0.0
    p.sweep(pts, prof, mat, twist_ref=tuple(ref))
    p.r = r0


def _surface_frame(sh, xn, s, back=0.0):
    """Centre, outward normal (towards the sitter) and up direction (along the profile)."""
    c = sh.point(xn, s, back=back)
    a = sh.point(xn, min(s + 0.02, 1.0), back=back)
    b = sh.point(xn, max(s - 0.02, 0.0), back=back)
    v = (a - b) / np.linalg.norm(a - b)
    n = sh.sitter(s)
    n = n - v * (n @ v)
    return c, n / np.linalg.norm(n), v


def _number_plate(name, text, fg="#f2f2ee", bg="#1d1e20", px=(128, 64)):
    return KD.image_material(name, albedo=KD.plate_image(name + "_img", text, fg=fg, bg=bg, px=px, border=3),
                             rough=0.35, metallic=0.4, extension="EXTEND")


def _plaque(p, c, n, v, w, h, mat, rim_mat, rivets=True):
    """A slightly proud aluminium plaque (2 mm, bevelled) with its print, and two rivets."""
    c, n, v = (np.asarray(x, float) for x in (c, n, v))
    u = np.cross(v, n)
    r0, p.r = p.r, 0.0008
    q = [c + n * 0.0005 + u * a * (w / 2 + 0.003) + v * b * (h / 2 + 0.003) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    top = [x + n * 0.002 for x in q]
    p.faces(q + top, [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)], rim_mat)
    p.r = r0
    p.decal(c + n * 0.0025, n, v, w, h, mat, lift=0.0003)
    if rivets:
        for a in (-1, 1):
            _bolt(p, c + n * 0.0025 + u * a * (w / 2 + 0.0015), n, 0.0018, rim_mat, h=0.0012, dome=True)


def _hinge_black():
    return flat("Seat_Hinge", "#1a1b1c", rough=0.55)


def _galv():
    """Hot-dip galvanised steel: mottled spangle grey (an image, so the game keeps it)."""
    name = "Seat_Galv"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    n = 512
    sp = KD.pnoise(n, n, 6, 6, seed=21)             # soft spangle patches (0.5-1 m in a 1 m tile)
    fine = KD.pnoise(n, n, 40, 40, seed=22)
    v = 0.42 + 0.08 * (sp - 0.5) * 2 + 0.025 * (fine - 0.5) * 2
    col = np.stack([v * 0.98, v, v * 1.02], -1)
    return KD.image_material(name, albedo=KD.image(name + "_img", np.clip(col, 0, 1)), rough=0.6, metallic=0.5)


def _fold(objs, pivot, angle):
    for o in objs:
        _rotate_about_x(o, pivot, angle)


def _wear(obj, style):
    """Wear in the vertex colours (multiplied into whatever colour an instance takes): dust and
    dirt low down, sun-faded upward faces against shaded undersides, grime in the dish."""
    def floor_dirt(P, N):
        g = np.clip(1 - (P[:, 2] - GROUT) / 0.12, 0, 1) ** 1.5
        return 1 - g[:, None] * (1 - np.array([0.72, 0.68, 0.6]))
    KD.tint_vertices(obj, floor_dirt)

    def fade(P, N):
        up = np.clip(N[:, 2], 0, 1)
        f = 0.86 + 0.14 * up
        return np.repeat(f[:, None], 3, 1)
    KD.tint_vertices(obj, fade, materials={"TINT_seat", "TINT_seat_bench", "TINT_seat_fabric"})


# -- styles -----------------------------------------------------------------------------------

def _back_depth(R=0.55, hw0=0.212, hw1=0.012, lumbar=0.022):
    def depth(xn, s):
        x = xn * (hw0 + hw1 * s)
        return (x * x / (2 * R) + lumbar * np.exp(-((s - 0.3) / 0.17) ** 2) * (1 - 0.4 * xn ** 2)
                - 0.01 * np.clip((s - 0.8) / 0.2, 0, 1) ** 2)
    return depth


def _pan(col, mat, bracket_mat, hinge, hw=0.21, length=0.39, lift=0.035, name="pan", lug_x=0.165):
    """A tip-up seat pan in its open position (hinge at its rear): dished, a rolled waterfall
    front, a 3 x 3 grid of moulded ribs underneath, drain holes at its lowest point, and two slim
    pressed-steel hinge brackets with their bolts. Returns (shell object, detail object)."""
    hy, hz = hinge
    z = hz + lift
    y0 = hy + 0.02
    prof = [(y0 + length, z - 0.026), (y0 + length - 0.014, z + 0.002), (y0 + length - 0.08, z + 0.007),
            (y0 + 0.2, z + 0.002), (y0 + 0.09, z + 0.003), (y0 + 0.015, z + 0.01)]
    sh = KD.Shell(prof, lambda s: hw + 0.008 * (1 - s),
                  lambda xn, s: -0.016 * (1 - xn ** 2) * np.sin(np.pi * np.clip(s, 0, 1)) ** 0.7 + 0.008 * xn ** 4)
    o = sh.build(name, mat, col, nu=10, nv=9, rim=0.012, flange=0.016, thick=0.006)
    p = _pm(0.002)
    for xn in (-0.58, 0.0, 0.58):
        _rib(p, sh, xn, np.linspace(0.1, 0.9, 7), 0.006, 0.014, 0.008, mat)
    for s in (0.36, 0.68):
        _rib(p, sh, np.linspace(-0.86, 0.86, 7), s, 0.006, 0.013, 0.008, mat)
    p.r = 0.003
    for side in (-1, 1):
        x = side * lug_x
        p.box((x - 0.003, hy - 0.028, hz - 0.024), (x + 0.003, hy + 0.06, z - 0.004), bracket_mat)
        p.cyl((x - 0.003, hy, hz), (x + 0.003, hy, hz), 0.024, bracket_mat, sides=14)
        for dy in (0.018, 0.046):
            _bolt(p, (x + side * 0.003, hy + dy, z - 0.016), (side, 0, 0), 0.0055, galvanised(), h=0.004)
    hole = flat("Seat_Hole", "#050505", rough=0.9)
    for xn in (-0.22, 0.22):
        c, n, _ = _surface_frame(sh, xn, 0.48)
        p.r = 0.0
        p.cyl(c - n * 0.0075, c + n * 0.0006, 0.0065, hole, sides=10)
    d = p.obj(name + "_detail", col)
    return o, d


def _hinge_axle(p, hinge, reach, cover, black, steel_m):
    """The axle through the pan brackets: a slim sprung cover (30 mm) either side of the mount and
    a nut and washer at each end."""
    p.r = 0.0
    p.cyl((-reach, *hinge), (reach, *hinge), 0.009, galvanised(), sides=10)
    for a, b in cover:
        p.cyl((a, *hinge), (b, *hinge), 0.015, black, sides=14)
    for side in (-1, 1):
        x = side * reach
        p.cyl((x - side * 0.004, *hinge), (x, *hinge), 0.014, galvanised(), sides=12)
        p.cyl((x, *hinge), (x + side * 0.006, *hinge), 0.011, steel_m, sides=6)


def build_tipup_classic(variant, col, width=0.5):
    """Moulded polypropylene tip-up on a central cast pedestal: the back on a steel spine, the pan
    on slim hinge brackets, folded up and parked parallel to the back as empty seats stand; the
    end of a row adds an aisle end frame with the row letter."""
    mat = seat_plastic()
    hw_mat = flat("Seat_Pedestal", "#3c3e41", rough=0.5)
    black = _hinge_black()
    hinge = (0.212, 0.42)
    prof = [(0.122, 0.47), (0.098, 0.60), (0.070, 0.73), (0.040, 0.862)]
    sh = KD.Shell(prof, lambda s: 0.212 + 0.012 * s, _back_depth())
    back = sh.build("back", mat, col, nu=10, nv=10, rim=0.012, flange=0.018, thick=0.006)
    pan, pan_d = _pan(col, mat, hw_mat, hinge)
    _fold([pan, pan_d], (0, *hinge), math.radians(100))
    p = _pm(0.004)
    # rear of the back: a 3 x 3 rib grid, the steel spine bolted up its centre, the seat number
    for xn in (-0.6, 0.6):
        _rib(p, sh, xn, np.linspace(0.06, 0.92, 7), 0.006, 0.014, 0.008, mat)
    for s in (0.35, 0.66):
        _rib(p, sh, np.linspace(-0.86, 0.86, 7), s, 0.006, 0.012, 0.008, mat)
    spine = sh.point(np.zeros(8), np.linspace(0.02, 0.7, 8), back=0.006 + 0.004)
    p.r = 0.0
    p.sweep(spine, [(-0.004, -0.022), (0.004, -0.022), (0.004, 0.022), (-0.004, 0.022)], hw_mat, twist_ref=(1, 0, 0))
    for s in (0.2, 0.6):
        c, n, _ = _surface_frame(sh, 0.0, s, back=0.006 + 0.008)
        _bolt(p, c, -n, 0.007, galvanised(), h=0.005, dome=True)
    c, n, v = _surface_frame(sh, 0.0, 0.86, back=0.006)
    _plaque(p, c, -n, v, 0.06, 0.03, _number_plate("Seat_No_Tipup", "27"), flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6))
    # cast pedestal on a bolted floor plate (on its grout bed) with gussets
    p.r = 0.006
    ped = [(0.15, 0.03), (0.33, 0.03), (0.33, 0.046), (0.29, 0.056), (0.272, 0.30), (0.272, 0.385), (0.26, 0.425),
           (0.225, 0.442), (0.15, 0.478), (0.105, 0.505), (0.088, 0.49), (0.13, 0.452), (0.205, 0.40), (0.218, 0.30),
           (0.215, 0.056), (0.15, 0.046)]
    p.extrude_yz(_ccw(ped), -0.026, 0.026, hw_mat)
    zt = _base_plate(p, -0.08, 0.08, 0.15, 0.35, hw_mat, galvanised())
    p.r = 0.002
    for side in (-1, 1):
        p.extrude(_ccw([(side * 0.025, zt), (side * 0.072, zt), (side * 0.025, zt + 0.075)]), 0.241, 0.247, hw_mat)
    # moulded hinge beam from the pedestal head to the pan's two hinge bosses (spring inside)
    p.r = 0.002
    p.cyl((-0.16, *hinge), (0.16, *hinge), 0.016, black, sides=14)
    for side in (-1, 1):
        p.cyl((side * 0.16, *hinge), (side * 0.198, *hinge), 0.0175, black, sides=14)
    p.r = 0.0
    hole = flat("Seat_Hole", "#050505", rough=0.9)
    for xs in (-0.0265, 0.0265):                                # lightening hole in the cast web
        p.decal((xs, 0.243, 0.24), (1 if xs > 0 else -1, 0, 0), (0, 0, 1), 0.026, 0.1, hole, lift=0.0003)
    if variant == "end":
        # aisle end frame: a cast standard with the row letter
        p.r = 0.006
        endp = [(0.10, 0.0), (0.40, 0.0), (0.40, 0.045), (0.33, 0.055), (0.30, 0.50), (0.245, 0.60), (0.15, 0.62),
                (0.12, 0.58), (0.20, 0.055), (0.10, 0.045)]
        p.extrude_yz(_ccw(endp), 0.236, 0.258, hw_mat)
        _plaque(p, (0.258, 0.215, 0.53), (1, 0, 0), (0, 0, 1), 0.06, 0.06,
                KD.image_material("Seat_Row_K", albedo=KD.plate_image("Seat_Row_K_img", "K", px=(96, 96), border=4),
                                  rough=0.35, metallic=0.4, extension="EXTEND"), flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6))
        for y in (0.13, 0.37):
            _bolt(p, (0.247, y, 0.045), (0, 0, 1), 0.008, galvanised(), h=0.006)
    det = p.obj("hardware", col)
    return [back, pan, pan_d, det]


def build_bucket_fixed(variant, col, width=0.48):
    """One-piece moulded bucket seat (rolled front and top lips, perimeter rib, three rear ribs
    running down under the pan) on two folded galvanised feet bolted to the tread."""
    mat = seat_plastic()
    galv = _galv()
    prof = [(0.468, 0.44), (0.455, 0.447), (0.39, 0.448), (0.30, 0.438), (0.21, 0.435), (0.14, 0.455),
            (0.105, 0.52), (0.083, 0.62), (0.06, 0.74), (0.036, 0.865)]
    hw = lambda s: 0.215 - 0.012 * np.clip((s - 0.45) / 0.55, 0, 1)
    depth = lambda xn, s: (0.03 * np.abs(xn) ** 3 * (1 - 0.4 * s) + 0.016 * np.exp(-((s - 0.68) / 0.12) ** 2)
                           - 0.012 * (1 - xn ** 2) * np.exp(-((s - 0.22) / 0.15) ** 2))
    sh = KD.Shell(prof, hw, depth)
    shell_o = sh.build("bucket", mat, col, nu=10, nv=14, rim=0.014, flange=0.016, thick=0.006)
    p = _pm(0.004)
    for xn in (-0.5, 0.0, 0.5):
        _rib(p, sh, xn, np.linspace(0.6, 0.95, 8), 0.006, 0.014, 0.008, mat)
    hole = flat("Seat_Hole", "#050505", rough=0.9)
    for xn in (-0.22, 0.22):
        c, n, _ = _surface_frame(sh, xn, 0.27)
        p.r = 0.0
        p.cyl(c - n * 0.0075, c + n * 0.0006, 0.0065, hole, sides=10)
    c, n, v = _surface_frame(sh, 0.0, 0.92)
    _plaque(p, c, n, v, 0.062, 0.03, _number_plate("Seat_No_Bucket", "112"), flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6))
    # perimeter rib round the back and its rim, moulded in
    for xn in (-0.9, 0.9):
        _rib(p, sh, xn, np.linspace(0.6, 0.97, 8), 0.006, 0.012, 0.004, mat)
    _rib(p, sh, np.linspace(-0.9, 0.9, 12), 0.97, 0.006, 0.012, 0.004, mat)
    # mounting: two folded steel feet bolted to the tread, a box beam under the pan, bosses
    for side in (-1, 1):
        x = side * 0.12
        zt = _base_plate(p, x - 0.035, x + 0.035, 0.18, 0.34, galv, galvanised(), t=0.008, bolts=2, inset=0.02)
        p.r = 0.003
        p.extrude_yz(_ccw([(0.2, zt), (0.32, zt), (0.32, zt + 0.012), (0.285, zt + 0.02), (0.275, 0.392),
                           (0.235, 0.392), (0.228, zt + 0.02), (0.2, zt + 0.012)]), x - 0.004, x + 0.004, galv)
    p.r = 0.004
    p.box((-0.15, 0.225, 0.392), (0.15, 0.285, 0.412), galv)
    for x in (-0.07, 0.07):
        for y in (0.235, 0.275):
            p.cyl((x, y, 0.412), (x, y, 0.421), 0.011, mat, sides=10)
            _bolt(p, (x, y, 0.392), (0, 0, -1), 0.007, galvanised(), h=0.005)
    det = p.obj("hardware", col)
    return [shell_o, det]


def _slat_poly(yc, ztop, w=0.086, t=0.032, r=0.007):
    """Extruded aluminium seat plank in section: rounded corners, two shallow grooves on top."""
    hw = w / 2
    pts = []
    for cy, cz, a0 in ((yc + hw - r, ztop - t + r, -90), (yc + hw - r, ztop - r, 0)):
        pts += [(cy + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))) for a in np.linspace(a0, a0 + 90, 4)]
    for g in (0.014, -0.014):
        gy = yc + g
        pts += [(gy + 0.0025, ztop), (gy + 0.0017, ztop - 0.002), (gy - 0.0017, ztop - 0.002), (gy - 0.0025, ztop)]
    for cy, cz, a0 in ((yc - hw + r, ztop - r, 90), (yc - hw + r, ztop - t + r, 180)):
        pts += [(cy + r * math.cos(math.radians(a)), cz + r * math.sin(math.radians(a))) for a in np.linspace(a0, a0 + 90, 4)]
    return _ccw(pts)


BENCH_SLATS = (0.13, 0.245, 0.36)
BENCH_TOP = 0.44


def _saddle(p, yc, mat, bolt_m):
    """A bent-plate saddle cupping one plank on the bracket arm, a bolt up through it."""
    r0, p.r = p.r, 0.0015
    z = BENCH_TOP - 0.03
    p.box((-0.032, yc - 0.05, z - 0.006), (0.032, yc + 0.05, z), mat)
    for sy in (-1, 1):
        p.box((-0.032, yc + sy * 0.05 - (0.003 if sy > 0 else 0.0), z - 0.006),
              (0.032, yc + sy * 0.05 + (0.0 if sy > 0 else 0.003), z + 0.012), mat)
    p.r = r0
    _bolt(p, (0, yc, z - 0.006), (0, 0, -1), 0.0075, bolt_m, h=0.006, washer=bolt_m)


def build_bench(variant, col, width=0.45):
    """Terrace bench bay: three extruded aluminium planks (305 mm deep together, top at 440 mm;
    instanced seat by seat they make one continuous plank run along the row), each plank cupped in
    a saddle on a rear-mounted cantilever arm. plank: the intermediate support, a cast aluminium
    leg rising at the back of the tread. bracket (every fourth seat and the row end): a
    galvanised folded-channel cantilever bracket with a gusset, and black end caps where the
    plank lengths join."""
    mat = seat_bench()
    p = _pm(0.0)
    capm = flat("Bench_Cap", "#141414", rough=0.6)
    bracket = variant == "bracket"
    e = 0.009 if bracket else 0.0
    for yc in BENCH_SLATS:
        p.extrude_yz(_slat_poly(yc, BENCH_TOP, w=0.103, t=0.034, r=0.012), -width / 2 + e, width / 2 - e, mat)
        if bracket:
            for x0, x1 in ((-width / 2, -width / 2 + e), (width / 2 - e, width / 2)):
                p.extrude_yz(_slat_poly(yc, BENCH_TOP + 0.0005, w=0.104, t=0.035, r=0.012), x0, x1, capm)
    screw = galvanised()
    for yc in BENCH_SLATS:
        for x in (-0.012, 0.012):
            p.cyl((x, yc, BENCH_TOP - 0.0005), (x, yc, BENCH_TOP + 0.0012), 0.005, screw, sides=8, r2=0.0038)
    _plaque(p, (0.0, BENCH_SLATS[-1] + 0.0515, BENCH_TOP - 0.017), (0, 1, 0), (0, 0, 1), 0.05, 0.02,
            _number_plate("Seat_No_Bench", "14", px=(128, 56)), flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6), rivets=True)
    frame = _galv() if bracket else flat("Bench_Leg", "#9a9ea2", rough=0.45, metallic=0.4)
    for yc in BENCH_SLATS:
        _saddle(p, yc, frame, galvanised())
    za = BENCH_TOP - 0.036                                      # underside of the saddles
    if bracket:
        zt = _base_plate(p, -0.075, 0.075, 0.02, 0.2, frame, galvanised())
        p.r = 0.003
        # folded channel: upright at the back of the tread, arm forward under the planks
        for x in (-0.03, 0.026):
            p.box((x, 0.06, zt), (x + 0.004, 0.13, za), frame)
            p.box((x, 0.06, za - 0.05), (x + 0.004, 0.42, za), frame)
        p.box((-0.03, 0.06, zt), (0.03, 0.064, za), frame)
        p.box((-0.03, 0.06, za - 0.004), (0.03, 0.42, za), frame)
        p.r = 0.0015
        p.extrude_yz(_ccw([(0.13, za - 0.26), (0.13, za - 0.05), (0.36, za - 0.05)]), -0.003, 0.003, frame)   # gusset
        for z in (zt + 0.06, zt + 0.2):
            _bolt(p, (0.03, 0.095, z), (1, 0, 0), 0.0065, galvanised(), h=0.004)
    else:
        zt = _base_plate(p, -0.06, 0.06, 0.03, 0.21, frame, galvanised(), bolts=2, inset=0.025)
        p.r = 0.005
        leg = [(0.035, zt), (0.205, zt), (0.205, zt + 0.014), (0.13, zt + 0.03), (0.105, za - 0.07), (0.17, za - 0.03),
               (0.42, za - 0.024), (0.42, za), (0.07, za), (0.065, za - 0.06), (0.075, zt + 0.03), (0.035, zt + 0.014)]
        p.extrude_yz(_ccw(leg), -0.014, 0.014, frame)
    return [p.obj("bench", col, smooth_deg=40)]


def _vip_fabric():
    """Charcoal woven upholstery with a fine weave normal (image, kept by the game export)."""
    name = "TINT_seat_fabric"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    n = 256
    yy, xx = np.mgrid[0:n, 0:n] / n
    k = 64                                   # 64 threads per 64 mm tile -> 1 mm yarn
    wx = np.sin(2 * np.pi * k * xx)
    wy = np.sin(2 * np.pi * k * yy)
    h = 0.5 + 0.25 * wx * np.sign(np.sin(np.pi * k * yy)) + 0.25 * wy * np.sign(np.sin(np.pi * k * xx + 1.0))
    h += 0.15 * (KD.pnoise(n, n, 20, 20, seed=31) - 0.5)
    nrm = KD.image(name + "_nrm", KD.normal_from_height(h, 1.5), non_color=True)
    tone = 0.9 + 0.2 * KD.pnoise(n, n, 6, 6, seed=32)
    base = KD.to_lin("#4a4c51")
    alb = KD.image(name + "_img", np.clip(base * tone[..., None], 0, 1))
    m = KD.image_material(name, albedo=alb, normal=nrm, rough=0.72, normal_strength=0.35, uv_scale=16.0)
    return m


def build_padded_vip(variant, col, width=0.56):
    """Hospitality seat: an upholstered back (lumbar, headrest roll, a welt seam at two thirds,
    perimeter piping) seated in a satin moulded back shell with a lip round it, a domed tip-up
    cushion with a rounded waterfall front and a dark under-shell on steel hinge brackets, cast
    side standards with PU armrest pads, number plaques front and back."""
    fab = _vip_fabric()
    shell_m = flat("VIP_Shell", "#303134", rough=0.45)
    frame = flat("VIP_Frame", "#1f2022", rough=0.4)
    pad_m = flat("VIP_Armrest", "#2b2c2e", rough=0.62)
    plaque_m = flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6)
    prof = [(0.20, 0.50), (0.17, 0.64), (0.13, 0.80), (0.095, 0.97)]

    def pad_depth(xn, s):
        x = xn * 0.215
        return (x * x / 1.8 + 0.026 * np.exp(-((s - 0.28) / 0.16) ** 2)
                + 0.018 * np.exp(-((s - 0.9) / 0.08) ** 2) * (1 - 0.5 * xn ** 2))
    sh = KD.Shell(prof, lambda s: 0.215 + 0 * s, pad_depth)
    back = sh.build("backpad", fab, col, nu=12, nv=12, rim=0.028, flange=0.032, thick=0.003)
    # the back shell, wrapped round the cushion from behind (walked top -> bottom so its lip
    # turns forward round the cushion's edges)
    rear_prof = [(y - 0.042, z - 0.008) for y, z in prof][::-1]
    rsh = KD.Shell(rear_prof, lambda s: 0.243 + 0 * s, lambda xn, s: -(xn * 0.243) ** 2 / 1.8)
    rear = rsh.build("backshell", shell_m, col, nu=8, nv=8, rim=0.012, flange=0.046, thick=0.004)
    # cushion: a crowned pad 70 mm thick at the centre, 45-50 mm at the edges, rolled waterfall front
    cprof = [(0.612, 0.405), (0.606, 0.44), (0.585, 0.458), (0.53, 0.466), (0.40, 0.463), (0.28, 0.46), (0.215, 0.462)]
    csh = KD.Shell(cprof, lambda s: 0.208 + 0 * s,
                   lambda xn, s: 0.02 * (1 - xn ** 2) ** 0.7 * np.sin(np.pi * np.clip(s * 0.9 + 0.05, 0, 1)) ** 0.6)
    cush = csh.build("cushion", fab, col, nu=12, nv=11, rim=0.03, flange=0.022, thick=0.003)
    ush = KD.Shell([(y - 0.012, z - 0.052) for y, z in cprof[2:]][::-1], lambda s: 0.226 + 0 * s, lambda xn, s: 0 * xn)
    under = ush.build("cushionpan", shell_m, col, nu=6, nv=6, rim=0.01, flange=0.016, thick=0.004)
    p = _pm(0.005)
    # piping round the back cushion and a welt seam across it
    loop = sh.outline(72, out=0.027, back=0.027)
    p.r = 0.0
    p.tube(list(loop) + [loop[0]], 0.0042, fab, sides=6)
    welt = sh.point(np.linspace(-0.93, 0.93, 16), np.full(16, 0.64), back=-0.0015)
    p.tube(list(welt), 0.0035, fab, sides=6)
    for xn in (-0.36, 0.36):                    # two vertical channel seams below the welt
        seam = sh.point(np.full(12, xn), np.linspace(0.04, 0.62, 12), back=-0.001)
        p.tube(list(seam), 0.003, fab, sides=6)
    cl = csh.outline(64, out=0.03, back=0.028)  # piping round the cushion's top edge
    p.tube(list(cl) + [cl[0]], 0.0042, fab, sides=6)
    c, n, v = _surface_frame(rsh, 0.0, 0.14)
    _plaque(p, c, n, v, 0.06, 0.03, _number_plate("Seat_No_VIP", "8", fg="#ffffff", bg="#151515"), plaque_m)
    xs = 0.255
    poly = [(0.08, 0.0), (0.42, 0.0), (0.42, 0.036), (0.28, 0.046), (0.22, 0.30), (0.215, 0.50), (0.30, 0.552),
            (0.585, 0.562), (0.59, 0.585), (0.12, 0.585), (0.115, 0.50), (0.15, 0.30), (0.16, 0.046), (0.08, 0.036)]
    plate = _number_plate("Seat_No_VIP", "8", fg="#ffffff", bg="#151515")
    for side in (-1, 1):
        x = side * xs
        p.r = 0.005
        p.extrude_yz(_ccw(poly), x - 0.02, x + 0.02, frame)
        p.r = 0.012
        p.box((x - 0.03, 0.112, 0.58), (x + 0.03, 0.604, 0.628), pad_m)         # one moulded soft-touch cap
        p.r = 0.0
        p.box((x - 0.022, 0.6025, 0.592), (x + 0.022, 0.6045, 0.616), frame)     # recess round the plate
        p.decal((x, 0.6045, 0.604), (0, 1, 0), (0, 0, 1), 0.04, 0.022, plate, lift=0.0004)
        for y in (0.12, 0.38):
            _bolt(p, (x, y, 0.036), (0, 0, 1), 0.008, galvanised(), h=0.006, washer=galvanised())
        p.r = 0.002
        p.cyl((side * 0.214, 0.27, 0.388), (side * 0.236, 0.27, 0.388), 0.03, frame, sides=16)   # pivot hub
        p.box((min(side * 0.21, side * 0.236), 0.105, 0.52), (max(side * 0.21, side * 0.236), 0.16, 0.575), frame)
    p.r = 0.0
    p.cyl((-0.214, 0.27, 0.388), (0.214, 0.27, 0.388), 0.013, frame, sides=12)
    p.r = 0.002
    p.r = 0.004
    p.cyl((0, 0.27, 0.388), (0, 0.38, 0.385), 0.016, frame, sides=12)          # return-spring housing
    p.r = 0.002
    for x in (-0.13, 0.13):                     # hinge brackets from the cushion pan to the pivot
        p.box((x - 0.003, 0.25, 0.375), (x + 0.003, 0.40, 0.408), frame)
        for y in (0.3, 0.37):
            _bolt(p, (x + 0.003, y, 0.395), (1, 0, 0), 0.005, galvanised(), h=0.003)
    if variant == "end":
        _plaque(p, (xs + 0.0205, 0.20, 0.45), (1, 0, 0), (0, 0, 1), 0.06, 0.06,
                KD.image_material("Seat_Row_C", albedo=KD.plate_image("Seat_Row_C_img", "C", fg="#ffffff", bg="#151515",
                                                                      px=(96, 96), border=4),
                                  rough=0.35, metallic=0.4, extension="EXTEND"), plaque_m)
    return [back, rear, cush, under, p.obj("frame", col)]


def build_rail_seat(variant, col, width=0.5):
    """Safe-standing rail seat: the seat's own 60 mm post (80 mm on a support post, every other
    seat and the row end) bolted down through a four-bolt plate, carrying the 48.3 mm barrier
    rail at 1.1 m in a clamp collar; the rail spans the whole bay and ends in a joint sleeve, so
    a row of seats makes one unbroken rail. A moulded back pad sits just under the rail; the pan,
    folded up and latched for standing, stows against it on two moulded hinge housings."""
    mat = seat_plastic()
    rail_m = flat("Rail_Steel", "#5a5e62", rough=0.5)
    black = _hinge_black()
    heavy = variant == "post"
    px = 0.04 if heavy else 0.03
    yc = 0.075
    hinge = (0.258, 0.43)
    prof = [(0.165, 0.73), (0.158, 0.86), (0.15, 1.0)]
    sh = KD.Shell(prof, lambda s: 0.21 + 0.005 * s, _back_depth(hw0=0.21, hw1=0.005, lumbar=0.012))
    back = sh.build("back", mat, col, nu=10, nv=8, rim=0.013, flange=0.018, thick=0.007)
    pan, pan_d = _pan(col, mat, rail_m, hinge, length=0.37, lug_x=0.055)
    _fold([pan, pan_d], (0, *hinge), math.radians(93))
    p = _pm(0.004)
    for xn in (-0.55, 0.0, 0.55):
        _rib(p, sh, xn, np.linspace(0.06, 0.94, 6), 0.007, 0.014, 0.004, mat)
    _rib(p, sh, np.linspace(-0.88, 0.88, 9), 0.5, 0.007, 0.012, 0.004, mat)
    c, n, v = _surface_frame(sh, 0.0, 0.8)
    _plaque(p, c, n, v, 0.06, 0.03, _number_plate("Seat_No_Rail", "43"), flat("Seat_Plaque", "#b9bcbe", 0.35, 0.6))
    # post on its base plate, welded cap, clamp collar round the rail
    if heavy:
        zt = _base_plate(p, -0.075, 0.075, yc - 0.1, yc + 0.125, rail_m, galvanised(), t=0.012)
    else:
        zt = _base_plate(p, -0.06, 0.06, yc - 0.065, yc + 0.125, rail_m, galvanised())
    p.r = 0.003
    p.box((-px, yc - px, zt), (px, yc + px, 1.06), rail_m)
    p.r = 0.002
    for s_ in (-1, 1):                                                   # welded foot gussets
        p.extrude(_ccw([(s_ * px, zt), (s_ * (px + 0.045), zt), (s_ * px, zt + 0.07)]), yc - 0.003, yc + 0.003, rail_m)
    p.r = 0.003
    p.cyl((-0.045, yc, 1.1), (0.045, yc, 1.1), 0.033, rail_m, sides=18)     # clamp collar
    p.box((-0.03, yc - 0.012, 1.06), (0.03, yc + 0.012, 1.08), rail_m)       # collar web to the post
    for x in (-0.025, 0.025):
        _bolt(p, (x, yc, 1.133), (0, 0, 1), 0.006, galvanised(), h=0.004)
    p.r = 0.0
    p.cyl((-width / 2, yc, 1.1), (width / 2, yc, 1.1), 0.0242, rail_m, sides=18, cap=False)
    p.cyl((width / 2 - 0.07, yc, 1.1), (width / 2, yc, 1.1), 0.0272, rail_m, sides=18, cap=False)   # joint sleeve
    for x in (width / 2 - 0.055, width / 2 - 0.015):
        _bolt(p, (x, yc, 1.1272), (0, 0, 1), 0.0045, galvanised(), h=0.003)
    # back pad brackets from the post, hinge cheeks and housings, the latch
    p.r = 0.003
    for z in (0.8, 0.95):
        y_back = float(sh.point(0.0, (z - 0.73) / 0.27, back=0.012)[1])
        p.box((-0.022, yc + px - 0.002, z - 0.018), (0.022, y_back, z + 0.018), rail_m)
    for side in (-1, 1):
        x = side * (px + 0.005)
        p.extrude_yz(_ccw([(yc + px, hinge[1] - 0.04), (hinge[0] - 0.02, hinge[1] - 0.025),
                           (hinge[0] + 0.015, hinge[1] - 0.012), (hinge[0] + 0.015, hinge[1] + 0.02),
                           (yc + px, hinge[1] + 0.03)]), x - 0.003, x + 0.003, rail_m)
        p.r = 0.008
        xx = side * 0.055
        p.box((xx - 0.012, hinge[0] - 0.03, hinge[1] - 0.03), (xx + 0.012, hinge[0] + 0.035, hinge[1] + 0.03), black)
    p.r = 0.0
    p.cyl((-0.07, *hinge), (0.07, *hinge), 0.008, galvanised(), sides=8, cap=False)     # spindle (hidden)
    p.r = 0.003
    p.box((-0.02, yc + px - 0.002, 0.70), (0.02, 0.205, 0.725), black)                    # latch
    return [back, pan, pan_d, p.obj("frame", col)]


def build_wooden_tipup(variant, col, width=0.5):
    """Heritage tip-up: a pair of cast-iron standards (tapered, filleted, a raised centre rib, an
    armrest stub, bolted feet), a hinged seat of four hardwood slats on steel bearers (tipped up,
    parked clear of the back) and a raked back of three evenly spaced slats; carriage bolts over
    the frame members, a brass seat number."""
    w = wood()
    iron = steel("Cast_Iron", "#1f2a23")
    bolt_m = steel("Seat_Bolt", "#3a3a36")
    stain = flat("Seat_RustStain", "#4a3424", rough=0.8)
    brass = KD.image_material("Seat_No_Brass", albedo=KD.plate_image("Seat_No_Brass_img", "23", fg="#2a1e10",
                                                                     bg="#b8913f", px=(128, 64), border=3),
                              rough=0.25, metallic=0.85, extension="EXTEND")
    hinge = (0.275, 0.405)
    std = [(0.05, 0.0), (0.41, 0.0), (0.41, 0.03), (0.37, 0.036), (0.335, 0.05), (0.318, 0.20), (0.308, 0.36),
           (0.305, 0.425), (0.27, 0.452), (0.215, 0.47), (0.17, 0.505), (0.15, 0.56), (0.112, 0.70), (0.08, 0.872),
           (0.052, 0.872), (0.074, 0.70), (0.108, 0.55), (0.128, 0.49), (0.18, 0.43), (0.235, 0.375), (0.252, 0.20),
           (0.258, 0.05), (0.22, 0.036), (0.05, 0.03)]
    rib = [(0.288, 0.04), (0.282, 0.2), (0.28, 0.38), (0.24, 0.43), (0.17, 0.47), (0.13, 0.53), (0.095, 0.7), (0.067, 0.86)]
    p = _pm(0.004)
    for side in (-1, 1):
        x = side * 0.225
        p.r = 0.005
        p.extrude_yz(_ccw(std), x - 0.011, x + 0.011, iron)
        p.r = 0.0
        pts = np.array([(x, a, b) for a, b in rib])
        p.sweep(pts, [(-0.003, -0.016), (0.003, -0.016), (0.003, 0.016), (-0.003, 0.016)], iron, twist_ref=(1, 0, 0))
        p.r = 0.003
        p.cyl((side * 0.204, *hinge), (side * 0.25, *hinge), 0.028, iron, sides=14)
        for y in (0.09, 0.38):
            _bolt(p, (x, y, 0.03), (0, 0, 1), 0.009, bolt_m, h=0.007, washer=bolt_m)
        # armrest stub: a cast arm off the back upright with a hardwood cap
        p.r = 0.004
        p.obox((x, 0.205, 0.585), (0.024, 0.2, 0.024), iron, pitch=math.radians(-8))
        p.r = 0.009
        p.box((x - 0.02, 0.11, 0.6), (x + 0.02, 0.34, 0.626), w)
        p.r = 0.003
        p.cyl((x - side * 0.011, 0.31, 0.47), (x - side * 0.024, 0.31, 0.47), 0.009, flat("Seat_Bumper", "#151515", 0.8), sides=10)
    # back slats on the arms' front edge, evenly spaced, carriage bolts over the frame
    a, b = np.array([0.165, 0.50]), np.array([0.077, 0.872])
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]])
    tilt = math.atan2(-d[0], d[1])
    p.r = 0.002
    for side in (-1, 1):                        # iron back rail on each standard: the slats sit on it
        c = a + d * 0.24 - n * 0.004
        p.obox((side * 0.225, *c), (0.04, 0.009, 0.36), iron, pitch=tilt)
    p.r = 0.003
    for t in (0.13, 0.233, 0.336):
        c = a + d * t + n * 0.0115
        p.obox((0, *c), (0.48, 0.022, 0.085), w, pitch=tilt)
        for side in (-1, 1):
            for dt in (-0.022, 0.022):
                q = a + d * (t + dt) + n * 0.0225
                _bolt(p, (side * 0.225, *q), (0, *n), 0.0072, bolt_m, h=0.0045, dome=True)
                r0, p.r = p.r, 0.0
                s0 = q - d * 0.022 + n * 0.0003
                p.decal((side * 0.225, *s0), (0, *n), (0, *d), 0.008, 0.026, stain, lift=0.0)
                p.r = r0
    c = a + d * 0.336 + n * 0.0225
    p.decal((0, *c), (0, *n), (0, *d), 0.06, 0.03, brass, lift=0.0012)
    q = _pm(0.003)
    for side in (-1, 1):                       # bearers, seat slats and the hinge pin (open position)
        q.box((side * 0.19 - 0.004, hinge[0] - 0.012, hinge[1] - 0.012), (side * 0.19 + 0.004, 0.565, hinge[1] + 0.0195),
              iron)
    q.r = 0.0
    q.cyl((-0.21, *hinge), (0.21, *hinge), 0.008, bolt_m, sides=8)
    q.r = 0.003
    for k in range(4):
        y0 = hinge[0] - 0.008 + k * 0.074
        q.box((-0.209, y0, hinge[1] + 0.02), (0.209, y0 + 0.065, hinge[1] + 0.042), w)
        for side in (-1, 1):
            _bolt(q, (side * 0.19, y0 + 0.0325, hinge[1] + 0.042), (0, 0, 1), 0.007, bolt_m, h=0.004, dome=True)
    seat_o = q.obj("seat", col)
    _fold([seat_o], (0, *hinge), math.radians(97))
    frame_o = p.obj("frame", col)
    return [frame_o, seat_o]


BUILDERS = {
    "tipup_classic": build_tipup_classic,
    "bucket_fixed": build_bucket_fixed,
    "bench": build_bench,
    "padded_vip": build_padded_vip,
    "rail_seat": build_rail_seat,
    "wooden_tipup": build_wooden_tipup,
}


def resolve(style):
    return catalog.preset("seats", style)


def build_variant(style, variant, col=None):
    spec = resolve(style)
    col = col or collection(f"Seats_{style}")
    parts = BUILDERS[style](variant, col, spec["width"])
    obj = join(parts, f"SEAT_{style}_{spec['variants'].index(variant):02d}_{variant}")
    KD.finish(obj, distance=0.3)
    _wear(obj, style)
    set_params(obj, f"seat.{style}.{variant}", "seat", "seat", {"style": style, "variant": variant})
    obj["width"] = spec["width"]
    return obj


LOD_TRIS = {1: 8000, 2: 260}


def _tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def lod(obj, level, col=None):
    """Copy of a seat variant for a level of detail (shares nothing with the original): level 1 is
    the full seat (decimated only beyond its budget), level 2 a coarse silhouette."""
    o = obj.copy()
    o.data = obj.data.copy()
    o.name = o.data.name = f"{obj.name}_LOD{level}"
    (col or obj.users_collection[0]).objects.link(o)
    n = _tris(o)
    if n > LOD_TRIS[level]:
        mod = o.modifiers.new("Decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = LOD_TRIS[level] / n
        mod.use_collapse_triangulate = True
        bake(o)
    o["lod"] = level
    return o


def build_style(style, col=None):
    """All variants of a style, in variant order (the order instancing picks by index)."""
    spec = resolve(style)
    col = col or collection(f"Seats_{style}")
    return [build_variant(style, v, col) for v in spec["variants"]]


def build(params=None, col=None, template=None):
    params = params or {}
    return build_variant(params.get("style", "tipup_classic"), params.get("variant", "std"), col)


def variant_index(spec, seat, of):
    if "row_end" in spec and seat == of - 1:
        return spec["row_end"]
    if "every" in spec and seat % spec["every"][1] == 0:
        return spec["every"][0]
    return 0
