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


# -- materials --------------------------------------------------------------------------------

def seat_plastic(default="#c8102e"):
    """Polypropylene shell: satin, faint mould texture, colour from the instancer if present."""
    name = "TINT_seat"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    col = nb.mix(attr[3], srgb(default), attr[0])  # Alpha is 0 when no instancer colour exists
    pos = nb.geometry()["Position"]
    grain = nb.noise(pos, 260.0, detail=2)
    dust = nb.smooth(0.6, 0.85, nb.noise(pos, 25.0, detail=3))
    col = nb.mix(nb.mul(dust, 0.05), col, srgb("#8d8a80"))
    bsdf = nb.principled(Base_Color=col, Roughness=nb.madd(dust, 0.12, 0.32), Coat_Weight=0.15,
                         Coat_Roughness=0.3, Subsurface_Weight=0.04, Subsurface_Radius=(0.01, 0.01, 0.01),
                         Normal=nb.bump(grain, 0.05, 0.001))
    nb.output(bsdf)
    return m


def seat_fabric(default="#3b3f46"):
    name = "TINT_seat_fabric"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    col = nb.mix(attr[3], srgb(default), attr[0])
    pos = nb.geometry()["Position"]
    weave = nb.noise(pos, 1400.0, detail=1)
    bsdf = nb.principled(Base_Color=col, Roughness=0.8, Sheen_Weight=0.6, Sheen_Roughness=0.4,
                         Normal=nb.bump(weave, 0.2, 0.0005))
    nb.output(bsdf)
    return m


def wood(name="Seat_Wood"):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    attr = nb.attribute("seat_color", "INSTANCER")
    base = nb.mix(attr[3], srgb("#7a4a2a"), attr[0])
    pos = nb.geometry()["Position"]
    stretch = nb.vmath("MULTIPLY", pos, (2.0, 40.0, 40.0))
    grain = nb.noise(stretch, 6.0, detail=6, rough=0.6, distortion=0.4)
    col = nb.mix(nb.mul(grain, 0.5), base, nb.mix(1.0, base, (0.55, 0.5, 0.45), blend="MULTIPLY"))
    bsdf = nb.principled(Base_Color=col, Roughness=nb.madd(grain, 0.2, 0.4), Coat_Weight=0.4, Coat_Roughness=0.2,
                         Normal=nb.bump(grain, 0.15, 0.001))
    nb.output(bsdf)
    return m


def steel(name="Seat_Steel", color="#2c2f33"):
    return MAT.painted_metal(name, color, rough=0.45, dirt=0.25, scuffs=0.25)


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


# -- styles -----------------------------------------------------------------------------------

def _tipup_standard(name, x, col):
    poly = [(0.10, 0.0), (0.34, 0.0), (0.34, 0.012), (0.25, 0.014), (0.232, 0.40), (0.17, 0.47),
            (0.135, 0.62), (0.105, 0.62), (0.125, 0.455), (0.19, 0.385), (0.205, 0.014), (0.10, 0.012)]
    return side_plate(name, poly, x, 0.014, col, steel())


def build_tipup_classic(variant, col, width=0.5):
    mat = seat_plastic()
    back = shell("back", [(0.125, 0.46), (0.118, 0.55), (0.105, 0.65), (0.09, 0.75), (0.075, 0.84)],
                 [0.42, 0.43, 0.44, 0.44, 0.43], 0.03, 0.006, col=col, mat=mat)
    pan = shell("pan", [(0.17, 0.445), (0.27, 0.44), (0.38, 0.445), (0.50, 0.455), (0.54, 0.44)],
                [0.42, 0.44, 0.44, 0.44, 0.42], 0.02, 0.008, col=col, mat=mat)
    _rotate_about_x(pan, (0, 0.20, 0.43), math.radians(84))   # empty tip-up seats stand folded
    parts = [back, pan, _tipup_standard("stdL", -width / 2 + 0.007, col)]
    if variant == "end":
        parts.append(_tipup_standard("stdR", width / 2 - 0.007, col))
    return parts


def build_bucket_fixed(variant, col, width=0.48):
    mat = seat_plastic("#1d4ea8")
    sh = shell("bucket", [(0.47, 0.425), (0.39, 0.415), (0.29, 0.405), (0.20, 0.415), (0.145, 0.46),
                          (0.13, 0.54), (0.125, 0.62), (0.12, 0.68)],
               [0.40, 0.43, 0.44, 0.44, 0.43, 0.42, 0.40, 0.37], 0.035, 0.007, n_across=9, col=col, mat=mat)
    ped = tube("pedestal", [(0, 0.29, 0.0), (0, 0.29, 0.40)], 0.028, col, steel())
    foot = rounded_box("foot", (0.16, 0.16, 0.012), (0, 0.29, 0.006), 0.004, col, steel())
    head = rounded_box("head", (0.2, 0.12, 0.02), (0, 0.29, 0.40), 0.005, col, steel())
    return [sh, ped, foot, head]


def build_bench(variant, col, width=0.45):
    mat = seat_plastic("#b9bcbf")
    parts = []
    for k, y in enumerate((0.19, 0.30, 0.41)):
        parts.append(rounded_box(f"plank{k}", (width + 0.002, 0.11, 0.035), (0, y, 0.425), 0.008, col, mat))
    if variant == "bracket":
        poly = [(0.22, 0.0), (0.36, 0.0), (0.36, 0.012), (0.31, 0.014), (0.305, 0.38), (0.465, 0.38),
                (0.465, 0.405), (0.13, 0.405), (0.13, 0.38), (0.255, 0.38), (0.25, 0.014), (0.22, 0.012)]
        parts.append(side_plate("bracket", poly, 0.0, 0.016, col, steel("Bench_Bracket", "#5c6064")))
    return parts


def build_padded_vip(variant, col, width=0.56):
    fab = seat_fabric()
    frame = steel("VIP_Frame", "#26282b")
    parts = [
        rounded_box("cushion", (0.46, 0.46, 0.11), (0, 0.39, 0.44), 0.035, col, fab),
        rounded_box("backpad", (0.46, 0.11, 0.6), (0, 0.17, 0.78), 0.04, col, fab, tilt_x=math.radians(-12)),
    ]
    for side, x in (("L", -width / 2 + 0.03), ("R", width / 2 - 0.03)):
        if side == "R" and variant != "end":
            continue
        parts.append(rounded_box(f"arm{side}", (0.055, 0.5, 0.6), (x, 0.36, 0.30), 0.012, col, frame))
        parts.append(rounded_box(f"armtop{side}", (0.07, 0.48, 0.035), (x, 0.38, 0.62), 0.012, col,
                                 MAT.plastic("VIP_Armrest", "#141414", rough=0.35, dirt=0.0)))
    return parts


def build_rail_seat(variant, col, width=0.5):
    g = galvanised()
    mat = seat_plastic()
    parts = [
        tube("rail", [(-width / 2, 0.07, 1.12), (width / 2, 0.07, 1.12)], 0.024, col, g),
        tube("midrail", [(-width / 2, 0.075, 0.62), (width / 2, 0.075, 0.62)], 0.016, col, g),
        rounded_box("backplate", (width - 0.08, 0.02, 0.17), (0, 0.10, 0.94), 0.008, col, mat),
        rounded_box("pan", (width - 0.12, 0.03, 0.30), (0, 0.13, 0.70), 0.01, col, mat),
    ]
    if variant == "post":
        x = -width / 2 + 0.03  # inside this seat's own width, so neighbours never overlap
        parts.append(rounded_box("post", (0.05, 0.05, 1.12), (x, 0.07, 0.56), 0.004, col, g))
        parts.append(rounded_box("plate", (0.06, 0.14, 0.01), (x, 0.08, 0.005), 0.002, col, g))
    return parts


def build_wooden_tipup(variant, col, width=0.5):
    w = wood()
    iron = steel("Cast_Iron", "#1e1f1f")
    parts = []
    for k, z in enumerate((0.55, 0.64, 0.73, 0.81)):
        y = 0.125 - (z - 0.55) * 0.18
        parts.append(rounded_box(f"backslat{k}", (0.44, 0.022, 0.07), (0, y, z), 0.006, col, w,
                                 tilt_x=math.radians(-10)))
    for k in range(4):
        sl = rounded_box(f"seatslat{k}", (0.44, 0.075, 0.022), (0, 0.24 + 0.085 * k, 0.44), 0.006, col, w)
        _rotate_about_x(sl, (0, 0.21, 0.43), math.radians(82))
        parts.append(sl)
    poly = [(0.08, 0.0), (0.36, 0.0), (0.36, 0.015), (0.27, 0.02), (0.255, 0.30), (0.235, 0.42),
            (0.17, 0.47), (0.13, 0.86), (0.095, 0.86), (0.13, 0.44), (0.205, 0.37), (0.22, 0.02),
            (0.08, 0.015)]
    parts.append(side_plate("stdL", poly, -width / 2 + 0.01, 0.02, col, iron, bevel=0.004))
    if variant == "end":
        parts.append(side_plate("stdR", poly, width / 2 - 0.01, 0.02, col, iron, bevel=0.004))
    return parts


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
    set_params(obj, f"seat.{style}.{variant}", "seat", "seat", {"style": style, "variant": variant})
    obj["width"] = spec["width"]
    return obj


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
