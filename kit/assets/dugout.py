"""Dugouts (team benches) and technical areas on the main-stand (south) side, either side of halfway.

Layout section:
  "dugouts": {"style": "shell" | "box" | "shelter", "seats": 14, "seat": "padded_vip", "color": "#c8102e"}
  shell    curved tinted polycarbonate canopy over padded bucket seats (modern elite)
  box      brick or concrete box with a flat roof and an open front (classic)
  shelter  steel frame with clear panels and a bench (grassroots)

Each dugout seats two rows. The technical area is marked 1 m either side of the seats and forward to
within 1 m of the touchline, in dashed paint (Law 1 / IFAB guidance).
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import material, srgb
from ..core.scene import collection, empty, join, link, mesh_object, set_params
from ..core.sockets import socket
from . import seat as SEAT
from .facade import brick
from .stand import _tubes_mesh, concrete, painted

FRONT_FROM_TOUCHLINE = 4.3
DEPTH = 2.5
GAP = 6.0          # between the two dugouts: fourth official and the tunnel


def polycarbonate(color):
    name = "Dugout_Polycarbonate"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=0.08, Transmission_Weight=0.85, IOR=1.58,
                            Coat_Weight=0.3))
    return m


def resolve(cfg):
    p = {"style": "shell", "seats": 14, "seat": "padded_vip", "color": "#c8102e"}
    p.update(cfg or {})
    if p["style"] == "shelter" and "seat" not in (cfg or {}):
        p["seat"] = "bench"
    return p


def length(cfg):
    p = resolve(cfg)
    per_row = math.ceil(p["seats"] / 2)
    return per_row * SEAT.resolve(p["seat"])["width"] + 1.0


def board_gap(cfg):
    """Opening needed in the perimeter boards for both dugouts and the tunnel between them."""
    return 2 * length(cfg) + GAP + 3.0


def _model(p, col, name):
    """One dugout, front centre at the origin, +Y towards the pitch."""
    Lx = length(p)
    parts = []
    slab = concrete("Dugout_Slab", "#8e8a82")
    V, F = _box(-Lx / 2, Lx / 2, -DEPTH, 0.0, 0.0, 0.12)
    parts.append(mesh_object("slab", V, F, col, materials=[slab]))
    V, F = _box(-Lx / 2, Lx / 2, -DEPTH, -1.15, 0.12, 0.42)
    parts.append(mesh_object("step", V, F, col, materials=[slab]))
    style = p["style"]
    frame = MAT.painted_metal("Dugout_Frame", "#d8dadb" if style == "shell" else "#2b2d30", rough=0.35, dirt=0.2)
    if style == "shell":
        shell = polycarbonate("#3d4954")
        prof = []
        for a in np.linspace(math.pi, math.pi / 2 + 0.25, 14):   # back, up and over to the front lip
            prof.append((-DEPTH + 0.1 + 2.35 * (1 + math.cos(a)), 0.12 + 2.05 * math.sin(a)))
        prof = np.array(prof)
        # canopy surface along x
        nx = 12
        xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
        Vv = [(x, y, z) for x in xs for (y, z) in prof]
        k = len(prof)
        Ff = [(i * k + j, i * k + j + 1, (i + 1) * k + j + 1, (i + 1) * k + j) for i in range(nx) for j in range(k - 1)]
        canopy = mesh_object("canopy", Vv, Ff, col, smooth=True, materials=[shell])
        mod = canopy.modifiers.new("t", "SOLIDIFY")
        mod.thickness = 0.012
        from ..core.scene import bake
        bake(canopy)
        parts.append(canopy)
        for k, x in enumerate((-Lx / 2, Lx / 2)):                 # glazed ends
            Ve = [(x, -DEPTH + 0.1, 0.12)] + [(x, y, z) for y, z in prof] + [(x, prof[-1][0], 0.12)]
            # never put floats in object names: Blender parses a trailing ".digits" as a counter
            parts.append(mesh_object(f"end_{k}", Ve, [tuple(range(len(Ve)))], col, materials=[shell]))
        ribs = [(np.array([[x, y, z] for y, z in prof]), 0.03) for x in np.linspace(-Lx / 2, Lx / 2, 5)]
        parts.append(_tubes_mesh("ribs", ribs, col, None, frame, sides=8))
    elif style == "box":
        wall = brick("Dugout_Brick", "#7a3426")
        for V, F in (_box(-Lx / 2, Lx / 2, -DEPTH - 0.25, -DEPTH, 0.0, 2.2),
                     _box(-Lx / 2 - 0.25, -Lx / 2, -DEPTH - 0.25, 0.0, 0.0, 2.2),
                     _box(Lx / 2, Lx / 2 + 0.25, -DEPTH - 0.25, 0.0, 0.0, 2.2)):
            parts.append(mesh_object("wall", V, F, col, materials=[wall]))
        V, F = _box(-Lx / 2 - 0.3, Lx / 2 + 0.3, -DEPTH - 0.3, 0.35, 2.2, 2.38)
        parts.append(mesh_object("roof", V, F, col, materials=[painted("TINT_dugout", p["color"], 0.45)]))
    else:  # shelter
        clear = polycarbonate("#dfe8ea")
        posts = []
        for x in (-Lx / 2, Lx / 2):
            for y in (-DEPTH + 0.05, -0.05):
                posts.append((np.array([[x, y, 0.12], [x, y, 2.15]]), 0.04))
        parts.append(_tubes_mesh("posts", posts, col, None, frame, sides=8))
        for V, F in (_box(-Lx / 2, Lx / 2, -DEPTH + 0.02, -DEPTH + 0.04, 0.3, 2.1),
                     _box(-Lx / 2 - 0.01, -Lx / 2 + 0.01, -DEPTH, -0.1, 0.3, 2.1),
                     _box(Lx / 2 - 0.01, Lx / 2 + 0.01, -DEPTH, -0.1, 0.3, 2.1)):
            parts.append(mesh_object("pane", V, F, col, materials=[clear]))
        V, F = _box(-Lx / 2 - 0.15, Lx / 2 + 0.15, -DEPTH - 0.15, 0.25, 2.15, 2.22)
        parts.append(mesh_object("roof", V, F, col, materials=[painted("TINT_dugout", p["color"], 0.45)]))
    obj = join([o for o in parts if o is not None], name)
    return obj


def _box(x0, x1, y0, y1, z0, z1):
    V = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return V, F


def _seats(p, col, parent, origin, flip):
    style = p["seat"]
    spec = SEAT.resolve(style)
    var_col = collection(f"SeatModels_{style}")
    if not var_col.objects:
        SEAT.build_style(style, var_col)
    bpy.context.view_layer.layer_collection.children[var_col.name].exclude = True
    models = list(var_col.objects)
    per_row = math.ceil(p["seats"] / 2)
    w = spec["width"]
    Lx = length(p)
    for row, (y, z) in enumerate(((-1.15, 0.12), (-DEPTH + 0.05, 0.42))):
        for j in range(per_row):
            v = SEAT.variant_index(spec, j, per_row)
            m = next(o for o in models if o.name.endswith(f"_{v:02d}_{spec['variants'][v]}"))
            o = link(bpy.data.objects.new(f"DugoutSeat_{row}_{j}", m.data), col, parent)
            local = Vector((-Lx / 2 + 0.5 + (j + 0.5) * w, y, z))
            o.matrix_world = origin @ Matrix.Translation(local)


def technical_areas(pitch, xs, Lx, col, parent):
    """Dashed technical-area lines: 1 m beyond the seats each side, to 1 m from the touchline."""
    from .pitch import crown_fn
    zf = crown_fn(pitch)
    W = pitch["width"]
    y_front = -(W / 2 + 1.0)
    y_back = -(W / 2 + FRONT_FROM_TOUCHLINE)
    paint = bpy.data.materials.get("TINT_Pitch_Paint")
    vs, fs, uvs, off = [], [], [], 0
    for xc in xs:
        x0, x1 = xc - Lx / 2 - 1.0, xc + Lx / 2 + 1.0
        lines = [((x0, y_back), (x0, y_front)), ((x0, y_front), (x1, y_front)), ((x1, y_front), (x1, y_back))]
        for a, b in lines:
            a, b = np.array(a), np.array(b)
            L = np.linalg.norm(b - a)
            n = int(L / 0.6)
            for k in range(n):
                if k % 2:
                    continue
                p0 = a + (b - a) * (k / n)
                p1 = a + (b - a) * ((k + 1) / n)
                V, F, UV = M.strip([p0, p1], 0.1)
                V[:, 2] = zf(V[:, 0], V[:, 1]) + 0.004
                vs.append(V)
                fs.append(F + off)
                uvs.append(UV)
                off += len(V)
    if vs:
        mesh_object("TechnicalAreas", np.concatenate(vs), np.concatenate(fs), col, parent,
                    materials=[paint] if paint else (), uvs=np.concatenate(uvs))


def build(cfg, pitch, night=False, col=None):
    if not cfg:
        return None
    p = resolve(cfg)
    col = col or collection("Dugouts")
    root = empty("Dugouts", col, size=1.0)
    set_params(root, f"dugouts.{p['style']}", "dugouts", "stadium_origin", p)
    Lx = length(p)
    model = _model(p, col, "DugoutModel")
    model.hide_render = model.hide_viewport = True
    y = -(pitch["width"] / 2 + FRONT_FROM_TOUCHLINE)
    xs = (-(GAP / 2 + Lx / 2), GAP / 2 + Lx / 2)
    for side, x in zip(("home", "away"), xs):
        o = link(bpy.data.objects.new(f"Dugout_{side}", model.data), col, root)
        origin = Matrix.Translation(Vector((x, y, 0.0)))
        o.matrix_world = origin
        _seats(p, col, root, origin, False)
        socket("dugout", side, (x, y, 0.0), 0.0, parent=root, col=col, seats=p["seats"], length=round(Lx, 2))
    technical_areas(pitch, xs, Lx, col, root)
    return {"root": root, "length": Lx, "xs": xs}
