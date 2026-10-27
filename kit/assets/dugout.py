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


def _segments(digit, x0, x1, z0, z1, t):
    """The lit segments of a seven-segment digit in the box x0..x1, z0..z1 (segment thickness t)."""
    zm = 0.5 * (z0 + z1)
    seg = {"a": (x0 + t, x1 - t, z1 - t, z1), "b": (x1 - t, x1, zm + t / 2, z1 - t), "c": (x1 - t, x1, z0 + t, zm - t / 2),
           "d": (x0 + t, x1 - t, z0, z0 + t), "e": (x0, x0 + t, z0 + t, zm - t / 2), "f": (x0, x0 + t, zm + t / 2, z1 - t),
           "g": (x0 + t, x1 - t, zm - t / 2, zm + t / 2)}
    lit = ["abcdef", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgedc", "abc", "abcdefg", "abcdfg"][digit % 10]
    return [seg[ch] for ch in lit]


def front_from_touchline(pitch, stands=None):
    """Dugout fronts sit 4.3 m behind the touchline, closer (not under 1.5 m) when the main stand's
    front is near, so the dugout and its roof never run into the stand."""
    if stands and "south" in (stands.get("runs") or {}):
        room = stands["B"] - pitch["width"] / 2 - DEPTH - 0.65
        return max(1.5, min(FRONT_FROM_TOUCHLINE, room))
    return FRONT_FROM_TOUCHLINE
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
        # base and lip rails along the shell, a club-colour back panel and a kick plate
        for y, z in (prof[0], prof[-1]):
            ribs.append((np.array([[-Lx / 2, y, z], [Lx / 2, y, z]]), 0.04))
        parts.append(_tubes_mesh("ribs", ribs, col, None, frame, sides=8))
        club = painted("TINT_dugout", p["color"], 0.45)
        for V, F in (_box(-Lx / 2 + 0.05, Lx / 2 - 0.05, -DEPTH + 0.14, -DEPTH + 0.2, 0.42, 1.55),
                     _box(-Lx / 2, Lx / 2, -0.08, 0.0, 0.12, 0.32)):
            parts.append(mesh_object("panel", V, F, col, materials=[club]))
    elif style == "box":
        # a brick base (back wall and low side walls) under clear polycarbonate sides and a sloping
        # polycarbonate roof on a steel frame, with a club-colour fascia
        wall = brick("Dugout_Brick", "#7a3426")
        for V, F in (_box(-Lx / 2 - 0.25, Lx / 2 + 0.25, -DEPTH - 0.25, -DEPTH, 0.0, 1.1),
                     _box(-Lx / 2 - 0.25, -Lx / 2, -DEPTH - 0.25, 0.0, 0.0, 1.0),
                     _box(Lx / 2, Lx / 2 + 0.25, -DEPTH - 0.25, 0.0, 0.0, 1.0)):
            parts.append(mesh_object("wall", V, F, col, materials=[wall]))
        V, F = _box(-Lx / 2 - 0.3, Lx / 2 + 0.3, -DEPTH - 0.3, -DEPTH + 0.05, 1.1, 1.18)
        parts.append(mesh_object("coping", V, F, col, materials=[slab]))
        parts += _canopy(p, Lx, 1.1, True, col, polycarbonate("#3d4954"), frame)
    else:  # shelter
        parts += _canopy(p, Lx, 0.3, False, col, polycarbonate("#dfe8ea"), frame)
    obj = join([o for o in parts if o is not None], name)
    return obj


def _canopy(p, Lx, z_base, walled, col, clear, frame):
    """Clear polycarbonate panels (back above z_base, sides from z_base) under a roof sloping from
    2.35 m at the back to 2.15 m over the front, on a steel frame of posts, rails and rafters, with a
    club-colour fascia along the front edge. walled: brick walls carry the back below z_base."""
    yb, yf, zb, zf = -DEPTH - 0.05, 0.3, 2.35, 2.15

    def zr(y):
        return zb + (zf - zb) * (y - yb) / (yf - yb)
    x0, x1 = -Lx / 2, Lx / 2
    quads = [[(x0, yb, zb), (x1, yb, zb), (x1, yf, zf), (x0, yf, zf)],
             [(x0, -DEPTH + 0.02, z_base), (x1, -DEPTH + 0.02, z_base), (x1, -DEPTH + 0.02, zr(-DEPTH)),
              (x0, -DEPTH + 0.02, zr(-DEPTH))]]
    quads += [[(x, -DEPTH, z_base), (x, 0.0, z_base), (x, 0.0, zr(0.0)), (x, -DEPTH, zr(-DEPTH))] for x in (x0, x1)]
    out = [mesh_object(f"pane_{k}", q, [(0, 1, 2, 3)], col, materials=[clear]) for k, q in enumerate(quads)]
    V, F = _box(x0 - 0.05, x1 + 0.05, yf - 0.06, yf + 0.02, zf - 0.18, zf + 0.04)
    out.append(mesh_object("fascia", V, F, col, materials=[painted("TINT_dugout", p["color"], 0.45)]))
    r = 0.03
    z_post = 1.0 if walled else 0.12
    lines = []
    for x in (x0, x1):
        lines += [(np.array([[x, -DEPTH, z_post], [x, -DEPTH, zr(-DEPTH)]]), r),
                  (np.array([[x, 0.0, z_post], [x, 0.0, zr(0.0)]]), r),
                  (np.array([[x, -DEPTH, z_base], [x, 0.0, z_base]]), r)]
    lines += [(np.array([[x0, yb, zb], [x1, yb, zb]]), r), (np.array([[x0, 0.0, zr(0.0)], [x1, 0.0, zr(0.0)]]), r),
              (np.array([[x0, -DEPTH + 0.02, z_base], [x1, -DEPTH + 0.02, z_base]]), r)]
    n = max(2, int(round(Lx / 1.2)))
    for k in range(n + 1):
        x = x0 + Lx * k / n
        lines.append((np.array([[x, yb, zb], [x, yf, zf]]), 0.022))
        if 0 < k < n:
            lines.append((np.array([[x, -DEPTH + 0.02, z_base], [x, -DEPTH + 0.02, zr(-DEPTH)]]), 0.02))
    out.append(_tubes_mesh("canopy_frame", lines, col, None, frame, sides=8))
    return out


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


def technical_areas(pitch, xs, Lx, col, parent, front=FRONT_FROM_TOUCHLINE):
    """Dashed technical-area lines: 1 m beyond the seats each side, to 1 m from the touchline."""
    from .pitch import crown_fn
    zf = crown_fn(pitch)
    W = pitch["width"]
    y_front = -(W / 2 + 1.0)
    y_back = -(W / 2 + front)
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


def build(cfg, pitch, night=False, col=None, stands=None):
    if not cfg:
        return None
    p = resolve(cfg)
    col = col or collection("Dugouts")
    root = empty("Dugouts", col, size=1.0)
    set_params(root, f"dugouts.{p['style']}", "dugouts", "stadium_origin", p)
    Lx = length(p)
    model = _model(p, col, "DugoutModel")
    model.hide_render = model.hide_viewport = True
    model["template"] = True
    front = front_from_touchline(pitch, stands)
    y = -(pitch["width"] / 2 + front)
    xs = (-(GAP / 2 + Lx / 2), GAP / 2 + Lx / 2)
    for side, x in zip(("home", "away"), xs):
        o = link(bpy.data.objects.new(f"Dugout_{side}", model.data), col, root)
        origin = Matrix.Translation(Vector((x, y, 0.0)))
        o.matrix_world = origin
        _seats(p, col, root, origin, False)
        socket("dugout", side, (x, y, 0.0), 0.0, parent=root, col=col, seats=p["seats"], length=round(Lx, 2))
    # the fourth official's desk beside the tunnel exit, between the dugouts
    dark = MAT.painted_metal("Dugout_Frame", "#2b2d30", rough=0.35, dirt=0.2)
    dx, dy = GAP / 2 - 0.6, y + 0.9
    boxes = [(_box(dx - 0.7, dx + 0.7, dy - 0.3, dy + 0.3, 0.72, 0.76), dark),
             (_box(dx - 0.7, dx + 0.7, dy + 0.26, dy + 0.3, 0.05, 0.72), painted("TINT_dugout", p["color"], 0.45))]
    boxes += [(_box(dx + sx * 0.66 - 0.03, dx + sx * 0.66 + 0.03, dy - 0.3, dy + 0.3, 0.0, 0.72), dark) for sx in (-1, 1)]
    # the electronic substitution board in its holder at one end: a slim housing with a carry
    # handle, and on both faces the two-digit number cells, red (off) over green (on)
    from .tunnel import _emissive
    red = _emissive("SubsBoard_Red", "#ff1508", 2.5, base="#101112")
    green = _emissive("SubsBoard_Green", "#1aff44", 2.5, base="#101112")
    bx0, by = dx - 0.62, dy - 0.05
    boxes.append((_box(bx0, bx0 + 0.62, by - 0.035, by + 0.035, 0.76, 1.24), dark))
    boxes += [(_box(hx - 0.015, hx + 0.015, by - 0.012, by + 0.012, 1.24, 1.33), dark) for hx in (bx0 + 0.1, bx0 + 0.52)]
    boxes.append((_box(bx0 + 0.085, bx0 + 0.535, by - 0.015, by + 0.015, 1.3, 1.34), dark))
    for sy in (-1.0, 1.0):
        ya, yb = (by - 0.039, by - 0.035) if sy < 0 else (by + 0.035, by + 0.039)
        for z0, mm, digits in ((1.02, red, (1, 4)), (0.8, green, (2, 3))):
            for c, dg in zip((bx0 + 0.07, bx0 + 0.33), digits):
                # read the right way round from the side it faces
                x0 = c if sy < 0 else 2 * bx0 + 0.62 - c - 0.22
                for sx0, sx1, sz0, sz1 in _segments(dg, x0 + 0.03, x0 + 0.19, z0 + 0.01, z0 + 0.17, 0.024):
                    if sy > 0:      # (the glyph itself mirrored too on the face read from +y)
                        sx0, sx1 = 2 * x0 + 0.22 - sx1, 2 * x0 + 0.22 - sx0
                    boxes.append((_box(sx0, sx1, ya, yb, sz0, sz1), mm))
    # water bottles at the other end
    bottle = MAT.plastic("FourthOfficial_Bottle", "#6fa6d6", 0.2)
    for k, bxx in enumerate((dx + 0.25, dx + 0.36, dx + 0.47)):
        byy = dy - 0.1 + 0.06 * k
        boxes.append((_box(bxx - 0.035, bxx + 0.035, byy - 0.035, byy + 0.035, 0.76, 0.98), bottle))
        boxes.append((_box(bxx - 0.015, bxx + 0.015, byy - 0.015, byy + 0.015, 0.98, 1.02), dark))
    for k, ((V, F), m) in enumerate(boxes):
        mesh_object(f"FourthOfficial_{k}", V, F, col, root, materials=[m])
    technical_areas(pitch, xs, Lx, col, root, front)
    return {"root": root, "length": Lx, "xs": xs}
