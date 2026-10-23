"""Big screens: LED video panels showing the scoreboard, mounted to suit the stadium.

Layout section:
  "screens": {"placement": "ends" | "corners" | "center" | "stand_top", "width": 18, "height": 8}
  ends       hung under the roof edge behind each goal (or on a frame on top of an end stand
             without a roof, or freestanding if there is no end stand)
  corners    two diagonal corners: freestanding on a frame (open corners) or hung (bowls)
  center     four-sided centre-hung cube over the centre circle, cabled to the roof edges
  stand_top  on a steel frame on top of the end stands
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

from ..ads import studio as STUDIO
from ..core.scene import collection, empty, link, mesh_object, set_params
from ..core.sockets import socket
from ..stands.geometry import WALL, stand_depth
from .ads import led_material
from .seat import galvanised, rounded_box
from .stand import _tubes_mesh, painted


def panel(name, w, h, col, mat_led, depth=0.9):
    """Screen model centred at the origin, face towards +Y."""
    body = painted("Screen_Body", "#141518", 0.5)
    parts = [rounded_box("case", (w + 0.5, depth, h + 0.5), (0, -depth / 2, 0), 0.05, col, body)]
    V = [(-w / 2, 0.01, -h / 2), (w / 2, 0.01, -h / 2), (w / 2, 0.01, h / 2), (-w / 2, 0.01, h / 2)]
    # wound to face +Y; seen from the front, local -X is the viewer's right, so u runs from +X to -X
    face = mesh_object("face", V, [(0, 3, 2, 1)], col, materials=[mat_led],
                       uvs=[(1, 0), (1, 1), (0, 1), (0, 0)])
    parts.append(face)
    from ..core.scene import join
    return join(parts, name)


def _place(obj, pos, yaw, tilt=0.0):
    """Yaw about Z, then tilt about the panel's own X (negative tips the face down to the pitch)."""
    obj.matrix_world = Matrix.Translation(Vector(pos)) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(tilt, 4, "X")


def _yaw_to_centre(x, y):
    # local +Y must point at the centre spot
    return math.atan2(x, -y)


def build(cfg, pitch, stands, night, col=None):
    if not cfg:
        return None
    col = col or collection("Screens")
    root = empty("Screens", col, size=1.5)
    set_params(root, "screens", "screens", "stadium_origin", cfg)
    w, h = cfg.get("width", 18.0), cfg.get("height", 8.0)
    img = bpy.data.images.load(str(STUDIO.render_scoreboard(**cfg.get("scoreboard", {}))), check_existing=True)
    mat = led_material("Screen_LED", img, 1.6 if night else 3.5, (1920, 1080))
    model = panel("ScreenModel", w, h, col, mat)
    model.hide_render = model.hide_viewport = True
    steel = galvanised()
    L, W = pitch["length"], pitch["width"]
    roofs = (stands or {}).get("roofs", {})
    runs = (stands or {}).get("runs", {})
    placement = cfg.get("placement", "ends")
    tubes, placed = [], []

    def put(name, pos, yaw, tilt=0.0):
        o = link(bpy.data.objects.new(name, model.data), col, root)
        _place(o, pos, yaw, tilt)
        placed.append(o)
        socket("screen", name, tuple(float(v) for v in pos), math.degrees(yaw), parent=root, col=col,
               width=w, height=h)
        return o

    def hang_from_edge(name, line):
        mid = line[len(line) // 2]
        inward = -np.array([mid[0], mid[1]]) / np.hypot(mid[0], mid[1])
        top = mid[2] - 0.6
        centre = np.array([mid[0] + inward[0] * 1.5, mid[1] + inward[1] * 1.5, top - h / 2])
        yaw = _yaw_to_centre(centre[0], centre[1])
        put(name, centre, yaw, math.radians(-8))
        side = np.array([-inward[1], inward[0]])
        for sgn in (-1, 1):
            a = np.array([*(centre[:2] + side * sgn * w * 0.4), top + 0.2])
            tubes.append((np.array([a, [a[0], a[1], mid[2]]]), 0.04))

    def on_frame(name, base_xy, bottom, ground=0.0):
        x, y = base_xy
        yaw = _yaw_to_centre(x, y)
        put(name, (x, y, bottom + h / 2 + 0.25), yaw, math.radians(-4))
        side = np.array([math.cos(yaw), math.sin(yaw)])
        back = -np.array([-math.sin(yaw), math.cos(yaw)])
        for sgn in (-0.38, 0.38):
            p = np.array([x, y]) + side * sgn * w + back * 1.2
            tubes.append((np.array([[p[0], p[1], ground], [p[0], p[1], bottom + h]]), 0.22))
            tubes.append((np.array([[p[0], p[1], bottom + 0.2], [*(p - back * 1.1), bottom + 0.2]]), 0.08))
            tubes.append((np.array([[p[0], p[1], bottom + h], [*(p - back * 1.1), bottom + h]]), 0.08))

    def stand_top(name, run_name):
        run, tiers = runs[run_name]
        t_mid = 0.5 * (run.t0 + run.t1)
        p, n = run.frame(t_mid)
        s = stand_depth(tiers) - 1.0
        top = max(t["h"][-1] for t in tiers) + 1.1
        xy = p + n * s
        on_frame(name, xy, top + 1.5, ground=top)

    if placement in ("ends", "stand_top"):
        for side, x in (("east", 1), ("west", -1)):
            name = f"Screen_{side}"
            if placement == "ends" and side in roofs:
                hang_from_edge(name, roofs[side]["edge_line"])
            elif side in runs:
                stand_top(name, side)
            else:
                on_frame(name, (x * (L / 2 + pitch["runoff_end"] + 4.0), 0.0), 4.0)
    elif placement == "corners":
        for k, (sx, sy) in enumerate(((1, 1), (-1, -1))):
            arc = {(1, 1): "NE", (-1, -1): "SW"}[(sx, sy)]
            if arc in roofs:
                hang_from_edge(f"Screen_{arc}", roofs[arc]["edge_line"])
            else:
                A = stands["A"] if stands else L / 2 + 10
                B = stands["B"] if stands else W / 2 + 8
                on_frame(f"Screen_{arc}", (sx * (A + 4.0), sy * (B + 4.0)), 8.0)
    elif placement == "center":
        if not roofs:
            raise ValueError("a centre-hung screen needs roofs to hang from")
        z_edge = min(float(np.min(r["edge_line"][:, 2])) for r in roofs.values())
        # real centre-hung boards clear the pitch by ~25 m; never higher than just under the roof edge
        cz = min(z_edge - 6.0 - h / 2, 25.0 + h / 2)
        half = w / 2 + 0.3
        body = painted("Screen_Body", "#141518", 0.5)
        for z in (cz - h / 2 - 0.4, cz + h / 2 + 0.4):          # close the cube top and bottom
            cap = rounded_box("ScreenCap", (2 * half + 0.6, 2 * half + 0.6, 0.6), (0, 0, z), 0.05, col, body)
            cap.parent = root
        for k, (ux, uy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
            # four faces of a cube, each looking outward (local +Y along (ux, uy))
            put(f"Screen_centre_{k}", (ux * half, uy * half, cz), math.atan2(-ux, uy))
        for line in [r["edge_line"] for r in roofs.values()]:
            mid = line[len(line) // 2]
            tubes.append((np.array([[0.0, 0.0, cz + h / 2 + 0.5], mid]), 0.035))
    else:
        raise ValueError(f"unknown screen placement {placement!r}")
    if tubes:
        _tubes_mesh("ScreenSteel", tubes, col, root, steel, sides=8)
    return {"root": root, "screens": len(placed)}
