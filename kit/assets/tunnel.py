"""Players' tunnel at halfway on the main-stand side: a portal in the stand front and a telescopic
tunnel (hoops under a fabric cover) reaching out towards the pitch.

Layout section:
  "tunnel": {"style": "telescopic" | "portal", "color": "#c8102e", "width": 3.6, "height": 2.7}
"""
import math

import bpy
import numpy as np

from ..core import materials as MAT
from ..core.nodes import material, srgb
from ..core.scene import collection, empty, mesh_object, set_params
from ..core.sockets import socket
from .dugout import _box, front_from_touchline
from .stand import _tubes_mesh, concrete, painted


def fabric(color):
    name = "TINT_tunnel"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=0.75, Sheen_Weight=0.6, Sheen_Roughness=0.35,
                            Normal=nb.bump(nb.noise(pos, 400.0, detail=1), 0.1, 0.001)))
    return m


def porch_depth(style, length):
    """Depth of the lit porch at the tunnel mouth: a telescopic tunnel keeps most of the run."""
    return max(min(2.4, 0.4 * length), 0.6) if style == "telescopic" else 2.4


def _emissive(name, color, strength, base=None):
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    nb.output(nb.principled(Base_Color=srgb(base or color), Roughness=0.3, Emission_Color=srgb(color),
                            Emission_Strength=strength))
    return m


def build(cfg, pitch, stands, col=None):
    if not cfg:
        return None
    p = {"style": "telescopic", "color": "#c8102e", "width": 3.6, "height": 2.7}
    p.update(cfg)
    col = col or collection("Tunnel")
    root = empty("Tunnel", col, size=1.0)
    set_params(root, f"tunnel.{p['style']}", "tunnel", "stadium_origin", p)
    W, w, h = pitch["width"], p["width"], p["height"]
    y_end = -(W / 2 + front_from_touchline(pitch, stands))
    has_stand = bool(stands) and "south" in (stands.get("runs") or {})
    y_wall = -stands["B"] if has_stand else y_end - 3.0
    trim = painted("Tunnel_Trim", "#1a1b1e", 0.4)
    building = concrete("Tunnel_Building", "#a8a399")
    club = fabric(p["color"])
    mat = MAT.rubber("Tunnel_Mat", "#1f2a22")
    light = _emissive("Tunnel_Light", "#fff8eb", 8.0)
    glass = _emissive("Tunnel_DoorGlass", "#fff2d9", 0.1, base="#1c2428")
    steel = MAT.painted_metal("Tunnel_Hoop", "#d9dbdc", 0.35, 0.1, 0.0)
    # a lit porch in front of the stand wall: side walls with club-colour panels, a roof slab with
    # light strips and a club-colour header, a floor mat, and double doors in the back wall
    dp = porch_depth(p["style"], y_end - y_wall)
    y0, y1 = y_wall, y_wall + dp
    boxes = [(_box(-w / 2, w / 2, y0 - 0.02, y0 + 0.01, 0.0, h), building),
             (_box(-1.05, 1.05, y0 + 0.01, y0 + 0.04, 0.0, 2.35), trim)]
    boxes += [(_box(x - 0.45, x + 0.45, y0 + 0.04, y0 + 0.05, 0.9, 2.2), glass) for x in (-0.52, 0.52)]
    for sx in (-1.0, 1.0):
        boxes.append((_box(sx * w / 2 - (0.3 if sx < 0 else 0.0), sx * w / 2 + (0.3 if sx > 0 else 0.0), y0, y1, 0.0,
                           h + 0.35), building))
        xa, xb = (-w / 2, -w / 2 + 0.02) if sx < 0 else (w / 2 - 0.02, w / 2)
        boxes.append((_box(xa, xb, y0 + 0.3, y1 - 0.2, 1.0, 2.2), club))
    for sx in (-1.0, 1.0):
        # the club panels framed in trim, a steel handrail along each wall below them
        def xi(o):
            return sx * (w / 2 - o)
        xa, xb = min(xi(0.02), xi(0.045)), max(xi(0.02), xi(0.045))
        for za, zb in ((0.96, 1.0), (2.2, 2.24)):
            boxes.append((_box(xa, xb, y0 + 0.26, y1 - 0.16, za, zb), trim))
        for ya, yb in ((y0 + 0.26, y0 + 0.3), (y1 - 0.2, y1 - 0.16)):
            boxes.append((_box(xa, xb, ya, yb, 0.96, 2.24), trim))
        boxes.append((_box(min(xi(0.07), xi(0.11)), max(xi(0.07), xi(0.11)), y0 + 0.15, y1 - 0.1, 0.86, 0.9), steel))
        for y in (y0 + 0.4, y1 - 0.3):
            boxes.append((_box(min(xi(0.0), xi(0.09)), max(xi(0.0), xi(0.09)), y - 0.02, y + 0.02, 0.8, 0.86), steel))
    # the doors: a meeting stile, a mid rail and stainless push bars across both leaves
    boxes.append((_box(-0.04, 0.04, y0 + 0.04, y0 + 0.07, 0.0, 2.3), trim))
    boxes.append((_box(-1.0, 1.0, y0 + 0.04, y0 + 0.07, 1.0, 1.1), trim))
    boxes += [(_box(x - 0.36, x + 0.36, y0 + 0.08, y0 + 0.12, 1.03, 1.07), steel) for x in (-0.52, 0.52)]
    boxes.append((_box(-w / 2 - 0.3, w / 2 + 0.3, y0, y1, h, h + 0.35), building))
    boxes.append((_box(-w / 2 - 0.3, w / 2 + 0.3, y1, y1 + 0.06, h - 0.05, h + 0.4), club))
    boxes += [(_box(x - 0.08, x + 0.08, y0 + 0.2, y1 - 0.2, h - 0.04, h), light) for x in (-w / 4, w / 4)]
    boxes.append((_box(-w / 2, w / 2, y0, y1, 0.0, 0.03), mat))
    if not has_stand:
        boxes.append((_box(-w / 2 - 1.2, w / 2 + 1.2, y_wall - 4.0, y_wall, 0.0, h + 0.8), building))
    for k, ((V, F), m) in enumerate(boxes):
        mesh_object(f"TunnelPorch_{k}", V, F, col, root, materials=[m])
    if p["style"] == "telescopic":
        # telescoping segments, each a little larger than the one it slides into, overlapping it;
        # fabric drooping between each segment's end hoops and middle rib, a rubber skirt at the foot
        ya, yb = y1 + 0.1, y_end
        n_seg = min(max(int(round((yb - ya) / 1.6)), 2), 7)
        seg = (yb - ya) / n_seg
        hw = max(0.5, h - w / 2)
        prof = [(-w / 2, 0.0)] + [((w / 2) * math.cos(a), hw + (w / 2) * math.sin(a))
                                  for a in np.linspace(math.pi, 0, 17)] + [(w / 2, 0.0)]
        tubes, Vv, Ff = [], [], []
        k = len(prof)
        for sg in range(n_seg):
            sc = 1.0 + 0.035 * sg
            pr = [(x * sc, z * sc) for x, z in prof]
            s0 = ya + sg * seg
            s1 = ya + (sg + 1) * seg + (0.12 if sg + 1 < n_seg else 0.0)
            base = len(Vv)
            for i in range(5):
                y = s0 + (s1 - s0) * i / 4
                sag = 0.035 if i % 2 == 1 else 0.0
                for x, z in pr:
                    if z <= 1e-6:
                        Vv.append((x, y, z))
                        continue
                    r = np.array([x, z - hw * sc])
                    q = np.array([0.0, hw * sc]) + r * (1 - sag / max(np.linalg.norm(r), 1e-3))
                    Vv.append((q[0], y, q[1]))
            for i in range(4):
                for j in range(k - 1):
                    Ff.append((base + i * k + j, base + (i + 1) * k + j, base + (i + 1) * k + j + 1, base + i * k + j + 1))
            for y, r in ((s0, 0.03), (0.5 * (s0 + s1), 0.025), (s1, 0.05)):
                tubes.append((np.array([[x * 1.01, y, z * 1.01] for x, z in pr]), r))
            xw = w / 2 * sc
            for sx in (-1.0, 1.0):
                V, F = _box(sx * xw - 0.03, sx * xw + 0.03, s0, s1, 0.0, 0.2)
                mesh_object("TunnelSkirt", V, F, col, root, materials=[mat])
        _tubes_mesh("TunnelHoops", tubes, col, root, MAT.painted_metal("Tunnel_Hoop", "#d9dbdc", 0.35, 0.1, 0.0), 8)
        mesh_object("TunnelCover", Vv, Ff, col, root, smooth=True, materials=[club])
        V, F = _box(-w / 2, w / 2, ya, yb, 0.0, 0.03)
        mesh_object("TunnelMat", V, F, col, root, materials=[mat])
    socket("tunnel", "players", (0.0, float(y_end), 0.0), 0.0, parent=root, col=col, width=w, height=h)
    return {"root": root, "from_y": y_wall, "to_y": y_end}
