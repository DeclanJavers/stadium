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
from .dugout import FRONT_FROM_TOUCHLINE, _box
from .stand import _tubes_mesh, concrete, painted


def fabric(color):
    name = "TINT_tunnel"
    if bpy.data.materials.get(name):
        return bpy.data.materials[name]
    m, nb = material(name)
    pos = nb.geometry()["Position"]
    nb.output(nb.principled(Base_Color=srgb(color), Roughness=0.6, Sheen_Weight=0.5, Coat_Weight=0.2,
                            Normal=nb.bump(nb.noise(pos, 400.0, detail=1), 0.1, 0.001)))
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
    y_end = -(W / 2 + FRONT_FROM_TOUCHLINE)
    has_stand = bool(stands) and "south" in (stands.get("runs") or {})
    y_wall = -stands["B"] if has_stand else y_end - 3.0
    dark = MAT.painted_metal("Tunnel_Mouth", "#050506", rough=0.9, dirt=0.0, scuffs=0.0)
    trim = painted("Tunnel_Trim", "#1a1b1e", 0.4)
    # portal: the opening in the stand front (a dark recess with a frame)
    V, F = _box(-w / 2, w / 2, y_wall - 0.02, y_wall + 0.01, 0.0, h)
    mesh_object("TunnelMouth", V, F, col, root, materials=[dark])
    for V, F in (_box(-w / 2 - 0.25, -w / 2, y_wall, y_wall + 0.12, 0.0, h + 0.25),
                 _box(w / 2, w / 2 + 0.25, y_wall, y_wall + 0.12, 0.0, h + 0.25),
                 _box(-w / 2 - 0.25, w / 2 + 0.25, y_wall, y_wall + 0.12, h, h + 0.25)):
        mesh_object("TunnelFrame", V, F, col, root, materials=[trim])
    if not has_stand:
        V, F = _box(-w / 2 - 1.2, w / 2 + 1.2, y_wall - 4.0, y_wall, 0.0, h + 0.8)
        mesh_object("TunnelBuilding", V, F, col, root, materials=[concrete("Tunnel_Building", "#a8a399")])
    if p["style"] == "telescopic":
        cloth = fabric(p["color"])
        n_hoops = max(3, int((y_end - y_wall) / 1.1) + 1)
        ys = np.linspace(y_wall + 0.15, y_end, n_hoops)
        # straight walls under a semicircular crown
        hw = max(0.5, h - w / 2)
        prof = [(-w / 2, 0.0)] + [((w / 2) * math.cos(a), hw + (w / 2) * math.sin(a))
                                  for a in np.linspace(math.pi, 0, 17)] + [(w / 2, 0.0)]
        tubes = []
        for y in ys:
            tubes.append((np.array([[x, y, z] for x, z in prof]), 0.035))
        _tubes_mesh("TunnelHoops", tubes, col, root, MAT.painted_metal("Tunnel_Hoop", "#d9dbdc", 0.35, 0.1, 0.0), 8)
        # fabric cover, sagging slightly between hoops
        Vv, Ff = [], []
        k = len(prof)
        n = (n_hoops - 1) * 4 + 1
        for i, y in enumerate(np.linspace(ys[0], ys[-1], n)):
            sag = 0.04 * math.sin(math.pi * ((i % 4) / 4))   # fabric droops inward between hoops
            for x, z in prof:
                if z <= 1e-6:
                    Vv.append((x, y, z))
                    continue
                r = np.array([x, z - hw])
                q = np.array([0.0, hw]) + r * (1 - sag / max(np.linalg.norm(r), 1e-3))
                Vv.append((q[0], y, q[1]))
        for i in range(n - 1):
            for j in range(k - 1):
                Ff.append((i * k + j, (i + 1) * k + j, (i + 1) * k + j + 1, i * k + j + 1))
        cover = mesh_object("TunnelCover", Vv, Ff, col, root, smooth=True, materials=[cloth])
        V, F = _box(-w / 2, w / 2, ys[0], ys[-1], 0.0, 0.03)
        mesh_object("TunnelMat", V, F, col, root, materials=[MAT.rubber("Tunnel_Mat", "#1f2a22")])
    socket("tunnel", "players", (0.0, float(y_end), 0.0), 0.0, parent=root, col=col, width=w, height=h)
    return {"root": root, "from_y": y_wall, "to_y": y_end}
