"""Materials the training-ground generators use (mirrors viewer/gen/src/site/mats.rs). Library
materials by name (tools/export_library.py), recoloured where a colour is given; grass surfaces
carry their turf for the renderer's procedural grass; fences and nets carry a cord pattern.
"""
from .model import Mat, rgb, rround


def named(key):
    return Mat(key)


def tint(key, hexs):
    return Mat(key, rgb(hexs))


def glow(key, hexs, strength):
    return Mat(key, rgb(hexs), (rgb(hexs), strength))


def _grass(pattern, width, strength, natural, color, dry, wear, height, origin, yaw):
    return {"pattern": pattern, "stripe_width": width, "angle_deg": 0.0, "strength": strength, "natural": natural,
            "color": color, "dry": dry, "wear": wear, "wetness": 0.0, "height_mm": height, "half": (500.0, 500.0),
            "origin": tuple(origin), "yaw": yaw, "wear_zones": False}


LAWNS = {
    "field": ("none", 6.0, 0.0, "#486329", "#857c4c", 0.25, 70.0),
    "rough": ("none", 6.0, 0.0, "#535f30", "#8f8457", 0.45, 110.0),
    "fine": ("stripes", 2.4, 0.22, "#3f6a25", "#8a8247", 0.1, 8.0),
    "berm": ("none", 6.0, 0.0, "#44602a", "#857c4c", 0.2, 45.0),
    "amenity": ("stripes", 4.5, 0.03, "#43642a", "#7f7a48", 0.15, 35.0),
    "hay": ("none", 6.0, 0.0, "#66703a", "#a3975e", 0.12, 140.0),
    "stubble": ("stripes", 2.2, 0.3, "#9a8a4f", "#b8a66c", 0.05, 30.0),
}


def lawn(kind, origin, yaw):
    """A lawn: amenity (site grass), field (farmland round the site), rough (unkempt, long), fine
    (a bowling green), berm (sloped banks)."""
    pattern, width, strength, color, dry, wear, height = LAWNS.get(kind, LAWNS["amenity"])
    return Mat("Pitch_Grass", texture=("grass", _grass(pattern, width, strength, True, color, dry, wear, height, origin, yaw)))


def turf_3g(origin, yaw):
    """Artificial turf outside the lines (cage aprons): as an artificial pitch, no wear zones."""
    return Mat("Pitch_Grass", texture=("grass", _grass("stripes", 4.0, 0.18, False, "#46742c", "#5d7031", 0.1, 40.0, origin, yaw)))


def net(mesh, size_mm, cord_mm, color):
    """A cord mesh card (chain link, ball-stop netting, welded mesh): UVs in metres."""
    c = rgb(color)
    spec = {"mesh": mesh, "mesh_size_mm": size_mm, "cord_tenths_mm": int(rround(cord_mm * 10.0)), "pattern": "solid",
            "band_cells": 1, "colors": [[int(rround(c[0] * 255.0)), int(rround(c[1] * 255.0)), int(rround(c[2] * 255.0))]],
            "metric": True}
    return Mat("TINT_net_card", texture=("net", spec))


def chainlink():
    return net("diamond", 55, 3.5, "#2f4a35")


def ball_stop():
    return net("square", 100, 3.0, "#1c1f1d")


def welded_mesh():
    return net("square", 75, 4.0, "#c9ccc8")
