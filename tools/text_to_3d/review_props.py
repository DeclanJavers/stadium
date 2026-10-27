"""Cycles review renders of finished generated props, plus a labelled contact sheet.

    blender -b --factory-startup -P tools/text_to_3d/review_props.py -- [id ...]

Each out/assets/genprop.<id>.glb is placed on grass under the kit's physical sky and shot from
the front three-quarter (+Y side) and the back three-quarter, framed to its bounding box, with a
1.8 m reference post beside it for scale. Writes out/review/genprops/<id>.png (both views side by
side) and out/review/genprops/sheet_<n>.png.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kit.core.render import camera, setup_cycles, setup_sky, surroundings  # noqa: E402
from kit.core.scene import OUT  # noqa: E402

DIR = OUT / "review" / "genprops"
SIZE = 640


def scene_for(glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_cycles(samples=96, res=(SIZE, SIZE))
    setup_sky(sun_elevation=35, sun_azimuth=150)
    surroundings(200)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Review_Ground"]
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    # scale post: 1.8 m, 4 cm square, grey, beside the prop
    bpy.ops.mesh.primitive_cube_add(size=1, location=(hi.x + 0.3 + 0.02, lo.y, 0.9))
    post = bpy.context.object
    post.scale = (0.04, 0.04, 1.8)
    m = bpy.data.materials.new("post")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
    post.data.materials.append(m)
    return lo, hi


def shoot(lo, hi, az, path):
    # Light every view from over the camera's shoulder so both sides of the prop can be judged.
    sun = bpy.data.objects["Sun"]
    s_az, s_el = az + math.radians(40), math.radians(40)
    to_sun = Vector((math.sin(s_az) * math.cos(s_el), math.cos(s_az) * math.cos(s_el), math.sin(s_el)))
    sun.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.15)
    dist = radius / math.tan(math.radians(30) / 2) * 1.15
    el = math.radians(18)
    loc = center + Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))) * dist
    loc.z = max(loc.z, 0.12)
    cam = camera("cam", tuple(loc), tuple(center), lens=36 / (2 * math.tan(math.radians(15))))
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def load(path):
    img = bpy.data.images.load(str(path))
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def save(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(path.stem, w, h)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()


def main():
    ids = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    glbs = [g for g in sorted((OUT / "assets").glob("genprop.*.glb")) if g.stem.count(".") == 1]  # not .anim
    if ids:
        glbs = [g for g in glbs if g.stem.split(".", 1)[1] in ids]
    DIR.mkdir(parents=True, exist_ok=True)
    pairs = []
    for g in glbs:
        pid = g.stem.split(".", 1)[1]
        lo, hi = scene_for(g)
        a, b = DIR / f"{pid}_front.png", DIR / f"{pid}_back.png"
        shoot(lo, hi, math.radians(35), a)       # +Y side, from front right
        shoot(lo, hi, math.radians(215), b)      # -Y side, from back left
        pair = np.concatenate([load(a), load(b)], axis=1)
        save(pair, DIR / f"{pid}.png")
        a.unlink()
        b.unlink()
        pairs.append(pair)
        print("REVIEWED", pid, flush=True)
    # contact sheets of 3 pairs per row, 4 rows (Blender rows run bottom-up, so stack reversed)
    for n in range(0, len(pairs), 12):
        chunk = pairs[n:n + 12]
        rows = []
        for r in range(0, len(chunk), 3):
            row = chunk[r:r + 3]
            row += [np.ones_like(chunk[0])] * (3 - len(row))
            rows.append(np.concatenate(row, axis=1))
        save(np.concatenate(rows[::-1], axis=0), DIR / f"sheet_{n // 12 + 1}.png")


main()
