"""Film strips of every action in the exported animated props, for review.

    blender -b --factory-startup -P tools/animate/review_anims.py -- [id ...] [--frames 8] [--mp4]
    blender -b --factory-startup -P tools/animate/review_anims.py -- --unimate <file.glb ...> [--mp4]

Imports out/assets/genprop.<id>.anim.glb (so the export itself is what gets checked), lights it
with the kit's physical sky on grass, and renders evenly spaced frames of each action from a
three-quarter side view: out/review/anims/<id>.<action>.png (a strip, left to right). --mp4
also writes a looping EEVEE preview video per action. --unimate reviews UniMate's animated GLBs
instead (canonical frame: facing glTF +Z, i.e. Blender -Y) as out/review/anims/unimate.<file>.png.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kit.core.render import camera, setup_cycles, setup_sky, surroundings  # noqa: E402

CFG = json.loads((ROOT / "catalog" / "creature_anims.json").read_text())
DIR = ROOT / "out" / "review" / "anims"
SIZE = 420


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    frames = int(a[a.index("--frames") + 1]) if "--frames" in a else 8
    ids = [x for x in a if not x.startswith("--") and not x.isdigit()]
    return ids, frames, "--mp4" in a, "--unimate" in a


def load(path):
    img = bpy.data.images.load(str(path))
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(SIZE, SIZE, 4)


def save(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(path.stem, w, h)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()


def fcurves(act):
    """An action's F-curves, in Blender 5's layered actions and in the old flat ones."""
    if getattr(act, "layers", None):
        for layer in act.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    yield from bag.fcurves
    else:
        yield from act.fcurves


def review(pid, n_frames, mp4, glb=None, fwd=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_cycles(samples=48, res=(SIZE, SIZE))
    setup_sky(sun_elevation=40, sun_azimuth=150)
    surroundings(60)
    bpy.ops.import_scene.gltf(filepath=str(glb or ROOT / "out" / "assets" / f"genprop.{pid}.anim.glb"))
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Review_Ground"]
    mesh = max(meshes, key=lambda o: len(o.data.vertices))
    for o in meshes:  # UniMate's canonical files carry a marker sphere
        o.hide_render = o is not mesh
    fwd = Vector(fwd or CFG["creatures"][pid]["forward"])
    left = Vector((0, 0, 1)).cross(fwd)
    size = max(mesh.dimensions)
    center = Vector((0, 0, mesh.dimensions.z * 0.42))
    d = (left * math.cos(math.radians(35)) + fwd * math.sin(math.radians(35))).normalized()
    loc = center + (d * math.cos(math.radians(14)) + Vector((0, 0, math.sin(math.radians(14))))) * size * 1.95
    cam = camera("cam", tuple(loc), tuple(center), lens=50)
    bpy.context.scene.camera = cam
    actions = [a for a in bpy.data.actions]
    arm.animation_data_create()
    for tr in list(arm.animation_data.nla_tracks):
        arm.animation_data.nla_tracks.remove(tr)
    DIR.mkdir(parents=True, exist_ok=True)
    for act in actions:
        name = act.name.split("|")[-1] if "|" in act.name else act.name
        arm.animation_data.action = act
        if hasattr(arm.animation_data, "action_slot") and act.slots:
            arm.animation_data.action_slot = act.slots[0]
        f0, f1 = (int(v) for v in act.frame_range)
        tiles = []
        for k in range(n_frames):
            f = f0 + round((f1 - f0) * k / n_frames)
            bpy.context.scene.frame_set(f)
            tmp = DIR / f"_{pid}_{k}.png"
            bpy.context.scene.render.filepath = str(tmp)
            bpy.ops.render.render(write_still=True)
            tiles.append(load(tmp))
            tmp.unlink()
        save(np.concatenate(tiles, axis=1), DIR / f"{pid}.{name}.png")
        if mp4:
            sc = bpy.context.scene
            eng, sc.render.engine = sc.render.engine, "BLENDER_EEVEE"
            # Repeat short cycles to about 3 s so the preview shows the loop, not one pass.
            for fc in fcurves(act):
                if not any(m.type == "CYCLES" for m in fc.modifiers):
                    fc.modifiers.new("CYCLES")
            sc.frame_start, sc.frame_end = f0, f0 + max(f1 - f0, round(72 / (f1 - f0)) * (f1 - f0)) - 1
            if hasattr(sc.render.image_settings, "media_type"):  # Blender 5: video is its own media type
                sc.render.image_settings.media_type = "VIDEO"
            sc.render.image_settings.file_format = "FFMPEG"
            sc.render.ffmpeg.format = "MPEG4"
            sc.render.ffmpeg.codec = "H264"
            sc.render.filepath = str(DIR / f"{pid}.{name}.mp4")
            bpy.ops.render.render(animation=True)
            sc.render.engine = eng
            if hasattr(sc.render.image_settings, "media_type"):
                sc.render.image_settings.media_type = "IMAGE"
            sc.render.image_settings.file_format = "PNG"
        print("REVIEWED", pid, name, f0, f1, flush=True)


def main():
    ids, n_frames, mp4, unimate = args()
    if unimate:
        for f in ids:
            review("unimate." + Path(f).stem, n_frames, mp4, glb=f, fwd=(0, -1, 0))
        return
    for pid in ids or CFG["creatures"]:
        if (ROOT / "out" / "assets" / f"genprop.{pid}.anim.glb").exists():
            review(pid, n_frames, mp4)


main()
