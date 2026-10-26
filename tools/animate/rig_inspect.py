"""Bring a UniRig result onto the finished, textured prop and show its skeleton for labelling.

    blender -b --factory-startup -P tools/animate/rig_inspect.py -- <rigged.glb> <genprop.glb> <out dir>

UniRig rigs an untextured copy of the prop (same geometry, same frame). This imports both, copies
the skin weights onto the textured mesh (nearest face, interpolated), and writes:
  <out>/<id>.rig.blend   armature + textured skinned mesh, ready for animate_creatures.py
  <out>/<id>.bones.json  every bone: index, name, parent, head/tail in kit space (Z up, front +Y)
  <out>/<id>.bones_front.png, .bones_side.png   the skeleton over the mesh, bones numbered
"""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    return Path(a[0]), Path(a[1]), Path(a[2])


def imported(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [o for o in bpy.data.objects if o not in before]


def main():
    rigged, prop, out = args()
    pid = prop.stem.split(".", 1)[1]
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    new = imported(rigged)
    arm = next(o for o in new if o.type == "ARMATURE")
    src = next(o for o in new if o.type == "MESH")
    new_t = imported(prop)
    tex = next(o for o in new_t if o.type == "MESH")
    for o in new_t:
        if o.type != "MESH":
            bpy.data.objects.remove(o)
    # Bake parenting so the textured mesh sits in world space like the rig.
    mw = tex.matrix_world.copy()
    tex.parent = None
    tex.matrix_world = mw

    # Copy weights: same vertex groups, nearest-face interpolation onto the textured mesh.
    for vg in src.vertex_groups:
        tex.vertex_groups.new(name=vg.name)
    dt = tex.modifiers.new("weights", "DATA_TRANSFER")
    dt.object = src
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    with bpy.context.temp_override(object=tex, active_object=tex):
        bpy.ops.object.modifier_apply(modifier=dt.name)
    bpy.data.objects.remove(src)
    tex.name = f"genprop.{pid}"
    mod = tex.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    tex.parent = arm
    tex.matrix_parent_inverse = arm.matrix_world.inverted()
    arm.name = f"genprop.{pid}.rig"

    bones = []
    names = [b.name for b in arm.data.bones]
    for i, b in enumerate(arm.data.bones):
        h, t = arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local
        bones.append({"i": i, "name": b.name, "parent": names.index(b.parent.name) if b.parent else None,
                      "head": [round(c, 4) for c in h], "tail": [round(c, 4) for c in t]})
    (out / f"{pid}.bones.json").write_text(json.dumps(bones, indent=1))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{pid}.rig.blend"))

    # Labelled skeleton render: a red stick per bone, its index at the midpoint.
    mat = bpy.data.materials.new("bone")
    mat.diffuse_color = (0.9, 0.1, 0.1, 1)
    size = max(tex.dimensions)
    for b in bones:
        h, t = Vector(b["head"]), Vector(b["tail"])
        d = t - h
        bpy.ops.mesh.primitive_cylinder_add(radius=size * 0.006, depth=max(d.length, 1e-4), location=(h + t) / 2)
        c = bpy.context.object
        c.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        c.data.materials.append(mat)
        c.show_in_front = True
        bpy.ops.object.text_add(location=(h + t) / 2 + Vector((0, 0, size * 0.015)))
        txt = bpy.context.object
        txt.data.body = str(b["i"])
        txt.data.size = size * 0.035
        txt.data.materials.append(mat)
    tex.hide_render = False
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_xray = True
    sc.display.shading.xray_alpha = 0.35
    sc.render.resolution_x, sc.render.resolution_y = 1000, 1000
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    lo = Vector([min(v[i] for b in bones for v in (b["head"], b["tail"])) for i in range(3)])
    hi = Vector([max(v[i] for b in bones for v in (b["head"], b["tail"])) for i in range(3)])
    center = (lo + hi) / 2
    cam.data.ortho_scale = size * 1.25
    for tag, d in (("front", Vector((0, 1, 0))), ("side", Vector((1, 0, 0)))):
        cam.location = center + d * size * 3
        cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
        for o in sc.objects:  # text shares the camera's orientation, so it reads facing it
            if o.type == "FONT":
                o.rotation_euler = cam.rotation_euler
        sc.render.filepath = str(out / f"{pid}.bones_{tag}.png")
        bpy.ops.render.render(write_still=True)
    print("RIGGED", pid, len(bones), "bones", flush=True)


main()
