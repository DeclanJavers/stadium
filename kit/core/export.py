"""glTF export and asset statistics for the manifest."""
import json
from pathlib import Path

import bpy
from mathutils import Vector

from .scene import descendants
from .sockets import PREFIX


def export_glb(root, path, **gltf):
    """Export root and its descendants. Extra keyword arguments go to the glTF exporter (e.g.
    export_image_format="JPEG"); without them every asset exports as it always has."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in descendants(root):
        o.select_set(True)
    bpy.context.view_layer.objects.active = root
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True,
                              export_extras=True, export_yup=True, export_apply=True, **gltf)
    return path


def stats(root):
    tris, mats, sockets = 0, set(), []
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    inv = root.matrix_world.inverted()
    for o in descendants(root):
        if o.name.startswith(PREFIX + ":"):
            sockets.append({"name": o.name, "kind": o["socket_kind"], "id": o["socket_id"],
                            "location": [round(v, 4) for v in (inv @ o.matrix_world.translation)],
                            "yaw_deg": round(o.rotation_euler.z * 57.29578, 3)})
        if o.type != "MESH":
            continue
        me = o.data
        tris += sum(len(p.vertices) - 2 for p in me.polygons)
        mats |= {m.name for m in me.materials if m}
        for v in me.vertices:
            c = inv @ (o.matrix_world @ v.co)
            lo = Vector(map(min, lo, c))
            hi = Vector(map(max, hi, c))
    return {
        "asset_id": root.get("asset_id"), "family": root.get("asset_family"), "fits": root.get("fits"),
        "params": json.loads(root.get("params", "{}")), "triangles": tris,
        "materials": sorted(mats), "tint_slots": sorted(m for m in mats if m.startswith("TINT_")),
        "sockets": sockets, "bounds_min": [round(v, 4) for v in lo], "bounds_max": [round(v, 4) for v in hi],
    }
