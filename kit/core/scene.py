"""Scene, collection and object helpers."""
import json
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"


def collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def link(obj, col=None, parent=None):
    (col or bpy.context.scene.collection).objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def empty(name, col=None, parent=None, location=(0, 0, 0), display="PLAIN_AXES", size=0.3):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    obj.location = location
    return link(obj, col, parent)


def safe_name(name):
    """Blender parses a trailing '.digits' as a duplicate counter (and crashes on huge ones)."""
    import re
    return re.sub(r"\.(\d{4,})$", lambda m: "_" + m.group(1), str(name))


def mesh_object(name, verts, faces, col=None, parent=None, smooth=False, materials=(), mat_index=None, uvs=None):
    """Build a mesh object from numpy-friendly verts (N,3) and faces (list of index tuples or (M,k) array).

    uvs: optional per-loop UVs (L,2) in face-corner order.
    """
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    name = safe_name(name)
    me = bpy.data.meshes.new(name)
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        k = faces.shape[1]
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", verts.ravel())
        me.loops.add(faces.size)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(len(faces))
        me.polygons.foreach_set("loop_start", np.arange(0, faces.size, k, dtype=np.int32))
        me.update(calc_edges=True)
    else:
        me.from_pydata(verts.tolist(), [], [tuple(f) for f in faces])
        me.update()
    if uvs is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    for m in materials:
        me.materials.append(m)
    if mat_index is not None and len(me.polygons):
        me.polygons.foreach_set("material_index", np.asarray(mat_index, dtype=np.int32))
    if smooth:
        me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    return link(obj, col, parent)


def join_meshes(name, parts, col=None, parent=None):
    """Join (verts, faces, uvs|None, mat_index|None) tuples of quad/tri arrays into one object."""
    vs, fs, uvs, mis, off = [], [], [], [], 0
    k = parts[0][1].shape[1]
    for v, f, uv, mi in parts:
        assert f.shape[1] == k
        vs.append(v)
        fs.append(f + off)
        off += len(v)
        if uv is not None:
            uvs.append(uv)
        mis.append(np.zeros(len(f), np.int32) if mi is None else np.broadcast_to(mi, (len(f),)))
    return np.concatenate(vs), np.concatenate(fs), (np.concatenate(uvs) if uvs else None), np.concatenate(mis)


def set_params(obj, asset_id, family, fits, params):
    obj["asset_id"] = asset_id
    obj["asset_family"] = family
    obj["fits"] = fits
    obj["params"] = json.dumps(params, sort_keys=True)


def descendants(obj):
    out = [obj]
    for c in obj.children:
        out.extend(descendants(c))
    return out


def look_at_rotation(loc, target):
    from mathutils import Vector
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y").to_euler()


def bake(obj):
    """Apply all modifiers (via the evaluated mesh) and clear them."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def join(objs, name):
    """Join objects into the first one."""
    objs = [o for o in objs if o is not None]
    with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs,
                                   selected_objects=objs, object=objs[0]):
        bpy.ops.object.join()
    objs[0].name = name
    objs[0].data.name = name
    return objs[0]
