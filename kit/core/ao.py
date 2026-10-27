"""Ambient occlusion baked into vertex colours (glTF COLOR_0, which engines multiply into the base
colour): creases, undersides and the ground contact of a prop darken as they do in daylight, so
props sit in a scene instead of floating on it.

Each object is baked on its own, standing on a ground plane, in a scene of its own (props are
built on top of one another at the origin).
"""
import bmesh
import bpy
import numpy as np

from .scene import mesh_object


def _scene():
    sc = bpy.data.scenes.get("AOScene")
    if sc:
        return sc
    sc = bpy.data.scenes.new("AOScene")
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = False
    w = bpy.data.worlds.get("AOWorld") or bpy.data.worlds.new("AOWorld")
    sc.world = w
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"
    g = mesh_object("AOGround", [(-60, -60, 0), (60, -60, 0), (60, 60, 0), (-60, 60, 0)], [(0, 1, 2, 3)], col=sc.collection)
    g["ao_ground"] = True
    return sc


def bake(obj, distance=0.6, strength=0.8, floor=0.3, ground=True):
    """Bake AO into obj's point colours ('Col'): 1 = open sky; mixed back towards 1 by `strength`.

    The bake runs on a copy whose face normals are made consistently outward (a face wound inwards,
    as some lofts and sweeps make them, would otherwise bake as fully enclosed and come out black);
    the result is copied back point for point, so the prop's own mesh is left as it was."""
    me = obj.data
    if not len(me.vertices):
        return
    sc = _scene()
    sc.world.light_settings.distance = distance
    g = sc.objects.get("AOGround")
    g.hide_render = not ground
    attr = me.color_attributes.get("Col") or me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    tmp = me.copy()
    bm = bmesh.new()
    bm.from_mesh(tmp)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(tmp)
    bm.free()
    tattr = tmp.color_attributes.get("Col")
    tmp.color_attributes.active_color = tattr
    probe = bpy.data.objects.new(obj.name + "_ao", tmp)
    probe.matrix_world = obj.matrix_world
    sc.collection.objects.link(probe)
    vl = sc.view_layers[0]
    for o in sc.objects:
        o.select_set(o == probe, view_layer=vl)
    with bpy.context.temp_override(scene=sc, view_layer=vl, active_object=probe, object=probe,
                                   selected_objects=[probe], selected_editable_objects=[probe]):
        bpy.ops.object.bake(type="AO", target="VERTEX_COLORS")
    c = np.zeros(len(tattr.data) * 4, np.float32)
    tattr.data.foreach_get("color", c)
    bpy.data.objects.remove(probe)
    bpy.data.meshes.remove(tmp)
    me.color_attributes.active_color = attr
    c = c.reshape(-1, 4)
    ao = np.clip(c[:, 0], 0, 1)
    ao = 1 - strength * (1 - ao)
    ao = floor + (1 - floor) * ao
    c[:, :3] = ao[:, None]
    c[:, 3] = 1
    attr.data.foreach_set("color", c.ravel())
