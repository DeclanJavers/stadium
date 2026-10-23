"""Corner flagpost: Law 1 post (>= 1.5 m, non-pointed top), spring or rigid base, cloth-draped flag.

Local frame: origin at the base on the corner of the lines, +Y towards the centre mark.
"""
import math

import bpy
import numpy as np

from .. import catalog
from ..core import materials as MAT
from ..core import mesh as M
from ..core.nodes import srgb
from ..core.scene import collection, empty, link, mesh_object, set_params
from .goal import duplicate

DEFAULTS = {"preset": "pro_spring", "wind": 0.15, "seed": 1}
POLE_R = 0.016


def resolve(params):
    params = dict(params or {})
    p = catalog.merge(DEFAULTS, catalog.preset("corner_flags", params.get("preset", DEFAULTS["preset"])))
    return catalog.merge(p, params)


def _pole(p, col, root):
    h = p["height"]
    colors = p["pole_colors"]
    mats = [MAT.plastic(f"TINT_pole_{'ab'[i]}", c, rough=0.3, dirt=0.4, sss=0.05) for i, c in enumerate(colors[:2])]
    prof = M.profile_ellipse(2 * POLE_R, 2 * POLE_R, 20)
    z0 = 0.16 if p["base"] == "spring" else -0.15
    bl = p["band_length"] if p["band_length"] > 0 and len(colors) > 1 else h
    zs = list(np.arange(z0, h, bl)) + [h]
    for k, (a, b) in enumerate(zip(zs[:-1], zs[1:])):
        V, F, UV = M.sweep([(0, 0, a), (0, 0, b)], prof, (1, 0, 0), caps=True)
        mesh_object(f"Pole_{k}", V, F, col, root, smooth=True, materials=[mats[k % len(mats)]], uvs=UV)
    V, F = M.uv_sphere((0, 0, h), POLE_R, 8, 20, z_min=0.0)
    mesh_object("PoleCap", V, F, col, root, smooth=True, materials=[mats[0]])
    rubber = MAT.rubber("FlagBase", "#141414")
    if p["base"] == "spring":
        # steel spring and rubber boot that let the post bend when a player runs into it
        t = np.linspace(0, 1, 400)
        turns = 9
        helix = np.stack([0.019 * np.cos(2 * np.pi * turns * t), 0.019 * np.sin(2 * np.pi * turns * t),
                          0.03 + 0.12 * t], 1)
        V, F, _ = M.sweep(helix, M.profile_ellipse(0.007, 0.007, 8), (0, 0, 1), caps=False)
        mesh_object("Spring", V, F, col, root, smooth=True, materials=[MAT.painted_metal("SpringSteel", "#2b2b2b", 0.4)])
        V, F, _ = M.sweep([(0, 0, -0.05), (0, 0, 0.035)], M.profile_ellipse(0.06, 0.06, 24), (1, 0, 0))
        mesh_object("Boot", V, F, col, root, smooth=True, materials=[rubber])
        V, F, _ = M.sweep([(0, 0, 0.14), (0, 0, 0.18)], M.profile_ellipse(0.042, 0.042, 24), (1, 0, 0))
        mesh_object("Collar", V, F, col, root, smooth=True, materials=[rubber])
    else:
        V, F, _ = M.sweep([(0, 0, -0.03), (0, 0, 0.012)], M.profile_ellipse(0.05, 0.05, 24), (1, 0, 0))
        mesh_object("GroundSleeve", V, F, col, root, smooth=True, materials=[rubber])
    return mats


def _flag_image(p):
    n = 256
    cols = [np.array(srgb(c)) for c in p["flag_colors"]]
    c0, c1 = cols[0], cols[1 % len(cols)]
    y, x = np.mgrid[0:n, 0:n] / n
    pat = p["flag_pattern"]
    if pat == "diagonal":
        m = (x + y) > 1.0
    elif pat == "checker":
        m = ((np.floor(x * 3) + np.floor(y * 3)) % 2) == 1
    elif pat == "quartered":
        m = (x < 0.5) ^ (y < 0.5)
    elif pat == "stripes":
        m = (np.floor(y * 4) % 2) == 1
    else:
        m = np.zeros_like(x, bool)
    img = np.where(m[..., None], c1, c0).astype(np.float32)
    return MAT.image_from_array(f"FlagPattern_{p['preset']}", img)


def _flag(p, col, root, pole_obj_names):
    fw, fh = p["flag_size"]
    h = p["height"]
    nx, nz = 18, 18
    x = POLE_R + 0.004 + np.linspace(0, fw, nx + 1)
    z = (h - 0.04) - np.linspace(0, fh, nz + 1)
    X, Z = np.meshgrid(x, z, indexing="ij")
    # a perfectly flat vertical sheet never buckles (gravity is in its plane): seed a ripple
    rng = np.random.default_rng(p["seed"])
    ripple = 0.006 * np.sin(X * 14 + rng.uniform(0, 6)) * (X - x[0]) / fw + rng.normal(0, 0.0008, X.shape)
    V = np.stack([X, ripple, Z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    a = i * (nz + 1) + j
    F = np.stack([a, a + nz + 1, a + nz + 2, a + 1], -1).reshape(-1, 4)
    uv_v = np.stack([(X - x[0]) / fw, 1 - (z[0] - Z) / fh], -1).reshape(-1, 2)
    UV = uv_v[F.ravel()]
    mat = MAT.fabric_image(f"TINT_flag_{p['preset']}", _flag_image(p))
    obj = mesh_object("Flag", V, F, col, root, smooth=True, materials=[mat], uvs=UV)
    pins = list(range(0, nz + 1))
    vg = obj.vertex_groups.new(name="pin")
    vg.add(pins, 1.0, "REPLACE")

    # temporary collider standing in for the pole
    Vp, Fp, _ = M.sweep([(0, 0, 0), (0, 0, h)], M.profile_ellipse(2 * POLE_R, 2 * POLE_R, 16), (1, 0, 0))
    collider = mesh_object("PoleCollider", Vp, Fp, col)
    collider.modifiers.new("Collision", "COLLISION")
    collider.collision.thickness_outer = 0.002

    mod = obj.modifiers.new("Cloth", "CLOTH")
    st = mod.settings
    st.quality = 12
    st.mass = 0.02
    st.air_damping = 1.5
    st.tension_stiffness = 15
    st.compression_stiffness = 15
    st.shear_stiffness = 5
    st.bending_stiffness = 0.03
    st.vertex_group_mass = "pin"
    mod.collision_settings.use_collision = True
    mod.collision_settings.distance_min = 0.003
    wind = None
    if p["wind"] > 0:
        # force-field settings only exist on effectors created through the operator
        bpy.ops.object.effector_add(type="WIND")
        wind = bpy.context.object
        wind.field.strength = 60 * p["wind"]
        wind.field.noise = 4.0 * p["wind"]
        wind.field.seed = p["seed"]
        wind.rotation_euler = (0, math.radians(90), math.radians(-25))
        wind.location = (-1, 0, h)
    frames = 70
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, frames
    mod.point_cache.frame_start, mod.point_cache.frame_end = 1, frames
    for f in range(1, frames + 1):
        sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg).data
    co = np.zeros(len(ev.vertices) * 3, np.float32)
    ev.vertices.foreach_get("co", co)
    sc.frame_set(1)
    obj.modifiers.remove(mod)
    obj.vertex_groups.clear()
    obj.data.vertices.foreach_set("co", co)
    obj.data.update()
    bpy.data.objects.remove(collider)
    if wind:
        bpy.data.objects.remove(wind)
    return obj


def build(params=None, col=None, template=None):
    col = col or collection("CornerFlag")
    if template is not None:
        return duplicate(template, col)
    p = resolve(params)
    root = empty("CornerFlag", col, display="ARROWS", size=0.5)
    set_params(root, f"corner_flag.{p['preset']}", "corner_flag", "corner_flag", p)
    _pole(p, col, root)
    _flag(p, col, root, None)
    root["height"] = p["height"]
    return root
