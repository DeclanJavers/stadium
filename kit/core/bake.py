"""Bake procedural Blender materials into game-ready PBR data.

- Tileable texture sets (albedo, roughness, normal) for large surfaces, baked from a plane with
  Cycles and made seamless by cross-fading with a half-offset copy (noise-based materials) or by
  baking an exact multiple of the pattern (bricks, profiled sheeting).
- Measured average albedo / roughness for small props whose shading is subtle (factor-only).
- A full-pitch albedo bake (wear, variation) so the engine's grass shader only adds stripes.

Everything lands in out/textures/ and is described by out/textures/manifest.json.
"""
import json
from pathlib import Path

import bpy
import numpy as np

from .scene import OUT, link, mesh_object

TEX = OUT / "textures"


def _scene():
    """A scene holding nothing but the bake plane, so Cycles never syncs the whole stadium."""
    sc = bpy.data.scenes.get("BakeScene") or bpy.data.scenes.new("BakeScene")
    return sc


def _plane(name, size, z=2.0, origin=(0.0, 0.0)):
    x0, y0 = origin
    V = [(x0, y0, z), (x0 + size[0], y0, z), (x0 + size[0], y0 + size[1], z), (x0, y0 + size[1], z)]
    o = mesh_object(name, V, [(0, 1, 2, 3)], col=_scene().collection, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    return o


def _setup(samples=8):
    sc = _scene()
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.render.bake.margin = 4
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"


def _bake(obj, mat, kind, px, colorspace):
    """Bake one pass of `mat` on `obj` into a new image; returns an (H, W, 4) float array."""
    img = bpy.data.images.new(f"bake_{mat.name}_{kind}", px[0], px[1], alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Linear Rec.709" if colorspace == "linear" else "Non-Color"
    nt = mat.node_tree
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    nt.nodes.active = node
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    sc = _scene()
    vl = sc.view_layers[0]
    obj.select_set(True, view_layer=vl)
    with bpy.context.temp_override(scene=sc, view_layer=vl, active_object=obj, object=obj,
                                   selected_objects=[obj], selected_editable_objects=[obj]):
        if kind == "albedo":
            sc.render.bake.use_pass_direct = False
            sc.render.bake.use_pass_indirect = False
            sc.render.bake.use_pass_color = True
            bpy.ops.object.bake(type="DIFFUSE")
        elif kind == "roughness":
            bpy.ops.object.bake(type="ROUGHNESS")
        elif kind == "normal":
            sc.render.bake.normal_space = "TANGENT"
            bpy.ops.object.bake(type="NORMAL")
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(px[1], px[0], 4)
    nt.nodes.remove(node)
    bpy.data.images.remove(img)
    return arr


def seamless(arr, feather=0.22):
    """Make an image tile by cross-fading with a half-offset copy near its edges."""
    h, w = arr.shape[:2]
    rolled = np.roll(np.roll(arr, h // 2, 0), w // 2, 1)
    yy = np.minimum(np.arange(h), np.arange(h)[::-1]) / (h * feather)
    xx = np.minimum(np.arange(w), np.arange(w)[::-1]) / (w * feather)
    wy = np.clip(yy, 0, 1)[:, None]
    wx = np.clip(xx, 0, 1)[None, :]
    wgt = (wy * wx)[..., None]
    wgt = wgt * wgt * (3 - 2 * wgt)
    return arr * wgt + rolled * (1 - wgt)


def _save(arr, path, srgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    a = arr.copy()
    if srgb:
        rgb = np.clip(a[..., :3], 0, 1)
        a[..., :3] = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    a[..., 3] = 1.0
    img = bpy.data.images.new(path.stem, w, h, alpha=False)
    img.pixels.foreach_set(np.clip(a, 0, 1).astype(np.float32).ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return path


def bake_tileable(mat, size_m=4.0, px=1024, exact=False, z=2.0, origin=(0.0, 0.0)):
    """Albedo / roughness / normal tileable set for `mat` covering size_m x size_m metres."""
    _setup()
    plane = _plane("BakePlane", (size_m, size_m), z, origin)
    out = {}
    for kind, space in (("albedo", "linear"), ("roughness", "data"), ("normal", "data")):
        arr = _bake(plane, mat, kind, (px, px), space)
        if not exact:
            arr = seamless(arr)
            if kind == "normal":
                n = arr[..., :3] * 2 - 1
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
                arr[..., :3] = n * 0.5 + 0.5
        out[kind] = arr
    bpy.data.objects.remove(plane)
    folder = TEX / mat.name
    paths = {k: str(_save(v, folder / f"{k}.png", srgb=(k == "albedo")).relative_to(OUT)) for k, v in out.items()}
    return {"type": "tileable", "size_m": size_m, "maps": paths,
            "albedo_mean": out["albedo"][..., :3].mean((0, 1)).round(4).tolist(),
            "roughness_mean": round(float(out["roughness"][..., 0].mean()), 3)}


def bake_factor(mat, z=2.0):
    """Average albedo and roughness of a small-prop material (for factor-only game materials)."""
    _setup(samples=4)
    plane = _plane("BakePlane", (1.0, 1.0), z)
    alb = _bake(plane, mat, "albedo", (64, 64), "linear")
    rough = _bake(plane, mat, "roughness", (64, 64), "data")
    bpy.data.objects.remove(plane)
    return {"type": "factor", "albedo_mean": alb[..., :3].mean((0, 1)).round(4).tolist(),
            "roughness_mean": round(float(rough[..., 0].mean()), 3)}


def bake_pitch(mat, length, width, px_per_m=16):
    """Whole-surface albedo of the pitch (wear, variation, no view-dependent stripes)."""
    _setup(samples=8)
    plane = _plane("BakePitch", (length, width), 0.0, (-length / 2, -width / 2))
    px = (int(length * px_per_m), int(width * px_per_m))
    alb = _bake(plane, mat, "albedo", px, "linear")
    rough = _bake(plane, mat, "roughness", (px[0] // 4, px[1] // 4), "data")
    bpy.data.objects.remove(plane)
    folder = TEX / mat.name
    return {"type": "pitch", "size_m": [length, width],
            "maps": {"albedo": str(_save(alb, folder / "albedo_full.png", True).relative_to(OUT))},
            "albedo_mean": alb[..., :3].mean((0, 1)).round(4).tolist(),
            "roughness_mean": round(float(rough[..., 0].mean()), 3)}


def write_manifest(entries):
    TEX.mkdir(parents=True, exist_ok=True)
    path = TEX / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(entries)
    path.write_text(json.dumps(old, indent=2))
    return path
