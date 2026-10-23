"""Turn an assembled (render-quality) stadium into game-ready glTF content.

1. Every procedural material is replaced by plain glTF PBR:
   - large surfaces: baked tileable albedo / roughness / normal, mapped in metres
     (KHR_texture_transform carries the tile size);
   - TINT_ surfaces: a greyscale detail texture times a colour factor, so engines can recolour
     them at runtime (the material's extras carry the gain to use);
   - the pitch: a full-surface albedo bake (wear, variation) plus tiling grass normal detail;
     mowing stripes stay an engine shader (docs/SPECS.md);
   - small props: measured average albedo and roughness.
2. Meshes using textured materials without UVs get metric box-projected UVs.
3. Bakes are cached by a fingerprint of each material's node values.
"""
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np

from . import bake as BK
from .scene import OUT

TILEABLE = {
    "Stand_Concrete": (4.0, False), "Stand_Block": (4.0, False), "Stand_Soffit": (4.0, False),
    "Facade_Concrete": (4.0, False), "Facade_Core": (4.0, False), "Facade_Stone": (4.0, False),
    "Facade_Brick": (3.6, True), "Dugout_Brick": (3.6, True), "Plaza_Paving": (3.6, True),
    "Apron_Concrete": (4.0, False), "Apron_Track": (4.0, False), "Apron_Turf": (4.0, False),
    "Roof_Underside": (4.0, False), "TINT_roof": (4.0, False), "TINT_roof_membrane": (4.0, False),
    "TINT_fascia": (4.0, False), "TINT_facade": (4.0, False), "TINT_facade_b": (4.0, False),
    "Dugout_Slab": (4.0, False), "Tunnel_Building": (4.0, False),
}
KEEP = ("Ads_", "Carpet_", "Screen_LED", "TINT_flag", "FlagPattern", "Balustrade_Glass",
        "Roof_Polycarbonate", "Dugout_Polycarbonate", "Facade_DarkGlass", "Fixture_Lens", "TINT_net_card")


def fingerprint(mat):
    h = hashlib.sha1()
    for n in mat.node_tree.nodes:
        h.update(n.bl_idname.encode())
        for s in n.inputs:
            if not s.is_linked and hasattr(s, "default_value"):
                v = s.default_value
                h.update(repr(tuple(v) if hasattr(v, "__len__") else v).encode())
        if n.bl_idname == "ShaderNodeGroup" and n.node_tree:
            h.update(n.node_tree.name.encode())
    return h.hexdigest()[:12]


def _cache():
    p = BK.TEX / "cache.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(c):
    BK.TEX.mkdir(parents=True, exist_ok=True)
    (BK.TEX / "cache.json").write_text(json.dumps(c, indent=2))


def _principled_defaults(mat):
    p = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if not p:
        return 0.0, 0.0, None
    met = p.inputs["Metallic"].default_value if not p.inputs["Metallic"].is_linked else 0.0
    tr = p.inputs["Transmission Weight"].default_value if not p.inputs["Transmission Weight"].is_linked else 0.0
    return met, tr, p


def _lum(c):
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def _tint_split(info):
    """Greyscale detail texture + colour factor for a TINT_ albedo bake."""
    path = OUT / info["maps"]["albedo"]
    img = bpy.data.images.load(str(path))
    w, h = img.size
    a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    rgb = np.where(a[..., :3] <= 0.04045, a[..., :3] / 12.92, ((a[..., :3] + 0.055) / 1.055) ** 2.4)
    lum = _lum(rgb)
    ratio = lum / max(1e-6, lum.mean())
    gain = float(ratio.max())
    detail = np.repeat((ratio / gain)[..., None], 3, -1)
    out = np.concatenate([detail, np.ones_like(detail[..., :1])], -1)
    dpath = path.with_name("detail.png")
    BK._save(out, dpath, srgb=True)
    mean = np.asarray(info["albedo_mean"])
    return str(dpath.relative_to(OUT)), (mean * gain).tolist(), gain


def _img(path, colorspace):
    img = bpy.data.images.load(str(OUT / path), check_existing=True)
    img.colorspace_settings.name = colorspace
    return img


def _rebuild(mat, base_rgb, rough, metallic, maps=None, size=None, offset=None, extend=False, tint_gain=None,
             transmission=0.0):
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(p.outputs[0], out.inputs["Surface"])
    p.inputs["Base Color"].default_value = (*base_rgb[:3], 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metallic
    if transmission:
        p.inputs["Transmission Weight"].default_value = transmission
    if maps:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1 / size[0], 1 / size[1], 1.0)
        if offset:
            mp.inputs["Location"].default_value = (offset[0], offset[1], 0.0)
        nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])

        def tex(path, cs):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = _img(path, cs)
            n.extension = "EXTEND" if extend else "REPEAT"
            nt.links.new(mp.outputs[0], n.inputs["Vector"])
            return n
        if "albedo" in maps:
            t = tex(maps["albedo"], "sRGB")
            if tint_gain is not None:
                mix = nt.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = 1.0
                nt.links.new(t.outputs["Color"], mix.inputs[6])
                mix.inputs[7].default_value = (*base_rgb[:3], 1.0)
                nt.links.new(mix.outputs[2], p.inputs["Base Color"])
                mat["tint_gain"] = round(tint_gain, 4)
            else:
                nt.links.new(t.outputs["Color"], p.inputs["Base Color"])
        if "roughness" in maps:
            t = tex(maps["roughness"], "Non-Color")
            nt.links.new(t.outputs["Color"], p.inputs["Roughness"])
        if "normal" in maps:
            t = tex(maps["normal"], "Non-Color")
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(t.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs[0], p.inputs["Normal"])
    mat["game_material"] = True


def box_uv(obj, scale=1.0):
    """Metric box projection: each face takes the two world axes it is most square to."""
    me = obj.data
    mw = np.array(obj.matrix_world)
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    layer = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    uv = np.zeros((len(me.loops), 2), np.float32)
    nrm = np.zeros(len(me.polygons) * 3)
    me.polygons.foreach_get("normal", nrm)
    nrm = np.abs(nrm.reshape(-1, 3) @ mw[:3, :3].T)
    loop_vert = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_vert)
    for p in me.polygons:
        ax = int(np.argmax(nrm[p.index]))
        keep = [i for i in range(3) if i != ax]
        for li in range(p.loop_start, p.loop_start + p.loop_total):
            uv[li] = co[loop_vert[li], keep] * scale
    layer.data.foreach_set("uv", uv.ravel())


def prepare(pitch_params=None, log=print):
    """Convert every material in the scene to game PBR. Returns a report dict."""
    cache = _cache()
    report = {}
    users = {}
    for o in bpy.data.objects:
        if o.type == "MESH":
            for m in o.data.materials:
                if m:
                    users.setdefault(m.name, []).append(o)
    for name, objs in sorted(users.items()):
        mat = bpy.data.materials[name]
        if mat.get("game_material") or any(name.startswith(k) for k in KEEP):
            continue
        has_img = any(n.bl_idname == "ShaderNodeTexImage" for n in mat.node_tree.nodes)
        if has_img:
            continue
        fp = fingerprint(mat)
        key = f"{name}:{fp}"
        met, trans, _ = _principled_defaults(mat)
        if name == "TINT_Pitch_Paint":
            # line paint: white (the coverage breakup is a render-scale detail)
            _rebuild(mat, (0.82, 0.83, 0.8), 0.6, 0.0)
            report[name] = "factor"
            continue
        if name == "Pitch_Grass" and pitch_params:
            sx = pitch_params["length"] + 2 * pitch_params["runoff_end"]
            sy = pitch_params["width"] + 2 * pitch_params["runoff_side"]
            info = cache.get(key) or BK.bake_pitch(mat, sx, sy)
            detail = cache.get(key + ":detail") or BK.bake_tileable(mat, 2.0, 1024, z=0.0, origin=(18.0, 18.0))
            cache[key], cache[key + ":detail"] = info, detail
            _rebuild(mat, (1, 1, 1), info["roughness_mean"], 0.0, maps={"albedo": info["maps"]["albedo"]},
                     size=(sx, sy), offset=(0.5, 0.5), extend=True)
            # tiling blade normal on its own (finer) mapping
            nt = mat.node_tree
            p = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
            tc = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexCoord")
            mp2 = nt.nodes.new("ShaderNodeMapping")
            mp2.inputs["Scale"].default_value = (0.5, 0.5, 1.0)
            nt.links.new(tc.outputs["UV"], mp2.inputs["Vector"])
            t = nt.nodes.new("ShaderNodeTexImage")
            t.image = _img(detail["maps"]["normal"], "Non-Color")
            nt.links.new(mp2.outputs[0], t.inputs["Vector"])
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(t.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs[0], p.inputs["Normal"])
            mat["stripe_shader"] = "docs/SPECS.md#grass"
            report[name] = "pitch"
        elif name in TILEABLE:
            size, exact = TILEABLE[name]
            info = cache.get(key) or BK.bake_tileable(mat, size, 1024, exact=exact)
            cache[key] = info
            maps = dict(info["maps"])
            gain = None
            base = (1, 1, 1)
            if name.startswith("TINT_"):
                maps["albedo"], base, gain = _tint_split(info)
            _rebuild(mat, base, info["roughness_mean"], met, maps=maps, size=(size, size), tint_gain=gain)
            for o in objs:
                if not o.data.uv_layers:
                    box_uv(o)
            report[name] = "tileable"
        else:
            info = cache.get(key) or BK.bake_factor(mat)
            cache[key] = info
            _rebuild(mat, info["albedo_mean"], info["roughness_mean"], met, transmission=trans)
            report[name] = "factor"
        _save_cache(cache)
    bs = bpy.data.scenes.get("BakeScene")
    if bs:
        for o in list(bs.objects):
            bpy.data.objects.remove(o)
        bpy.data.scenes.remove(bs)
    log(f"[game] materials: {sum(v == 'tileable' for v in report.values())} tileable, "
        f"{sum(v == 'factor' for v in report.values())} factor-only, pitch baked: {'Pitch_Grass' in report}")
    return report


def area_lights_to_spots():
    """glTF has no area lights: replace each with a wide downward spot of matching intensity.
    A one-sided area light of power P has peak intensity P/pi W/sr; a Blender spot of power Q has
    Q/(4 pi), so Q = 4 P keeps the light on the seats the same."""
    import math
    n = 0
    for o in list(bpy.data.objects):
        if o.type == "LIGHT" and o.data.type == "AREA":
            spot = bpy.data.lights.get("BowlSpot") or bpy.data.lights.new("BowlSpot", "SPOT")
            spot.energy = 4.0 * o.data.energy
            spot.spot_size = math.radians(120)
            spot.spot_blend = 1.0
            spot.color = o.data.color
            s = bpy.data.objects.new(o.name.replace("BowlLight", "BowlSpot"), spot)
            for c in o.users_collection:
                c.objects.link(s)
            s.parent = o.parent
            s.matrix_world = o.matrix_world.copy()
            bpy.data.objects.remove(o)
            n += 1
    return n
