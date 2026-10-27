"""Stage 3 (local Blender): turn make_props.py results into kit-ready props.

    blender -b --factory-startup -P tools/text_to_3d/finish_props.py -- <results dir> [id ...]

<results dir>/<id>/ holds high.glb (Hi3D geometry) and paint.glb (40k faces + painted texture).
For each prop in catalog/generated_props.json:
  - front (the side the reference image shows) turned to +Y, scaled to the real size in metres,
    origin on the ground at the centre of the footprint;
  - a low-poly copy decimated to the triangle budget, smooth shaded, Smart-UV unwrapped;
  - 1024 px textures baked onto it: base colour from paint.glb, tangent normals from high.glb,
    and a glTF metallic-roughness map (G roughness, B metal) derived from the colour where the
    prop's "metal" rule says which colours are bare metal;
  - optional extras ("glass_case");
  - exported to out/assets/genprop.<id>.glb with the kit's asset extras.
Textures also land in out/text_to_3d/baked/ for inspection.
"""
import json
import math
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kit.core.export import export_glb  # noqa: E402
from kit.core.scene import OUT, set_params  # noqa: E402

PX = 1024
BAKED = OUT / "text_to_3d" / "baked"
# A/B runs: GENPROP_PAINT=paint_sr.glb GENPROP_SUFFIX=.sr finishes another painted variant
# under its own name, leaving the main asset alone.
VARIANT = {k: os.environ[e] for k, e in (("paint", "GENPROP_PAINT"), ("suffix", "GENPROP_SUFFIX")) if e in os.environ}


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return Path(a[0]), set(a[1:])


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path, name):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    for o in bpy.data.objects:
        if o not in before and o.type != "MESH":
            o.select_set(False)
    # Bake parenting/transforms into the mesh so every object lives in world space.
    for o in meshes:
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
    if len(meshes) > 1:
        with bpy.context.temp_override(active_object=meshes[0], selected_editable_objects=meshes,
                                       selected_objects=meshes, object=meshes[0]):
            bpy.ops.object.join()
    obj = meshes[0]
    for o in [o for o in bpy.data.objects if o not in before and o.type != "MESH"]:
        bpy.data.objects.remove(o)
    obj.name = name
    return obj


def bbox(obj):
    co = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(0), co.max(0)


def place(objs, ref, size, lay_flat=False):
    """Front to +Y, real size, origin at the ground centre. glTF imports put the reference
    image's view on -Y, so a half turn about Z brings it round to the kit's +Y. lay_flat turns
    the front to face up instead, for things photographed from above (gloves palm up)."""
    turn = Matrix.Rotation(math.pi, 4, "Z")
    if lay_flat:
        turn = Matrix.Rotation(math.pi / 2, 4, "X") @ turn
    for o in objs:
        o.data.transform(turn)
    lo, hi = bbox(ref)
    ext = hi - lo
    s = size["h"] / ext[2] if "h" in size else size["l"] / max(ext[0], ext[1])
    shift = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    m = Matrix.Scale(s, 4) @ Matrix.Translation(shift)
    for o in objs:
        o.data.transform(m)
        o.data.update()
    return ext * s


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def apply_mods(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    obj.modifiers.clear()
    old, obj.data = obj.data, me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def decimated(src, budget, name, weld):
    low = src.copy()
    low.data = src.data.copy()
    low.name = name
    bpy.context.collection.objects.link(low)
    low.data.materials.clear()
    if weld:
        bm = bmesh.new()
        bm.from_mesh(low.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(low.data)
        bm.free()
    low.modifiers.new("tri", "TRIANGULATE")
    apply_mods(low)
    dec = low.modifiers.new("dec", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = min(1.0, budget / max(1, tri_count(low)))
    dec.use_collapse_triangulate = True
    apply_mods(low)
    return low


def agrees(low, src):
    """Share of the low mesh's faces that face the same way as the nearest source surface."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons([v.co for v in src.data.vertices], [p.vertices[:] for p in src.data.polygons])
    polys = low.data.polygons
    step = max(1, len(polys) // 2000)
    ok = n = 0
    for i in range(0, len(polys), step):
        poly = polys[i]
        hit = tree.find_nearest(poly.center)
        if hit[0] is not None:
            n += 1
            ok += poly.normal.dot(hit[1]) > 0
    return ok / max(1, n)


def make_low(src, budget, name):
    """Decimate the painted copy to `budget` triangles. The painted glTF is split at its UV seams
    (glTF stores vertices per UV), so its surface is hundreds of separate patches, and decimated
    like that every seam opens into a hairline crack. Welding first fixes that on organic shapes,
    but on some generated hard surfaces the welded mesh tangles: it stops short of the budget and
    turns faces inside out (they bake black). So: weld, check the result against the source, and
    fall back to the unwelded mesh if it misses the budget or faces the wrong way."""
    low = decimated(src, budget, name, weld=True)
    agree = agrees(low, src)
    if tri_count(low) > 1.05 * budget or agree < 0.93:  # a tangled weld scores 0.77-0.88
        print(f"{name}: welded decimation unusable ({tri_count(low)} tris, {agree:.0%} facing right): unwelded", flush=True)
        bpy.data.objects.remove(low)
        low = decimated(src, budget, name, weld=False)
    for p in low.data.polygons:
        p.use_smooth = True
    return low


def unwrap(obj):
    """xatlas charts and packing (a few large islands, little padding waste, few seams to bleed
    at a distance); Smart UV Project if xatlas isn't installed in Blender's Python. Smart UV on a
    decimated organic mesh shatters it into thousands of tiny islands."""
    try:
        import xatlas
    except ImportError:
        print("xatlas missing: falling back to Smart UV Project", flush=True)
        return smart_unwrap(obj)
    me = obj.data
    v = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", v)
    f = np.array([p.vertices[:] for p in me.polygons], np.uint32)  # triangles (make_low triangulates)
    atlas = xatlas.Atlas()
    atlas.add_mesh(v.reshape(-1, 3), f)
    pack = xatlas.PackOptions()
    pack.resolution, pack.padding, pack.bilinear = PX, 4, True
    chart = xatlas.ChartOptions()
    chart.max_cost, chart.normal_deviation_weight = 4.0, 1.0  # generated meshes are noisy: accept bigger charts
    atlas.generate(chart_options=chart, pack_options=pack)
    print(f"xatlas: {atlas.chart_count} charts, {float(atlas.utilization):.0%} of the texture used", flush=True)
    vmap, idx, uv = atlas[0]
    if len(idx) != len(f):
        print("xatlas changed the face count: falling back", flush=True)
        return smart_unwrap(obj)
    layer = me.uv_layers.new(name="UVMap") if not me.uv_layers else me.uv_layers[0]
    loops = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loops)
    out = np.empty((len(me.loops), 2), np.float32)
    for p in me.polygons:  # xatlas keeps face order; match each corner by its original vertex
        corners = {int(vmap[k]): uv[k] for k in idx[p.index]}
        for li in p.loop_indices:
            out[li] = corners[loops[li]]
    layer.data.foreach_set("uv", out.ravel())


def smart_unwrap(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.004, area_weight=0.6)
    bpy.ops.uv.pack_islands(margin=0.004, rotate=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def new_image(name, colorspace):
    img = bpy.data.images.new(name, PX, PX, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace
    return img


def bake(low, source, kind, img, reach):
    sc = bpy.context.scene
    mat = bpy.data.materials.new(f"bake_{kind}")
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = img
    mat.node_tree.nodes.active = node
    low.data.materials.clear()
    low.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    source.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    sc.render.bake.margin = 12
    kw = dict(type=kind, use_selected_to_active=True, cage_extrusion=reach, max_ray_distance=reach * 3,
              target="IMAGE_TEXTURES", margin=12)
    if kind == "DIFFUSE":
        kw["pass_filter"] = {"COLOR"}
    else:
        kw["normal_space"] = "TANGENT"
    bpy.ops.object.bake(**kw)


def pixels(img):
    a = np.empty(PX * PX * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(PX, PX, 4)


def box(a, r):
    """Mean over a (2r+1)^2 window, edges clamped (integral image)."""
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return (s / n ** 2)[: a.shape[0], : a.shape[1]].astype(np.float32)


def metal_rough(albedo, pbr):
    """glTF metallic-roughness texture from the baked colour, and a cleaned base colour.

    Rules pick which colours are bare metal: 'silver' = unsaturated, 'medal' = that plus warm
    golds. The paint model bakes the reference photo's reflections into silver as dark smears,
    so the mask is closed over them and metal gets a clean base colour (polished silver, or the
    gold's own hue brightened): reflections then come from the scene, as they should."""
    px = pixels(albedo)
    rgb = px[..., :3]
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = (np.degrees(np.arctan2(math.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    rule = pbr.get("metal")
    silver = np.zeros(rgb.shape[:2], bool)
    gold = np.zeros(rgb.shape[:2], bool)
    if rule in ("silver", "medal"):
        silver = (sat < 0.2) & (mx > 0.3)
        silver = (box(silver, 6) > 0.45) & (sat < 0.3)   # close over dark reflections
    if rule == "medal":
        gold = (hue > 20) & (hue < 60) & (sat > 0.25) & (mx > 0.3)
        gold = (box(gold, 6) > 0.45) & (hue > 15) & (hue < 65)
    metal = silver | gold
    if rule == "all":
        metal[:] = True
    lum = rgb.mean(-1, keepdims=True)
    rgb[silver] = (np.array([0.94, 0.93, 0.90]) * (0.85 + 0.15 * lum[silver] / max(float(lum.mean()), 1e-3))).clip(0, 1)
    rgb[gold] = (rgb[gold] / np.maximum(rgb[gold].max(-1, keepdims=True), 1e-3) * 0.92).clip(0, 1)
    px[..., :3] = rgb
    albedo.pixels.foreach_set(px.ravel())
    rough = np.full(metal.shape, pbr.get("rough", 0.6), np.float32)
    rough[metal] = pbr.get("metal_rough", 0.2)
    out = np.ones((PX, PX, 4), np.float32)
    out[..., 1] = rough
    out[..., 2] = metal.astype(np.float32)
    img = new_image(albedo.name.replace("albedo", "mr"), "Non-Color")
    img.pixels.foreach_set(out.ravel())
    return img, float(metal.mean())


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()


def final_material(name, albedo, normal, mr):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t_col = nt.nodes.new("ShaderNodeTexImage")
    t_col.image = albedo
    nt.links.new(t_col.outputs["Color"], bsdf.inputs["Base Color"])
    t_mr = nt.nodes.new("ShaderNodeTexImage")
    t_mr.image = mr
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t_mr.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if normal is None:
        return mat
    t_n = nt.nodes.new("ShaderNodeTexImage")
    t_n.image = normal
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(t_n.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# Levels of detail: share of the full prop's triangle budget, and texture size. Each level is cut
# from the source meshes and unwrapped and baked on its own: decimating the finished prop instead
# collapses edges across UV seams and smears the texture.
LODS = {1: (0.35, 512), 2: (0.12, 256)}
LOD_MIN_TRIS = 250
# JPEG at q90 for every map: about a quarter of PNG with no visible loss (checked side by side)
JPEG = dict(export_image_format="JPEG", export_image_quality=90, export_jpeg_quality=90)


def glass_case(low, dims):
    """A five-sided glass cover over the whole prop, open at the floor."""
    mx, my, h = dims[0] / 2 + 0.03, dims[1] / 2 + 0.03, dims[2] + 0.04
    V = [(-mx, -my, 0), (mx, -my, 0), (mx, my, 0), (-mx, my, 0),
         (-mx, -my, h), (mx, -my, h), (mx, my, h), (-mx, my, h)]
    F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]
    me = bpy.data.meshes.new(low.name + ".glass")
    me.from_pydata(V, [], F)
    o = bpy.data.objects.new(low.name + ".glass", me)
    bpy.context.collection.objects.link(o)
    mat = bpy.data.materials.new("GenProp_Glass")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.03
    b.inputs["IOR"].default_value = 1.5
    me.materials.append(mat)
    o.parent = low
    return o


def textured(name, tag, paint, high, budget, px, pbr, reach, normals=True):
    """A finished mesh: decimated from the painted copy to `budget` triangles, unwrapped, and
    given baked colour (from paint), normals (from high) and metal/roughness at px x px.
    normals=False skips the normal map: on a coarse distant mesh the gap to the detail is too
    big and the map comes out as noise, so the geometry shades it instead."""
    global PX
    PX = px
    low = make_low(paint, budget, name)
    low.data.name = name
    unwrap(low)
    albedo = new_image(f"{tag}_albedo", "sRGB")
    bake(low, paint, "DIFFUSE", albedo, reach)
    normal = None
    if normals:
        normal = new_image(f"{tag}_normal", "Non-Color")
        bake(low, high, "NORMAL", normal, reach)
    mr, metal_share = metal_rough(albedo, pbr)
    for img in (albedo, normal, mr):
        if img is not None:
            save(img, BAKED / f"{img.name}.png")
    low.data.materials.clear()
    low.data.materials.append(final_material(f"GenProp_{tag}", albedo, normal, mr))
    return low, metal_share


def finish(p, src):
    reset()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 4
    high = import_glb(src / "high.glb", "high")
    paint = import_glb(src / VARIANT.get("paint", "paint.glb"), "paint")
    # The painted glTF leaves metallicFactor unset, which glTF reads as 1.0: a fully metallic
    # surface has no diffuse colour to bake. Only its texture matters here.
    for m in paint.data.materials:
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf:
            for link in list(bsdf.inputs["Metallic"].links):
                m.node_tree.links.remove(link)
            bsdf.inputs["Metallic"].default_value = 0.0
    dims = place([high, paint], high, p["size"], p.get("lay_flat", False))
    name = f"genprop.{p['id']}{VARIANT.get('suffix', '')}"
    tag = p["id"] + VARIANT.get("suffix", "")
    reach = 0.015 * float(max(dims))
    low, metal_share = textured(name, tag, paint, high, p["budget"], 1024, p.get("pbr", {}), reach)
    lods = []
    for level, (share, px) in LODS.items():
        budget = max(LOD_MIN_TRIS, int(p["budget"] * share))
        # a coarser surface strays further from the detailed one: search further for it
        lods.append((level, textured(f"{name}.lod{level}", f"{tag}.lod{level}", paint, high, budget, px,
                                     p.get("pbr", {}), reach * share ** -0.75, normals=False)[0]))
    if p.get("glass_case"):
        glass_case(low, dims)
    bpy.data.objects.remove(high)
    bpy.data.objects.remove(paint)
    set_params(low, name, "genprop", "ground", {"group": p["group"], "size_m": [round(float(d), 3) for d in dims],
                                                "source": "text_to_3d"})
    path = export_glb(low, OUT / "assets" / f"{name}.glb", **JPEG)
    lod_tris = {}
    for level, lo in lods:
        set_params(lo, f"{name}.lod{level}", "genprop", "ground",
                   {"group": p["group"], "size_m": [round(float(d), 3) for d in dims], "source": "text_to_3d", "lod": level})
        export_glb(lo, OUT / "assets" / f"{name}.lod{level}.glb", **JPEG)
        lod_tris[level] = tri_count(lo)
    stats = {"id": tag, "tris": tri_count(low), "lod_tris": lod_tris, "dims_m": [round(float(d), 3) for d in dims],
             "metal_share": round(metal_share, 3), "glb": str(path.relative_to(ROOT))}
    print("FINISHED", json.dumps(stats), flush=True)
    return stats


def main():
    results, only = args()
    spec = json.loads((ROOT / "catalog" / "generated_props.json").read_text())
    done = []
    for p in spec["props"]:
        if only and p["id"] not in only:
            continue
        src = results / p["id"]
        if not (src / "high.glb").exists() or not (src / VARIANT.get("paint", "paint.glb")).exists():
            print("MISSING", p["id"], flush=True)
            continue
        try:
            done.append(finish(p, src))
        except Exception as exc:
            print("FAILED", p["id"], type(exc).__name__, exc, flush=True)
    report = OUT / "text_to_3d" / "finish_report.json"
    old = json.loads(report.read_text()) if report.exists() else {}
    old.update({d["id"]: d for d in done})
    report.write_text(json.dumps(old, indent=1))


main()
