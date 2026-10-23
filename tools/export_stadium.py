"""Assemble a layout and export the whole stadium as one .glb.

Seats ship as one mesh per style variant plus EXT_mesh_gpu_instancing arrays (position, rotation,
stretch, _SEAT_COLOR), so a 150k-seat bowl stays a few tens of MB and draws in a handful of calls.

blender -b -P tools/export_stadium.py -- layouts/premier.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from kit.assembler import assemble  # noqa: E402
from kit.assets import seat as SEAT  # noqa: E402
from kit.core.gltf_instancing import add_instances, add_lods, count_instances  # noqa: E402
from kit.core.scene import OUT, collection, empty, link, reset  # noqa: E402


def main(path, game=False, seat_lod=None):
    t = time.time()
    reset()
    layout = json.loads(path.read_text())
    if game and layout.get("floodlights"):
        layout["floodlights"] = dict(layout["floodlights"], on=True)   # ship the lights; the game switches them
    st = assemble(layout)
    stands = st.get("stands")
    for o in list(bpy.data.objects):  # render-only helpers, templates and GN seat clouds never ship
        if o.get("render_only") or o.get("seat_style") or o.get("template"):
            bpy.data.objects.remove(o)
    groups = {}
    lods = {}
    if stands:
        col = collection("SeatInstances")
        holder = empty("SeatInstances", col)
        anchors = stands["anchors"]
        colors = np.asarray(stands["seat_colors"])
        for style in sorted({a["style"] for a in anchors}):
            spec = SEAT.resolve(style)
            models = bpy.data.collections[f"SeatModels_{style}"].objects
            for vi, v in enumerate(spec["variants"]):
                idx = [i for i, a in enumerate(anchors) if a["style"] == style
                       and SEAT.variant_index(spec, a["seat"], a["of"]) == vi]
                name = f"SEATS:{style}:{v}"
                src = next(o for o in models if o.name.endswith(f"_{vi:02d}_{v}"))
                level = (1 if game else 0) if seat_lod is None else seat_lod
                if level:   # keep only the mesh: the helper object would ship as a stray seat
                    lo = SEAT.lod(src, level, col)
                    mesh = lo.data
                    bpy.data.objects.remove(lo)
                else:
                    mesh = src.data
                link(bpy.data.objects.new(name, mesh), col, holder)
                if game and level < 2:   # a coarser level for distance, linked by MSFT_lod
                    far = SEAT.lod(src, 2, col)
                    link(bpy.data.objects.new(name + ":LOD2", far.data), col, holder)
                    bpy.data.objects.remove(far)
                    lods[name] = name + ":LOD2"
                groups[name] = {"pos": [anchors[i]["pos"] for i in idx],
                                "yaw": [anchors[i]["yaw"] for i in idx],
                                "sx": [anchors[i]["pitch"] / spec["width"] if spec.get("stretch") else 1.0
                                       for i in idx],
                                "color": colors[idx]}
        for c in list(bpy.data.collections):  # template seat models would ship as stray seats at the centre
            if c.name.startswith("SeatModels_"):
                for o in list(c.objects):
                    bpy.data.objects.remove(o)
    if game:
        from kit.core import game as GAME
        for o in list(bpy.data.objects):          # hero-only geometry: nets ship as their LOD1 cards
            if o.get("lod") == 0 and o.get("has_lod1"):
                bpy.data.objects.remove(o)
        GAME.prepare(st["pitch_params"])
        GAME.area_lights_to_spots()
    out = OUT / "stadiums" / f"{path.stem}{'.game' if game else ''}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_extras=True, export_yup=True,
                              export_apply=True, export_lights=game)
    n = add_instances(out, groups)
    if lods:
        add_lods(out, lods)
    cap = stands["capacity"] if stands else None
    print(f"[export] {out.name}: {out.stat().st_size / 1e6:.1f} MB, {n} seat instances "
          f"in {len(count_instances(out))} meshes, capacity {cap}, {time.time() - t:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    lod_arg = int(argv[argv.index("--seat-lod") + 1]) if "--seat-lod" in argv else None
    main(Path(argv[0]), game="--game" in argv, seat_lod=lod_arg)
