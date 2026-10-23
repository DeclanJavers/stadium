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
from kit.core.gltf_instancing import add_instances, count_instances  # noqa: E402
from kit.core.scene import OUT, collection, empty, link, reset  # noqa: E402


def main(path):
    t = time.time()
    reset()
    st = assemble(json.loads(path.read_text()))
    stands = st.get("stands")
    for o in list(bpy.data.objects):  # render-only helpers and GN seat clouds never ship
        if o.get("render_only") or o.get("seat_style"):
            bpy.data.objects.remove(o)
    groups = {}
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
                link(bpy.data.objects.new(name, src.data), col, holder)
                groups[name] = {"pos": [anchors[i]["pos"] for i in idx],
                                "yaw": [anchors[i]["yaw"] for i in idx],
                                "sx": [anchors[i]["pitch"] / spec["width"] if spec.get("stretch") else 1.0
                                       for i in idx],
                                "color": colors[idx]}
        for c in list(bpy.data.collections):  # template seat models would ship as stray seats at the centre
            if c.name.startswith("SeatModels_"):
                for o in list(c.objects):
                    bpy.data.objects.remove(o)
    out = OUT / "stadiums" / f"{path.stem}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_extras=True, export_yup=True,
                              export_apply=True)
    n = add_instances(out, groups)
    cap = stands["capacity"] if stands else None
    print(f"[export] {out.name}: {out.stat().st_size / 1e6:.1f} MB, {n} seat instances "
          f"in {len(count_instances(out))} meshes, capacity {cap}, {time.time() - t:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--") + 1]))
