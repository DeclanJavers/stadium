"""Export the runtime asset library for the Rust stadium editor: every material the generators
reference (game-ready: baked tileable PBR, TINT detail + gain) as a small swatch, and every prop the
editor instances (seats at LOD1 + LOD2, goals with their net cards, corner flags, floodlight
fixtures). One .glb; the editor finds things by node name:

  MAT:<material>          a 1 m swatch carrying the material
  PROP:<asset key>        the root of a prop, e.g. PROP:seat.tipup_classic.std, PROP:goal.box
  PROP:<key>.lod2         the coarse level of a seat

blender -b --factory-startup -P tools/export_library.py
(LIBRARY_ONLY=tree,hedge exports only the props whose keys contain one of those words: a quick
loop while tuning a few props; run it in full before using the library for anything else.)
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

from kit import catalog  # noqa: E402
from kit.assets import corner_flag, dugout, facade, floodlight, goal, props, roof, seat, site_materials, stand, tunnel  # noqa: E402
from kit.core import game as GAME  # noqa: E402
from kit.core import materials as MAT  # noqa: E402
from kit.core.scene import OUT, collection, empty, link, mesh_object, reset  # noqa: E402

from kit.assets.library import MATERIALS  # noqa: E402


def swatch(name, mat):
    """A 1 m square in the XZ plane far below the origin (never drawn), metric UVs."""
    V = [(0, 0, -100), (1, 0, -100), (1, 0, -99), (0, 0, -99)]
    o = mesh_object(f"MAT:{name}", V, [(0, 1, 2, 3)], collection("Materials"), materials=[mat],
                    uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    return o


def prop_root(key, objs):
    root = empty(f"PROP:{key}", collection("Props"))
    for o in objs:
        o.parent = root
    return root


def main():
    t = time.time()
    reset()
    made = {}
    for name, (fn, base) in MATERIALS.items():
        m = fn()
        assert m.name == name, f"{name}: built {m.name}"
        made[name] = (m, base)
        swatch(name, m)
    # props
    for style, spec in catalog.load("seats").items():
        for v in spec["variants"]:
            base = seat.build_variant(style, v, collection("SeatBase"))
            for lvl, suffix in ((1, ""), (2, ".lod2")):
                lo = seat.lod(base, lvl, collection("Props"))
                lo.name = f"SEAT_{style}_{v}_LOD{lvl}"
                prop_root(f"seat.{style}.{v}{suffix}", [lo])
            bpy.data.objects.remove(base)
    for style in catalog.load("goals"):
        g = goal.build({"style": style}, collection(f"Goal_{style}"))
        for o in list(g.children_recursive):
            if o.get("lod") == 0 and o.get("has_lod1"):   # the hero net: games use its card
                bpy.data.objects.remove(o)
            else:
                o.hide_render = False
        g.name = f"PROP:goal.{style}"
    for preset in catalog.load("corner_flags"):
        f = corner_flag.build({"preset": preset}, collection(f"Flag_{preset}"))
        f.name = f"PROP:corner_flag.{preset}"
    for kind in ("led", "halide"):
        fx = floodlight.fixture_model(kind, collection("Fixtures"), 5700, False)
        fx.hide_render = fx.hide_viewport = False
        prop_root(f"floodlight_fixture.{kind}", [fx])
        lo = floodlight.fixture_lod2(kind, collection("Fixtures"), 5700, False)
        prop_root(f"floodlight_fixture.{kind}.lod2", [lo])
    # training grounds: site materials and the props library (street, vehicles, construction,
    # training gear, vegetation)
    for name, (fn, base) in site_materials.MATERIALS.items():
        m = fn()
        assert m.name == name, f"{name}: built {m.name}"
        made[name] = (m, base)
        swatch(name, m)
    only = [w for w in os.environ.get("LIBRARY_ONLY", "").split(",") if w]
    for key in props.keys():
        if not only or any(w in key for w in only):
            props.build(key, collection("Props"))
    GAME.prepare(None)
    for name, (m, base) in made.items():
        if base:
            m["base_hex"] = base
    # LIBRARY_OUT: write elsewhere (a partial library for reviewing a few props; the viewer reads it
    # with STADIUM_LIBRARY=<path>)
    out = Path(os.environ["LIBRARY_OUT"]) if os.environ.get("LIBRARY_OUT") else OUT / "library" / "library.game.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    for o in bpy.data.objects:
        o.hide_viewport = False
        o.hide_render = False
        o.hide_set(False)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_extras=True, export_yup=True,
                              export_apply=True, use_visible=False, export_vertex_color="ACTIVE")
    print(f"[library] {out}: {out.stat().st_size / 1e6:.1f} MB, {len(made)} materials, "
          f"{sum(1 for o in bpy.data.objects if o.name.startswith('PROP:'))} props, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
