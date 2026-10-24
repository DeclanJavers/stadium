"""Export the runtime asset library for the Rust stadium editor: every material the generators
reference (game-ready: baked tileable PBR, TINT detail + gain) as a small swatch, and every prop the
editor instances (seats at LOD1 + LOD2, goals with their net cards, corner flags, floodlight
fixtures). One .glb; the editor finds things by node name:

  MAT:<material>          a 1 m swatch carrying the material
  PROP:<asset key>        the root of a prop, e.g. PROP:seat.tipup_classic.std, PROP:goal.box
  PROP:<key>.lod2         the coarse level of a seat

blender -b --factory-startup -P tools/export_library.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

from kit import catalog  # noqa: E402
from kit.assets import corner_flag, dugout, facade, floodlight, goal, roof, seat, stand, tunnel  # noqa: E402
from kit.core import game as GAME  # noqa: E402
from kit.core import materials as MAT  # noqa: E402
from kit.core.scene import OUT, collection, empty, link, mesh_object, reset  # noqa: E402

# (material, the sRGB colour it is baked with) -- the editor recolours relative to it
MATERIALS = {
    "Stand_Concrete": (lambda: stand.concrete(), "#9d998f"),
    "Stand_Block": (lambda: stand.concrete("Stand_Block", "#b3aea4"), "#b3aea4"),
    "Stand_Soffit": (lambda: stand.painted("Stand_Soffit", "#d9d6cf", 0.8), "#d9d6cf"),
    "TINT_fascia": (lambda: stand.painted("TINT_fascia", "#1b1d20", 0.4), "#1b1d20"),
    "Balustrade_Glass": (lambda: stand.glass(), None),
    "Galvanised": (lambda: seat.galvanised(), None),
    "Step_Nosing": (lambda: stand.painted("Step_Nosing", "#f2c200", 0.55), "#f2c200"),
    "Apron_Track": (lambda: stand.painted("Apron_Track", "#9c3f2c", 0.85), "#9c3f2c"),
    "Apron_Concrete": (lambda: stand.concrete("Apron_Concrete", "#77746e"), "#77746e"),
    "Plaza_Paving": (lambda: stand.paving(), None),
    "TINT_roof": (lambda: roof.deck_material("#d9dbdc"), "#d9dbdc"),
    "TINT_roof_membrane": (lambda: roof.deck_material("#f4f3ef", membrane=True), "#f4f3ef"),
    "Roof_Polycarbonate": (lambda: roof.translucent_material(), None),
    "TINT_roof_fascia": (lambda: stand.painted("TINT_roof_fascia", "#1b1d20", 0.45), "#1b1d20"),
    "Roof_Underside": (lambda: stand.painted("Roof_Underside", "#d8d8d4", 0.65), "#d8d8d4"),
    "Roof_Steel": (lambda: MAT.painted_metal("Roof_Steel", "#eceeee", rough=0.4, dirt=0.0, scuffs=0.1), "#eceeee"),
    "Facade_Brick": (lambda: facade.brick(color="#8a3b2a"), "#8a3b2a"),
    "Facade_Stone": (lambda: stand.concrete("Facade_Stone", "#d6ccb9"), "#d6ccb9"),
    "Facade_Concrete": (lambda: stand.concrete("Facade_Concrete", "#b8b3a8"), "#b8b3a8"),
    "Facade_Trim": (lambda: MAT.painted_metal("Facade_Trim", "#d9dad6", rough=0.4, dirt=0.2, scuffs=0.0), "#d9dad6"),
    "TINT_facade": (lambda: stand.painted("TINT_facade", "#c7c9c9", 0.35), "#c7c9c9"),
    "TINT_facade_b": (lambda: stand.painted("TINT_facade_b", "#9e0c24", 0.35), "#9e0c24"),
    "Facade_DarkGlass": (lambda: facade.dark_glass(), None),
    "TINT_facade_etfe": (lambda: facade.etfe("#f2f2f0", None, 0.0), "#f2f2f0"),
    "Facade_Core": (lambda: stand.concrete("Facade_Core", "#bdb8ad"), "#bdb8ad"),
    "Facade_Frame": (lambda: MAT.painted_metal("Facade_Frame", "#f1f2f0", rough=0.35, dirt=0.0, scuffs=0.0), "#f1f2f0"),
    "Board_Housing": (lambda: stand.painted("Board_Housing", "#34363a", 0.5), "#34363a"),
    "Screen_Body": (lambda: stand.painted("Screen_Body", "#45484d", 0.5), "#45484d"),
    "Dugout_Slab": (lambda: stand.concrete("Dugout_Slab", "#8e8a82"), "#8e8a82"),
    "Dugout_Polycarbonate": (lambda: dugout.polycarbonate("#3d4954"), "#3d4954"),
    "Dugout_Frame": (lambda: MAT.painted_metal("Dugout_Frame", "#d8dadb", rough=0.35, dirt=0.2), "#d8dadb"),
    "Dugout_Brick": (lambda: facade.brick("Dugout_Brick", "#7a3426"), "#7a3426"),
    "TINT_dugout": (lambda: stand.painted("TINT_dugout", "#c8102e", 0.45), "#c8102e"),
    "Tunnel_Mouth": (lambda: MAT.painted_metal("Tunnel_Mouth", "#050506", rough=0.9, dirt=0.0, scuffs=0.0), "#050506"),
    "Tunnel_Trim": (lambda: stand.painted("Tunnel_Trim", "#1a1b1e", 0.4), "#1a1b1e"),
    "Tunnel_Building": (lambda: stand.concrete("Tunnel_Building", "#a8a399"), "#a8a399"),
    "TINT_tunnel": (lambda: tunnel.fabric("#c8102e"), "#c8102e"),
    "Tunnel_Mat": (lambda: MAT.rubber("Tunnel_Mat", "#1f2a22"), "#1f2a22"),
    "Tunnel_Hoop": (lambda: MAT.painted_metal("Tunnel_Hoop", "#d9dbdc", 0.35, 0.1, 0.0), "#d9dbdc"),
}


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
    GAME.prepare(None)
    for name, (m, base) in made.items():
        if base:
            m["base_hex"] = base
    out = OUT / "library" / "library.game.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    for o in bpy.data.objects:
        o.hide_viewport = False
        o.hide_render = False
        o.hide_set(False)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_extras=True, export_yup=True,
                              export_apply=True, use_visible=False)
    print(f"[library] {out.relative_to(ROOT)}: {out.stat().st_size / 1e6:.1f} MB, {len(made)} materials, "
          f"{sum(1 for o in bpy.data.objects if o.name.startswith('PROP:'))} props, {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
