"""Build every catalog variant to out/assets/*.glb and write out/manifest.json.

blender -b -P tools/build.py [-- --only goal]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit import catalog  # noqa: E402
from kit.assets import corner_flag, goal, pitch, seat  # noqa: E402
from kit.core.export import export_glb, stats  # noqa: E402
from kit.core.scene import OUT, reset  # noqa: E402


def variants():
    yield "pitch.standard", lambda: pitch.build({})[0]
    for style in catalog.load("goals"):
        yield f"goal.{style}", lambda style=style: goal.build({"style": style})
    for preset in catalog.load("corner_flags"):
        yield f"corner_flag.{preset}", lambda preset=preset: corner_flag.build({"preset": preset})
    for style, spec in catalog.load("seats").items():
        for v in spec["variants"]:
            yield f"seat.{style}.{v}", lambda style=style, v=v: seat.build_variant(style, v)
    from kit.assets import dugout, floodlight, screen
    from kit.core.scene import collection, set_params
    for style in ("shell", "box", "shelter"):
        def make(style=style):
            p = dugout.resolve({"style": style})
            o = dugout._model(p, collection("Dugout"), f"Dugout_{style}")
            set_params(o, f"dugout.{style}", "dugout", "dugout", p)
            return o
        yield f"dugout.{style}", make
    for kind in ("led", "halide"):
        def fx(kind=kind):
            o = floodlight.fixture_model(kind, collection("Fixture"), 5700, False)
            set_params(o, f"floodlight_fixture.{kind}", "floodlight_fixture", "roof_edge", {"fixture": kind})
            return o
        yield f"floodlight_fixture.{kind}", fx

    def scr():
        from kit.ads import studio
        import bpy
        img = bpy.data.images.load(str(studio.render_scoreboard()), check_existing=True)
        from kit.assets.ads import led_material
        o = screen.panel("Screen_16x9", 16.0, 9.0, collection("Screen"), led_material("Screen_LED", img, 3.5, (1920, 1080)))
        set_params(o, "screen.16x9", "screen", "roof_edge", {"width": 16.0, "height": 9.0})
        return o
    yield "screen.16x9", scr


def main(only=None):
    manifest = {"schema": "stadium-kit-manifest/1", "assets": []}
    for name, fn in variants():
        if only and not name.startswith(only):
            continue
        t = time.time()
        reset()
        root = fn()
        info = stats(root)
        path = export_glb(root, OUT / "assets" / f"{name}.glb")
        info["file"] = str(path.relative_to(OUT))
        info["bytes"] = path.stat().st_size
        manifest["assets"].append(info)
        print(f"[build] {name}: {info['triangles']} tris, {len(info['sockets'])} sockets, "
              f"{info['bytes'] / 1e6:.1f} MB, {time.time() - t:.1f}s", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv[argv.index("--only") + 1] if "--only" in argv else None)
