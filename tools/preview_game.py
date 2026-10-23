"""Render an exported game .glb on its own (imported into an empty scene, no procedural shaders),
to see exactly what an engine receives.

blender -b -P tools/preview_game.py -- out/stadiums/classic.game.glb [--night]
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402

from kit.core import render as R  # noqa: E402
from kit.core.scene import OUT, reset  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    path = Path(argv[0])
    night = "--night" in argv
    reset()
    R.setup_cycles(64, (1600, 900))
    bpy.ops.import_scene.gltf(filepath=str(path))
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    if night:
        R.setup_night()
    else:
        for o in lights:
            o.hide_render = True
        R.setup_sky(38, 215)
    R.surroundings()
    W = 68.0
    mounts = [o for o in bpy.data.objects if o.name.startswith("SOCKET:dugout")]
    cam = R.camera("Broadcast", (0, -(W / 2 + 34), 22), (0, 6, 0), lens=21)
    out = OUT / "review" / f"{path.stem.replace('.', '_')}{'_night' if night else ''}_engine_view.png"
    R.render(cam, out)
    print(f"[preview] {out} ({len(lights)} lights)", flush=True)
