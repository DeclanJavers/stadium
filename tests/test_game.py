"""Phase 5 tests: game-ready export, LODs, lights, and the Rust viewer. Run inside Blender:

blender -b --factory-startup -P tests/test_game.py
"""
import json
import struct
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from kit.core import bake as BK  # noqa: E402
from kit.core.scene import OUT, reset  # noqa: E402

TESTS = []
_EXPORTS = {}


def test(fn):
    TESTS.append(fn)
    return fn


def gltf(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n])


def game(name):
    if name not in _EXPORTS:
        import export_stadium
        _EXPORTS[name] = export_stadium.main(ROOT / "layouts" / f"{name}.json", game=True)
    return _EXPORTS[name]


@test
def seamless_tiling_has_no_edge_seam():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 4)).astype(np.float32)
    s = BK.seamless(a)
    # wrapping neighbours across the tile edge must be as similar as ordinary neighbours
    edge = np.abs(s[:, 0] - s[:, -1]).mean()
    inner = np.abs(s[:, 31] - s[:, 32]).mean()
    assert edge <= inner * 1.6, (edge, inner)


@test
def every_game_material_carries_colour():
    g = gltf(game("classic"))
    missing = [m["name"] for m in g["materials"]
               if "baseColorFactor" not in m.get("pbrMetallicRoughness", {})
               and "baseColorTexture" not in m.get("pbrMetallicRoughness", {})]
    assert not missing, f"materials an engine would draw white: {missing}"
    tiled = [m for m in g["materials"] if "KHR_texture_transform" in json.dumps(m)]
    assert len(tiled) >= 8, f"only {len(tiled)} tiling materials"


@test
def tint_materials_expose_their_gain():
    g = gltf(game("classic"))
    for m in g["materials"]:
        if m["name"] in ("TINT_fascia", "TINT_facade", "TINT_roof") and "baseColorTexture" in m.get("pbrMetallicRoughness", {}):
            assert m.get("extras", {}).get("tint_gain"), f"{m['name']} lacks tint_gain"


@test
def lod_nodes_are_linked_and_never_drawn_twice():
    g = gltf(game("premier"))
    roots = set(g["scenes"][0]["nodes"])
    children = {c for n in g["nodes"] for c in n.get("children", [])}
    reachable = roots | children
    bases = [n for n in g["nodes"] if "MSFT_lod" in n.get("extensions", {})]
    assert bases, "no MSFT_lod links"
    for n in bases:
        for lod in n["extensions"]["MSFT_lod"]["ids"]:
            assert lod not in reachable, "a LOD node is still in the scene graph (engines would draw it)"
            assert "mesh" in g["nodes"][lod]
    assert "MSFT_lod" in g["extensionsUsed"]


@test
def no_stray_meshes_at_the_origin():
    g = gltf(game("premier"))
    children = {c for n in g["nodes"] for c in n.get("children", [])}
    stray = [n["name"] for i, n in enumerate(g["nodes"])
             if "mesh" in n and n.get("name", "").startswith(("SEAT_", "ScreenModel", "DugoutModel", "Fixture_"))
             and "EXT_mesh_gpu_instancing" not in n.get("extensions", {}) and i in children
             and not any(k in n for k in ("translation", "rotation", "matrix"))]
    assert not stray, f"template meshes shipped at the origin: {stray[:5]}"


@test
def nets_ship_as_cards_and_lights_ship_as_spots():
    g = gltf(game("premier"))
    names = [n.get("name", "") for n in g["nodes"]]
    assert any(nm.startswith("Net_LOD1") for nm in names), "net cards missing"
    assert not any(nm == "Net" or nm.startswith("Net.") for nm in names), "hero cord nets shipped in the game file"
    lights = g["extensions"]["KHR_lights_punctual"]["lights"]
    assert all(l["type"] == "spot" for l in lights)
    n_light_nodes = sum("KHR_lights_punctual" in n.get("extensions", {}) for n in g["nodes"])
    assert n_light_nodes >= 150, n_light_nodes


def _viewer():
    exe = ROOT / "viewer" / "target" / "release" / "stadium-viewer"
    if not exe.exists():
        subprocess.run([str(ROOT / "viewer" / "build.sh"), "--offline"], check=True, capture_output=True)
    return exe


@test
def viewer_renders_day_and_night():
    exe = _viewer()
    for night in (False, True):
        out = OUT / "review" / f"test_viewer_{'night' if night else 'day'}.png"
        cmd = [str(exe), str(game("premier")), "--screenshot", str(out), "--size", "640x360"] + (["--night"] if night else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        img = bpy.data.images.load(str(out))
        px = np.array(img.pixels[:]).reshape(360, 640, 4)[..., :3]
        pitch = px[60:200, 200:440].mean(axis=(0, 1))   # lower-middle band is pitch (rows are bottom-up)
        assert 0.08 < px.mean() < 0.9, f"image blank or blown out: mean {px.mean():.3f}"
        assert pitch[1] > pitch[0] and pitch[1] > pitch[2], f"pitch is not green: {pitch}"


@test
def viewer_lod_and_irradiance_volume_report():
    exe = _viewer()
    r = subprocess.run([str(exe), str(game("premier")), "--night", "--screenshot",
                        str(OUT / "review" / "test_viewer_lod.png"), "--size", "320x180"], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if "ms/frame" in l][0]
    fine, coarse = [int(x) for x in line.split("seats ")[1].replace(" coarse", "").split(" fine / ")]
    assert coarse > fine > 0, line


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed", flush=True)
    sys.exit(1 if failed else 0)
