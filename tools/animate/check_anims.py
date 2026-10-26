"""Check the exported animated props without Blender: every looping action must end where it
starts (or the prop pops each cycle in a game), every action in the armature's action table must
exist as a glTF animation, and the LODs must carry the same actions as the full prop.

    python3 tools/animate/check_anims.py
"""
import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {5120: (np.int8, 1), 5121: (np.uint8, 1), 5122: (np.int16, 1), 5123: (np.uint16, 1),
              5125: (np.uint32, 1), 5126: (np.float32, 1)}
WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
ROT_TOL = 0.5   # degrees
LOC_TOL = 1e-3  # metres


def load(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + n])
    binary = b[20 + n + 8:]
    return g, binary


def accessor(g, binary, i):
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    dtype, _ = COMPONENTS[a["componentType"]]
    w = WIDTH[a["type"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    out = np.frombuffer(binary, dtype, a["count"] * w, start).reshape(a["count"], w)
    return out.astype(np.float64)


def check(path):
    g, binary = load(path)
    problems = []
    table = next((json.loads(x["extras"]["actions"]) for x in g["nodes"] if "actions" in x.get("extras", {})), None)
    if table is None:
        return [f"{path.name}: no action table"], {}
    anims = {a["name"]: a for a in g.get("animations", [])}
    for name, meta in table.items():
        if name not in anims:
            problems.append(f"{path.name}: '{name}' in the table but not exported")
            continue
        if not meta.get("loop", True):
            continue
        anim = anims[name]
        worst_rot = worst_loc = 0.0
        for ch in anim["channels"]:
            s = anim["samplers"][ch["sampler"]]
            vals = accessor(g, binary, s["output"])
            a, z = vals[0], vals[-1]
            if ch["target"]["path"] == "rotation":
                dot = min(1.0, abs(float(np.dot(a / np.linalg.norm(a), z / np.linalg.norm(z)))))
                worst_rot = max(worst_rot, np.degrees(2 * np.arccos(dot)))
            elif ch["target"]["path"] == "translation":
                worst_loc = max(worst_loc, float(np.abs(a - z).max()))
        if worst_rot > ROT_TOL or worst_loc > LOC_TOL:
            problems.append(f"{path.name}: '{name}' does not loop (seam {worst_rot:.2f} deg, {worst_loc * 1000:.1f} mm)")
    return problems, table


def main():
    problems, checked = [], 0
    for full in sorted(glob.glob(str(ROOT / "out" / "assets" / "genprop.*.anim.glb"))):
        full = Path(full)
        p, table = check(full)
        problems += p
        checked += 1
        for level in (1, 2):
            lod = full.with_name(full.name.replace(".anim.glb", f".anim.lod{level}.glb"))
            if not lod.exists():
                problems.append(f"{lod.name}: missing")
                continue
            p, lod_table = check(lod)
            problems += p
            checked += 1
            if set(lod_table) != set(table):
                problems.append(f"{lod.name}: actions {sorted(lod_table)} differ from {sorted(table)}")
    for line in problems:
        print("FAIL", line)
    print(f"{checked} files checked, {len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
