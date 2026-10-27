"""Write EXT_mesh_gpu_instancing data into an exported .glb.

Blender's exporter only instances linked duplicates, not geometry-node instances, so stadium seats
(up to 150k) are exported as one mesh per seat variant plus per-instance attribute arrays that we
append here: TRANSLATION, ROTATION, SCALE and the custom _SEAT_COLOR (linear RGB).
"""
import json
import math
import struct

import numpy as np

EXT = "EXT_mesh_gpu_instancing"


def _read(path):
    data = open(path, "rb").read()
    magic, version, _ = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67, "not a glb"
    jlen, jtype = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + jlen])
    off = 20 + jlen
    blen, btype = struct.unpack("<II", data[off:off + 8])
    return gltf, bytearray(data[off + 8:off + 8 + blen])


def _write(path, gltf, binary):
    j = json.dumps(gltf, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(j) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A) + j)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + bytes(binary))


def _accessor(gltf, binary, arr, kind):
    arr = np.ascontiguousarray(arr, np.float32)
    while len(binary) % 4:
        binary.append(0)
    view = {"buffer": 0, "byteOffset": len(binary), "byteLength": arr.nbytes}
    binary += arr.tobytes()
    gltf["bufferViews"].append(view)
    acc = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": 5126, "count": len(arr), "type": kind}
    if kind == "VEC3":
        acc["min"] = arr.min(0).tolist()
        acc["max"] = arr.max(0).tolist()
    gltf["accessors"].append(acc)
    return len(gltf["accessors"]) - 1


def add_instances(path, groups):
    """groups: {node_name: dict(pos=(N,3) Blender Z-up, yaw=(N,), sx=(N,), color=(N,3) linear)}.
    Optionally tilt=(N,) (about the turned local X, after yaw) and scale=(N,3) (Blender axes) in
    place of sx; color may be omitted (no instance colour)."""
    gltf, binary = _read(path)
    names = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
    total = 0
    for name, g in groups.items():
        if name not in names or not len(g["pos"]):
            continue
        p = np.asarray(g["pos"], float)
        t = np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)          # Blender Z-up -> glTF Y-up
        half = np.asarray(g["yaw"], float) / 2                  # yaw about Blender Z == about glTF +Y
        q = np.stack([np.zeros_like(half), np.sin(half), np.zeros_like(half), np.cos(half)], 1)
        if g.get("tilt") is not None:
            # q = q_yaw * q_tilt, the tilt about local X (Blender X == glTF X)
            th = np.asarray(g["tilt"], float) / 2
            sx_, cx_ = np.sin(th), np.cos(th)
            qy, qw = q[:, 1], q[:, 3]
            q = np.stack([qw * sx_, qy * cx_, -qy * sx_, qw * cx_], 1)
        if g.get("scale") is not None:
            sc = np.asarray(g["scale"], float)
            s = np.stack([sc[:, 0], sc[:, 2], sc[:, 1]], 1)     # Blender (x, y, z) -> glTF (x, z, y)
        else:
            s = np.stack([np.asarray(g["sx"], float), np.ones(len(p)), np.ones(len(p))], 1)
        attrs = {"TRANSLATION": _accessor(gltf, binary, t, "VEC3"),
                 "ROTATION": _accessor(gltf, binary, q, "VEC4"),
                 "SCALE": _accessor(gltf, binary, s, "VEC3")}
        if g.get("color") is not None:
            attrs["_SEAT_COLOR"] = _accessor(gltf, binary, np.asarray(g["color"], float)[:, :3], "VEC3")
        node = gltf["nodes"][names[name]]
        node.setdefault("extensions", {})[EXT] = {"attributes": attrs}
        total += len(p)
    gltf["buffers"][0]["byteLength"] = len(binary) + (-len(binary) % 4)
    used = gltf.setdefault("extensionsUsed", [])
    if EXT not in used:
        used.append(EXT)
    _write(path, gltf, binary)
    return total


def count_instances(path):
    gltf, _ = _read(path)
    out = {}
    for n in gltf["nodes"]:
        ext = n.get("extensions", {}).get(EXT)
        if ext:
            out[n["name"]] = gltf["accessors"][ext["attributes"]["TRANSLATION"]]["count"]
    return out


def add_lods(path, pairs, coverage=(0.004, 0.0)):
    """Link LOD meshes with MSFT_lod. pairs: {base node name: lod node name}. The LOD nodes are
    detached from the scene (engines without MSFT_lod never draw them) and referenced from the base
    node; MSFT_screencoverage in extras gives the switch thresholds."""
    gltf, binary = _read(path)
    names = {n.get("name"): i for i, n in enumerate(gltf["nodes"])}
    lod_ids = {names[l] for b, l in pairs.items() if b in names and l in names}
    for n in gltf["nodes"]:
        if "children" in n:
            n["children"] = [c for c in n["children"] if c not in lod_ids]
            if not n["children"]:
                del n["children"]
    for sc in gltf.get("scenes", []):
        sc["nodes"] = [c for c in sc["nodes"] if c not in lod_ids]
    linked = 0
    for base, lod in pairs.items():
        if base in names and lod in names:
            node = gltf["nodes"][names[base]]
            node.setdefault("extensions", {})["MSFT_lod"] = {"ids": [names[lod]]}
            node.setdefault("extras", {})["MSFT_screencoverage"] = list(coverage)
            linked += 1
    if linked:
        used = gltf.setdefault("extensionsUsed", [])
        if "MSFT_lod" not in used:
            used.append("MSFT_lod")
    _write(path, gltf, binary)
    return linked
