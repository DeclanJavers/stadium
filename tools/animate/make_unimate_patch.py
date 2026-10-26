"""Write out/animate/unimate_patch.json: the reviewed UniMate joint labels and facing pairs from
catalog/creature_anims.json, keyed by each rig's raw bone names (out/animate/rigs/<id>.bones.json),
for apply_unimate_labels.py on Colab.

    python3 tools/animate/make_unimate_patch.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cfg = json.loads((ROOT / "catalog" / "creature_anims.json").read_text())
patch = {}
for pid, spec in cfg["creatures"].items():
    if "unimate" not in spec:
        continue
    names = [b["name"] for b in json.loads((ROOT / "out" / "animate" / "rigs" / f"{pid}.bones.json").read_text())]
    u = spec["unimate"]
    patch[pid] = {"labels": {names[int(i)]: label for i, label in u["labels"].items()},
                  "face": [names[u["face"][0]], names[u["face"][1]]]}
out = ROOT / "out" / "animate" / "unimate_patch.json"
out.write_text(json.dumps(patch))
print("wrote", out, {k: v["face"] for k, v in patch.items()})
