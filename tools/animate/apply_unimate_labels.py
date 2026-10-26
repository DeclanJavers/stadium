# Colab: apply the reviewed labels and facing pairs (make_unimate_patch.py) to UniMate's
# annotation.json files, by raw bone name.
import json, sys
patch = json.load(open("/content/anim/unimate_patch.json"))
for pid, p in patch.items():
    f = f"/content/anim/unimate/rig/{pid}/annotation.json"
    a = json.load(open(f))
    for j in a["joints"]:
        if j["raw"] in p["labels"]:
            j["label"], j["check"], j["source"] = p["labels"][j["raw"]], False, "review"
    a["face_pair"].update(right=p["face"][0], left=p["face"][1], body_axis=False, check=False, source="review")
    json.dump(a, open(f, "w"), indent=1)
    print(pid, "patched", sum(j["raw"] in p["labels"] for j in a["joints"]), "of", len(a["joints"]))
