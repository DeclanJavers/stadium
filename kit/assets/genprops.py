"""Generated props as library props: the text-to-3D props of catalog/generated_props.json, finished
by tools/text_to_3d/finish_props.py into out/assets/genprop.<id>.glb, imported under the key
genprop.<id> (metres, Z up, front +Y, origin on the ground; baked colour, normal and
metallic-roughness maps, so the game-material conversion leaves them as they are).

They are made off-line on a GPU, so any of them may not exist yet, and some are rejected in review
(the manifest's "rejected"): keys() lists only the finished, accepted ones, and a placement of one
that is missing draws its manifest "fallback" (a code prop) instead, or nothing (props.build, and
the editor's Library::resolve).
"""
import json
from functools import lru_cache

import bpy

from ..core.scene import OUT, ROOT, empty

PREFIX = "genprop."


@lru_cache(maxsize=1)
def manifest():
    """{id: spec} from catalog/generated_props.json."""
    data = json.loads((ROOT / "catalog" / "generated_props.json").read_text())
    return {p["id"]: p for p in data.get("props", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}


def path(pid):
    return OUT / "assets" / f"{PREFIX}{pid}.glb"


def usable(pid):
    """Finished and not rejected in review (a rejected prop's file is ignored)."""
    return path(pid).exists() and not manifest().get(pid, {}).get("rejected")


def keys():
    """The keys of the generated props that have been finished and not rejected."""
    return [PREFIX + pid for pid in manifest() if usable(pid)]


def fallback(key):
    """The code prop drawn while a generated prop is missing, or None."""
    fb = manifest().get(key[len(PREFIX):], {}).get("fallback")
    return fb if isinstance(fb, str) else None


def build(key, col):
    """Import a finished generated prop under a PROP:<key> root in col; None if it is missing."""
    f = path(key[len(PREFIX):])
    if not usable(key[len(PREFIX):]):
        return None
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(f))
    objs = [o for o in bpy.data.objects if o not in before]
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    root = empty(f"PROP:{key}", col)
    for o in objs:
        if o.parent is None:
            o.parent = root
    root["asset_id"] = key
    root["asset_family"] = "genprop"
    return root
