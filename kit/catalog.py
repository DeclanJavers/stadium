import copy
import json

from .core.scene import ROOT


def load(name):
    data = json.loads((ROOT / "catalog" / f"{name}.json").read_text())
    data.pop("_doc", None)
    return data


def preset(name, key):
    data = load(name)
    if key not in data:
        raise KeyError(f"{key!r} not in catalog/{name}.json; options: {sorted(data)}")
    return copy.deepcopy(data[key])


def merge(base, override):
    """Deep-merge override into a copy of base."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out
