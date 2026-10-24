"""Layout sanitising: unknown catalog ids, option names and colours fall back to a default with a
warning instead of failing the whole stadium. The Rust editor does the same (viewer/gen/src/sanitize.rs).
"""
import copy
import re

from . import catalog

MOW = ("none", "stripes", "stripes_long", "checkerboard", "diagonal", "diamond", "circles")
NET_PATTERNS = ("solid", "bands", "checker")
NET_MESHES = ("hex", "square")
SEAT_PATTERNS = ("solid", "tiers", "rows", "bays", "gradient", "random")
SCREEN_PLACEMENTS = ("ends", "stand_top", "corners", "center")
FLOODLIGHT_TYPES = ("corner_pylons", "roof", "poles")

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def is_hex(v):
    return isinstance(v, str) and _HEX.fullmatch(v) is not None


class _Ctx:
    def __init__(self):
        self.warnings = []

    def choice(self, obj, what, key, ok, default):
        if key in obj and not (isinstance(obj[key], str) and ok(obj[key])):
            self.warnings.append(f"unknown {what} {obj[key]!r}; using {default}")
            obj[key] = default

    def color(self, obj, what, key, default):
        if key in obj and not is_hex(obj[key]):
            self.warnings.append(f"{what} {obj[key]!r} is not a #rrggbb colour; using {default}")
            obj[key] = default

    def colors(self, obj, what, key):
        v = obj.get(key)
        if isinstance(v, list):
            good = [c for c in v if is_hex(c)]
            if len(good) != len(v):
                self.warnings.append(f"{what}: ignoring colours that are not #rrggbb")
                if good:
                    obj[key] = good
                else:
                    del obj[key]

    def preset(self, holder, key, what, names):
        v = holder.get(key)
        if isinstance(v, str):
            name = v
        elif isinstance(v, dict) and "preset" in v:
            name = v["preset"]
        else:
            return
        if name in names or not names:
            return
        self.warnings.append(f"unknown {what} preset {name!r}; using {names[0]}")
        if isinstance(v, dict):
            v["preset"] = names[0]
        else:
            holder[key] = names[0]


def _seat_colors(c, holder, what):
    col = holder.get("colors")
    if isinstance(col, dict):
        c.choice(col, f"{what} seat pattern", "pattern", lambda s: s in SEAT_PATTERNS, "solid")
        c.color(col, f"{what} seat colour", "color", "#c8102e")
        c.colors(col, f"{what} seat colours", "colors")


def sanitize(layout):
    """A sanitised copy of the layout and the warnings for what was changed."""
    lay = copy.deepcopy(layout)
    c = _Ctx()
    if not isinstance(lay, dict):
        return lay, c.warnings
    grass, goals, flags = catalog.load("grass"), catalog.load("goals"), catalog.load("corner_flags")
    seats, profiles = catalog.load("seats"), catalog.load("stand_profiles")
    roofs, facades = list(catalog.load("roofs")), list(catalog.load("facades"))
    lights, brands = catalog.load("floodlights"), catalog.load("brands")
    p = lay.get("pitch")
    if isinstance(p, dict):
        c.choice(p, "grass", "grass", lambda s: s in grass, "perennial_ryegrass")
        c.color(p, "line colour", "line_color", "#f5f5f0")
        if isinstance(p.get("mow"), dict):
            c.choice(p["mow"], "mowing pattern", "pattern", lambda s: s in MOW, "stripes")
    g = lay.get("goals")
    if isinstance(g, dict):
        c.choice(g, "goal style", "style", lambda s: s in goals, "classic")
        c.choice(g, "net pattern", "net_pattern", lambda s: s in NET_PATTERNS, "solid")
        c.choice(g, "net mesh", "mesh", lambda s: s in NET_MESHES, "hex")
        c.color(g, "goal frame colour", "frame_color", "#f4f4f0")
        c.colors(g, "net colours", "net_colors")
    f = lay.get("corner_flags")
    if isinstance(f, dict):
        c.choice(f, "corner flag preset", "preset", lambda s: s in flags, "pro_spring")
    s = lay.get("stands")
    if isinstance(s, dict):
        c.choice(s, "hospitality seat", "hospitality_seat", lambda x: x in seats, "padded_vip")
        c.color(s, "concrete colour", "concrete", "#9d998f")
        c.color(s, "fascia colour", "fascia_color", "#1b1d20")
        c.preset(s, "roof", "roof", roofs)
        c.preset(s, "facade", "facade", facades)
        _seat_colors(c, s, "stand")
        used = []
        sides = s.get("sides")
        if isinstance(sides, dict):
            valid = [v["profile"] for v in sides.values() if isinstance(v, dict) and v.get("profile") in profiles]
            fallback = valid[0] if valid else next(iter(profiles))
            for name, sc in sides.items():
                if not isinstance(sc, dict):
                    continue
                c.choice(sc, f"{name} stand profile", "profile", lambda x: x in profiles, fallback)
                c.choice(sc, f"{name} seat", "seat", lambda x: x in seats, "tipup_classic")
                c.preset(sc, "roof", f"{name} roof", roofs)
                c.preset(sc, "facade", f"{name} facade", facades)
                _seat_colors(c, sc, name)
                if isinstance(sc.get("profile"), str):
                    used.append(sc["profile"])
        if s.get("corners") == "curved" and (len(used) < 4 or len(set(used)) != 1):
            c.warnings.append("curved corners need all four sides with one profile (a bowl); using open corners")
            s["corners"] = "open"
    fl = lay.get("floodlights")
    if isinstance(fl, dict):
        if isinstance(fl.get("preset"), str) and fl["preset"] not in lights:
            c.warnings.append(f"unknown floodlight preset {fl['preset']!r}; using corner_pylons")
            fl["preset"] = "corner_pylons"
        c.choice(fl, "floodlight type", "type", lambda x: x in FLOODLIGHT_TYPES, "corner_pylons")
        if "preset" not in fl and "type" not in fl:
            fl["type"] = "corner_pylons"
    a = lay.get("ads")
    if isinstance(a, dict) and isinstance(a.get("brands"), list):
        good = [k for k in a["brands"] if isinstance(k, str) and k in brands]
        if len(good) != len(a["brands"]):
            bad = [repr(k) for k in a["brands"] if k not in good]
            c.warnings.append(f"unknown brands {', '.join(bad)}; left out")
            a["brands"] = good
    sc = lay.get("screens")
    if isinstance(sc, dict):
        c.choice(sc, "screen placement", "placement", lambda x: x in SCREEN_PLACEMENTS, "ends")
    return lay, c.warnings
