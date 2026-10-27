"""The training-ground generator's output model (mirrors viewer/gen/src/site/mod.rs): materials by
reference, generated parts (meshes or tube sets), instanced props, pitches, lights and sockets,
collected by an Out while the plots are generated. Plain Python, no Blender.
"""
import math

from ..stands.patterns import hex_rgb
from ..stands.prng import uniform
from . import kernel

# heights of the layered ground (no holes are cut: covers sit on the lawn)
FIELD_Z = -0.06
LAWN_Z = -0.03
COVER_Z = 0.02
PATH_Z = 0.035
ROAD_Z = 0.05

STATES = ("occupied", "demolition", "cleared", "construction", "built")
BOUNDARIES = ("chainlink", "palisade", "railings", "brick_wall", "hedge")
SCREENING = ("none", "trees", "berm")


def rgb(hexs):
    """sRGB 0-1 triple of a #rrggbb colour (as plain floats)."""
    return tuple(float(c) for c in hex_rgb(hexs))


class Mat:
    """A material reference: a library material by key, optionally recoloured (sRGB), glowing, or
    carrying a procedural texture ("grass", spec) / ("net", spec)."""
    __slots__ = ("key", "color", "emission", "texture")

    def __init__(self, key, color=None, emission=None, texture=None):
        self.key, self.color, self.emission, self.texture = key, color, emission, texture

    def __repr__(self):
        return f"Mat({self.key!r}, {self.color}, {self.emission}, {self.texture})"


class Placement:
    """A prop instance: position, yaw about z, then tilt about the turned local x, scale."""
    __slots__ = ("pos", "yaw", "tilt", "scale")

    def __init__(self, pos, yaw, tilt=0.0, scale=(1.0, 1.0, 1.0)):
        self.pos, self.yaw, self.tilt, self.scale = tuple(pos), yaw, tilt, tuple(scale)

    @staticmethod
    def aim(pos, d):
        """Local +y pointing along d (Blender's track-to Y, up Z)."""
        ln = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        r = 1.0 / ln
        n = (d[0] * r, d[1] * r, d[2] * r)
        return Placement(pos, math.atan2(-n[0], n[1]), math.asin(min(max(n[2], -1.0), 1.0)))


class PropSet:
    __slots__ = ("key", "items", "tints", "net")

    def __init__(self, key, items, tints, net=None):
        self.key, self.items, self.tints, self.net = key, items, tints, net


class SitePart:
    """A mesh (or a set of tubes: [(points, radius)]) with one material per face tag."""
    __slots__ = ("name", "mesh", "tubes", "sides", "mats", "smooth_deg")

    def __init__(self, name, mesh, tubes, sides, mats, smooth_deg):
        self.name, self.mesh, self.tubes, self.sides, self.mats, self.smooth_deg = name, mesh, tubes, sides, mats, smooth_deg


class SitePitch:
    """A pitch on the site: the pitch generator's parameters and where it lies (yaw in radians:
    its length runs along that direction)."""
    __slots__ = ("plot", "params", "origin", "yaw", "goals", "both_goals", "flags", "lit", "surface")

    def __init__(self, plot, params, origin, yaw, goals, both_goals, flags, lit, surface):
        self.plot, self.params, self.origin, self.yaw = plot, params, origin, yaw
        self.goals, self.both_goals, self.flags, self.lit, self.surface = goals, both_goals, flags, lit, surface


class Spot:
    __slots__ = ("pos", "dir", "power", "angle_deg", "blend", "color")

    def __init__(self, pos, d, power, angle_deg, blend, color):
        self.pos, self.dir, self.power, self.angle_deg, self.blend, self.color = pos, d, power, angle_deg, blend, color


class Socket:
    __slots__ = ("kind", "id", "loc", "n", "props")

    def __init__(self, kind, id_, loc, n, props):
        self.kind, self.id, self.loc, self.n, self.props = kind, id_, loc, n, props


class Club:
    __slots__ = ("name", "primary", "secondary")

    def __init__(self, name, primary, secondary):
        self.name, self.primary, self.secondary = name, primary, secondary


class Plot:
    """A plot ready to generate: its rectangle, frame, frontage (a local unit vector towards the
    road or path) and its resolved configuration."""
    __slots__ = ("id", "kind", "w", "d", "frame", "front", "cfg", "seed")

    def __init__(self, id_, kind, w, d, frame, front, cfg, seed):
        self.id, self.kind, self.w, self.d, self.frame, self.front, self.cfg, self.seed = id_, kind, w, d, frame, front, cfg, seed

    def state(self):
        return text(self.cfg, "state", "occupied")

    def r(self, i):
        return uniform(self.seed, i)

    def front_yaw(self):
        """Yaw (radians, about z) that turns a prop's local +y to face the frontage."""
        return math.atan2(-self.front[0], self.front[1])


class Out:
    """Collects a site's output: parts in order, props grouped by asset and recolour."""

    def __init__(self, warnings, night, club):
        self.parts, self.props, self.pitches, self.spots, self.sockets = [], [], [], [], []
        self.warnings = warnings
        self.parking = 0
        self.entrance = None
        self.night = night
        self.club = club

    def part(self, name, mesh, mats, smooth_deg=0.0):
        if mesh.faces:
            self.parts.append(SitePart(name, mesh, [], 0, mats, smooth_deg))

    def part_at(self, f, name, mesh, mats, smooth_deg=0.0):
        f.mesh(mesh)
        self.part(name, mesh, mats, smooth_deg)

    def tubes(self, name, lines, mat, sides):
        if lines:
            self.parts.append(SitePart(name, kernel.Mesh(), lines, sides, [mat], 60.0))

    def tubes_at(self, f, name, lines, mat, sides):
        self.tubes(name, f.tubes(lines), mat, sides)

    def prop(self, key, pl, tints=()):
        t = [(n, rgb(h)) for n, h in tints]
        for ps in self.props:
            if ps.key == key and ps.tints == t:
                ps.items.append(pl)
                return
        self.props.append(PropSet(key, [pl], t))

    def prop_at(self, f, key, u, v, z, yaw, tints=()):
        p = f.pt(u, v)
        self.prop(key, Placement((p[0], p[1], z), f.yaw + yaw), tints)

    def prop_full(self, f, key, u, v, z, yaw, tilt, scale, tints=()):
        p = f.pt(u, v)
        self.prop(key, Placement((p[0], p[1], z), f.yaw + yaw, tilt, scale), tints)

    def socket_at(self, f, kind, id_, u, v, z, d, props):
        p = f.pt(u, v)
        self.sockets.append(Socket(kind, id_, (p[0], p[1], z), f.vec(d[0], d[1]), props))

    def spot(self, pos, d, power, angle_deg, color):
        ln = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        r = 1.0 / ln
        self.spots.append(Spot(pos, (d[0] * r, d[1] * r, d[2] * r), power, angle_deg, 0.6, color))


# -- catalogue helpers (catalog.rs num / text) ------------------------------------------------------

def num(v, k, d):
    x = v.get(k) if isinstance(v, dict) else None
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else d


def text(v, k, d):
    x = v.get(k) if isinstance(v, dict) else None
    return x if isinstance(x, str) else d


def tier_spec(cat, facility, tier):
    tiers = cat["facilities"].get(facility, {}).get("tiers")
    if not isinstance(tiers, list):
        return None
    i = max(tier, 1) - 1
    return tiers[i] if i < len(tiers) else None


def fits(cat, facility, tier, kind):
    t = tier_spec(cat, facility, tier)
    return isinstance(t, dict) and isinstance(t.get("fits"), list) and kind in t["fits"]


def tier_count(cat, facility):
    f = cat["facilities"].get(facility)
    tiers = f.get("tiers") if isinstance(f, dict) else None
    return len(tiers) if isinstance(tiers, list) else 0


def option(cat, facility, tier, opts, key):
    """A facility option: the plot's value, else the option default, else the tier's value."""
    if isinstance(opts, dict) and opts.get(key) is not None:
        return opts[key]
    d = cat["facilities"].get(facility, {}).get("options", {}).get(key, {})
    if isinstance(d, dict) and d.get("default") is not None:
        return d["default"]
    t = tier_spec(cat, facility, tier)
    return t.get(key) if isinstance(t, dict) else None


def option_bool(cat, facility, tier, opts, key):
    o = cat["facilities"].get(facility, {}).get("options", {}).get(key, {})
    mt = o.get("min_tier") if isinstance(o, dict) else None
    min_tier = int(mt) if isinstance(mt, int) and not isinstance(mt, bool) and mt >= 0 else 1
    v = option(cat, facility, tier, opts, key)
    return tier >= min_tier and v is True


def rround(x):
    """Rust's f64::round: halves away from zero (Python's round() goes to even)."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


# -- 2D vectors as tuples ---------------------------------------------------------------------------

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, k):
    return (a[0] * k, a[1] * k)


def length(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def normalize(a):
    r = 1.0 / length(a)
    return (a[0] * r, a[1] * r)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]
