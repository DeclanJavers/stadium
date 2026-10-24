"""Stand geometry: cross-sections swept along the stand front line. Pure numpy, no Blender.

A stand is a set of *runs* around the pitch (straight sides and, for bowls, corner arcs). Each run is
divided into *bays* (~8 m, sized to fit). Each bay of each tier is a few closed prisms: the seating
block, a half-aisle at each end with intermediate steps, and optionally a vomitory. Sections live in
(s, z): s outward from the front line, z up. A run maps (t, s, z) to the world.

Outputs are plain arrays (verts, faces, per-face material tags, UVs) and seat anchors, so the game
can run this same code (or a port of it) when the player edits a stand.
"""
import math

import numpy as np

from .sightlines import WALL, soffit_z

BAY_TARGET = 8.0
HALF_AISLE = 0.5
AISLE_RISER_MAX = 0.19
VOM_WIDTH = 2.2
VOM_HEADROOM = 2.2

TAG_CONCRETE, TAG_FASCIA, TAG_SOFFIT, TAG_BLOCK = 0, 1, 2, 3


# -- runs -------------------------------------------------------------------------------------

class StraightRun:
    def __init__(self, name, start, direction, length):
        self.name = name
        self.p0 = np.asarray(start, float)
        self.t = np.asarray(direction, float) / np.linalg.norm(direction)
        self.n = np.array([self.t[1], -self.t[0]])  # outward = right-hand normal (runs go anticlockwise)
        self.t0, self.t1 = 0.0, float(length)
        self.curved = False

    def frame(self, t):
        return self.p0 + self.t * t, self.n

    def metric(self, s):
        return 1.0

    def samples(self, ta, tb, s_max):
        return np.array([ta, tb])


class ArcRun:
    def __init__(self, name, centre, radius, a0, a1):
        self.name = name
        self.c = np.asarray(centre, float)
        self.r0 = float(radius)
        self.t0, self.t1 = float(a0), float(a1)
        self.curved = True

    def frame(self, t):
        n = np.array([math.cos(t), math.sin(t)])
        return self.c + self.r0 * n, n

    def metric(self, s):
        return self.r0 + s

    def samples(self, ta, tb, s_max):
        n = max(2, int(math.ceil(abs(tb - ta) * (self.r0 + s_max) / 1.0)) + 1)
        return np.linspace(ta, tb, n)


def perimeter_runs(A, B, sides, corners, corner_radius):
    """Front-line runs. A, B: half-sizes of the front-line rectangle. sides: {name: cfg or None}."""
    runs = []
    R = corner_radius if corners == "curved" else 0.0
    defs = {
        "south": ((-A + R, -B), (1, 0), 2 * A - 2 * R),
        "east": ((A, -B + R), (0, 1), 2 * B - 2 * R),
        "north": ((A - R, B), (-1, 0), 2 * A - 2 * R),
        "west": ((-A, B - R), (0, -1), 2 * B - 2 * R),
    }
    for name in ("south", "east", "north", "west"):
        cfg = sides.get(name)
        if cfg is None or cfg is False:
            continue
        start, d, length = defs[name]
        if corners != "curved" and cfg.get("length"):
            # shorter stand, centred on its side
            full = length
            length = min(full, float(cfg["length"]))
            start = np.asarray(start, float) + np.asarray(d, float) * (full - length) / 2
        runs.append((name, StraightRun(name, start, d, length)))
    if corners == "curved":
        for name, c, a0 in (("SE", (A - R, -B + R), -math.pi / 2), ("NE", (A - R, B - R), 0.0),
                            ("NW", (-A + R, B - R), math.pi / 2), ("SW", (-A + R, -B + R), math.pi)):
            runs.append((name, ArcRun(name, c, R, a0, a0 + math.pi / 2)))
    return runs


def max_corner_radius(A, B, half_length, half_width, margin=2.0):
    """Largest corner radius for a bowl whose front line is the A x B rectangle: each corner arc
    must stay `margin` metres outside the corner of the field of play, and the straight sides must
    keep some length."""
    def ok(R):
        cx, cy = A - R, B - R
        if half_length <= cx or half_width <= cy:
            return True                      # the field corner is not inside the arc's quadrant
        return math.hypot(half_length - cx, half_width - cy) <= R - margin
    lo, hi = 0.0, min(A, B) - 1.0
    if ok(hi):
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def bays(run, s_mid):
    width = (run.t1 - run.t0) * run.metric(s_mid)
    n = max(1, int(round(width / BAY_TARGET)))
    edges = np.linspace(run.t0, run.t1, n + 1)
    return list(zip(edges[:-1], edges[1:]))


# -- sections ---------------------------------------------------------------------------------

def _tread_path(tier, aisle=False, rows=None):
    """Top surface polyline of the rows (front nosing -> back of last tread)."""
    s, h, T = tier["s"], tier["h"], tier["tread"]
    rows = range(tier["rows"]) if rows is None else rows
    pts = []
    for i in rows:
        pts.append((s[i], h[i]))
        if i + 1 < tier["rows"] and (i + 1) in rows:
            R = h[i + 1] - h[i]
            n = max(1, int(math.ceil(R / AISLE_RISER_MAX))) if aisle else 1
            for k in range(1, n):
                pts.append((s[i] + k * T / n, h[i] + (k - 1) * R / n))
                pts.append((s[i] + k * T / n, h[i] + k * R / n))
            pts.append((s[i] + T, h[i] + (n - 1) * R / n))
        else:
            pts.append((s[i] + T, h[i]))
    return pts


def _bottom(tier, s):
    return 0.0 if tier["soffit_front"] is None else soffit_z(tier, s)


def tier_section(tier, aisle=False, back_parapet=1.1):
    """Closed polygon [(s, z)] and per-edge tags for a full tier block."""
    s0, h0, sb = tier["s0"], tier["h"][0], tier["s_back"]
    top = h0 + tier["parapet"]
    hb = tier["h"][-1]
    pts = [(s0 - WALL, _bottom(tier, s0 - WALL)), (s0 - WALL, top), (s0, top)]
    tags = [TAG_FASCIA, TAG_FASCIA, TAG_CONCRETE]
    path = _tread_path(tier, aisle)
    pts += path
    tags += [TAG_CONCRETE] * len(path)
    pts += [(sb, hb + back_parapet), (sb + WALL, hb + back_parapet), (sb + WALL, _bottom(tier, sb + WALL))]
    tags += [TAG_CONCRETE, TAG_CONCRETE, TAG_SOFFIT if tier["soffit_front"] is not None else TAG_CONCRETE]
    return _clean(pts, tags)


def vom_sections(tier, r0, k):
    """Two polygons for a vomitory bay: the floor block, and the block above the tunnel roof."""
    s, h, T = tier["s"], tier["h"], tier["tread"]
    s0, sb, hb = tier["s0"], tier["s_back"], tier["h"][-1]
    floor = h[r0]
    lower = [(s0 - WALL, _bottom(tier, s0 - WALL)), (s0 - WALL, h[0] + tier["parapet"]), (s0, h[0] + tier["parapet"])]
    lower += _tread_path(tier, rows=range(0, r0 + 1))[:-1]
    lower += [(s[r0], floor), (sb + WALL, floor), (sb + WALL, _bottom(tier, sb + WALL))]
    ltags = [TAG_FASCIA, TAG_FASCIA] + [TAG_CONCRETE] * (len(lower) - 3) + [TAG_SOFFIT]
    ceil = floor + VOM_HEADROOM
    upper = [(s[r0 + k], ceil), (s[r0 + k], h[r0 + k])]
    upper += _tread_path(tier, rows=range(r0 + k, tier["rows"]))[1:]
    upper += [(sb, hb + 1.1), (sb + WALL, hb + 1.1), (sb + WALL, ceil)]
    utags = [TAG_CONCRETE] * len(upper)
    return [_clean(lower, ltags), _clean(upper, utags)]


def vom_rows(tier):
    """Row index where the vomitory opens and how many rows it removes (None if it cannot fit)."""
    if not tier["vomitories"] or tier["rows"] < 10:
        return None
    r0 = max(2, tier["rows"] // 4) if tier["index"] == 0 else 2
    for k in range(2, tier["rows"] - r0):
        if tier["h"][r0 + k] - tier["h"][r0] >= VOM_HEADROOM + 0.3:
            return r0, k
    return None


def block_section(lower, upper):
    """Concourse block under an upper tier, from the back of the tier below to the back of the stand."""
    sa = lower["s_back"] + WALL
    sb = max(upper["s_back"] + WALL, sa + 3.0)
    front = upper["s0"] - WALL
    pts = [(sa, 0.0), (sb, 0.0)]
    top_b = soffit_z(upper, min(sb, upper["s_back"] + WALL))
    pts.append((sb, top_b))
    if front > sa:  # set-back tier: a flat concourse deck in front of it
        pts += [(front, soffit_z(upper, front)), (front, upper["soffit_front"]), (sa, upper["soffit_front"])]
    else:
        pts.append((sa, soffit_z(upper, sa)))
    return _clean(pts, [TAG_BLOCK] * len(pts))


def _clean(pts, tags):
    out, otags = [], []
    for p, t in zip(pts, tags):
        if out and abs(p[0] - out[-1][0]) < 1e-6 and abs(p[1] - out[-1][1]) < 1e-6:
            continue
        out.append(p)
        otags.append(t)
    if abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
        otags.pop()
    return np.array(out, float), otags


# -- extrusion --------------------------------------------------------------------------------

class MeshAcc:
    """Accumulates polygons (mixed sizes), per-face tags and per-loop UVs."""

    def __init__(self):
        self.verts, self.faces, self.tags, self.uvs = [], [], [], []
        self.n = 0

    def add_prism(self, run, ta, tb, section, cap_tag=TAG_CONCRETE):
        poly, tags = section
        K = len(poly)
        smax = poly[:, 0].max()
        ts = run.samples(ta, tb, smax)
        J = len(ts)
        base = self.n
        per = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1))])
        for t in ts:
            p, n = run.frame(t)
            xy = p[None, :] + poly[:, :1] * n[None, :]
            self.verts.append(np.column_stack([xy, poly[:, 1]]))
        self.n += J * K
        along = (ts - ts[0]) * run.metric(0.0)
        for j in range(J - 1):
            for i in range(K):
                a, b = base + j * K + i, base + j * K + (i + 1) % K
                self.faces.append((a, b, b + K, a + K))
                self.tags.append(tags[i])
                self.uvs += [(along[j], per[i]), (along[j], per[i + 1]), (along[j + 1], per[i + 1]),
                             (along[j + 1], per[i])]
        for j, flip in ((0, True), (J - 1, False)):
            idx = [base + j * K + i for i in range(K)]
            self.faces.append(tuple(reversed(idx)) if flip else tuple(idx))
            self.tags.append(cap_tag)
            uv = [tuple(q) for q in poly]
            self.uvs += list(reversed(uv)) if flip else uv

    def arrays(self):
        V = np.concatenate(self.verts) if self.verts else np.zeros((0, 3))
        return V, self.faces, np.array(self.tags, np.int32), np.array(self.uvs, float)


# -- a whole run --------------------------------------------------------------------------------

def build_run(run, tiers, seat_pitch_default=0.5, vom_every=3):
    """Geometry, seat anchors, rail polylines and sockets for one run of a stand."""
    acc = MeshAcc()
    anchors = []      # dicts: pos, yaw, tier, row, t, pitch, bay
    rails = []        # (polyline (N,3), radius)
    nosings = []      # aisle step nosing segments: (p0, p1) world
    barriers = []
    sockets = []
    s_mid = np.mean([t["s0"] for t in tiers] + [t["s_back"] for t in tiers])
    bay_list = bays(run, s_mid)
    for tier in tiers:
        full = tier_section(tier)
        aisle = tier_section(tier, aisle=True)
        vr = vom_rows(tier)
        vom_sec = vom_sections(tier, *vr) if vr else None
        pitch_target = tier.get("seat_pitch", seat_pitch_default)
        for b, (ta, tb) in enumerate(bay_list):
            ha = HALF_AISLE / run.metric(s_mid)
            acc.add_prism(run, ta, ta + ha, aisle)
            acc.add_prism(run, tb - ha, tb, aisle)
            width_mid = (tb - ta) * run.metric(s_mid)
            has_vom = vom_sec is not None and b % vom_every == 1 and width_mid >= VOM_WIDTH + 3.0
            if has_vom:
                tm = 0.5 * (ta + tb)
                hv = 0.5 * VOM_WIDTH / run.metric(tier["s"][vr[0]])
                acc.add_prism(run, ta + ha, tm - hv, full)
                acc.add_prism(run, tm + hv, tb - ha, full)
                for sec in vom_sec:
                    acc.add_prism(run, tm - hv, tm + hv, sec)
                rails += _vom_rails(run, tier, vr, tm, hv)
            else:
                acc.add_prism(run, ta + ha, tb - ha, full)
            if tier["kind"] == "seated":
                anchors += _anchors(run, tier, b, ta + ha, tb - ha, pitch_target,
                                    vom=(vr, 0.5 * (ta + tb), VOM_WIDTH / 2) if has_vom else None)
            else:
                barriers += _barriers(run, tier, b, ta + ha, tb - ha)
            nosings += _aisle_nosings(run, tier, ta, ha)
            nosings += _aisle_nosings(run, tier, tb - ha, ha)
            # fascia (ads / LED ribbon) and roof sockets, one per bay
            p, n = run.frame(0.5 * (ta + tb))
            sockets.append(("fascia", f"{run.name}_T{tier['index']}_B{b}",
                            (*(p + n * (tier["s0"] - WALL)), tier["h"][0] + tier["parapet"]), n,
                            {"width": round((tb - ta) * run.metric(tier["s0"]), 3), "tier": tier["index"]}))
        # aisle handrails on steep tiers, on every bay boundary
        risers = np.diff(tier["h"])
        if len(risers) and risers.max() / tier["tread"] > 0.45:
            for ta, _ in bay_list[1:]:
                rails.append((_row_line(run, ta, tier, 0.9, range(1, tier["rows"])), 0.022))
        if tier is tiers[-1]:
            for b, (ta, tb) in enumerate(bay_list):
                p, n = run.frame(0.5 * (ta + tb))
                sockets.append(("roof_anchor", f"{run.name}_B{b}",
                                (*(p + n * (tier["s_back"] + WALL)), tier["h"][-1] + 1.1), n,
                                {"width": round((tb - ta) * run.metric(tier["s_back"]), 3)}))
    for lower, upper in zip(tiers[:-1], tiers[1:]):
        blk = block_section(lower, upper)
        for b, (ta, tb) in enumerate(bay_list):
            acc.add_prism(run, ta, tb, blk)
        _rakers(acc, run, upper, bay_list)
    return {"mesh": acc.arrays(), "anchors": anchors, "rails": rails, "nosings": nosings,
            "barriers": barriers, "sockets": sockets, "bays": len(bay_list), "bay_list": bay_list}


def stand_depth(tiers):
    """Furthest-back structure of the stand section (tiers and concourse blocks)."""
    s = max(t["s_back"] + WALL for t in tiers)
    for lower, upper in zip(tiers[:-1], tiers[1:]):
        s = max(s, upper["s_back"] + WALL, lower["s_back"] + WALL + 3.0)
    return s


def section_hull(tiers):
    """Outline of the whole stand section (for gable walls): convex hull of every block polygon."""
    pts = []
    for t in tiers:
        pts += tier_section(t)[0].tolist()
    for lower, upper in zip(tiers[:-1], tiers[1:]):
        pts += block_section(lower, upper)[0].tolist()
    P = sorted(set(map(tuple, pts)))
    if len(P) < 3:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower_h, upper_h = [], []
    for q in P:
        while len(lower_h) >= 2 and cross(lower_h[-2], lower_h[-1], q) <= 0:
            lower_h.pop()
        lower_h.append(q)
    for q in reversed(P):
        while len(upper_h) >= 2 and cross(upper_h[-2], upper_h[-1], q) <= 0:
            upper_h.pop()
        upper_h.append(q)
    return np.array(lower_h[:-1] + upper_h[:-1])


def _world(run, t, s, z):
    p, n = run.frame(t)
    q = p + n * s
    return np.array([q[0], q[1], z])


def _yaw_facing_pitch(n):
    # local +Y must point along -n (towards the pitch)
    return math.atan2(n[0], -n[1])


def _anchors(run, tier, bay, ta, tb, pitch_target, vom=None):
    out = []
    T = tier["tread"]
    for i in range(tier["rows"]):
        s = tier["s"][i] + T
        m = run.metric(s)
        segments = [(ta, tb)]
        if vom is not None:
            (r0, k), tm, half = vom
            if r0 <= i < r0 + k:
                hv = (half + 0.1) / m
                segments = [(ta, tm - hv), (tm + hv, tb)]
        for sa, sb in segments:
            avail = (sb - sa) * m
            n = int(avail // pitch_target)
            if n <= 0:
                continue
            pitch = avail / n
            for j in range(n):
                t = sa + (j + 0.5) * (sb - sa) / n
                p, nrm = run.frame(t)
                out.append({"pos": _world(run, t, s, tier["h"][i]), "yaw": _yaw_facing_pitch(nrm),
                            "tier": tier["index"], "row": i, "t": t, "pitch": pitch, "bay": bay,
                            "seat": j, "of": n, "run": run.name, "riser": (tier["h"][i + 1] - tier["h"][i])
                            if i + 1 < tier["rows"] else 0.0})
    return out


def _row_line(run, t, tier, height, rows):
    pts = []
    for i in rows:
        pts.append(_world(run, t, tier["s"][i] + 0.05, tier["h"][i] + height))
    return np.array(pts)


def _vom_rails(run, tier, vr, tm, hv):
    r0, k = vr
    rails = []
    for side in (-1, 1):
        t = tm + side * hv
        rails.append((_row_line(run, t, tier, 1.0, range(r0 + 1, r0 + k + 1)), 0.024))
    s_front = tier["s"][r0 + k] + 0.06
    z = tier["h"][r0 + k] + 1.0
    n = 8
    ts = np.linspace(tm - hv, tm + hv, n)
    rails.append((np.array([_world(run, t, s_front, z) for t in ts]), 0.024))
    return rails


def _aisle_nosings(run, tier, ta, ha):
    """Yellow safety nosings on every aisle step edge."""
    segs = []
    pts = _tread_path(tier, aisle=True)
    for (s1, z1), (s2, z2) in zip(pts[:-1], pts[1:]):
        if abs(z2 - z1) < 1e-6 and s2 > s1:  # a horizontal going: nosing at its front edge
            segs.append((_world(run, ta, s1, z1), _world(run, ta + ha, s1, z1)))
    return segs


def _barriers(run, tier, bay, ta, tb):
    """Crush barriers on standing terraces: every fifth step, staggered bay to bay."""
    out = []
    for i in range(2 + (bay % 2) * 2, tier["rows"], 5):
        s = tier["s"][i] + 0.12
        z = tier["h"][i] + 1.1
        m = run.metric(s)
        ts = np.linspace(ta + 0.3 / m, tb - 0.3 / m, max(2, int((tb - ta) * m / 1.0) + 1))
        out.append(np.array([_world(run, t, s, z) for t in ts]))
    return out


def _rakers(acc, run, tier, bay_list):
    """Raking beams under an upper tier's soffit, one on every bay boundary."""
    sa, sb = tier["s0"] - WALL + 0.4, tier["s_back"] + WALL
    za, zb = soffit_z(tier, sa), soffit_z(tier, sb)
    depth = 0.9
    poly = np.array([(sa, za), (sb, zb), (sb, zb - depth), (sa + 1.5, za - 0.35)])
    for ta, _ in bay_list:
        w = 0.2 / run.metric(sa)
        acc.add_prism(run, ta - w, ta + w, (poly, [TAG_SOFFIT] * 4))
    ta = bay_list[-1][1]
    w = 0.2 / run.metric(sa)
    acc.add_prism(run, ta - w, ta + w, (poly, [TAG_SOFFIT] * 4))
