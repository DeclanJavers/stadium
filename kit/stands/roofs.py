"""Roofs over a stand run. Pure numpy, same (t, s, z) run coordinates as the stands.

Two structural systems:
  cantilever  trusses at every bay boundary, fixed to a back column with a tied back-span; the roof
              rises towards the pitch. The modern default; wraps bowls continuously.
  goalpost    a front girder carried on posts standing in the front rows, rafters back to the rear
              wall. The classic British grandstand (posts obstruct some views, as they really do);
              with posts on every bay it is the small covered stand of non-league grounds.

The solver keeps the leading edge clear of the back row's view of a ball 15 m above the far
touchline (raising the edge if needed) and caps cantilevers at `max_cantilever`.
"""
import math

import numpy as np

from .geometry import MeshAcc, WALL, _world, stand_depth
from .sightlines import SEATED_EYE, SEATED_EYE_FWD

TAG_DECK, TAG_TRANSLUCENT, TAG_FASCIA, TAG_UNDERSIDE = 0, 1, 2, 3
BALL_HEIGHT = 15.0
HEADROOM = 2.5

DEFAULTS = {
    "cantilever": {"rise_deg": 5.0, "back_clearance": 3.0, "cover": 0.0, "back_span": 0.22,
                   "translucent": 0.25, "max_cantilever": 60.0, "truss_every": 1},
    "goalpost": {"slope_deg": 3.0, "back_clearance": 2.6, "cover": 1.0, "post_every": 3,
                 "girder_depth": None, "translucent": 0.0, "clad_girder": True},
}


def _line_z(eye, target, s):
    (se, ze), (sp, zp) = eye, target
    return ze + (zp - ze) * (se - s) / (se - sp)


def solve(tiers, spec, d_focal, far_width):
    """Roof section for one run. far_width: distance from the near to the far touchline/goal line."""
    kind = spec.get("type", "cantilever")
    p = dict(DEFAULTS[kind], **{k: v for k, v in spec.items() if v is not None})
    top = tiers[-1]
    s_back = stand_depth(tiers)
    h_last = max(t["h"][-1] for t in tiers)
    s_front = tiers[0]["s0"] - WALL
    eye = (top["s"][-1] + top["tread"] - SEATED_EYE_FWD, top["h"][-1] + SEATED_EYE)
    target = (-(d_focal + far_width), BALL_HEIGHT)
    r = {"type": kind, "s_back": s_back, "params": p}
    if kind == "cantilever":
        s_f = max(s_front - p["cover"], s_back - p["max_cantilever"])
        L = s_back - s_f
        z_b = h_last + p["back_clearance"]
        z_f = z_b + L * math.tan(math.radians(p["rise_deg"]))
        need = _line_z(eye, target, s_f) + 0.5
        raised = max(0.0, need - z_f)
        z_f += raised
        d_b = float(np.clip(L / 7.0, 1.2, 9.0))
        d_f = float(np.clip(L / 40.0, 0.5, 1.5))
        r.update(s_front=s_f, z_back=z_b, z_front=z_f, depth_back=d_b, depth_front=d_f, length=L,
                 back_span=L * p["back_span"], raised_for_sightline=round(raised, 3))
    else:
        s_g = s_front + p["cover"]
        L = s_back - s_g
        z_b = h_last + p["back_clearance"]
        z_g = z_b - L * math.tan(math.radians(p["slope_deg"]))
        need = _line_z(eye, target, s_g) + 0.3
        raised = max(0.0, need - z_g)
        z_g += raised
        r.update(s_front=s_g, z_back=z_b, z_front=z_g, length=L, raised_for_sightline=round(raised, 3),
                 s_post=s_front + 0.45)
    # headroom: the underside (a straight line back -> front) must stay >= 2.5 m over every covered tread
    zb, zf, sb, sf = r["z_back"], r["z_front"], r["s_back"], r["s_front"]
    need_f = zf
    for t in tiers:
        for s_i, h_i in zip(t["s"], t["h"]):
            if sf <= s_i < sb - 1e-6:
                need_f = max(need_f, zb + (h_i + HEADROOM - zb) * (sb - sf) / (sb - s_i))
    r["raised_for_headroom"] = round(need_f - zf, 3)
    r["z_front"] = need_f
    r["clearance_to_sightline"] = round(r["z_front"] - _line_z(eye, target, r["s_front"]), 3)
    return r


# -- geometry ---------------------------------------------------------------------------------

def _truss_nodes(roof):
    s_b, s_f = roof["s_back"], roof["s_front"]
    z_b, z_f = roof["z_back"], roof["z_front"]
    d_b, d_f = roof["depth_back"], roof["depth_front"]
    n = max(3, int(round(roof["length"] / max(1.5, 0.6 * (d_b + d_f)))))
    u = np.linspace(0, 1, n + 1)
    s = s_b + (s_f - s_b) * u
    zb = z_b + (z_f - z_b) * u
    zt = zb + d_b + (d_f - d_b) * u
    return s, zb, zt


def _top_z(roof, s):
    """Top chord height of a cantilever at section position s (extended over the back-span)."""
    s_b, s_f = roof["s_back"], roof["s_front"]
    zt_b = roof["z_back"] + roof["depth_back"]
    zt_f = roof["z_front"] + roof["depth_front"]
    return zt_b + (zt_f - zt_b) * (s_b - s) / (s_b - s_f)


def _deck(acc, run, roof, s_a, s_b_end, z_of, tags_main, translucent):
    """Deck plate along the top of the structure, front translucent strip optional."""
    edges = [s_a]
    if translucent > 0:
        edges.append(s_a + translucent * (s_b_end - s_a))
    edges.append(s_b_end)
    for k, (s1, s2) in enumerate(zip(edges[:-1], edges[1:])):
        tag = TAG_TRANSLUCENT if (translucent > 0 and k == 0) else TAG_DECK
        z1, z2 = z_of(s1), z_of(s2)
        poly = np.array([(s1, z1), (s2, z2), (s2, z2 + 0.12), (s1, z1 + 0.12)])
        acc.add_prism(run, run.t0, run.t1, (poly, [TAG_UNDERSIDE, TAG_FASCIA, tag, TAG_FASCIA]), cap_tag=TAG_FASCIA)


def build(run, roof, bay_list, include_ends=True):
    """Mesh (deck, fascia), structural tubes [(polyline, radius)] and sockets for one run."""
    acc = MeshAcc()
    tubes = []
    sockets = []
    bounds = [ta for ta, _ in bay_list] + [bay_list[-1][1]]
    if not include_ends:
        bounds = bounds[1:-1]
    if roof["type"] == "cantilever":
        s, zb, zt = _truss_nodes(roof)
        L = roof["length"]
        r_chord = 0.09 + L / 450
        r_web = 0.05 + L / 900
        bs = roof["back_span"]
        s_tail = roof["s_back"] + bs
        every = roof["params"]["truss_every"]
        for k, t in enumerate(bounds):
            if k % every and t not in (bounds[0], bounds[-1]):
                continue
            W = lambda ss, zz, t=t: _world(run, t, ss, zz)
            tubes.append((np.array([W(a, b) for a, b in zip(s, zb)]), r_chord))
            tubes.append((np.array([W(s_tail, _top_z(roof, s_tail))] + [W(a, b) for a, b in zip(s, zt)]), r_chord))
            for i in range(len(s) - 1):  # Warren web
                a, b = (i, i + 1)
                tubes.append((np.array([W(s[a], zb[a] if i % 2 == 0 else zt[a]),
                                        W(s[b], zt[b] if i % 2 == 0 else zb[b])]), r_web))
            tubes.append((np.array([W(s[0], zb[0]), W(s[0], zt[0])]), r_web))
            tubes.append((np.array([W(s[0], 0.0), W(s[0], zt[0])]), max(0.3, r_chord * 2.2)))       # column
            tubes.append((np.array([W(s_tail, 0.0), W(s_tail, _top_z(roof, s_tail))]), r_chord * 0.8))  # tie
        # purlins along the run at every top-chord node
        ts = run.samples(run.t0, run.t1, roof["s_back"])
        for a, b in zip(s[1:], zt[1:]):
            tubes.append((np.array([_world(run, t, a, b + r_chord + 0.02) for t in ts]), 0.06))
        lift = r_chord + 0.08  # sheeting sits on purlins above the truss chords
        _deck(acc, run, roof, roof["s_front"] - 0.35, s_tail, lambda ss: _top_z(roof, ss) + lift,
              None, roof["params"]["translucent"])
        z_edge_lo = roof["z_front"] - 0.35
        z_edge_hi = _top_z(roof, roof["s_front"]) + 0.3
    else:
        p = roof["params"]
        s_g, z_g, s_b = roof["s_front"], roof["z_front"], roof["s_back"]
        bay_w = (bay_list[0][1] - bay_list[0][0]) * run.metric(s_g)
        span = p["post_every"] * bay_w
        dg = p["girder_depth"] or float(np.clip(span / 10.0, 0.6, 3.0))
        roof["girder_depth"] = dg
        z_top_back = roof["z_back"] + 0.5
        z_of = lambda ss: (z_g + dg) + (z_top_back - z_g - dg) * (ss - s_g) / (s_b - s_g)
        ts = run.samples(run.t0, run.t1, s_g)
        ts = np.linspace(run.t0, run.t1, max(len(ts), int((run.t1 - run.t0) * run.metric(s_g) / dg) + 1))
        r = 0.06 + dg / 30
        tubes.append((np.array([_world(run, t, s_g, z_g) for t in ts]), r))
        tubes.append((np.array([_world(run, t, s_g, z_g + dg) for t in ts]), r))
        for i in range(len(ts) - 1):
            tubes.append((np.array([_world(run, ts[i], s_g, z_g + (dg if i % 2 else 0)),
                                    _world(run, ts[i + 1], s_g, z_g + (0 if i % 2 else dg))]), r * 0.6))
        for k, t in enumerate(bounds):
            tubes.append((np.array([_world(run, t, s_g, z_g + dg), _world(run, t, s_b, z_top_back)]), 0.12))
            if k % p["post_every"] == 0 or t == bounds[-1]:
                tubes.append((np.array([_world(run, t, roof["s_post"], 0.0), _world(run, t, roof["s_post"], z_g)]),
                              0.16))
        _deck(acc, run, roof, s_g - 0.3, s_b + 0.3, z_of, None, p["translucent"])
        if p["clad_girder"]:
            poly = np.array([(s_g - 0.35, z_g - 0.05), (s_g + 0.25, z_g - 0.05), (s_g + 0.25, z_g + dg + 0.25),
                             (s_g - 0.35, z_g + dg + 0.25)])
            acc.add_prism(run, run.t0, run.t1, (poly, [TAG_UNDERSIDE, TAG_FASCIA, TAG_DECK, TAG_FASCIA]),
                          cap_tag=TAG_FASCIA)
        z_edge_lo, z_edge_hi = z_g - 0.05, z_g + dg + 0.25
    for b, (ta, tb) in enumerate(bay_list):
        tm = 0.5 * (ta + tb)
        p_, n = run.frame(tm)
        loc = p_ + n * (roof["s_front"] - 0.4)
        sockets.append(("roof_edge", f"{run.name}_B{b}", (float(loc[0]), float(loc[1]), float(z_edge_lo)), n,
                        {"width": round((tb - ta) * run.metric(roof["s_front"]), 3),
                         "edge_height": round(z_edge_hi - z_edge_lo, 3)}))
    edge_line = np.array([_world(run, t, roof["s_front"] - 0.4, z_edge_lo)
                          for t in run.samples(run.t0, run.t1, roof["s_front"])])
    if roof["type"] == "cantilever":
        tail_s, tail_top = roof["s_back"] + roof["back_span"], _top_z(roof, roof["s_back"] + roof["back_span"]) + 0.3
    else:
        tail_s, tail_top = roof["s_back"] + 0.3, roof["z_back"] + 0.7
    return {"mesh": acc.arrays(), "tubes": tubes, "sockets": sockets, "edge_line": edge_line,
            "tail_s": tail_s, "tail_top": tail_top}
