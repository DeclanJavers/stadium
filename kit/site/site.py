"""Training grounds: a club-owned site from a site-layout/1 document (mirrors
viewer/gen/src/site/mod.rs). Plain Python: no Blender until the result is realised.

The land is fixed (a template from catalog/sites.json) and split into plots. Every plot is in one
state: occupied by something the club inherited, being demolished, cleared, under construction,
or built on with a facility at a tier, in a style, with options. The output is renderer-neutral:
parts (meshes or tube sets) with material references, instanced props, pitches (built by the
pitch generator), spot lights, sockets and a report.

    cat = load_catalog(); site = build(layout, cat)
"""
import json
import math
import re
from pathlib import Path

from ..stands.prng import crc32, uniform
from . import academy, construction, facilities, kernel, landscape, occupants
from .kernel import Frame
from .model import BOUNDARIES, SCREENING, STATES, Club, Out, Plot, fits, num, text, tier_count, tier_spec

ROOT = Path(__file__).resolve().parents[2]


def load_catalog():
    """The catalogues the site generator reads (as the Rust Catalog: `_doc` removed)."""
    out = {}
    for name in ("sites", "occupants", "facilities", "styles", "rooms"):
        d = json.loads((ROOT / "catalog" / f"{name}.json").read_text())
        d.pop("_doc", None)
        out[name] = d
    return out


_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def is_hex(v):
    return isinstance(v, str) and _HEX.fullmatch(v) is not None


def _q(s):
    """Rust's {:?} for a string."""
    return json.dumps(s)


def sanitize(layout, cat):
    """A sanitised copy of the layout and the warnings for what was changed. Every plot of the
    template gets an entry; merged plots replace their parts."""
    w = []
    lay = json.loads(json.dumps(layout)) if isinstance(layout, dict) else {}
    tpl_name = text(lay, "template", "edge_of_town")
    if not (isinstance(cat["sites"].get(tpl_name), dict) and tpl_name != "plot_kinds"):
        w.append(f"unknown site template {_q(tpl_name)}; using edge_of_town")
        tpl_name = "edge_of_town"
    lay["template"] = tpl_name
    tpl = cat["sites"][tpl_name]
    club = lay["club"] if isinstance(lay.get("club"), dict) else {}
    c2 = {"name": text(club, "name", "KIT FC")}
    for k, d in (("primary", "#c8102e"), ("secondary", "#f2f2f2")):
        v = club.get(k, d)
        if not is_hex(v):
            w.append(f"club {k} colour {json.dumps(v)} is not a #rrggbb colour; using {d}")
            c2[k] = d
        else:
            c2[k] = v
    lay["club"] = c2
    style = text(lay, "style", "classic")
    if style not in cat["styles"]:
        w.append(f"unknown style {_q(style)}; using classic")
        lay["style"] = "classic"
    else:
        lay["style"] = style
    style = lay["style"]
    b = lay["boundary"] if isinstance(lay.get("boundary"), dict) else {}
    b2 = {"type": text(b, "type", "chainlink"), "screening": text(b, "screening", "none")}
    if b2["type"] not in BOUNDARIES:
        w.append(f"unknown boundary {json.dumps(b2['type'])}; using chainlink")
        b2["type"] = "chainlink"
    if b2["screening"] not in SCREENING:
        w.append(f"unknown screening {json.dumps(b2['screening'])}; using none")
        b2["screening"] = "none"
    lay["boundary"] = b2
    g = lay["gate"] if isinstance(lay.get("gate"), dict) else {}
    gs = g.get("style")
    t = min(max(float(_rround(num(g, "tier", 1.0))), 1.0), 3.0)
    lay["gate"] = {"tier": int(t), "style": gs if isinstance(gs, str) and gs in cat["styles"] else style}
    ls = lay["landscape"] if isinstance(lay.get("landscape"), dict) else {}
    lay["landscape"] = {"trees": min(max(num(ls, "trees", 0.5), 0.0), 1.0)}
    given = lay["plots"] if isinstance(lay.get("plots"), dict) else {}
    tpl_ids = [p["id"] for p in tpl.get("plots", []) if isinstance(p.get("id"), str)]
    merges = tpl.get("merges") if isinstance(tpl.get("merges"), list) else []
    out_plots = {}
    taken = []
    for m in merges:
        mid = m.get("id", "")
        if isinstance(given.get(mid), dict):
            parts = [x for x in m.get("plots", []) if isinstance(x, str)]
            for x in parts:
                if x in given:
                    w.append(f"plot {x} is part of {mid}; its own entry is ignored")
            taken.extend(parts)
            out_plots[mid] = given[mid]
    for k in given:
        if k not in tpl_ids and not any(m.get("id") == k for m in merges):
            w.append(f"unknown plot {_q(k)} in template {tpl_name}")
    for pid in tpl_ids:
        if pid not in taken:
            out_plots[pid] = given.get(pid)

    def kind_of(pid):
        d = next((p for p in tpl.get("plots", []) if p.get("id") == pid), None) or next((m for m in merges if m.get("id") == pid), None)
        k = d.get("kind") if d else None
        return k if isinstance(k, str) else "medium"
    seed = int(num(lay, "seed", 1.0))
    occ_keys = list(cat["occupants"].keys())
    for pid in list(out_plots):
        kind = kind_of(pid)
        c = out_plots[pid]
        c = dict(c) if isinstance(c, dict) else {}
        state = text(c, "state", "occupied")
        if state not in STATES:
            w.append(f"{pid}: unknown state {_q(state)}; using occupied")
            state = "occupied"

        def occ_fits(o):
            v = cat["occupants"].get(o)
            return isinstance(v, dict) and isinstance(v.get("fits"), list) and kind in v["fits"]
        if state in ("occupied", "demolition"):
            occ = text(c, "occupant", "")
            if not occ_fits(occ):
                fitting = [o for o in occ_keys if occ_fits(o)]
                pick = fitting[int(uniform(seed + crc32(pid), 0) * len(fitting)) % len(fitting)]
                if occ:
                    w.append(f"{pid}: occupant {_q(occ)} does not fit a {kind} plot; using {pick}")
                c["occupant"] = pick
        if state in ("construction", "built"):
            fac = text(c, "facility", "")
            n = tier_count(cat, fac)
            if n == 0:
                w.append(f"{pid}: unknown facility {_q(fac)}; plot left cleared")
                state = "cleared"
            else:
                want = min(int(max(_rround(num(c, "tier", 1.0)), 1.0)), n)
                tier = next((tt for tt in range(want, 0, -1) if fits(cat, fac, tt, kind)), None)
                if tier is not None:
                    if tier != want:
                        w.append(f"{pid}: tier {want} {fac} does not fit a {kind} plot; using tier {tier}")
                    c["tier"] = tier
                else:
                    w.append(f"{pid}: a {fac} does not fit a {kind} plot; plot left cleared")
                    state = "cleared"
                s = c.get("style")
                if isinstance(s, str) and s not in cat["styles"]:
                    w.append(f"{pid}: unknown style {_q(s)}; using the site style")
                    del c["style"]
                if not isinstance(c.get("options"), dict):
                    c["options"] = {}
        c["state"] = state
        out_plots[pid] = c
    lay["plots"] = out_plots
    return lay, w


def _rround(x):
    from .model import rround
    return rround(x)


def front_vec(f):
    return {"x": (1.0, 0.0), "-x": (-1.0, 0.0), "y": (0.0, 1.0)}.get(f, (0.0, -1.0))


def plots(layout, cat):
    """The plots to generate, in template order (a merge takes its first part's place)."""
    tpl = cat["sites"][layout["template"]]
    kinds = cat["sites"]["plot_kinds"]
    seed = int(num(layout, "seed", 1.0))
    out, done = [], []
    merges = tpl.get("merges") if isinstance(tpl.get("merges"), list) else []
    for p in tpl.get("plots", []):
        pid = p.get("id", "")
        merged = next((m for m in merges if pid in m.get("plots", []) and m.get("id", "") in layout["plots"]), None)
        d = merged or p
        did = d.get("id", "")
        if did in done:
            continue
        done.append(did)
        kind = d.get("kind") if isinstance(d.get("kind"), str) else "medium"
        size = kinds.get(kind, {}).get("size", [50.0, 40.0])
        cfg = layout["plots"].get(did)
        out.append(Plot(did, kind, float(size[0]), float(size[1]), Frame(num(d, "x", 0.0), num(d, "y", 0.0), num(d, "yaw", 0.0)),
                        front_vec(text(d, "front", "-y")), cfg if cfg is not None else {"state": "occupied"},
                        (seed * 1000003 + crc32(did)) & ((1 << 64) - 1)))
    return out


class Report:
    def __init__(self, template):
        self.template = template
        self.plots = []
        self.pitches = self.grass = self.hybrid = self.artificial = self.lit = 0
        self.replica = False
        self.parking = 0
        self.area_m2 = self.developed_m2 = 0.0
        self.walk_m = None
        self.academy = None
        self.warnings = []


class Site:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build(layout, cat):
    """Build a training ground from a site-layout/1 document."""
    lay, warnings = sanitize(layout, cat)
    tpl_name = lay["template"]
    tpl = cat["sites"][tpl_name]
    night = text(lay, "lighting", "day") == "night"
    club = Club(text(lay["club"], "name", "KIT FC"), text(lay["club"], "primary", "#c8102e"), text(lay["club"], "secondary", "#f2f2f2"))
    o = Out(warnings, night, club)
    bnd = [(float(p[0]), float(p[1])) for p in tpl.get("boundary", [])]
    ps = plots(lay, cat)
    style = lay["style"]
    seed = int(num(lay, "seed", 1.0))
    landscape.context(o, tpl, bnd, seed)
    landscape.site_ground(o, bnd)
    landscape.roads(o, tpl, ps)
    landscape.boundary(o, tpl, bnd, lay["boundary"], seed)
    landscape.gate(o, cat, tpl, lay["gate"], seed)
    report = Report(tpl_name)
    for p in ps:
        state = p.state()
        pr = {"id": p.id, "kind": p.kind, "state": state, "label": "", "facility": None, "tier": 0, "style": None, "area_m2": p.w * p.d}
        if state in ("occupied", "demolition"):
            occ = text(p.cfg, "occupant", "scrubland")
            v = cat["occupants"].get(occ)
            pr["label"] = text(v, "label", occ) if isinstance(v, dict) else occ
            occupants.build(o, p, occ, state == "demolition")
            if state == "demolition":
                construction.demolition(o, p)
                pr["label"] = f"Demolishing {pr['label'].lower()}"
        elif state == "cleared":
            pr["label"] = "Cleared"
            construction.cleared(o, p)
        elif state in ("construction", "built"):
            fac = text(p.cfg, "facility", "")
            tier = int(num(p.cfg, "tier", 1.0))
            st = p.cfg.get("style") if isinstance(p.cfg.get("style"), str) else style
            ts = tier_spec(cat, fac, tier)
            tl = text(ts, "label", "") if isinstance(ts, dict) else ""
            fs = cat["facilities"].get(fac)
            fl = text(fs, "label", fac) if isinstance(fs, dict) else fac
            pr.update(facility=fac, tier=tier, style=st)
            if state == "construction":
                pr["label"] = f"Building: {fl} ({tl})"
                construction.site(o, cat, p, fac, tier, st)
            else:
                pr["label"] = f"{fl}: {tl}"
                facilities.build(o, cat, p, fac, tier, st)
                report.developed_m2 += p.w * p.d
        o.socket_at(p.frame, "plot", p.id, 0.0, 0.0, 0.0, p.front, [("width", p.w), ("depth", p.d)])
        report.plots.append(pr)
    landscape.trees(o, tpl, bnd, ps, num(lay["landscape"], "trees", 0.5), seed)
    for sp in o.pitches:
        if "gk" in sp.plot:
            continue
        report.pitches += 1
        if sp.surface == "artificial":
            report.artificial += 1
        elif sp.surface == "hybrid":
            report.hybrid += 1
        else:
            report.grass += 1
        report.lit += 1 if sp.lit else 0
    report.replica = any(p.state() == "built" and text(p.cfg, "facility", "") == "pitch" and int(num(p.cfg, "tier", 1.0)) >= 4 for p in ps)
    report.parking = o.parking
    report.academy = academy.grade(ps, cat, o.pitches)
    report.area_m2 = abs(kernel.area2(bnd)) / 2.0
    if o.entrance is not None and o.pitches:
        e = o.entrance
        report.walk_m = max(math.sqrt((sp.origin[0] - e[0]) ** 2 + (sp.origin[1] - e[1]) ** 2) for sp in o.pitches)
    report.warnings = o.warnings
    lo = (min(p[0] for p in bnd), min(p[1] for p in bnd))
    hi = (max(p[0] for p in bnd), max(p[1] for p in bnd))
    c = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    span = math.sqrt((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2)
    camera = ((c[0] - span * 0.45, c[1] - span * 0.62, span * 0.42), (c[0], c[1] + span * 0.02, 0.0))
    return Site(name=text(lay, "name", "Training ground"), night=night, parts=o.parts, props=o.props, pitches=o.pitches, spots=o.spots,
                sockets=o.sockets, camera=camera, boundary=bnd, report=report)
