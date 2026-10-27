"""Academy grading (mirrors viewer/gen/src/site/academy.rs): which academy category a site's
built facilities reach. The requirements are modelled on the facility criteria of the Premier
League's Elite Player Performance Plan (Category 4, the entry level, up to Category 1), simplified
to what the kit builds; each category also needs everything below it.
"""
from .model import num, option, text

LEVELS = (4, 3, 2, 1)


def requirements(ps, cat, pitches):
    """The requirements per category: {category: [(label, met)]}."""
    built = []
    for p in ps:
        if p.state() != "built":
            continue
        fac = text(p.cfg, "facility", "")
        tier = int(num(p.cfg, "tier", 1.0))
        opts = p.cfg.get("options") if isinstance(p.cfg.get("options"), dict) else {}
        built.append((p.id, fac, tier, opts))

    def has(fac, tier=1):
        return any(f == fac and t >= tier for _, f, t, _ in built)

    def modules(fac, tier):
        for _, f, t, o in built:
            if f == fac and t >= tier:
                mv = option(cat, f, t, o, "modules")
                if isinstance(mv, list):
                    yield [m for m in mv if isinstance(m, str)]
    formats = set()
    for _, f, t, o in built:
        if f == "youth_pitches":
            fv = option(cat, f, t, o, "format")
            formats.add(fv if isinstance(fv, str) else "9v9")
    full_size = sum(1 for _, f, t, _ in built if f in ("pitch", "showpiece_pitch") or (f == "artificial_pitch" and t >= 2))
    building = max([t for _, f, t, _ in built if f in ("academy_building", "training_building")] or [0])
    art_ids = {pid for pid, f, _, _ in built if f == "artificial_pitch"}
    lit = sum(1 for sp in pitches if sp.lit)
    lit_art = any(sp.lit and sp.plot in art_ids for sp in pitches)
    return {
        4: [("Two full-size pitches", full_size >= 2),
            ("Changing rooms (an academy or training building)", building >= 1)],
        3: [("Foundation-phase pitches (5v5 or 7v7)", bool(formats & {"5v5", "7v7"})),
            ("9v9 pitches", "9v9" in formats),
            ("U13-U14 11v11 pitches", "11v11_u13" in formats),
            ("A floodlit pitch", lit >= 1),
            ("An academy block or better", building >= 2)],
        2: [("Indoor training (an air dome or hall)", has("indoor_hall")),
            ("A floodlit artificial pitch", lit_art),
            ("A goalkeeper area", has("goalkeeper_area")),
            ("A showpiece match pitch", has("showpiece_pitch")),
            ("An academy centre with classrooms and sports science",
             any("classrooms" in m and "sports_science" in m for m in modules("academy_building", 3)))],
        1: [("A full-size indoor hall", has("indoor_hall", 3)),
            ("A showpiece pitch with a covered stand", has("showpiece_pitch", 2)),
            ("A Category One centre with a scholars' lodge", any("boarding" in m for m in modules("academy_building", 4))),
            ("Secure arrival (drop-off with controlled entry)", has("drop_off", 3)),
            ("Four floodlit pitches", lit >= 4),
            ("Three full-size pitches", full_size >= 3)],
    }


def grade(ps, cat, pitches):
    """{"category": the best category met (None: not yet Category 4), "next": the one above
    (None at Category 1), "missing": what the next one still needs}."""
    req = requirements(ps, cat, pitches)
    category = None
    for level in LEVELS:
        if all(ok for _, ok in req[level]):
            category = level
        else:
            return {"category": category, "next": level, "missing": [label for label, ok in req[level] if not ok]}
    return {"category": category, "next": None, "missing": []}
