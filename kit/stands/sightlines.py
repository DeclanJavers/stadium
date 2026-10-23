"""Sightline solver: tread heights for every row of every tier from the C-value.

Pure Python/numpy (no Blender) so the game can port it. Section coordinates: s = horizontal distance
outward from the stand front line, z = height above pitch level. The focal point is the nearest
touchline (or goal line for end stands) at pitch level, `d_focal` metres in front of s = 0.

For a spectator whose eye is D horizontal and N vertical from the focal point, the next row
(tread T further back) must rise by

    R = (N + C)(D + T) / D - N

for their sightline to clear the eye in front by C (Green Guide, ch. 12). Risers are rounded up to
5 mm (constructable), clamped to a 34 deg maximum rake, and the achieved C is recorded per row.
"""
import math

SEATED_EYE = 1.15        # eye above tread, seated
STANDING_EYE = 1.60
SEATED_EYE_FWD = 0.30    # eye forward of the back of the tread (over the seat)
MAX_RAKE_DEG = 34.0
RISER_STEP = 0.005
WALL = 0.25              # front/back wall thickness
C_MIN = 0.06             # Green Guide: 90 mm good, 60 mm the acceptable minimum

TIER_DEFAULTS = {
    "seated": {"tread": 0.80, "c_value": 0.09, "parapet": 0.20, "slab": 0.75},
    "standing": {"tread": 0.38, "c_value": 0.06, "parapet": 0.20, "slab": 0.75},
}


def _eye(kind, s_back_of_tread, tread):
    if kind == "standing":
        return s_back_of_tread - tread / 2, STANDING_EYE
    return s_back_of_tread - SEATED_EYE_FWD, SEATED_EYE


def solve_tier(rows, s0, h0, d_focal, kind="seated", tread=None, c_value=None, min_riser=0.0):
    d = TIER_DEFAULTS[kind]
    T = tread or d["tread"]
    C = c_value if c_value is not None else d["c_value"]
    max_r = T * math.tan(math.radians(MAX_RAKE_DEG))
    h = [h0]
    c_achieved = []
    for i in range(rows - 1):
        ex, eh = _eye(kind, s0 + (i + 1) * T, T)
        D = d_focal + ex
        N = h[-1] + eh
        R = (N + C) * (D + T) / D - N
        R = math.ceil(R / RISER_STEP) * RISER_STEP
        R = min(max(R, min_riser), max_r)
        h.append(h[-1] + R)
        c_achieved.append((N + R) * D / (D + T) - N)
    return {"rows": rows, "kind": kind, "tread": T, "c_target": C, "s0": s0,
            "h": h, "s": [s0 + i * T for i in range(rows)], "c_achieved": c_achieved,
            "s_back": s0 + rows * T}


def solve_profile(profile, d_focal):
    """Resolve every tier of a stand profile into concrete row positions."""
    tiers = []
    for k, spec in enumerate(profile["tiers"]):
        kind = spec.get("kind", "seated")
        dflt = TIER_DEFAULTS[kind]
        if k == 0:
            s0 = WALL
            h0 = profile.get("front_height", 0.9)
            soffit_front = None
        else:
            prev = tiers[-1]
            s0 = prev["s_back"] - spec.get("overhang", 4.0)
            slab = spec.get("slab", dflt["slab"])
            soffit_front = prev["h"][-1] + spec.get("clearance", 3.0)
            h0 = soffit_front + slab
        args = (spec["rows"], s0, h0, d_focal, kind, spec.get("tread"), spec.get("c_value"))
        t = solve_tier(*args)
        requested_s0 = s0
        c_min = spec.get("c_min", C_MIN)
        # An upper tier pushed too far forward is steeper than the maximum rake from the touchline and
        # no riser can give it a view. Step it back until every row clears c_min, as a designer would.
        while k > 0 and t["c_achieved"] and min(t["c_achieved"]) < c_min - 1e-6 and s0 < requested_s0 + 80:
            s0 += 0.25
            t = solve_tier(spec["rows"], s0, h0, d_focal, kind, spec.get("tread"), spec.get("c_value"))
        t["setback"] = round(s0 - requested_s0, 2)
        t.update(index=k, parapet=spec.get("parapet", dflt["parapet"] if k == 0 else 0.85),
                 glass=spec.get("glass", 0.3 if k > 0 else 0.0), slab=spec.get("slab", dflt["slab"]),
                 soffit_front=soffit_front, seat_pitch=spec.get("seat_pitch", 0.5),
                 vomitories=spec.get("vomitories", spec["rows"] >= 12),
                 hospitality=spec.get("hospitality", False))
        if soffit_front is not None:
            # soffit runs parallel to the nosing line, `slab` below it
            rake = (t["h"][-1] - t["h"][0]) / max(1e-6, (t["rows"] - 1) * t["tread"])
            t["soffit_slope"] = rake
        tiers.append(t)
    return tiers


def soffit_z(tier, s):
    """Underside height of an upper tier at section position s."""
    return tier["soffit_front"] + tier.get("soffit_slope", 0.0) * (s - (tier["s0"] - WALL))


def report(tiers):
    out = []
    for t in tiers:
        risers = [b - a for a, b in zip(t["h"][:-1], t["h"][1:])]
        rake = math.degrees(math.atan(max(risers) / t["tread"])) if risers else 0.0
        out.append({"tier": t["index"], "rows": t["rows"], "kind": t["kind"],
                    "front_tread_m": round(t["h"][0], 3), "back_tread_m": round(t["h"][-1], 3),
                    "max_rake_deg": round(rake, 1),
                    "min_c_mm": round(1000 * min(t["c_achieved"]), 1) if t["c_achieved"] else None,
                    "front_s_m": round(t["s0"], 2), "setback_m": t.get("setback", 0.0)})
    return out
