"""Reference assembler: turns a stadium-layout/1 JSON document into a Blender scene by snapping
assets onto sockets. Game engines implement the same rule (docs/ARCHITECTURE.md)."""
import importlib

from .core.scene import collection
from .core.sockets import attach, sockets_of


def _family(name):
    return importlib.import_module(f"kit.assets.{name}")


def assemble(layout):
    layout = layout or {}
    state = {}
    pitch_mod = _family("pitch")
    root, p, lines = pitch_mod.build(layout.get("pitch"))
    state.update(pitch=root, pitch_params=p, lines=lines)

    goals_cfg = layout.get("goals", {"style": "classic"})
    if goals_cfg is not None:
        goal_mod = _family("goal")
        built = None
        for sock in sockets_of(root, "goal"):
            # build once, then instance the same data for the other end (identical nets are realistic
            # enough, and it halves cloth-simulation time)
            g = goal_mod.build(goals_cfg, collection(f"Goal_{sock['socket_id']}"), template=built)
            built = built or g
            attach(g, sock)
        state["goals"] = True

    night = layout.get("lighting") == "night"
    state["night"] = night
    if layout.get("stands"):
        st = _family("stand").build(dict(layout["stands"], _night=night), p, collection("Stands"))
        state["stands"] = st
        print(f"[assemble] capacity {st['capacity']}", flush=True)

    dug = layout.get("dugouts")
    tun = layout.get("tunnel")
    gap = _family("dugout").board_gap(dug) if dug else (6.0 if tun else 0.0)
    if layout.get("ads"):
        state["ads"] = _family("ads").build(layout["ads"], p, state.get("stands"), night, collection("Ads"), gap=gap)
    if layout.get("screens"):
        state["screens"] = _family("screen").build(layout["screens"], p, state.get("stands"), night,
                                                   collection("Screens"))
    if dug:
        state["dugouts"] = _family("dugout").build(dug, p, night, collection("Dugouts"))
    if tun:
        state["tunnel"] = _family("tunnel").build(tun, p, state.get("stands"), collection("Tunnel"))

    if layout.get("floodlights"):
        state["floodlights"] = _family("floodlight").build(layout["floodlights"], p, state.get("stands"), night,
                                                          collection("Floodlights"))

    flags_cfg = layout.get("corner_flags", {"preset": "pro_spring"})
    if flags_cfg is not None:
        flag_mod = _family("corner_flag")
        template = None
        for sock in sockets_of(root, "corner_flag"):
            f = flag_mod.build(flags_cfg, collection(f"CornerFlag_{sock['socket_id']}"), template=template)
            template = template or f
            attach(f, sock)
    return state
