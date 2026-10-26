"""Author looping animation cycles on UniRig-rigged props and export them.

    blender -b --factory-startup -P tools/animate/animate_creatures.py -- [id ...]

Reads out/animate/rigs/<id>.rig.blend (rig_inspect.py) and catalog/creature_anims.json, builds
each action from periodic functions of the loop phase (so the last frame meets the first),
and writes out/assets/genprop.<id>.anim.glb (and .anim.lod1/.anim.lod2 from the static LODs):
the textured, skinned prop with one glTF animation per action ("idle", "walk", "trot", "peck", "flap", "sniff", "call", "takeoff", "land", "startle", "alert"). Walk and trot cycles plant the feet with IK
targets on a stance/swing path, then bake to plain bone keys; they play in place, and the
action's root_speed (m/s) is how fast to move the prop while it plays.

Bone rotations are given about kit-space axes (forward, left, up) with the sign chosen from
the rest geometry (e.g. "turn the neck so the head moves down"), because UniRig bones have
arbitrary roll and no names.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[2]
RIGS = ROOT / "out" / "animate" / "rigs"
OUT = ROOT / "out" / "assets"
CFG = json.loads((ROOT / "catalog" / "creature_anims.json").read_text())
FPS = CFG["fps"]
TAU = 2 * math.pi


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Rig:
    def __init__(self, pid, spec):
        bpy.ops.wm.open_mainfile(filepath=str(RIGS / f"{pid}.rig.blend"))
        self.pid, self.spec = pid, spec
        self.arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        self.mesh = next(o for o in bpy.data.objects if o.type == "MESH")
        self.names = [b.name for b in self.arm.data.bones]
        self.to_arm = self.arm.matrix_world.inverted().to_3x3()  # world -> armature space
        up = Vector((0, 0, 1))
        self.fwd = Vector(spec["forward"]).normalized()
        self.left = up.cross(self.fwd).normalized()
        self.up = up
        self.size = max(self.mesh.dimensions)
        for pb in self.arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
        bpy.context.scene.render.fps = FPS

    def bone(self, i):
        return self.arm.data.bones[self.names[i]]

    def pbone(self, i):
        return self.arm.pose.bones[self.names[i]]

    def head(self, i):
        return self.arm.matrix_world @ self.bone(i).head_local

    def tail(self, i):
        return self.arm.matrix_world @ self.bone(i).tail_local

    def sign(self, i, axis, toward, tip=None):
        """+1 or -1: the rotation direction about `axis` at bone i's head that moves `tip`
        (default: the bone's tail) along `toward`."""
        tip = self.tail(i) if tip is None else tip
        r = self.head(i)
        moved = Matrix.Rotation(0.05, 3, axis) @ (tip - r)
        return 1.0 if (moved - (tip - r)).dot(toward) >= 0 else -1.0

    def local_q(self, i, axis, deg):
        """Rotation by deg about a world-space axis, expressed in bone i's rest frame."""
        a = (self.bone(i).matrix_local.to_3x3().inverted() @ (self.to_arm @ axis)).normalized()
        return Quaternion(a, math.radians(deg))


class Pose:
    """Per-frame accumulator: bone index -> quaternion, root offset."""

    def __init__(self, rig):
        self.rig, self.q, self.root = rig, {}, Vector()
        # Rest-pose correction applied before any motion: a generated prop posed looking at the
        # camera (head_forward off the body's forward) turns its head to face where it goes.
        turn = rig.spec.get("head_turn")
        if turn:
            face = rig.tail(rig.spec["head"][-1]) + Vector(rig.spec["head_forward"]) * rig.size * 0.12
            for i, share in ((rig.spec["neck"][0], 0.15), (rig.spec["head"][0], 0.85)):  # twist at the skull, not the throat
                self.rot(i, rig.up, turn * share * rig.sign(i, rig.up, rig.fwd, tip=face))

    def rot(self, i, axis, deg):
        self.q[i] = self.rig.local_q(i, axis, deg) @ self.q.get(i, Quaternion())

    def apply(self, frame):
        rig = self.rig
        for pb in rig.arm.pose.bones:
            pb.rotation_quaternion = Quaternion()
            pb.location = Vector()
        for i, q in self.q.items():
            rig.pbone(i).rotation_quaternion = q
        r = rig.spec["root"]
        rig.pbone(r).location = rig.bone(r).matrix_local.to_3x3().inverted() @ (rig.to_arm @ self.root)
        for pb in rig.arm.pose.bones:
            pb.keyframe_insert("rotation_quaternion", frame=frame)
            pb.keyframe_insert("location", frame=frame)


# ---------------------------------------------------------------- motion layers

def breathe(rig, p, t, period, deg=1.6):
    for i in rig.spec.get("spine", []):
        p.rot(i, rig.left, deg * math.sin(TAU * t / period) * rig.sign(i, rig.left, rig.up))


def look_around(rig, p, frame, n):
    """Hold, turn, hold, turn back: the head yaws to two sides and rolls a little."""
    keys = [(0, 0), (0.10, 0), (0.17, 26), (0.45, 26), (0.52, -22), (0.85, -22), (0.93, 0), (1.0, 0)]
    u = frame / n
    for (u0, a0), (u1, a1) in zip(keys, keys[1:]):
        if u0 <= u <= u1:
            yaw = a0 + (a1 - a0) * smooth((u - u0) / max(u1 - u0, 1e-6))
            break
    neck = rig.spec["neck"]
    for i in neck:
        p.rot(i, rig.up, yaw / len(neck))
    p.rot(rig.spec["head"][0], rig.fwd, yaw * 0.25)


def tail_flick(rig, p, frame, n, at=0.62, deg=10):
    u = frame / n
    k = math.exp(-((u - at) * 22) ** 2)
    for j, i in enumerate(rig.spec.get("tail", [])[:2]):
        p.rot(i, rig.left, deg * k * rig.sign(i, rig.left, rig.up) * (1 - 0.4 * j))


def head_bob(rig, p, t, half, deg):
    """Pigeon head-bob, one per step: the head thrusts forward quickly, then is held still
    in space while the body walks under it (in place: it glides back)."""
    u = (t % half) / half
    x = smooth(u / 0.3) if u < 0.3 else 1 - smooth((u - 0.3) / 0.7)
    neck = rig.spec["neck"]
    base, top = neck[0], neck[-1]
    s = rig.sign(base, rig.left, rig.fwd, tip=rig.tail(rig.spec["head"][-1]))
    p.rot(base, rig.left, deg * (x - 0.5) * s)
    p.rot(top, rig.left, -0.8 * deg * (x - 0.5) * s)  # keep the head level


def wings_flap(rig, p, t, n, amp):
    """Downstroke over 40% of the beat, upstroke over 60%; the elbow and hand lag the shoulder."""
    def stroke(u):
        u %= 1.0
        return math.cos(math.pi * u / 0.4) if u < 0.4 else -math.cos(math.pi * (u - 0.4) / 0.6)
    for chain in rig.spec["wings"].values():
        tip = rig.tail(chain[-1])
        for j, i in enumerate(chain):
            s = rig.sign(i, rig.fwd, -rig.up, tip=tip)
            lag = 0.08 * j
            # +amp is the top of the stroke: stroke() runs +1 (up) -> -1 (down) -> +1
            share = [1.0, 0.35, 0.25, 0.15][min(j, 3)]
            p.rot(i, rig.fwd, -amp * share * stroke(t / n - lag) * s)
    p.root = rig.up * (0.012 * rig.size * math.sin(TAU * (t / n) - 0.6))


def takeoff(rig, p, f, n, act):
    """One shot, with root motion: a crouch with the wings cocked, then flaps that start hard and
    fast and settle as the bird climbs away forward; the legs tuck back once it is airborne."""
    u = f / n
    lift_at = 0.18
    crouch = smooth(u / lift_at) * (1 - smooth((u - lift_at) / 0.06))
    air = max(0.0, (u - lift_at) / (1 - lift_at))
    beat = act.get("beat", 9)  # frames per wingbeat
    wings_flap(rig, p, f, beat, act["amplitude"] * (1.15 - 0.25 * air))
    p.root = (rig.up * (act["height"] * air ** 1.6 - 0.05 * rig.size * crouch)
              + rig.fwd * (act["distance"] * air ** 2))
    # The prop stands upright; in the air the body tips forward towards level flight.
    root = rig.spec["root"]
    p.rot(root, rig.left, act.get("pitch", 55) * smooth((air - 0.1) / 0.6) * rig.sign(root, rig.left, rig.fwd, tip=nose(rig)))
    tuck = smooth((air - 0.15) / 0.3)
    for leg in rig.spec.get("legs", {}).values():
        hip = leg["chain"][0]
        p.rot(hip, rig.left, 70 * tuck * rig.sign(hip, rig.left, -rig.fwd))


def land(rig, p, f, n, act):
    """One shot, with root motion, ending at the prop's origin: a level glide in from behind and
    above on shallow beats, a flare (body swings upright, big braking beats, legs reach down),
    touchdown at 70% of the clip, then the wings settle to rest and the body bobs once."""
    u = f / n
    touch = 0.7
    air = 1 - smooth(u / touch)  # 1 = still at the start of the approach, 0 = on the ground
    flare = smooth((u - 0.35) / 0.3)
    p.root = (rig.up * (act["height"] * air ** 1.3) - rig.fwd * (act["distance"] * air ** 1.6))
    settle = max(0.0, (u - touch) / (1 - touch))
    p.root += rig.up * (-0.04 * rig.size * math.sin(math.pi * min(1.0, settle * 1.6)))
    beat = act.get("beat", 9)
    amp = act["amplitude"] * (0.35 + 0.85 * flare) * (1 - smooth(settle / 0.6))
    root_shift = p.root.copy()
    wings_flap(rig, p, f, beat, amp)
    p.root = root_shift  # wings_flap sets a flap bob; the glide path wins here
    root = rig.spec["root"]
    p.rot(root, rig.left, act.get("pitch", 55) * (1 - flare) * rig.sign(root, rig.left, rig.fwd, tip=nose(rig)))
    tuck = 1 - smooth((u - 0.3) / 0.3)
    for leg in rig.spec.get("legs", {}).values():
        hip = leg["chain"][0]
        p.rot(hip, rig.left, 70 * tuck * rig.sign(hip, rig.left, -rig.fwd))


def startle(rig, p, f, n, act):
    """One shot, in place: the head snaps up and back, the body crouches and springs into a short
    hop (legs tucked in the air), lands on the spot, then the head eases back down and the
    feathers settle with a shake."""
    u = f / n
    snap = smooth(u / 0.08) * (1 - smooth((u - 0.45) / 0.35))
    neck_pitch(rig, p, -22 * snap, head_deg=-10 * snap)
    crouch = math.exp(-((u - 0.14) * 22) ** 2)
    hop = math.sin(math.pi * min(1.0, max(0.0, (u - 0.18) / 0.22)))  # airborne 18%..40%
    p.root = rig.up * (act["hop"] * hop - 0.03 * rig.size * crouch)
    for leg in rig.spec.get("legs", {}).values():
        hip = leg["chain"][0]
        p.rot(hip, rig.left, 35 * hop * rig.sign(hip, rig.left, -rig.fwd))
    shake = math.sin(TAU * u * 11) * math.exp(-((u - 0.72) * 9) ** 2)
    for i in rig.spec.get("spine", [rig.spec["root"]]):
        p.rot(i, rig.fwd, 6 * shake)
    tail_flick(rig, p, f, n, at=0.2, deg=14)


def alert(rig, p, f, n, act):
    """One shot, in place: the body freezes and rises a little onto straighter legs, the head lifts
    and turns sharply towards a sound, holds a long still stare (one small ear-twitch of the
    head), then turns back and relaxes. The tail goes still and low."""
    u = f / n
    up = smooth(u / 0.12) * (1 - smooth((u - 0.78) / 0.18))
    turn = smooth((u - 0.06) / 0.08) * (1 - smooth((u - 0.74) / 0.12))
    neck_pitch(rig, p, -18 * up, head_deg=-6 * up)
    side = act.get("side", 1)
    for i in rig.spec["neck"]:
        p.rot(i, rig.up, side * 35 * turn / len(rig.spec["neck"]))
    p.rot(rig.spec["head"][0], rig.up, side * 15 * turn)
    twitch = math.exp(-((u - 0.45) * 40) ** 2)
    p.rot(rig.spec["head"][0], rig.fwd, 6 * twitch)
    p.root = rig.up * (0.012 * rig.size * up)
    for j, i in enumerate(rig.spec.get("tail", [])):
        p.rot(i, rig.left, -8 * up * rig.sign(i, rig.left, -rig.up) / (1 + j))


def wing_balance(rig, p, f, n):
    """Raised wings held, not frozen: slow uneven swaying at the shoulder, the two sides a little
    out of step, and one small half-beat midway, as a bird balancing in a breeze."""
    u = f / n
    for k, chain in enumerate(rig.spec["wings"].values()):
        tip = rig.tail(chain[-1])
        sway = 4 * math.sin(TAU * (u * 2 + 0.17 * k)) + 2 * math.sin(TAU * (u * 3 + 0.4 * k))
        beat = 14 * math.exp(-((u - 0.55) * 18) ** 2)
        s = rig.sign(chain[0], rig.fwd, -rig.up, tip=tip)
        p.rot(chain[0], rig.fwd, (sway - beat) * s)
        p.rot(chain[1], rig.fwd, 0.5 * sway * rig.sign(chain[1], rig.fwd, -rig.up, tip=tip))


# ---------------------------------------------------------------- IK walk

def plant_feet(rig):
    """IK targets at each foot's rest position (the walk animates them; other actions leave
    them planted) and the foot/toe locks. Returns {side: (target, rest)} and the helpers."""
    scene = bpy.context.scene
    targets, helpers = {}, []
    for side, leg in rig.spec["legs"].items():
        chain = leg["chain"]
        ik = leg.get("ik", chain[2] if len(chain) > 2 else chain[-1])
        e = bpy.data.objects.new(f"ik_{side}", None)
        scene.collection.objects.link(e)
        helpers.append(e)
        e.location = rest = rig.tail(ik)
        targets[side] = (e, rest)
        c = rig.pbone(ik).constraints.new("IK")
        c.target = e
        # the whole leg from the hip down to the IK bone: with only the two lower bones, a
        # target ahead of the foot is out of reach and the foot lands frames late
        c.chain_count = chain.index(ik) + 1 if ik in chain else 2
        # The foot (or toes) keeps its rest orientation, so it lands flat instead of tipping
        # with the shin.
        lock = leg.get("lock", chain[3] if len(chain) > 3 else None)
        if lock is not None:
            r = bpy.data.objects.new(f"lock_{side}", None)
            scene.collection.objects.link(r)
            helpers.append(r)
            r.rotation_mode = "QUATERNION"
            r.rotation_quaternion = (rig.arm.matrix_world @ rig.bone(lock).matrix_local).to_quaternion()
            cr = rig.pbone(lock).constraints.new("COPY_ROTATION")
            cr.target = r
            cr.target_space = cr.owner_space = "WORLD"
        for dup, src in leg.get("copy", {}).items():
            cr = rig.pbone(int(dup)).constraints.new("COPY_ROTATION")
            cr.target, cr.subtarget = rig.arm, rig.names[src]
            cr.target_space = cr.owner_space = "WORLD"
    return targets, helpers


def bake(rig, n, helpers):
    """Bake IK and copy constraints to plain keys, then drop the helpers."""
    bpy.context.view_layer.objects.active = rig.arm
    rig.arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(frame_start=0, frame_end=n, only_selected=False, visual_keying=True,
                     clear_constraints=True, use_current_action=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    for e in helpers:
        bpy.data.objects.remove(e)


def walk(rig, act):
    n, stride, lift = act["frames"], act["stride"], act["lift"]
    legs = rig.spec["legs"]
    targets, helpers = plant_feet(rig)
    duty = act.get("duty", 0.62)  # share of the cycle a foot is on the ground
    for f in range(n + 1):
        t = f / n
        for k, (side, (e, rest)) in enumerate(targets.items()):
            # two legs alternate by default; a trot pairs diagonals via each leg's "phase"
            u = (t + legs[side].get("phase", 0.5 * k)) % 1.0
            if u < duty:  # stance: the foot is planted and slides back as the body passes
                x, z = 0.5 - u / duty, 0.0
            else:          # swing: lift and carry forward
                v = (u - duty) / (1 - duty)
                x, z = -0.5 + smooth(v), math.sin(math.pi * v)
            e.location = rest + rig.fwd * (stride * x) + rig.up * (lift * z)
            e.keyframe_insert("location", frame=f)
        p = Pose(rig)
        # body: drops at each foot contact, sways towards the stance leg
        p.root = rig.up * (-0.10 * lift * (1 + math.cos(TAU * 2 * t)) / 2)
        if "waddle" in act:
            for i in rig.spec.get("spine", [rig.spec["root"]]):
                p.rot(i, rig.fwd, act["waddle"] * math.sin(TAU * t))
        if "tail_sway" in act:  # a trotting tail swings side to side, once per stride
            for j, i in enumerate(rig.spec.get("tail", [])):
                p.rot(i, rig.up, act["tail_sway"] * math.sin(TAU * t - 0.6 * j) / (1 + j))
        if "head_bob" in act:
            head_bob(rig, p, f, n / 2, act["head_bob"])
        tail_flick(rig, p, f, n, at=0.5, deg=4)
        p.apply(f)
    bake(rig, n, helpers)
    # A planted foot slides back one stride while it is down (duty of the cycle): that is the
    # ground speed the body must travel at for the feet not to skate.
    return stride / (duty * n / FPS)


# ---------------------------------------------------------------- actions

def build(rig, name, act):
    n = act["frames"]
    action = bpy.data.actions.new(f"{name}")
    rig.arm.animation_data_create()
    rig.arm.animation_data.action = action
    speed = 0.0
    if name in ("walk", "trot"):
        speed = walk(rig, act)
    else:
        planted = act.get("plant") and rig.spec.get("legs")
        if planted:
            targets, helpers = plant_feet(rig)
        for f in range(n + 1):
            p = Pose(rig)
            if name == "idle":
                breathe(rig, p, f, n / 2)
                look_around(rig, p, f, n)
                tail_flick(rig, p, f, n)
                if "wings" in rig.spec:
                    wing_balance(rig, p, f, n)
            elif name == "peck":
                peck(rig, p, f, n)
            elif name == "flap":
                wings_flap(rig, p, f, n, act["amplitude"])
            elif name == "sniff":
                sniff(rig, p, f, n)
            elif name == "call":
                long_call(rig, p, f, n)
            elif name == "takeoff":
                takeoff(rig, p, f, n, act)
            elif name == "land":
                land(rig, p, f, n, act)
            elif name == "startle":
                startle(rig, p, f, n, act)
            elif name == "alert":
                alert(rig, p, f, n, act)
            p.apply(f)
        if planted:
            bake(rig, n, helpers)
    action = rig.arm.animation_data.action
    action.name = name
    action.use_fake_user = True
    action["root_speed"] = round(speed, 3)
    action["frames"] = n
    track = rig.arm.animation_data.nla_tracks.new()
    track.name = name
    track.strips.new(name, 0, action)
    rig.arm.animation_data.action = None
    meta = {"frames": n, "seconds": round(n / FPS, 2), "root_speed": round(speed, 3)}
    if act.get("one_shot"):  # plays once; root_motion: it carries its own movement (else it ends in place)
        meta.update(loop=False, root_motion=act.get("root_motion", True))
    return meta


def peck(rig, p, f, n):
    """Two quick jabs at the ground, then a short look up and a return."""
    u = f / n
    jab = sum(math.exp(-((u - c) * 30) ** 2) for c in (0.12, 0.30))
    lift = smooth((u - 0.45) / 0.12) * (1 - smooth((u - 0.80) / 0.15))
    neck = rig.spec["neck"]
    tip = rig.tail(rig.spec["head"][-1])
    for i in neck:
        s = rig.sign(i, rig.left, -rig.up, tip=tip)
        p.rot(i, rig.left, (26 * jab - 16 * lift) * s / len(neck))
    p.root = rig.up * (-0.012 * rig.size * jab)  # the body dips into each jab
    p.rot(rig.spec["head"][0], rig.up, 18 * lift * math.sin(TAU * u * 2))
    tail_flick(rig, p, f, n, at=0.2, deg=6)


def nose(rig):
    """A point just in front of the head: what "head down" or "beak up" is judged by."""
    return rig.tail(rig.spec["head"][-1]) + rig.fwd * rig.size * 0.12


def neck_pitch(rig, p, deg, head_deg=0.0):
    """Bend the neck so the nose goes down (positive deg) or up, spread over the neck bones,
    and tip the head itself by head_deg the same way."""
    neck = rig.spec["neck"]
    for i in neck:
        p.rot(i, rig.left, deg * rig.sign(i, rig.left, -rig.up, tip=nose(rig)) / len(neck))
    h = rig.spec["head"][0]
    p.rot(h, rig.left, head_deg * rig.sign(h, rig.left, -rig.up, tip=nose(rig)))


def sniff(rig, p, f, n):
    """The shoulders drop (front legs bend: the feet stay planted), the nose goes to the
    ground for a run of quick sniffs and sweeps a little side to side, then comes back up."""
    u = f / n
    down = smooth((u - 0.06) / 0.2) * (1 - smooth((u - 0.76) / 0.18))
    sniffs = 4 * math.sin(TAU * u * 10) * smooth((u - 0.26) / 0.05) * (1 - smooth((u - 0.70) / 0.05))
    spine = rig.spec.get("spine", [])
    for i in spine:  # pitch the back so the shoulders go down
        p.rot(i, rig.left, 30 * down * rig.sign(i, rig.left, -rig.up, tip=rig.tail(spine[-1])) / len(spine))
    neck_pitch(rig, p, 55 * down + sniffs, head_deg=5 * down)
    sweep = 10 * down * math.sin(TAU * (u - 0.26) * 1.2)
    for i in rig.spec["neck"]:
        p.rot(i, rig.up, sweep / len(rig.spec["neck"]))
    p.root = rig.up * (-0.07 * rig.size * down)
    for j, i in enumerate(rig.spec.get("tail", [])):
        p.rot(i, rig.up, 6 * math.sin(TAU * u - 0.5 * j) / (1 + j))


def long_call(rig, p, f, n):
    """A gull's long call: the head dips, then is thrown back with the beak to the sky and held
    while calling (a fast shiver), then settles."""
    u = f / n
    dip = smooth(u / 0.16) * (1 - smooth((u - 0.16) / 0.10))
    up = smooth((u - 0.22) / 0.10) * (1 - smooth((u - 0.74) / 0.16))
    shake = 3 * math.sin(TAU * u * 16) * up
    neck_pitch(rig, p, 26 * dip - 22 * up, head_deg=22 * dip - 62 * up + shake)
    for i in rig.spec.get("spine", []):
        p.rot(i, rig.left, -6 * up * rig.sign(i, rig.left, -rig.up, tip=nose(rig)))
    breathe(rig, p, f, n, deg=1.0)


def export(rig, actions):
    """The prop, its skin and one glTF animation per action. glTF has no place for per-animation
    properties, so the action table (frames, seconds, looping, root_speed: how fast the game moves
    the prop while a cycle plays) goes on the armature node's extras as "actions"."""
    rig.arm["actions"] = json.dumps({name: {"loop": True, **a} for name, a in actions.items()}, sort_keys=True)
    rig.arm["fps"] = FPS
    rig.mesh.data.name = rig.mesh.name  # a game sees mesh names: not the generator's temp file name
    path = OUT / f"genprop.{rig.pid}.anim.glb"
    export_skinned(rig, rig.mesh, path)
    # Levels of detail: the static prop's LOD meshes (own UVs and textures), skinned to the same
    # skeleton by copying weights from the full mesh, with the same actions.
    for level in (1, 2):
        src = OUT / f"genprop.{rig.pid}.lod{level}.glb"
        if not src.exists():
            continue
        lod = skinned_lod(rig, src, level)
        export_skinned(rig, lod, OUT / f"genprop.{rig.pid}.anim.lod{level}.glb")
        bpy.data.objects.remove(lod)
    return path


def skinned_lod(rig, src, level):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(src))
    new = [o for o in bpy.data.objects if o not in before]
    lod = next(o for o in new if o.type == "MESH")
    mw = lod.matrix_world.copy()
    lod.parent = None
    lod.matrix_world = mw
    for o in new:
        if o is not lod:
            bpy.data.objects.remove(o)
    lod.name = lod.data.name = f"genprop.{rig.pid}.lod{level}"
    for vg in rig.mesh.vertex_groups:
        lod.vertex_groups.new(name=vg.name)
    dt = lod.modifiers.new("weights", "DATA_TRANSFER")
    dt.object = rig.mesh
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    rig.arm.data.pose_position = "REST"  # copy weights between the two rest shapes
    with bpy.context.temp_override(object=lod, active_object=lod):
        bpy.ops.object.modifier_apply(modifier=dt.name)
    rig.arm.data.pose_position = "POSE"
    lod.modifiers.new("Armature", "ARMATURE").object = rig.arm
    lod.parent = rig.arm
    lod.matrix_parent_inverse = rig.arm.matrix_world.inverted()
    return lod


def plumage(rgb, rule):
    """Recolour the low-saturation plumage of an albedo (grey body, black bars), leaving saturated
    parts (iridescent neck, pink feet, eye) alone."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-4, (mx - mn) / np.maximum(mx, 1e-4), 0)
    grey = np.clip((0.32 - sat) / 0.12, 0, 1)[..., None]  # soft mask
    lum = rgb.mean(-1, keepdims=True)
    if rule["kind"] == "dark":
        new = rgb * rule.get("value", 0.45)
    elif rule["kind"] == "tint":
        new = lum * np.array(rule["rgb"]) * rule.get("gain", 1.1)
    elif rule["kind"] == "mantle":  # darken the mid-grey (back and wings) only; white stays white
        mid = np.clip(1 - np.abs(lum - 0.55) / 0.3, 0, 1)
        f = 1 - (1 - rule.get("value", 0.4)) * mid
        new = rgb * f
    elif rule["kind"] == "pale":
        new = lum * (1 - rule.get("lift", 0.5)) + rule.get("lift", 0.5) * np.array(rule.get("rgb", [0.9, 0.9, 0.88]))
    return np.clip(rgb * (1 - grey) + new * grey, 0, 1)


def add_variants(rig, mesh):
    """KHR_materials_variants: one alternative material per plumage in the spec's "variants",
    sharing the mesh, normals and roughness map; a game picks one per instance."""
    variants = rig.spec.get("variants")
    if not variants:
        return
    prefs = bpy.context.preferences.addons["io_scene_gltf2"].preferences
    if not prefs.KHR_materials_variants_ui:
        prefs.KHR_materials_variants_ui = True
    scene = bpy.data.scenes[0]
    names = [v.name for v in scene.gltf2_KHR_materials_variants_variants]
    base = mesh.data.materials[0]
    for vname, rule in variants.items():
        if vname not in names:
            v = scene.gltf2_KHR_materials_variants_variants.add()
            v.variant_idx = len(scene.gltf2_KHR_materials_variants_variants) - 1
            v.name = vname
            names.append(vname)
        idx = names.index(vname)
        mat = base
        if rule:
            mat = base.copy()
            mat.name = f"{base.name}.{vname}"
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image and node.image.colorspace_settings.name == "sRGB":
                    img = node.image.copy()
                    img.name = f"{node.image.name}.{vname}"
                    w, h = img.size
                    px = np.empty(w * h * 4, np.float32)
                    img.pixels.foreach_get(px)
                    px = px.reshape(-1, 4)
                    px[:, :3] = plumage(px[:, :3], rule)
                    img.pixels.foreach_set(px.ravel())
                    img.pack()
                    node.image = img
        prim = mesh.data.gltf2_variant_mesh_data.add()
        prim.material_slot_index = 0
        prim.material = mat
        prim.variants.add().variant.variant_idx = idx


def export_skinned(rig, mesh, path):
    add_variants(rig, mesh)
    bpy.ops.object.select_all(action="DESELECT")
    rig.arm.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig.arm
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_yup=True,
                              export_animations=True, export_animation_mode="NLA_TRACKS",
                              export_force_sampling=True, export_extras=True, export_skins=True,
                              # JPEG at q90: about a fifth of PNG, no visible loss on these maps
                              export_image_format="JPEG", export_image_quality=90, export_jpeg_quality=90)


def main():
    want = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    report = {}
    for pid, spec in CFG["creatures"].items():
        if want and pid not in want:
            continue
        rig = Rig(pid, spec)
        report[pid] = {name: build(rig, name, act) for name, act in spec["actions"].items()}
        bpy.ops.wm.save_as_mainfile(filepath=str(RIGS / f"{pid}.anim.blend"))
        report[pid]["glb"] = str(export(rig, dict(report[pid])).relative_to(ROOT))
        print("ANIMATED", pid, json.dumps(report[pid]), flush=True)
    path = ROOT / "out" / "animate" / "anim_report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, indent=1))


main()
