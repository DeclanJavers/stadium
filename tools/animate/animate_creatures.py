"""Author looping animation cycles on UniRig-rigged props and export them.

    blender -b --factory-startup -P tools/animate/animate_creatures.py -- [id ...]

Reads out/animate/rigs/<id>.rig.blend (rig_inspect.py) and catalog/creature_anims.json, builds
each action from periodic functions of the loop phase (so the last frame meets the first),
and writes out/assets/genprop.<id>.anim.glb: the textured, skinned prop with one glTF
animation per action ("idle", "walk", "trot", "peck", "flap", "sniff", "call"). Walk and trot cycles plant the feet with IK
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
        c.chain_count = 2
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
            elif name == "peck":
                peck(rig, p, f, n)
            elif name == "flap":
                wings_flap(rig, p, f, n, act["amplitude"])
            elif name == "sniff":
                sniff(rig, p, f, n)
            elif name == "call":
                long_call(rig, p, f, n)
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
    return {"frames": n, "seconds": round(n / FPS, 2), "root_speed": round(speed, 3)}


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
    rig.arm["actions"] = json.dumps({name: dict(a, loop=True) for name, a in actions.items()}, sort_keys=True)
    rig.arm["fps"] = FPS
    rig.mesh.data.name = rig.mesh.name  # a game sees mesh names: not the generator's temp file name
    bpy.ops.object.select_all(action="DESELECT")
    rig.arm.select_set(True)
    rig.mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig.arm
    path = OUT / f"genprop.{rig.pid}.anim.glb"
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True, export_yup=True,
                              export_animations=True, export_animation_mode="NLA_TRACKS",
                              export_force_sampling=True, export_extras=True, export_skins=True)
    return path


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
