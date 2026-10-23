"""Review rendering: Cycles on the GPU under a physical sky."""
import math
from pathlib import Path

import bpy
from mathutils import Vector

from .nodes import NB
from .scene import link, look_at_rotation


def setup_cycles(samples=128, res=(1600, 900), denoise=True):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        sc.cycles.device = "GPU"
    except Exception:
        sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 0.0


def setup_sky(sun_elevation=38.0, sun_azimuth=200.0, strength=0.05, sun_strength=4.5):
    """Physical sky for ambient light plus a matching sun lamp for crisp shadows.

    The multiple-scattering sky is in physical units; strength 0.05 makes skylight ~18% of the
    illuminance on the ground next to a 4.5 W/m² sun, as on a clear day (measured, see git log).
    """
    sc = bpy.context.scene
    world = bpy.data.worlds.new("Sky")
    sc.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nb = NB(world.node_tree)
    sky = nb.node("ShaderNodeTexSky", sky_type="MULTIPLE_SCATTERING")
    sky.sun_disc = False
    sky.sun_elevation = math.radians(sun_elevation)
    sky.sun_rotation = math.radians(sun_azimuth)
    bg = nb.node("ShaderNodeBackground")
    nb.connect(sky.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    out = nb.node("ShaderNodeOutputWorld")
    nb.connect(bg.outputs[0], out.inputs["Surface"])

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = sun_strength
    sun.angle = math.radians(0.53)
    sun.color = (1.0, 0.97, 0.92)
    obj = link(bpy.data.objects.new("Sun", sun))
    # Blender sun shines along its local -Z. Point it from the sky direction.
    el, az = math.radians(sun_elevation), math.radians(sun_azimuth)
    to_sun = Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)))
    obj.rotation_euler = to_sun.to_track_quat("Z", "Y").to_euler()
    return sky, obj


def camera(name, location, target, lens=35.0, sensor=36.0, dof=None):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.sensor_width = sensor
    cam.clip_start = 0.02
    cam.clip_end = 2000
    if dof:
        cam.dof.use_dof = True
        cam.dof.focus_distance = (Vector(target) - Vector(location)).length
        cam.dof.aperture_fstop = dof
    obj = link(bpy.data.objects.new(name, cam))
    obj.location = location
    obj.rotation_euler = look_at_rotation(location, target)
    return obj


def render(cam, path):
    sc = bpy.context.scene
    sc.camera = cam
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return path


def surroundings(radius=4000.0):
    """Rough field grass out to the horizon so review renders get realistic ground bounce."""
    from .nodes import material, srgb
    import numpy as np
    from .mesh import disc
    from .scene import mesh_object
    m, nb = material("Review_Ground")
    pos = nb.geometry()["Position"]
    v = nb.noise(pos, 0.02, detail=4)
    col = nb.mix(v, srgb("#3c5524"), srgb("#5a6630"))
    nb.output(nb.principled(Base_Color=col, Roughness=0.85, Specular_IOR_Level=0.2))
    V, F, UV = disc((0, 0), radius, 64, z=-0.03)
    o = mesh_object("Review_Ground", V, F, materials=[m])
    o["render_only"] = True
    return o


def setup_night(strength=1.0):
    """Clear night sky: deep blue, a faint horizon glow from the city. No sun."""
    sc = bpy.context.scene
    world = bpy.data.worlds.new("Night")
    sc.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nb = NB(world.node_tree)
    geo = nb.node("ShaderNodeTexCoord").outputs["Generated"]
    z = nb.sep(nb.node("ShaderNodeNewGeometry").outputs["Incoming"])[2]
    up = nb.math("POWER", nb.clamp01(nb.mul(z, -1.0)), 0.5)
    col = nb.mix(up, (0.018, 0.016, 0.02, 1.0), (0.0015, 0.0025, 0.007, 1.0))
    bg = nb.node("ShaderNodeBackground")
    nb.connect(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    out = nb.node("ShaderNodeOutputWorld")
    nb.connect(bg.outputs[0], out.inputs["Surface"])
