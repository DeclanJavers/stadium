"""Sockets: named empties that define where other assets attach (see docs/ARCHITECTURE.md)."""
import math

import bpy

from .scene import link

PREFIX = "SOCKET"


def socket(kind, ident, location, yaw_deg=0.0, parent=None, col=None, **props):
    """Create a socket. yaw_deg rotates about Z; the attached asset's local +Y follows it."""
    e = bpy.data.objects.new(f"{PREFIX}:{kind}:{ident}", None)
    e.empty_display_type = "ARROWS"
    e.empty_display_size = 0.6
    e.location = location
    e.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
    e["socket_kind"] = kind
    e["socket_id"] = ident
    for k, v in props.items():
        e[k] = v
    return link(e, col, parent)


def sockets_of(root, kind=None):
    from .scene import descendants
    bpy.context.view_layer.update()  # matrix_world is stale until the depsgraph re-evaluates
    out = []
    for o in descendants(root):
        if o.name.startswith(PREFIX + ":") and (kind is None or o.get("socket_kind") == kind):
            out.append(o)
    return out


def attach(asset_root, sock):
    """Snap an asset onto a socket: origin to socket, orientation to socket."""
    bpy.context.view_layer.update()
    asset_root.matrix_world = sock.matrix_world.copy()
    asset_root["attached_to"] = sock.name
