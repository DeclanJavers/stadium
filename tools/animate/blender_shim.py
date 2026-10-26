#!/content/envs/unimate/bin/python
"""`blender -b [--python-exit-code N] -P script.py -- args` for an env that has Blender only as
the pip bpy module (UniMate's mesh-animation step shells out to a `blender` command)."""
import runpy
import sys

import bpy  # noqa: F401  (loads Blender before the script imports it)

a = sys.argv[1:]
script = a[a.index("-P") + 1]
rest = a[a.index("--") + 1:] if "--" in a else []
sys.argv = ["blender", "--"] + rest
runpy.run_path(script, run_name="__main__")
