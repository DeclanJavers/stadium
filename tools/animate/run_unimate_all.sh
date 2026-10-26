#!/usr/bin/env bash
# Colab: the whole UniMate comparison in one go (survives being re-run on a fresh runtime):
# install, label the uploaded UniRig rigs, apply the reviewed labels, build, sample, animate.
set -uo pipefail
cd /content/anim
[ -x /content/envs/unimate/bin/blender ] || bash tools/colab_setup_anim.sh unimate
T="bash tools/run_unimate.sh"
IDS="pigeon_standing pigeon_pecking seagull_standing seagull_wings_up"
$T label $IDS
/content/envs/unimate/bin/python tools/apply_unimate_labels.py
$T build $IDS
$T sample pigeon_standing "An object walks forward." "An object pecks at the ground."
$T sample pigeon_pecking "An object pecks at the ground."
$T sample seagull_standing "An object walks forward." "An object looks around."
$T sample seagull_wings_up "An object flaps its wings."
cd /content/anim/unimate && rm -f ../unimate_out.zip && find samples -name "*.glb" -path "*animated*" | zip -q ../unimate_out.zip -@ && echo "UNIMATE DONE $(ls samples/*/animated/*.glb | wc -l) animated GLBs"
