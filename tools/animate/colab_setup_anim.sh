#!/usr/bin/env bash
# One-time Colab setup (an L4 is enough) for rigging and animating generated props:
#   /content/envs/unirig   Python 3.11, UniRig (MIT): skeleton + skinning prediction
#   /content/envs/unimate  Python 3.10, UniMate: text-to-motion for any skeleton. Code MIT,
#                          checkpoints CC BY-NC 4.0, so its motions are reference only
# Work happens under /content/anim.
set -euo pipefail
unset PYTHONPATH
export HF_HOME=/content/hf_cache
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p /content/envs /content/anim "$HF_HOME"
cd /content
pip -q install uv
# Blender-as-a-module needs these X/GL client libraries even headless.
apt-get -qq install -y libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libegl1 > /dev/null

WHAT=${1:-all}   # all | unirig | unimate
if [ "$WHAT" != unimate ]; then
echo "== UniRig"
[ -d /content/UniRig ] || git clone -q --depth 1 https://github.com/VAST-AI-Research/UniRig.git
[ -x /content/envs/unirig/bin/python ] || uv venv -q /content/envs/unirig --python 3.11 --seed
U=/content/envs/unirig/bin/python
$U -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
grep -v -E '^(flash_attn|bpy)' /content/UniRig/requirements.txt > /content/anim/unirig_req.txt
$U -m pip install -q -r /content/anim/unirig_req.txt
$U -m pip install -q bpy==4.2.0
$U -m pip install -q \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
$U -m pip install -q spconv-cu120
$U -m pip install -q torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
$U -m pip install -q numpy==1.26.4
$U -c "import torch, bpy, flash_attn, spconv, torch_scatter, torch_cluster; print('UniRig env ready, CUDA', torch.cuda.is_available())"
fi
[ "$WHAT" = unirig ] && exit 0

echo "== UniMate"
[ -d /content/UniMate ] || git clone -q --depth 1 https://github.com/Friedrich-M/UniMate.git
[ -x /content/envs/unimate/bin/python ] || uv venv -q /content/envs/unimate --python 3.10 --seed
M=/content/envs/unimate/bin/python
$M -m pip install -q "setuptools<81" wheel
cd /content/UniMate
$M -m pip install -q -r requirements.txt --no-build-isolation
/content/envs/unimate/bin/hf download Linzhan/UniMate --repo-type model --local-dir outputs \
  --include "unimate_uniml3d_f60_v3/*.json" --include "unimate_uniml3d_f60_v3/*.npy" \
  --include "unimate_uniml3d_f60_v3/checkpoints/checkpoint_step_150000.pt" > /dev/null
# UniMate's animate step shells out to `blender`; this env only has Blender as the bpy module.
install -m 755 /content/anim/tools/blender_shim.py /content/envs/unimate/bin/blender
$M -c "import torch, bpy, unimate; print('UniMate env ready, CUDA', torch.cuda.is_available())"
echo "setup done"
