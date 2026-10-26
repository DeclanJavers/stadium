#!/usr/bin/env bash
# Colab: rig every /content/anim/geo/<id>.glb with UniRig -> /content/anim/rigged/<id>.glb
# (skeleton prediction, skinning prediction, then merged back onto the input mesh).
#   SEED=12345 bash run_unirig.sh [id ...]
set -uo pipefail
unset PYTHONPATH
export HF_HOME=/content/hf_cache
export PATH=/content/envs/unirig/bin:$PATH
cd /content/UniRig
mkdir -p /content/anim/rigged /content/anim/unirig_tmp
ids=("$@")
if [ ${#ids[@]} -eq 0 ]; then
  for f in /content/anim/geo/*.glb; do ids+=("$(basename "$f" .glb)"); done
fi
for id in "${ids[@]}"; do
  echo "== $id $(date +%T)"
  t=/content/anim/unirig_tmp/$id
  mkdir -p "$t"
  bash launch/inference/generate_skeleton.sh --input /content/anim/geo/$id.glb --output "$t/skeleton.fbx" --seed "${SEED:-12345}" \
    && bash launch/inference/generate_skin.sh --input "$t/skeleton.fbx" --output "$t/skin.fbx" \
    && bash launch/inference/merge.sh --source "$t/skin.fbx" --target /content/anim/geo/$id.glb --output /content/anim/rigged/$id.glb \
    && echo "RIGGED $id" || echo "FAILED $id"
done
