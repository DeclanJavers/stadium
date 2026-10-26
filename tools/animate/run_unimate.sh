#!/usr/bin/env bash
# Colab: UniMate motions for UniRig-rigged props, for comparison with the authored cycles.
#   bash run_unimate.sh label   <id ...>   rule-based joint labels -> /content/anim/unimate/rig/<id>/annotation.json
#   bash run_unimate.sh build   <id ...>   build the asset dir from the (reviewed) annotation.json
#   bash run_unimate.sh sample  <id> "<prompt>" ["<prompt>" ...]   generate + animate -> .../samples/<id>/animated
# UniMate checkpoints are CC BY-NC 4.0: these motions are for comparison, not for shipping.
set -uo pipefail
unset PYTHONPATH
export HF_HOME=/content/hf_cache
export PATH=/content/envs/unimate/bin:$PATH
cd /content/UniMate
R=/content/anim/unimate
mkdir -p $R/rig $R/samples
cmd=$1; shift
case $cmd in
  label)
    for id in "$@"; do
      python -m data_process.rig_preprocess run --input /content/anim/rigged/$id.glb --output_dir $R/rig/$id \
        --annotate rule --overwrite && echo "LABELLED $id" || echo "FAILED $id"
    done ;;
  build)
    for id in "$@"; do
      python -m data_process.rig_preprocess run --input /content/anim/rigged/$id.glb --output_dir $R/rig/$id \
        --annotation $R/rig/$id/annotation.json --overwrite && echo "BUILT $id" || echo "FAILED $id"
    done ;;
  sample)
    id=$1; shift
    python -m unimate.inference.sample --exp_dir outputs/unimate_uniml3d_f60_v3 --asset $R/rig/$id \
      --prompt "$@" --num_repetitions 2 --cfg_scale 5 --output_dir $R/samples/$id \
      && bash scripts/run_animate_motion.sh $R/samples/$id && echo "SAMPLED $id" || echo "FAILED $id" ;;
esac
