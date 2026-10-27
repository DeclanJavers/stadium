#!/usr/bin/env bash
# Colab: reference images -> BiRefNet cut-outs -> review sheets, all under /content/props.
#   REDO=a,b bash tools/run_stage1.sh     (REDO regenerates those images after a prompt edit)
# Leaves review/stage1.zip: contact sheets of the images and the cut-outs, for review before
# make_props.py spends GPU time on 3D.
set -uo pipefail
cd /content/props
M=/content/envs/moss/bin/python
H=/content/envs/hi3d/bin/python
while pgrep -f tools/gen_images.py > /dev/null; do sleep 10; done
if [ -n "${REDO:-}" ]; then
  $M tools/gen_images.py --manifest generated_props.json --out images --ids "$REDO" --force
fi
$M tools/gen_images.py --manifest generated_props.json --out images
$H tools/make_cutouts.py --manifest generated_props.json --images images --out cutouts

mkdir -p review
mapfile -t IDS < <($M -c "import json; [print(p['id']) for p in json.load(open('generated_props.json'))['props']]")
half=$(( (${#IDS[@]} + 1) / 2 ))
for part in 1 2; do
  start=$(( (part - 1) * half ))
  imgs=(); cuts=()
  for i in "${IDS[@]:$start:$half}"; do imgs+=("images/$i.png"); cuts+=("cutouts/$i.png"); done
  $M tools/contact_sheet.py review/images_$part.jpg "${imgs[@]}" 2>&1 | tail -1
  $M tools/contact_sheet.py review/cutouts_$part.jpg "${cuts[@]}" 2>&1 | tail -1
done
cd review && rm -f stage1.zip && zip -q stage1.zip images_*.jpg cutouts_*.jpg && echo "stage1 ready"
