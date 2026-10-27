#!/usr/bin/env bash
# One-time Colab setup for the prop batch, on top of the text_to_3d notebook's two venvs
# (/content/envs/moss and /content/envs/hi3d). Adds Hunyuan3D-Paint for textures:
#   - CUDA 12.4 toolkit, because Colab ships CUDA 13 and the paint rasterizer must be
#     compiled against the hi3d env's torch 2.5 + cu124;
#   - the custom_rasterizer CUDA extension and the mesh_processor C++ extension;
#   - Hi3D weights as plain files (the HF cache's symlinks trip Hi3D's loader at res 512)
#     and the paint + delight weights.
set -euo pipefail
unset PYTHONPATH
export HF_HOME=/content/hf_cache
H=/content/envs/hi3d/bin/python
CONSTRAINTS=/content/Hi3Dv1.0/docker/constraints.txt

if [ ! -x /usr/local/cuda-12.4/bin/nvcc ]; then
  echo "== CUDA 12.4 toolkit"
  wget -q -O /content/cuda124.run \
    https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
  mkdir -p /content/cuda_tmp
  sh /content/cuda124.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.4 \
    --no-man-page --no-drm --override --tmpdir=/content/cuda_tmp
  rm -rf /content/cuda124.run /content/cuda_tmp
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
nvcc --version | tail -1

echo "== texture extensions"
$H -m pip install -q -c "$CONSTRAINTS" pybind11 xatlas pygltflib
cd /content/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
$H -m pip install -q --no-build-isolation .
cd /content/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
# mesh_render imports it relatively, so the .so has to sit next to it.
$H setup.py build_ext --inplace > /dev/null

echo "== weights"
$H - <<'EOF'
from huggingface_hub import snapshot_download
files = ["model_index.json", "scheduler/*", "vae/*", "transformer_512/*", "transformer_1024/*"]
snapshot_download("hi3d-official/Hi3Dv1.0", local_dir="/content/models/hi3d", allow_patterns=files)
snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"])
# the refiner's image conditioners (Sparc3DPipeline loads them with local_files_only)
snapshot_download("facebook/dinov2-large")
snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.txt", "*.safetensors"])
EOF

cd /content
$H -c "import torch, custom_rasterizer, xatlas; from hy3dgen.texgen import Hunyuan3DPaintPipeline; print('paint ready')"
