#!/usr/bin/env bash
# Run Bringing Old Photos Back to Life (Wan et al., 2020) on the exported denoise_inpaint
# inputs, on CPU, in its own venv outside this repo (decisions D12).
#
# Runs BOPBtL's stage 1 only (scratch detection + global restoration), the same two
# commands its run.py issues with --with_scratch. Stages 2-4 (face detection/enhancement)
# need dlib and are skipped: run.py's final_output equals stage 1's output for faceless
# images, and our test crops are mostly scenes.
#
# Usage:  python scripts/export_bopbtl_inputs.py --limit 8 && bash scripts/run_bopbtl_local.sh
# Env:    AAI521_DATA_ROOT (else .env, like config.py), BOPBTL_HOME (~/bopbtl), BOPBTL_VENV (~/.venvs/bopbtl)
set -euo pipefail

# Data root resolved like restoration.config.data_root(): env, then the repo's .env, then data/.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${AAI521_DATA_ROOT:-}" ] && [ -f "$REPO/.env" ]; then
  AAI521_DATA_ROOT=$(sed -n 's/^[[:space:]]*AAI521_DATA_ROOT[[:space:]]*=[[:space:]]*//p' \
    "$REPO/.env" | tail -1 | tr -d "\"'\r")
fi
AAI521_DATA_ROOT="${AAI521_DATA_ROOT:-$REPO/data}"
AAI521_DATA_ROOT="${AAI521_DATA_ROOT/#\~/$HOME}"
BOPBTL_HOME="${BOPBTL_HOME:-$HOME/bopbtl}"
BOPBTL_VENV="${BOPBTL_VENV:-$HOME/.venvs/bopbtl}"
COMMIT=33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c # master, 2022-07-23
ROOT="$AAI521_DATA_ROOT/outputs/bopbtl"
IN="$ROOT/inputs"
WORK="$ROOT/work"
OUT="$ROOT/outputs"
n_in=$(find "$IN" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
[ "$n_in" -gt 0 ] || { echo "No inputs in $IN; run scripts/export_bopbtl_inputs.py" >&2; exit 1; }

# 1. Code, SyncBN dependency and stage-1 checkpoints (once).
[ -d "$BOPBTL_HOME/.git" ] ||
  git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git "$BOPBTL_HOME"
git -C "$BOPBTL_HOME" checkout -q "$COMMIT"
cd "$BOPBTL_HOME/Global"
if [ ! -d detection_models/sync_batchnorm ]; then
  tmp=$(mktemp -d)
  git clone -q https://github.com/vacancy/Synchronized-BatchNorm-PyTorch "$tmp"
  cp -r "$tmp/sync_batchnorm" detection_models/
  rm -rf "$tmp"
fi
if [ ! -f checkpoints/detection/FT_Epoch_latest.pt ]; then
  curl -fL -o global_checkpoints.zip \
    https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
  unzip -q global_checkpoints.zip && rm global_checkpoints.zip
fi

# 2. Venv: their requirements.txt minus dlib (faces) and PySimpleGUI (GUI).
if [ ! -x "$BOPBTL_VENV/bin/python" ]; then
  uv venv "$BOPBTL_VENV" --python 3.11
  uv pip install --python "$BOPBTL_VENV/bin/python" \
    torch==2.11.0 torchvision==0.26.0 numpy pillow opencv-python scikit-image easydict \
    PyYAML 'dominate>=2.3.1' dill tensorboardX scipy einops matplotlib
fi

# 3. Stage 1 on CPU. Their CUDA-saved checkpoints predate torch's weights_only=True default
#    and test.py loads them without map_location, so patch torch.load instead of their code.
SHIM='import functools, runpy, sys, torch
torch.load = functools.partial(torch.load, map_location="cpu", weights_only=False)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")'
PY="$BOPBTL_VENV/bin/python"
rm -rf "$WORK" "$OUT"
mkdir -p "$WORK" "$OUT"
start=$(date +%s)
"$PY" -c "$SHIM" detection.py --test_path "$IN" --output_dir "$WORK/masks" \
  --input_size full_size --GPU -1
"$PY" -c "$SHIM" test.py --Scratch_and_Quality_restore --test_input "$WORK/masks/input" \
  --test_mask "$WORK/masks/mask" --outputs_dir "$WORK" --gpu_ids -1
cp "$WORK"/restored_image/*.png "$OUT"/

# test.py skips images that raise, so check none went missing.
n_out=$(find "$OUT" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
secs=$(($(date +%s) - start))
echo "BOPBtL: $n_out/$n_in images in ${secs}s ($((secs / n_in)) s/image) -> $OUT"
[ "$n_out" -eq "$n_in" ] || { echo "Missing outputs; see the log above" >&2; exit 1; }
