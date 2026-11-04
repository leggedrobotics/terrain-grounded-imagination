#!/bin/bash
# End-to-end pipeline, from scratch: world model + FPM + policy training in imagination (the defaults of
# scripts/train.py are the full recipe), background course checks every 2k / 20k iterations, then the final export,
# final evaluation (sim2sim courses + base env validation in Isaac Lab) and videos.
#
# Usage: scripts/pipeline.sh <run_name> [extra scripts/train.py args]
#   RESUME=1 scripts/pipeline.sh <run_name> [same extra args]   only to recover from a crash of the same run (continues
#   from last.pt; pass the original extra args again, a resume does not read them from the checkpoint)
set -e
cd "$(dirname "$0")/.."
NAME=$1; shift
DIR=logs/$NAME
if [ -d "$DIR" ] && [ "${RESUME:-0}" != 1 ]; then echo "$DIR exists; the pipeline runs from scratch (RESUME=1 only after a crash)"; exit 1; fi
mkdir -p "$DIR"
EXTRA=(); [ "${RESUME:-0}" = 1 ] && EXTRA=(--resume "$DIR/last.pt")
echo $$ > "$DIR/pipeline.pid"
PY=${TGI_PYTHON:-python}
export TGI_PYTHON=$PY

# 1. training (GPU)
OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/train.py --log_dir "$DIR" "${EXTRA[@]}" "$@" >> "$DIR/raw.log" 2>&1

# 2. final export + evaluation (sim2sim on CPU MuJoCo with Unitree's G1 MJCF, Isaac Lab on the GPU)
CKPT=$(ls "$DIR"/ckpt_*.pt | sort | tail -1)
IT=$($PY -c "import re,sys; print(int(re.findall(r'ckpt_(\d+)', sys.argv[1])[0]))" "$CKPT")
RUN=$DIR/export_$(printf %06d "$IT")
ISAAC_GPU=1 bash scripts/check.sh "$CKPT" 4 > "$DIR/final_eval.txt" 2>&1
for t in stairs:0.20 fpmstones:0.6 fpmstones:1.0; do
  CUDA_VISIBLE_DEVICES= $PY scripts/sim2sim.py --export_dir "$RUN" --terrains "$t" --seconds 12 --workers 1 \
    --video "$DIR/final_${t//[:.]/}.mp4" > /dev/null 2>&1 || true
done
echo "pipeline done: $RUN (play it from fpm/ with: python scripts/play.py --checkpoint $(realpath $RUN)/model_$IT.pt)" | tee -a "$DIR/final_eval.txt"
