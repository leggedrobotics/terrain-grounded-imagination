#!/bin/bash
# Export a checkpoint and run the sim2sim courses (CPU MuJoCo); a full check adds the stand-still check and the
# base env's locomotion validation (Isaac Lab, PhysX). Training runs it in the background every 2k / 20k iterations.
# Usage: scripts/check.sh logs/<run>/ckpt_<it>.pt [repeats]
#   QUICK=1      the 12 courses that separate runs (about 30 s), no stand check, no Isaac evaluation
#   ISAAC_GPU=1  Isaac evaluation on the GPU with 520 envs (default: CPU PhysX, 260 envs, next to training)
set -e
cd "$(dirname "$0")/.."
PY=${TGI_PYTHON:-python}
CKPT=$(realpath "$1"); REP=${2:-2}
IT=$($PY -c "import re,sys; print(int(re.findall(r'ckpt_(\d+)', sys.argv[1])[0]))" "$CKPT")
DIR=$(dirname "$CKPT")
RUN=$DIR/export_$(printf %06d $IT)
CUDA_VISIBLE_DEVICES= $PY scripts/export.py --ckpt "$CKPT" --out "$RUN" 2>&1 | tail -1
[ -f "$RUN/model_$IT.pt" ] || { echo "  export failed: $RUN/model_$IT.pt missing"; exit 1; }
if [ ! -f "$RUN/exported/deploy.json" ]; then  # --scan full or --fpm 0: no foot model, which sim2sim and the stand check need
  echo "  no foot model (--scan full or --fpm 0): sim2sim courses and stand check skipped"
  [ "${QUICK:-0}" = 1 ] && exit 0
  SCAN=$($PY -c "import json, sys; print(json.load(open(sys.argv[1])).get('scan', 'foothold'))" "$DIR/args.json")
  if [ "$SCAN" = full ]; then export FPM_POLICY_SCAN=full; fi
else
  if [ "${QUICK:-0}" = 1 ]; then
    TERRAINS="stairs:0.15 stairs:0.20 stairs:0.25 blocks:0.22 stones:0.20 stones:0.17:0.28 fpmstones:0.6 fpmstones:0.9 pit:0.30 pit:0.40 stones:0.20:0.20 noise:0.20:0.2"; OUT=$DIR/quick_$IT.json
  else
    TERRAINS="flat drop:0.10 drop:0.15 drop:0.20 stairs stairs:0.15 stairs:0.20 stairs:0.25 blocks blocks:0.22 stones:0.10 stones:0.20 stones:0.10:0.35 stones:0.17:0.28 stones:0.25:0.20 fpmstones:0.3 fpmstones:0.6 fpmstones:0.9 fpmstones:1.0 pit:0.10 pit:0.20 pit:0.30 pit:0.40 stones:0.20:0.20 noise:0.15:0.1 noise:0.20:0.2"
    OUT=$DIR/check_$IT.json
  fi
  CUDA_VISIBLE_DEVICES= $PY scripts/sim2sim.py --export_dir "$RUN" --terrains $TERRAINS --repeats $REP --workers 12 --out "$OUT" > /dev/null 2>&1
  $PY -c "
import json, sys
s = json.load(open(sys.argv[1]))['summary']
for t, v in s.items():
    print(f'  {t:12s} crossed {v[\"crossed\"]:.2f} surv {v[\"survival\"]:.2f} lin_err {v[\"lin_err\"]:.3f} waist_yaw {v[\"waist_yaw_mean\"]:+.3f}')
" "$OUT"
  [ "${QUICK:-0}" = 1 ] && exit 0
  CUDA_VISIBLE_DEVICES= $PY scripts/stand_check.py --export_dir "$RUN" 2>/dev/null | tail -1
fi
echo "  base G1 env (Isaac Lab 3, PhysX), success % / falls %:"
if [ "${ISAAC_GPU:-0}" = 1 ]; then bash scripts/isaac_eval.sh "$RUN/model_$IT.pt" 520 cuda:0 | sed 's/^/  /'
else bash scripts/isaac_eval.sh "$RUN/model_$IT.pt" 260 cpu | sed 's/^/  /'; fi
