#!/bin/bash
# Locomotion validation of an exported checkpoint in the base G1 env (Watch Your Step: 13 terrains at difficulty 0.5,
# Isaac Lab 3, PhysX), with fpm/scripts/validate_locomotion.py. A quick check: 1100 steps with the startup randomization on,
# so not comparable with the README's results, which use the full protocol (README, Evaluation).
# Usage: scripts/isaac_eval.sh logs/<run>/export_<it>/model_<it>.pt [num_envs] [device]
#   device cpu (default): CPU PhysX, about 0.25 GB of GPU memory next to training (260 envs, about 3 min); cuda:0 for big runs
PY=${TGI_PYTHON:-python}
CK=$(realpath "$1"); N=${2:-260}; DEV=${3:-cpu}
FPM_REPO=$($PY -c "import fpm, os; print(os.path.dirname(os.path.dirname(os.path.dirname(fpm.__file__))))") || exit 1  # fpm project root
cd "$FPM_REPO" || exit 1
[ "$DEV" = cpu ] && export FPM_PHYSX_SMALL=1
OMNI_KIT_ACCEPT_EULA=YES PYTHONHASHSEED=0 $PY scripts/validate_locomotion.py --headless \
  --device "$DEV" --num_envs "$N" --num_steps 1100 --checkpoint "$CK" 2>&1 | grep -A19 "^terrain "
