# FPM: foothold prediction model and G1 locomotion environments

The foothold prediction model (FPM) and the Unitree G1 locomotion environments of [Watch Your Step](https://leggedrobotics.github.io/watch-your-step/) ([code](https://github.com/leggedrobotics/foothold-prediction-model), [arXiv](https://arxiv.org/abs/2610.29612)), as an Isaac Lab 3 task package. The FPM predicts the next two footholds from proprioception and the velocity command, and the policy observes the elevation map only in 7 x 7 patches around them. TGI (`../tgi`) trains its policies in these environments and exports them in this package's formats.

## Installation

Tested with Isaac Lab 3.0.0 (Isaac Sim 6.1, PhysX). In Isaac Lab's Python environment:

```bash
pip install -e fpm --no-deps
```

Importing `fpm.tasks` registers the tasks:

| Task | Use |
|---|---|
| `FPM-G1-v0` | PPO training |
| `FPM-G1-Play-v0` | playback |
| `FPM-G1-Validate-Locomotion-v0` | 13 terrains at difficulty 0.5 |
| `FPM-G1-Validate-Corridor-v0` | 13 sections in each of three walled corridors |
| `FPM-G1-Validate-FootPred-v0` | FPM prediction error |

## Structure

```
src/fpm/
  tasks/locomotion/
    mdp/              foothold-gated height scan, rewards, events, commands, symmetry augmentation
    config/g1/        environment configurations, task registration, PPO configuration (agents/)
  terrains/           training, locomotion validation and corridor validation terrains
  foot_pred/          FPM network, foothold labels, refitting and training schedule
  assets/             Unitree G1 (29 DoF) USD and articulation configuration
  vecenv.py           rsl-rl wrapper for finite-horizon validation
scripts/              train.py, play.py, validate_locomotion.py, validate_corridor.py, validate_foot_pred.py
demo/                 pretrained policy and FPM
```

## Training

```bash
python scripts/train.py --headless --seed 1
```

For the first 500 iterations the footholds are fixed under the base. After that, the FPM is retrained from scratch every 500 iterations on 5 x 10^5 transitions of the current policy. FPM checkpoints are saved next to the policy checkpoints as `foot_model_<iteration>.pt` (TorchScript, input normalization included). To resume a run, pass `--resume <run>/model_<iteration>.pt`.

## Playback and export

```bash
python scripts/play.py --show_footholds
```

Without `--checkpoint`, the latest checkpoint in `logs/rsl_rl/g1_fpm/` is played, or the pretrained policy in `demo/` if there are no logs. Training uses the rsl-rl that ships with Isaac Lab 3; playback and the validation scripts also load checkpoints in the rsl-rl 3.x layout of the original release, the demo and TGI exports. Writes `policy.pt`, `policy.onnx` and `foot_model.onnx` to `exported/` next to the checkpoint. Use `--joystick` for gamepad control and `--video` to record a clip.

## Evaluation

Each script loads the policy checkpoint and the latest `foot_model_*.pt` from the same directory. Set `PYTHONHASHSEED=0` so that the corridor terrain is identical across runs, and add `--save_results` to save the per-episode results in the run directory.

```bash
export PYTHONHASHSEED=0
python scripts/validate_locomotion.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_corridor.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_foot_pred.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
```

With many environments, raise PhysX's pair buffers (`env.sim.physics.gpu_found_lost_pairs_capacity=16777216 env.sim.physics.gpu_found_lost_aggregate_pairs_capacity=67108864`), otherwise missed contacts show up as falls. TGI policies trained with `--scan full` read the full height scan; set `FPM_POLICY_SCAN=full` to evaluate them.

## Citation

```bibtex
@misc{heyrman2026watchstepanticipatedperception,
  title         = {Watch Your Step: Anticipated Perception for Humanoid Locomotion},
  author        = {Matthias Heyrman and Chenhao Li and Ioannis Dadiotis and Marco Hutter},
  year          = {2026},
  eprint        = {2610.29612},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.29612}
}
```

## License

© 2026 ETH Zurich<br>
Created by: Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, and Marco Hutter<br>
Licensed under the MIT License (see [LICENSE](../LICENSE)). The G1 USD in `src/fpm/assets/data/G1/` is from [unitree_ros](https://github.com/unitreerobotics/unitree_ros), BSD-3-Clause (see [licenses/dependencies](../licenses/dependencies)).
