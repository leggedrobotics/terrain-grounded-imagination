import os

import gymnasium as gym
import torch

import fpm.tasks  # noqa: F401  (registers the tasks)
from fpm.foot_pred import latest_foot_model
from fpm.vecenv import FiniteRslRlVecEnvWrapper


def make_env_and_policy(task: str, env_cfg, agent_cfg, checkpoint: str, device: str | None = None):
    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    env_cfg.log_dir = run_dir
    env_cfg.load_run_dir = run_dir
    if device is not None:
        env_cfg.sim.device = device
        agent_cfg.device = device
    print(f"[INFO] policy {checkpoint}\n[INFO] FPM {latest_foot_model(run_dir)}")
    env = FiniteRslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    return env, load_actor(checkpoint, env.unwrapped.device), run_dir


def is_legacy_checkpoint(checkpoint: str) -> bool:
    """True for an rsl-rl 3.x checkpoint (the FPM release, the demo, TGI exports)."""
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False).get("model_state_dict", {})
    return any(k.startswith("actor.") for k in sd)


def load_actor_module(checkpoint: str, device) -> torch.nn.Sequential:
    """Deterministic actor (ELU MLP, no normalizer) of an rsl-rl 3.x checkpoint, independent of the installed rsl-rl."""
    sd = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})
    layers = []
    for j, i in enumerate(idx):
        w = sd[f"actor.{i}.weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"actor.{i}.bias"])
        layers += [lin] if j == len(idx) - 1 else [lin, torch.nn.ELU()]
    return torch.nn.Sequential(*layers).eval()


def load_actor(checkpoint: str, device) -> callable:
    actor = load_actor_module(checkpoint, device)
    return lambda obs: actor(obs["policy"])


def terrain_column(env) -> torch.Tensor:
    u = env.unwrapped
    centers = u.scene.terrain.terrain_origins.mean(dim=0)[:, :2].to(u.device)
    return torch.cdist(u.scene["robot"].data.root_pos_w.torch[:, :2], centers).argmin(dim=-1)


def hydra_task_config(task: str, agent_entry_point: str):
    """Isaac Lab 2.x ``hydra_task_config`` on top of 3.0's ``resolve_task_config``."""
    from isaaclab_tasks.utils.hydra import resolve_task_config

    def deco(fn):
        def wrapper():
            import sys

            overrides = [a for a in sys.argv[1:] if not a.startswith("-")]  # AppLauncher flags are not Hydra's
            env_cfg, agent_cfg = resolve_task_config(task, agent_entry_point, overrides=overrides)
            return fn(env_cfg, agent_cfg)

        return wrapper

    return deco
