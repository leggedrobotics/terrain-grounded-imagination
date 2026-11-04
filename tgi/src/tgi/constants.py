"""Constants shared by the training simulator, imagination and validation."""

import os

NUM_J = 29
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J  # p, quat(wxyz), v_w, w_w, q, qd
# foot friction, torso com offset(3), encoder bias(29), joint friction loss, joint damping, armature scale,
# actuation lag (physics steps), torso added mass
XI_DIM = 1 + 3 + NUM_J + 5
MAX_LAG = 4  # physics steps of 5 ms -> up to 20 ms actuation latency

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# G1 MJCF for the differentiable kinematics (mjlab's G1 without meshes)
G1_MJCF = os.path.join(ASSETS, "g1_kinematics.xml")
# Newton-compatible copy of the base env's G1 USD (scripts/convert_g1_usd.py)
G1_NEWTON_USD = os.path.join(ASSETS, "g1_newton.usd")
# Unitree's official G1 (29 DoF) for sim2sim validation
UNITREE_SCENE = os.path.join(ASSETS, "unitree_g1", "scene_29dof.xml")
UNITREE_G1 = os.path.join(ASSETS, "unitree_g1", "g1_29dof.xml")

# per-joint std of the pose reward (mjlab's G1 velocity task "pose" term)
POSE_STD = {
    "std_standing": {".*": 0.05},
    "std_walking": {".*hip_pitch.*": 0.3, ".*hip_roll.*": 0.15, ".*hip_yaw.*": 0.15, ".*knee.*": 0.35,
                    ".*ankle_pitch.*": 0.25, ".*ankle_roll.*": 0.1, ".*waist_yaw.*": 0.2, ".*waist_roll.*": 0.08,
                    ".*waist_pitch.*": 0.1, ".*shoulder_pitch.*": 0.15, ".*shoulder_roll.*": 0.15,
                    ".*shoulder_yaw.*": 0.1, ".*elbow.*": 0.15, ".*wrist.*": 0.3},
    "std_running": {".*hip_pitch.*": 0.5, ".*hip_roll.*": 0.2, ".*hip_yaw.*": 0.2, ".*knee.*": 0.6,
                    ".*ankle_pitch.*": 0.35, ".*ankle_roll.*": 0.15, ".*waist_yaw.*": 0.3, ".*waist_roll.*": 0.08,
                    ".*waist_pitch.*": 0.2, ".*shoulder_pitch.*": 0.5, ".*shoulder_roll.*": 0.2,
                    ".*shoulder_yaw.*": 0.15, ".*elbow.*": 0.35, ".*wrist.*": 0.3},
}  # fmt: skip


def match_values(patterns: dict[str, float], names: list[str]) -> list[float]:
    """Value of the first regex in ``patterns`` that fully matches each name."""
    import re

    out = []
    for n in names:
        v = [val for pat, val in patterns.items() if re.fullmatch(pat, n)]
        assert v, n
        out.append(v[0])
    return out
