"""Train a Terrain Grounded Imagination policy for the Unitree G1 (see README.md for the options).

    python scripts/train.py --log_dir logs/<run_name> [options]
"""

from tgi.training.config import parse_args


def main():
    cfg = parse_args()  # before torch and Isaac Lab are imported
    from tgi.training.trainer import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
