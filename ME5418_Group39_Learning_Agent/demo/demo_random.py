"""Show that the environment runs end to end without a trained controller."""

import argparse
from pathlib import Path
import numpy as np
from environment import AerialSprayingEnv
from visualization.plot_trajectory import save_episode_plots


def run(seed: int = 42, render: bool = False, save_plot: str | None = None) -> dict:
    """Run random actions until the episode ends and return a short summary."""
    env = AerialSprayingEnv(render_mode="human" if render else None)
    _, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
    if save_plot:
        save_episode_plots(env.history, env.trajectory.waypoints, save_plot,
                           title="Random-policy trajectory")
    stats = {"steps": env.step_count, "reward": env.episode_reward,
             "final_error": info["tracking_error_norm"], "reason": info["termination_reason"]}
    print("Random policy:", stats)
    env.close()
    return stats


def main() -> None:
    """Read command-line options and launch the random-policy demonstration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-plot", nargs="?", const="outputs/random_trajectory.png")
    args = parser.parse_args()
    run(args.seed, args.render, args.save_plot)


if __name__ == "__main__":
    main()
