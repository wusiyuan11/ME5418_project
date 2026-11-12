"""A straightforward fixed-gain controller for checking the environment."""

import argparse
import numpy as np
from environment import AerialSprayingEnv
from visualization.plot_trajectory import save_episode_plots


class FixedGainController:
    """Track the reference using only state information available to a policy.

    The gains are intentionally fixed.  This controller is a useful baseline
    and debugging tool, not a claim of optimal PID tuning.
    """

    def __init__(self, dt: float):
        """Store the update period and the manually selected baseline gains."""
        self.dt = dt
        self.integral_z = 0.0
        self.kp_xy, self.kd_xy = 0.32, 0.35
        self.kp_z, self.kd_z, self.ki_z = 0.55, 0.35, 0.06

    def action(self, observation: np.ndarray, reference_velocity: np.ndarray) -> np.ndarray:
        """Convert position and velocity errors into the environment's four actions."""
        velocity = observation[3:6]
        error = observation[12:15]
        velocity_error = reference_velocity - velocity
        self.integral_z = np.clip(self.integral_z + error[2] * self.dt, -2.0, 2.0)
        # With the sign convention in our dynamics, nose-up pitch moves along
        # +x, while a negative roll command moves along +y.
        pitch = self.kp_xy * error[0] + self.kd_xy * velocity_error[0]
        roll = -(self.kp_xy * error[1] + self.kd_xy * velocity_error[1])
        thrust = self.kp_z * error[2] + self.kd_z * velocity_error[2] + self.ki_z * self.integral_z
        return np.clip(np.array([thrust, roll, pitch, 0.0], dtype=np.float32), -1.0, 1.0)


def run(seed: int = 7, render: bool = False, save_plot: str | None = None,
        error_plot: str | None = None, animation: str | None = None) -> dict:
    """Run one baseline-controlled episode and report tracking and effort."""
    env = AerialSprayingEnv(render_mode="human" if render else None)
    observation, _ = env.reset(seed=seed)
    controller = FixedGainController(env.config.dt)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = controller.action(observation, env.reference_velocity)
        observation, _, terminated, truncated, info = env.step(action)
    if save_plot:
        save_episode_plots(env.history, env.trajectory.waypoints, save_plot, error_plot,
                           "Fixed-gain controller trajectory")
    if animation:
        from visualization.animate_episode import save_episode_animation
        save_episode_animation(env.history, env.trajectory.waypoints, animation)
    effort = float(np.mean(np.sum(np.asarray(env.history["action"]) ** 2, axis=1)))
    stats = {"steps": env.step_count, "reward": env.episode_reward,
             "mean_error": float(np.mean(env.history["error"])),
             "final_error": info["tracking_error_norm"], "mean_control_effort": effort,
             "waypoints_achieved": info["waypoints_achieved"],
             "total_waypoints": len(env.trajectory.waypoints),
             "reason": info["termination_reason"]}
    print("Fixed-gain controller:", stats)
    env.close()
    return stats


def main() -> None:
    """Read command-line options and launch the fixed-gain demonstration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-plot", nargs="?", const="outputs/pid_trajectory.png")
    parser.add_argument("--error-plot", default="outputs/tracking_error.png")
    parser.add_argument("--animation", help="Save telemetry replay as a GIF")
    args = parser.parse_args()
    run(args.seed, args.render, args.save_plot, args.error_plot if args.save_plot else None, args.animation)


if __name__ == "__main__":
    main()
