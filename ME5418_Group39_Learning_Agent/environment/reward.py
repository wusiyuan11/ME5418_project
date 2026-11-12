"""Calculate the feedback signal used by a future reinforcement-learning agent."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardWeights:
    """Keep the reward weights together so experiments can change them safely."""
    position: float = 1.0
    velocity: float = 0.20
    angular_velocity: float = 0.05
    control_effort: float = 0.02
    waypoint_bonus: float = 3.0
    crash_penalty: float = 100.0


def compute_reward(position_error: np.ndarray, velocity_error: np.ndarray,
                   angular_velocity: np.ndarray, normalized_action: np.ndarray,
                   waypoint_reached: bool, crashed: bool,
                   weights: RewardWeights, dt: float) -> tuple[float, dict]:
    """Combine tracking quality, stability, effort and task events.

    The individual terms are returned as well as the total.  This makes it much
    easier to understand why an episode received a particular reward.
    """
    position_term = -weights.position * np.linalg.norm(position_error) * dt
    velocity_term = -weights.velocity * np.linalg.norm(velocity_error) * dt
    angular_term = -weights.angular_velocity * np.linalg.norm(angular_velocity) * dt
    # The first action is a change around hover rather than absolute thrust.
    # Consequently, holding the drone up does not receive an artificial penalty.
    control_term = -weights.control_effort * float(np.dot(normalized_action, normalized_action)) * dt
    waypoint_term = weights.waypoint_bonus if waypoint_reached else 0.0
    crash_term = -weights.crash_penalty if crashed else 0.0
    total = position_term + velocity_term + angular_term + control_term + waypoint_term + crash_term
    return float(total), {
        "position_penalty": float(position_term),
        "velocity_penalty": float(velocity_term),
        "angular_velocity_penalty": float(angular_term),
        "control_effort_penalty": float(control_term),
        "waypoint_bonus": float(waypoint_term),
        "crash_penalty": float(crash_term),
        "total_reward": float(total),
    }
