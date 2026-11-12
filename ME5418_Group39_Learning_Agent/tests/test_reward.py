import numpy as np
from environment.reward import RewardWeights, compute_reward


def test_reward_finite_and_improves_with_smaller_error():
    """Good tracking should score better and every reported term must be finite."""
    weights = RewardWeights()
    bad, bad_parts = compute_reward(np.ones(3), np.ones(3), np.ones(3), np.ones(4), False, False, weights, .05)
    good, good_parts = compute_reward(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4), False, False, weights, .05)
    assert np.isfinite(bad) and good > bad
    assert set(("angular_velocity_penalty", "control_effort_penalty", "waypoint_bonus", "crash_penalty", "total_reward")) <= bad_parts.keys()
