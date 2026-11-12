"""Focused regressions for the third milestone's waypoint-aware observation."""
import numpy as np
import torch
from environment import AerialSprayingEnv
from pure_rl.agent import PurePPO, PPOConfig
from pure_rl.task import mission_observation


def test_same_motion_different_visit_history_produces_different_mission_inputs():
    env = AerialSprayingEnv()
    try:
        env.reset(seed=7)
        # Counterfactual histories at the same physical/timed state: the vehicle
        # is at C, but either missed B or previously visited B. This constructs
        # an observation-contract test, not an achieved test mission.
        point_c = env.trajectory.waypoints[2]
        env.time = env.trajectory.cumulative_lengths[2] / env.config.reference_speed
        env.reference_position, env.reference_velocity, env.current_waypoint, env.progress = \
            env.trajectory.reference(env.time)
        env.dynamics.set_state(point_c)
        raw = env._observation()
        env.waypoints_achieved = 1
        missed_b = mission_observation(raw, env, 'goal')
        env.waypoints_achieved = 2
        visited_b = mission_observation(raw, env, 'goal')
        np.testing.assert_array_equal(missed_b[:18], visited_b[:18])
        np.testing.assert_allclose(missed_b[18:21], [0., -2., 0.])
        np.testing.assert_allclose(visited_b[18:21], [0., 0., 0.])
        assert np.isclose(missed_b[21], .1)
        assert np.isclose(visited_b[21], .2)
        assert missed_b[22] == visited_b[22]
    finally:
        env.close()


def test_moving_reference_does_not_erase_an_outstanding_waypoint():
    env = AerialSprayingEnv()
    try:
        env.reset(seed=7)
        env.dynamics.set_state(env.trajectory.waypoints[2])
        env.waypoints_achieved = 1  # A counted; B still owed.
        for time in (8., 20.):
            env.time = time
            env.reference_position, env.reference_velocity, env.current_waypoint, env.progress = \
                env.trajectory.reference(time)
            assert env.current_waypoint > env.waypoints_achieved
            assert not env._check_waypoint_arrival()
            feature = mission_observation(env._observation(), env, 'goal')
            np.testing.assert_allclose(feature[18:21], [0., -2., 0.])
            assert np.isclose(feature[21], .1)
    finally:
        env.close()


def test_actor_action_is_independent_of_critic_output():
    agent = PurePPO(PPOConfig(features='goal', action_repeat=2))
    observation = np.linspace(-.2, .3, 23, dtype=np.float32)
    before = agent.act(observation)
    with torch.no_grad():
        old_value = agent.model(torch.as_tensor(observation))[2].item()
        # Change the value branch only. Actor inputs and parameters are intact.
        for parameter in agent.model.critic.parameters():
            parameter.fill_(.123)
        new_value = agent.model(torch.as_tensor(observation))[2].item()
    assert abs(new_value - old_value) > 1e-4
    np.testing.assert_array_equal(before, agent.act(observation))
