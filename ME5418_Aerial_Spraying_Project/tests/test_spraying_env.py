"""Automated tests for the simplified aerial-spraying environment."""

# Import NumPy for numerical comparisons and test actions.
import numpy as np
# Import the environment and its reference controller.
from spraying_env import SprayingEnv, waypoint_controller


# Verify that reset restores the initial position, velocity, mass, and target.
def test_environment_reset():
    # Create and disturb a deterministic environment.
    env = SprayingEnv(seed=7)
    env.step(np.ones(2, dtype=np.float32))
    # Reset to the initial episode state.
    observation, info = env.reset(seed=7)
    # Check all important reset invariants.
    assert np.allclose(env.pos, [1.0, 2.0])
    assert np.allclose(env.vel, [0.0, 0.0])
    assert env.mass == env.dry_mass + env.initial_water
    assert env.target_idx == 1
    assert observation.shape == (9,)


# Verify that one step advances time and returns Gymnasium's data types.
def test_environment_step():
    # Create a wind-free environment for a controlled transition.
    env = SprayingEnv(wind_max=0.0, seed=1)
    # Apply a positive x acceleration.
    observation, reward, terminated, truncated, info = env.step([1.0, 0.0])
    # Confirm motion and return-value contracts.
    assert env.step_count == 1
    assert env.pos[0] > 1.0
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert observation.dtype == np.float32


# Verify the declared and actual observation shape.
def test_observation_shape():
    # Reset a default environment.
    env = SprayingEnv(seed=2)
    observation, info = env.reset(seed=2)
    # Require a valid nine-component observation.
    assert observation.shape == (9,)
    assert env.observation_space.contains(observation)


# Verify action limits and shape validation.
def test_action_limits():
    # Compare an oversized action with its explicitly clipped equivalent.
    env_large = SprayingEnv(wind_max=0.0, seed=3)
    env_clipped = SprayingEnv(wind_max=0.0, seed=3)
    env_large.step([5.0, -5.0])
    env_clipped.step([1.0, -1.0])
    # The two resulting states must match exactly within numerical tolerance.
    assert np.allclose(env_large.pos, env_clipped.pos)
    assert np.allclose(env_large.vel, env_clipped.vel)


# Verify wind range, vector construction, and direction sampling.
def test_wind_sampling():
    # Check multiple independent seeded episodes.
    for seed in range(30):
        # Sample one episode wind.
        env = SprayingEnv(wind_max=3.0, seed=seed)
        # Verify magnitude bounds and the polar-to-Cartesian equation.
        assert 0.0 <= env.wind_speed < 3.0
        assert 0.0 <= env.wind_angle < 2.0 * np.pi
        assert np.allclose(env.wind, env.wind_speed * np.array([np.cos(env.wind_angle), np.sin(env.wind_angle)]))


# Verify exact payload reduction, dry-mass floor, and depletion stop.
def test_mass_reduction():
    # Use a small payload and fast spray rate for a short depletion test.
    env = SprayingEnv(wind_max=0.0, initial_water=0.02, spray_rate=0.1, seed=4)
    # Advance beyond the liquid depletion time.
    for unused_step in range(10):
        # Use zero action so only the mass equation is under test.
        env.step([0.0, 0.0])
    # Require liquid depletion without crossing the empty mass.
    assert env.water == 0.0
    assert env.mass == env.dry_mass
    # Advance again to prove mass stops decreasing.
    env.step([0.0, 0.0])
    assert env.mass == env.dry_mass


# Verify the exact ten-point alternating lawnmower geometry.
def test_waypoint_generation():
    # Construct a default route.
    env = SprayingEnv(seed=5)
    # Require ten points with expected endpoints and alternating directions.
    assert env.waypoints.shape == (10, 2)
    assert np.allclose(env.waypoints[0], [1.0, 2.0])
    assert np.allclose(env.waypoints[1], [19.0, 2.0])
    assert np.allclose(env.waypoints[2], [19.0, 6.0])
    assert np.allclose(env.waypoints[-1], [19.0, 18.0])


# Verify that reaching the final waypoint terminates successfully.
def test_mission_completion():
    # Create a wind-free controlled state.
    env = SprayingEnv(wind_max=0.0, seed=6)
    # Place the drone at the final target with no velocity.
    env.target_idx = len(env.waypoints) - 1
    env.pos = env.waypoints[-1].copy()
    env.vel = np.zeros(2)
    # Advance once to evaluate the success condition.
    observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
    # Require natural successful termination.
    assert terminated is True
    assert truncated is False
    assert info["success"] is True


# Verify the documented Euclidean active-waypoint error calculation.
def test_tracking_error_calculation():
    # Place a wind-free drone three-four-five metres from its target.
    env = SprayingEnv(wind_max=0.0, seed=7)
    env.pos = env.waypoints[env.target_idx] - np.array([3.0, 4.0])
    # Require a five-metre Euclidean error.
    assert np.isclose(env.tracking_error, 5.0)


# Verify that equal seeds produce equal wind and trajectories.
def test_reproducibility_using_random_seeds():
    # Construct two identical seeded environments.
    first = SprayingEnv(seed=11)
    second = SprayingEnv(seed=11)
    # Advance both with independently computed but deterministic controller actions.
    for unused_step in range(50):
        # Apply corresponding reference-controller commands.
        first.step(waypoint_controller(first))
        second.step(waypoint_controller(second))
    # Require identical disturbance and state histories.
    assert np.allclose(first.wind, second.wind)
    assert np.allclose(first.history, second.history)


# Verify that wind changes motion and lower mass increases wind-force acceleration.
def test_wind_and_mass_affect_dynamics():
    # Construct equal fixed-wind cases with heavy and light payloads.
    heavy = SprayingEnv(wind_max=3.0, initial_water=0.8, wind_angle=0.0, seed=9)
    light = SprayingEnv(wind_max=3.0, initial_water=0.0, wind_angle=0.0, seed=9)
    calm = SprayingEnv(wind_max=0.0, initial_water=0.8, wind_angle=0.0, seed=9)
    # Apply zero command for one step in each case.
    heavy.step([0.0, 0.0])
    light.step([0.0, 0.0])
    calm.step([0.0, 0.0])
    # Require wind-induced motion and the expected inverse-mass effect.
    assert heavy.pos[0] > calm.pos[0]
    assert light.pos[0] > heavy.pos[0]
