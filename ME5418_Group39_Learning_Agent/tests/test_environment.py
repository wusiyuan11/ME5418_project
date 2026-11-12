import subprocess
import sys
from pathlib import Path
import numpy as np
from gymnasium.utils.env_checker import check_env
from environment import AerialSprayingEnv, EnvironmentConfig


def test_spaces_reset_and_checker():
    """The public API must match the proposal and pass Gymnasium's own checker."""
    env = AerialSprayingEnv()
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=42)
    second, _ = env.reset(seed=42)
    assert first.shape == (15,) and first.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert np.all(np.isfinite(first)) and np.array_equal(first, second)
    assert info["physics_backend"] == "pybullet_3.2.7"


def test_pybullet_rgb_array_render():
    """The engine should provide a real headless camera image."""
    env = AerialSprayingEnv(render_mode="rgb_array")
    env.reset(seed=42)
    image = env.render()
    assert image.shape == (600, 800, 3) and image.dtype == np.uint8
    env.close()


def test_step_and_time_limit():
    """A short episode should return valid data and stop at its time limit."""
    config = EnvironmentConfig(max_episode_time=0.10, wind_mode="none")
    env = AerialSprayingEnv(config)
    env.reset(seed=1)
    result = env.step(np.zeros(4, dtype=np.float32))
    assert len(result) == 5 and np.isfinite(result[1])
    _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert not terminated and truncated and info["termination_reason"] == "time_limit"


def test_crash_and_boundary_terminate():
    """Unsafe ground contact and leaving the field must end the task."""
    env = AerialSprayingEnv(); env.reset(seed=2)
    env.dynamics.set_state(np.array([0.0, 0.0, 0.04]))
    assert env.step(np.zeros(4))[2]
    env.reset(seed=2); env.dynamics.set_state(np.array([100.0, 0.0, 2.0]))
    _, _, terminated, _, info = env.step(np.zeros(4))
    assert terminated and info["termination_reason"] == "boundary_violation"


def test_demos_generate_plots(tmp_path):
    """Both user-facing demos should run headlessly and create real images."""
    root = Path(__file__).parents[1]
    random_plot, pid_plot, error_plot = tmp_path/"random.png", tmp_path/"pid.png", tmp_path/"error.png"
    subprocess.run([sys.executable, "-m", "demo.demo_random", "--seed", "3", "--save-plot", str(random_plot)],
                   cwd=root, check=True, timeout=60)
    subprocess.run([sys.executable, "-m", "demo.demo_pid", "--seed", "3", "--save-plot", str(pid_plot),
                    "--error-plot", str(error_plot)], cwd=root, check=True, timeout=60)
    assert random_plot.stat().st_size > 1000
    assert pid_plot.stat().st_size > 1000 and error_plot.stat().st_size > 1000


def test_actual_ordered_waypoints_and_success(monkeypatch):
    """Reference progress, skipped points and hovering cannot earn arrivals."""
    env = AerialSprayingEnv(EnvironmentConfig(wind_mode="none", max_episode_time=100))
    try:
        env.reset(seed=7)
        assert env.waypoints_achieved == 1  # spawn counts without reset reward
        monkeypatch.setattr(env.dynamics, "step", lambda *args: None)
        env.time = env.trajectory.duration
        env.dynamics.set_state(env.trajectory.waypoints[-1])
        _, _, terminated, _, info = env.step(np.zeros(4))
        assert not terminated and info["waypoint_bonus"] == 0
        assert info["trajectory_progress"] == 1 and info["waypoints_achieved"] == 1
        for index in range(1, len(env.trajectory.waypoints)):
            env.dynamics.set_state(env.trajectory.waypoints[index])
            _, _, terminated, _, info = env.step(np.zeros(4))
            assert info["waypoint_bonus"] == 3
            assert info["waypoints_achieved"] == index + 1
            assert terminated == (index == len(env.trajectory.waypoints)-1)
            if not terminated:
                assert env.step(np.zeros(4))[4]["waypoint_bonus"] == 0
        assert info["is_success"] and info["termination_reason"] == "mission_complete"
        assert info["next_waypoint"] is None
        env.reset(seed=7)
        assert env.waypoints_achieved == 1
    finally:
        env.close()


def test_arrival_requires_3d_tolerance_and_reference_finish(monkeypatch):
    env = AerialSprayingEnv(EnvironmentConfig(wind_mode="none"))
    try:
        env.reset(seed=7)
        monkeypatch.setattr(env.dynamics, "step", lambda *args: None)
        target = env.trajectory.waypoints[1]
        env.dynamics.set_state(target + np.array([0, 0, .51]))
        assert env.step(np.zeros(4))[4]["waypoint_bonus"] == 0
        env.dynamics.set_state(target + np.array([0, 0, .49]))
        assert env.step(np.zeros(4))[4]["waypoint_bonus"] == 3
        for target in env.trajectory.waypoints[2:]:
            env.dynamics.set_state(target)
            _, _, terminated, _, info = env.step(np.zeros(4))
        assert not terminated and not info["is_success"]
        env.time = env.trajectory.duration
        env.dynamics.set_state(target + np.array([.51, 0, 0]))
        assert not env.step(np.zeros(4))[2]
        env.dynamics.set_state(target)
        assert env.step(np.zeros(4))[4]["is_success"]
    finally:
        env.close()
