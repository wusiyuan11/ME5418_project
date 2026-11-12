import numpy as np
import pybullet as p
from environment import AerialSprayingEnv, EnvironmentConfig
from environment.payload_model import PayloadModel


def test_payload_decreases_and_never_negative():
    """Spraying empties the tank but cannot create a negative liquid mass."""
    payload = PayloadModel(); payload.reset(0.4, 0.1)
    assert payload.update(1.0) < 0.4
    assert payload.update(100.0) == 0.0


def test_pybullet_mass_and_inertia_are_updated():
    """Payload changes must reach the physics engine, not only an info field."""
    env = AerialSprayingEnv(EnvironmentConfig(wind_mode="none")); env.reset(seed=8)
    requested_mass = env.config.dry_mass + 0.1
    env.dynamics.update_mass(requested_mass)
    dynamics_info = p.getDynamicsInfo(env.dynamics.body_id, -1,
                                      physicsClientId=env.dynamics.client)
    assert np.isclose(dynamics_info[0], requested_mass)
    assert np.all(np.asarray(dynamics_info[2]) > 0.0)
    env.close()


def test_thrust_and_tilt_affect_motion():
    """Low thrust should descend, while pitching should create forward motion."""
    env = AerialSprayingEnv(EnvironmentConfig(wind_mode="none")); env.reset(seed=4)
    initial_vz = env.dynamics.velocity[2]
    env.step(np.array([-1.0, 0, 0, 0], dtype=np.float32))
    assert env.dynamics.velocity[2] < initial_vz
    env.reset(seed=4)
    # The first PyBullet step builds angular velocity; subsequent steps tilt
    # the body and redirect part of the thrust horizontally.
    for _ in range(5):
        env.step(np.array([0.5, 0, 1.0, 0], dtype=np.float32))
    assert env.dynamics.velocity[0] > 0.0


def test_wind_changes_motion_and_payload_changes_response():
    """Wind must move the drone, and a lighter drone must accelerate more."""
    no_wind = AerialSprayingEnv(EnvironmentConfig(wind_mode="none")); no_wind.reset(seed=5)
    wind = AerialSprayingEnv(EnvironmentConfig(wind_mode="constant", max_wind_speed=4.0)); wind.reset(seed=5)
    action = np.zeros(4, dtype=np.float32)
    for _ in range(10): no_wind.step(action); wind.step(action)
    assert np.linalg.norm(wind.dynamics.position[:2] - no_wind.dynamics.position[:2]) > 1e-4
    env = AerialSprayingEnv(EnvironmentConfig(wind_mode="none")); env.reset(seed=6)
    env.payload.liquid_mass = 0.6
    env.dynamics.set_state(env.dynamics.position, velocity=np.zeros(3))
    env.step(np.array([0.4, 0, 0, 0])); heavy_vz = env.dynamics.velocity[2]
    env.reset(seed=6); env.payload.liquid_mass = 0.0
    env.dynamics.set_state(env.dynamics.position, velocity=np.zeros(3))
    env.step(np.array([0.4, 0, 0, 0])); light_vz = env.dynamics.velocity[2]
    assert light_vz > heavy_vz
