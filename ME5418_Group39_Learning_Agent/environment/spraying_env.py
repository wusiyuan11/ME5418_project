"""Bring the trajectory, disturbances and drone model together as a Gymnasium task."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .drone_dynamics import DroneDynamics
from .payload_model import PayloadModel
from .reward import RewardWeights, compute_reward
from .trajectory import LawnMowerTrajectory
from .wind_model import WindModel


@dataclass
class EnvironmentConfig:
    """Settings that define the field, vehicle limits and episode conditions."""
    dt: float = 0.05
    max_episode_time: float = 45.0
    field_width: float = 8.0
    field_length: float = 12.0
    lane_spacing: float = 2.0
    altitude: float = 2.0
    reference_speed: float = 2.0
    waypoint_tolerance: float = 0.5
    boundary_margin: float = 2.0
    dry_mass: float = 1.5
    liquid_mass_range: tuple[float, float] = (0.35, 0.65)
    mass_flow_range: tuple[float, float] = (0.008, 0.014)
    wind_mode: str = "gusts"
    max_wind_speed: float = 3.0
    max_tilt_radians: float = np.deg2rad(25.0)
    max_yaw_rate: float = np.deg2rad(90.0)
    thrust_deviation_fraction: float = 0.60
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


class AerialSprayingEnv(gym.Env):
    """A seeded aerial-spraying task with 15 observations and four actions.

    A zero action asks for nominal hover at the midpoint configured mass.  The
    first action changes that thrust by up to 60 percent; the other three ask
    for roll, pitch and yaw rate within the documented vehicle limits.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: EnvironmentConfig | None = None, render_mode: str | None = None):
        """Construct the reusable models and declare the public Gymnasium spaces."""
        super().__init__()
        self.config = config or EnvironmentConfig()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.trajectory = LawnMowerTrajectory(
            self.config.field_width, self.config.field_length, self.config.lane_spacing,
            self.config.altitude, self.config.reference_speed, self.config.waypoint_tolerance)
        self.dynamics = DroneDynamics(dt=self.config.dt, gui=render_mode == "human")
        self.payload = PayloadModel(dry_mass=self.config.dry_mass)
        self.wind = WindModel(self.config.wind_mode, self.config.max_wind_speed)
        mean_liquid = sum(self.config.liquid_mass_range) / 2.0
        self.nominal_hover_thrust = (self.config.dry_mass + mean_liquid) * self.dynamics.gravity
        self.max_steps = int(np.ceil(self.config.max_episode_time / self.config.dt))
        self._figure = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a fresh, reproducible spraying episode.

        The path stays fixed, while the starting offset, wind, tank fill and
        spray rate are sampled again.  Passing the same seed reproduces all of
        these choices.
        """
        super().reset(seed=seed)
        self.time, self.step_count, self.episode_reward = 0.0, 0, 0.0
        start = self.trajectory.waypoints[0] + self.np_random.uniform(-0.08, 0.08, size=3)
        start[2] = max(start[2], 0.5)
        self.payload.reset(self.np_random.uniform(*self.config.liquid_mass_range),
                           self.np_random.uniform(*self.config.mass_flow_range))
        self.dynamics.reset(start, self.payload.total_mass)
        self.wind.reset(self.np_random)
        self.reference_position, self.reference_velocity, self.current_waypoint, self.progress = \
            self.trajectory.reference(self.time)
        self.waypoints_achieved = 0
        self._check_waypoint_arrival()
        self.history = {key: [] for key in ("time", "position", "reference", "error", "action", "reward",
            "attitude", "velocity", "wind", "payload", "command", "waypoints_achieved")}
        observation = self._observation()
        return observation, self._info("running", {})

    def _check_waypoint_arrival(self) -> bool:
        """Record at most one ordered arrival at the current sampled position."""
        index = self.waypoints_achieved
        if index < len(self.trajectory.waypoints) and np.linalg.norm(
                self.dynamics.position - self.trajectory.waypoints[index]) <= self.config.waypoint_tolerance:
            self.waypoints_achieved += 1
            return True
        return False

    def _physical_action(self, action: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Translate normalized policy outputs into thrust, angles and yaw rate."""
        thrust = self.nominal_hover_thrust * (1.0 + self.config.thrust_deviation_fraction * action[0])
        attitude = np.array([action[1], action[2], 0.0]) * self.config.max_tilt_radians
        yaw_rate = action[3] * self.config.max_yaw_rate
        return float(max(0.0, thrust)), attitude, float(yaw_rate)

    def _observation(self) -> np.ndarray:
        """Assemble the exact 15-value observation promised by the interface."""
        error = self.reference_position - self.dynamics.position
        observation = np.concatenate((self.dynamics.state_vector(), error)).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Non-finite environment state")
        return observation

    def _info(self, reason: str, components: dict[str, float]) -> dict[str, Any]:
        """Collect diagnostics without leaking hidden wind or payload to the policy."""
        error = self.reference_position - self.dynamics.position
        info = {
            "position_error": error.copy(),
            "velocity_error": (self.dynamics.velocity - self.reference_velocity).copy(),
            "tracking_error_norm": float(np.linalg.norm(error)),
            "remaining_liquid_mass": float(self.payload.liquid_mass),
            "wind_velocity": self.wind.velocity(self.time).copy(),
            "trajectory_progress": float(self.progress),
            "current_waypoint": int(self.current_waypoint),  # timed reference target
            "waypoints_achieved": self.waypoints_achieved,
            "next_waypoint": (self.waypoints_achieved if self.waypoints_achieved <
                              len(self.trajectory.waypoints) else None),
            "actual_waypoint_progress": self.waypoints_achieved / len(self.trajectory.waypoints),
            "is_success": reason == "mission_complete",
            "termination_reason": reason,
            "physics_backend": "pybullet_3.2.7",
        }
        info.update(components)
        return info

    def step(self, action):
        """Apply one command and return the standard Gymnasium five-item result."""
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != (4,) or not np.all(np.isfinite(raw_action)):
            raise ValueError("Action must be a finite array with shape (4,)")
        # Clipping here gives the environment a final safety guard even if a
        # controller accidentally produces a value outside the declared space.
        action = np.clip(raw_action, -1.0, 1.0)
        thrust, commanded_attitude, yaw_rate = self._physical_action(action)
        wind_velocity = self.wind.velocity(self.time)
        wind_force = self.wind.aerodynamic_force(self.dynamics.velocity, wind_velocity)
        self.dynamics.step(thrust, commanded_attitude, yaw_rate, self.payload.total_mass,
                           wind_force, self.config.dt)
        self.time += self.config.dt
        self.step_count += 1
        self.payload.update(self.time)
        self.reference_position, self.reference_velocity, self.current_waypoint, self.progress = \
            self.trajectory.reference(self.time)
        # Only the next ordered waypoint is eligible; reference time cannot
        # award a bonus. Sampled 3-D arrival pays once, including the final point.
        waypoint_reached = self._check_waypoint_arrival()

        p = self.dynamics.position
        margin = self.config.boundary_margin
        boundary = not (-margin <= p[0] <= self.config.field_length + margin and
                        -margin <= p[1] <= self.config.field_width + margin and
                        0.0 <= p[2] <= 2.5 * self.config.altitude)
        crash = bool(p[2] <= 0.05)
        success = bool(self.waypoints_achieved == len(self.trajectory.waypoints) and
                       self.progress >= 1.0 and np.linalg.norm(p - self.trajectory.waypoints[-1])
                       <= self.config.waypoint_tolerance)
        terminated = crash or boundary or success
        truncated = (self.step_count >= self.max_steps or self.time >= self.config.max_episode_time) and not terminated
        reason = "crash" if crash else "boundary_violation" if boundary else "mission_complete" if success \
            else "time_limit" if truncated else "running"

        position_error = self.reference_position - p
        velocity_error = self.dynamics.velocity - self.reference_velocity
        reward, components = compute_reward(position_error, velocity_error,
                                             self.dynamics.angular_velocity, action,
                                             waypoint_reached, crash or boundary,
                                             self.config.reward_weights, self.config.dt)
        self.episode_reward += reward
        self.history["time"].append(self.time)
        self.history["position"].append(p.copy())
        self.history["reference"].append(self.reference_position.copy())
        self.history["error"].append(float(np.linalg.norm(position_error)))
        self.history["action"].append(action.copy())
        self.history["reward"].append(reward)
        self.history["attitude"].append(self.dynamics.attitude.copy())
        self.history["velocity"].append(self.dynamics.velocity.copy())
        self.history["wind"].append(wind_velocity.copy())  # applied over this interval
        self.history["payload"].append(self.payload.liquid_mass)
        self.history["command"].append(np.array([thrust, *commanded_attitude[:2], yaw_rate]))
        self.history["waypoints_achieved"].append(self.waypoints_achieved)
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, self._info(reason, components)

    def render(self):
        """Draw the field, reference path and recorded vehicle track."""
        if self.render_mode == "rgb_array":
            return self.dynamics.camera_image()
        if self.render_mode == "human":
            # The GUI is updated whenever PyBullet advances the simulation.
            return None
        import matplotlib.pyplot as plt
        if self._figure is None:
            self._figure, self._axis = plt.subplots(figsize=(7, 5))
        self._axis.clear()
        self._axis.plot(self.trajectory.waypoints[:, 0], self.trajectory.waypoints[:, 1], "k--", label="Reference")
        if self.history["position"]:
            positions = np.asarray(self.history["position"])
            self._axis.plot(positions[:, 0], positions[:, 1], color="#168AAD", label="Drone")
        self._axis.scatter(*self.dynamics.position[:2], c="#E76F51", s=55, label="Current")
        self._axis.set(xlim=(-0.5, self.config.field_length + 0.5),
                       ylim=(-0.5, self.config.field_width + 0.5), xlabel="x (m)", ylabel="y (m)",
                       title="Aerial spraying coverage")
        self._axis.set_aspect("equal")
        self._axis.grid(alpha=0.25)
        self._axis.legend(loc="upper right")
        self._figure.tight_layout()
        self._figure.canvas.draw()
        if self.render_mode == "human":
            plt.pause(0.001)
            return None
        rgba = np.asarray(self._figure.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self):
        """Release the Matplotlib window when interactive rendering was used."""
        if self._figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self._figure)
            self._figure = None
        self.dynamics.close()
