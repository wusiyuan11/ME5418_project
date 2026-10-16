"""Educational 2-D Gymnasium aerial-spraying environment with explained code."""

# Import terminal, data-export, executable-detection, and path utilities.
import argparse
import csv
import shutil
from pathlib import Path

# Import the official reinforcement-learning environment interface.
import gymnasium as gym
# Import Matplotlib and select a display-independent file-rendering backend.
import matplotlib
matplotlib.use("Agg")
# Import plotting and animation tools for PNG, GIF, and MP4 output.
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
# Import NumPy for vectors, random sampling, and dynamics calculations.
import numpy as np


# Define the simplified horizontal aerial-spraying environment.
class SprayingEnv(gym.Env):
    """Simulate a 2-D point mass following a lawnmower route.

    Inputs configure maximum wind speed, liquid loss rate, initial liquid,
    optional fixed wind angle in radians, and random seed. This model has no
    altitude, attitude, motors, rotor aerodynamics, or payload sloshing.
    """

    # Declare supported display modes and a nominal display rate.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Initialize physical settings, Gymnasium spaces, path, and episode state.
    def __init__(self, wind_max=3.0, spray_rate=0.02, initial_water=0.8, wind_angle=None, seed=7):
        # Initialize the inherited Gymnasium implementation.
        super().__init__()
        # Define the fixed numerical time step in seconds.
        self.dt = 0.1
        # Define the square agricultural field size in metres.
        self.field_size = 20.0
        # Define the empty drone mass in kilograms.
        self.dry_mass = 1.2
        # Validate and store the initial liquid mass in kilograms.
        self.initial_water = max(0.0, float(initial_water))
        # Validate and store the liquid spray rate in kilograms per second.
        self.spray_rate = max(0.0, float(spray_rate))
        # Validate and store the largest random wind speed in metres per second.
        self.wind_max = max(0.0, float(wind_max))
        # Store an optional fixed world-frame wind direction.
        self.fixed_wind_angle = None if wind_angle is None else float(wind_angle)
        # Set an illustrative, uncalibrated quadratic drag-force scale.
        self.drag_coefficient = 0.12
        # Set the largest commanded planar acceleration in metres per second squared.
        self.max_accel = 2.0
        # Set the planar speed limit in metres per second.
        self.max_speed = 2.0
        # Limit each episode to 1,100 steps, or 110 simulated seconds.
        self.max_steps = 1100
        # Create a reproducible local random-number generator.
        self.rng = np.random.default_rng(seed)
        # Declare two normalized horizontal acceleration commands.
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Declare the nine values returned to a controller or learning policy.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        # Generate the alternating field-coverage waypoints.
        self.waypoints = self._make_path()
        # Create a valid initial state immediately.
        self.reset(seed=seed)

    # Generate five alternating horizontal spraying lanes.
    def _make_path(self):
        """Return ten two-dimensional lane endpoints in metres."""
        # Start an empty waypoint list.
        points = []
        # Visit five lanes spaced four metres apart.
        for lane, y_position in enumerate([2.0, 6.0, 10.0, 14.0, 18.0]):
            # Alternate horizontal direction to form a continuous lawnmower path.
            x_positions = (1.0, 19.0) if lane % 2 == 0 else (19.0, 1.0)
            # Add the current lane's start and end points.
            points.extend([(x_positions[0], y_position), (x_positions[1], y_position)])
        # Return the points as a double-precision array.
        return np.asarray(points, dtype=np.float64)

    # Start a fresh episode and sample constant wind for that episode.
    def reset(self, *, seed=None, options=None):
        """Reset state and return Gymnasium's ``(observation, info)`` pair."""
        # Let Gymnasium perform its standard seed bookkeeping.
        super().reset(seed=seed)
        # Recreate the local generator when an explicit seed is supplied.
        if seed is not None:
            # Seed NumPy independently for exact episode reproducibility.
            self.rng = np.random.default_rng(seed)
        # Place the drone at the first path point.
        self.pos = np.array([1.0, 2.0], dtype=np.float64)
        # Start with zero planar velocity.
        self.vel = np.zeros(2, dtype=np.float64)
        # Refill the liquid tank.
        self.water = float(self.initial_water)
        # Clear elapsed-step state.
        self.step_count = 0
        # Target the second waypoint because the first is the starting point.
        self.target_idx = 1
        # Sample wind magnitude uniformly between zero and the configured maximum.
        self.wind_speed = float(self.rng.uniform(0.0, self.wind_max))
        # Sample direction uniformly over a full circle unless fixed by the user.
        self.wind_angle = float(self.rng.uniform(0.0, 2.0 * np.pi)) if self.fixed_wind_angle is None else self.fixed_wind_angle
        # Convert wind magnitude and angle into a world-frame x-y vector.
        self.wind = self.wind_speed * np.array([np.cos(self.wind_angle), np.sin(self.wind_angle)])
        # Initialize synchronized histories for output and animation.
        self.history = [self.pos.copy()]
        self.mass_history = [self.mass]
        self.error_history = [self.tracking_error]
        self.target_history = [self.target_idx]
        # Return the policy observation and diagnostic information.
        return self._observe(), self._info(False, False)

    # Expose simulated time as a read-only value.
    @property
    def time(self):
        """Return simulated elapsed time in seconds."""
        # Convert completed steps into physical time.
        return self.step_count * self.dt

    # Expose total vehicle mass as a read-only value.
    @property
    def mass(self):
        """Return dry mass plus remaining liquid mass in kilograms."""
        # Add the non-negative payload to empty vehicle mass.
        return self.dry_mass + self.water

    # Expose distance to the active waypoint as a read-only value.
    @property
    def tracking_error(self):
        """Return active-waypoint Euclidean distance in metres."""
        # Measure the norm of the target displacement vector.
        return float(np.linalg.norm(self.waypoints[self.target_idx] - self.pos))

    # Construct the nine-component policy observation.
    def _observe(self):
        """Return position, velocity, target error, progress, time, and speed."""
        # Calculate the two-component displacement to the active target.
        error = self.waypoints[self.target_idx] - self.pos
        # Normalize route progress by the final waypoint index.
        waypoint_fraction = self.target_idx / float(len(self.waypoints) - 1)
        # Normalize episode time by the step limit.
        time_fraction = self.step_count / float(self.max_steps)
        # Calculate scalar planar speed.
        speed = float(np.linalg.norm(self.vel))
        # Pack exactly nine single-precision features.
        return np.asarray([*self.pos, *self.vel, *error, waypoint_fraction, time_fraction, speed], dtype=np.float32)

    # Construct diagnostics that are not included in the policy observation.
    def _info(self, success, crashed):
        """Return mass, wind, error, time, target, and status diagnostics."""
        # Return a fresh dictionary of scalar and vector diagnostics.
        return {"mass_kg": self.mass, "water_kg": self.water, "wind_mps": self.wind.copy(), "wind_speed_mps": self.wind_speed, "wind_angle_rad": self.wind_angle, "target_index": self.target_idx, "tracking_error_m": self.tracking_error, "time_s": self.time, "success": bool(success), "crashed": bool(crashed)}

    # Apply one normalized acceleration action and advance simplified dynamics.
    def step(self, action):
        """Return ``observation, reward, terminated, truncated, info``."""
        # Convert the supplied command into a numerical array.
        action_array = np.asarray(action, dtype=np.float64)
        # Reject a wrong shape instead of allowing unsafe NumPy broadcasting.
        if action_array.shape != self.action_space.shape:
            # Explain the required and received shapes in the exception.
            raise ValueError(f"Action shape {action_array.shape} does not match {self.action_space.shape}.")
        # Enforce normalized action limits.
        bounded_action = np.clip(action_array, -1.0, 1.0)
        # Record pre-transition target distance for reward shaping.
        previous_distance = self.tracking_error
        # Calculate velocity relative to the world-frame moving air.
        relative_air_velocity = self.vel - self.wind
        # Apply illustrative quadratic drag opposite relative air motion.
        wind_force = -self.drag_coefficient * np.linalg.norm(relative_air_velocity) * relative_air_velocity
        # Convert the wind force to mass-dependent acceleration and add commanded acceleration.
        acceleration = self.max_accel * bounded_action + wind_force / self.mass
        # Integrate velocity with forward Euler integration.
        self.vel = self.vel + acceleration * self.dt
        # Measure speed before enforcing the safety limit.
        speed = float(np.linalg.norm(self.vel))
        # Restrict excessive speed while preserving direction.
        if speed > self.max_speed:
            # Scale velocity back to the permitted magnitude.
            self.vel *= self.max_speed / speed
        # Integrate planar position using updated velocity.
        self.pos = self.pos + self.vel * self.dt
        # Apply the exact analytical payload equation at the new time.
        self.water = max(self.initial_water - self.spray_rate * ((self.step_count + 1) * self.dt), 0.0)
        # Advance the discrete time counter.
        self.step_count += 1
        # Measure post-transition distance to the current target.
        distance = self.tracking_error
        # Declare a waypoint reached inside a 0.38-metre radius.
        waypoint_reached = distance < 0.38
        # Default mission completion to false.
        success = False
        # Process the reached waypoint when appropriate.
        if waypoint_reached:
            # Complete the mission at the final path point.
            if self.target_idx == len(self.waypoints) - 1:
                # Mark successful full-path completion.
                success = True
            # Otherwise target the next path point.
            else:
                # Increment the active waypoint index.
                self.target_idx += 1
        # Treat leaving the rectangular field as a crash.
        crashed = bool(np.any(self.pos < 0.0) or np.any(self.pos > self.field_size))
        # End naturally on success or boundary failure.
        terminated = success or crashed
        # Truncate at the time limit only when not naturally terminated.
        truncated = self.step_count >= self.max_steps and not terminated
        # Reward progress and penalize target distance and control effort.
        reward = 3.0 * (previous_distance - distance) - 0.08 * distance - 0.005 * float(np.dot(bounded_action, bounded_action))
        # Add a waypoint-reaching bonus.
        reward += 4.0 if waypoint_reached else 0.0
        # Add a mission-completion bonus.
        reward += 40.0 if success else 0.0
        # Add a boundary-failure penalty.
        reward -= 40.0 if crashed else 0.0
        # Append synchronized histories for analysis and animation.
        self.history.append(self.pos.copy())
        self.mass_history.append(self.mass)
        self.error_history.append(self.tracking_error)
        self.target_history.append(self.target_idx)
        # Return Gymnasium's standard transition tuple.
        return self._observe(), float(reward), bool(terminated), bool(truncated), self._info(success, crashed)

    # Draw the final field, routes, vehicle, target, wind, and telemetry.
    def render(self):
        """Return a Matplotlib figure representing the current episode."""
        # Create a square field plot.
        figure, axis = plt.subplots(figsize=(7.5, 7.0))
        # Shade the field light green.
        axis.set_facecolor("#edf7e8")
        # Draw reference and actual paths with distinct colors.
        axis.plot(self.waypoints[:, 0], self.waypoints[:, 1], "--", color="#1565c0", linewidth=2, label="Reference trajectory")
        actual = np.asarray(self.history)
        axis.plot(actual[:, 0], actual[:, 1], color="#ef6c00", linewidth=2, label="Actual trajectory")
        # Mark current drone and target positions.
        axis.scatter(*self.pos, s=100, marker="^", color="#212121", label="Drone", zorder=5)
        axis.scatter(*self.waypoints[self.target_idx], s=140, marker="*", color="#c2185b", label="Target waypoint", zorder=5)
        # Draw wind direction and relative magnitude in world coordinates.
        axis.quiver(10, 10, self.wind[0], self.wind[1], color="#00897b", scale=8, width=0.009, label="Wind")
        # Determine whether the final target is currently reached.
        status = "complete" if self.target_idx == len(self.waypoints) - 1 and self.tracking_error < 0.38 else "in progress"
        # Apply field limits, unit labels, and numerical status text.
        axis.set(xlim=(0, self.field_size), ylim=(0, self.field_size), xlabel="World x (m)", ylabel="World y (m)", title=f"Aerial spraying demo | t={self.time:.1f} s | mass={self.mass:.2f} kg\nwind={self.wind_speed:.2f} m/s | error={self.tracking_error:.2f} m | mission={status}")
        # Preserve true geometric scale and add grid and legend.
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
        # Return the completed figure.
        return figure

    # Release any Matplotlib resources after use.
    def close(self):
        """Close every open figure created by this process."""
        # Close all figures to prevent resource warnings.
        plt.close("all")


# Implement the current proportional-derivative-style waypoint controller.
def waypoint_controller(env):
    """Return one bounded normalized action; this is not PPO or attitude PID."""
    # Calculate position error to the active target.
    position_error = env.waypoints[env.target_idx] - env.pos
    # Convert position error into a bounded desired velocity.
    desired_velocity = np.clip(position_error, -1.6, 1.6)
    # Convert velocity error into a normalized acceleration command.
    command = 2.0 * (desired_velocity - env.vel) / env.max_accel
    # Return a valid single-precision action.
    return np.clip(command, -1.0, 1.0).astype(np.float32)


# Run the demonstration controller until the episode ends.
def run_episode(env, seed):
    """Run one seeded episode and return its final diagnostic dictionary."""
    # Reset state and reproduce the requested wind.
    observation, info = env.reset(seed=seed)
    # Allow at most the configured number of transitions.
    for unused_step in range(env.max_steps):
        # Compute the reference controller's next action.
        action = waypoint_controller(env)
        # Advance the simulation by one transition.
        observation, reward, terminated, truncated, info = env.step(action)
        # Stop after success, crash, or time limit.
        if terminated or truncated:
            # Exit while retaining the final diagnostics.
            break
    # Return the actual final episode information.
    return info


# Save synchronized numerical results for independent analysis.
def save_results_csv(env, output_path):
    """Write time, position, mass, error, target, and wind histories to CSV."""
    # Open a UTF-8 CSV file with portable newline handling.
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        # Define explicit result columns.
        writer = csv.DictWriter(stream, fieldnames=["time_s", "x_m", "y_m", "mass_kg", "tracking_error_m", "target_index", "wind_x_mps", "wind_y_mps"])
        # Write the header row.
        writer.writeheader()
        # Iterate through synchronized history samples.
        for index, position in enumerate(env.history):
            # Write one complete numerical record.
            writer.writerow({"time_s": index * env.dt, "x_m": position[0], "y_m": position[1], "mass_kg": env.mass_history[index], "tracking_error_m": env.error_history[index], "target_index": env.target_history[index], "wind_x_mps": env.wind[0], "wind_y_mps": env.wind[1]})


# Save requested static trajectory, mass, and tracking-error figures.
def save_static_figures(env, folder):
    """Create three publication-ready PNG diagnostics."""
    # Render and save the final trajectory comparison.
    trajectory_figure = env.render()
    trajectory_figure.savefig(folder / "trajectory.png", dpi=180, bbox_inches="tight")
    plt.close(trajectory_figure)
    # Build a time vector aligned with the recorded histories.
    times = np.arange(len(env.mass_history), dtype=float) * env.dt
    # Plot total mass and the empty-vehicle lower bound.
    mass_figure, mass_axis = plt.subplots(figsize=(8, 4.2))
    mass_axis.plot(times, env.mass_history, color="#2e7d32", linewidth=2.2)
    mass_axis.axhline(env.dry_mass, color="#616161", linestyle="--", label="Empty drone mass")
    mass_axis.set(xlabel="Simulation time (s)", ylabel="Total mass (kg)", title="Time-varying payload mass")
    mass_axis.grid(True, alpha=0.25)
    mass_axis.legend()
    mass_figure.savefig(folder / "mass_curve.png", dpi=180, bbox_inches="tight")
    plt.close(mass_figure)
    # Plot active-waypoint tracking error over time.
    error_figure, error_axis = plt.subplots(figsize=(8, 4.2))
    error_axis.plot(times, env.error_history, color="#c62828", linewidth=1.6)
    error_axis.set(xlabel="Simulation time (s)", ylabel="Active-waypoint distance (m)", title="Waypoint tracking error versus time")
    error_axis.grid(True, alpha=0.25)
    error_figure.savefig(folder / "tracking_error.png", dpi=180, bbox_inches="tight")
    plt.close(error_figure)


# Save a continuously updated visual demonstration.
def save_animation(env, folder):
    """Save a GIF and, when FFmpeg exists, an MP4; return MP4 availability."""
    # Convert trajectory history to an array and select every third sample.
    positions = np.asarray(env.history)
    frame_indices = np.arange(0, len(positions), 3, dtype=int)
    # Guarantee that the final episode state appears.
    if frame_indices[-1] != len(positions) - 1:
        # Append the missing final sample.
        frame_indices = np.append(frame_indices, len(positions) - 1)
    # Create the animated agricultural field.
    figure, axis = plt.subplots(figsize=(8.2, 7))
    axis.set_facecolor("#edf7e8")
    axis.plot(env.waypoints[:, 0], env.waypoints[:, 1], "--", color="#1565c0", linewidth=2, label="Reference trajectory")
    actual_line, = axis.plot([], [], color="#ef6c00", linewidth=2.2, label="Actual trajectory")
    drone_marker, = axis.plot([], [], "^", color="#212121", markersize=10, label="Drone")
    target_marker, = axis.plot([], [], "*", color="#c2185b", markersize=14, label="Target waypoint")
    axis.quiver(10, 10, env.wind[0], env.wind[1], color="#00897b", scale=8, width=0.009, label="Wind direction")
    telemetry = axis.text(0.02, 0.02, "", transform=axis.transAxes, va="bottom", fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    axis.set(xlim=(0, env.field_size), ylim=(0, env.field_size), xlabel="World x (m)", ylabel="World y (m)", title="ME5418 aerial spraying trajectory demonstration")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)

    # Update moving artists and telemetry for one animation frame.
    def update(frame_number):
        """Return the artists changed for one recorded frame."""
        # Map rendered frame number to simulation-history index.
        index = int(frame_indices[frame_number])
        # Extend the path and move drone and target markers.
        actual_line.set_data(positions[: index + 1, 0], positions[: index + 1, 1])
        drone_marker.set_data([positions[index, 0]], [positions[index, 1]])
        target_index = int(env.target_history[index])
        target_marker.set_data([env.waypoints[target_index, 0]], [env.waypoints[target_index, 1]])
        # Report completion only at a successful final state.
        status = "COMPLETE" if index == len(positions) - 1 and target_index == len(env.waypoints) - 1 and env.error_history[index] < 0.38 else "IN PROGRESS"
        # Display all requested live numerical fields.
        telemetry.set_text(f"Time: {index * env.dt:5.1f} s\nWind: {env.wind_speed:4.2f} m/s at {np.degrees(env.wind_angle) % 360:5.1f} deg\nPayload: {env.mass_history[index] - env.dry_mass:4.2f} kg\nTotal mass: {env.mass_history[index]:4.2f} kg\nTracking error: {env.error_history[index]:5.2f} m\nMission: {status}")
        # Return all changing Matplotlib artists.
        return actual_line, drone_marker, target_marker, telemetry

    # Construct and save the GIF at 15 rendered frames per second.
    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=1000 / 15, blit=True)
    animation.save(folder / "flight_demo.gif", writer=PillowWriter(fps=15))
    # Detect and use FFmpeg for optional MP4 output.
    mp4_supported = shutil.which("ffmpeg") is not None
    if mp4_supported:
        # Encode the same animation as an MP4 file.
        animation.save(folder / "flight_demo.mp4", writer=FFMpegWriter(fps=15, bitrate=1800))
    # Release animation resources and report MP4 support.
    plt.close(figure)
    return mp4_supported


# Provide the terminal entry point for the complete demonstration.
def main():
    """Parse settings, run one episode, and export all requested artifacts."""
    # Define documented terminal options.
    parser = argparse.ArgumentParser(description="ME5418 2-D aerial-spraying demonstration")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--output", default="outputs", help="artifact directory")
    parser.add_argument("--wind-max", type=float, default=3.0, help="maximum wind speed in m/s")
    parser.add_argument("--wind-angle-deg", type=float, default=None, help="fixed world-frame wind angle")
    parser.add_argument("--initial-payload", type=float, default=0.8, help="initial liquid in kg")
    parser.add_argument("--spray-rate", type=float, default=0.02, help="liquid loss in kg/s")
    # Parse user-supplied arguments.
    arguments = parser.parse_args()
    # Convert optional degrees to radians.
    fixed_angle = None if arguments.wind_angle_deg is None else np.deg2rad(arguments.wind_angle_deg)
    # Create the requested output folder.
    output_folder = Path(arguments.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    # Construct and run the configured seeded environment.
    environment = SprayingEnv(arguments.wind_max, arguments.spray_rate, arguments.initial_payload, fixed_angle, arguments.seed)
    final_info = run_episode(environment, arguments.seed)
    # Export numerical data, figures, and animations.
    save_results_csv(environment, output_folder / "simulation_results.csv")
    save_static_figures(environment, output_folder)
    mp4_saved = save_animation(environment, output_folder)
    # Calculate the measured active-waypoint error statistic.
    tracking_rmse = float(np.sqrt(np.mean(np.square(environment.error_history))))
    # Print actual runtime, disturbance, outcome, and output facts.
    print(f"Gymnasium version: {gym.__version__}")
    print(f"Wind: {environment.wind_speed:.3f} m/s at {np.degrees(environment.wind_angle) % 360:.2f} degrees")
    print(f"Success: {final_info['success']}; steps: {environment.step_count}; final mass: {environment.mass:.3f} kg; waypoint RMSE: {tracking_rmse:.3f} m")
    print(f"MP4 saved: {mp4_saved}")
    print(f"Outputs saved in: {output_folder.resolve()}")
    # Close remaining resources.
    environment.close()


# Run the terminal workflow only when this file is executed directly.
if __name__ == "__main__":
    # Start the demonstration.
    main()
