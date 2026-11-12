"""Replay recorded telemetry as a timestamped GIF without a display server."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def save_episode_animation(history: dict, waypoints: np.ndarray, output_path: str | Path,
                           fps: int = 10, playback_speed: float = 2.0) -> None:
    """Show path, applied wind, actual attitude and held interval commands.

    Frames are decimated for playback only; waypoint detection uses every physics
    step. Wind is a world-frame velocity vector pointing toward its destination.
    """
    if fps <= 0 or playback_speed <= 0 or not history["time"]:
        raise ValueError("Positive fps, playback speed and nonempty history required")
    path = Path(output_path)
    if path.suffix.lower() != ".gif":
        raise ValueError("Animation output must use .gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: np.asarray(value) for key, value in history.items()}
    time = data["time"]
    targets = np.arange(time[0], time[-1], playback_speed / fps)
    indices = np.unique(np.r_[np.searchsorted(time, targets), len(time)-1])
    fig = plt.figure(figsize=(11, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1))
    ax = fig.add_subplot(grid[:, 0])
    attitude_ax = fig.add_subplot(grid[0, 1])
    text_ax = fig.add_subplot(grid[1, 1]); text_ax.axis("off")
    ax.plot(waypoints[:, 0], waypoints[:, 1], "k--", alpha=.4, label="Planned path")
    ax.scatter(waypoints[:, 0], waypoints[:, 1], s=20, color="gray")
    trail, = ax.plot([], [], color="#168AAD", label="Vehicle track")
    vehicle, = ax.plot([], [], "o", color="#168AAD")
    reference, = ax.plot([], [], "x", color="#E76F51", label="Timed reference")
    achieved = ax.scatter([], [], color="#2A9D8F", s=45, label="Achieved in order")
    wind = ax.quiver([1], [1], [0], [0], angles="xy", scale_units="xy", scale=1,
                     color="#9750A0")
    ax.set(xlim=(-2, waypoints[:, 0].max()+2), ylim=(-2, waypoints[:, 1].max()+2),
           xlabel="x (m)", ylabel="y (m)", title="Wind arrow: 1 m per 1 m/s, points toward flow")
    ax.set_aspect("equal"); ax.legend(loc="upper center", fontsize=8); ax.grid(alpha=.2)
    angles = np.rad2deg(data["attitude"])
    commands = np.rad2deg(data["command"][:, 1:3])
    for j, (name, color) in enumerate(zip(("Roll", "Pitch", "Yaw"), ("#168AAD", "#E76F51", "#2A9D8F"))):
        attitude_ax.plot(time, angles[:, j], color=color, label=name)
        if j < 2:
            attitude_ax.plot(time, commands[:, j], "--", color=color, alpha=.5, label=name+" cmd")
    cursor = attitude_ax.axvline(time[0], color="black")
    attitude_ax.set(xlabel="Simulation time (s)", ylabel="Angle (deg)", title="Attitude response and commands")
    attitude_ax.legend(fontsize=7, ncol=3); attitude_ax.grid(alpha=.2)
    status = text_ax.text(0, 1, "", va="top", family="monospace", fontsize=10)
    title = fig.suptitle("")

    def update(i):
        p = data["position"][i]; w = data["wind"][i]; c = data["command"][i]
        trail.set_data(data["position"][:i+1, 0], data["position"][:i+1, 1])
        vehicle.set_data([p[0]], [p[1]])
        reference.set_data([data["reference"][i, 0]], [data["reference"][i, 1]])
        achieved.set_offsets(waypoints[:int(data["waypoints_achieved"][i]), :2])
        wind.set_UVC([w[0]], [w[1]])
        cursor.set_xdata([time[i], time[i]])
        direction = np.degrees(np.arctan2(w[1], w[0])) % 360
        title.set_text(f"Aerial spraying | t = {time[i]:.2f} s | playback {playback_speed:g}x")
        status.set_text(
            f"Applied wind: {np.linalg.norm(w):.2f} m/s\n"
            f"Flow direction: {direction:.0f} deg (+x toward +y)\n"
            f"Remaining liquid: {data['payload'][i]:.3f} kg\n"
            f"Altitude: {p[2]:.2f} m | speed: {np.linalg.norm(data['velocity'][i]):.2f} m/s\n"
            f"Tracking error: {data['error'][i]:.2f} m\n"
            f"Waypoints: {int(data['waypoints_achieved'][i])}/{len(waypoints)}\n\n"
            f"Held commands for preceding interval\n"
            f"Thrust: {c[0]:.2f} N\n"
            f"Roll / pitch: {np.rad2deg(c[1]):+.1f} / {np.rad2deg(c[2]):+.1f} deg\n"
            f"Yaw rate: {np.rad2deg(c[3]):+.1f} deg/s")
        return trail, vehicle, reference, achieved, wind, cursor, status, title

    animation = FuncAnimation(fig, update, frames=indices, interval=1000/fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
