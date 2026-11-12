"""Turn a completed episode into figures that are easy to inspect and submit."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_episode_plots(history: dict, waypoints: np.ndarray, output_path: str | Path,
                       error_path: str | Path | None = None, title: str = "Episode trajectory") -> None:
    """Save the flown path and, when requested, a separate tracking-error plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(history["position"])
    reference = np.asarray(history["reference"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(waypoints[:, 0], waypoints[:, 1], "--", color="#495057", lw=1.5, label="Waypoints")
    axes[0].plot(reference[:, 0], reference[:, 1], color="#F4A261", lw=2, label="Reference")
    axes[0].plot(actual[:, 0], actual[:, 1], color="#168AAD", lw=1.6, label="Drone")
    axes[0].scatter(*waypoints[0, :2], c="#2A9D8F", s=55, label="Start")
    axes[0].scatter(*waypoints[-1, :2], c="#E76F51", marker="*", s=85, label="Goal")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Top view")
    axes[0].axis("equal"); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
    axes[1].plot(history["time"], actual[:, 2], label="Actual altitude", color="#168AAD")
    axes[1].plot(history["time"], reference[:, 2], "--", label="Reference altitude", color="#F4A261")
    axes[1].set(xlabel="Time (s)", ylabel="z (m)", title="Altitude")
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold"); fig.tight_layout(); fig.savefig(output_path, dpi=170); plt.close(fig)
    if error_path is not None:
        error_path = Path(error_path); error_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        ax.plot(history["time"], history["error"], color="#D1495B", lw=1.5)
        ax.set(xlabel="Time (s)", ylabel="Position error (m)", title="Trajectory tracking error")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(error_path, dpi=170); plt.close(fig)
