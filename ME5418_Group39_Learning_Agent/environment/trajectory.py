"""Build the alternating passes used to cover the spraying field."""

from dataclasses import dataclass
import numpy as np


@dataclass
class LawnMowerTrajectory:
    """Generate a repeatable lawn-mower path and interpolate along it in time."""
    field_width: float = 8.0
    field_length: float = 12.0
    lane_spacing: float = 2.0
    altitude: float = 2.0
    speed: float = 2.0
    waypoint_tolerance: float = 0.5

    def __post_init__(self) -> None:
        """Create the waypoint list after the trajectory settings are known."""
        if min(self.field_width, self.field_length, self.lane_spacing, self.speed) <= 0:
            raise ValueError("Field dimensions, lane spacing, and speed must be positive")
        lanes = np.arange(0.0, self.field_width + 1e-9, self.lane_spacing)
        if lanes[-1] < self.field_width:
            lanes = np.append(lanes, self.field_width)
        points = []
        # Adjacent lanes are flown in opposite directions.  This avoids an
        # unnecessary return to the same side of the field after every pass.
        for index, y in enumerate(lanes):
            xs = (0.0, self.field_length) if index % 2 == 0 else (self.field_length, 0.0)
            if not points:
                points.append([xs[0], y, self.altitude])
            points.append([xs[1], y, self.altitude])
            if index < len(lanes) - 1:
                points.append([xs[1], lanes[index + 1], self.altitude])
        self.waypoints = np.asarray(points, dtype=float)
        self.segment_vectors = np.diff(self.waypoints, axis=0)
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        self.cumulative_lengths = np.concatenate(([0.0], np.cumsum(self.segment_lengths)))
        self.path_length = float(self.cumulative_lengths[-1])
        self.duration = self.path_length / self.speed

    def reference(self, time: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Return reference position, velocity, waypoint number and progress."""
        distance = min(max(time, 0.0) * self.speed, self.path_length)
        index = min(np.searchsorted(self.cumulative_lengths, distance, side="right") - 1,
                    len(self.segment_lengths) - 1)
        fraction = (distance - self.cumulative_lengths[index]) / self.segment_lengths[index]
        position = self.waypoints[index] + fraction * self.segment_vectors[index]
        velocity = self.speed * self.segment_vectors[index] / self.segment_lengths[index]
        if distance >= self.path_length:
            velocity = np.zeros(3)
        return position, velocity, index + 1, distance / self.path_length
