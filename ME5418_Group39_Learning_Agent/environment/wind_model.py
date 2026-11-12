"""Generate repeatable wind conditions for each episode."""

from dataclasses import dataclass
import numpy as np


@dataclass
class WindModel:
    """Represent calm air, steady wind, random wind, or wind with smooth gusts."""

    mode: str = "gusts"
    max_speed: float = 3.0
    gust_amplitude: float = 0.7
    gust_frequency: float = 0.25

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        """Choose the episode's wind using Gymnasium's seeded random generator."""
        if self.mode not in {"none", "constant", "random", "gusts"}:
            raise ValueError(f"Unsupported wind mode: {self.mode}")
        if self.mode == "none":
            magnitude, direction = 0.0, 0.0
        elif self.mode == "constant":
            magnitude, direction = self.max_speed, 0.0
        else:
            magnitude = rng.uniform(0.0, self.max_speed)
            direction = rng.uniform(0.0, 2.0 * np.pi)
        self.base_velocity = np.array(
            [magnitude * np.cos(direction), magnitude * np.sin(direction), 0.0], dtype=float
        )
        self.gust_phase = rng.uniform(0.0, 2.0 * np.pi)
        self.gust_direction = rng.uniform(0.0, 2.0 * np.pi)
        return self.base_velocity.copy()

    def velocity(self, time: float) -> np.ndarray:
        """Return the wind velocity at the requested simulation time."""
        wind = self.base_velocity.copy()
        if self.mode == "gusts":
            value = self.gust_amplitude * np.sin(2.0 * np.pi * self.gust_frequency * time + self.gust_phase)
            wind[:2] += value * np.array([np.cos(self.gust_direction), np.sin(self.gust_direction)])
        return wind

    @staticmethod
    def aerodynamic_force(drone_velocity: np.ndarray, wind_velocity: np.ndarray) -> np.ndarray:
        """Convert relative airflow into a force acting on the vehicle.

        We use the familiar quadratic drag approximation
        ``0.5 * rho * Cd * A * speed * relative_airflow``.  It is deliberately
        simple, but it gives wind a measurable and physically sensible effect.
        """
        relative_airflow = wind_velocity - drone_velocity
        rho, drag_coefficient, reference_area = 1.225, 0.8, 0.10
        speed = np.linalg.norm(relative_airflow)
        return 0.5 * rho * drag_coefficient * reference_area * speed * relative_airflow
