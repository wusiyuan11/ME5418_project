"""Keep track of the spray liquid carried by the drone."""

from dataclasses import dataclass


@dataclass
class PayloadModel:
    """Drain the tank at a constant rate and report the current vehicle mass."""

    dry_mass: float = 1.50
    initial_liquid_mass: float = 0.50
    mass_flow_rate: float = 0.010
    liquid_mass: float = 0.50

    def reset(self, initial_liquid_mass: float, mass_flow_rate: float) -> None:
        """Fill the tank for a new episode and store the selected spray rate."""
        self.initial_liquid_mass = float(max(0.0, initial_liquid_mass))
        self.mass_flow_rate = float(max(0.0, mass_flow_rate))
        self.liquid_mass = self.initial_liquid_mass

    def update(self, elapsed_time: float) -> float:
        """Calculate the remaining liquid directly from elapsed episode time.

        Computing from the initial mass avoids slowly accumulating subtraction
        error over hundreds of simulation steps.
        """
        self.liquid_mass = max(self.initial_liquid_mass - self.mass_flow_rate * elapsed_time, 0.0)
        return self.liquid_mass

    @property
    def total_mass(self) -> float:
        """Return the dry-airframe mass plus whatever remains in the tank."""
        return self.dry_mass + self.liquid_mass
