"""Entities for Robin Labs."""

from src.robin.demand.entities import Demand
from src.robin.kernel.entities import Kernel
from src.robin.supply.entities import Supply

from typing import Tuple


class RobinLab:
    """Robin Lab experiment."""
    def __init__(self,
                 path_config_supply: str,
                 path_config_demand: str):
        """
        Initialize Robin Lab experiment.

        Args:
            path_config_supply: Path to supply config.
            path_config_demand: Path to demand config.
        """
        self.default_supply = Supply.from_yaml(path_config_supply)
        self.default_demand = Demand.from_yaml(path_config_demand)

        # Experimental data should be generated based on a specific experiment requirements.
        self.experimental_data = [("supply", "demand")]

    def prices(self,
               price_percentages: Tuple[float, float],
               step: int = 1):
        """Run experiment."""
        # Specify path to save yamls.

        # Generate yamls for each price percentage.
        # 1. Modify loaded supply
        # 2. Save modified supply
        pass

    def run(self):
        # Define experiment main loop.
        for supply, demand in self.experimental_data:
            # Run experiment.
            kernel = Kernel(path_config_supply=supply, path_config_demand=demand)
            kernel.simulate(output_path="output")