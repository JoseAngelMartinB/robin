"""Entities for Robin Labs."""

import numpy as np
import os

from src.robin.demand.entities import Demand
from src.robin.kernel.entities import Kernel
from src.robin.supply.entities import Supply
from src.robin.scraping.entities import SupplySaver
from copy import deepcopy
from typing import Mapping

TMP_PATH = "data/labs/tmp/"


class RobinLab:
    """Robin Lab experiment."""

    def __init__(self,
                 path_config_supply: str,
                 path_config_demand: str,
                 lab_config: Mapping):
        """
        Initialize Robin Lab experiment.

        Args:
            path_config_supply: Path to supply config.
            path_config_demand: Path to demand config.
        """
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)

        self.lab_config = lab_config
        self.default_supply = Supply.from_yaml(path_config_supply)
        self.default_demand = Demand.from_yaml(path_config_demand)

        if not any(self.lab_config.values()):
            raise ValueError("At least one of the lab configs must be non-empty.")

        # Check if it's a supply or demand experiment.
        if self.lab_config["supply"]:
            self._supply_yaml_editor()
        elif self.lab_config["demand"]:
            self._demand_yaml_editor()
        else:
            raise ValueError("At least one of the lab configs must be non-empty.")

    def _supply_yaml_editor(self):
        supply_lab_config = self.lab_config["supply"]

        print(f"Generating supply data...")
        arange_args = supply_lab_config
        for i in np.arange(**arange_args):
            tmp_services = []
            for service in self.default_supply.services:
                tmp_service = deepcopy(service)
                for origin_destination in tmp_service.prices:
                    for seat in tmp_service.prices[origin_destination]:
                        tmp_service.prices[origin_destination][seat] *= (1 + i)
                tmp_services.append(tmp_service)

            SupplySaver(services=tmp_services).to_yaml(filename=f"supply_{i}.yml", save_path=TMP_PATH)

    def _demand_yaml_editor(self):
        pass

    def run(self):
        pass
