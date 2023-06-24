"""Entities for Robin Labs."""

import numpy as np
import os
import shutil

from src.robin.supply.entities import Supply
from src.robin.demand.entities import Demand
from src.robin.kernel.entities import Kernel
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
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand

        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)

        if not os.path.exists(TMP_PATH+"supply/"):
            os.makedirs(TMP_PATH+"supply/")

        if not os.path.exists(TMP_PATH+"demand/"):
            os.makedirs(TMP_PATH+"demand/")

        if not os.path.exists(TMP_PATH+"output/"):
            os.makedirs(TMP_PATH+"output/")

        self.lab_config = lab_config
        self.default_supply = Supply.from_yaml(path_config_supply)
        self.default_demand = Demand.from_yaml(path_config_demand)

        if not any(self.lab_config.values()):
            raise ValueError("At least one of the lab configs must be non-empty.")

        # Check if it's a supply or demand experiment.
        if self.lab_config["supply"]:
            self._supply_yaml_editor()
            shutil.copy(path_config_demand, TMP_PATH + "demand/" + os.path.basename(path_config_demand))
        else:
            raise ValueError("At least one of the lab configs must be non-empty.")

        self.sim_sequence = {}

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

            supply_file_name = f"supply_{i}.yml"
            SupplySaver(services=tmp_services).to_yaml(filename=supply_file_name, save_path=TMP_PATH+"supply/")

    def simulate(self):
        """Simulate the experiment."""
        # For each supply file, run the simulation.
        for supply_file in sorted(os.listdir(TMP_PATH+"supply/")):
            kernel = Kernel(path_config_supply=TMP_PATH+"supply/"+supply_file,
                            path_config_demand=TMP_PATH+"demand/"+os.path.basename(self.path_config_demand))
            kernel.simulate(output_path=TMP_PATH+"output/"+supply_file.split("_")[-1].split(".")[0]+".csv")
            print("Successfully simulated supply file: ", supply_file)

    def get_elasticity_curve(self):
        pass
