"""Entities for Robin Labs."""

import numpy as np
import os
import pandas as pd
import shutil

from src.robin.supply.entities import Supply
from src.robin.demand.entities import Demand
from src.robin.kernel.entities import Kernel
from src.robin.scraping.entities import SupplySaver
from copy import deepcopy
from typing import Mapping

from tqdm.auto import tqdm
from matplotlib import pyplot as plt


class RobinLab:
    """Robin Lab experiment."""

    def __init__(self,
                 path_config_supply: str,
                 path_config_demand: str,
                 tmp_path: str,
                 lab_config: Mapping,
                 verbose=0):
        """
        Initialize Robin Lab experiment.

        Args:
            path_config_supply: Path to supply config.
            path_config_demand: Path to demand config.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.tmp_path = tmp_path

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

        if not os.path.exists(self.tmp_path + "/supply"):
            os.makedirs(self.tmp_path + "/supply")

        if not os.path.exists(self.tmp_path + "/demand"):
            os.makedirs(self.tmp_path + "/demand")

        if not os.path.exists(self.tmp_path + "/output"):
            os.makedirs(self.tmp_path + "/output")

        self.lab_config = lab_config
        self.default_supply = Supply.from_yaml(path_config_supply)
        self.default_demand = Demand.from_yaml(path_config_demand)

        if not any(self.lab_config.values()):
            raise ValueError("At least one of the lab configs must be non-empty.")

        # Check if it's a supply or demand experiment.
        if self.lab_config["supply"]:
            self._supply_yaml_editor()
            shutil.copy(path_config_demand, self.tmp_path + "/demand/" + os.path.basename(path_config_demand))
        else:
            raise ValueError("At least one of the lab configs must be non-empty.")

        self.sim_sequence = {}

    def _supply_yaml_editor(self):
        def multiply_price_values(dictionary: Mapping, factor: float):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    multiply_price_values(value, factor)
                elif isinstance(value, (int, float)):
                    dictionary[key] *= (1 + factor / 100)

        supply_lab_config = self.lab_config["supply"]

        print(f"Generating supply data...")
        arange_args = supply_lab_config
        for i, factor in enumerate(tqdm(np.arange(**arange_args)), start=1):
            tmp_services = deepcopy(self.default_supply.services)
            for tmp_service in tmp_services:
                multiply_price_values(tmp_service.prices, factor)

            supply_file_name = f"/supply_{i}.yml"
            SupplySaver(services=tmp_services).to_yaml(filename=supply_file_name, save_path=self.tmp_path+"/supply")

    def simulate(self):
        """Simulate the experiment."""
        # For each supply file, run the simulation.
        for i, supply_file in enumerate(tqdm(sorted(os.listdir(self.tmp_path + "/supply"), key=lambda x: int(x.split(".")[0].split("_")[-1]))), start=1):
            kernel = Kernel(path_config_supply=self.tmp_path + "/supply/" + supply_file,
                            path_config_demand=self.tmp_path + "/demand/" + os.path.basename(self.path_config_demand))
            kernel.simulate(output_path=f"{self.tmp_path}/output/output_{i}.csv")

    def _get_tickets_sold(self):
        tickets_sold = {}
        for output_file in sorted(os.listdir(self.tmp_path + "/output"), key=lambda x: int(x.split(".")[0].split("_")[-1])):
            df = pd.read_csv(self.tmp_path + "/output/" + output_file)
            buffer_tickets_sold = df.groupby(by=['seat']).size().to_dict()
            buffer_tickets_sold['Total'] = sum(buffer_tickets_sold.values())
            tickets_sold[output_file.split(".")[0].split("_")[-1]] = buffer_tickets_sold

        return tickets_sold

    def plot_elasticity_curve(self):
        tickets_sold = self._get_tickets_sold()

        supply_lab_config = self.lab_config["supply"]
        arange_args = supply_lab_config
        t = np.arange(**arange_args)
        series_keys = set(k for v in tickets_sold.values() for k in v)
        series = {k: [] for k in series_keys}

        for k in tickets_sold:
            for kk in series_keys:
                series[kk].append(tickets_sold[k].get(kk, 0))

        plt.figure(figsize=(10, 5))
        plt.title("Price elasticity curve")
        for k in series:
            plt.plot(t, series[k], label=k)

        plt.xticks(ticks=t[::3], labels=[f"{x:.0f}%" for x in t][::3])
        plt.xlabel("Price increase (%)")
        plt.ylabel("Tickets sold")
        plt.legend()
        plt.show()
