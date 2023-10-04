"""Entities for Robin Labs."""

import copy
import os
import progressbar
import shutil
import yaml

from src.robin.kernel.entities import Kernel
from src.robin.labs.utils import *
from src.robin.plotter.utils import plot_series

from matplotlib import pyplot as plt
from pathlib import Path
from typing import Mapping


class RobinLab:
    """
    Robin Lab class.

    This class is intended to run Robin simulation experiments.
    """
    def __init__(self,
                 path_config_supply: Path,
                 path_config_demand: Path,
                 tmp_path: Path,
                 verbose=0):
        """
        Initialize Robin Lab experiment.

        Args:
            path_config_supply: Path to supply config.
            path_config_demand: Path to demand config.
            tmp_path: Path to temporary directory.
            verbose: Verbosity level.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.tmp_path = tmp_path
        self.lab_config = None
        self.verbose = verbose
        self.seed = None

    def set_lab_config(self, config: Mapping):
        """
        Define the lab environment.

        Args:
            config: Lab configuration.
        """
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self._create_tmp_dir()
        self.lab_config = config

        if not any(self.lab_config.values()):
            raise ValueError("At least one of the lab configs must be non-empty.")

        if self.lab_config["supply"]:  # Check if it's a supply or demand experiment
            self._supply_yaml_editor()
            shutil.copy(self.path_config_demand, self.tmp_path / "demand" / self.path_config_demand.name)
        else:
            raise ValueError("At least one of the lab configs must be non-empty.")

        self.seed = self.lab_config.get("seed", None)

    def _create_tmp_dir(self):
        """
        Create temporary directory and folders.
        """
        if not self.tmp_path.exists():
            os.makedirs(self.tmp_path)

        if not os.path.exists(self.tmp_path / "supply"):
            os.makedirs(self.tmp_path / "supply")

        if not os.path.exists(self.tmp_path / "demand"):
            os.makedirs(self.tmp_path / "demand")

        if not os.path.exists(self.tmp_path / "output"):
            os.makedirs(self.tmp_path / "output")

    def _supply_yaml_editor(self) -> None:
        """
        Modify default supply based on provided simulation config.
        """
        def modify_prices(data: Mapping, factor: float):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == 'price':
                        data[key] = float(data[key] * (1 + factor / 100))
                    else:
                        modify_prices(value, factor)
            elif isinstance(data, list):
                for item in data:
                    modify_prices(item, factor)

        with open(self.path_config_supply, 'r') as file:
            original_data = yaml.load(file, Loader=yaml.CSafeLoader)

        supply_lab_config = self.lab_config["supply"]
        arange_args = supply_lab_config
        with progressbar.ProgressBar(min_value=1, max_value=len(np.arange(**arange_args))) as bar:
            for i, factor in enumerate(np.arange(**arange_args), start=1):
                modified_data = copy.deepcopy(original_data)
                modified_services = modified_data.get('service')
                assert modified_services, "No services found in the supply config."
                for service in modified_services:
                    modify_prices(service['origin_destination_tuples'], factor)

                modified_data['service'] = modified_services
                supply_file_name = f"supply_{i}.yml"
                save_path = f"{self.tmp_path}/supply/{supply_file_name}"
                with open(save_path, 'w') as file:
                    yaml.safe_dump(modified_data, file)
                bar.update(i)

    def simulate(self) -> None:
        """Simulate the experiment."""
        def file_number(file) -> int:
            """
            Get number from file name.

            Args:
                file: File name.

            Returns:
                int number from file name.
            """
            file = Path(file)
            file_name = file.stem
            return int(file_name.split("_")[-1])

        # Run simulation for each supply file
        sorted_supply_files = sorted(os.listdir(self.tmp_path / "supply"), key=file_number)
        with progressbar.ProgressBar(min_value=1, max_value=len(sorted_supply_files)) as bar:
            for i, supply_file in enumerate(sorted_supply_files, start=1):
                kernel = Kernel(path_config_supply=self.tmp_path / "supply" / supply_file,
                                path_config_demand=self.tmp_path / "demand" / self.path_config_demand.name,
                                seed=self.seed)
                kernel.simulate(output_path=Path(f"{self.tmp_path}/output/output_{i}.csv"))
                bar.update(i)

    def _get_tickets_sold(self) -> Mapping:
        """Get the number of tickets sold for each supply file."""
        tickets_sold = {}
        for output_file in sorted(os.listdir(self.tmp_path / "output"), key=lambda x: int(x.split(".")[0].split("_")[-1])):
            df = pd.read_csv(self.tmp_path / "output" / output_file)
            buffer_tickets_sold = df.groupby(by=['seat']).size().to_dict()
            buffer_tickets_sold['Total'] = sum(buffer_tickets_sold.values())
            tickets_sold[output_file.split(".")[0].split("_")[-1]] = buffer_tickets_sold

        return tickets_sold

    def plot_elasticity_curve(self, save_path: Path = None) -> None:
        """Plot the elasticity curve."""
        tickets_sold = self._get_tickets_sold()

        supply_lab_config = self.lab_config["supply"]
        arange_args = supply_lab_config
        x_data = np.arange(**arange_args)
        series_keys = set(key for value in tickets_sold.values() for key in value)
        series = {key: [] for key in series_keys}

        for file_key in tickets_sold:
            for seat_key in series_keys:
                series[seat_key].append(tickets_sold[file_key].get(seat_key, 0))

        fig, ax = plot_series(x_data=tuple(x_data),
                              y_data=series,
                              title="Prices elasticity curve",
                              xlabel="Price increase (%)",
                              ylabel="Tickets sold",
                              xticks=x_data[::3],
                              xticks_labels=tuple([f"{x:.0f}%" for x in x_data][::3]),
                              figsize=(10, 6))

        plt.show()

        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)


class RobinKPIs:
    """
        Robin KPIs class.

        This class is intended to calculate KPIs from multiple Robin's output files.
    """
    def __init__(self,
                 path_supply_files: Path,
                 path_demand_files: Path,
                 path_output_files: Path
        ):
        """
        Initialize Robin Lab experiment.

        Args:
            path_supply_files: Path to supply configurations.
            path_demand_files: Path to demand configurations.
            path_output_files: Path to output files.
        """
        self.path_supply_files = path_supply_files
        self.path_demand_files = path_demand_files
        self.path_output_files = path_output_files

    # TODO: Add KPIs calculation
    # Beneficio (bruto) total y desagregado por tipo de asiento
    # Nº pasajeros (Total y desagregado por tipos)
    # Ranking trayectos más demandados
    # Ranking trayectos mayor beneficio
    # Ranking trayectos mayor ocupación
    # Utilidad media (total y desagregada por tipos)
    # Ocupación media (total y desagregada por tipos) -> Revisar si es posible en servicio completo
    # o desagregado por tramos
    # Estudiar pasajeros con utilidad negativa (tipos, trayectos, etc.)
