"""Entities for the labs module."""

import copy
import os
import numpy as np
import pandas as pd
import random
import shutil
import yaml

from robin.kernel.entities import Kernel
from robin.supply.entities import Supply
from robin.demand.entities import Demand
from robin.labs.utils import (
    get_file_key, get_passenger_status, get_tickets_by_date_user_seat, get_tickets_by_pair_seat, get_pairs_sold
)
from robin.plotter.utils import plot_series

from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
from typing import List, Mapping


# TODO: Inherit from KernelPlotter and delete utils methods
class Labs:
    """
    Robin Lab class.

    This class is intended to run Robin simulation experiments.
    """

    def __init__(
        self,
        path_config_supply: str,
        path_config_demand: str,
        tmp_path: str,
        verbose=0
    ) -> None:
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
        self.default_supply = Supply.from_yaml(path_config_supply)
        self.default_demand = Demand.from_yaml(path_config_demand)
        self.stations_dict = self.default_supply.get_stations_dict()
        self.tmp_path = tmp_path
        self.lab_config = None
        self.verbose = verbose

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
            raise ValueError('At least one of the lab configs must be non-empty.')

        if self.lab_config['supply']:  # Check if it's a supply or demand experiment
            self._supply_yaml_editor()
        else:
            raise ValueError('At least one of the lab configs must be non-empty.')

    def _create_tmp_dir(self):
        """
        Create temporary directory and folders.
        """
        if not self.tmp_path.exists():
            os.makedirs(self.tmp_path)

        if not os.path.exists(self.tmp_path / 'supply'):
            os.makedirs(self.tmp_path / 'supply')

        if not os.path.exists(self.tmp_path / 'demand'):
            os.makedirs(self.tmp_path / 'demand')

        if not os.path.exists(self.tmp_path / 'output'):
            os.makedirs(self.tmp_path / 'output')

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

        # TODO: Temporary fix only to test prices
        supply_lab_config = self.lab_config['supply']
        arange_args = supply_lab_config['prices']
        for i, factor in enumerate(tqdm(np.arange(**arange_args)), start=1):
            modified_data = copy.deepcopy(original_data)
            modified_services = modified_data.get('service')
            assert modified_services, 'No services found in the supply config.'
            for service in modified_services:
                modify_prices(service['origin_destination_tuples'], factor)

            modified_data['service'] = modified_services
            supply_file_name = f'supply_{i}.yaml'
            save_path_supply = f'{self.tmp_path}/supply/{supply_file_name}'

            with open(save_path_supply, 'w') as file:
                yaml.safe_dump(modified_data, file)

            shutil.copy(self.path_config_demand, self.tmp_path / 'demand' / f'demand_{i}.yaml')

    def simulate(self, runs: int = 1) -> None:
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
            return int(file_name.split('_')[-1])

        # Run simulation for each supply file
        sorted_supply_files = sorted(os.listdir(self.tmp_path / 'supply'), key=file_number)
        sorted_demand_files = sorted(os.listdir(self.tmp_path / 'demand'), key=file_number)
        for r in tqdm(range(runs), desc='Runs: '):
            seed = random.randint(0, 1000000)
            print(f'Seed used run {r}: {seed}')
            for i, supply_file, demand_file in zip(tqdm(range(1, len(sorted_supply_files) + 1),
                                                        desc='Iters: ',
                                                        leave=True),
                                                   sorted_supply_files,
                                                   sorted_demand_files):
                kernel = Kernel(path_config_supply=self.tmp_path / 'supply' / supply_file,
                                path_config_demand=self.tmp_path / 'demand' / demand_file)
                kernel.simulate(output_path=Path(f'{self.tmp_path}/output/output_{r}_{i}.csv'),
                                seed=seed,
                                calculate_global_utility=True)

    def _get_tickets_sold(self) -> Mapping:
        """Get the number of tickets sold for each supply file."""
        tickets_sold = {}
        output_files = sorted(os.listdir(self.tmp_path / 'output'), key=lambda x: int(x.split('.')[0].split('_')[-1]))
        for _, output_file in zip(tqdm(range(1, len(output_files) + 1)), output_files):
            df = pd.read_csv(self.tmp_path / 'output' / output_file)
            buffer_tickets_sold = df.groupby(by=['seat']).size().to_dict()
            buffer_tickets_sold['Total'] = sum(buffer_tickets_sold.values())
            tickets_sold[output_file.split('.')[0].split('_')[-1]] = buffer_tickets_sold

        return tickets_sold

    def plot_seat_elasticity_curve(self, save_path: str = None) -> None:
        """Plot the seat elasticity curve."""
        tickets_sold = self._get_tickets_sold()

        # TODO: Temporary fix only to test prices
        supply_lab_config = self.lab_config['supply']
        arange_args = supply_lab_config['prices']
        x_data = np.arange(**arange_args)
        series_keys = set(key for value in tickets_sold.values() for key in value)
        series = {key: [] for key in series_keys}

        for file_key in tickets_sold:
            for seat_key in series_keys:
                series[seat_key].append(tickets_sold[file_key].get(seat_key, 0))

        fig, ax = plot_series(x_data=tuple(x_data),
                              y_data=series,
                              title='Prices elasticity curve (seat types)',
                              xlabel='Price increase (%)',
                              ylabel='Tickets sold',
                              xticks=x_data[::3],
                              xticks_labels=tuple([f'{x:.0f}%' for x in x_data][::3]),
                              figsize=(10, 6))

        plt.show()

        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)

    def plot_demand_status(self, save_path: str = None) -> None:
        """Plot the demand status for different user categories over time."""
        demand_status = self._get_demand_status()

        time_periods = demand_status.keys()
        user_categories = ["User found \nany service that\nmet his needs\nbut couldn't purchase.",
                           'User bought\na service which\nwas not the one\nwith the best utility.',
                           'User bought\nthe ticket with\nbest utility.',
                           "User didn't find\nany ticket\nthat met his needs."]

        series = {category: [] for category in user_categories}

        for period in time_periods:
            user_data, _ = demand_status[period]
            for category in user_categories:
                series[category].append(user_data.get(user_categories.index(category), 0))

        # TODO: Temporary fix only to test prices
        supply_lab_config = self.lab_config["supply"]
        arange_args = supply_lab_config["prices"]
        x_data = np.arange(**arange_args)

        fig, ax = plot_series(x_data=tuple(x_data),
                              y_data=series,
                              title='User Demand Status Over Time',
                              xlabel='Price increase (%)',
                              ylabel='Number of Users',
                              xticks=x_data[::3],
                              xticks_labels=tuple([f'{x:.0f}%' for x in x_data][::3]),
                              figsize=(10, 6))

        plt.show()

        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)

    def _get_demand_status(self):
        file_number = lambda file: int(Path(file).stem.split('_')[-1])
        output_files = sorted(os.listdir(self.tmp_path / 'output'), key=file_number)
        supply_files = sorted(os.listdir(self.tmp_path / 'supply'), key=file_number)
        demand_files = sorted(os.listdir(self.tmp_path / 'demand'), key=file_number)

        passenger_status = {}
        for i, supply_file, demand_file, output_file in zip(tqdm(range(1, len(output_files) + 1)),
                                                            supply_files,
                                                            demand_files,
                                                            output_files):
            output = pd.read_csv(self.tmp_path / 'output' / output_file,
                                 dtype={'departure_station': str, 'arrival_station': str})
            passenger_status[i] = get_passenger_status(output)

        return passenger_status

    def plot_markets(self, save_path: str = None) -> None:
        """Plot the sum of tickets sold for different market routes over time."""
        markets_data = self._get_markets_data()

        series = {}
        markets = tuple(set(key for value in markets_data.values() for key in value))
        for period, market_data in markets_data.items():
            for market in markets:
                market_values = market_data.get(market)
                result = sum(market_values.values()) if market_values else 0
                if market not in series:
                    series[market] = []
                series[market].append(result)

        # TODO: Temporary fix only to test prices
        supply_lab_config = self.lab_config['supply']
        arange_args = supply_lab_config['prices']
        x_data = np.arange(**arange_args)

        fig, ax = plot_series(x_data=tuple(x_data),
                              y_data=series,
                              title='User Demand Status Over Time',
                              xlabel='Price increase (%)',
                              ylabel='Number of Users',
                              xticks=x_data[::3],
                              xticks_labels=tuple([f'{x:.0f}%' for x in x_data][::3]),
                              figsize=(10, 6))

        plt.show()

        ax.set_title('Sum of Tickets Sold for Different Market Routes Over Time')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Total Number of Tickets Sold')
        ax.legend()

        plt.show()

        if save_path:
            fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)

    def _get_markets_data(self):
        file_number = lambda file: int(Path(file).stem.split('_')[-1])
        output_files = sorted(os.listdir(self.tmp_path / 'output'), key=file_number)
        supply_files = sorted(os.listdir(self.tmp_path / 'supply'), key=file_number)
        demand_files = sorted(os.listdir(self.tmp_path / 'demand'), key=file_number)

        tickets_by_pair_seat = {}
        pairs_sold = {}
        for i, supply_file, demand_file, output_file in zip(tqdm(range(1, len(output_files) + 1)),
                                                            supply_files,
                                                            demand_files,
                                                            output_files):
            output = pd.read_csv(self.tmp_path / 'output' / output_file,
                                 dtype={'departure_station': str, 'arrival_station': str})
            tickets_by_pair_seat[i] = get_tickets_by_pair_seat(output, self.stations_dict)
            pairs_sold[i] = get_pairs_sold(output, self.stations_dict)

        return tickets_by_pair_seat

    def get_markets_df(self, output_files: List[str]):
        """
        Get the dataframe for the markets plot.

        Args:
            output_files (List[str]): List of output files.

        Returns:
            pd.DataFrame: Dataframe for the markets plot.
        """
        df_markets = pd.DataFrame(columns=['run', 'iter', 'tickets_sold', 'trip'])

        for i, output_file in zip(tqdm(range(1, len(output_files) + 1)), output_files):
            run, iter_num = get_file_key(output_file)
            output = pd.read_csv(self.tmp_path / 'output' / output_file,
                                 dtype={'departure_station': str, 'arrival_station': str})
            tickets_by_pair_seat = get_tickets_by_pair_seat(output, self.stations_dict)
            total_by_trip = {}
            for seat in tickets_by_pair_seat:
                for trip in tickets_by_pair_seat[seat]:
                    if trip not in total_by_trip:
                        total_by_trip[trip] = 0
                    total_by_trip[trip] += tickets_by_pair_seat[seat][trip]

            pairs_sold = get_pairs_sold(output, self.stations_dict)
            for trip in pairs_sold:
                df_m_row = [run, iter_num, pairs_sold[trip], trip]
                df_markets.loc[len(df_markets)] = df_m_row

        return df_markets

    def get_tickets_seat_df(self, output_files: List[str]):
        """
        Get the dataframe for the tickets seat plot.

        Args:
            output_files (List[str]): List of output files.

        Returns:
            pd.DataFrame: Dataframe for the tickets seat plot.
        """
        df_tickets_seat = pd.DataFrame(columns=['run', 'iter', 'tickets_sold', 'seat_type'])

        for i, output_file in zip(tqdm(range(1, len(output_files) + 1)), output_files):
            run, iter_num = get_file_key(output_file)
            output = pd.read_csv(self.tmp_path / 'output' / output_file,
                                 dtype={'departure_station': str, 'arrival_station': str})
            tickets_by_date_user_seat = get_tickets_by_date_user_seat(output)

            total_by_seat = {}
            for purchase_date in tickets_by_date_user_seat:
                for user in tickets_by_date_user_seat[purchase_date]:
                    for seat in tickets_by_date_user_seat[purchase_date][user]:
                        if seat not in total_by_seat:
                            total_by_seat[seat] = 0
                        total_by_seat[seat] += tickets_by_date_user_seat[purchase_date][user][seat]

            total_by_seat['Total'] = sum(total_by_seat.values())
            for seat in total_by_seat:
                df_ts_row = [run, iter_num, total_by_seat[seat], seat]
                df_tickets_seat.loc[len(df_tickets_seat)] = df_ts_row

        return df_tickets_seat

    def get_demand_status_df(self, output_files: List[str]):
        """
        Get the dataframe for the demand status plot.

        Args:
            output_files (List[str]): List of output files.

        Returns:
            pd.DataFrame: Dataframe for the demand status plot.
        """
        df_demand_status = pd.DataFrame(columns=['run', 'iter', 'users', 'status'])

        for i, output_file in zip(tqdm(range(1, len(output_files) + 1)), output_files):
            run, iter_num = get_file_key(output_file)
            output = pd.read_csv(self.tmp_path / 'output' / output_file,
                                 dtype={'departure_station': str, 'arrival_station': str})
            demand_status_dict, labels = get_passenger_status(output)
            for k, v in demand_status_dict.items():
                df_ds_row = [run, iter_num, v, labels[k]]
                df_demand_status.loc[len(df_demand_status)] = df_ds_row

        return df_demand_status

    def get_sns_dfs(self):
        """
        Get the dataframes for seaborn plots.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Dataframes for seaborn plots.
        """
        output_files = sorted(os.listdir(self.tmp_path / 'output'), key=get_file_key)

        df_markets = self.get_markets_df(output_files=output_files)
        df_tickets_seat = self.get_tickets_seat_df(output_files=output_files)
        df_demand_status = self.get_demand_status_df(output_files=output_files)

        return df_markets, df_tickets_seat, df_demand_status

    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
