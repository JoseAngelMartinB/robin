"""Entities for the calibration module."""

import numpy as np
import pandas as pd
import optuna
import yaml

from .constants import HOURS_IN_DAY
from src.robin.kernel.entities import Kernel
from src.robin.supply.entities import Supply

from sklearn.metrics import mean_squared_error
from typing import Union


class Calibration:
    """
    The calibration class optimize the demand hyperparameters to match the target output.
    
    Attributes:
        path_config_supply (str): Path to the supply configuration file.
        path_config_demand (str): Path to the demand configuration file.
        df_target_output (pd.DataFrame): DataFrame with the number of tickets sold for each service.
    """
    
    def __init__(
        self,
        path_config_supply: str,
        path_config_demand: str,
        target_output_path: str,
        seed: Union[int, None] = None
    ) -> None:
        """
        Initialize a calibration object.
        
        Args:
            path_config_supply (str): Path to the supply configuration file.
            path_config_demand (str): Path to the demand configuration file.
            target_output_path (str): File path to the target output CSV file.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.df_target_output = self._get_df_target_output(target_output_path)
        self.arrival_time = np.zeros(HOURS_IN_DAY)
        self.seed = seed
        
    def _get_df_target_output(self, target_output_path: str) -> pd.DataFrame:
        """
        Reads the target output CSV file and returns a DataFrame with the number of tickets sold for each service.
        
        Args:
            target_output_path (str): File path to the target output CSV file.

        Returns:
            pd.DataFrame: DataFrame with 'tickets_sold_target' and 'tickets_sold_prediction'
                columns and service IDs as the index.
        """
        df_target = pd.read_csv(target_output_path)
        df_result = df_target.groupby(by='service').size().to_frame()
        df_result.columns = ['tickets_sold_target']

        services = [service.id for service in Supply.from_yaml(self.path_config_supply).services]
        df_target_output = pd.DataFrame(0, columns=['tickets_sold_target', 'tickets_sold_prediction'], index=services)
        df_target_output.update(df_result)
        return df_target_output
    
    def _suggest_arrival_time(self, trial: optuna.Trial) -> None:
        """
        Suggest arrival time hyperparameters.
        
        It is important to note that the arrival time hyperparameters are normalized to sum to 1.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self.arrival_time = [
            trial.suggest_float(name=f'arrival_time_{hour}', low=0.0, high=1.0) for hour in range(HOURS_IN_DAY)
        ]
        total_arrival_time = sum(self.arrival_time)
        for hour in range(HOURS_IN_DAY):
            self.arrival_time[hour] /= total_arrival_time
            trial.set_user_attr(key=f'arrival_time_{hour}', value=self.arrival_time[hour])

    def create_study(
        self,
        direction: str = 'minimize',
        study_name: Union[int, None] = None,
        storage: Union[str, None] = None,
        n_trials: Union[int, None] = None,
        timeout: Union[int, None] = None,
        show_progress_bar: bool = False
    ) -> None:
        """
        Creates an Optuna study and optimize the demand hyperparameters.
        """
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(
            func=self.optimize,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )

    def objetive_function(self, trial: optuna.Trial) -> float:
        """
        """
        df_temp = pd.read_csv(f'temp_{trial.number}.csv')
        df_temp = df_temp.groupby(by='service').size().to_frame()
        df_temp.columns = ['tickets_sold_prediction']
        self.df_target_output.update(df_temp)
        actual = self.df_target_output['tickets_sold_target'].values
        prediction = self.df_target_output['tickets_sold_prediction'].values
        print(self.df_target_output)
        return mean_squared_error(actual, prediction)

    def optimize(self, trial: optuna.Trial) -> float:
        """
        Optimize the demand hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Objective function value.
        """
        self.suggest_hyperparameters(trial)
        self.simulate(trial)
        return self.objetive_function(trial)
    
    def simulate(self, trial: optuna.Trial) -> None:
        """
        Simulate the demand with the suggested hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        kernel = Kernel(self.path_config_supply, f'temp_{trial.number}.yaml', self.seed)
        kernel.simulate(output_path=f'temp_{trial.number}.csv', departure_time_hard_restriction=True)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> None:
        """
        Suggest demand hyperparameters.
        
        List of hyperparameters:
            - arrival_time_{hour}: Arrival time for each hour of the day.
            - demand_pattern_{day}: Demand pattern for each day of the week.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self._suggest_arrival_time(trial)
        # generate yaml file with the suggested hyperparameters
        with open(self.path_config_demand, 'r') as f:
            demand_yaml = f.read()
        data = yaml.load(demand_yaml, Loader=yaml.CSafeLoader)
        
        arrival_time = {}
        for hour in range(HOURS_IN_DAY):
            arrival_time[str(hour)] = self.arrival_time[hour]
            
        for user_pattern in data['userPattern']:
            user_pattern['arrival_time_kwargs'] = arrival_time
        
        with open(f'temp_{trial.number}.yaml', 'w') as f:
            yaml.dump(data, f, sort_keys=False, Dumper=yaml.CSafeDumper)


if __name__ == '__main__':
    calibration = Calibration(
        path_config_supply='configs/calibration/supply_data.yml',
        path_config_demand='configs/calibration/demand_data.yml',
        target_output_path='data/calibration/target.csv',
        seed=0
    )
    calibration.create_study(study_name='distributed-example', storage='sqlite:///example.db', n_trials=100, show_progress_bar=True)
    # include max number of trials
    # rename temp to checkpoint
