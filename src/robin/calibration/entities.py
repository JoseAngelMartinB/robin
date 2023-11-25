"""Entities for the calibration module."""

import numpy as np
import pandas as pd
import optuna
import os
import yaml

from .constants import HOURS_IN_DAY, LOW_ARRIVAL_TIME, HIGH_ARRIVAL_TIME
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
        calibration_logs: str = 'calibration_logs',
        departure_time_hard_restriction: bool = True,
        seed: Union[int, None] = None
    ) -> None:
        """
        Initialize a calibration object.
        
        Args:
            path_config_supply (str): Path to the supply configuration file.
            path_config_demand (str): Path to the demand configuration file.
            target_output_path (str): File path to the target output CSV file.
            calibration_logs (str): Path to the calibration logs. Defaults to 'calibration_logs'.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not
                be assigned to a service with a departure time that is not valid. Defaults to True.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.df_target_output = self._get_df_target_output(target_output_path)
        self.calibration_logs = self._create_directory(calibration_logs)
        self.departure_time_hard_restriction = departure_time_hard_restriction
        self.arrival_time = np.zeros(HOURS_IN_DAY)
        self.seed = seed
    
    def _create_directory(self, directory: str) -> str:
        """
        Creates a directory if it does not exist.
        
        Args:
            directory (str): Path to the directory.
        
        Returns:
            str: Path to the directory.
        """
        os.makedirs(directory, exist_ok=True)
        return directory
    
    def _get_df_target_output(self, target_output_path: str) -> pd.DataFrame:
        """
        Reads the target output CSV file and returns a DataFrame with the number of tickets sold for each service.
        
        Args:
            target_output_path (str): File path to the target output CSV file.

        Returns:
            pd.DataFrame: DataFrame with 'tickets_sold_target' and 'tickets_sold_prediction'
                columns and service IDs as the index.
        """
        services = [service.id for service in Supply.from_yaml(self.path_config_supply).services]
        df_target_output = pd.DataFrame(0, columns=['tickets_sold_target', 'tickets_sold_prediction'], index=services)
        df_target = pd.read_csv(target_output_path)
        df_result = df_target.groupby(by='service').size().to_frame()
        df_result.columns = ['tickets_sold_target']
        df_target_output.update(df_result)
        return df_target_output

    def _suggest_arrival_time(self, trial: optuna.Trial) -> None:
        """
        Suggest arrival time hyperparameters.
        
        Arrival time hyperparameters are normalized to sum to 1.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self.arrival_time = [
            trial.suggest_float(name=f'arrival_time_{hour}', low=LOW_ARRIVAL_TIME, high=HIGH_ARRIVAL_TIME) for hour in range(HOURS_IN_DAY)
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

    def optimize(self, trial: optuna.Trial) -> float:
        """
        Optimize the demand hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Objective function value.
        """
        trial_directory = self._create_directory(f'{self.calibration_logs}/trial_{trial.number}')
        self.suggest_hyperparameters(trial, trial_directory)
        self.simulate(trial, trial_directory)
        error = self.objetive_function(trial, trial_directory)
        return error
    
    def suggest_hyperparameters(self, trial: optuna.Trial, trial_directory: str) -> None:
        """
        Suggest demand hyperparameters.
        
        List of hyperparameters:
            - arrival_time_{hour}: Arrival time for each hour of the day.
            - demand_pattern_{day}: Demand pattern for each day of the week.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
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
        
        yaml_file = f'{trial_directory}/checkpoint_{trial.number}.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f, sort_keys=False, Dumper=yaml.CSafeDumper)

    def simulate(self, trial: optuna.Trial, trial_directory: str) -> None:
        """
        Simulate the demand with the suggested hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
        """
        kernel = Kernel(
            path_config_supply=self.path_config_supply,
            path_config_demand=f'{trial_directory}/checkpoint_{trial.number}.yaml',
            seed=self.seed
        )
        kernel.simulate(
            output_path=f'{trial_directory}/checkpoint_{trial.number}.csv',
            departure_time_hard_restriction=self.departure_time_hard_restriction
        )

    def _update_df_target_output(self, trial: optuna.Trial, trial_directory: str) -> None:
        """
        Update the target output DataFrame with the predicted number of tickets sold.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
        """
        df_checkpoint = pd.read_csv(f'{trial_directory}/checkpoint_{trial.number}.csv')
        df_checkpoint = df_checkpoint.groupby(by='service').size().to_frame()
        df_checkpoint.columns = ['tickets_sold_prediction']
        self.df_target_output.update(df_checkpoint)

    def objetive_function(self, trial: optuna.Trial, trial_directory: str) -> float:
        """
        Objective function to optimize.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
        
        Returns:
            float: Mean squared error between the target tickets sold and the predicted tickets sold.
        """
        self._update_df_target_output(trial, trial_directory)
        actual = self.df_target_output['tickets_sold_target'].values
        prediction = self.df_target_output['tickets_sold_prediction'].values
        error = mean_squared_error(actual, prediction)
        return error


if __name__ == '__main__':
    calibration = Calibration(
        path_config_supply='configs/calibration/supply_data.yml',
        path_config_demand='configs/calibration/demand_data.yml',
        target_output_path='data/calibration/target.csv',
        seed=0
    )
    calibration.create_study(
        study_name='distributed-example',
        storage='sqlite:///example.db',
        n_trials=100,
        show_progress_bar=True
    )
