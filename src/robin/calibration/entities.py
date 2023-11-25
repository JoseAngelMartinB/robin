"""Entities for the calibration module."""

import pandas as pd
import optuna
import os
import yaml

from .constants import *
from .exceptions import InvalidArrivalTimeDistribution, InvalidPenaltyFunction
from src.robin.kernel.entities import Kernel
from src.robin.supply.entities import Supply

from sklearn.metrics import mean_squared_error
from typing import Any, Dict, List, Union


class Calibration:
    """
    The calibration class optimize the demand hyperparameters to match the target output.
    
    Attributes:
        path_config_supply (str): Path to the supply configuration file.
        path_config_demand (str): Path to the demand configuration file.
        df_target_output (pd.DataFrame): DataFrame with the number of tickets sold for each service.
        calibration_logs (str): Path to the calibration logs.
        departure_time_hard_restriction (bool): If True, the passenger will not be assigned to a service
            with a departure time that is not valid.
        seed (int): Seed for the random number generator.
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
        Suggest demand hyperparameters and save them to a YAML file.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
        """
        hyperparameters = Hyperparameters(self.path_config_demand)
        hyperparameters.suggest_hyperparameters(trial)
        hyperparameters.save_demand_yaml(f'{trial_directory}/checkpoint_{trial.number}.yaml')

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

    def _save_df_target_output(self, trial: optuna.Trial, trial_directory: str) -> None:
        """
        Save the target output DataFrame.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
        """
        self.df_target_output.to_csv(f'{trial_directory}/df_target_output_{trial.number}.csv', index_label='service')

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
        self._save_df_target_output(trial, trial_directory)
        actual = self.df_target_output['tickets_sold_target'].values
        prediction = self.df_target_output['tickets_sold_prediction'].values
        error = mean_squared_error(actual, prediction)
        return error


class Hyperparameters:
    """
    Hyperparameters to optimize.
    
    Attributes:
        path_config_demand (str): Path to the demand configuration file.
        demand_yaml (Dict[str, Any]): Demand configuration file content.
        arrival_time_kwargs (Dict[str, Dict[str, Union[float, None]]]): Arrival time hyperparameters per user pattern.
        seats_utility (Dict[str, List[Dict[str, Union[int, None]]]]): Seats utility hyperparameters per user pattern.
        penalty_arrival_time (Dict[str, Dict[str, Union[float, None]]]): Arrival time penalty hyperparameters per user pattern.
        penalty_departure_time (Dict[str, Dict[str, Union[float, None]]]): Departure time penalty hyperparameters per user pattern.
        penalty_cost (Dict[str, Dict[str, Union[float, None]]]): Cost penalty hyperparameters per user pattern.
        penalty_travel_time (Dict[str, Dict[str, Union[float, None]]]): Travel time penalty hyperparameters per user pattern.
    """
    
    def __init__(self, path_config_demand: str) -> None:
        """
        Initialize a hyperparameters object.
        
        Args:
            path_config_demand (str): Path to the demand configuration file.
        """
        self.path_config_demand = path_config_demand
        self.demand_yaml = self._get_demand_yaml()
        self.arrival_time_kwargs = self._get_arrival_time_kwargs()
        self.seats_utility = self._get_seats_utility()
        self.penalty_arrival_time_kwargs = self._get_penalty_kwargs(penalty_name='arrival_time')
        self.penalty_departure_time_kwargs = self._get_penalty_kwargs(penalty_name='departure_time')
        self.penalty_cost_kwargs = self._get_penalty_kwargs(penalty_name='cost')
        self.penalty_travel_time_kwargs = self._get_penalty_kwargs(penalty_name='travel_time')
        
    def _get_demand_yaml(self) -> Dict[str, Any]:
        """
        Reads the demand configuration file and returns a string with the content.
        
        Returns:
            str: Demand configuration file content.
        """
        with open(self.path_config_demand, 'r') as f:
            data = f.read()
        demand_yaml = yaml.load(data, Loader=yaml.CSafeLoader)
        return demand_yaml
        
    def _get_arrival_time_kwargs(self) -> Dict[str, Dict[str, Union[float, None]]]:
        """
        Get arrival time hyperparameters from the demand configuration file.
        """
        arrival_time_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            if user_pattern['arrival_time'] != 'custom_arrival_time':
                raise InvalidArrivalTimeDistribution(distribution_name=user_pattern['arrival_time'])
            arrival_time_kwargs[user_pattern['name']] = user_pattern['arrival_time_kwargs']
        return arrival_time_kwargs
    
    def _get_seats_utility(self) -> Dict[str, List[Dict[str, Union[int, None]]]]:
        """
        Get seats utility hyperparameters from the demand configuration file.
        """
        seats_utility = {}
        for user_pattern in self.demand_yaml['userPattern']:
            seats_utility[user_pattern['name']] = user_pattern['seats']
        return seats_utility
    
    def _get_penalty_kwargs(self, penalty_name: str) -> Dict[str, Dict[str, Union[float, None]]]:
        """
        Get penalty hyperparameters from the demand configuration file.
        
        Args:
            penalty_name (str): Name of the penalty. It can be 'arrival_time', 'departure_time', 'cost' or 'travel_time'.
        """
        penalty_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            if user_pattern[f'penalty_{penalty_name}'] != 'polynomial':
                raise InvalidPenaltyFunction(function_name=user_pattern[f'penalty_{penalty_name}'])
            penalty_kwargs[user_pattern['name']] = user_pattern[f'penalty_{penalty_name}_kwargs']
        return penalty_kwargs
    
    def suggest_arrival_time_kwargs(self, trial: optuna.Trial) -> None:
        """
        Suggest arrival time hyperparameters.
        
        Arrival time hyperparameters are normalized to sum to 1.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, arrival_time_kwargs in self.arrival_time_kwargs.items():
            for hour, value in arrival_time_kwargs.items():
                # Suggestions are only made for None values
                if value is None:
                    arrival_time_kwargs[hour] = trial.suggest_float(
                        name=f'{user_pattern}_arrival_time_{hour}',
                        low=LOW_ARRIVAL_TIME,
                        high=HIGH_ARRIVAL_TIME
                    )
            # Normalize arrival time hyperparameters to sum to 1
            total_arrival_time = sum(arrival_time_kwargs.values())
            for hour in range(len(arrival_time_kwargs)):
                hour = str(hour)
                arrival_time_kwargs[hour] /= total_arrival_time
                trial.set_user_attr(
                    key=f'{user_pattern}_arrival_time_{hour}',
                    value=arrival_time_kwargs[hour]
                )
    
    def suggest_seats_utility(self, trial: optuna.Trial) -> None:
        """
        Suggest seats utility hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, seats_utility in self.seats_utility.items():
            for i, seat in enumerate(seats_utility):
                seat_id = seat['id']
                value = seat['utility']
                # Suggestions are only made for None values
                if value is None:
                    seats_utility[i]['utility'] = trial.suggest_int(
                        name=f'{user_pattern}_seats_utility_{seat_id}',
                        low=LOW_SEATS_UTILITY,
                        high=HIGH_SEATS_UTILITY
                    )

    def _suggest_penalty_kwargs(
        self,
        trial: optuna.Trial,
        penalty_name: str,
        penalty_kwargs: Dict[str, Dict[str, Union[float, None]]]
    ) -> None:
        """
        Suggest penalty hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            penalty_kwargs (Dict[str, Dict[str, Union[float, None]]]): Penalty hyperparameters.
        """
        for user_pattern, penalty_kwargs in penalty_kwargs.items():
            for beta, value in penalty_kwargs.items():
                # Suggestions are only made for None values
                if value is None:
                    penalty_kwargs[beta] = trial.suggest_float(
                        name=f'{user_pattern}_penalty_{penalty_name}_{beta}',
                        low=LOW_PENALTY_UTILITY,
                        high=HIGH_PENALTY_UTILITY
                    )
    
    def suggest_penalty_kwargs(self, trial: optuna.Trial) -> None:
        """
        Suggest penalty hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        penalties = [
            ('arrival_time', self.penalty_arrival_time_kwargs),
            ('departure_time', self.penalty_departure_time_kwargs),
            ('cost', self.penalty_cost_kwargs),
            ('travel_time', self.penalty_travel_time_kwargs)
        ]
        for penalty_name, penalty_kwargs in penalties:
            self._suggest_penalty_kwargs(trial, penalty_name, penalty_kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> None:
        """
        Suggestions for all hyperparameters.
        
        List of hyperparameters:
            - {user_pattern}_arrival_time_{hour}: Arrival time for each hour of the day.
            - {user_pattern}_seats_utility_{seat_id}: Seats utility for each seat.
            - {user_pattern}_penalty_arrival_time_{beta}: Arrival time penalty for each beta.
            - {user_pattern}_penalty_departure_time_{beta}: Departure time penalty for each beta.
            - {user_pattern}_penalty_cost_{beta}: Cost penalty for each beta.
            - {user_pattern}_penalty_travel_time_{beta}: Travel time penalty for each beta.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self.suggest_arrival_time_kwargs(trial)
        self.suggest_seats_utility(trial)
        self.suggest_penalty_kwargs(trial)
    
    def _update_user_patterns(self) -> None:
        """
        Update the user patterns with the suggested hyperparameters.
        """
        for user_pattern in self.demand_yaml['userPattern']:
            user_pattern['arrival_time_kwargs'] = self.arrival_time_kwargs[user_pattern['name']]
            user_pattern['seats'] = self.seats_utility[user_pattern['name']]
            user_pattern['penalty_arrival_time_kwargs'] = self.penalty_arrival_time_kwargs[user_pattern['name']]
            user_pattern['penalty_departure_time_kwargs'] = self.penalty_departure_time_kwargs[user_pattern['name']]
            user_pattern['penalty_cost_kwargs'] = self.penalty_cost_kwargs[user_pattern['name']]
            user_pattern['penalty_travel_time_kwargs'] = self.penalty_travel_time_kwargs[user_pattern['name']]
    
    def _update_demand_patterns(self) -> None:
        pass
    
    def _update_demand_yaml(self) -> None:
        """
        Update the demand configuration file with the suggested hyperparameters.
        """
        self._update_user_patterns()
        self._update_demand_patterns()
    
    def save_demand_yaml(self, path: str) -> None:
        """
        Save the demand configuration file.
        
        Args:
            path (str): Path to the demand configuration file.
        """
        self._update_demand_yaml()
        with open(path, 'w') as f:
            yaml.dump(self.demand_yaml, f, sort_keys=False, Dumper=yaml.CSafeDumper)


if __name__ == '__main__':
    calibration = Calibration(
        path_config_supply='configs/calibration/supply_data.yml',
        path_config_demand='configs/calibration/demand_data.yml',
        target_output_path='data/calibration/target.csv',
        seed=0
    )
    calibration.create_study(
        study_name='calibration_test',
        storage='sqlite:///calibration_test.db',
        n_trials=100,
        show_progress_bar=True
    )
    # purchase day, que es None, pues suggest todo
    # que nos dan la distrubucion, pues usamos esa, y los parámetros que nos den
