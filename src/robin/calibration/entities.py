"""Entities for the calibration module."""

import numpy as np
import optuna
import os
import pandas as pd
import shutil
import yaml

from robin.calibration.constants import (
    CHOICES_CONTINUOUS, CHOICES_DISCRETE, CHOICES_POTENCIAL_DEMAND, DEFAULT_KEEP_TOP_K, LOW_ARRIVAL_TIME, LOW_NORM,
    LOW_PENALTY_UTILITY, LOW_POISSON, LOW_RANDINT, LOW_SEATS_UTILITY, LOW_TSPS_UTILITY, LOW_USER_PATTERN_DISTRIBUTION,
    HIGH_ARRIVAL_TIME, HIGH_NORM, HIGH_PENALTY_UTILITY, HIGH_POISSON, HIGH_RANDINT, HIGH_SEATS_UTILITY,
    HIGH_TSPS_UTILITY, HIGH_USER_PATTERN_DISTRIBUTION
)
from robin.calibration.exceptions import InvalidArrivalTimeDistribution, InvalidPenaltyFunction
from robin.kernel.entities import Kernel
from robin.supply.entities import Supply

from pathlib import Path
from sklearn.metrics import mean_squared_error
from typing import Any, Dict, List, Tuple, Union


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
        top_k_trials (Dict[int, float]): Top k trials with the lowest error.
        keep_top_k (int): Number of top k trials to keep.
        seed (int): Seed for the random number generator.
    """
    
    def __init__(
        self,
        path_config_supply: str,
        path_config_demand: str,
        target_output_path: str,
        calibration_logs: str = 'calibration_logs',
        departure_time_hard_restriction: bool = True,
        keep_top_k: int = DEFAULT_KEEP_TOP_K
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
            keep_top_k (int, optional): Number of top k trials to keep. Defaults to 3.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.df_target_output = self._get_df_target_output(target_output_path)
        self.calibration_logs = self._create_directory(calibration_logs)
        self.departure_time_hard_restriction = departure_time_hard_restriction
        self.top_k_trials = {}
        self.keep_top_k = keep_top_k
    
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
        seed: Union[int, None] = None,
        show_progress_bar: bool = False
    ) -> None:
        """
        Creates an Optuna study and optimize the demand hyperparameters.

        Args:
            direction (str): Direction of the optimization. It can be 'minimize' or 'maximize'. Defaults to 'minimize'.
            study_name (Union[int, None]): Name of the study. Defaults to None.
            storage (Union[str, None]): Storage for the study. Defaults to None.
            n_trials (Union[int, None]): Number of trials to run. Defaults to None.
            timeout (Union[int, None]): Timeout for the study in seconds. Defaults to None.
            seed (int, optional): Seed for the random number generator. Defaults to None.
            show_progress_bar (bool): If True, show a progress bar. Defaults to False.
        """
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(
            func=lambda trial: self.optimize(trial, seed),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )

    def optimize(self, trial: optuna.Trial, seed: Union[int, None] = None) -> float:
        """
        Optimize the demand hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            float: Objective function value.
        """
        trial_directory = self._create_directory(f'{self.calibration_logs}/trial_{trial.number}')
        self.suggest_hyperparameters(trial, trial_directory)
        self.simulate(trial, trial_directory, seed)
        error = self.objetive_function(trial, trial_directory)
        self.keep_top_k_trials(trial, trial_directory, error)
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

    def simulate(self, trial: optuna.Trial, trial_directory: str, seed: Union[int, None] = None) -> None:
        """
        Simulate the demand with the suggested hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        kernel = Kernel(
            path_config_supply=self.path_config_supply,
            path_config_demand=f'{trial_directory}/checkpoint_{trial.number}.yaml',
        )
        kernel.simulate(
            output_path=f'{trial_directory}/checkpoint_{trial.number}.csv',
            seed=seed,
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

    def _replace_highest_error(self, trial: optuna.Trial, trial_directory: str, error: float) -> None:
        """
        Replace the trial with the highest error if the error is lower than the highest one.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
            error (float): Error of the trial.
        """
        errors = np.array(list(self.top_k_trials.values()))
        max_index = np.argmax(errors)
        # Replace if the error is lower than the highest one
        if error < errors[max_index]:
            trial_number = list(self.top_k_trials.keys())[max_index]
            self.top_k_trials[trial.number] = error
            # Delete previous one
            del self.top_k_trials[trial_number]
            shutil.rmtree(f'{self.calibration_logs}/trial_{trial_number}')
        else:
            # Delete current one
            shutil.rmtree(trial_directory)

    def keep_top_k_trials(self, trial: optuna.Trial, trial_directory: str, error: float) -> None:
        """
        Keep the top k trials with the lowest error.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            trial_directory (str): Path to the trial directory.
            error (float): Error of the trial.
        """
        if len(self.top_k_trials) < self.keep_top_k:
            self.top_k_trials[trial.number] = error
        else:
            self._replace_highest_error(trial, trial_directory, error)


class Hyperparameters:
    """
    Hyperparameters to optimize.
    
    Attributes:
        path_config_demand (str): Path to the demand configuration file.
        demand_yaml (Dict[str, Any]): Demand configuration file content.
        arrival_time_kwargs (Dict[str, Dict[str, Union[float, None]]]): Arrival time hyperparameters per user pattern.
        purchase_day (Dict[str, str]): Purchase distribution name per user pattern.
        purchase_day_kwargs (Dict[str, Dict[str, Union[float, None]]]): Purchase day hyperparameters per user pattern.
        seats_utility (Dict[str, List[Dict[str, Union[int, None]]]]): Seats utility hyperparameters per user pattern.
        tsps_utility (Dict[str, List[Dict[str, Union[int, None]]]]): TSPs utility hyperparameters per user pattern.
        penalty_arrival_time (Dict[str, Dict[str, Union[float, None]]]): Arrival time penalty hyperparameters per user pattern.
        penalty_departure_time (Dict[str, Dict[str, Union[float, None]]]): Departure time penalty hyperparameters per user pattern.
        penalty_cost (Dict[str, Dict[str, Union[float, None]]]): Cost penalty hyperparameters per user pattern.
        penalty_travel_time (Dict[str, Dict[str, Union[float, None]]]): Travel time penalty hyperparameters per user pattern.
        error (Dict[str, str]): Error distribution name per user pattern.
        error_kwargs (Dict[str, Dict[str, Union[float, None]]]): Error hyperparameters per user pattern.
        potential_demand (Dict[str, Dict[int, str]]): Potential demand distribution name per demand pattern.
        potential_demand_kwargs (Dict[str, Dict[int, Dict[str, Union[float, None]]]]): Potential demand hyperparameters per demand pattern.
        user_pattern_distribution (Dict[str, Dict[int, Dict[int, float]]]): User pattern distribution per demand pattern.
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
        self.purchase_day, self.purchase_day_kwargs = self._get_purchase_day()
        self.seats_utility = self._get_utility(key='seats')
        self.tsps_utility = self._get_utility(key='train_service_providers')
        self.penalty_arrival_time_kwargs = self._get_penalty_kwargs(penalty_name='arrival_time')
        self.penalty_departure_time_kwargs = self._get_penalty_kwargs(penalty_name='departure_time')
        self.penalty_cost_kwargs = self._get_penalty_kwargs(penalty_name='cost')
        self.penalty_travel_time_kwargs = self._get_penalty_kwargs(penalty_name='travel_time')
        self.error, self.error_kwargs = self._get_error()
        self.potential_demand, self.potential_demand_kwargs = self._get_potential_demand()
        self.user_pattern_distribution = self._get_user_pattern_distribution()
        
    def _get_demand_yaml(self) -> Dict[str, Any]:
        """
        Reads the demand configuration file and returns a string with the content.
        
        Returns:
            Dict[str, Any]: Demand configuration file content.
        """
        with open(self.path_config_demand, 'r') as f:
            data = f.read()
        demand_yaml = yaml.load(data, Loader=yaml.CSafeLoader)
        return demand_yaml
        
    def _get_arrival_time_kwargs(self) -> Dict[str, Dict[str, Union[float, None]]]:
        """
        Get arrival time hyperparameters from the demand configuration file.
        
        Returns:
            Dict[str, Dict[str, Union[float, None]]]: Arrival time hyperparameters per user pattern.
        """
        arrival_time_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            if user_pattern['arrival_time'] != 'hourly':
                raise InvalidArrivalTimeDistribution(distribution_name=user_pattern['arrival_time'])
            arrival_time_kwargs[user_pattern['name']] = user_pattern['arrival_time_kwargs']
        return arrival_time_kwargs
    
    def _get_purchase_day(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Union[float, None]]]]:
        """
        Get purchase day hyperparameters from the demand configuration file.
        
        Returns:
            Tuple[Dict[str, str], Dict[str, Dict[str, Union[float, None]]]]:
                Purchase distribution name and hyperparameters and per purchase day and user pattern.
        """
        purchase_day = {}
        purchase_day_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            purchase_day[user_pattern['name']] = user_pattern['purchase_day']
            purchase_day_kwargs[user_pattern['name']] = user_pattern['purchase_day_kwargs'] or {}
        return purchase_day, purchase_day_kwargs

    def _get_utility(self, key: str) -> Dict[str, List[Dict[str, Union[int, None]]]]:
        """
        Get utility hyperparameters from the demand configuration file.
        
        Args:
            key (str): Key of the utility hyperparameter.
        
        Returns:
            Dict[str, List[Dict[str, Union[int, None]]]]: Utility hyperparameters per user pattern.
        """
        utility = {}
        for user_pattern in self.demand_yaml['userPattern']:
            utility[user_pattern['name']] = user_pattern[key]
        return utility
    
    def _get_penalty_kwargs(self, penalty_name: str) -> Dict[str, Dict[str, Union[float, None]]]:
        """
        Get penalty hyperparameters from the demand configuration file.
        
        Args:
            penalty_name (str): Name of the penalty. It can be 'arrival_time', 'departure_time', 'cost' or 'travel_time'.
        
        Returns:
            Dict[str, Dict[str, Union[float, None]]]: Penalty hyperparameters per user pattern.
        """
        penalty_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            if user_pattern[f'penalty_{penalty_name}'] != 'polynomial':
                raise InvalidPenaltyFunction(function_name=user_pattern[f'penalty_{penalty_name}'])
            penalty_kwargs[user_pattern['name']] = user_pattern[f'penalty_{penalty_name}_kwargs']
        return penalty_kwargs

    def _get_error(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, Union[float, None]]]]:
        """
        Get error hyperparameters from the demand configuration file.
        
        Returns:
            Tuple[Dict[str, str], Dict[str, Dict[str, Union[float, None]]]]:
                Error distribution name and hyperparameters per user pattern.
        """
        error = {}
        error_kwargs = {}
        for user_pattern in self.demand_yaml['userPattern']:
            error[user_pattern['name']] = user_pattern['error']
            error_kwargs[user_pattern['name']] = user_pattern['error_kwargs'] or {}
        return error, error_kwargs

    def _get_potential_demand(self) -> Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, Dict[str, Union[float, None]]]]]:
        """
        Get potential demand hyperparameters from the demand configuration file.
        
        Returns:
            Tuple[Dict[str, Dict[int, str]], Dict[str, Dict[int, Dict[str, Union[float, None]]]]]:
                Potential demand distribution name and hyperparameters per market and demand pattern.
        """
        potential_demand = {}
        potential_demand_kwargs = {}
        for demand_pattern in self.demand_yaml['demandPattern']:
            potential_demand[demand_pattern['name']] = {}
            potential_demand_kwargs[demand_pattern['name']] = {}
            for market in demand_pattern['markets']:
                potential_demand[demand_pattern['name']][market['market']] = market['potential_demand']
                potential_demand_kwargs[demand_pattern['name']][market['market']] = market['potential_demand_kwargs'] or {}
        return potential_demand, potential_demand_kwargs
    
    def _get_user_pattern_distribution(self) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Get user pattern distribution name from the demand configuration file.
        
        Returns:
            Dict[str, Dict[int, Dict[int, float]]]: User pattern distribution per demand pattern.
        """
        user_pattern_distribution = {}
        for demand_pattern in self.demand_yaml['demandPattern']:
            user_pattern_distribution[demand_pattern['name']] = {}
            for market in demand_pattern['markets']:
                user_pattern_distribution[demand_pattern['name']][market['market']] = {}
                for user_pattern in market['user_pattern_distribution']:
                    id_user_pattern = user_pattern['id']
                    percentage = user_pattern['percentage']
                    user_pattern_distribution[demand_pattern['name']][market['market']][id_user_pattern] = percentage
        return user_pattern_distribution

    def _suggest_poisson_kwargs(
        self,
        trial: optuna.Trial,
        pattern: str,
        hyperparameter_name: str,
        distribution_kwargs: Dict[str, float]
    ) -> None:
        """
        Suggest Poisson distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            pattern (str): Name of the pattern.
            hyperparameter_name (str): Name of the hyperparameter.
            distribution_kwargs (Dict[str, float]): Poisson distribution hyperparameters.
        """
        if distribution_kwargs.get('mu') is None:
            distribution_kwargs['mu'] = trial.suggest_float(
                name=f'{pattern}_{hyperparameter_name}_mu',
                low=LOW_POISSON[hyperparameter_name]['mu'],
                high=HIGH_POISSON[hyperparameter_name]['mu']
            )
    
    def _suggest_norm_kwargs(
        self,
        trial: optuna.Trial,
        pattern: str,
        hyperparameter_name: str,
        distribution_kwargs: Dict[str, float],
    ) -> None:
        """
        Suggest normal distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            pattern (str): Name of the pattern.
            hyperparameter_name (str): Name of the hyperparameter.
            distribution_kwargs (Dict[str, float]): Normal distribution hyperparameters.
        """
        if distribution_kwargs.get('loc') is None:
            distribution_kwargs['loc'] = trial.suggest_float(
                name=f'{pattern}_{hyperparameter_name}_loc',
                low=LOW_NORM[hyperparameter_name]['loc'],
                high=HIGH_NORM[hyperparameter_name]['loc']
            )
        if distribution_kwargs.get('scale') is None:
            distribution_kwargs['scale'] = trial.suggest_float(
                name=f'{pattern}_{hyperparameter_name}_scale',
                low=LOW_NORM[hyperparameter_name]['scale'],
                high=HIGH_NORM[hyperparameter_name]['scale']
            )
    
    def _suggest_randint_kwargs(
        self,
        trial: optuna.Trial,
        pattern: str,
        hyperparameter_name: str,
        distribution_kwargs: Dict[str, float],
    ) -> None:
        """
        Suggest randint distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            pattern (str): Name of the pattern.
            hyperparameter_name (str): Name of the hyperparameter.
            distribution_kwargs (Dict[str, float]): Randint distribution hyperparameters.
        """
        if distribution_kwargs.get('low') is None:
            distribution_kwargs['low'] = trial.suggest_int(
                name=f'{pattern}_{hyperparameter_name}_low',
                low=LOW_RANDINT[hyperparameter_name]['low'],
                high=HIGH_RANDINT[hyperparameter_name]['low']
            )
        if distribution_kwargs.get('high') is None:
            distribution_kwargs['high'] = trial.suggest_int(
                name=f'{pattern}_{hyperparameter_name}_high',
                low=LOW_RANDINT[hyperparameter_name]['high'],
                high=HIGH_RANDINT[hyperparameter_name]['high']
            )
    
    def _suggest_distribution(
        self,
        trial: optuna.Trial,
        distribution_name: str,
        pattern: str,
        hyperparameter_name: str,
        distribution_kwargs: Dict[str, float]
    ) -> None:
        """
        Suggest distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            distribution_name (str): Name of the distribution.
            pattern (str): Name of the pattern.
            hyperparameter_name (str): Name of the hyperparameter.
            distribution_kwargs (Dict[str, float]): Distribution hyperparameters.
        """
        args = (trial, pattern, hyperparameter_name, distribution_kwargs)
        if distribution_name == 'poisson':
            self._suggest_poisson_kwargs(*args)
        elif distribution_name == 'norm':
            self._suggest_norm_kwargs(*args)
        elif distribution_name == 'randint':
            self._suggest_randint_kwargs(*args) 
 
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
    
    def suggest_purchase_day(self, trial: optuna.Trial) -> None:
        """
        Suggest purchase day hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, purchase_day in self.purchase_day.items():
            # Suggestions are only made for None values
            if purchase_day is None:
                self.purchase_day[user_pattern] = trial.suggest_categorical(
                    name=f'{user_pattern}_purchase_day',
                    choices=CHOICES_DISCRETE
                )       

    def suggest_purchase_day_kwargs(self, trial: optuna.Trial) -> None:
        """
        Suggest purchase day hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, purchase_day_kwargs in self.purchase_day_kwargs.items():
            self._suggest_distribution(
                trial=trial,
                distribution_name=self.purchase_day[user_pattern],
                pattern=user_pattern,
                hyperparameter_name='purchase_day_kwargs',
                distribution_kwargs=purchase_day_kwargs
            )
    
    def _suggest_utility(
        self,
        trial: optuna.Trial,
        hyperparameter_name: str,
        utility_dict: Dict[str, List[Dict[str, Union[int, None]]]],
        low: int,
        high: int
    ) -> None:
        """
        Suggest seats utility hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            hyperparameter_name (str): Name of the hyperparameter.
            utility_dict (Dict[str, List[Dict[str, Union[int, None]]]]): Utility hyperparameters.
            low (int): Low value for the utility.
            high (int): High value for the utility.
        """
        for user_pattern, utility in utility_dict.items():
            for i, key in enumerate(utility):
                _id = key['id']
                value = key['utility']
                # Suggestions are only made for None values
                if value is None:
                    utility[i]['utility'] = trial.suggest_int(
                        name=f'{user_pattern}_{hyperparameter_name}_utility_{_id}',
                        low=low,
                        high=high
                    )

    def suggest_utility(self, trial: optuna.Trial) -> None:
        """
        Suggest utility hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self._suggest_utility(trial, 'seats', self.seats_utility, LOW_SEATS_UTILITY, HIGH_SEATS_UTILITY)
        self._suggest_utility(trial, 'train_service_providers', self.tsps_utility, LOW_TSPS_UTILITY, HIGH_TSPS_UTILITY)

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

    def suggest_error(self, trial: optuna.Trial) -> None:
        """
        Suggest error hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, error in self.error.items():
            # Suggestions are only made for None values
            if error is None:
                self.error[user_pattern] = trial.suggest_categorical(
                    name=f'{user_pattern}_error',
                    choices=CHOICES_CONTINUOUS
                )
    
    def suggest_error_kwargs(self, trial: optuna.Trial) -> None:
        """
        Suggest error hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for user_pattern, error_kwargs in self.error_kwargs.items():
            self._suggest_distribution(
                trial=trial,
                distribution_name=self.error[user_pattern],
                pattern=user_pattern,
                hyperparameter_name='error_kwargs',
                distribution_kwargs=error_kwargs
            )

    def suggest_potential_demand(self, trial: optuna.Trial) -> None:
        """
        Suggest potential demand hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for demand_pattern, potential_demand in self.potential_demand.items():
            for market, value in potential_demand.items():
                # Suggestions are only made for None values
                if value is None:
                    self.potential_demand[demand_pattern][market] = trial.suggest_categorical(
                        name=f'{demand_pattern}_{market}_potential_demand',
                        choices=CHOICES_POTENCIAL_DEMAND
                    )
    
    def suggest_potential_demand_kwargs(self, trial: optuna.Trial) -> None:
        """
        Suggest potential demand hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for demand_pattern, potential_demand_kwargs in self.potential_demand_kwargs.items():
            for market, value in potential_demand_kwargs.items():
                self._suggest_distribution(
                    trial=trial,
                    distribution_name=self.potential_demand[demand_pattern][market],
                    pattern=demand_pattern,
                    hyperparameter_name=f'{market}_potential_demand_kwargs',
                    distribution_kwargs=value
                )

    def suggest_user_pattern_distribution(self, trial: optuna.Trial) -> None:
        """
        Suggest user pattern distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        for demand_pattern, user_pattern_distribution in self.user_pattern_distribution.items():
            for market, user_pattern_distribution in user_pattern_distribution.items():
                for user_pattern, percentage in user_pattern_distribution.items():
                    # Suggestions are only made for None values
                    if percentage is None:
                        user_pattern_distribution[user_pattern] = trial.suggest_float(
                            name=f'{demand_pattern}_{market}_user_pattern_distribution_{user_pattern}',
                            low=LOW_USER_PATTERN_DISTRIBUTION,
                            high=HIGH_USER_PATTERN_DISTRIBUTION
                        )
                # Normalize user pattern distribution hyperparameters to sum to 1
                total_percentage = sum(user_pattern_distribution.values())
                for id_user_pattern in user_pattern_distribution:
                    user_pattern_distribution[id_user_pattern] /= total_percentage
                    trial.set_user_attr(
                        key=f'{demand_pattern}_{market}_user_pattern_distribution_{user_pattern}_{id_user_pattern}',
                        value=user_pattern_distribution[id_user_pattern]
                    )

    def _suggest_user_patterns(self, trial: optuna.Trial) -> None:
        """
        Suggestions for hyperparameters per user pattern.
        
        List of hyperparameters:
            - {user_pattern}_arrival_time_{hour}: Arrival time for each hour of the day.
            - {user_pattern}_purchase_day: Purchase day distribution name.
            - {user_pattern}_purchase_day_kwargs: Purchase day distribution hyperparameters.
            - {user_pattern}_seats_utility_{seat_id}: Seats utility for each seat.
            - {user_pattern}_train_service_providers_utility_{tsp_id}: TSPs utility for each TSP.
            - {user_pattern}_penalty_arrival_time_{beta}: Arrival time penalty for each beta.
            - {user_pattern}_penalty_departure_time_{beta}: Departure time penalty for each beta.
            - {user_pattern}_penalty_cost_{beta}: Cost penalty for each beta.
            - {user_pattern}_penalty_travel_time_{beta}: Travel time penalty for each beta.
            - {user_pattern}_error: Error distribution name.
            - {user_pattern}_error_kwargs: Error distribution hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self.suggest_arrival_time_kwargs(trial)
        self.suggest_purchase_day(trial)
        self.suggest_purchase_day_kwargs(trial)
        self.suggest_utility(trial)
        self.suggest_penalty_kwargs(trial)
        self.suggest_error(trial)
        self.suggest_error_kwargs(trial)

    def _suggest_demand_patterns(self, trial: optuna.Trial) -> None:
        """
        Suggestions for hyperparameters per demand pattern.
        
        List of hyperparameters:
            - {demand_pattern}_{market}_potential_demand: Potential demand distribution name.
            - {demand_pattern}_{market}_potential_demand_kwargs: Potential demand distribution hyperparameters.
            - {demand_pattern}_{market}_user_pattern_distribution_{user_pattern}_{id_user_pattern}: User pattern distribution for each user pattern.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self.suggest_potential_demand(trial)
        self.suggest_potential_demand_kwargs(trial)
        self.suggest_user_pattern_distribution(trial)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> None:
        """
        Suggestions for all hyperparameters, including user patterns and demand patterns.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
        """
        self._suggest_user_patterns(trial)
        self._suggest_demand_patterns(trial)
    
    def _update_user_patterns(self) -> None:
        """
        Update the user patterns with the suggested hyperparameters.
        """
        for user_pattern in self.demand_yaml['userPattern']:
            user_pattern['arrival_time_kwargs'] = self.arrival_time_kwargs[user_pattern['name']]
            user_pattern['purchase_day'] = self.purchase_day[user_pattern['name']]
            user_pattern['purchase_day_kwargs'] = self.purchase_day_kwargs[user_pattern['name']]
            user_pattern['seats'] = self.seats_utility[user_pattern['name']]
            user_pattern['penalty_arrival_time_kwargs'] = self.penalty_arrival_time_kwargs[user_pattern['name']]
            user_pattern['penalty_departure_time_kwargs'] = self.penalty_departure_time_kwargs[user_pattern['name']]
            user_pattern['penalty_cost_kwargs'] = self.penalty_cost_kwargs[user_pattern['name']]
            user_pattern['penalty_travel_time_kwargs'] = self.penalty_travel_time_kwargs[user_pattern['name']]
            user_pattern['error'] = self.error[user_pattern['name']]
            user_pattern['error_kwargs'] = self.error_kwargs[user_pattern['name']]
    
    def _update_demand_patterns(self) -> None:
        """
        Update the demand patterns with the suggested hyperparameters.
        """
        for demand_pattern in self.demand_yaml['demandPattern']:
            for market in demand_pattern['markets']:
                market['potential_demand'] = self.potential_demand[demand_pattern['name']][market['market']]
                market['potential_demand_kwargs'] = self.potential_demand_kwargs[demand_pattern['name']][market['market']]
                for user_pattern in market['user_pattern_distribution']:
                    id_user_pattern = user_pattern['id']
                    user_pattern['percentage'] = self.user_pattern_distribution[demand_pattern['name']][market['market']][id_user_pattern]
    
    def _update_demand_yaml(self) -> None:
        """
        Update the demand configuration file with the suggested hyperparameters.
        """
        self._update_user_patterns()
        self._update_demand_patterns()
    
    def save_demand_yaml(self, output_path: str) -> None:
        """
        Save the demand configuration file.
        
        Args:
            output_path (str): Path to the output demand configuration file.
        """
        self._update_demand_yaml()
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as file:
            yaml.dump(self.demand_yaml, file, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True)
