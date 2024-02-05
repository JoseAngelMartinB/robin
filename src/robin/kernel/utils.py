import numpy as np

from src.robin.demand.entities import Passenger

from typing import Union


def get_constrain_value(passenger: Passenger,
                        variable_name: str,
                        ) -> Union[float, np.NaN]:
    """
    Get the maximum value of a variable in the user pattern rules.

    Args:
        passenger: Passenger object.
        variable_name: Name of the variable.

    Returns:
        The maximum value of the variable in the user pattern.
    """
    max_value = 1.0
    for rule in passenger.user_pattern.behaviour_rules:
        for proposition in rule.antecedent.proposiciones:
            if proposition.variable.name == variable_name:
                max_value = max(proposition.term.values)
    return max_value
