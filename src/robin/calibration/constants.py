"""Constants for the ROBIN calibration module."""

DEFAULT_KEEP_TOP_K = 3

LOW_ARRIVAL_TIME = 0.0
HIGH_ARRIVAL_TIME = 1.0

LOW_SEATS_UTILITY = 1
HIGH_SEATS_UTILITY = 40

LOW_PENALTY_UTILITY = 0.0
HIGH_PENALTY_UTILITY = 1.0

LOW_USER_PATTERN_DISTRIBUTION = 0.0
HIGH_USER_PATTERN_DISTRIBUTION = 1.0

CHOICES_DISCRETE = ['randint', 'poisson']
CHOICES_CONTINUOUS = ['norm']
CHOICES_POTENCIAL_DEMAND = ['randint']

LOW_POISSON = {
    'purchase_day_kwargs': {
        'mu': 0.0
    }
}
HIGH_POISSON = {
    'purchase_day_kwargs': {
        'mu': 5.0
    }
}

LOW_NORM = {
    'error_kwargs': {
        'loc': 0.0,
        'scale': 0.0
    }
}
HIGH_NORM = {
    'error_kwargs': {
        'loc': 3.0,
        'scale': 2.0
    }
}

LOW_RANDINT = {
    'purchase_day_kwargs': {
        'low': 0,
        'high': 1
    },
    '3_potential_demand_kwargs': {
        'low': 0,
        'high': 5000
    }
}
HIGH_RANDINT = {
    'purchase_day_kwargs': {
        'low': 0,
        'high': 7
    },
    '3_potential_demand_kwargs': {
        'low': 5000,
        'high': 10000
    }
}
