"""Constants for the ROBIN calibration module."""

LOW_ARRIVAL_TIME = 0.0
HIGH_ARRIVAL_TIME = 1.0

CHOICES_PURCHASE_DAY = ['randint', 'poisson']

LOW_SEATS_UTILITY = 1
HIGH_SEATS_UTILITY = 40

LOW_PENALTY_UTILITY = 0.0
HIGH_PENALTY_UTILITY = 1.0

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
    'potential_demand_kwargs': {
        'low': 0,
        'high': 5000
    }
}
HIGH_RANDINT = {
    'purchase_day_kwargs': {
        'low': 0,
        'high': 7
    },
    'potential_demand_kwargs': {
        'low': 5000,
        'high': 10000
    }
}
