"""Constants for the calibration module."""

CHOICES_CONTINUOUS = ['norm']
"""List of continuous distributions available for the calibration process."""

CHOICES_ARRIVAL_TIME = CHOICES_CONTINUOUS + ['hourly']
"""List of distributions available for modeling arrival time."""

CHOICES_DISCRETE = ['randint', 'poisson']
"""List of discrete distributions available for the calibration process."""

CHOICES_POTENCIAL_DEMAND = ['randint']
"""List of distributions available for modeling potential demand."""

DEFAULT_KEEP_TOP_K = 3
"""Default number of best trials per execution to keep during calibration."""

LOW_ARRIVAL_TIME_HOURLY = 0.0
"""Minimum value for hourly arrival time parameters."""

LOW_NORM = {
    'arrival_time_kwargs': {
        'loc': 6.0,
        'scale': 0.0
    },
    'error_kwargs': {
        'loc': 0.0,
        'scale': 0.0
    }
}
"""Minimum values for normal distribution parameters."""

LOW_PENALTY_UTILITY = 0.0
"""Minimum value for utility penalty function parameters."""

LOW_POISSON = {
    'purchase_day_kwargs': {
        'mu': 0.0
    }
}
"""Minimum values for Poisson distribution parameters."""

LOW_RANDINT = {
    'purchase_day_kwargs': {
        'low': 0,
        'high': 1
    },
    '1_potential_demand_kwargs': {
        'low': 12500,
        'high': 17500
    },
    '2_potential_demand_kwargs': {
        'low': 1250,
        'high': 1750
    },
    '3_potential_demand_kwargs': {
        'low': 1250,
        'high': 1750
    },
    '4_potential_demand_kwargs': {
        'low': 1250,
        'high': 1750
    },
    '5_potential_demand_kwargs': {
        'low': 1250,
        'high': 1750
    },
    '6_potential_demand_kwargs': {
        'low': 1250,
        'high': 1750
    }
}
"""Minimum values for randint distribution parameters."""

LOW_SEATS_UTILITY = 1
"""Minimum value for seats utility in the preference model."""

LOW_TSPS_UTILITY = 0
"""Minimum value for train service providers (TSPs) utility."""

LOW_USER_PATTERN_DISTRIBUTION = 0.0
"""Minimum value for user pattern distribution."""

HIGH_ARRIVAL_TIME_HOURLY = 1.0
"""Maximum value for hourly arrival time parameters."""

HIGH_NORM = {
    'arrival_time_kwargs': {
        'loc': 24.0,
        'scale': 12.0
    },
    'error_kwargs': {
        'loc': 3.0,
        'scale': 2.0
    }
}
"""Maximum values for normal distribution parameters."""

HIGH_PENALTY_UTILITY = 10.0
"""Maximum value for utility penalty function parameters."""

HIGH_POISSON = {
    'purchase_day_kwargs': {
        'mu': 5.0
    }
}
"""Maximum values for Poisson distribution parameters."""

HIGH_RANDINT = {
    'purchase_day_kwargs': {
        'low': 0,
        'high': 7
    },
    '1_potential_demand_kwargs': {
        'low': 15000,
        'high': 35000
    },
    '2_potential_demand_kwargs': {
        'low': 1500,
        'high': 3500
    },
    '3_potential_demand_kwargs': {
        'low': 1500,
        'high': 3500
    },
    '4_potential_demand_kwargs': {
        'low': 1500,
        'high': 3500
    },
    '5_potential_demand_kwargs': {
        'low': 1500,
        'high': 3500
    },
    '6_potential_demand_kwargs': {
        'low': 1500,
        'high': 3500
    }
}
"""Maximum values for randint distribution parameters."""

HIGH_SEATS_UTILITY = 40
"""Maximum value for seats utility in the preference model."""

HIGH_TSPS_UTILITY = 30
"""Maximum value for train service providers (TSPs) utility."""

HIGH_USER_PATTERN_DISTRIBUTION = 1.0
"""Maximum value for user pattern distribution."""

OTHER_MARKET_ID = -1
"""Identifier for 'other' market."""
