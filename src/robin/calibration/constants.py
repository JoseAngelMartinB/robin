"""Constants for the calibration module."""

# List of continuous distributions available for the calibration process
CHOICES_CONTINUOUS = ['norm']

# List of discrete distributions available for the calibration process
CHOICES_DISCRETE = ['randint', 'poisson']

# List of distributions available for modeling potential demand
CHOICES_POTENCIAL_DEMAND = ['randint']

# Default number of best trials per execution to keep during calibration
DEFAULT_KEEP_TOP_K = 3

# Minimum value for arrival time parameters
LOW_ARRIVAL_TIME = 0.0

# Minimum values for normal distribution parameters
LOW_NORM = {
    'error_kwargs': {
        'loc': 0.0,
        'scale': 0.0
    }
}

# Minimum value for utility penalty function parameters
LOW_PENALTY_UTILITY = 0.0

# Minimum values for Poisson distribution parameters
LOW_POISSON = {
    'purchase_day_kwargs': {
        'mu': 0.0
    }
}

# Minimum values for randint distribution parameters
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

# Minimum value for seats utility in the preference model
LOW_SEATS_UTILITY = 1

# Minimum value for train service providers (TSPs) utility
LOW_TSPS_UTILITY = 0

# Minimum value for user pattern distribution
LOW_USER_PATTERN_DISTRIBUTION = 0.0

# Maximum value for arrival time parameters
HIGH_ARRIVAL_TIME = 1.0

# Maximum values for normal distribution parameters
HIGH_NORM = {
    'error_kwargs': {
        'loc': 3.0,
        'scale': 2.0
    }
}

# Maximum value for utility penalty function parameters
HIGH_PENALTY_UTILITY = 1.0

# Maximum values for Poisson distribution parameters
HIGH_POISSON = {
    'purchase_day_kwargs': {
        'mu': 5.0
    }
}

# Maximum values for randint distribution parameters
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

# Maximum value for seats utility in the preference model
HIGH_SEATS_UTILITY = 40

# Maximum value for train service providers (TSPs) utility
HIGH_TSPS_UTILITY = 10

# Maximum value for user pattern distribution
HIGH_USER_PATTERN_DISTRIBUTION = 1.0
