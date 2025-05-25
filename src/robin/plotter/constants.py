"""Constants for the plotter module."""

COLORS = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']
"""Colors used in the plots."""

DARK_GRAY = '#A9A9A9'
"""Grid color."""

MARKERS = {
    'departure': {'marker': '^', 'label': 'Departure Station'},
    'arrival': {'marker': 's', 'label': 'Arrival Station'},
    'intermediate': {'marker': 'o', 'label': 'Intermediate Station'}
}
"""Markers for departure, arrival, and intermediate stations in Marey charts."""

MINUTES_IN_A_DAY = 24 * 60
"""Number of minutes in a day."""

SCALE_MAX = 1000
"""Maximum scale for normalization in Marey charts."""

SAFETY_GAP = 10
"""Safety gap in minutes."""

STYLE = 'seaborn-v0_8-pastel'
"""Seaborn style for the plots."""

WHITE_SMOKE = '#F5F5F5'
"""Plot background color."""
