"""Constants for the plotter module."""

# Colors used in the plots
COLORS = ['lemonchiffon', 'lightsteelblue', 'palegreen', 'lightsalmon', 'lavender', 'lightgray']

# Grid color
DARK_GRAY = '#A9A9A9'

# Markers for departure, arrival, and intermediate stations in Marey charts
MARKERS = {
    'departure': {'marker': '^', 'label': 'Departure Station'},
    'arrival': {'marker': 's', 'label': 'Arrival Station'},
    'intermediate': {'marker': 'o', 'label': 'Intermediate Station'}
}

# Safety gap in minutes
SAFETY_GAP = 10

# Seaborn style for the plots
STYLE = 'seaborn-v0_8-pastel'

# Plot background color
WHITE_SMOKE = '#F5F5F5'
