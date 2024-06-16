# design_options.py

# Define colors for dark mode
DARK_EMISSIONS_COLOR = '#fc8d62'  # Orange
DARK_REMOVALS_COLOR = '#66c2a5'   # Green
DARK_COCOA_YIELD_COLOR = '#8da0cb' # Purple

# Define colors for light mode
LIGHT_EMISSIONS_COLOR = '#d95f02'  # Darker Orange
LIGHT_REMOVALS_COLOR = '#1b9e77'   # Darker Green
LIGHT_COCOA_YIELD_COLOR = '#7570b3' # Darker Purple

def get_colors(mode):
    if mode == "Light Mode":
        return {
            'EMISSIONS_COLOR': LIGHT_EMISSIONS_COLOR,
            'REMOVALS_COLOR': LIGHT_REMOVALS_COLOR,
            'COCOA_YIELD_COLOR': LIGHT_COCOA_YIELD_COLOR
        }
    else:
        return {
            'EMISSIONS_COLOR': DARK_EMISSIONS_COLOR,
            'REMOVALS_COLOR': DARK_REMOVALS_COLOR,
            'COCOA_YIELD_COLOR': DARK_COCOA_YIELD_COLOR
        }
