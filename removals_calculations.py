import numpy as np
from helper_functions import exponential_growth, logarithmic_growth, power_function_growth

def calculate_removals(time_horizon, cultivation_cycle_duration, growth_params_df):
    annual_removals = np.zeros((time_horizon, len(growth_params_df.columns)))
    
    for idx, tree_type in enumerate(growth_params_df.columns):
        params = growth_params_df[tree_type]
        
        if tree_type == 'Cocoa':
            beta, L, k = 0.5, 100, 0.1  # Example parameters for Cocoa
            for year in range(time_horizon):
                annual_removals[year, idx] = exponential_growth(year + 1, beta, L, k)
        elif tree_type == 'Shade':
            coefficient, intercept = 39.819, 14.817  # Example parameters for Shade
            for year in range(time_horizon):
                annual_removals[year, idx] = logarithmic_growth(year + 1, coefficient, intercept)
        elif tree_type == 'Timber':
            alpha, beta = 1.6721, 0.88  # Example parameters for Timber
            for year in range(time_horizon):
                annual_removals[year, idx] = power_function_growth(year + 1, alpha, beta, 0.47, 3.667)

    return annual_removals
