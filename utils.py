import numpy as np
import pandas as pd
from helper_functions import (exponential_growth, logarithmic_growth, 
                              power_function_growth, get_equation_latex)

def calculate_annual_emissions_removals(time_horizon, cultivation_cycle_duration, growth_params_df, emissions_df, luc_emissions, replanting_emissions, luc_emission_approach, amortization_years):
    # Initialize arrays for annual emissions and removals
    annual_emissions = np.zeros((time_horizon, len(emissions_df.columns)))
    annual_removals = np.zeros((time_horizon, len(growth_params_df.columns)))

    # Populate annual emissions
    for year in range(time_horizon):
        current_year = year % cultivation_cycle_duration
        annual_emissions[year, :] = emissions_df.iloc[current_year, :].values

    # Populate annual removals based on growth models
    for tree_type in growth_params_df.columns:
        params = growth_params_df[tree_type]

        if 'beta' in params and 'L' in params and 'k' in params:  # Exponential Plateau
            beta, L, k = params['beta'], params['L'], params['k']
            for year in range(time_horizon):
                annual_removals[year, tree_type] = L * (1 - np.exp(-k * (year + 1) / beta))
        elif 'Coefficient' in params and 'Intercept' in params:  # Logarithmic Growth (Terra Global Method)
            coefficient, intercept = params['Coefficient'], params['Intercept']
            for year in range(time_horizon):
                annual_removals[year, tree_type] = coefficient * np.log(year + 1) + intercept
        elif 'alpha' in params and 'beta' in params:  # Power Function (Cool Farm Method)
            alpha, beta = params['alpha'], params['beta']
            for year in range(time_horizon):
                annual_removals[year, tree_type] = alpha * (year + 1) ** beta

    return annual_emissions, annual_removals