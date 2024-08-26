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


def calculate_discounted_luc_emissions(luc_emissions, luc_event_year, amortization_years=20):
    discount_factors = [
        0.0975, 0.0925, 0.0875, 0.0825, 0.0775, 0.0725, 0.0675, 0.0625, 0.0575, 0.0525,
        0.0475, 0.0425, 0.0375, 0.0325, 0.0275, 0.0225, 0.0175, 0.0125, 0.0075, 0.0025
    ]
    total_emissions = 0.0
    for i in range(amortization_years):
        year_emissions = luc_emissions * discount_factors[i]
        total_emissions += year_emissions
    return total_emissions
