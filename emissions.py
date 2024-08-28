import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as hf
from LUC import generate_luc_emissions_df

def update_emissions_df(cultivation_cycle_duration=20):
    emission_sources = ['Year Number', 'Year', 'Equipment Fuel', 'Fertilizer Estimate', 'Soil Amendments', 'Pesticide Estimate', 'Drying Estimate']

    # Determine the start year based on land preparation year
    land_prep_year = st.session_state.get('land_prep_year', 2015)
    year_numbers = np.arange(1, cultivation_cycle_duration + 1)
    actual_years = land_prep_year + (year_numbers - 1)

    # Initial emissions values (for a 20-year cycle)
    initial_values = np.array([
        [0.022, 0.000, 0.000, 0.000, 0.000],
        [0.118, 0.000, 0.000, 0.000, 0.009],
        [0.163, 0.055, 0.394, 0.000, 0.210],
        [0.550, 0.409, 0.469, 0.004, 0.300],
        [0.490, 0.240, 0.453, 0.002, 0.480],
        [0.829, 0.272, 0.622, 0.002, 0.600],
        [0.978, 0.389, 0.639, 0.004, 0.720],
        [0.978, 0.584, 0.160, 0.004, 0.720],
        [1.222, 0.729, 0.200, 0.004, 0.900],
        [1.467, 0.875, 0.060, 0.005, 1.080]
    ])

    # Adjust initial_values to match cultivation_cycle_duration
    if cultivation_cycle_duration > len(initial_values):
        # If cultivation_cycle_duration is longer, repeat the last row
        extra_rows = cultivation_cycle_duration - len(initial_values)
        initial_values = np.vstack((initial_values, np.tile(initial_values[-1], (extra_rows, 1))))
    elif cultivation_cycle_duration < len(initial_values):
        # If cultivation_cycle_duration is shorter, truncate the array
        initial_values = initial_values[:cultivation_cycle_duration]

    # Create the DataFrame
    emissions_df = pd.DataFrame(np.column_stack((year_numbers, actual_years, initial_values)), 
                                columns=emission_sources)
    
    # Calculate total emissions
    emissions_df['Total'] = emissions_df.iloc[:, 2:].sum(axis=1)

    # Save to session state
    st.session_state.emissions_df = emissions_df

    return emissions_df

def generate_emissions_forecast(time_horizon, cultivation_cycle_duration):
    # Get the farm management emissions template
    template_df = st.session_state.emissions_df

    # Create a new DataFrame for the forecast
    forecast_df = pd.DataFrame(index=range(time_horizon))
    
    # Fill in the farm management emissions
    for year in range(time_horizon):
        cycle_year = year % cultivation_cycle_duration
        for col in template_df.columns:
            if col not in ['Year Number', 'Year']:
                forecast_df.loc[year, col] = template_df.loc[min(cycle_year, len(template_df) - 1), col]

    # Add Year Number and Year columns
    forecast_df['Year Number'] = range(1, time_horizon + 1)
    forecast_df['Year'] = st.session_state.get('land_prep_year', 2015) + forecast_df['Year Number'] - 1

    # Add LUC emissions
    luc_emissions_df = generate_luc_emissions_df(
        time_horizon, 
        st.session_state.get('luc_event_year', 2015),
        st.session_state.get('luc_emissions', 5.0),
        st.session_state.get('amortization_years', 5),
        st.session_state.get('amortization_method', 'Equal Amortization'),
        st.session_state.get('luc_application', "At LUC event year"),
        st.session_state.get('amortize_luc', False)
    )
    forecast_df['LUC Emissions'] = luc_emissions_df['LUC Emissions']

    # Add Removal Reversals
    if 'removal_reversals' in st.session_state:
        forecast_df['Removal Reversals'] = st.session_state['removal_reversals'][:time_horizon]
    else:
        forecast_df['Removal Reversals'] = 0

    # Calculate total emissions
    forecast_df['Total'] = forecast_df.drop(['Year Number', 'Year'], axis=1).sum(axis=1)

    return forecast_df



def update_annual_emissions_df(time_horizon, cultivation_cycle_duration):
    if 'emissions_df' not in st.session_state or st.session_state.emissions_df is None or st.session_state.emissions_df.empty:
        st.write("No emissions data available. Generating default emissions data.")
        emissions_df = update_emissions_df(cultivation_cycle_duration)
    else:
        emissions_df = st.session_state.emissions_df

    if emissions_df.empty:
        st.error("Unable to generate emissions data. Please check your inputs and try again.")
        return None

    annual_emissions = np.zeros((time_horizon, len(emissions_df.columns)))

    # Get the initial year from the start of the first cycle
    initial_year = emissions_df['Year'].iloc[0]

    for year in range(time_horizon):
        cycle_year = year % cultivation_cycle_duration
        annual_emissions[year] = emissions_df.iloc[cycle_year].values

    # Calculate the actual year based on the initial year and the year number
    actual_years = initial_year + np.arange(time_horizon)

    annual_emissions_df = pd.DataFrame(annual_emissions, columns=emissions_df.columns)
    annual_emissions_df['Year Number'] = np.arange(1, time_horizon + 1)
    annual_emissions_df['Year'] = actual_years  # Assign the correct continuous years

    return annual_emissions_df



def emissions_input(cultivation_cycle_duration):
    st.header('Emissions Parametrization', divider="gray")

    # Initialize or update emissions_df with the cultivation cycle duration
    emissions_df = update_emissions_df(cultivation_cycle_duration)

    if emissions_df is not None and not emissions_df.empty:
        st.subheader(f'Table: Farm Management Emissions Template ({cultivation_cycle_duration} years)')
        st.dataframe(emissions_df)
    else:
        st.error("Failed to generate emissions template. Please check your inputs and try again.")
        return None

    # Generate and store the full emissions forecast
    time_horizon = st.session_state.get('time_horizon', 50)
    forecast_df = generate_emissions_forecast(time_horizon, cultivation_cycle_duration)
    
    # Add removal reversals to the forecast
    if 'removal_reversals' in st.session_state:
        forecast_df['Removal Reversals'] = st.session_state['removal_reversals']
    else:
        forecast_df['Removal Reversals'] = 0
    
    # Recalculate total emissions including removal reversals
    forecast_df['Total'] = forecast_df.drop(['Year Number', 'Year'], axis=1).sum(axis=1)
    
    # Reorder columns to ensure Year Number and Year are first
    columns_order = ['Year Number', 'Year'] + [col for col in forecast_df.columns if col not in ['Year Number', 'Year']]
    forecast_df = forecast_df[columns_order]
    
    st.session_state.annual_emissions_df = forecast_df

    return emissions_df



def emissions_analysis(time_horizon):
    if 'annual_emissions_df' in st.session_state and st.session_state.annual_emissions_df is not None:
        emissions_df = st.session_state.annual_emissions_df

        # Ensure Year Number and Year are the first two columns
        columns_order = ['Year Number', 'Year'] + [col for col in emissions_df.columns if col not in ['Year Number', 'Year']]
        emissions_df = emissions_df[columns_order]

        st.subheader('Graph: Annual Emissions Over Time')
        
        # Exclude 'Year Number', 'Year', and 'Total' from emission sources for the chart
        emissions_chart_df = emissions_df.drop(columns=['Year Number', 'Total'])
        annual_emissions_chart = hf.create_annual_emissions_stacked_bar_chart(emissions_chart_df, 'Year', 'Emissions', 'Emission Source')
        st.altair_chart(annual_emissions_chart, use_container_width=True)

        st.subheader('Table: Annual Emissions Over Time')
        format_dict = {col: '{:.3f}' for col in emissions_df.columns if col not in ['Year Number', 'Year']}
        format_dict.update({
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}'
        })
        st.dataframe(emissions_df.style.format(format_dict))

        st.subheader('Graph: Cumulative Emissions Over Time')
        # Calculate the cumulative emissions
        cumulative_emissions_df = emissions_df.drop(columns=['Year Number', 'Year', 'Total']).cumsum()
        cumulative_emissions_df['Year'] = emissions_df['Year']
        cumulative_emissions_df['Year Number'] = emissions_df['Year Number']
        
        # Calculate the total for each year
        cumulative_emissions_df['Total'] = cumulative_emissions_df.drop(['Year', 'Year Number'], axis=1).sum(axis=1)
        
        # Reorder columns to ensure Year Number and Year are first
        columns_order = ['Year Number', 'Year'] + [col for col in cumulative_emissions_df.columns if col not in ['Year Number', 'Year']]
        cumulative_emissions_df = cumulative_emissions_df[columns_order]
        
        cumulative_emissions_chart = hf.create_cumulative_emissions_stacked_bar_chart(cumulative_emissions_df, 'Year', 'Cumulative Emissions', 'Emission Source')
        st.altair_chart(cumulative_emissions_chart, use_container_width=True)

        st.subheader('Table: Cumulative Emissions Over Time')
        format_dict = {col: '{:.3f}' for col in cumulative_emissions_df.columns if col not in ['Year Number', 'Year']}
        format_dict.update({
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}'
        })
        st.dataframe(cumulative_emissions_df.style.format(format_dict))

    else:
        st.warning("No emissions data available. Please ensure the emissions input section is completed and data is properly calculated.")
