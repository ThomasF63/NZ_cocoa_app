import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as hf
from utils import calculate_annual_emissions_removals

def update_emissions_df(cultivation_cycle_duration, luc_emissions, replanting_emissions):
    emission_sources = ['Year', 'Equipment', 'Fuel', 'Fertilizer', 'Soil Amendments', 'Pesticide', 'Drying', 'LUC Emissions', 'Replanting Emissions', 'Total']
    years = np.arange(1, cultivation_cycle_duration + 1)
    
    # Initial values for emissions, assuming these are defined for 5 years and need to be repeated
    initial_values = np.array([
        [0.5, 0.3, 0.4, 0.2, 0.25, 0.3],
        [0.55, 0.35, 0.45, 0.25, 0.3, 0.35],
        [0.6, 0.4, 0.5, 0.3, 0.35, 0.4],
        [0.8, 0.45, 0.55, 0.35, 0.4, 0.45],
        [1.0, 0.55, 0.8, 0.4, 0.45, 0.6]
    ])
    
    # Repeat the last row to fill up to the cultivation cycle duration
    if cultivation_cycle_duration > 5:
        repeated_values = np.tile(initial_values[-1, :], (cultivation_cycle_duration - 5, 1))
        emissions_values = np.vstack((initial_values, repeated_values))
    else:
        emissions_values = initial_values[:cultivation_cycle_duration, :]

    # Add LUC and replanting columns
    luc_column = np.zeros(cultivation_cycle_duration)
    replanting_column = np.zeros(cultivation_cycle_duration)

    if st.session_state.get('luc_emission_approach', 'At Start') == 'At Start':
        luc_column[0] = luc_emissions
    else:
        luc_column[:st.session_state.get('amortization_years', 10)] = luc_emissions / st.session_state.get('amortization_years', 10)

    emissions_values = np.column_stack((emissions_values, luc_column, replanting_column))
    emissions_df = pd.DataFrame(np.column_stack((years, emissions_values, np.zeros((cultivation_cycle_duration, 1)))), columns=emission_sources)
    emissions_df['Total'] = emissions_df.iloc[:, 1:8].sum(axis=1)  # Sum only the main emissions columns, excluding LUC and replanting emissions
    st.session_state.emissions_df = emissions_df

    update_annual_emissions_df(time_horizon=st.session_state['time_horizon'], cultivation_cycle_duration=cultivation_cycle_duration, luc_emissions=luc_emissions, replanting_emissions=replanting_emissions)

def update_annual_emissions_df(time_horizon, cultivation_cycle_duration, luc_emissions, replanting_emissions):
    if 'growth_params_df' not in st.session_state or st.session_state.growth_params_df is None:
        st.write("No growth data available.")
        return

    growth_params_df = st.session_state.growth_params_df
    emissions_df = st.session_state.emissions_df
    luc_emission_approach = st.session_state.get('luc_emission_approach', 'At Start')
    amortization_years = st.session_state.get('amortization_years', 10)

    annual_emissions, _ = calculate_annual_emissions_removals(time_horizon, cultivation_cycle_duration, growth_params_df, emissions_df, luc_emissions, replanting_emissions, luc_emission_approach, amortization_years)
    emissions_sources = emissions_df.columns.tolist()
    annual_emissions_df = pd.DataFrame(annual_emissions, columns=emissions_sources)
    if 'Year' not in annual_emissions_df.columns:
        annual_emissions_df.insert(0, 'Year', np.arange(1, time_horizon + 1))
    else:
        annual_emissions_df['Year'] = np.arange(1, time_horizon + 1)
    st.session_state.annual_emissions_df = annual_emissions_df

def emissions_input(cultivation_cycle_duration):
    st.subheader('LUC and Replanting/Recycling emissions')

    # Initialize emissions_df if it doesn't exist
    if 'emissions_df' not in st.session_state:
        update_emissions_df(cultivation_cycle_duration, st.session_state.get('luc_emissions', 100.0), st.session_state.get('replanting_emissions', 50.0))

    # Create columns for input boxes and buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        luc_emissions = st.number_input('Enter LUC emissions (tons of CO2):', value=st.session_state.get('luc_emissions', 100.0), step=0.1)
    with col2:
        if st.button('Update LUC Emissions'):
            st.session_state.luc_emissions = luc_emissions  # Save to session state
            update_emissions_df(cultivation_cycle_duration, luc_emissions, st.session_state.get('replanting_emissions', 50.0))

    col3, col4 = st.columns([3, 1])
    with col3:
        replanting_emissions = st.number_input('Enter Replanting/Recycling emissions (tons of CO2):', value=st.session_state.get('replanting_emissions', 50.0), step=0.1)
    with col4:
        if st.button('Update Replanting Emissions'):
            st.session_state.replanting_emissions = replanting_emissions  # Save to session state
            update_emissions_df(cultivation_cycle_duration, st.session_state.get('luc_emissions', 100.0), replanting_emissions)

    st.subheader('Table: Annual Emissions Input for One Cultivation Cycle')
    st.dataframe(st.session_state.emissions_df)

    annual_emissions_csv = st.session_state.emissions_df.to_csv(index=False)
    st.download_button(
        label="Download Annual Emissions Data as CSV",
        data=annual_emissions_csv,
        file_name='annual_emissions_data.csv',
        mime='text/csv',
        key='download_emissions_input'
    )

    return luc_emissions, replanting_emissions, st.session_state.emissions_df

def emissions_analysis(time_horizon):
    if 'annual_emissions_df' in st.session_state:
        emissions_df = st.session_state.annual_emissions_df

        st.subheader('Graph: Annual Emissions Over Time')
        annual_emissions_chart = hf.create_annual_emissions_stacked_bar_chart(emissions_df, 'Year', 'Emissions', 'Emission Source', title='Annual Emissions Over Time')
        st.altair_chart(annual_emissions_chart, use_container_width=True)

        st.subheader('Table: Annual Emissions Over Time')
        st.dataframe(emissions_df)

        csv = emissions_df.to_csv(index=False)
        st.download_button(
            label="Download Annual Emissions Data as CSV",
            data=csv,
            file_name='annual_emissions_data.csv',
            mime='text/csv',
            key='download_annual_emissions_data'
        )

        st.subheader('Graph: Cumulative Emissions Over Time')
        cumulative_emissions_chart = hf.create_cumulative_emissions_stacked_bar_chart(emissions_df, 'Year', 'Cumulative Emissions', 'Emission Source', title='Cumulative Emissions Over Time')
        st.altair_chart(cumulative_emissions_chart, use_container_width=True)

        st.subheader('Table: Cumulative Emissions Over Time')
        cumulative_emissions_df = emissions_df.copy()
        cumulative_emissions_df.iloc[:, 1:] = cumulative_emissions_df.iloc[:, 1:].cumsum()
        st.dataframe(cumulative_emissions_df)

        csv_cumulative = cumulative_emissions_df.to_csv(index=False)
        st.download_button(
            label="Download Cumulative Emissions Data as CSV",
            data=csv_cumulative,
            file_name='cumulative_emissions_data.csv',
            mime='text/csv',
            key='download_cumulative_emissions_data'
        )
