# data_handling.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import design_options as do
import helper_functions as hf
from emissions import update_emissions_df
from scenario_management import load_scenario
from reserve_rate_widget import reserve_rate_widget



def initialize_inputs():
    if 'time_horizon' not in st.session_state:
        st.session_state['time_horizon'] = 50 # Default to 50 years
    if 'cultivation_cycle_duration' not in st.session_state:
        st.session_state['cultivation_cycle_duration'] = 10 # Default to 20 years
    if 'luc_emission_approach' not in st.session_state:
        st.session_state['luc_emission_approach'] = 'Amortized'  # Default to "Amortized"
    if 'amortization_years' not in st.session_state:
        st.session_state['amortization_years'] = 20  # Default to 20 years
    if 'growth_curve_selections' not in st.session_state:
        st.session_state['growth_curve_selections'] = {}
    if 'default_params' not in st.session_state:
        st.session_state['default_params'] = {}
    if 'emissions_df' not in st.session_state:
        st.session_state['emissions_df'] = pd.DataFrame(columns=['Year', 'LUC Emissions', 'Replanting Emissions'])
    if 'planting_densities' not in st.session_state:
        st.session_state['planting_densities'] = {
            'reference': {'Cocoa': 1800, 'Shade': 150, 'Timber': 33},
            'modified': {'Cocoa': 1950, 'Shade': 0, 'Timber': 33}
        }
        st.session_state.selected_density = 'reference'
    if 'scenarios_df' not in st.session_state:
        default_data = {
            "Scenario": ["A", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            "Block": [1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Area (ha)": [500, 500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "Year": [1, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Start from year 1
        }
        st.session_state.scenarios_df = pd.DataFrame(default_data)



def simulation_parameters():
    st.subheader('Simulation Parameters')
    
    # Time horizon input
    time_horizon = st.number_input(
        'Enter the time horizon for the simulation (years):',
        min_value=1,
        value=st.session_state.get('time_horizon', 50),
        step=1,
        key='time_horizon_input'
    )

    # Cultivation cycle duration input
    cultivation_cycle_duration = st.number_input(
        'Enter the duration of the cultivation cycle (years):',
        min_value=1,
        value=st.session_state.get('cultivation_cycle_duration', 10),
        step=1,
        key='cultivation_cycle_duration_input'
    )

    # Update button
    if st.button('Update Simulation Parameters', key='update_sim_params_button'):
        st.session_state['time_horizon'] = time_horizon
        st.session_state['cultivation_cycle_duration'] = cultivation_cycle_duration
        st.success('Simulation parameters updated successfully!')

        # Info boxes for current values
        st.info(f"Time horizon: {st.session_state['time_horizon']} years")
        st.info(f"Cultivation cycle duration: {st.session_state['cultivation_cycle_duration']} years")

    return time_horizon, cultivation_cycle_duration



def apply_global_reserve_rate():
    apply_reserve_rate = st.session_state.get('apply_reserve_rate', False)
    reserve_rate = st.session_state.get('reserve_rate', 0.0)

    if 'annual_removals_df' in st.session_state and 'cumulative_removals_df' in st.session_state:
        annual_removals_df = st.session_state['annual_removals_df'].copy()
        cumulative_removals_df = st.session_state['cumulative_removals_df'].copy()

        for col in annual_removals_df.columns:
            if col not in ['Year Number', 'Year', 'Total_Removals', 'Removal Reversals']:
                if apply_reserve_rate:
                    annual_removals_df[col] *= (1 - reserve_rate)
                    cumulative_removals_df[col] *= (1 - reserve_rate)
                else:
                    annual_removals_df[col] = st.session_state['original_annual_removals_df'][col]
                    cumulative_removals_df[col] = st.session_state['original_cumulative_removals_df'][col]

        # Recalculate total removals
        annual_removals_df['Total_Removals'] = annual_removals_df.drop(columns=['Year Number', 'Year', 'Removal Reversals']).sum(axis=1)
        cumulative_removals_df['Total_Removals'] = cumulative_removals_df.drop(columns=['Year Number', 'Year', 'Removal Reversals']).sum(axis=1)

        st.session_state['adjusted_annual_removals_df'] = annual_removals_df
        st.session_state['adjusted_cumulative_removals_df'] = cumulative_removals_df

    # Reset the change flag
    st.session_state['reserve_rate_changed'] = False



def calculate_scenarios(time_horizon):
    if 'update_flag' in st.session_state and st.session_state.update_flag:
        st.session_state.update_flag = False

        if 'annual_removals_df' not in st.session_state:
            st.error("Annual removals data not found. Please run the 'One Hectare Model' tab first.")
            return None

        if 'annual_emissions_df' not in st.session_state:
            st.error("Annual emissions data not found. Please run the 'One Hectare Model' tab first.")
            return None

        kpi_df = pd.DataFrame({'Year': np.arange(1, time_horizon + 1)})

        for index, row in st.session_state.scenarios_df.iterrows():
            scenario_name = row['Scenario']
            load_scenario(scenario_name)

            annual_removals_df = st.session_state['annual_removals_df']
            annual_emissions_df = st.session_state['annual_emissions_df']
            cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 20)
            luc_emissions = st.session_state.get('luc_emissions', 0.5648)
            replanting_emissions = st.session_state.get('replanting_emissions', 0.0)

            area = row['Area (ha)']
            start_year = int(row['Year'])

            scenario_data = {
                "emissions": np.zeros(time_horizon),
                "removals": np.zeros(time_horizon),
                "area_planted": np.zeros(time_horizon),
                "cocoa_yield": np.zeros(time_horizon),
                "annual_removals_by_tree_type": {year: {tree_type: 0 for tree_type in annual_removals_df.columns if tree_type not in ['Year', 'Total_Removals']} for year in range(1, time_horizon + 1)}
            }

            for year in range(start_year, time_horizon + 1):
                scenario_data["area_planted"][year - 1] += area
                cycle_year = (year - start_year) % cultivation_cycle_duration

                for tree_type in annual_removals_df.columns:
                    if tree_type not in ['Year', 'Total_Removals']:
                        annual_growth = annual_removals_df[tree_type].iloc[cycle_year] * area
                        scenario_data["annual_removals_by_tree_type"][year][tree_type] += annual_growth
                        scenario_data["removals"][year - 1] += annual_growth

                scenario_data["emissions"][year - 1] += annual_emissions_df['Total'].iloc[cycle_year] * area

                if 'Cocoa Yield (t/ha/yr)' in annual_removals_df.columns:
                    scenario_data['cocoa_yield'][year - 1] += annual_removals_df['Cocoa Yield (t/ha/yr)'].iloc[cycle_year] * area

                if cycle_year == 0 and year != start_year:
                    scenario_data["emissions"][year - 1] += annual_emissions_df['Total'].iloc[0] * area

            if start_year <= time_horizon:
                scenario_data["emissions"][start_year - 1] += area * luc_emissions
            last_year_of_cycle = start_year + cultivation_cycle_duration - 1
            if last_year_of_cycle <= time_horizon:
                scenario_data["emissions"][last_year_of_cycle - 1] += area * replanting_emissions

            cumulative_emissions = np.cumsum(scenario_data["emissions"])
            cumulative_removals = np.cumsum(scenario_data["removals"])
            carbon_balance = cumulative_removals - cumulative_emissions

            farm_block_name = f'{row["Farm"]}-{row["Block"]}'
            kpi_df[f'{farm_block_name}: Cumulative Emissions'] = cumulative_emissions
            kpi_df[f'{farm_block_name}: Cumulative Removals'] = cumulative_removals
            kpi_df[f'{farm_block_name}: Carbon Balance'] = carbon_balance
            kpi_df[f'{farm_block_name}: Total Area Planted (ha)'] = scenario_data["area_planted"]

            st.session_state[f'{farm_block_name}_annual_removals_by_tree_type'] = scenario_data["annual_removals_by_tree_type"]

        st.session_state.kpi_df = kpi_df



