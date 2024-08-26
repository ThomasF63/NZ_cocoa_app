# planting_schedule.py

import streamlit as st
import pandas as pd
from data_handling import calculate_scenarios
from helper_functions import plot_total_area_planted, plot_emissions_vs_removals_planting_scenarios, plot_carbon_balance_planting_scenarios, plot_removals_by_tree_type, plot_cumulative_carbon_intensity, plot_annual_carbon_intensity
from scenario_management import get_saved_scenarios, load_scenario

def planting_schedule():
    st.subheader('Planting Scenarios')

    if 'scenarios_df' not in st.session_state:
        st.session_state.scenarios_df = pd.DataFrame({
            "Farm": [],
            "Block": [],
            "Area (ha)": [],
            "Year": [],
            "Scenario": []
        })

    edited_scenarios_df = st.data_editor(data=st.session_state.scenarios_df, num_rows="dynamic", key="scenarios")

    time_horizon = st.session_state.get('time_horizon', 50)

    if st.button("Update Graphs and Tables"):
        st.session_state.scenarios_df = edited_scenarios_df
        st.session_state.update_flag = True
        calculate_scenarios(time_horizon)

    saved_scenarios = get_saved_scenarios()

    st.subheader("Add New Block")
    new_farm = st.text_input("Farm Name")
    new_block = st.text_input("Block Name")
    new_area = st.number_input("Area (ha)", min_value=0.0)
    new_year = st.number_input("Year", min_value=1, step=1)
    new_scenario = st.selectbox("Select Scenario", saved_scenarios)
    if st.button("Add Block"):
        new_entry = {
            "Farm": new_farm,
            "Block": new_block,
            "Area (ha)": new_area,
            "Year": new_year,
            "Scenario": new_scenario
        }
        st.session_state.scenarios_df = st.session_state.scenarios_df.append(new_entry, ignore_index=True)

    if 'kpi_df' not in st.session_state:
        calculate_scenarios(time_horizon)

    if 'kpi_df' in st.session_state:
        st.subheader('Graph: Total Planted Area Over Time')
        plot_total_area_planted(st.session_state.kpi_df)

        st.subheader('Graph: Emissions vs. Removals Over Time')
        plot_emissions_vs_removals_planting_scenarios()

        st.subheader('Graph: Carbon Balance Over Time')
        plot_carbon_balance_planting_scenarios()

        st.subheader('Annual Removals by Tree Type')
        for scenario in st.session_state.scenarios_df['Scenario'].unique():
            plot_removals_by_tree_type(scenario)

        st.subheader('Graph: Cumulative Carbon Intensity Over Time')
        plot_cumulative_carbon_intensity(st.session_state.kpi_df)

        st.subheader('Graph: Annual Carbon Intensity Over Time')
        plot_annual_carbon_intensity(st.session_state.kpi_df)
    else:
        st.write("No KPI data available. Please complete the necessary sections first.")

