import streamlit as st
from data_handling import initialize_inputs
from helper_functions import plot_removals_by_tree_type, plot_total_area_planted, plot_emissions_vs_removals_planting_scenarios, plot_carbon_balance_planting_scenarios

def planting_schedule():
    st.title("Planting Schedule")
    initialize_inputs()
    plot_total_area_planted(st.session_state.kpi_df)
    plot_emissions_vs_removals_planting_scenarios()
    plot_carbon_balance_planting_scenarios()
    for scenario in st.session_state.scenarios_df['Scenario'].unique():
        plot_removals_by_tree_type(scenario)
