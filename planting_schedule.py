import streamlit as st
from data_handling import calculate_scenarios
from helper_functions import plot_total_area_planted, plot_emissions_vs_removals_planting_scenarios, plot_carbon_balance_planting_scenarios, plot_removals_by_tree_type

def planting_schedule():
    st.subheader('Planting Scenarios')

    scenarios_df = st.session_state.scenarios_df
    edited_scenarios_df = st.data_editor(data=scenarios_df, num_rows="dynamic", key="scenarios")

    time_horizon = st.session_state.get('time_horizon', 50)

    if st.button("Update Graphs and Tables"):
        st.session_state.scenarios_df = edited_scenarios_df
        st.session_state.update_flag = True
        calculate_scenarios(time_horizon)

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
    else:
        st.write("No KPI data available. Please complete the necessary sections first.")
