# one_hectare_model.py

import streamlit as st
from data_handling import simulation_parameters
from tree_growth import tree_growth_parameters
from emissions import emissions_input, emissions_analysis
from removals import removals_analysis
from scenario_management import save_scenario, load_scenario, get_saved_scenarios, duplicate_scenario, delete_scenario
from cocoa_yield import cocoa_yield_curve
from LUC import luc_parameters
from planting_density import planting_density_section
from carbon_intensity import carbon_intensity_section
from carbon_balance import carbon_balance_section
from summary import summary_section
from plantation_timeline import plantation_timeline
from data_handling import apply_global_reserve_rate


def one_hectare_model():
    st.sidebar.title("One Hectare Model Sections")
    sections = [
        "Simulation Parameters",
        "LUC Parameters",
        "Plantation Timeline",
        "Tree Growth Parameters",
        "Planting Density",
        "Cocoa Yield Curve",
        "Removals",
        "Emissions",
        "Carbon Balance",
        "Carbon Intensity",
        "Summary",
        "Save/Load Scenario"
    ]
    section = st.sidebar.radio("Go to Section", sections, index=0)

    time_horizon = st.session_state.get('time_horizon', 50)
    cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 10)

    if section == "Simulation Parameters":
        time_horizon, cultivation_cycle_duration = simulation_parameters()
        st.session_state['time_horizon'] = time_horizon
        st.session_state['cultivation_cycle_duration'] = cultivation_cycle_duration

    elif section == "LUC Parameters":
        luc_event_year, luc_emissions, luc_application, amortize_luc, amortization_years, amortization_method = luc_parameters()

    elif section == "Plantation Timeline":
        land_prep_year, shade_planting_year, cocoa_planting_year, timber_planting_year = plantation_timeline()

    elif section == "Tree Growth Parameters":
        growth_params_df = tree_growth_parameters(time_horizon)

    elif section == "Planting Density":
        density_df = planting_density_section()

    elif section == "Cocoa Yield Curve":
        cocoa_yield_curve(cultivation_cycle_duration, time_horizon)

    elif section == "Removals":
        removals_analysis(time_horizon)

    elif section == "Emissions":
        emissions_df = emissions_input(cultivation_cycle_duration)
        emissions_analysis(time_horizon)

    elif section == "Carbon Balance":
        carbon_balance_section(time_horizon)

    elif section == "Carbon Intensity":
        carbon_intensity_section(time_horizon)

    elif section == "Summary":
        summary_section(time_horizon)

    elif section == "Save/Load Scenario":
        st.subheader("Save Current Scenario")
        scenario_name = st.text_input("Scenario Name")
        if st.button("Save Scenario"):
            save_scenario(scenario_name)
        
        st.subheader("Load Saved Scenario")
        saved_scenarios = get_saved_scenarios()
        selected_scenario = st.selectbox("Select Scenario to Load", saved_scenarios)
        if st.button("Load Scenario"):
            load_scenario(selected_scenario)

        st.subheader("Duplicate Scenario")
        duplicate_scenario_name = st.selectbox("Select Scenario to Duplicate", saved_scenarios)
        new_scenario_name = st.text_input("New Scenario Name")
        if st.button("Duplicate Scenario"):
            duplicate_scenario(duplicate_scenario_name, new_scenario_name)

        st.subheader("Delete Scenario")
        delete_scenario_name = st.selectbox("Select Scenario to Delete", saved_scenarios)
        if st.button("Delete Scenario"):
            delete_scenario(delete_scenario_name)


        if st.button("Recalculate All Sections"):
            apply_global_reserve_rate()
            st.experimental_rerun()

