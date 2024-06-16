import streamlit as st
import pandas as pd
import numpy as np
from data_handling import simulation_parameters, cocoa_yield_curve, carbon_balance_section, carbon_intensity_section, summary_section
from tree_growth import tree_growth_parameters
from emissions import emissions_input, emissions_analysis
from removals import removals_analysis



def one_hectare_model():
    st.sidebar.title("One Hectare Model Sections")
    sections = ["Simulation Parameters", "Tree Growth Parameters", "Planting Density", "Cocoa Yield Curve", "Emissions", "Removals", "Carbon Balance", "Carbon Intensity", "Summary"]
    section = st.sidebar.radio("Go to Section", sections)

    if section == "Simulation Parameters":
        time_horizon, cultivation_cycle_duration, luc_emission_approach, amortization_years = simulation_parameters()
        st.session_state['time_horizon'] = time_horizon
        st.session_state['cultivation_cycle_duration'] = cultivation_cycle_duration

    time_horizon = st.session_state.get('time_horizon', 10)
    cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 20)

    if section == "Tree Growth Parameters":
        growth_params_df = tree_growth_parameters(time_horizon)
    elif section == "Planting Density":
        planting_density_section()
    elif section == "Cocoa Yield Curve":
        cocoa_yield_df = cocoa_yield_curve(cultivation_cycle_duration, time_horizon)
    elif section == "Emissions":
        luc_emissions, replanting_emissions, emissions_df = emissions_input(cultivation_cycle_duration)
        emissions_analysis(time_horizon)
    elif section == "Removals":
        removals_analysis(time_horizon)
    elif section == "Carbon Balance":
        carbon_balance_section(time_horizon)
    elif section == "Carbon Intensity":
        carbon_intensity_section(time_horizon)
    elif section == "Summary":
        summary_section(time_horizon)




def planting_density_section():
    st.header("Planting Density")

    if 'planting_densities' not in st.session_state:
        st.session_state.planting_densities = {
            'reference': {'Cocoa': 1800, 'Shade': 150, 'Timber': 33},
            'modified': {'Cocoa': 1950, 'Shade': 0, 'Timber': 33}
        }
        st.session_state.selected_density = 'reference'

    st.subheader("Reference Density")
    reference_cocoa_density = st.number_input("Reference Cocoa Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['reference']['Cocoa'], key='reference_cocoa_density')
    reference_shade_density = st.number_input("Reference Shade Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['reference']['Shade'], key='reference_shade_density')
    reference_timber_density = st.number_input("Reference Timber Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['reference']['Timber'], key='reference_timber_density')

    st.subheader("Modified Density")
    modified_cocoa_density = st.number_input("Modified Cocoa Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['modified']['Cocoa'], key='modified_cocoa_density')
    modified_shade_density = st.number_input("Modified Shade Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['modified']['Shade'], key='modified_shade_density')
    modified_timber_density = st.number_input("Modified Timber Density (trees/ha):", min_value=0, value=st.session_state.planting_densities['modified']['Timber'], key='modified_timber_density')

    density_option = st.radio("Select Density to Use:", ('Reference', 'Modified'), key='density_option')

    if st.button("Validate Planting Densities"):
        st.session_state.planting_densities['reference']['Cocoa'] = reference_cocoa_density
        st.session_state.planting_densities['reference']['Shade'] = reference_shade_density
        st.session_state.planting_densities['reference']['Timber'] = reference_timber_density

        st.session_state.planting_densities['modified']['Cocoa'] = modified_cocoa_density
        st.session_state.planting_densities['modified']['Shade'] = modified_shade_density
        st.session_state.planting_densities['modified']['Timber'] = modified_timber_density

        st.session_state.selected_density = 'reference' if density_option == 'Reference' else 'modified'
        
        reference_densities = st.session_state.planting_densities['reference']
        modified_densities = st.session_state.planting_densities[st.session_state.selected_density]

        # Recalculate growth data based on the selected densities
        adjust_growth_data_based_on_density(reference_densities, modified_densities)

        st.success("Planting densities validated and growth data updated!")
        st.write("Debug: planting_densities and selected_density saved in session state")
        st.write(st.session_state.planting_densities)
        st.write(st.session_state.selected_density)

def adjust_growth_data_based_on_density(reference_densities, selected_densities):
    for tree_type in reference_densities:
        ratio = selected_densities[tree_type] / reference_densities[tree_type]
        st.session_state.growth_params_df[tree_type] *= ratio
