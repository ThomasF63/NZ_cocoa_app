# planting_density.py

import streamlit as st

def planting_density_section():

    # Initialize the planting densities in session state if not already present
    if 'planting_densities' not in st.session_state:
        st.session_state.planting_densities = {
            'reference': {'Cocoa': 1800, 'Shade': 150, 'Timber': 33},
            'modified': {'Cocoa': 1950, 'Shade': 0, 'Timber': 33}
        }
        st.session_state.selected_density = 'reference'

    st.header("Planting Density Parametrization", divider="gray")

    # Input fields for reference planting densities
    st.subheader("Reference Density", divider="gray")
    reference_cocoa_density = st.number_input("Reference Cocoa Density (trees/ha):", 
                                              min_value=0, 
                                              value=st.session_state.planting_densities['reference']['Cocoa'], 
                                              key='reference_cocoa_density')
    reference_shade_density = st.number_input("Reference Shade Density (trees/ha):", 
                                              min_value=0, 
                                              value=st.session_state.planting_densities['reference']['Shade'], 
                                              key='reference_shade_density')
    reference_timber_density = st.number_input("Reference Timber Density (trees/ha):", 
                                               min_value=0, 
                                               value=st.session_state.planting_densities['reference']['Timber'], 
                                               key='reference_timber_density')

    # Input fields for modified planting densities
    st.subheader("Modified Density", divider="gray")
    modified_cocoa_density = st.number_input("Modified Cocoa Density (trees/ha):", 
                                             min_value=0, 
                                             value=st.session_state.planting_densities['modified']['Cocoa'], 
                                             key='modified_cocoa_density')
    modified_shade_density = st.number_input("Modified Shade Density (trees/ha):", 
                                             min_value=0, 
                                             value=st.session_state.planting_densities['modified']['Shade'], 
                                             key='modified_shade_density')
    modified_timber_density = st.number_input("Modified Timber Density (trees/ha):", 
                                              min_value=0, 
                                              value=st.session_state.planting_densities['modified']['Timber'], 
                                              key='modified_timber_density')

    # Radio button to select which density to use
    density_option = st.radio("Select Density to Use:", ('Reference', 'Modified'), key='density_option')

    # Button to validate and update planting densities
    if st.button("Validate Planting Densities"):
        # Update session state with new values
        st.session_state.planting_densities['reference']['Cocoa'] = reference_cocoa_density
        st.session_state.planting_densities['reference']['Shade'] = reference_shade_density
        st.session_state.planting_densities['reference']['Timber'] = reference_timber_density

        st.session_state.planting_densities['modified']['Cocoa'] = modified_cocoa_density
        st.session_state.planting_densities['modified']['Shade'] = modified_shade_density
        st.session_state.planting_densities['modified']['Timber'] = modified_timber_density

        # Set the selected density in session state
        st.session_state.selected_density = 'reference' if density_option == 'Reference' else 'modified'
        
        reference_densities = st.session_state.planting_densities['reference']
        selected_densities = st.session_state.planting_densities[st.session_state.selected_density]

        # Recalculate growth data based on the selected densities
        adjust_growth_data_based_on_density(reference_densities, selected_densities)

        st.success("Planting densities validated and growth data updated!")
        st.write("Debug: planting_densities and selected_density saved in session state")
        st.write(st.session_state.planting_densities)
        st.write(st.session_state.selected_density)


def adjust_growth_data_based_on_density(reference_densities, selected_densities):
    # Adjust the growth parameters in the session state based on the selected densities
    for tree_type in reference_densities:
        ratio = selected_densities[tree_type] / reference_densities[tree_type]
        st.session_state.growth_params_df[tree_type] *= ratio

