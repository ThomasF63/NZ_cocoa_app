# LUC.py

import streamlit as st
import numpy as np
import pandas as pd

def generate_luc_emissions_df(time_horizon, luc_event_year, luc_emissions, amortization_years, amortization_method, luc_application, amortize_luc):
    # Generate a range of years from the LUC event year up to the time horizon
    years = np.arange(luc_event_year, luc_event_year + time_horizon)
    
    # Initialize a DataFrame with these years and zero emissions
    luc_emissions_df = pd.DataFrame({'Year': years, 'LUC Emissions': np.zeros(time_horizon)})

    if amortize_luc:
        if amortization_method == 'Equal Amortization':
            amortized_emission = luc_emissions / amortization_years
            for i in range(amortization_years):
                if i < time_horizon:
                    luc_emissions_df.loc[i, 'LUC Emissions'] = amortized_emission

        elif amortization_method == 'Linear Discounting':
            discount_factors = [9.7, 9.25, 8.75, 8.25, 7.75, 7.25, 6.75, 6.25, 5.75, 5.25,
                                4.75, 4.25, 3.75, 3.25, 2.75, 2.25, 1.75, 1.25, 0.75, 0.25]
            for i in range(min(amortization_years, len(discount_factors))):
                discounted_emission = luc_emissions * (discount_factors[i] / 100)
                if i < time_horizon:
                    luc_emissions_df.loc[i, 'LUC Emissions'] = discounted_emission

    else:
        # If not amortizing, apply all emissions at once
        if luc_application == "At LUC event year":
            # Apply emissions at the LUC event year
            index = 0  # The first index corresponds to the LUC event year
        else:
            # Apply emissions at the plantation start year
            plantation_start_year = st.session_state.get('land_prep_year', luc_event_year)
            index = plantation_start_year - luc_event_year  # Find the correct index

        if index < time_horizon:
            luc_emissions_df.loc[index, 'LUC Emissions'] = luc_emissions

    return luc_emissions_df


def luc_parameters():
    st.subheader('Land Use Change (LUC) Parameters')
    
    luc_event_year = st.number_input('Year of LUC event:', 
                                     min_value=1900, 
                                     max_value=2100, 
                                     value=st.session_state.get('luc_event_year', 2015), 
                                     step=1)
    luc_emissions = st.number_input('LUC emissions (tCO2e/ha):', 
                                    min_value=0.0, 
                                    value=st.session_state.get('luc_emissions', 5.0), 
                                    step=0.1)
    
    luc_application = st.radio(
        "When to apply LUC emissions:",
        ["At LUC event year", "At plantation start year"],
        index=0  
    )
    
    amortize_luc = st.checkbox('Amortize LUC emissions', value=st.session_state.get('amortize_luc', True))
    
    amortization_years = 20  
    amortization_method = 'Linear Amortization'
    if amortize_luc:
        amortization_method = st.radio(
            "Select Amortization Method:",
            ["Equal Amortization", "Linear Discounting"],
            index=1
        )
        if amortization_method == "Equal Amortization":
            amortization_years = st.number_input('Number of years for LUC emissions amortization:', 
                                                 min_value=1, 
                                                 value=st.session_state.get('amortization_years', 20), 
                                                 step=1)
    
    if st.button('Update LUC Parameters'):
        st.session_state['luc_event_year'] = luc_event_year
        st.session_state['luc_emissions'] = luc_emissions
        st.session_state['luc_application'] = luc_application
        st.session_state['amortize_luc'] = amortize_luc
        st.session_state['amortization_years'] = amortization_years
        st.session_state['amortization_method'] = amortization_method
        st.success('LUC parameters updated successfully!')

        # Generate the LUC emissions DataFrame based on the selected parameters
        time_horizon = st.session_state.get('time_horizon', 20)
        luc_emissions_df = generate_luc_emissions_df(time_horizon, luc_event_year, luc_emissions, amortization_years, amortization_method, luc_application, amortize_luc)

        # Display the LUC emissions DataFrame below the parameters
        st.subheader('LUC Emissions DataFrame')
        st.dataframe(luc_emissions_df)

    return luc_event_year, luc_emissions, luc_application, amortize_luc, amortization_years, amortization_method
