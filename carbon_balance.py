# carbon_balance.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import design_options as do
import helper_functions as hf
from reserve_rate_widget import reserve_rate_widget
from data_handling import apply_global_reserve_rate

def carbon_balance_section(time_horizon):
    if st.session_state.get('reserve_rate_changed', False):
        apply_global_reserve_rate()

    st.subheader('Graph: Emissions vs. Removals Over Time')

    if 'annual_emissions_df' not in st.session_state or st.session_state.annual_emissions_df is None:
        st.write("No emissions data available. Please complete the emissions section first.")
        return
    
    if 'adjusted_annual_removals_df' not in st.session_state or st.session_state.adjusted_annual_removals_df is None:
        st.write("No removals data available. Please complete the removals section first.")
        return

    emissions_df = st.session_state['annual_emissions_df']
    removals_df = st.session_state['adjusted_annual_removals_df']

    # Ensure we have the correct Year column
    if 'Year' not in emissions_df.columns or emissions_df['Year'].dtype == int:
        # If 'Year' is missing or is just a year number, create it based on land_prep_year
        land_prep_year = st.session_state.get('land_prep_year', 2015)
        emissions_df['Year'] = land_prep_year + emissions_df['Year Number'] - 1
        removals_df['Year'] = land_prep_year + removals_df['Year Number'] - 1

    cumulative_emissions = emissions_df['Total'].cumsum()
    cumulative_removals = removals_df['Total_Removals'].cumsum()
    carbon_balance = cumulative_removals - cumulative_emissions

    kpi_df = pd.DataFrame({
        'Year Number': np.arange(1, time_horizon + 1),
        'Year': emissions_df['Year'],
        'Emissions': cumulative_emissions,
        'Removals': cumulative_removals,
        'Carbon Balance': carbon_balance
    })
    st.session_state['kpi_df'] = kpi_df  # Save KPI DataFrame to session state

    st.altair_chart(hf.plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon), use_container_width=True)
    st.subheader('Graph: Carbon Balance Over Time')
    st.altair_chart(hf.plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon), use_container_width=True)
    st.subheader('Table: Carbon Balance Over Time')
    
    # Format the dataframe
    format_dict = {
        'Year Number': '{:0.0f}',
        'Year': '{:0.0f}',
        'Emissions': '{:.3f}',
        'Removals': '{:.3f}',
        'Carbon Balance': '{:.3f}'
    }
    st.dataframe(kpi_df.style.format(format_dict))
    
    st.subheader('Graph: Annual Total Emissions vs. Annual Total Removals')
    annual_emissions_vs_removals_df = pd.DataFrame({
        'Year Number': np.arange(1, time_horizon + 1),
        'Year': emissions_df['Year'],
        'Annual Emissions': emissions_df['Total'],
        'Annual Removals': removals_df['Total_Removals']
    })
    color_scale = alt.Scale(
        domain=['Annual Emissions', 'Annual Removals'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]
    )
    annual_emissions_vs_removals_chart = alt.Chart(annual_emissions_vs_removals_df).transform_fold(
        fold=['Annual Emissions', 'Annual Removals'],
        as_=['Type', 'Value']
    ).mark_line().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y='Value:Q',
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=[alt.Tooltip('Year:O', title='Year', format='d'), 'Value:Q', 'Type:N']
    ).interactive().properties(title='Annual Total Emissions vs. Annual Total Removals')
    st.altair_chart(annual_emissions_vs_removals_chart, use_container_width=True)
    st.subheader('Table: Annual Total Emissions vs. Annual Total Removals')
    
    # Format the dataframe
    format_dict = {
        'Year Number': '{:0.0f}',
        'Year': '{:0.0f}',
        'Annual Emissions': '{:.3f}',
        'Annual Removals': '{:.3f}'
    }
    st.dataframe(annual_emissions_vs_removals_df.style.format(format_dict))