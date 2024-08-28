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

    st.header('Carbon Balance Section', divider="gray")

    if 'annual_emissions_df' not in st.session_state or st.session_state.annual_emissions_df is None:
        st.write("No emissions data available. Please complete the emissions section first.")
        return
    
    if 'adjusted_annual_removals_df' not in st.session_state or st.session_state.adjusted_annual_removals_df is None:
        st.write("No removals data available. Please complete the removals section first.")
        return

    emissions_df = st.session_state['annual_emissions_df']
    removals_df = st.session_state['adjusted_annual_removals_df']

    cumulative_emissions = emissions_df['Total'].cumsum()
    cumulative_removals = removals_df['Total_Removals'].cumsum()
    cumulative_carbon_balance = cumulative_emissions - cumulative_removals
    annual_carbon_balance = emissions_df['Total'] - removals_df['Total_Removals']

    kpi_df = pd.DataFrame({
        'Year Number': np.arange(1, time_horizon + 1),
        'Year': emissions_df['Year'],
        'Cumulative Emissions': cumulative_emissions,
        'Cumulative Removals': cumulative_removals,
        'Cumulative Carbon Balance': cumulative_carbon_balance,
        'Annual Emissions': emissions_df['Total'],
        'Annual Removals': removals_df['Total_Removals'],
        'Annual Carbon Balance': annual_carbon_balance
    })
    st.session_state['kpi_df'] = kpi_df  # Save KPI DataFrame to session state

    st.subheader('Graph: Cumulative Emissions vs. Cumulative Removals Over Time')
    st.altair_chart(hf.plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon), use_container_width=True)
    
    st.subheader('Graph: Cumulative Carbon Balance Over Time')
    st.altair_chart(hf.plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon), use_container_width=True)
    
    st.subheader('Graph: Annual Emissions vs. Annual Removals Over Time')
    st.altair_chart(hf.plot_annual_emissions_vs_removals_one_hectare(kpi_df, time_horizon), use_container_width=True)
    
    st.subheader('Graph: Annual Carbon Balance Over Time')
    st.altair_chart(hf.plot_annual_carbon_balance_bars_one_hectare(kpi_df, time_horizon), use_container_width=True)
    
    st.subheader('Table: Annual and Cumulative Carbon Balance Over Time')
    
    # Format the dataframe
    format_dict = {
        'Year Number': '{:0.0f}',
        'Year': '{:0.0f}',
        'Cumulative Emissions': '{:.3f}',
        'Cumulative Removals': '{:.3f}',
        'Cumulative Carbon Balance': '{:.3f}',
        'Annual Emissions': '{:.3f}',
        'Annual Removals': '{:.3f}',
        'Annual Carbon Balance': '{:.3f}'
    }
    st.dataframe(kpi_df.style.format(format_dict))