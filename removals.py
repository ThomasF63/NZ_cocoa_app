import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from removals_calculations import calculate_removals
from helper_functions import create_removals_line_chart

def calculate_annual_removals(carbon_stocks_df):
    # Calculate the annual removals as the difference in carbon stocks between consecutive years
    annual_removals_df = carbon_stocks_df.diff().fillna(0)
    annual_removals_df['Year'] = carbon_stocks_df['Year']
    annual_removals_df.iloc[0, 1:] = carbon_stocks_df.iloc[0, 1:]  # First year removals are the same as carbon stocks
    return annual_removals_df

def prepare_removals_dataframes(time_horizon, cultivation_cycle_duration):
    if 'growth_params_df' not in st.session_state or st.session_state.growth_params_df is None:
        st.write("No growth data available.")
        return None

    growth_params_df = st.session_state['growth_params_df']

    # Calculate annual removals
    annual_removals_df = calculate_annual_removals(growth_params_df)

    # Ensure the 'Year' column is correctly inserted
    if 'Year' not in annual_removals_df.columns:
        year_data = np.arange(1, time_horizon + 1)
        annual_removals_df.insert(0, 'Year', year_data)
    else:
        annual_removals_df['Year'] = np.arange(1, time_horizon + 1)
    
    annual_removals_df['Total_Removals'] = annual_removals_df.drop(columns='Year').sum(axis=1)
    st.session_state['annual_removals_df'] = annual_removals_df  # Save to session state

    return annual_removals_df

def removals_analysis(time_horizon):
    prepare_removals_dataframes(time_horizon, st.session_state.get('cultivation_cycle_duration', 20))

    if 'annual_removals_df' not in st.session_state or st.session_state.annual_removals_df is None:
        st.write("No removals data available. Please complete the tree growth section first.")
        return

    annual_removals_df = st.session_state['annual_removals_df']

    st.subheader('Graph: Annual Removals Over Time')
    removals_chart = create_removals_line_chart(annual_removals_df, 'Year', 'Total_Removals', 'Tree Type', 'Annual Removals Over Time')
    st.altair_chart(removals_chart, use_container_width=True)

    st.subheader('Table: Annual Removals Over Time')
    st.dataframe(annual_removals_df)

    csv = annual_removals_df.to_csv(index=False)
    st.download_button(
        label="Download Annual Removals Data as CSV",
        data=csv,
        file_name='annual_removals_data.csv',
        mime='text/csv',
        key='download_annual_removals_data'
    )

    st.subheader('Graph: Cumulative Removals Over Time')
    cumulative_removals_df = annual_removals_df.copy()
    cumulative_removals_df.iloc[:, 1:-1] = cumulative_removals_df.iloc[:, 1:-1].cumsum()
    cumulative_removals_chart = create_removals_line_chart(cumulative_removals_df, 'Year', 'Total_Removals', 'Tree Type', 'Cumulative Removals Over Time')
    st.altair_chart(cumulative_removals_chart, use_container_width=True)

    st.subheader('Table: Cumulative Removals Over Time')
    st.dataframe(cumulative_removals_df)

    csv_cumulative = cumulative_removals_df.to_csv(index=False)
    st.download_button(
        label="Download Cumulative Removals Data as CSV",
        data=csv_cumulative,
        file_name='cumulative_removals_data.csv',
        mime='text/csv',
        key='download_cumulative_removals_data'
    )

def plot_removals_by_tree_type(scenario):
    st.subheader(f'Annual Removals by Tree Type for Scenario {scenario}')
    
    annual_removals_by_tree_type = st.session_state.get(f'{scenario}_annual_removals_by_tree_type', None)
    if annual_removals_by_tree_type is None:
        st.write("No data available for this scenario.")
        return

    removals_df = pd.DataFrame.from_dict(annual_removals_by_tree_type, orient='index')
    removals_df['Year'] = removals_df.index
    removals_df = removals_df.melt(id_vars='Year', var_name='Tree Type', value_name='Carbon Stock (tCO2e/ha)')
    
    stacked_bar_chart = alt.Chart(removals_df).mark_bar().encode(
        x='Year:Q',
        y='Carbon Stock (tCO2e/ha):Q',
        color='Tree Type:N',
        tooltip=['Year:Q', 'Carbon Stock (tCO2e/ha):Q', 'Tree Type:N']
    ).interactive().properties(title=f'Annual Removals by Tree Type for Scenario {scenario}')
    
    st.altair_chart(stacked_bar_chart, use_container_width=True)
