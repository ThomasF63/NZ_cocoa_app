import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from reserve_rate_widget import reserve_rate_widget
from data_handling import apply_global_reserve_rate

def calculate_carbon_intensity(annual_removals_df, annual_emissions_df, extended_yield_df):
    # Ensure all dataframes have the same length
    min_length = min(len(annual_removals_df), len(annual_emissions_df), len(extended_yield_df))
    
    total_emissions = annual_emissions_df['Total'].values[:min_length]
    total_removals = annual_removals_df['Total_Removals'].values[:min_length]
    cocoa_yield = extended_yield_df['Cocoa Yield (kg/ha/yr)'].values[:min_length]
    year_numbers = annual_emissions_df['Year Number'].values[:min_length]
    years = annual_emissions_df['Year'].values[:min_length]

    # Annual Carbon Intensity
    net_emissions = total_emissions - total_removals
    annual_carbon_intensity = net_emissions / (cocoa_yield / 1000)  # Convert kg to tonnes
    annual_carbon_intensity = np.where(cocoa_yield == 0, np.nan, annual_carbon_intensity)  # Handle division by zero

    annual_carbon_intensity_df = pd.DataFrame({
        'Year Number': year_numbers,
        'Year': years,
        'Total Emissions': total_emissions,
        'Total Removals': total_removals,
        'Net Emissions': net_emissions,
        'Cocoa Yield (kg/ha/yr)': cocoa_yield,
        'Annual Carbon Intensity (tCO2e/t Cocoa)': annual_carbon_intensity
    })

    # Cumulative Carbon Intensity
    cumulative_emissions = np.cumsum(total_emissions)
    cumulative_removals = np.cumsum(total_removals)
    cumulative_net_emissions = cumulative_emissions - cumulative_removals
    cumulative_yield = np.cumsum(cocoa_yield)
    cumulative_carbon_intensity = cumulative_net_emissions / (cumulative_yield / 1000)  # Convert kg to tonnes
    cumulative_carbon_intensity = np.where(cumulative_yield == 0, np.nan, cumulative_carbon_intensity)  # Handle division by zero

    cumulative_carbon_intensity_df = pd.DataFrame({
        'Year Number': year_numbers,
        'Year': years,
        'Cumulative Emissions': cumulative_emissions,
        'Cumulative Removals': cumulative_removals,
        'Net Emissions': cumulative_net_emissions,
        'Cumulative Cocoa Yield (kg/ha)': cumulative_yield,
        'Cumulative Carbon Intensity (tCO2e/t Cocoa)': cumulative_carbon_intensity
    })

    return annual_carbon_intensity_df, cumulative_carbon_intensity_df

def create_carbon_intensity_chart(df, x_col, y_col, title):
    base = alt.Chart(df).encode(
        x=alt.X(f'{x_col}:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y(f'{y_col}:Q', title='Carbon Intensity (tCO2e/t Cocoa)'),
        tooltip=[alt.Tooltip(f'{x_col}:O', title='Year', format='d'), alt.Tooltip(y_col, format='.3f')]
    )

    line = base.mark_line(color='grey')

    points = base.mark_circle(size=60).encode(
        color=alt.condition(
            alt.datum[y_col] > 0,
            alt.value('red'),
            alt.value('green')
        )
    )

    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black').encode(y='y')

    chart = (line + points + zero_line).properties(
        title=title,
        width=600,
        height=400
    ).interactive()

    return chart

def carbon_intensity_section(time_horizon):

    if st.session_state.get('reserve_rate_changed', False):
        apply_global_reserve_rate()

    st.header('Carbon Intensity', divider="gray")

    if 'adjusted_annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'extended_yield_df' in st.session_state:
        annual_removals_df = st.session_state['adjusted_annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        extended_yield_df = st.session_state['extended_yield_df']

        annual_carbon_intensity_df, cumulative_carbon_intensity_df = calculate_carbon_intensity(
            annual_removals_df, annual_emissions_df, extended_yield_df
        )

        # Display annual carbon intensity
        st.subheader('Annual Carbon Intensity')
        annual_ci_chart = create_carbon_intensity_chart(
            annual_carbon_intensity_df,
            'Year',
            'Annual Carbon Intensity (tCO2e/t Cocoa)',
            'Annual Carbon Intensity Over Time'
        )
        st.altair_chart(annual_ci_chart, use_container_width=True)
        
        # Table
        st.subheader('Table: Annual Carbon Intensity')
        format_dict = {
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}',
            'Total Emissions': '{:.3f}',
            'Total Removals': '{:.3f}',
            'Cocoa Yield (kg/ha/yr)': '{:.3f}',
            'Annual Carbon Intensity (tCO2e/t Cocoa)': '{:.3f}'
        }
        st.dataframe(annual_carbon_intensity_df.style.format(format_dict))

        # Display cumulative carbon intensity
        st.subheader('Cumulative Carbon Intensity')
        cumulative_ci_chart = create_carbon_intensity_chart(
            cumulative_carbon_intensity_df,
            'Year',
            'Cumulative Carbon Intensity (tCO2e/t Cocoa)',
            'Cumulative Carbon Intensity Over Time'
        )
        st.altair_chart(cumulative_ci_chart, use_container_width=True)
        
        # Table
        st.subheader('Table: Cumulative Carbon Intensity')
        format_dict = {
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}',
            'Cumulative Emissions': '{:.3f}',
            'Cumulative Removals': '{:.3f}',
            'Cumulative Cocoa Yield (kg/ha)': '{:.3f}',
            'Cumulative Carbon Intensity (tCO2e/t Cocoa)': '{:.3f}'
        }
        st.dataframe(cumulative_carbon_intensity_df.style.format(format_dict))

    else:
        st.write("No data available for carbon intensity calculation. Please complete the emissions, removals, and cocoa yield sections first.")