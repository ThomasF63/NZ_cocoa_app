import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import design_options as do
from reserve_rate_widget import reserve_rate_widget
from data_handling import apply_global_reserve_rate

def summary_section(time_horizon):

    if st.session_state.get('reserve_rate_changed', False):
        apply_global_reserve_rate()

    st.subheader('Summary Section')

    if 'adjusted_annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'extended_yield_df' in st.session_state:
        annual_removals_df = st.session_state['adjusted_annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        extended_yield_df = st.session_state['extended_yield_df']

        summary_df = pd.DataFrame({
            'Year Number': annual_emissions_df['Year Number'],
            'Year': annual_emissions_df['Year'],
            'Emissions': annual_emissions_df['Total'].cumsum(),
            'Removals': annual_removals_df['Total_Removals'].cumsum(),
            'Cocoa Yield (kg/ha/yr)': extended_yield_df['Cocoa Yield (kg/ha/yr)']
        })

        st.session_state['summary_df'] = summary_df

        st.subheader('Graph: Cumulative Emissions, Removals, and Cocoa Yield Over Time')
        
        # Create base chart
        base = alt.Chart(summary_df).encode(x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')))

        # Create lines for Emissions and Removals
        emissions_removals = base.mark_line().encode(
            y=alt.Y('Emissions:Q', axis=alt.Axis(title='Cumulative Emissions and Removals (tCO2e)')),
            color=alt.value(do.EMISSIONS_COLOR)
        ) + base.mark_line().encode(
            y=alt.Y('Removals:Q', axis=alt.Axis(title='Cumulative Emissions and Removals (tCO2e)')),
            color=alt.value(do.REMOVALS_COLOR)
        )

        # Create line for Cocoa Yield
        cocoa_yield = base.mark_line().encode(
            y=alt.Y('Cocoa Yield (kg/ha/yr):Q', axis=alt.Axis(title='Cocoa Yield (kg/ha/yr)')),
            color=alt.value(do.COCOA_YIELD_COLOR)
        )

        # Combine charts
        cumulative_summary_chart = alt.layer(emissions_removals, cocoa_yield).resolve_scale(
            y='independent'
        ).properties(
            title='Cumulative Emissions, Removals, and Cocoa Yield Over Time',
            width=600,
            height=400
        ).interactive()

        st.altair_chart(cumulative_summary_chart, use_container_width=True)

        st.subheader('Table: Cumulative Summary Data')
        format_dict = {
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}',
            'Emissions': '{:.3f}',
            'Removals': '{:.3f}',
            'Cocoa Yield (kg/ha/yr)': '{:.3f}'
        }
        st.dataframe(summary_df.style.format(format_dict))

        annual_summary_df = pd.DataFrame({
            'Year Number': annual_emissions_df['Year Number'],
            'Year': annual_emissions_df['Year'],
            'Annual Emissions': annual_emissions_df['Total'],
            'Annual Removals': annual_removals_df['Total_Removals'],
            'Cocoa Yield': extended_yield_df['Cocoa Yield (kg/ha/yr)']
        })

        st.session_state['annual_summary_df'] = annual_summary_df

        st.subheader('Graph: Annual Emissions, Removals, and Cocoa Yield Over Time')
        
        # Create base chart for annual data
        annual_base = alt.Chart(annual_summary_df).encode(x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')))

        # Create lines for Annual Emissions and Removals
        annual_emissions_removals = annual_base.mark_line().encode(
            y=alt.Y('Annual Emissions:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e)')),
            color=alt.value(do.EMISSIONS_COLOR)
        ) + annual_base.mark_line().encode(
            y=alt.Y('Annual Removals:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e)')),
            color=alt.value(do.REMOVALS_COLOR)
        )

        # Create line for Annual Cocoa Yield
        annual_cocoa_yield = annual_base.mark_line().encode(
            y=alt.Y('Cocoa Yield:Q', axis=alt.Axis(title='Cocoa Yield (kg/ha/yr)')),
            color=alt.value(do.COCOA_YIELD_COLOR)
        )

        # Combine annual charts
        annual_summary_chart = alt.layer(annual_emissions_removals, annual_cocoa_yield).resolve_scale(
            y='independent'
        ).properties(
            title='Annual Emissions, Removals, and Cocoa Yield Over Time',
            width=600,
            height=400
        ).interactive()

        st.altair_chart(annual_summary_chart, use_container_width=True)

        st.subheader('Table: Annual Summary Data')
        format_dict = {
            'Year Number': '{:0.0f}',
            'Year': '{:0.0f}',
            'Annual Emissions': '{:.3f}',
            'Annual Removals': '{:.3f}',
            'Cocoa Yield': '{:.3f}'
        }
        st.dataframe(annual_summary_df.style.format(format_dict))
    else:
        st.write("No data available for summary. Please complete the emissions, removals, and cocoa yield sections first.")