import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import design_options as do
import helper_functions as hf
from emissions import update_emissions_df



def initialize_inputs():
    if 'time_horizon' not in st.session_state:
        st.session_state['time_horizon'] = 10
    if 'cultivation_cycle_duration' not in st.session_state:
        st.session_state['cultivation_cycle_duration'] = 20
    if 'luc_emission_approach' not in st.session_state:
        st.session_state['luc_emission_approach'] = 'At Start'
    if 'amortization_years' not in st.session_state:
        st.session_state['amortization_years'] = 10
    if 'growth_curve_selections' not in st.session_state:
        st.session_state['growth_curve_selections'] = {}
    if 'default_params' not in st.session_state:
        st.session_state['default_params'] = {}
    if 'emissions_df' not in st.session_state:
        st.session_state['emissions_df'] = pd.DataFrame(columns=['Year', 'LUC Emissions', 'Replanting Emissions'])
    if 'planting_densities' not in st.session_state:
        st.session_state['planting_densities'] = {
            'reference': {'Cocoa': 1800, 'Shade': 150, 'Timber': 33},
            'modified': {'Cocoa': 1950, 'Shade': 0, 'Timber': 33}
        }
        st.session_state.selected_density = 'reference'



def simulation_parameters():
    st.subheader('Simulation Parameters')

    time_horizon = st.number_input('Enter the time horizon for the simulation (years):', min_value=1, value=st.session_state.get('time_horizon', 10), step=1, key='time_horizon_widget')
    cultivation_cycle_duration = st.number_input('Enter the duration of the cultivation cycle (years):', min_value=1, value=st.session_state.get('cultivation_cycle_duration', 20), step=1, key='cultivation_cycle_duration_widget')
    luc_emission_approach = st.radio(
        "Select LUC Emissions Approach",
        options=["At Start", "Amortized"],
        index=0 if st.session_state.get('luc_emission_approach', 'At Start') == 'At Start' else 1,
        key='luc_emission_approach_widget'
    )

    amortization_years = st.session_state.get('amortization_years', 10)

    if luc_emission_approach == "Amortized":
        col1, col2 = st.columns([3, 1])
        with col1:
            amortization_years = st.number_input('Enter the number of years for LUC emissions amortization:', min_value=1, value=amortization_years, step=1, key='amortization_years_widget')
        with col2:
            if st.button('Update Amortization Years'):
                st.session_state['amortization_years'] = amortization_years
                update_emissions_df(st.session_state.get('cultivation_cycle_duration', 20), st.session_state.get('luc_emissions', 100.0), st.session_state.get('replanting_emissions', 50.0))

    st.session_state['time_horizon'] = time_horizon
    st.session_state['cultivation_cycle_duration'] = cultivation_cycle_duration
    st.session_state['luc_emission_approach'] = luc_emission_approach
    if luc_emission_approach == "Amortized":
        st.session_state['amortization_years'] = amortization_years

    return time_horizon, cultivation_cycle_duration, luc_emission_approach, amortization_years




def cocoa_yield_curve(cultivation_cycle_duration):
    st.subheader('Production Curve: Cocoa Yield Over Time')

    default_yield = {
        'Year': list(range(1, 21)),
        'Cocoa Yield (t/ha/yr)': [None if x == 0 else x for x in [0, 0, 153, 583, 1013, 1442, 1872, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250]]
    }

    max_yield = max(filter(lambda x: x is not None, default_yield['Cocoa Yield (t/ha/yr)']))
    default_yield['Relative Yield (%maximum)'] = [f"{int((x / max_yield) * 100)}%" if x is not None else "0%" for x in default_yield['Cocoa Yield (t/ha/yr)']]

    if 'cocoa_yield_df' not in st.session_state:
        st.session_state.cocoa_yield_df = pd.DataFrame(default_yield)
    
    cocoa_yield_df = st.data_editor(data=st.session_state.cocoa_yield_df, key="cocoa_yield")
    st.session_state.cocoa_yield_df = cocoa_yield_df

    uploaded_yield_file = st.file_uploader("Upload your cocoa yield data CSV", type=['csv'])
    if uploaded_yield_file is not None:
        custom_yield_df = pd.read_csv(uploaded_yield_file)
        if len(custom_yield_df) == cultivation_cycle_duration:
            st.session_state.cocoa_yield_df = custom_yield_df
        else:
            st.error(f"Please ensure the uploaded CSV has exactly {cultivation_cycle_duration} rows to match the cultivation cycle duration.")

    st.subheader('Graph: Annual Cocoa Yield Over Time')
    yield_chart = alt.Chart(cocoa_yield_df).mark_bar().encode(
        x='Year:Q',
        y='Cocoa Yield (t/ha/yr):Q',
        tooltip=['Year:Q', 'Cocoa Yield (t/ha/yr):Q']
    ).properties(title='Annual Cocoa Yield Over Time')
    st.altair_chart(yield_chart, use_container_width=True)
    return cocoa_yield_df




def carbon_balance_section(time_horizon):
    st.subheader('Graph: Emissions vs. Removals Over Time')

    if 'annual_emissions_df' not in st.session_state or st.session_state.annual_emissions_df is None:
        st.write("No emissions data available. Please complete the emissions section first.")
        return
    
    if 'annual_removals_df' not in st.session_state or st.session_state.annual_removals_df is None:
        st.write("No removals data available. Please complete the removals section first.")
        return

    emissions_df = st.session_state['annual_emissions_df']
    removals_df = st.session_state['annual_removals_df']

    # Calculate cumulative values and carbon balance
    cumulative_emissions = emissions_df['Total'].cumsum()
    cumulative_removals = removals_df['Total_Removals'].cumsum()
    carbon_balance = cumulative_removals - cumulative_emissions

    # Create the KPI DataFrame
    kpi_df = pd.DataFrame({
        'Year': np.arange(1, time_horizon + 1),
        'Emissions': cumulative_emissions,
        'Removals': cumulative_removals,
        'Carbon Balance': carbon_balance
    })
    st.session_state['kpi_df'] = kpi_df  # Save KPI DataFrame to session state

    # Graph: Emissions vs. Removals Over Time
    st.altair_chart(hf.plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon), use_container_width=True)

    # Graph: Carbon Balance Over Time
    st.subheader('Graph: Carbon Balance Over Time')
    st.altair_chart(hf.plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon), use_container_width=True)

    # Table: Carbon Balance Over Time
    st.subheader('Table: Carbon Balance Over Time')
    st.dataframe(kpi_df)

    # Graph: Annual Total Emissions vs. Annual Total Removals
    st.subheader('Graph: Annual Total Emissions vs. Annual Total Removals')
    annual_emissions_vs_removals_df = pd.DataFrame({
        'Year': np.arange(1, time_horizon + 1),
        'Annual Emissions': np.diff(emissions_df['Total'], prepend=0),
        'Annual Removals': np.diff(removals_df['Total_Removals'], prepend=0)
    })

    color_scale = alt.Scale(
        domain=['Annual Emissions', 'Annual Removals'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]
    )

    annual_emissions_vs_removals_chart = alt.Chart(annual_emissions_vs_removals_df).transform_fold(
        fold=['Annual Emissions', 'Annual Removals'],
        as_=['Type', 'Value']
    ).mark_line().encode(
        x='Year:Q',
        y='Value:Q',
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=['Year:Q', 'Value:Q', 'Type:N']
    ).interactive().properties(title='Annual Total Emissions vs. Annual Total Removals')

    st.altair_chart(annual_emissions_vs_removals_chart, use_container_width=True)

    # Table: Annual Total Emissions vs. Annual Total Removals
    st.subheader('Table: Annual Total Emissions vs. Annual Total Removals')
    st.dataframe(annual_emissions_vs_removals_df)




def carbon_intensity_section(time_horizon):
    st.subheader('Carbon Intensity Section')

    if 'annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'cocoa_yield_df' in st.session_state:
        annual_removals_df = st.session_state['annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        cocoa_yield_df = st.session_state['cocoa_yield_df']

        total_emissions = annual_emissions_df['Total'].values
        total_removals = annual_removals_df['Total_Removals'].values
        cocoa_yield = cocoa_yield_df['Cocoa Yield (t/ha/yr)'].values

        # Ensure all arrays are of the same length
        min_length = min(len(total_emissions), len(total_removals), len(cocoa_yield))
        total_emissions = total_emissions[:min_length]
        total_removals = total_removals[:min_length]
        cocoa_yield = cocoa_yield[:min_length]

        carbon_intensity = (total_emissions - total_removals) / cocoa_yield
        carbon_intensity = np.where(cocoa_yield == 0, np.nan, carbon_intensity)  # Handle division by zero

        carbon_intensity_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Total Emissions': total_emissions,
            'Total Removals': total_removals,
            'Cocoa Yield (t/ha/yr)': cocoa_yield,
            'Carbon Intensity (tCO2e/t Cocoa)': carbon_intensity
        })

        # Replace NaN with a placeholder value compatible with Arrow serialization
        carbon_intensity_df['Carbon Intensity (tCO2e/t Cocoa)'] = carbon_intensity_df['Carbon Intensity (tCO2e/t Cocoa)'].replace({np.nan: None})

        st.session_state['carbon_intensity_df'] = carbon_intensity_df

        st.subheader('Graph: Carbon Intensity Over Time')
        carbon_intensity_chart = alt.Chart(carbon_intensity_df).mark_line().encode(
            x='Year:Q',
            y=alt.Y('Carbon Intensity (tCO2e/t Cocoa):Q', scale=alt.Scale(zero=False)),
            tooltip=['Year', 'Carbon Intensity (tCO2e/t Cocoa)']
        ).properties(title='Carbon Intensity Over Time')
        st.altair_chart(carbon_intensity_chart, use_container_width=True)

        st.subheader('Table: Carbon Intensity Data')
        st.dataframe(carbon_intensity_df)

        csv = carbon_intensity_df.to_csv(index=False)
        st.download_button(
            label="Download Carbon Intensity Data as CSV",
            data=csv,
            file_name='carbon_intensity_data.csv',
            mime='text/csv'
        )
    else:
        st.write("No data available for carbon intensity calculation. Please complete the emissions, removals, and cocoa yield sections first.")




def summary_section(time_horizon):
    st.subheader('Summary Section')

    if 'annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'cocoa_yield_df' in st.session_state:
        annual_removals_df = st.session_state['annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        cocoa_yield_df = st.session_state['cocoa_yield_df']

        total_emissions = annual_emissions_df['Total'].values
        total_removals = annual_removals_df['Total_Removals'].values
        cocoa_yield = cocoa_yield_df['Cocoa Yield (t/ha/yr)'].values

        # Ensure all arrays are of the same length
        min_length = min(len(total_emissions), len(total_removals), len(cocoa_yield))
        total_emissions = total_emissions[:min_length]
        total_removals = total_removals[:min_length]
        cocoa_yield = cocoa_yield[:min_length]

        summary_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Emissions': total_emissions,
            'Removals': total_removals,
            'Cocoa Yield (t/ha/yr)': cocoa_yield
        })

        st.session_state['summary_df'] = summary_df

        st.subheader('Graph: Cumulative Emissions, Removals, and Cocoa Yield Over Time')
        cumulative_summary_chart = hf.plot_summary_emissions_removals_cocoa_production(summary_df)
        st.altair_chart(cumulative_summary_chart, use_container_width=True)

        st.subheader('Table: Cumulative Summary Data')
        st.dataframe(summary_df)

        # Annual Analysis
        annual_emissions = np.diff(np.insert(total_emissions, 0, 0))
        annual_removals = np.diff(np.insert(total_removals, 0, 0))

        annual_summary_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Annual Emissions': annual_emissions,
            'Annual Removals': annual_removals,
            'Cocoa Yield': cocoa_yield
        })

        st.session_state['annual_summary_df'] = annual_summary_df

        st.subheader('Graph: Annual Emissions, Removals, and Cocoa Yield Over Time')
        annual_summary_chart = hf.plot_annual_emissions_removals_cocoa_yield(annual_summary_df)
        st.altair_chart(annual_summary_chart, use_container_width=True)

        st.subheader('Table: Annual Summary Data')
        st.dataframe(annual_summary_df)
    else:
        st.write("No data available for summary. Please complete the emissions, removals, and cocoa yield sections first.")




