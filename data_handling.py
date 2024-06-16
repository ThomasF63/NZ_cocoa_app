# data_handling.py
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
    if 'scenarios_df' not in st.session_state:
        default_data = {
            "Scenario": ["A", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
            "Block": [1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Area (ha)": [500, 500, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "Year": [1, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Start from year 1
        }
        st.session_state.scenarios_df = pd.DataFrame(default_data)




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

    # Add reserve rate input
    apply_reserve_rate = st.checkbox('Apply Reserve Rate to Removals', value=st.session_state.get('apply_reserve_rate', True))

    if apply_reserve_rate:
        # Set default value to 50% if the reserve rate is not already set
        default_reserve_rate = st.session_state.get('reserve_rate', 0.5) * 100.0 if 'reserve_rate' in st.session_state else 50.0
        reserve_rate = st.number_input('Reserve Rate (%)', min_value=0.0, max_value=100.0, value=default_reserve_rate, step=0.1)
        
        if st.button('Update Reserve Rate'):
            st.session_state['apply_reserve_rate'] = apply_reserve_rate
            st.session_state['reserve_rate'] = reserve_rate / 100.0  # Store as a fraction
            st.success(f"Reserve rate updated to {reserve_rate}%")
    else:
        st.session_state['apply_reserve_rate'] = False
        st.session_state['reserve_rate'] = 0.0  # Directly set to 0% reserve rate when not applying

    st.session_state['time_horizon'] = time_horizon
    st.session_state['cultivation_cycle_duration'] = cultivation_cycle_duration
    st.session_state['luc_emission_approach'] = luc_emission_approach
    if luc_emission_approach == "Amortized":
        st.session_state['amortization_years'] = amortization_years

    return time_horizon, cultivation_cycle_duration, luc_emission_approach, amortization_years




def cocoa_yield_curve(cultivation_cycle_duration, time_horizon):
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

    # Extend cocoa yield data to match the time horizon
    yield_values = st.session_state.cocoa_yield_df['Cocoa Yield (t/ha/yr)'].values
    extended_yield = np.tile(yield_values, (time_horizon // cultivation_cycle_duration) + 1)[:time_horizon]

    # Create extended dataframe
    extended_yield_df = pd.DataFrame({
        'Year': np.arange(1, time_horizon + 1),
        'Cocoa Yield (t/ha/yr)': extended_yield
    })
    st.session_state.extended_yield_df = extended_yield_df

    st.subheader('Graph: Annual Cocoa Yield Over Time')
    yield_chart = alt.Chart(extended_yield_df).mark_bar().encode(
        x=alt.X('Year:Q', title='Year', scale=alt.Scale(domain=[1, time_horizon]), axis=alt.Axis(tickMinStep=1, tickCount=time_horizon, labelExpr="datum.value % 1 === 0 ? datum.value : ''")),
        y=alt.Y('Cocoa Yield (t/ha/yr):Q', title='Cocoa Yield (t/ha/yr)'),
        tooltip=['Year:Q', 'Cocoa Yield (t/ha/yr):Q']
    ).properties(title='Annual Cocoa Yield Over Time')
    st.altair_chart(yield_chart, use_container_width=True)
    
    return extended_yield_df




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

    cumulative_emissions = emissions_df['Total'].cumsum()
    cumulative_removals = removals_df['Total_Removals'].cumsum()
    carbon_balance = cumulative_removals - cumulative_emissions

    kpi_df = pd.DataFrame({
        'Year': np.arange(1, time_horizon + 1),
        'Emissions': cumulative_emissions,
        'Removals': cumulative_removals,
        'Carbon Balance': carbon_balance
    })
    st.session_state['kpi_df'] = kpi_df  # Save KPI DataFrame to session state

    # Display reserve rate information
    if st.session_state.get('apply_reserve_rate', True):
        st.info(f"Reserve rate of {st.session_state.get('reserve_rate', 0.5) * 100:.0f}% applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")
    else:
        st.info("No reserve rate applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")

    st.altair_chart(hf.plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon), use_container_width=True)
    st.subheader('Graph: Carbon Balance Over Time')
    st.altair_chart(hf.plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon), use_container_width=True)
    st.subheader('Table: Carbon Balance Over Time')
    st.dataframe(kpi_df)
    
    st.subheader('Graph: Annual Total Emissions vs. Annual Total Removals')
    annual_emissions_vs_removals_df = pd.DataFrame({
        'Year': np.arange(1, time_horizon + 1),
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
        x='Year:Q',
        y='Value:Q',
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=['Year:Q', 'Value:Q', 'Type:N']
    ).interactive().properties(title='Annual Total Emissions vs. Annual Total Removals')
    st.altair_chart(annual_emissions_vs_removals_chart, use_container_width=True)
    st.subheader('Table: Annual Total Emissions vs. Annual Total Removals')
    st.dataframe(annual_emissions_vs_removals_df)



def carbon_intensity_section(time_horizon):
    st.subheader('Carbon Intensity Section')

    if 'annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'extended_yield_df' in st.session_state:
        annual_removals_df = st.session_state['annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        extended_yield_df = st.session_state['extended_yield_df']

        total_emissions = annual_emissions_df['Total'].values
        total_removals = annual_removals_df['Total_Removals'].values
        cocoa_yield = extended_yield_df['Cocoa Yield (t/ha/yr)'].values

        min_length = min(len(total_emissions), len(total_removals), len(cocoa_yield))
        total_emissions = total_emissions[:min_length]
        total_removals = total_removals[:min_length]
        cocoa_yield = cocoa_yield[:min_length]

        # Annual Carbon Intensity
        annual_carbon_intensity = (total_emissions - total_removals) / cocoa_yield
        annual_carbon_intensity = np.where(cocoa_yield == 0, np.nan, annual_carbon_intensity)  # Handle division by zero

        annual_carbon_intensity_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Total Emissions': total_emissions,
            'Total Removals': total_removals,
            'Cocoa Yield (t/ha/yr)': cocoa_yield,
            'Annual Carbon Intensity (tCO2e/t Cocoa)': annual_carbon_intensity
        })

        annual_carbon_intensity_df['Annual Carbon Intensity (tCO2e/t Cocoa)'] = annual_carbon_intensity_df['Annual Carbon Intensity (tCO2e/t Cocoa)'].replace({np.nan: None})

        # Cumulative Carbon Intensity
        cumulative_emissions = np.cumsum(total_emissions)
        cumulative_removals = np.cumsum(total_removals)
        cumulative_carbon_intensity = (cumulative_emissions - cumulative_removals) / cocoa_yield
        cumulative_carbon_intensity = np.where(cocoa_yield == 0, np.nan, cumulative_carbon_intensity)  # Handle division by zero

        cumulative_carbon_intensity_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Cumulative Emissions': cumulative_emissions,
            'Cumulative Removals': cumulative_removals,
            'Cocoa Yield (t/ha/yr)': cocoa_yield,
            'Cumulative Carbon Intensity (tCO2e/t Cocoa)': cumulative_carbon_intensity
        })

        cumulative_carbon_intensity_df['Cumulative Carbon Intensity (tCO2e/t Cocoa)'] = cumulative_carbon_intensity_df['Cumulative Carbon Intensity (tCO2e/t Cocoa)'].replace({np.nan: None})

        # Display reserve rate information
        if st.session_state.get('apply_reserve_rate', True):
            st.info(f"Reserve rate of {st.session_state.get('reserve_rate', 0.5) * 100:.0f}% applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")
        else:
            st.info("No reserve rate applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")

        # Annual Carbon Intensity Graph and Table
        st.subheader('Graph: Annual Carbon Intensity Over Time')
        annual_carbon_intensity_chart = alt.Chart(annual_carbon_intensity_df).mark_line().encode(
            x='Year:Q',
            y=alt.Y('Annual Carbon Intensity (tCO2e/t Cocoa):Q', scale=alt.Scale(zero=False)),
            tooltip=['Year', 'Annual Carbon Intensity (tCO2e/t Cocoa)']
        ).properties(title='Annual Carbon Intensity Over Time')
        st.altair_chart(annual_carbon_intensity_chart, use_container_width=True)

        st.subheader('Table: Annual Carbon Intensity Data')
        st.dataframe(annual_carbon_intensity_df)

        # Cumulative Carbon Intensity Graph and Table
        st.subheader('Graph: Cumulative Carbon Intensity Over Time')
        cumulative_carbon_intensity_chart = alt.Chart(cumulative_carbon_intensity_df).mark_line().encode(
            x='Year:Q',
            y=alt.Y('Cumulative Carbon Intensity (tCO2e/t Cocoa):Q', scale=alt.Scale(zero=False)),
            tooltip=['Year', 'Cumulative Carbon Intensity (tCO2e/t Cocoa)']
        ).properties(title='Cumulative Carbon Intensity Over Time')
        st.altair_chart(cumulative_carbon_intensity_chart, use_container_width=True)

        st.subheader('Table: Cumulative Carbon Intensity Data')
        st.dataframe(cumulative_carbon_intensity_df)
    else:
        st.write("No data available for carbon intensity calculation. Please complete the emissions, removals, and cocoa yield sections first.")






def summary_section(time_horizon):
    st.subheader('Summary Section')

    if 'annual_removals_df' in st.session_state and 'annual_emissions_df' in st.session_state and 'extended_yield_df' in st.session_state:
        annual_removals_df = st.session_state['annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']
        extended_yield_df = st.session_state['extended_yield_df']

        total_emissions = annual_emissions_df['Total'].values
        total_removals = annual_removals_df['Total_Removals'].values
        cocoa_yield = extended_yield_df['Cocoa Yield (t/ha/yr)'].values

        min_length = min(len(total_emissions), len(total_removals), len(cocoa_yield))
        total_emissions = total_emissions[:min_length]
        total_removals = total_removals[:min_length]
        cocoa_yield = cocoa_yield[:min_length]

        cumulative_emissions = np.cumsum(total_emissions)
        cumulative_removals = np.cumsum(total_removals)

        summary_df = pd.DataFrame({
            'Year': np.arange(1, min_length + 1),
            'Emissions': cumulative_emissions,
            'Removals': cumulative_removals,
            'Cocoa Yield (t/ha/yr)': cocoa_yield
        })

        st.session_state['summary_df'] = summary_df

        # Display reserve rate information
        if st.session_state.get('apply_reserve_rate', True):
            st.info(f"Reserve rate of {st.session_state.get('reserve_rate', 0.5) * 100:.0f}% applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")
        else:
            st.info("No reserve rate applied on removals. You can apply/change the reserve rate in the Simulation Parameters.")

        st.subheader('Graph: Cumulative Emissions, Removals, and Cocoa Yield Over Time')
        cumulative_summary_chart = hf.plot_summary_emissions_removals_cocoa_production(summary_df)
        st.altair_chart(cumulative_summary_chart, use_container_width=True)

        st.subheader('Table: Cumulative Summary Data')
        st.dataframe(summary_df)

        annual_emissions = annual_emissions_df['Total']
        annual_removals = annual_removals_df['Total_Removals']

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



def calculate_scenarios(time_horizon):
    if 'update_flag' in st.session_state and st.session_state.update_flag:
        st.session_state.update_flag = False

        if 'annual_removals_df' not in st.session_state:
            st.error("Annual removals data not found. Please run the 'One Hectare Model' tab first.")
            return None

        if 'annual_emissions_df' not in st.session_state:
            st.error("Annual emissions data not found. Please run the 'One Hectare Model' tab first.")
            return None

        annual_removals_df = st.session_state['annual_removals_df']
        annual_emissions_df = st.session_state['annual_emissions_df']

        cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 20)
        luc_emissions = st.session_state.get('luc_emissions', 0.5648)
        replanting_emissions = st.session_state.get('replanting_emissions', 0.0)

        # Adjust year range to start from 1 to time_horizon inclusive
        kpi_df = pd.DataFrame({'Year': np.arange(1, time_horizon + 1)})

        for scenario in st.session_state.scenarios_df['Scenario'].unique():
            scenario_data = {
                "emissions": np.zeros(time_horizon),
                "removals": np.zeros(time_horizon),
                "area_planted": np.zeros(time_horizon),
                "annual_removals_by_tree_type": {year: {tree_type: 0 for tree_type in annual_removals_df.columns if tree_type not in ['Year', 'Total_Removals']} for year in range(1, time_horizon + 1)}
            }
            scenario_df = st.session_state.scenarios_df[st.session_state.scenarios_df['Scenario'] == scenario]
            for _, row in scenario_df.iterrows():
                area = row['Area (ha)']
                start_year = int(row['Year'])
                for year in range(start_year, time_horizon + 1):
                    scenario_data["area_planted"][year - 1] += area
                    cycle_year = (year - start_year) % cultivation_cycle_duration

                    for tree_type in annual_removals_df.columns:
                        if tree_type not in ['Year', 'Total_Removals']:
                            annual_growth = annual_removals_df[tree_type].iloc[cycle_year] * area
                            scenario_data["annual_removals_by_tree_type"][year][tree_type] += annual_growth
                            scenario_data["removals"][year - 1] += annual_growth

                    scenario_data["emissions"][year - 1] += annual_emissions_df['Total'].iloc[cycle_year] * area

                    if cycle_year == 0 and year != start_year:
                        scenario_data["emissions"][year - 1] += annual_emissions_df['Total'].iloc[0] * area

                if start_year <= time_horizon:
                    scenario_data["emissions"][start_year - 1] += area * luc_emissions
                last_year_of_cycle = start_year + cultivation_cycle_duration - 1
                if last_year_of_cycle <= time_horizon:
                    scenario_data["emissions"][last_year_of_cycle - 1] += area * replanting_emissions

            cumulative_emissions = np.cumsum(scenario_data["emissions"])
            cumulative_removals = np.cumsum(scenario_data["removals"])
            carbon_balance = cumulative_removals - cumulative_emissions

            kpi_df[f'{scenario}: Cumulative Emissions'] = cumulative_emissions
            kpi_df[f'{scenario}: Cumulative Removals'] = cumulative_removals
            kpi_df[f'{scenario}: Carbon Balance'] = carbon_balance
            kpi_df[f'{scenario}: Total Area Planted (ha)'] = scenario_data["area_planted"]

            st.session_state[f'{scenario}_annual_removals_by_tree_type'] = scenario_data["annual_removals_by_tree_type"]

        st.session_state.kpi_df = kpi_df
