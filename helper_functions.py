# helper_functions.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import design_options as do

def exponential_growth(t, beta, L, k):
    return L * (1 - np.exp(-k * t))

def logarithmic_growth(t, coefficient, intercept):
    return coefficient * np.log(t) + intercept

def power_function_growth(t, alpha, beta, carbon_content, conversion_factor):
    return (alpha * t ** beta) * carbon_content * conversion_factor

def get_equation_latex(model_name):
    if model_name == "Exponential Plateau":
        return r"L \cdot (1 - e^{-k \cdot t})"
    elif model_name == "Logarithmic Growth (Terra Global Method)":
        return r"C \cdot \ln(t) + I"
    elif model_name == "Power Function (Cool Farm Method)":
        return r"\alpha \cdot t^{\beta} \cdot C_c \cdot C_f"
    return ""



def plot_tree_growth(df, tree_type):
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y(f'{tree_type}:Q', title='Growth (units)')
    ).properties(title=f'{tree_type} Growth Over Time')
    return chart

def plot_combined_tree_growth(df):
    melted_df = df.melt('Year', var_name='Tree Type', value_name='Growth')
    chart = alt.Chart(melted_df).mark_line().encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('Growth:Q', title='Growth (units)'),
        color='Tree Type:N'
    ).properties(title='Combined Tree Growth Models')
    return chart



def create_removals_stacked_bar_chart(dataframe, x_axis, y_axis, color_category, title="Removals Stacked Bar Chart"):
    # Exclude 'Year', 'Year Number', and 'Total_Removals' columns
    melt_columns = [col for col in dataframe.columns if col not in [x_axis, 'Year Number', 'Total_Removals']]
    long_df = dataframe.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Removals_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_axis}:N', axis=alt.Axis(title=x_axis, labelAngle=0)),  # Ensure labels are vertical
        y=alt.Y('Removals_Value:Q', stack='zero', axis=alt.Axis(title='Removals (tCO2e)')),
        color=alt.Color(f'{color_category}:N', title='Tree Type'),
        tooltip=[f'{x_axis}:N', f'{color_category}:N', 'Removals_Value:Q']
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart



def create_removals_line_chart(dataframe, x_axis, y_axis, color_category, title="Line Chart"):
    # Melt the DataFrame to long format, excluding 'Total_Removals'
    melt_columns = [col for col in dataframe.columns if col not in [x_axis, y_axis]]
    long_df = dataframe.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Removals')
    
    chart = alt.Chart(long_df).mark_line().encode(
        x=alt.X(f'{x_axis}:Q', axis=alt.Axis(title=x_axis)),
        y=alt.Y('Removals:Q', axis=alt.Axis(title=y_axis)),
        color=f'{color_category}:N',
        tooltip=[f'{x_axis}:Q', f'{color_category}:N', 'Removals:Q']
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart



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



def create_annual_emissions_stacked_bar_chart(dataframe, x_axis, y_axis, color_category, title="Annual Emissions Stacked Bar Chart"):
    # Exclude the 'Year' column from melt_columns
    melt_columns = [col for col in dataframe.columns if col != x_axis]
    long_df = dataframe.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Emissions_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_axis}:O', axis=alt.Axis(title='Year', format='d', labelAngle=0)),  # Use 'd' format for no comma
        y=alt.Y('Emissions_Value:Q', stack='zero', axis=alt.Axis(title='Emissions (tCO2e)')),
        color=alt.Color(f'{color_category}:N', title='Source'),
        tooltip=[alt.Tooltip(f'{x_axis}:O', title='Year', format='d'), f'{color_category}:N', alt.Tooltip('Emissions_Value:Q', format='.3f')]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart


def create_cumulative_emissions_stacked_bar_chart(dataframe, x_axis, y_axis, color_category, title="Cumulative Emissions Stacked Bar Chart"):
    # The dataframe should already contain cumulative values, so we don't need to recalculate
    melt_columns = [col for col in dataframe.columns if col not in [x_axis, 'Year Number', 'Total']]
    long_df = dataframe.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Cumulative_Emissions_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_axis}:O', axis=alt.Axis(title='Year', format='d', labelAngle=0)),  # Use 'd' format for no comma
        y=alt.Y('Cumulative_Emissions_Value:Q', stack='zero', axis=alt.Axis(title='Cumulative Emissions (tCO2e)')),
        color=alt.Color(f'{color_category}:N', title='Source'),
        tooltip=[alt.Tooltip(f'{x_axis}:O', title='Year', format='d'), f'{color_category}:N', alt.Tooltip('Cumulative_Emissions_Value:Q', format='.3f')]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart



def plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon):
    chart = alt.Chart(kpi_df).transform_calculate(
        negative='datum["Cumulative Carbon Balance"] < 0'
    ).mark_bar().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Cumulative Carbon Balance:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.condition(
            alt.datum['Cumulative Carbon Balance'] < 0,
            alt.value('green'),
            alt.value('red')
        ),
        tooltip=[alt.Tooltip('Year:O', title='Year', format='d'), 'Cumulative Carbon Balance:Q']
    )
    return chart

def plot_annual_carbon_balance_bars_one_hectare(kpi_df, time_horizon):
    chart = alt.Chart(kpi_df).transform_calculate(
        negative='datum["Annual Carbon Balance"] < 0'
    ).mark_bar().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Annual Carbon Balance:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.condition(
            alt.datum['Annual Carbon Balance'] < 0,
            alt.value('green'),
            alt.value('red')
        ),
        tooltip=[alt.Tooltip('Year:O', title='Year', format='d'), 'Annual Carbon Balance:Q']
    )
    return chart



def plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon):
    df_filtered = kpi_df.melt(
        id_vars=['Year', 'Year Number'],
        value_vars=['Cumulative Emissions', 'Cumulative Removals'],
        var_name='Type',
        value_name='tCO2e'
    )
    color_scale = alt.Scale(
        domain=['Cumulative Emissions', 'Cumulative Removals'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]
    )
    chart = alt.Chart(df_filtered).mark_line().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('tCO2e:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=[alt.Tooltip('Year:O', title='Year', format='d'), 'tCO2e:Q', 'Type:N']
    )
    return chart


def plot_annual_emissions_vs_removals_one_hectare(kpi_df, time_horizon):
    df_filtered = kpi_df.melt(
        id_vars=['Year', 'Year Number'],
        value_vars=['Annual Emissions', 'Annual Removals'],
        var_name='Type',
        value_name='tCO2e'
    )
    color_scale = alt.Scale(
        domain=['Annual Emissions', 'Annual Removals'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]
    )
    chart = alt.Chart(df_filtered).mark_line().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('tCO2e:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=[alt.Tooltip('Year:O', title='Year', format='d'), 'tCO2e:Q', 'Type:N']
    )
    return chart


def plot_summary_emissions_removals_cocoa_production(summary_df):
    summary_df_melted = summary_df.melt(id_vars='Year', value_vars=['Emissions', 'Removals', 'Cocoa Yield (t/ha/yr)'], var_name='Type', value_name='Value')
    color_scale = alt.Scale(
        domain=['Emissions', 'Removals', 'Cocoa Yield (t/ha/yr)'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR, do.COCOA_YIELD_COLOR]
    )
    base = alt.Chart(summary_df_melted).encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        color=alt.Color('Type:N', scale=color_scale)
    )
    emissions_removals = base.transform_filter(
        alt.FieldOneOfPredicate(field='Type', oneOf=['Emissions', 'Removals'])
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Emissions and Removals (tCO2e/ha/yr)'))
    )
    cocoa_yield = base.transform_filter(
        alt.FieldEqualPredicate(field='Type', equal='Cocoa Yield (t/ha/yr)')
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Cocoa Yield (t/ha/yr)'))
    )
    combined_chart = alt.layer(
        emissions_removals, cocoa_yield
    ).resolve_scale(
        y='independent'
    ).properties(
        width=600,
        height=400,
        title='Emissions, Removals, and Cocoa Production Over Time'
    ).interactive()
    return combined_chart



def plot_annual_emissions_removals_cocoa_yield(annual_summary_df):
    annual_summary_df_melted = annual_summary_df.melt(id_vars='Year', value_vars=['Annual Emissions', 'Annual Removals', 'Cocoa Yield'], var_name='Type', value_name='Value')
    color_scale_annual = alt.Scale(
        domain=['Annual Emissions', 'Annual Removals', 'Cocoa Yield'],
        range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR, do.COCOA_YIELD_COLOR]
    )
    base_annual = alt.Chart(annual_summary_df_melted).encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        color=alt.Color('Type:N', scale=color_scale_annual)
    )
    annual_emissions_removals = base_annual.transform_filter(
        alt.FieldOneOfPredicate(field='Type', oneOf=['Annual Emissions', 'Annual Removals'])
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e/ha/yr)'))
    )
    annual_cocoa_yield = base_annual.transform_filter(
        alt.FieldEqualPredicate(field='Type', equal='Cocoa Yield')
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Cocoa Yield (t/ha/yr)'))
    )
    combined_annual_chart = alt.layer(
        annual_emissions_removals, annual_cocoa_yield
    ).resolve_scale(
        y='independent'
    ).properties(
        width=600,
        height=400,
        title='Annual Emissions and Removals vs. Cocoa Yield Over Time'
    ).interactive()
    return combined_annual_chart




def plot_emissions_vs_removals_planting_scenarios():
    if 'kpi_df' in st.session_state:
        kpi_df = st.session_state.kpi_df

        if 'filter_options' not in st.session_state:
            st.session_state.filter_options = {
                "show_emissions": True,
                "show_removals": True,
                "selected_scenarios": st.session_state.scenarios_df['Scenario'].unique().tolist()
            }

        show_emissions = st.checkbox("Show Emissions", value=st.session_state.filter_options["show_emissions"])
        show_removals = st.checkbox("Show Removals", value=st.session_state.filter_options["show_removals"])
        selected_scenarios = st.multiselect(
            "Select Scenarios",
            options=st.session_state.scenarios_df['Scenario'].unique(),
            default=st.session_state.filter_options["selected_scenarios"]
        )

        st.session_state.filter_options["show_emissions"] = show_emissions
        st.session_state.filter_options["show_removals"] = show_removals
        st.session_state.filter_options["selected_scenarios"] = selected_scenarios

        metrics = []
        if show_emissions:
            metrics += [f'{scenario}: Cumulative Emissions' for scenario in selected_scenarios]
        if show_removals:
            metrics += [f'{scenario}: Cumulative Removals' for scenario in selected_scenarios]

        if metrics:
            # Display the columns for debugging
            st.write("Columns in kpi_df:", kpi_df.columns)
            
            melted_kpi_df = kpi_df.melt(id_vars='Year', value_vars=metrics, var_name='Metric', value_name='Value')
            chart = alt.Chart(melted_kpi_df).mark_line().encode(
                x='Year:Q',
                y='Value:Q',
                color='Metric:N',
                tooltip=['Year', 'Metric', 'Value']
            ).properties(title='Emissions vs. Removals Over Time')
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Please select at least one metric to display.")




def plot_carbon_balance_planting_scenarios():
    if 'kpi_df' in st.session_state:
        kpi_df = st.session_state.kpi_df
        
        # Ensure only relevant columns are selected
        value_vars = [col for col in kpi_df.columns if 'Carbon Balance' in col]
        if not value_vars:
            st.write("No carbon balance data available.")
            return
        
        # Add checkboxes for filtering
        selected_scenarios = st.multiselect(
            "Select Scenarios for Carbon Balance",
            options=st.session_state.scenarios_df['Scenario'].unique(),
            default=st.session_state.scenarios_df['Scenario'].unique().tolist()
        )
        
        selected_value_vars = [f'{scenario}: Carbon Balance' for scenario in selected_scenarios]
        
        # Display the columns for debugging
        st.write("Columns in kpi_df:", kpi_df.columns)
        
        melted_kpi_df = kpi_df.melt(id_vars='Year', value_vars=selected_value_vars, var_name='Scenario', value_name='Carbon Balance Value')
        
        carbon_balance_chart = alt.Chart(melted_kpi_df).mark_line().encode(
            x='Year:Q',
            y='Carbon Balance Value:Q',
            color='Scenario:N',
            tooltip=['Year', 'Scenario', 'Carbon Balance Value']
        ).properties(title='Carbon Balance Over Time')
        
        st.altair_chart(carbon_balance_chart, use_container_width=True)



def plot_removals_by_tree_type(scenario):
    if f'{scenario}_annual_removals_by_tree_type' in st.session_state:
        annual_removals_by_tree_type = st.session_state[f'{scenario}_annual_removals_by_tree_type']
        removals_df = pd.DataFrame(annual_removals_by_tree_type).T.reset_index().melt(id_vars='index', var_name='Tree Type', value_name='Removals')
        removals_df = removals_df[removals_df['Tree Type'] != 'Total_Removals']  # Exclude 'Total_Removals'
        removals_df.rename(columns={'index': 'Year'}, inplace=True)
        removals_chart = alt.Chart(removals_df).mark_bar().encode(
            x='Year:O',
            y='Removals:Q',
            color='Tree Type:N',
            tooltip=['Year:O', 'Tree Type:N', 'Removals:Q']
        ).properties(title=f'Annual Removals by Tree Type for Scenario {scenario}')
        st.altair_chart(removals_chart, use_container_width=True)



def plot_cumulative_carbon_intensity(kpi_df):
    if 'kpi_df' in st.session_state:
        kpi_df = st.session_state.kpi_df

        cumulative_df = kpi_df.copy()
        cumulative_df['Total Cocoa Yield'] = cumulative_df[[col for col in cumulative_df.columns if 'Cocoa Yield' in col]].sum(axis=1)

        for scenario in st.session_state.scenarios_df['Scenario'].unique():
            cumulative_df[f'{scenario}: Cumulative Carbon Intensity'] = cumulative_df[f'{scenario}: Cumulative Emissions'] / cumulative_df['Total Cocoa Yield']

        value_vars = [f'{scenario}: Cumulative Carbon Intensity' for scenario in st.session_state.scenarios_df['Scenario'].unique()]
        melted_df = cumulative_df.melt(id_vars='Year', value_vars=value_vars, var_name='Scenario', value_name='Carbon Intensity')

        carbon_intensity_chart = alt.Chart(melted_df).mark_line().encode(
            x='Year:Q',
            y='Carbon Intensity:Q',
            color='Scenario:N',
            tooltip=['Year', 'Scenario', 'Carbon Intensity']
        ).properties(title='Cumulative Carbon Intensity Over Time')

        st.altair_chart(carbon_intensity_chart, use_container_width=True)



def plot_annual_carbon_intensity(kpi_df):
    if 'kpi_df' in st.session_state:
        kpi_df = st.session_state.kpi_df

        annual_df = kpi_df.copy()
        annual_df['Total Cocoa Yield'] = annual_df[[col for col in annual_df.columns if 'Cocoa Yield' in col]].sum(axis=1)

        for scenario in st.session_state.scenarios_df['Scenario'].unique():
            annual_df[f'{scenario}: Annual Carbon Intensity'] = annual_df[f'{scenario}: Cumulative Emissions'] / annual_df['Total Cocoa Yield']

        value_vars = [f'{scenario}: Annual Carbon Intensity' for scenario in st.session_state.scenarios_df['Scenario'].unique()]
        melted_df = annual_df.melt(id_vars='Year', value_vars=value_vars, var_name='Scenario', value_name='Carbon Intensity')

        carbon_intensity_chart = alt.Chart(melted_df).mark_line().encode(
            x='Year:Q',
            y='Carbon Intensity:Q',
            color='Scenario:N',
            tooltip=['Year', 'Scenario', 'Carbon Intensity']
        ).properties(title='Annual Carbon Intensity Over Time')

        st.altair_chart(carbon_intensity_chart, use_container_width=True)


# Helper functions for the farm-related plots
def plot_farm_total_area_planted(kpi_df):
    # Identify columns that represent individual block areas
    block_columns = [col for col in kpi_df.columns if col.endswith('Area Planted')]
    block_names = [col.replace(' Area Planted', '') for col in block_columns]

    # Create a new dataframe for the stacked area chart
    area_df = pd.DataFrame({
        'Year': kpi_df['Year']
    })

    for block, col in zip(block_names, block_columns):
        area_df[block] = kpi_df[col]

    # Melt the dataframe for Altair
    melted_df = area_df.melt('Year', var_name='Block', value_name='Area')

    # Determine the year each block was first added
    block_years = melted_df[melted_df['Area'] > 0].groupby('Block')['Year'].min().reset_index()

    # Merge the year of first addition back into the melted_df
    melted_df = melted_df.merge(block_years, on='Block', suffixes=('', '_FirstAdded'))

    # Sort blocks by the year they were first added
    block_order = block_years.sort_values('Year')['Block'].tolist()

    # Create the stacked area chart with step interpolation and no border
    area_chart = alt.Chart(melted_df).mark_area(
        interpolate='step-after',
        line=False
    ).encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', labelAngle=0)),
        y=alt.Y('Area:Q', stack='zero', axis=alt.Axis(title='Planted Area (ha)')),
        color=alt.Color('Block:N', scale=alt.Scale(scheme='category20'), sort=block_order),  # Ensure legend is sorted correctly
        order=alt.Order('Year_FirstAdded:O', sort='ascending'),  # Order by the year of first addition
        tooltip=['Year:O', 'Block:N', alt.Tooltip('Area:Q', format='.2f')]
    )

    # Create the line chart for total area with step interpolation
    line_chart = alt.Chart(kpi_df).mark_line(
        color='black',
        strokeWidth=2,
        interpolate='step-after'
    ).encode(
        x='Year:O',
        y='Total Area Planted (ha):Q',
        tooltip=[
            alt.Tooltip('Year:O'),
            alt.Tooltip('Total Area Planted (ha):Q', format='.2f')
        ]
    )

    # Combine the charts
    combined_chart = (area_chart + line_chart).properties(
        title='Total Planted Area Over Time',
        width=700,
        height=400
    ).interactive()

    st.altair_chart(combined_chart, use_container_width=True)



def plot_farm_cumulative_emissions_removals(kpi_df):
    combined_df = kpi_df[['Year', 'Cumulative Emissions', 'Cumulative Removals']]
    cumulative_chart = combined_df.melt('Year', var_name='Type', value_name='Value')
    cumulative_chart = alt.Chart(cumulative_chart).mark_line().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Value:Q', axis=alt.Axis(title='Cumulative Emissions and Removals (tCO2e)')),
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Cumulative Emissions', 'Cumulative Removals'],
                                                 range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]))
    ).properties(
        width=700,
        height=400
    )

    st.altair_chart(cumulative_chart, use_container_width=True)

def display_farm_simulation_data(timeseries_df):
    necessary_columns = ['Cumulative Emissions', 'Cumulative Removals', 'Annual Emissions', 'Annual Removals', 'Cumulative Cocoa Yield', 'Annual Cocoa Yield']
    for column in necessary_columns:
        if column not in timeseries_df.columns:
            timeseries_df[column] = np.nan

    timeseries_df['Year Number'] = range(1, len(timeseries_df) + 1)
    timeseries_df = timeseries_df[['Year Number', 'Year', 'Cumulative Emissions', 'Cumulative Removals', 'Annual Emissions', 'Annual Removals', 'Cumulative Cocoa Yield', 'Annual Cocoa Yield']]
    
    timeseries_df = timeseries_df.rename(columns={
        'Cumulative Emissions': 'Total Cumulative Emissions',
        'Cumulative Removals': 'Total Cumulative Removals',
        'Annual Emissions': 'Total Annual Emissions',
        'Annual Removals': 'Total Annual Removals',
        'Cumulative Cocoa Yield': 'Cumulative Cocoa Yield',
        'Annual Cocoa Yield': 'Annual Cocoa Yield'
    })
    
    timeseries_df['Total Cumulative Emissions'] = timeseries_df['Total Cumulative Emissions'].round(2)
    timeseries_df['Total Cumulative Removals'] = timeseries_df['Total Cumulative Removals'].round(2)
    timeseries_df['Total Annual Emissions'] = timeseries_df['Total Annual Emissions'].round(2)
    timeseries_df['Total Annual Removals'] = timeseries_df['Total Annual Removals'].round(2)
    timeseries_df['Cumulative Cocoa Yield'] = timeseries_df['Cumulative Cocoa Yield'].round(2)
    timeseries_df['Annual Cocoa Yield'] = timeseries_df['Annual Cocoa Yield'].round(2)
    
    st.dataframe(timeseries_df.style.format({
        'Year': '{:0.0f}',
        'Total Cumulative Emissions': '{:.2f}',
        'Total Cumulative Removals': '{:.2f}',
        'Total Annual Emissions': '{:.2f}',
        'Total Annual Removals': '{:.2f}',
        'Cumulative Cocoa Yield': '{:.2f}',
        'Annual Cocoa Yield': '{:.2f}'
    }))



def display_farm_block_breakdown_annual(timeseries_df, kpi_df):
    st.subheader("Farm Block Breakdown of Annual Emissions and Removals")
    annual_emissions_df = kpi_df.filter(like='Annual Emissions').copy()
    annual_removals_df = kpi_df.filter(like='Annual Removals').copy()
    breakdown_annual_df = pd.concat([timeseries_df[['Year Number', 'Year']], annual_emissions_df, annual_removals_df], axis=1)
    st.dataframe(breakdown_annual_df.style.format({
        'Year': '{:0.0f}',
        'Annual Emissions': '{:.2f}',
        'Annual Removals': '{:.2f}'
    }))

def display_farm_block_breakdown_cumulative(timeseries_df, kpi_df):
    st.subheader("Farm Block Breakdown of Cumulative Emissions and Removals")
    cumulative_emissions_df = kpi_df.filter(like='Cumulative Emissions').copy()
    cumulative_removals_df = kpi_df.filter(like='Cumulative Removals').copy()
    breakdown_cumulative_df = pd.concat([timeseries_df[['Year Number', 'Year']], cumulative_emissions_df, cumulative_removals_df], axis=1)
    st.dataframe(breakdown_cumulative_df.style.format({
        'Year': '{:0.0f}',
        'Cumulative Emissions': '{:.2f}',
        'Cumulative Removals': '{:.2f}'
    }))

def display_farm_block_breakdown_cocoa(timeseries_df, kpi_df):
    st.subheader("Farm Block Breakdown of Cocoa Production")
    cocoa_yield_df = kpi_df.filter(like='Cocoa Yield').copy()
    cocoa_yield_df['Yearly Total'] = cocoa_yield_df.sum(axis=1)
    breakdown_cocoa_df = pd.concat([timeseries_df[['Year Number', 'Year']], cocoa_yield_df], axis=1)
    st.dataframe(breakdown_cocoa_df.style.format({
        'Year': '{:0.0f}',
        'Cocoa Yield': '{:.2f}',
        'Yearly Total': '{:.2f}'
    }))
    


def plot_farm_annual_emissions_removals(kpi_df):
    annual_chart = alt.Chart(kpi_df).mark_line().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Value:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e)')),
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Annual Emissions', 'Annual Removals'],
                                                 range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]))
    ).transform_fold(
        ['Annual Emissions', 'Annual Removals'],
        as_=['Type', 'Value']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(annual_chart, use_container_width=True)

def plot_farm_annual_carbon_balance(kpi_df):
    kpi_df['Annual Carbon Balance'] = kpi_df['Annual Removals'] - kpi_df['Annual Emissions']
    balance_chart = alt.Chart(kpi_df).mark_bar().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Annual Carbon Balance:Q', axis=alt.Axis(title='Annual Carbon Balance (tCO2e)')),
        color=alt.condition(
            alt.datum['Annual Carbon Balance'] > 0,
            alt.value(do.REMOVALS_COLOR),
            alt.value(do.EMISSIONS_COLOR)
        )
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(balance_chart, use_container_width=True)

def plot_farm_cumulative_carbon_balance(kpi_df):
    kpi_df['Cumulative Carbon Balance'] = kpi_df['Cumulative Removals'] - kpi_df['Cumulative Emissions']
    balance_chart = alt.Chart(kpi_df).mark_bar().encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')),
        y=alt.Y('Cumulative Carbon Balance:Q', axis=alt.Axis(title='Cumulative Carbon Balance (tCO2e)')),
        color=alt.condition(
            alt.datum['Cumulative Carbon Balance'] > 0,
            alt.value(do.REMOVALS_COLOR),
            alt.value(do.EMISSIONS_COLOR)
        )
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(balance_chart, use_container_width=True)

def display_farm_annual_carbon_balance(kpi_df):
    annual_balance_df = kpi_df[['Year', 'Annual Emissions', 'Annual Removals']]
    annual_balance_df['Annual Carbon Balance'] = annual_balance_df['Annual Removals'] - annual_balance_df['Annual Emissions']
    st.dataframe(annual_balance_df.style.format({col: '{:.2f}' for col in annual_balance_df.columns if col != 'Year'}))

def display_farm_cumulative_carbon_balance(kpi_df):
    cumulative_balance_df = kpi_df[['Year', 'Cumulative Emissions', 'Cumulative Removals']]
    cumulative_balance_df['Cumulative Carbon Balance'] = cumulative_balance_df['Cumulative Removals'] - cumulative_balance_df['Cumulative Emissions']
    st.dataframe(cumulative_balance_df.style.format({col: '{:.2f}' for col in cumulative_balance_df.columns if col != 'Year'}))

def plot_farm_annual_carbon_intensity(kpi_df):
    kpi_df['Annual Carbon Intensity'] = np.where(
        kpi_df['Annual Cocoa Yield'] > 0,
        kpi_df['Annual Emissions'] / kpi_df['Annual Cocoa Yield'],
        np.nan
    )
    
    base = alt.Chart(kpi_df).encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d'))
    )
    
    line = base.mark_line(color='black').encode(
        y=alt.Y('Annual Carbon Intensity:Q', 
                axis=alt.Axis(title='Annual Carbon Intensity (tCO2e/t cocoa)'),
                scale=alt.Scale(zero=False))
    )
    
    points = base.mark_circle(size=60).encode(
        y='Annual Carbon Intensity:Q',
        color=alt.condition(
            alt.datum['Annual Carbon Intensity'] > 0,
            alt.value('red'),
            alt.value('green')
        ),
        tooltip=['Year:O', alt.Tooltip('Annual Carbon Intensity:Q', format='.2f')]
    )
    
    chart = (line + points).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

def plot_farm_cumulative_carbon_intensity(kpi_df):
    kpi_df['Cumulative Carbon Intensity'] = np.where(
        kpi_df['Cumulative Cocoa Yield'] > 0,
        kpi_df['Cumulative Emissions'] / kpi_df['Cumulative Cocoa Yield'],
        np.nan
    )
    
    base = alt.Chart(kpi_df).encode(
        x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d'))
    )
    
    line = base.mark_line(color='black').encode(
        y=alt.Y('Cumulative Carbon Intensity:Q', 
                axis=alt.Axis(title='Cumulative Carbon Intensity (tCO2e/t cocoa)'),
                scale=alt.Scale(zero=False))
    )
    
    points = base.mark_circle(size=60).encode(
        y='Cumulative Carbon Intensity:Q',
        color=alt.condition(
            alt.datum['Cumulative Carbon Intensity'] > 0,
            alt.value('red'),
            alt.value('green')
        ),
        tooltip=['Year:O', alt.Tooltip('Cumulative Carbon Intensity:Q', format='.2f')]
    )
    
    chart = (line + points).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

def display_farm_annual_carbon_intensity(kpi_df):
    annual_intensity_df = kpi_df[['Year', 'Annual Emissions', 'Annual Cocoa Yield']]
    annual_intensity_df['Annual Carbon Intensity'] = np.where(
        annual_intensity_df['Annual Cocoa Yield'] > 0,
        annual_intensity_df['Annual Emissions'] / annual_intensity_df['Annual Cocoa Yield'],
        np.nan
    )
    st.dataframe(annual_intensity_df.style.format({
        'Year': '{:.0f}',
        'Annual Emissions': '{:.2f}',
        'Annual Cocoa Yield': '{:.2f}',
        'Annual Carbon Intensity': '{:.2f}'
    }))

def display_farm_cumulative_carbon_intensity(kpi_df):
    cumulative_intensity_df = kpi_df[['Year', 'Cumulative Emissions', 'Cumulative Cocoa Yield']]
    cumulative_intensity_df['Cumulative Carbon Intensity'] = np.where(
        cumulative_intensity_df['Cumulative Cocoa Yield'] > 0,
        cumulative_intensity_df['Cumulative Emissions'] / cumulative_intensity_df['Cumulative Cocoa Yield'],
        np.nan
    )
    st.dataframe(cumulative_intensity_df.style.format({
        'Year': '{:.0f}',
        'Cumulative Emissions': '{:.2f}',
        'Cumulative Cocoa Yield': '{:.2f}',
        'Cumulative Carbon Intensity': '{:.2f}'
    }))

def plot_farm_annual_summary(kpi_df):
    st.write("Columns in kpi_df:", kpi_df.columns)
    st.write("First few rows of kpi_df:", kpi_df.head())

    base = alt.Chart(kpi_df).encode(x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')))

    emissions_removals = base.transform_fold(
        ['Annual Emissions', 'Annual Removals'],
        as_=['Variable', 'Value']
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e)'), scale=alt.Scale(zero=False)),
        color=alt.Color('Variable:N', scale=alt.Scale(domain=['Annual Emissions', 'Annual Removals'],
                                                     range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]))
    )

    cocoa_production_column = 'Annual Cocoa Production (t/ha/yr)' if 'Annual Cocoa Production (t/ha/yr)' in kpi_df.columns else 'Annual Cocoa Production'
    
    cocoa_production = base.mark_line(color=do.COCOA_YIELD_COLOR).encode(
        y=alt.Y(f'{cocoa_production_column}:Q', axis=alt.Axis(title='Cocoa Production (t/ha/yr)'), scale=alt.Scale(zero=False))
    )

    chart = alt.layer(emissions_removals, cocoa_production).resolve_scale(
        y='independent'
    ).properties(
        width=700,
        height=400,
        title='Annual Emissions, Removals, and Cocoa Production Over Time'
    )

    st.altair_chart(chart, use_container_width=True)

def plot_farm_cumulative_summary(kpi_df):
    st.write("Columns in kpi_df:", kpi_df.columns)
    st.write("First few rows of kpi_df:", kpi_df.head())

    base = alt.Chart(kpi_df).encode(x=alt.X('Year:O', axis=alt.Axis(title='Year', format='d')))

    emissions_removals = base.transform_fold(
        ['Cumulative Emissions', 'Cumulative Removals'],
        as_=['Variable', 'Value']
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Cumulative Emissions and Removals (tCO2e)'), scale=alt.Scale(zero=False)),
        color=alt.Color('Variable:N', scale=alt.Scale(domain=['Cumulative Emissions', 'Cumulative Removals'],
                                                     range=[do.EMISSIONS_COLOR, do.REMOVALS_COLOR]))
    )

    cocoa_production_column = 'Cumulative Cocoa Production (t/ha)' if 'Cumulative Cocoa Production (t/ha)' in kpi_df.columns else 'Cumulative Cocoa Production'
    
    cocoa_production = base.mark_line(color=do.COCOA_YIELD_COLOR).encode(
        y=alt.Y(f'{cocoa_production_column}:Q', axis=alt.Axis(title='Cocoa Production (t/ha)'), scale=alt.Scale(zero=False))
    )

    chart = alt.layer(emissions_removals, cocoa_production).resolve_scale(
        y='independent'
    ).properties(
        width=700,
        height=400,
        title='Cumulative Emissions, Removals, and Cocoa Production Over Time'
    )

    st.altair_chart(chart, use_container_width=True)

def display_farm_annual_summary(kpi_df):
    st.write("Columns in kpi_df:", kpi_df.columns)
    
    columns = ['Year Number', 'Year', 'Total Area Planted (ha)', 'Annual Emissions', 'Annual Removals', 'Annual Cocoa Production', 'Annual Cocoa Production (t/ha/yr)']
    
    # Check which columns are actually present in kpi_df
    available_columns = [col for col in columns if col in kpi_df.columns]
    
    annual_summary_df = kpi_df[available_columns].copy()
    
    # If 'Year Number' is missing, create it
    if 'Year Number' not in annual_summary_df.columns:
        annual_summary_df['Year Number'] = range(1, len(annual_summary_df) + 1)
    
    # If 'Annual Cocoa Production (t/ha/yr)' is missing but 'Annual Cocoa Production' is present, calculate it
    if 'Annual Cocoa Production (t/ha/yr)' not in annual_summary_df.columns and 'Annual Cocoa Production' in annual_summary_df.columns and 'Total Area Planted (ha)' in annual_summary_df.columns:
        annual_summary_df['Annual Cocoa Production (t/ha/yr)'] = annual_summary_df['Annual Cocoa Production'] / annual_summary_df['Total Area Planted (ha)']
    
    st.dataframe(annual_summary_df.style.format({
        'Year Number': '{:.0f}',
        'Year': '{:.0f}',
        'Total Area Planted (ha)': '{:.2f}',
        'Annual Emissions': '{:.2f}',
        'Annual Removals': '{:.2f}',
        'Annual Cocoa Production': '{:.2f}',
        'Annual Cocoa Production (t/ha/yr)': '{:.2f}'
    }))

def display_farm_cumulative_summary(kpi_df):
    st.write("Columns in kpi_df:", kpi_df.columns)
    
    columns = ['Year Number', 'Year', 'Total Area Planted (ha)', 'Cumulative Emissions', 'Cumulative Removals', 'Cumulative Cocoa Production', 'Cumulative Cocoa Production (t/ha)']
    
    # Check which columns are actually present in kpi_df
    available_columns = [col for col in columns if col in kpi_df.columns]
    
    cumulative_summary_df = kpi_df[available_columns].copy()
    
    # If 'Year Number' is missing, create it
    if 'Year Number' not in cumulative_summary_df.columns:
        cumulative_summary_df['Year Number'] = range(1, len(cumulative_summary_df) + 1)
    
    # If 'Cumulative Cocoa Production (t/ha)' is missing but 'Cumulative Cocoa Production' is present, calculate it
    if 'Cumulative Cocoa Production (t/ha)' not in cumulative_summary_df.columns and 'Cumulative Cocoa Production' in cumulative_summary_df.columns and 'Total Area Planted (ha)' in cumulative_summary_df.columns:
        cumulative_summary_df['Cumulative Cocoa Production (t/ha)'] = cumulative_summary_df['Cumulative Cocoa Production'] / cumulative_summary_df['Total Area Planted (ha)']
    
    st.dataframe(cumulative_summary_df.style.format({
        'Year Number': '{:.0f}',
        'Year': '{:.0f}',
        'Total Area Planted (ha)': '{:.2f}',
        'Cumulative Emissions': '{:.2f}',
        'Cumulative Removals': '{:.2f}',
        'Cumulative Cocoa Production': '{:.2f}',
        'Cumulative Cocoa Production (t/ha)': '{:.2f}'
    }))