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



def create_removals_stacked_bar_chart(df, x_col, y_col, color_col, title):
    import altair as alt

    df_melted = df.melt(id_vars=[x_col], value_vars=df.columns.difference([x_col, 'Total']), var_name=color_col, value_name=y_col)
    
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X(f'{x_col}:O', title='Year', axis=alt.Axis(labelAngle=0)),  # Ensure labels are vertical
        y=alt.Y(f'{y_col}:Q', title='Removals'),
        color=color_col,
        tooltip=[x_col, y_col, color_col]
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()

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
    # Exclude the 'Total' column
    melt_columns = [col for col in dataframe.columns if col not in [x_axis, 'Total']]
    long_df = dataframe.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Emissions_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_axis}:N', axis=alt.Axis(title=x_axis, labelAngle=0)),  # Ensure labels are vertical
        y=alt.Y('Emissions_Value:Q', stack='zero', axis=alt.Axis(title='Emissions (tCO2e)')),
        color=alt.Color(f'{color_category}:N', title='Source'),
        tooltip=[f'{x_axis}:N', f'{color_category}:N', 'Emissions_Value:Q']
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart


def create_cumulative_emissions_stacked_bar_chart(dataframe, x_axis, y_axis, color_category, title="Cumulative Emissions Stacked Bar Chart"):
    # Calculate cumulative emissions
    cumulative_df = dataframe.copy()
    melt_columns = [col for col in cumulative_df.columns if col not in [x_axis, 'Total']]
    cumulative_df[melt_columns] = cumulative_df[melt_columns].cumsum()

    long_df = cumulative_df.melt(id_vars=[x_axis], value_vars=melt_columns, var_name=color_category, value_name='Cumulative_Emissions_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_axis}:N', axis=alt.Axis(title=x_axis, labelAngle=0)),  # Ensure labels are vertical
        y=alt.Y('Cumulative_Emissions_Value:Q', stack='zero', axis=alt.Axis(title='Cumulative Emissions (tCO2e)')),
        color=alt.Color(f'{color_category}:N', title='Source'),
        tooltip=[f'{x_axis}:N', f'{color_category}:N', 'Cumulative_Emissions_Value:Q']
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart



def plot_emissions_vs_removals_one_hectare(kpi_df, time_horizon):
    mode = st.session_state.get('mode', 'Dark Mode')
    colors = do.get_colors(mode)
    
    df_filtered = kpi_df.melt(id_vars='Year', value_vars=['Emissions', 'Removals'], var_name='Type', value_name='tCO2e')
    
    color_scale = alt.Scale(
        domain=['Emissions', 'Removals'],
        range=[colors['EMISSIONS_COLOR'], colors['REMOVALS_COLOR']]
    )
    
    chart = alt.Chart(df_filtered).mark_line().encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        y=alt.Y('tCO2e:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=['Year:Q', 'tCO2e:Q', 'Type:N']
    ).interactive().properties(title='Emissions vs. Removals Over Time')
    
    return chart




def plot_carbon_balance_bars_one_hectare(kpi_df, time_horizon):
    chart = alt.Chart(kpi_df).transform_calculate(
        positive='datum["Carbon Balance"] < 0'
    ).mark_bar().encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        y=alt.Y('Carbon Balance:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.condition(
            alt.datum['Carbon Balance'] < 0,
            alt.value('green'),
            alt.value('red')
        ),
        tooltip=['Year:Q', 'Carbon Balance:Q']
    ).properties(title='Carbon Balance Over Time')
    return chart



def plot_summary_emissions_removals_cocoa_production(summary_df):
    mode = st.session_state.get('mode', 'Dark Mode')
    colors = do.get_colors(mode)
    
    summary_df_melted = summary_df.melt(id_vars='Year', value_vars=['Emissions', 'Removals', 'Cocoa Yield (t/ha/yr)'], var_name='Type', value_name='Value')

    color_scale = alt.Scale(
        domain=['Emissions', 'Removals', 'Cocoa Yield (t/ha/yr)'],
        range=[colors['EMISSIONS_COLOR'], colors['REMOVALS_COLOR'], colors['COCOA_YIELD_COLOR']]
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
    mode = st.session_state.get('mode', 'Dark Mode')
    colors = do.get_colors(mode)
    
    # Melt the DataFrame to plot multiple lines
    annual_summary_df_melted = annual_summary_df.melt(id_vars='Year', value_vars=['Annual Emissions', 'Annual Removals', 'Cocoa Yield'], var_name='Type', value_name='Value')

    # Define color scale for annual chart
    color_scale_annual = alt.Scale(
        domain=['Annual Emissions', 'Annual Removals', 'Cocoa Yield'],
        range=[colors['EMISSIONS_COLOR'], colors['REMOVALS_COLOR'], colors['COCOA_YIELD_COLOR']]
    )

    # Base chart
    base_annual = alt.Chart(annual_summary_df_melted).encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        color=alt.Color('Type:N', scale=color_scale_annual)
    )

    # Annual Emissions and Removals
    annual_emissions_removals = base_annual.transform_filter(
        alt.FieldOneOfPredicate(field='Type', oneOf=['Annual Emissions', 'Annual Removals'])
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Annual Emissions and Removals (tCO2e/ha/yr)'))
    )

    # Annual Cocoa Yield
    annual_cocoa_yield = base_annual.transform_filter(
        alt.FieldEqualPredicate(field='Type', equal='Cocoa Yield')
    ).mark_line().encode(
        y=alt.Y('Value:Q', axis=alt.Axis(title='Cocoa Yield (t/ha/yr)'))
    )

    # Combine the charts
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




def plot_total_area_planted(kpi_df):
    st.subheader('Graph: Total Planted Area Over Time')
    for scenario in st.session_state.scenarios_df['Scenario'].unique():
        kpi_df[f'{scenario}: Total Area Planted (ha)'] = 0
        for _, row in st.session_state.scenarios_df[st.session_state.scenarios_df['Scenario'] == scenario].iterrows():
            kpi_df.loc[kpi_df['Year'] >= row['Year'], f'{scenario}: Total Area Planted (ha)'] += row['Area (ha)']

    total_area_planted_df = kpi_df.melt(id_vars='Year', value_vars=[col for col in kpi_df.columns if 'Total Area Planted' in col], var_name='Scenario', value_name='Area (ha)')
    total_area_planted_chart = alt.Chart(total_area_planted_df).mark_line().encode(
        x='Year:Q',
        y='Area (ha):Q',
        color='Scenario:N',
        tooltip=['Year:Q', 'Area (ha):Q', 'Scenario:N']
    ).interactive().properties(title='Total Planted Area Over Time')
    st.altair_chart(total_area_planted_chart, use_container_width=True)



def plot_emissions_vs_removals_planting_scenarios():
    st.subheader('Graph: Emissions vs. Removals Over Time')

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

    if show_emissions != st.session_state.filter_options["show_emissions"] or show_removals != st.session_state.filter_options["show_removals"] or selected_scenarios != st.session_state.filter_options["selected_scenarios"]:
        st.session_state.filter_options["show_emissions"] = show_emissions
        st.session_state.filter_options["show_removals"] = show_removals
        st.session_state.filter_options["selected_scenarios"] = selected_scenarios

    filtered_columns = []
    if st.session_state.filter_options["show_emissions"]:
        filtered_columns.extend([f'{scenario}: Emissions' for scenario in st.session_state.filter_options["selected_scenarios"]])
    if st.session_state.filter_options["show_removals"]:
        filtered_columns.extend([f'{scenario}: Removals' for scenario in st.session_state.filter_options["selected_scenarios"]])

    if not filtered_columns:
        st.write("Please select at least one scenario and type to display.")
        return

    kpi_df = st.session_state.kpi_df
    
    # Ensure the necessary columns are present
    for col in filtered_columns:
        if col not in kpi_df.columns:
            kpi_df[col] = 0

    emissions_removals_df = kpi_df.melt(id_vars='Year', value_vars=filtered_columns, var_name='Type', value_name='tCO2e')
    emissions_removals_chart = alt.Chart(emissions_removals_df).mark_line().encode(
        x='Year:Q',
        y='tCO2e:Q',
        color='Type:N',
        tooltip=['Year:Q', 'tCO2e:Q', 'Type:N']
    ).interactive().properties(title='Emissions vs. Removals Over Time')
    st.altair_chart(emissions_removals_chart, use_container_width=True)




def plot_carbon_balance_planting_scenarios():
    st.subheader('Graph: Carbon Balance Over Time')
    
    if 'filter_options_carbon_balance' not in st.session_state:
        st.session_state.filter_options_carbon_balance = st.session_state.scenarios_df['Scenario'].unique().tolist()

    selected_scenarios = st.multiselect(
        "Select Scenarios for Carbon Balance",
        options=st.session_state.scenarios_df['Scenario'].unique(),
        default=st.session_state.filter_options_carbon_balance
    )

    st.session_state.filter_options_carbon_balance = selected_scenarios

    filtered_columns = [f'{scenario}: Carbon Balance' for scenario in st.session_state.filter_options_carbon_balance]

    if not filtered_columns:
        st.write("Please select at least one scenario to display.")
        return

    kpi_df = st.session_state.kpi_df
    carbon_balance_df = kpi_df.melt(id_vars='Year', value_vars=filtered_columns, var_name='Scenario', value_name='tCO2e')
    carbon_balance_chart = alt.Chart(carbon_balance_df).mark_line().encode(
        x='Year:Q',
        y='tCO2e:Q',
        color='Scenario:N',
        tooltip=['Year:Q', 'tCO2e:Q', 'Scenario:N']
    ).interactive().properties(title='Carbon Balance Over Time')

    st.altair_chart(carbon_balance_chart, use_container_width=True)