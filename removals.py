import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from reserve_rate_widget import reserve_rate_widget
import helper_functions as hf


def calculate_annual_removals(carbon_stocks_df, cultivation_cycle_duration, plantation_timeline):
    land_prep_year = plantation_timeline['land_prep_year']
    time_horizon = len(carbon_stocks_df)
    annual_removals = np.zeros((time_horizon, 3))  # 3 columns for Cocoa, Shade, and Timber
    cumulative_removals = np.zeros((time_horizon, 3))
    removal_reversals = np.zeros(time_horizon)
    
    tree_types = ['Cocoa', 'Shade', 'Timber']
    
    for i, tree_type in enumerate(tree_types):
        planting_year = plantation_timeline[f'{tree_type.lower()}_planting_year']
        remove_at_cycle_end = st.session_state.get(f'remove_{tree_type.lower()}_at_cycle_end', True)
        replanting_delay = st.session_state.get(f'{tree_type.lower()}_replanting_delay', 0) if remove_at_cycle_end else 0
        
        tree_age = 0
        for year in range(time_horizon):
            actual_year = land_prep_year + year
            years_since_planting = actual_year - planting_year
            cycle_year = years_since_planting % cultivation_cycle_duration if years_since_planting >= 0 else -1
            
            if cycle_year == -1:
                # Before planting
                annual_removals[year, i] = 0
                cumulative_removals[year, i] = 0
            elif remove_at_cycle_end and cycle_year == cultivation_cycle_duration - 1:
                # Last year of cycle for trees that are removed
                current_stock = carbon_stocks_df.iloc[min(tree_age, len(carbon_stocks_df) - 1)][tree_type]
                annual_removals[year, i] = max(0, current_stock - cumulative_removals[year-1, i])
                cumulative_removals[year, i] = current_stock
                reversal = cumulative_removals[year, i]
                removal_reversals[year + 1] = reversal  # Store reversal for next year
                tree_age = 0
            elif remove_at_cycle_end and cycle_year < replanting_delay:
                # During replanting delay for trees that are removed
                annual_removals[year, i] = 0
                cumulative_removals[year, i] = 0
            else:
                # Normal growth or continued growth for trees not removed
                tree_age += 1
                current_stock = carbon_stocks_df.iloc[min(tree_age - 1, len(carbon_stocks_df) - 1)][tree_type]
                previous_stock = cumulative_removals[year-1, i] if year > 0 else 0
                annual_removals[year, i] = max(0, current_stock - previous_stock)
                cumulative_removals[year, i] = current_stock

    return annual_removals, cumulative_removals, removal_reversals



def prepare_removals_dataframes(time_horizon, cultivation_cycle_duration):
    if 'growth_params_df' not in st.session_state or st.session_state.growth_params_df is None:
        st.write("No growth data available.")
        return None, None

    growth_params_df = st.session_state['growth_params_df']

    # Check if the growth_params_df has the expected structure
    expected_columns = ['Year Number', 'Cocoa', 'Shade', 'Timber']
    if not all(col in growth_params_df.columns for col in expected_columns):
        st.write("Error: growth_params_df does not have the expected structure.")
        st.write("Columns:", growth_params_df.columns)
        return None, None

    plantation_timeline = {
        'land_prep_year': st.session_state.get('land_prep_year', 2015),
        'shade_planting_year': st.session_state.get('shade_planting_year', 2016),
        'cocoa_planting_year': st.session_state.get('cocoa_planting_year', 2017),
        'timber_planting_year': st.session_state.get('timber_planting_year', 2016)
    }

    annual_removals, cumulative_removals, removal_reversals = calculate_annual_removals(growth_params_df, cultivation_cycle_duration, plantation_timeline)

    land_prep_year = plantation_timeline['land_prep_year']
    year_numbers = np.arange(1, time_horizon + 1)
    actual_years = land_prep_year + (year_numbers - 1)

    annual_removals_df = pd.DataFrame(annual_removals, columns=['Cocoa', 'Shade', 'Timber'])
    annual_removals_df.insert(0, 'Year Number', year_numbers)
    annual_removals_df.insert(1, 'Year', actual_years)
    
    cumulative_removals_df = pd.DataFrame(cumulative_removals, columns=['Cocoa', 'Shade', 'Timber'])
    cumulative_removals_df.insert(0, 'Year Number', year_numbers)
    cumulative_removals_df.insert(1, 'Year', actual_years)
    
    annual_removals_df['Total_Removals'] = annual_removals_df[['Cocoa', 'Shade', 'Timber']].sum(axis=1)
    cumulative_removals_df['Total_Removals'] = cumulative_removals_df[['Cocoa', 'Shade', 'Timber']].sum(axis=1)

    annual_removals_df['Removal Reversals'] = removal_reversals
    cumulative_removals_df['Removal Reversals'] = np.cumsum(removal_reversals)  # Use cumulative sum for cumulative dataframe

    # Store removal reversals in session state for use in emissions calculations
    st.session_state['removal_reversals'] = removal_reversals

    return annual_removals_df, cumulative_removals_df



def create_removals_stacked_bar_chart(dataframe, x_col, y_col, color_col, title):
    # Exclude 'Year', 'Year Number', and 'Total_Removals' columns
    melt_columns = [col for col in dataframe.columns if col not in [x_col, 'Year Number', 'Total_Removals', 'Removal Reversals']]
    long_df = dataframe.melt(id_vars=[x_col], value_vars=melt_columns, var_name=color_col, value_name='Removals_Value')
    
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X(f'{x_col}:O', axis=alt.Axis(title='Year', format='d', labelAngle=0)),  # Use 'd' format for no comma
        y=alt.Y('Removals_Value:Q', stack='zero', axis=alt.Axis(title='Removals (tCO2e)')),
        color=alt.Color(f'{color_col}:N', title='Tree Type'),
        tooltip=[alt.Tooltip(f'{x_col}:O', title='Year', format='d'), f'{color_col}:N', alt.Tooltip('Removals_Value:Q', format='.2f')]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart


def removals_analysis(time_horizon):
    st.header('Removals Analysis', divider="gray")
    
    apply_reserve_rate, reserve_rate = reserve_rate_widget()
    
    cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 20)
    annual_removals_df, cumulative_removals_df = prepare_removals_dataframes(time_horizon, cultivation_cycle_duration)
    
    if annual_removals_df is None or annual_removals_df.empty:
        st.write("No removals data available. Please complete the tree growth section first.")
        return

    # Store the original removals for reference
    st.session_state['original_annual_removals_df'] = annual_removals_df.copy()
    st.session_state['original_cumulative_removals_df'] = cumulative_removals_df.copy()

    # Apply reserve rate if enabled
    if apply_reserve_rate:
        for col in annual_removals_df.columns:
            if col not in ['Year Number', 'Year', 'Total_Removals', 'Removal Reversals']:
                annual_removals_df[col] *= (1 - reserve_rate)
                cumulative_removals_df[col] *= (1 - reserve_rate)
        annual_removals_df['Total_Removals'] *= (1 - reserve_rate)
        cumulative_removals_df['Total_Removals'] *= (1 - reserve_rate)

    # Recalculate total removals after applying reserve rate
    annual_removals_df['Total_Removals'] = annual_removals_df.drop(columns=['Year Number', 'Year', 'Removal Reversals']).sum(axis=1)
    cumulative_removals_df['Total_Removals'] = cumulative_removals_df.drop(columns=['Year Number', 'Year', 'Removal Reversals']).sum(axis=1)

    # Reorder columns to ensure Year Number and Year are first
    column_order = ['Year Number', 'Year'] + [col for col in annual_removals_df.columns if col not in ['Year Number', 'Year']]
    annual_removals_df = annual_removals_df[column_order]
    cumulative_removals_df = cumulative_removals_df[column_order]

    st.session_state['annual_removals_df'] = annual_removals_df
    st.session_state['cumulative_removals_df'] = cumulative_removals_df

    st.subheader('Graph: Annual Removals Over Time')
    annual_removals_chart = create_removals_stacked_bar_chart(annual_removals_df, 'Year', 'Removals', 'Tree Type', "Annual Removals Over Time")
    st.altair_chart(annual_removals_chart, use_container_width=True)

    st.subheader('Table: Annual Removals Over Time')
    format_dict = {col: '{:.2f}' for col in annual_removals_df.columns}
    format_dict.update({
        'Year Number': '{:0.0f}',
        'Year': '{:0.0f}',
        'Total_Removals': '{:.2f}',
        'Removal Reversals': '{:.2f}'
    })
    st.dataframe(annual_removals_df.style.format(format_dict))

    st.subheader('Graph: Cumulative Removals Over Time')
    cumulative_removals_chart = create_removals_stacked_bar_chart(cumulative_removals_df, 'Year', 'Removals', 'Tree Type', "Cumulative Removals Over Time")
    st.altair_chart(cumulative_removals_chart, use_container_width=True)

    st.subheader('Table: Cumulative Removals Over Time')
    format_dict = {col: '{:.2f}' for col in cumulative_removals_df.columns}
    format_dict.update({
        'Year Number': '{:0.0f}',
        'Year': '{:0.0f}',
        'Total_Removals': '{:.2f}',
        'Removal Reversals': '{:.2f}'
    })
    st.dataframe(cumulative_removals_df.style.format(format_dict))



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
