import streamlit as st
import pandas as pd
import numpy as np
import altair as alt



# Define the growth function for tree carbon sequestration
def exponential_growth(t, beta, L, k):
    """
    Exponential growth model for carbon stock.
    
    Parameters:
    - t: Time in years (age)
    - beta: Lower asymptote
    - L: Upper asymptote
    - k: Growth rate
    """
    return L - (L - beta) * np.exp(-k * t)


# Create an Altair chart for the Tree Growth
def plot_tree_growth(growth_params_df):
    # Time range for the simulation
    t = np.linspace(0, 50, 500)
    growth_curves = pd.DataFrame(t, columns=['Year'])
     
    # Generate growth curves for each tree type based on CO2e parameters
    for tree_type, params in growth_params_df.iterrows():
        beta, L, k = params['beta'], params['L'], params['k']
        growth_curves[tree_type] = exponential_growth(t, beta, L, k)
    
    # Melt the DataFrame for Altair plotting
    long_df = growth_curves.melt('Year', var_name='Tree Type', value_name='Carbon Stock (tCO2e/ha)')

    # Calculate the maximum carbon stock value
    max_carbon_stock =  growth_curves.drop('Year', axis=1).max().max()

    # Use this value to set the y-axis domain dynamically
    chart = alt.Chart(long_df).mark_line().encode(
        x='Year:Q',
        y=alt.Y('Carbon Stock (tCO2e/ha):Q', scale=alt.Scale(domain=(0, max_carbon_stock + 0.2 * max_carbon_stock))),
        color='Tree Type:N',
        tooltip=['Year:Q', 'Carbon Stock (tCO2e/ha):Q', 'Tree Type:N']
    ).interactive().properties(title='Tree Growth Over Time (CO2e)')
    
    return chart



# Create an Altair chart for the KPIs
def altair_plot_kpis(kpi_df):
    # Melting the DataFrame to long format
    long_df = kpi_df.melt('Year', var_name='variable', value_name='value')
    
    # Define a manual domain and range for colors
    domain = ['Cumulative Emissions', 'Cumulative Removals', 'Carbon Balance']
    range_ = ['brown', 'lightgreen', 'purple']  # Custom colors for each series

    # Creating the chart
    chart = alt.Chart(long_df).mark_line().encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        y=alt.Y('value:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.Color('variable:N', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Series")),
        tooltip=['Year:Q', 'value:Q', 'variable:N']
    ).interactive()

    return chart

## Streamlit application
def main():
    st.title('Cocoa Plantation Carbon Footprint Model')



    # Inputs for the simulation time horizon and cultivation cycle duration
    time_horizon = st.number_input('Enter the time horizon for the simulation (years):', min_value=1, value=50, step=1)
    cultivation_cycle_duration = st.number_input('Enter the duration of the cultivation cycle (years):', min_value=1, value=20, step=1)



    # Tree Growth Parameters
    st.subheader('Tree Growth Parameters')
    
    # Define default parameters for tree growth
    tree_types = ['Cocoa Trees', 'Shade Trees', 'Timber Trees']

    default_params = {
        'beta': [0.01, 0.01, 0.01],  # Assuming these are correct lower asymptotes
        'L': [44.04, 36.7, 18.35],   # Upper asymptotes for each tree type
        'k': [0.3, 0.2, 0.1]         # Growth rates for each tree type
    }
    
    if 'growth_params_df' not in st.session_state:
        st.session_state.growth_params_df = pd.DataFrame(default_params, index=tree_types)
    growth_params_df = st.data_editor(data=st.session_state.growth_params_df, key="growth_params")

    # Plot tree growth based on current parameters
    st.altair_chart(plot_tree_growth(growth_params_df), use_container_width=True)



    # LUC and replanting/recycling emissions inputs
    st.subheader('LUC and Replanting/Recycling emissions')
    luc_emissions = st.number_input('Enter LUC emissions (tons of CO2):', value=0.0, step=0.1)
    replanting_emissions = st.number_input('Enter Replanting/Recycling emissions (tons of CO2):', value=0.0, step=0.1)



    # Annual Emissions Input for One Cultivation Cycle
    st.subheader('Annual Emissions Input for One Cultivation Cycle')
    emission_sources = ['Equipment', 'Fuel', 'Fertilizer', 'Soil Amendments', 'Pesticide', 'Drying']

    if 'emissions_df' not in st.session_state or len(st.session_state.emissions_df) != cultivation_cycle_duration:
        # Generate initial emissions values
        initial_values = np.linspace(start=[0.5, 0.3, 0.4, 0.2, 0.25, 0.3], 
                                    stop=[1.0, 0.55, 0.8, 0.4, 0.45, 0.6], 
                                    num=5)
        # Repeat the last row for the rest of the years
        repeated_values = np.tile(initial_values[-1, :], (cultivation_cycle_duration - 5, 1))
        # Combine initial values with repeated values
        emissions_values = np.vstack((initial_values, repeated_values))
        
        # Create DataFrame with the generated values
        st.session_state.emissions_df = pd.DataFrame(emissions_values, columns=emission_sources)

    # Display the emissions DataFrame for editing
    edited_emissions_df = st.data_editor(data=st.session_state.emissions_df, key="emissions")

    # Process the edited rows if any
    if 'edited_rows' in st.session_state:
        for row_index, changes in st.session_state.edited_rows.items():
            for col, value in changes.items():
                # You can add validation logic here if necessary
                st.session_state.emissions_df.at[row_index, col] = value



    # Initialize arrays to store annual emissions and removals
    annual_emissions = np.zeros(time_horizon)
    annual_removals = np.zeros(time_horizon)


    # Calculate annual emissions from editable inputs
    for cycle_start in range(0, time_horizon, cultivation_cycle_duration):
        for year in range(cultivation_cycle_duration):
            if cycle_start + year < time_horizon:
                # Add emissions for each source; LUC and replanting handled separately
                annual_emissions[cycle_start + year] += edited_emissions_df.iloc[year].sum()
        
        # Handle LUC emissions at the start of each cycle and replanting at the end
        if cycle_start < time_horizon:
            annual_emissions[cycle_start] += luc_emissions
        if cycle_start + cultivation_cycle_duration - 1 < time_horizon:
            annual_emissions[cycle_start + cultivation_cycle_duration - 1] += replanting_emissions


    # Loop for calculating annual removals and adjusting emissions at the end of each cycle
    for year in range(time_horizon):
        cycle_year = year % cultivation_cycle_duration
        if cycle_year == cultivation_cycle_duration - 1:
            # At the end of the cycle, count the carbon stored in trees as emissions
            carbon_stock_final_year = 0
            for tree_type, params in growth_params_df.iterrows():
                beta, L, k = params['beta'], params['L'], params['k']
                carbon_stock_final_year += exponential_growth(cycle_year + 1, beta, L, k)
            # Add this carbon stock to the emissions for the year
            annual_emissions[year] += carbon_stock_final_year
            # Reset removals to zero for this year
            annual_removals[year] = 0
        else:
            for tree_type, params in growth_params_df.iterrows():
                beta, L, k = params['beta'], params['L'], params['k']
                # Continue to calculate removals for all other years
                annual_removals[year] += exponential_growth(cycle_year + 1, beta, L, k)





    # Calculate cumulative emissions, removals, and carbon balance
    cumulative_emissions = np.cumsum(annual_emissions)
    cumulative_removals = np.cumsum(annual_removals)
    carbon_balance = cumulative_removals - cumulative_emissions                             


    # Prepare DataFrame for plotting and display with updated calculations
    kpi_df = pd.DataFrame({
        'Year': np.arange(time_horizon),
        'Cumulative Emissions': cumulative_emissions,
        'Cumulative Removals': cumulative_removals,
        'Carbon Balance': carbon_balance
    })




    # Plotting the KPIs with Altair
    st.subheader('Interactive Cumulative Emissions, Removals, and Carbon Balance Over Time')
    st.altair_chart(altair_plot_kpis(kpi_df), use_container_width=True)



    # Display the data table with KPIs
    st.subheader('Cumulative Emissions, Removals, and Carbon Balance Over Time')
    st.dataframe(kpi_df)



if __name__ == "__main__":
    main()
