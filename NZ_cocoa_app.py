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


def plot_emissions_vs_removals(kpi_df):
    # Define the color scheme explicitly for the chart
    color_scale = alt.Scale(domain=['Emissions', 'Removals'],
                            range=['brown', 'lightgreen'])

    df_filtered = kpi_df.melt(id_vars='Year', value_vars=['Emissions', 'Removals'], var_name='Type', value_name='tCO2e')
    chart = alt.Chart(df_filtered).mark_line().encode(
        x='Year:Q',
        y=alt.Y('tCO2e:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.Color('Type:N', scale=color_scale),
        tooltip=['Year:Q', 'tCO2e:Q', 'Type:N']
    ).interactive().properties(title='Emissions vs. Removals Over Time')
    return chart


def plot_carbon_balance_bars(kpi_df):
    chart = alt.Chart(kpi_df).transform_calculate(
        positive='datum["Carbon Balance"] >= 0'  # Determines if the balance is positive
    ).mark_bar().encode(
        x=alt.X('Year:Q', axis=alt.Axis(title='Year')),
        y=alt.Y('Carbon Balance:Q', axis=alt.Axis(title='tCO2e')),
        color=alt.condition(
            alt.datum['Carbon Balance'] >= 0,
            alt.value('green'),  # Positive values colored green
            alt.value('red')     # Negative values colored red
        ),
        tooltip=['Year:Q', 'Carbon Balance:Q']
    ).properties(title='Carbon Balance Over Time')

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

    # File Uploader for Custom Emissions Data
    uploaded_file = st.file_uploader("Upload your emissions data CSV", type=['csv'])
    if uploaded_file is not None:
        custom_emissions_df = pd.read_csv(uploaded_file)
        if len(custom_emissions_df) == cultivation_cycle_duration:
            st.session_state.emissions_df = custom_emissions_df
        else:
            st.error(f"Please ensure the uploaded CSV has exactly {cultivation_cycle_duration} rows to match the cultivation cycle duration.")

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

    # Convert the annual emissions DataFrame to CSV for download
    annual_emissions_csv = edited_emissions_df.to_csv(index=False)

    # New: Create a download button for the annual emissions CSV data
    st.download_button(
        label="Download Annual Emissions Data as CSV",
        data=annual_emissions_csv,
        file_name='annual_emissions_data.csv',
        mime='text/csv',
    )


    # Initialize arrays to store annual emissions and removals
    annual_emissions = np.zeros(time_horizon)
    annual_removals = np.zeros(time_horizon)

    # Adjust the logic to account for LUC emissions only in the first year
    # and replanting/recycling emissions in the last year of each cultivation cycle
    for cycle_start in range(0, time_horizon, cultivation_cycle_duration):
        for year in range(cultivation_cycle_duration):
            current_year = cycle_start + year
            if current_year < time_horizon:
                # Add emissions for each source; LUC handled separately
                annual_emissions[current_year] += edited_emissions_df.iloc[year % cultivation_cycle_duration].sum()

        # Handle LUC emissions only in the first year of the simulation
        if cycle_start == 0:
            annual_emissions[0] += luc_emissions

        # Handle replanting/recycling emissions at the end of each cultivation cycle,
        # but not beyond the simulation time horizon
        last_year_of_cycle = cycle_start + cultivation_cycle_duration - 1
        if last_year_of_cycle < time_horizon:
            annual_emissions[last_year_of_cycle] += replanting_emissions



    # Adjusted loop for calculating annual removals with resetting
    for year in range(time_horizon):
        cycle_year = year % cultivation_cycle_duration  # Year within the current cycle
        annual_removals[year] = 0  # Reset at the start of the loop
        for tree_type, params in growth_params_df.iterrows():
            beta, L, k = params['beta'], params['L'], params['k']
            # Calculate growth only for the current year within the cycle
            annual_removals[year] += exponential_growth(cycle_year + 1, beta, L, k)


    # Calculate cumulative emissions, removals, and carbon balance
    cumulative_emissions = np.cumsum(annual_emissions)
    carbon_balance = annual_removals - cumulative_emissions                            


    # Prepare DataFrame for plotting and display with updated calculations
    kpi_df = pd.DataFrame({
    'Year': np.arange(time_horizon),
    'Emissions': cumulative_emissions,
    'Removals': annual_removals,  # Updated to show current year's removals
    'Carbon Balance': carbon_balance  # Adjusted carbon balance calculation
    })


    # Plotting the KPIs with Altair
    st.subheader('Graph: Emissions, Removals, and Carbon Balance Over Time')   
     # Plotting Emissions vs. Removals
    st.altair_chart(plot_emissions_vs_removals(kpi_df), use_container_width=True)
    # Plotting Carbon Balance
    st.altair_chart(plot_carbon_balance_bars(kpi_df), use_container_width=True)





    # Display the data table with KPIs
    st.subheader('Data: Emissions, Removals, and Carbon Balance Over Time')
    # Round all values in the DataFrame to two decimal places
    kpi_df = kpi_df.round(2)
    st.dataframe(kpi_df)

    # Convert DataFrame to CSV string
    csv = kpi_df.to_csv(index=False)

    # Create a download button for the CSV data
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='carbon_balance_data.csv',
        mime='text/csv',
    )



if __name__ == "__main__":
    main()
