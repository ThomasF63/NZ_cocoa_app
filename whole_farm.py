# whole_farm.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import helper_functions as hf
import design_options as do
from scenario_management import get_saved_scenarios, load_scenario, save_farm, load_farm, get_saved_farms


def generate_farm_simulation_data(farm_blocks, time_horizon=None):
    if time_horizon is None:
        time_horizon = st.session_state.get('time_horizon', 50)
    
    land_prep_year = st.session_state.get('land_prep_year', 2015)
    kpi_df = pd.DataFrame({
        'Year Number': range(1, time_horizon + 1),
        'Year': range(land_prep_year, land_prep_year + time_horizon)
    })
    
    total_emissions = np.zeros(time_horizon)
    total_removals = np.zeros(time_horizon)
    total_area_planted = np.zeros(time_horizon)
    annual_cocoa_production = np.zeros(time_horizon)
    cumulative_cocoa_production = np.zeros(time_horizon)
    
    # Dictionary to store block-specific area data
    area_planted_dict = {}

    for _, block in farm_blocks.iterrows():
        scenario_name = block['Scenario']
        scenario_data = load_scenario(scenario_name)
        
        if scenario_data is None:
            st.error(f"Scenario '{scenario_name}' not found. Please check your scenarios.")
            continue

        block_area = block['Area (ha)']
        start_year = int(block['Year'])
        
        annual_emissions = scenario_data['annual_emissions_df']['Total'].values
        annual_removals = scenario_data['annual_removals_df']['Total_Removals'].values
        cocoa_yield = scenario_data['cocoa_yield_df'].get('Cocoa Yield (kg/ha/yr)', np.zeros(time_horizon))
        
        # Convert to numpy array if it's a pandas Series
        if isinstance(cocoa_yield, pd.Series):
            cocoa_yield = cocoa_yield.values
        
        # Ensure cocoa_yield has the correct length
        if len(cocoa_yield) < time_horizon:
            cocoa_yield = np.pad(cocoa_yield, (0, time_horizon - len(cocoa_yield)), 'edge')
        elif len(cocoa_yield) > time_horizon:
            cocoa_yield = cocoa_yield[:time_horizon]

        area_planted_per_block = np.zeros(time_horizon)
        
        for year in range(time_horizon):
            if year >= start_year - land_prep_year:
                scenario_year = year - (start_year - land_prep_year)
                if scenario_year < len(annual_emissions):
                    total_emissions[year] += annual_emissions[scenario_year] * block_area
                    total_removals[year] += annual_removals[scenario_year] * block_area
                    annual_cocoa_production[year] += cocoa_yield[scenario_year] * block_area / 1000  # Convert kg to tonnes
                    area_planted_per_block[year] = block_area
                total_area_planted[year] += block_area

        area_planted_dict[block['Block']] = area_planted_per_block

    kpi_df['Total Area Planted (ha)'] = total_area_planted
    kpi_df['Annual Emissions'] = total_emissions
    kpi_df['Annual Removals'] = total_removals
    kpi_df['Cumulative Emissions'] = np.cumsum(total_emissions)
    kpi_df['Cumulative Removals'] = np.cumsum(total_removals)
    kpi_df['Annual Cocoa Production'] = annual_cocoa_production
    kpi_df['Cumulative Cocoa Production'] = np.cumsum(annual_cocoa_production)
    
    # Calculate cocoa production in t/ha/yr and t/ha
    kpi_df['Annual Cocoa Production (t/ha/yr)'] = np.where(kpi_df['Total Area Planted (ha)'] > 0, 
                                                           kpi_df['Annual Cocoa Production'] / kpi_df['Total Area Planted (ha)'], 
                                                           0)
    kpi_df['Cumulative Cocoa Production (t/ha)'] = np.where(kpi_df['Total Area Planted (ha)'] > 0, 
                                                            kpi_df['Cumulative Cocoa Production'] / kpi_df['Total Area Planted (ha)'], 
                                                            0)

    # Add block-specific area data to kpi_df
    for block, area in area_planted_dict.items():
        kpi_df[f'{block} Area Planted'] = area

    st.session_state.kpi_df = kpi_df

    return kpi_df


def whole_farm_simulation():
    st.sidebar.title("Whole Farm Simulation")
    steps = ["Farm Blocks", "Carbon Balance", "Carbon Intensity", "Summary", "Save/Load Farm"]
    choice = st.sidebar.radio("Go to Section", steps)

    if choice == "Farm Blocks":
        st.header('Farm Blocks', divider="gray")

        if 'farm_blocks' not in st.session_state:
            st.session_state.farm_blocks = pd.DataFrame({
                "Block": [],
                "Area (ha)": [],
                "Year": [],
                "Scenario": []
            })

        # Add "Update Graphs and Tables" button at the top
        if st.button("Update Graphs and Tables", key="update_top"):
            st.session_state.update_flag = True
            generate_farm_simulation_data(st.session_state.farm_blocks, st.session_state.get('time_horizon', 50))

        # Farm blocks table (non-editable)
        st.dataframe(st.session_state.farm_blocks.style.format({
            'Area (ha)': '{:.2f}',
            'Year': '{:.0f}'
        }))

        # Graph: Total Area Planted Over Time
        if 'kpi_df' in st.session_state:
            st.subheader('Graph: Total Planted Area Over Time')
            hf.plot_farm_total_area_planted(st.session_state.kpi_df)

        # Controls for adding new blocks
        st.subheader("Add New Block")
        new_block = st.text_input("Block Name")
        new_area = st.number_input("Area (ha)", min_value=0.0, step=0.1)
        new_year = st.number_input("Planting Year", min_value=1900, max_value=2100, step=1)
        saved_scenarios = get_saved_scenarios()
        new_scenario = st.selectbox("Select Scenario", saved_scenarios)

        # Preventing duplicate block names
        if st.button("Add Block"):
            if new_block in st.session_state.farm_blocks["Block"].values:
                st.error("A block with this name already exists. Please choose a different name.")
            else:
                new_entry = pd.DataFrame({
                    "Block": [new_block],
                    "Area (ha)": [new_area],
                    "Year": [new_year],
                    "Scenario": [new_scenario]
                })
                st.session_state.farm_blocks = pd.concat([st.session_state.farm_blocks, new_entry], ignore_index=True)
                st.rerun()


        # Editable table for managing blocks
        st.subheader("Manage Blocks")

        edited_farm_blocks = st.data_editor(
            st.session_state.farm_blocks,
            num_rows="dynamic",
            key="farm_blocks_editor"
        )

        # Update button for editable table
        if st.button("Update Blocks", key="update_blocks"):
            st.session_state.farm_blocks = edited_farm_blocks
            st.session_state.update_flag = True
            generate_farm_simulation_data(st.session_state.farm_blocks, st.session_state.get('time_horizon', 50))
            st.success("Farm blocks updated successfully!")
            st.rerun()

        # Block duplication and deletion controls
        col1, col2 = st.columns(2)
        with col1:
            block_to_duplicate = st.selectbox("Select Block to Duplicate", st.session_state.farm_blocks["Block"].tolist())
            duplicated_block_name = st.text_input("New Name for Duplicated Block")
            if st.button("Duplicate Block"):
                if duplicated_block_name in st.session_state.farm_blocks["Block"].values:
                    st.error("A block with this name already exists. Please choose a different name.")
                else:
                    block_data = st.session_state.farm_blocks.loc[st.session_state.farm_blocks["Block"] == block_to_duplicate].copy()
                    block_data["Block"] = duplicated_block_name
                    st.session_state.farm_blocks = pd.concat([st.session_state.farm_blocks, block_data], ignore_index=True)
                    st.rerun()

        with col2:
            block_to_delete = st.selectbox("Select Block to Delete", st.session_state.farm_blocks["Block"].tolist())
            if st.button("Delete Block"):
                st.session_state.farm_blocks = st.session_state.farm_blocks[st.session_state.farm_blocks["Block"] != block_to_delete]
                st.rerun()  

        # Generate farm simulation data and update graphs
        if st.button("Update Graphs and Tables", key="update_bottom"):
            st.session_state.update_flag = True
            generate_farm_simulation_data(st.session_state.farm_blocks, st.session_state.get('time_horizon', 50))


    elif choice == "Carbon Balance":
        st.header('Carbon Balance', divider="gray")

        if 'kpi_df' in st.session_state:
            # Annual Emissions and Removals
            st.subheader('Graph: Annual Emissions and Removals Over Time')
            hf.plot_farm_annual_emissions_removals(st.session_state.kpi_df)

            # Annual Carbon Balance
            st.subheader('Graph: Annual Carbon Balance Over Time')
            hf.plot_farm_annual_carbon_balance(st.session_state.kpi_df)

            # Cumulative Emissions and Removals
            st.subheader('Graph: Cumulative Emissions and Removals Over Time')
            hf.plot_farm_cumulative_emissions_removals(st.session_state.kpi_df)

            # Cumulative Carbon Balance
            st.subheader('Graph: Cumulative Carbon Balance Over Time')
            hf.plot_farm_cumulative_carbon_balance(st.session_state.kpi_df)

            # Tables
            with st.expander("View Carbon Balance Data Tables"):
                st.subheader("Annual Carbon Balance Data")
                hf.display_farm_annual_carbon_balance(st.session_state.kpi_df)

                st.subheader("Cumulative Carbon Balance Data")
                hf.display_farm_cumulative_carbon_balance(st.session_state.kpi_df)
        else:
            st.write("No data available. Please complete the Farm Blocks section first.")

    elif choice == "Carbon Intensity":
        st.header('Carbon Intensity', divider="gray")

        if 'kpi_df' in st.session_state:
            # Annual Carbon Intensity
            st.subheader('Graph: Annual Carbon Intensity Over Time')
            hf.plot_farm_annual_carbon_intensity(st.session_state.kpi_df)

            # Cumulative Carbon Intensity
            st.subheader('Graph: Cumulative Carbon Intensity Over Time')
            hf.plot_farm_cumulative_carbon_intensity(st.session_state.kpi_df)

            # Tables
            with st.expander("View Carbon Intensity Data Tables"):
                st.subheader("Annual Carbon Intensity Data")
                hf.display_farm_annual_carbon_intensity(st.session_state.kpi_df)

                st.subheader("Cumulative Carbon Intensity Data")
                hf.display_farm_cumulative_carbon_intensity(st.session_state.kpi_df)
        else:
            st.write("No data available. Please complete the Farm Blocks section first.")

    elif choice == "Summary":
        st.header('Summary', divider="gray")

        if 'kpi_df' in st.session_state:
            # Annual Summary Graph
            st.subheader('Graph: Annual Emissions, Removals, and Cocoa Production Over Time')
            hf.plot_farm_annual_summary(st.session_state.kpi_df)

            # Cumulative Summary Graph
            st.subheader('Graph: Cumulative Emissions, Removals, and Cocoa Production Over Time')
            hf.plot_farm_cumulative_summary(st.session_state.kpi_df)

            # Tables
            with st.expander("View Summary Data Tables"):
                st.subheader("Annual Summary Data")
                hf.display_farm_annual_summary(st.session_state.kpi_df)

                st.subheader("Cumulative Summary Data")
                hf.display_farm_cumulative_summary(st.session_state.kpi_df)
        else:
            st.write("No data available. Please complete the Farm Blocks section first.")

    elif choice == "Save/Load Farm":
        st.header('Save/Load Farm', divider="gray")

        # Save farm
        st.subheader("Save Current Farm")
        farm_name = st.text_input("Farm Name")
        farm_comment = st.text_area("Farm Comment")
        if st.button("Save Farm"):
            save_farm(farm_name, st.session_state.farm_blocks, farm_comment)
            st.success(f"Farm '{farm_name}' saved successfully!")

        # Display saved farms
        st.subheader("Saved Farms")
        saved_farms = get_saved_farms()
        if saved_farms:
            for farm in saved_farms:
                farm_data = load_farm(farm)
                st.write(f"Farm Name: {farm}")
                st.write(f"Blocks: {', '.join(farm_data['blocks']['Block'].tolist())}")
                st.write(f"Areas: {', '.join(farm_data['blocks']['Area (ha)'].astype(str).tolist())}")
                st.write(f"Comment: {farm_data['comment']}")

                if st.button(f"Delete {farm}"):
                    del st.session_state.farms[farm]
                    st.success(f"Farm '{farm}' deleted successfully!")
        else:
            st.write("No farms saved yet.")



# Farm comparison function
def farm_comparison():
    st.subheader('Farm Comparison')

    saved_farms = get_saved_farms()
    selected_farms = st.multiselect("Select Farms to Compare", saved_farms)

    if selected_farms:
        comparison_data = []
        for farm_name in selected_farms:
            farm_data = load_farm(farm_name)
            if farm_data is not None:
                farm_kpi = generate_farm_simulation_data(farm_data['blocks'])
                farm_kpi['Farm'] = farm_name
                comparison_data.append(farm_kpi)

        if comparison_data:
            comparison_df = pd.concat(comparison_data)
            
            chart = alt.Chart(comparison_df).mark_line().encode(
                x='Year:O',
                y='Carbon Balance:Q',
                color='Farm:N',
                tooltip=['Farm', 'Year', 'Carbon Balance']
            ).properties(
                title='Carbon Balance Comparison',
                width=600,
                height=400
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data available for the selected farms.")
    else:
        st.write("Please select farms to compare.")
