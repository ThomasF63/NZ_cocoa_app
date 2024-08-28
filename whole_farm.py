import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scenario_management import get_saved_scenarios, load_scenario, save_farm, load_farm, get_saved_farms
import helper_functions as hf

def generate_farm_simulation_data(farm_blocks, time_horizon=None):
    if time_horizon is None:
        time_horizon = st.session_state.get('time_horizon', 50)
    
    land_prep_year = st.session_state.get('land_prep_year', 2015)
    kpi_df = pd.DataFrame({'Year': range(land_prep_year, land_prep_year + time_horizon)})
    
    total_emissions = np.zeros(time_horizon)
    total_removals = np.zeros(time_horizon)
    total_area_planted = np.zeros(time_horizon)
    annual_cocoa_yield = np.zeros(time_horizon)
    cumulative_cocoa_yield = np.zeros(time_horizon)
    
    # Dictionaries to store block-specific data
    annual_emissions_dict = {}
    annual_removals_dict = {}
    cumulative_emissions_dict = {}
    cumulative_removals_dict = {}
    cocoa_yield_dict = {}

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
        cocoa_yield = scenario_data['cocoa_yield_df'].get('Yield', np.zeros(time_horizon)).values
        
        annual_emissions_per_block = np.zeros(time_horizon)
        annual_removals_per_block = np.zeros(time_horizon)
        cumulative_emissions_per_block = np.zeros(time_horizon)
        cumulative_removals_per_block = np.zeros(time_horizon)
        cocoa_yield_per_block = np.zeros(time_horizon)
        
        for year in range(time_horizon):
            if year >= start_year - land_prep_year:
                scenario_year = year - (start_year - land_prep_year)
                if scenario_year < len(annual_emissions):
                    total_emissions[year] += annual_emissions[scenario_year] * block_area
                    total_removals[year] += annual_removals[scenario_year] * block_area
                    annual_cocoa_yield[year] += cocoa_yield[scenario_year] * block_area
                    annual_emissions_per_block[year] = annual_emissions[scenario_year] * block_area
                    annual_removals_per_block[year] = annual_removals[scenario_year] * block_area
                    cocoa_yield_per_block[year] = cocoa_yield[scenario_year] * block_area
                total_area_planted[year] += block_area

        cumulative_emissions_per_block = np.cumsum(annual_emissions_per_block)
        cumulative_removals_per_block = np.cumsum(annual_removals_per_block)
        
        annual_emissions_dict[block['Block']] = annual_emissions_per_block
        annual_removals_dict[block['Block']] = annual_removals_per_block
        cumulative_emissions_dict[block['Block']] = cumulative_emissions_per_block
        cumulative_removals_dict[block['Block']] = cumulative_removals_per_block
        cocoa_yield_dict[block['Block']] = cocoa_yield_per_block

    kpi_df['Cumulative Emissions'] = np.cumsum(total_emissions)
    kpi_df['Cumulative Removals'] = np.cumsum(total_removals)
    kpi_df['Annual Emissions'] = total_emissions
    kpi_df['Annual Removals'] = total_removals
    kpi_df['Carbon Balance'] = kpi_df['Cumulative Removals'] - kpi_df['Cumulative Emissions']
    kpi_df['Total Area Planted'] = total_area_planted
    kpi_df['Annual Cocoa Yield'] = annual_cocoa_yield
    kpi_df['Cumulative Cocoa Yield'] = np.cumsum(annual_cocoa_yield)

    # Add block-specific data to kpi_df
    for block, emissions in annual_emissions_dict.items():
        kpi_df[f'{block} Annual Emissions'] = emissions
    for block, removals in annual_removals_dict.items():
        kpi_df[f'{block} Annual Removals'] = removals
    for block, emissions in cumulative_emissions_dict.items():
        kpi_df[f'{block} Cumulative Emissions'] = emissions
    for block, removals in cumulative_removals_dict.items():
        kpi_df[f'{block} Cumulative Removals'] = removals
    for block, yield_vals in cocoa_yield_dict.items():
        kpi_df[f'{block} Cocoa Yield'] = yield_vals

    st.session_state.kpi_df = kpi_df

    return kpi_df

def whole_farm_simulation():
    st.subheader('Whole Farm Simulation')

    steps = ["Farm Blocks", "Results", "Save/Load Farm"]
    choice = st.sidebar.radio("Select Step", steps)

    if choice == "Farm Blocks":
        st.subheader("Farm Blocks")

        if 'farm_blocks' not in st.session_state:
            st.session_state.farm_blocks = pd.DataFrame({
                "Block": [],
                "Area (ha)": [],
                "Year": [],
                "Scenario": []
            })

        # Planting scenario table
        st.subheader("Farm Blocks")
        st.data_editor(
            data=st.session_state.farm_blocks,
            num_rows="dynamic",
            key="farm_blocks_editor"
        )

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
                st.experimental_rerun()  # Trigger rerun to update the table

        # Block duplication and deletion controls
        st.subheader("Manage Blocks")
        block_to_duplicate = st.selectbox("Select Block to Duplicate", st.session_state.farm_blocks["Block"].tolist())
        duplicated_block_name = st.text_input("New Name for Duplicated Block")
        if st.button("Duplicate Block"):
            if duplicated_block_name in st.session_state.farm_blocks["Block"].values:
                st.error("A block with this name already exists. Please choose a different name.")
            else:
                block_data = st.session_state.farm_blocks.loc[st.session_state.farm_blocks["Block"] == block_to_duplicate].copy()
                block_data["Block"] = duplicated_block_name
                st.session_state.farm_blocks = pd.concat([st.session_state.farm_blocks, block_data], ignore_index=True)
                st.experimental_rerun()  # Trigger rerun to update the table

        block_to_delete = st.selectbox("Select Block to Delete", st.session_state.farm_blocks["Block"].tolist())
        if st.button("Delete Block"):
            st.session_state.farm_blocks = st.session_state.farm_blocks[st.session_state.farm_blocks["Block"] != block_to_delete]
            st.experimental_rerun()  # Trigger rerun to update the table

        # Generate farm simulation data and update graphs
        if st.button("Update Graphs and Tables"):
            st.session_state.update_flag = True
            generate_farm_simulation_data(st.session_state.farm_blocks, st.session_state.get('time_horizon', 50))

    elif choice == "Results":
        st.subheader("Results")

        # Display total emissions and removals graph
        if 'kpi_df' in st.session_state:
            st.subheader('Graph: Cumulative Emissions and Removals Over Time')
            hf.plot_farm_cumulative_emissions_removals(st.session_state.kpi_df)

            # Display the Total Farm Emissions and Removals
            st.subheader("Total Farm Emissions and Removals")
            hf.display_farm_simulation_data(st.session_state.kpi_df)

            # Farm Block Breakdown of Annual Emissions and Removals
            hf.display_farm_block_breakdown_annual(st.session_state.kpi_df, st.session_state.kpi_df)

            # Farm Block Breakdown of Cumulative Emissions and Removals
            hf.display_farm_block_breakdown_cumulative(st.session_state.kpi_df, st.session_state.kpi_df)

            # Farm Block Breakdown of Cocoa Production
            hf.display_farm_block_breakdown_cocoa(st.session_state.kpi_df, st.session_state.kpi_df)

    elif choice == "Save/Load Farm":
        st.subheader("Save/Load Farm")

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
