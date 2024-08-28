import streamlit as st
import pandas as pd

def initialize_scenarios():
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}

def save_scenario(name, comment):
    initialize_scenarios()
    st.session_state.scenarios[name] = {
        'time_horizon': st.session_state.get('time_horizon', 50),
        'cultivation_cycle_duration': st.session_state.get('cultivation_cycle_duration', 20),
        'growth_params_df': st.session_state.get('growth_params_df'),
        'cocoa_yield_df': st.session_state.get('cocoa_yield_df'),
        'emissions_df': st.session_state.get('emissions_df'),
        'annual_emissions_df': st.session_state.get('annual_emissions_df'),
        'annual_removals_df': st.session_state.get('annual_removals_df'),
        'comment': comment
    }

def load_scenario(name):
    initialize_scenarios()
    if name in st.session_state.scenarios:
        scenario = st.session_state.scenarios[name]
        for key, value in scenario.items():
            if key != 'comment':
                st.session_state[key] = value
        return scenario
    return None

def get_saved_scenarios():
    initialize_scenarios()
    return list(st.session_state.scenarios.keys())

def duplicate_scenario(name, new_name):
    initialize_scenarios()
    if name in st.session_state.scenarios:
        st.session_state.scenarios[new_name] = st.session_state.scenarios[name].copy()
        st.session_state.scenarios[new_name]['comment'] += " (Duplicated)"

def delete_scenario(name):
    initialize_scenarios()
    if name in st.session_state.scenarios:
        del st.session_state.scenarios[name]

def display_saved_scenarios_table():
    initialize_scenarios()
    if st.session_state.scenarios:
        scenarios_data = [
            {
                'Name': name,
                'Time Horizon': scenario['time_horizon'],
                'Cultivation Cycle': scenario['cultivation_cycle_duration'],
                'Comment': scenario.get('comment', '')
            }
            for name, scenario in st.session_state.scenarios.items()
        ]
        scenarios_df = pd.DataFrame(scenarios_data)
        st.table(scenarios_df)
    else:
        st.write("No scenarios saved yet.")

def save_farm(name, blocks, comment):
    if 'farms' not in st.session_state:
        st.session_state.farms = {}
    st.session_state.farms[name] = {
        'blocks': blocks,
        'comment': comment
    }

def load_farm(name):
    if 'farms' in st.session_state and name in st.session_state.farms:
        return st.session_state.farms[name]
    return None

def get_saved_farms():
    if 'farms' not in st.session_state:
        st.session_state.farms = {}
    return list(st.session_state.farms.keys())

def scenario_management_section():
    initialize_scenarios()

    st.header("Manage Scenarios", divider="gray")
    st.subheader("Recalculate All Sections")
    
    if st.button("Recalculate All Sections"):
        from data_handling import apply_global_reserve_rate
        apply_global_reserve_rate()
        st.success("All sections have been recalculated!")
        st.rerun()

    st.subheader("Save Current Scenario")
    scenario_name = st.text_input("Scenario Name")
    scenario_comment = st.text_area("Scenario Comment")
    if st.button("Save Scenario"):
        if scenario_name:
            save_scenario(scenario_name, scenario_comment)
            st.success(f"Scenario '{scenario_name}' saved successfully!")
            st.rerun()
        else:
            st.error("Please enter a name for the scenario.")

    st.subheader("Load Saved Scenario")
    saved_scenarios = get_saved_scenarios()
    if saved_scenarios:
        selected_scenario = st.selectbox("Select Scenario to Load", saved_scenarios)
        if st.button("Load Scenario"):
            load_scenario(selected_scenario)
            st.success(f"Scenario '{selected_scenario}' loaded successfully!")
            st.rerun()
    else:
        st.info("No saved scenarios available.")

    st.subheader("Duplicate Scenario")
    if saved_scenarios:
        duplicate_scenario_name = st.selectbox("Select Scenario to Duplicate", saved_scenarios)
        new_scenario_name = st.text_input("New Scenario Name")
        if st.button("Duplicate Scenario"):
            if new_scenario_name:
                duplicate_scenario(duplicate_scenario_name, new_scenario_name)
                st.success(f"Scenario '{duplicate_scenario_name}' duplicated as '{new_scenario_name}'")
                st.rerun()
            else:
                st.error("Please enter a name for the new scenario.")
    else:
        st.info("No scenarios available to duplicate.")

    st.subheader("Delete Scenario")
    if saved_scenarios:
        delete_scenario_name = st.selectbox("Select Scenario to Delete", saved_scenarios)
        if st.button("Delete Scenario"):
            delete_scenario(delete_scenario_name)
            st.success(f"Scenario '{delete_scenario_name}' deleted successfully!")
            st.rerun()
    else:
        st.info("No scenarios available to delete.")

    st.subheader("Saved Scenarios")
    display_saved_scenarios_table()