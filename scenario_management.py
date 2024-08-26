# scenario_management.py

import streamlit as st

def save_scenario(name):
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {}
    st.session_state.scenarios[name] = {
        'time_horizon': st.session_state.get('time_horizon', 10),
        'cultivation_cycle_duration': st.session_state.get('cultivation_cycle_duration', 20),
        'luc_emission_approach': st.session_state.get('luc_emission_approach', 'At Start'),
        'amortization_years': st.session_state.get('amortization_years', 10),
        'growth_params_df': st.session_state.get('growth_params_df'),
        'cocoa_yield_df': st.session_state.get('cocoa_yield_df'),
        'emissions_df': st.session_state.get('emissions_df'),
        'annual_emissions_df': st.session_state.get('annual_emissions_df'),
        'annual_removals_df': st.session_state.get('annual_removals_df'),
    }
    st.success(f"Scenario '{name}' saved successfully!")

def load_scenario(name):
    if 'scenarios' in st.session_state and name in st.session_state.scenarios:
        scenario = st.session_state.scenarios[name]
        st.session_state['time_horizon'] = scenario['time_horizon']
        st.session_state['cultivation_cycle_duration'] = scenario['cultivation_cycle_duration']
        st.session_state['luc_emission_approach'] = scenario['luc_emission_approach']
        st.session_state['amortization_years'] = scenario['amortization_years']
        st.session_state['growth_params_df'] = scenario['growth_params_df']
        st.session_state['cocoa_yield_df'] = scenario['cocoa_yield_df']
        st.session_state['emissions_df'] = scenario['emissions_df']
        st.session_state['annual_emissions_df'] = scenario['annual_emissions_df']
        st.session_state['annual_removals_df'] = scenario['annual_removals_df']
        st.success(f"Scenario '{name}' loaded successfully!")
    else:
        st.error(f"Scenario '{name}' not found!")

def duplicate_scenario(name, new_name):
    if 'scenarios' in st.session_state and name in st.session_state.scenarios:
        st.session_state.scenarios[new_name] = st.session_state.scenarios[name]
        st.success(f"Scenario '{name}' duplicated successfully as '{new_name}'!")
    else:
        st.error(f"Scenario '{name}' not found!")

def delete_scenario(name):
    if 'scenarios' in st.session_state and name in st.session_state.scenarios:
        del st.session_state.scenarios[name]
        st.success(f"Scenario '{name}' deleted successfully!")
    else:
        st.error(f"Scenario '{name}' not found!")

def get_saved_scenarios():
    if 'scenarios' in st.session_state:
        return list(st.session_state.scenarios.keys())
    return []
