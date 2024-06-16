import streamlit as st
import pandas as pd
import numpy as np
from helper_functions import exponential_growth, logarithmic_growth, power_function_growth, get_equation_latex

def tree_growth_parameters(time_horizon):
    st.title("Tree Growth")
    st.header("Growth Parametrization")

    tree_types = ['Cocoa', 'Shade', 'Timber']
    growth_models = ["Logarithmic Growth (Terra Global Method)", "Exponential Plateau", "Power Function (Cool Farm Method)"]
    
    # Ensure growth_curve_selections is initialized
    if 'growth_curve_selections' not in st.session_state:
        st.session_state.growth_curve_selections = {tree_type: growth_models[0] for tree_type in tree_types}

    default_params = {
        "Exponential Plateau": {
            'beta': [0.5, 0.4, 0.3],
            'L': [100, 90, 80],
            'k': [0.1, 0.2, 0.3]
        },
        "Logarithmic Growth (Terra Global Method)": {
            'Coefficient': [39.819, 39.819, 39.819],
            'Intercept': [14.817, 14.817, 14.817],
            'Partitioning Cocoa': [0.77, 0.0, 0.0],
            'Partitioning Shade': [0.0, 0.22, 0.0],
            'Partitioning Timber': [0.0, 0.0, 0.01]
        },
        "Power Function (Cool Farm Method)": {
            'alpha': [1.6721, 0.5, 0.5],
            'beta': [0.88, 0.5, 0.5],
            'carbon_content': [0.47, 0.47, 0.47],
            'conversion_factor': [3.667, 3.667, 3.667]
        }
    }

    # Ensure default_params is initialized in session state
    if 'default_params' not in st.session_state or not st.session_state.default_params:
        st.session_state.default_params = default_params
    else:
        # Update existing default_params with any new keys/values
        for model, params in default_params.items():
            if model not in st.session_state.default_params:
                st.session_state.default_params[model] = params
            else:
                for key, value in params.items():
                    if key not in st.session_state.default_params[model]:
                        st.session_state.default_params[model][key] = value
                    elif len(st.session_state.default_params[model][key]) < len(tree_types):
                        # Extend the list to match the length of tree_types
                        st.session_state.default_params[model][key].extend(value[len(st.session_state.default_params[model][key]):len(tree_types)])

    for i, tree_type in enumerate(tree_types):
        with st.expander(f"{tree_type}", expanded=False):
            if tree_type not in st.session_state.growth_curve_selections:
                st.session_state.growth_curve_selections[tree_type] = growth_models[0]

            st.session_state.growth_curve_selections[tree_type] = st.selectbox(
                f"Select Growth Curve Model for {tree_type}",
                growth_models,
                index=growth_models.index(st.session_state.growth_curve_selections[tree_type]),
                key=f"{tree_type}_growth_model"
            )
            selected_model = st.session_state.growth_curve_selections[tree_type]
            st.latex(get_equation_latex(selected_model))

            params = {key: [st.session_state.default_params[selected_model][key][i]] for key in st.session_state.default_params[selected_model]}
            params_df = pd.DataFrame(params, index=[tree_type])
            edited_params = st.data_editor(params_df, key=f"{tree_type}_params_editor")

            for key in edited_params.columns:
                st.session_state.default_params[selected_model][key][i] = edited_params.at[tree_type, key]

            growth_curve_df = calculate_growth_curve(edited_params, time_horizon, tree_type)
            st.line_chart(growth_curve_df.set_index('Year'))

            st.dataframe(growth_curve_df)

    # Calculate combined growth models
    combined_df = combine_tree_growth_models(st.session_state.growth_curve_selections, st.session_state.default_params, time_horizon)

    st.subheader("Combined Tree Growth Models")
    summary_df = pd.DataFrame.from_dict(st.session_state.growth_curve_selections, orient='index', columns=['Selected Growth Model'])
    st.table(summary_df)
    st.dataframe(combined_df)
    st.line_chart(combined_df.set_index('Year'))

    if st.button("Validate Tree Growth Parameters"):
        st.session_state['growth_params_df'] = combined_df
        st.success("Tree Growth Parameters validated and saved!")
        st.write("Debug: growth_params_df saved in session state")
        st.write(st.session_state['growth_params_df'])

    return combined_df

def calculate_growth_curve(params_df, time_horizon, tree_type):
    t = np.linspace(1, time_horizon, time_horizon)
    selected_model = st.session_state.growth_curve_selections[tree_type]

    if selected_model == "Exponential Plateau":
        beta, L, k = params_df.at[tree_type, 'beta'], params_df.at[tree_type, 'L'], params_df.at[tree_type, 'k']
        y = exponential_growth(t, beta, L, k)
    elif selected_model == "Logarithmic Growth (Terra Global Method)":
        coefficient = params_df.at[tree_type, 'Coefficient']
        intercept = params_df.at[tree_type, 'Intercept']
        y = logarithmic_growth(t, coefficient, intercept)
        partitioning = params_df.at[tree_type, f'Partitioning {tree_type}']
        y = y * partitioning
    elif selected_model == "Power Function (Cool Farm Method)":
        alpha, beta = params_df.at[tree_type, 'alpha'], params_df.at[tree_type, 'beta']
        carbon_content = params_df.at[tree_type, 'carbon_content']
        conversion_factor = params_df.at[tree_type, 'conversion_factor']
        y = power_function_growth(t, alpha, beta, carbon_content, conversion_factor)

    growth_curve_df = pd.DataFrame({'Year': t, 'Carbon Stock (tCO2e/ha)': y})
    return growth_curve_df

def combine_tree_growth_models(growth_curve_selections, default_params, time_horizon):
    t = np.linspace(1, time_horizon, time_horizon)
    combined_df = pd.DataFrame(t, columns=['Year'])

    for i, (tree_type, growth_curve_type) in enumerate(growth_curve_selections.items()):
        if growth_curve_type not in default_params:
            st.error(f"Model '{growth_curve_type}' not found in default parameters.")
            return combined_df

        params = {key: default_params[growth_curve_type][key][i] for key in default_params[growth_curve_type]}
        params_df = pd.DataFrame(params, index=[tree_type])

        if growth_curve_type == "Exponential Plateau":
            beta, L, k = params_df.at[tree_type, 'beta'], params_df.at[tree_type, 'L'], params_df.at[tree_type, 'k']
            y = exponential_growth(t, beta, L, k)
        elif growth_curve_type == "Logarithmic Growth (Terra Global Method)":
            coefficient = params_df.at[tree_type, 'Coefficient']
            intercept = params_df.at[tree_type, 'Intercept']
            y = logarithmic_growth(t, coefficient, intercept)
            partitioning = params_df.at[tree_type, f'Partitioning {tree_type.split()[0]}']
            y = y * partitioning
        elif growth_curve_type == "Power Function (Cool Farm Method)":
            alpha, beta = params_df.at[tree_type, 'alpha'], params_df.at[tree_type, 'beta']
            carbon_content = params_df.at[tree_type, 'carbon_content']
            conversion_factor = params_df.at[tree_type, 'conversion_factor']
            y = power_function_growth(t, alpha, beta, carbon_content, conversion_factor)

        combined_df[tree_type] = y

    return combined_df
