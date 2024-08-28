import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from helper_functions import exponential_growth, logarithmic_growth, power_function_growth, get_equation_latex

def tree_growth_parameters(time_horizon):
    st.header("Tree Growth Parametrization", divider="gray")
    st.subheader("Individual Tree Growth Models", divider="gray")

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
    if 'default_params' not in st.session_state:
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

            # Display the formatted version of the parameters
            st.dataframe(edited_params.style.format("{:.2f}"))

            for key in edited_params.columns:
                st.session_state.default_params[selected_model][key][i] = edited_params.at[tree_type, key]

            growth_curve_df = calculate_growth_curve(edited_params, time_horizon, tree_type)
            
            chart = alt.Chart(growth_curve_df).mark_line().encode(
                x=alt.X('Age:Q', axis=alt.Axis(format='d')),
                y=alt.Y('Carbon Stock (tCO2e/ha):Q', axis=alt.Axis(format='d')),
                tooltip=[
                    alt.Tooltip('Age:Q', format='d'),
                    alt.Tooltip('Carbon Stock (tCO2e/ha):Q', format='.2f')
                ]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(growth_curve_df.style.format({
                'Age': '{:0.0f}',
                'Carbon Stock (tCO2e/ha)': '{:.2f}'
            }))

    # Calculate combined growth models
    combined_df = combine_tree_growth_models(st.session_state.growth_curve_selections, st.session_state.default_params, time_horizon)

    st.subheader("Combined Tree Growth Models", divider="gray")
    summary_df = pd.DataFrame.from_dict(st.session_state.growth_curve_selections, orient='index', columns=['Selected Growth Model'])
    st.table(summary_df)
    
    st.dataframe(combined_df.style.format({
        'Year Number': '{:0.0f}',
        'Cocoa': '{:.2f}',
        'Shade': '{:.2f}',
        'Timber': '{:.2f}'
    }))
    
    # Create a melted dataframe for plotting
    melted_df = combined_df.melt('Year Number', var_name='Tree Type', value_name='Carbon Stock (tCO2e/ha)')
    
    # Create the line chart
    chart = alt.Chart(melted_df).mark_line().encode(
        x=alt.X('Year Number:Q', axis=alt.Axis(format='d', title='Year')),
        y=alt.Y('Carbon Stock (tCO2e/ha):Q', axis=alt.Axis(format='d')),
        color='Tree Type:N',
        tooltip=[
            alt.Tooltip('Year Number:Q', title='Year', format='d'),
            alt.Tooltip('Tree Type:N'),
            alt.Tooltip('Carbon Stock (tCO2e/ha):Q', format='.2f')
        ]
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

    if st.button("Validate Tree Growth Parameters"):
        st.session_state['growth_params_df'] = combined_df
        st.success("Tree Growth Parameters validated and saved!")
        st.info(f"Growth parameters saved for {', '.join(tree_types)}")

    return combined_df

def calculate_growth_curve(params_df, time_horizon, tree_type):
    t = np.arange(1, time_horizon + 1, dtype=int)
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

    growth_curve_df = pd.DataFrame({'Age': t, 'Carbon Stock (tCO2e/ha)': y})
    return growth_curve_df

def combine_tree_growth_models(growth_curve_selections, default_params, time_horizon):
    year_numbers = np.arange(1, time_horizon + 1, dtype=int)
    combined_df = pd.DataFrame({'Year Number': year_numbers})

    for tree_type, growth_curve_type in growth_curve_selections.items():
        age = year_numbers  # Use year numbers directly as age

        # Safely get parameters for the current tree type
        params = {}
        for key in default_params[growth_curve_type]:
            param_list = default_params[growth_curve_type][key]
            param_index = min(list(growth_curve_selections.keys()).index(tree_type), len(param_list) - 1)
            params[key] = param_list[param_index]

        params_df = pd.DataFrame(params, index=[tree_type])

        y = np.zeros_like(age, dtype=float)  # Initialize with zeros

        if growth_curve_type == "Exponential Plateau":
            beta, L, k = params_df.at[tree_type, 'beta'], params_df.at[tree_type, 'L'], params_df.at[tree_type, 'k']
            y = exponential_growth(age, beta, L, k)
        elif growth_curve_type == "Logarithmic Growth (Terra Global Method)":
            coefficient = params_df.at[tree_type, 'Coefficient']
            intercept = params_df.at[tree_type, 'Intercept']
            y = logarithmic_growth(age, coefficient, intercept)
            partitioning = params_df.at[tree_type, f'Partitioning {tree_type.split()[0]}']
            y *= partitioning
        elif growth_curve_type == "Power Function (Cool Farm Method)":
            alpha, beta = params_df.at[tree_type, 'alpha'], params_df.at[tree_type, 'beta']
            carbon_content = params_df.at[tree_type, 'carbon_content']
            conversion_factor = params_df.at[tree_type, 'conversion_factor']
            y = power_function_growth(age, alpha, beta, carbon_content, conversion_factor)

        combined_df[tree_type] = y

    return combined_df