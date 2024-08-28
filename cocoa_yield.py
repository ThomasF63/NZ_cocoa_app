import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

def calculate_extended_yield(cocoa_yield_df, time_horizon, cultivation_cycle_duration, cocoa_planting_year, land_prep_year):
    # Create a DataFrame for the extended yield
    extended_yield_df = pd.DataFrame({'Year': range(land_prep_year, land_prep_year + time_horizon)})
    
    # Calculate the cocoa age for each year
    extended_yield_df['Cocoa Age'] = np.maximum(0, extended_yield_df['Year'] - cocoa_planting_year)
    
    # Get the maximum age in the original yield curve
    max_age = len(cocoa_yield_df) - 1
    
    # Apply the yield based on the cocoa age, resetting at the end of each cycle
    # and repeating the last year's yield for ages beyond the original curve
    extended_yield_df['Cocoa Yield (kg/ha/yr)'] = extended_yield_df['Cocoa Age'].apply(
        lambda age: cocoa_yield_df['Cocoa Yield (kg/ha/yr)'].iloc[min(age % cultivation_cycle_duration, max_age)] 
        if age > 0 else 0
    )
    
    return extended_yield_df

def cocoa_yield_curve(cultivation_cycle_duration, time_horizon):
    st.header('Cocoa Yield Parametrization', divider="gray")
    st.subheader('Cocoa Yield Curve', divider="gray")

    # Default yield data
    default_yield = {
        'Year': list(range(1, 21)),
        'Cocoa Yield (kg/ha/yr)': [0, 0, 153, 583, 1013, 1442, 1872, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250],
        'Relative Yield (%maximum)': ['0%'] * 2 + [f"{int((153/2250)*100)}%"] + [f"{int((x/2250)*100)}%" for x in [583, 1013, 1442, 1872, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250, 2250]]
    }

    # Load data from CSV if provided
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="cocoa_yield_csv_uploader")
    if uploaded_file is not None:
        st.session_state.cocoa_yield_df = pd.read_csv(uploaded_file)
    else:
        if 'cocoa_yield_df' not in st.session_state:
            st.session_state.cocoa_yield_df = pd.DataFrame(default_yield)

    # Rename "Year" to "Cocoa Age" if necessary
    if 'Year' in st.session_state.cocoa_yield_df.columns:
        st.session_state.cocoa_yield_df.rename(columns={'Year': 'Cocoa Age'}, inplace=True)

    # Ensure the "Cocoa Age" and "Cocoa Yield (kg/ha/yr)" columns are numeric
    st.session_state.cocoa_yield_df['Cocoa Age'] = pd.to_numeric(st.session_state.cocoa_yield_df['Cocoa Age'], errors='coerce')
    st.session_state.cocoa_yield_df['Cocoa Yield (kg/ha/yr)'] = pd.to_numeric(st.session_state.cocoa_yield_df['Cocoa Yield (kg/ha/yr)'], errors='coerce')

    # Drop any NaN values
    st.session_state.cocoa_yield_df.dropna(inplace=True)

    # Editable dataframe for cocoa yield and relative yield
    edited_df = st.data_editor(
        st.session_state.cocoa_yield_df,
        key="cocoa_yield_editor",
        disabled=["Cocoa Age"]
    )

    # Update session state with edited values
    st.session_state.cocoa_yield_df = edited_df

    # Function to update relative yield
    def update_relative_yield(df):
        max_yield = df['Cocoa Yield (kg/ha/yr)'].max()
        df['Relative Yield (%maximum)'] = df['Cocoa Yield (kg/ha/yr)'].apply(lambda x: f"{int((x / max_yield) * 100)}%" if max_yield > 0 else "0%")
        return df

    # Function to update cocoa yield based on relative yield
    def update_cocoa_yield(df):
        max_yield = df['Cocoa Yield (kg/ha/yr)'].max()
        df['Cocoa Yield (kg/ha/yr)'] = df['Relative Yield (%maximum)'].apply(lambda x: (int(x.strip('%')) / 100) * max_yield if max_yield > 0 else 0)
        return df

    # Buttons to update either relative yield or cocoa yield
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Update Relative Yield'):
            st.session_state.cocoa_yield_df = update_relative_yield(st.session_state.cocoa_yield_df)
            st.success("Relative Yield updated!")
    with col2:
        if st.button('Update Cocoa Yield'):
            st.session_state.cocoa_yield_df = update_cocoa_yield(st.session_state.cocoa_yield_df)
            st.success("Cocoa Yield updated!")

    # Display the updated dataframe with both cocoa yield and relative yield
    st.subheader("Cocoa Yield and Relative Yield", divider="gray")
    st.dataframe(st.session_state.cocoa_yield_df.style.format({
        'Cocoa Age': '{:0.0f}',
        'Cocoa Yield (kg/ha/yr)': '{:.2f}',
        'Relative Yield (%maximum)': '{}'
    }))

    # Display the column chart
    chart = alt.Chart(st.session_state.cocoa_yield_df).mark_bar().encode(
        x=alt.X('Cocoa Age:Q', axis=alt.Axis(title='Cocoa Age')),
        y=alt.Y('Cocoa Yield (kg/ha/yr):Q', axis=alt.Axis(title='Cocoa Yield (kg/ha/yr)')),
        tooltip=['Cocoa Age:Q', 'Cocoa Yield (kg/ha/yr):Q', 'Relative Yield (%maximum):N']
    ).properties(title='Cocoa Yield Over Time')

    st.altair_chart(chart, use_container_width=True)

    if st.button("Validate Cocoa Yield Curve"):
        st.session_state['cocoa_yield_df'] = st.session_state.cocoa_yield_df
        st.success("Cocoa Yield Curve validated and saved!")

    # Calculate the extended yield dataframe
    cocoa_planting_year = st.session_state.get('cocoa_planting_year', st.session_state.get('land_prep_year', 2015) + 2)
    land_prep_year = st.session_state.get('land_prep_year', 2015)
    extended_yield_df = calculate_extended_yield(
        st.session_state.cocoa_yield_df,
        time_horizon,
        cultivation_cycle_duration,
        cocoa_planting_year,
        land_prep_year
    )

    # Store the extended yield dataframe in session state
    st.session_state['extended_yield_df'] = extended_yield_df

    st.subheader("Extended Cocoa Yield Forecast", divider="gray")
    st.dataframe(extended_yield_df.style.format({
        'Year': '{:0.0f}',
        'Cocoa Age': '{:0.0f}',
        'Cocoa Yield (kg/ha/yr)': '{:.2f}'
    }))

    # Create a bar chart for the extended yield
    chart = alt.Chart(extended_yield_df).mark_bar(color='#8B4513').encode(  # #8B4513 is a brown color
        x=alt.X('Year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('Cocoa Yield (kg/ha/yr):Q', axis=alt.Axis(title='Cocoa Yield (kg/ha/yr)')),
        tooltip=['Year:O', 'Cocoa Age:Q', 'Cocoa Yield (kg/ha/yr):Q']
    ).properties(
        title='Extended Cocoa Yield Forecast',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    return st.session_state.cocoa_yield_df