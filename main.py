# main.py

import streamlit as st
from whole_farm import whole_farm_simulation, farm_comparison
from data_handling import initialize_inputs
from one_hectare_model import one_hectare_model

def main():
    st.title("Cocoa Farm Emissions and Carbon Storage Simulator")

    menu = ["One Hectare Model", "Whole Farm Simulation", "Farm Comparison"]
    choice = st.sidebar.selectbox("Select Mode", menu)

    initialize_inputs()

    if choice == "One Hectare Model":
        one_hectare_model()
    elif choice == "Whole Farm Simulation":
        whole_farm_simulation()
    elif choice == "Farm Comparison":
        farm_comparison()

if __name__ == '__main__':
    main()
