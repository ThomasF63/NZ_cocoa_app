import streamlit as st
from one_hectare_model import one_hectare_model
from planting_schedule import planting_schedule
from data_handling import initialize_inputs

def main():
    st.title("Cocoa Farm Emissions and Carbon Storage Simulator")

    menu = ["One Hectare Model", "Planting Schedule"]
    choice = st.sidebar.selectbox("Select Mode", menu)

    initialize_inputs()

    if choice == "One Hectare Model":
        one_hectare_model()
    elif choice == "Planting Schedule":
        planting_schedule()

if __name__ == '__main__':
    main()
