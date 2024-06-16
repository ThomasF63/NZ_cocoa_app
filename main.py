import streamlit as st
from one_hectare_model import one_hectare_model
from planting_schedule import planting_schedule

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["One Hectare Model", "Planting Schedule"])

    if app_mode == "One Hectare Model":
        one_hectare_model()
    elif app_mode == "Planting Schedule":
        planting_schedule()

if __name__ == "__main__":
    main()
