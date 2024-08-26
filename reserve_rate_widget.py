import streamlit as st

def reserve_rate_widget():
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])
    
    # Initialize session state variables if they don't exist
    if 'apply_reserve_rate' not in st.session_state:
        st.session_state['apply_reserve_rate'] = True
    if 'reserve_rate' not in st.session_state:
        st.session_state['reserve_rate'] = 0.5  # 50% default
    if 'temp_reserve_rate' not in st.session_state:
        st.session_state['temp_reserve_rate'] = st.session_state['reserve_rate']

    with col1:
        apply_reserve_rate = st.toggle("Apply reserve rate", value=st.session_state['apply_reserve_rate'])
    
    if apply_reserve_rate:
        with col2:
            reserve_rate = st.number_input("Rate (%)", 
                                           min_value=0.0, 
                                           max_value=100.0, 
                                           value=float(st.session_state['temp_reserve_rate'] * 100), 
                                           step=1.0,
                                           key="reserve_rate_input")
            st.session_state['temp_reserve_rate'] = reserve_rate / 100
        
        with col3:
            st.write("")  # Empty space to align the button vertically
            st.write("")
            confirm = st.button("Confirm", key="confirm_reserve_rate")
        
        if confirm:
            st.session_state['reserve_rate'] = st.session_state['temp_reserve_rate']
            st.session_state['reserve_rate_changed'] = True  # Flag to indicate change
            with col4:
                st.success("Updated")
        
        effective_rate = st.session_state['reserve_rate']
        st.info(f"Reserve rate of {effective_rate:.0%} applied on removals. You can apply/change the rate using the controls above.")
    else:
        effective_rate = 0.0
        st.session_state['apply_reserve_rate'] = False
        st.session_state['reserve_rate'] = 0.0
        st.session_state['reserve_rate_changed'] = True  # Flag to indicate change
        st.info("No reserve rate applied on removals. You can apply/change the rate using the controls above.")

    # Update the apply_reserve_rate in session state
    st.session_state['apply_reserve_rate'] = apply_reserve_rate

    # Return the effective rate (either the confirmed rate or 0 if not applied)
    return apply_reserve_rate, effective_rate