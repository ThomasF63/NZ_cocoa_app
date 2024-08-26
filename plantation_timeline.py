import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

def plantation_timeline():
    st.subheader('Plantation Timeline')
    
    # Initialize session state variables if they don't exist
    if 'timeline_updated' not in st.session_state:
        st.session_state.timeline_updated = False
    if 'show_messages' not in st.session_state:
        st.session_state.show_messages = False

    # Get the necessary data from session state
    luc_year = st.session_state.get('luc_event_year', 2015)
    land_prep_year = st.session_state.get('land_prep_year', 2015)
    shade_planting_year = st.session_state.get('shade_planting_year', 2016)
    cocoa_planting_year = st.session_state.get('cocoa_planting_year', 2017)
    timber_planting_year = st.session_state.get('timber_planting_year', 2016)
    cultivation_cycle_duration = st.session_state.get('cultivation_cycle_duration', 10)

    # Create the timeline dataframe
    timeline_df = create_timeline_dataframe(luc_year, land_prep_year, shade_planting_year, cocoa_planting_year, timber_planting_year, cultivation_cycle_duration)

    # Create and display the timeline chart
    create_timeline_chart(timeline_df)
    
    # Display debugging dataframe
    st.subheader("Debugging: Plantation Timeline Events")
    st.dataframe(timeline_df)
    
    # Input fields for years
    land_prep_year = st.number_input('Year of land preparation:', 
                                     min_value=2000, 
                                     max_value=2100, 
                                     value=land_prep_year, 
                                     step=1)
    
    shade_planting_year = st.number_input('Year of shade tree planting:', 
                                          min_value=2000, 
                                          max_value=2100, 
                                          value=shade_planting_year, 
                                          step=1)
    
    cocoa_planting_year = st.number_input('Year of cocoa tree planting:', 
                                          min_value=2000, 
                                          max_value=2100, 
                                          value=cocoa_planting_year, 
                                          step=1)
    
    timber_planting_year = st.number_input('Year of timber tree planting:', 
                                           min_value=2000, 
                                           max_value=2100, 
                                           value=timber_planting_year, 
                                           step=1)
    
    tree_types = ['Cocoa', 'Shade', 'Timber']
    for tree_type in tree_types:
        remove_at_cycle_end = st.checkbox(f'Remove {tree_type} trees at end of cultivation cycle?', 
                                          value=st.session_state.get(f'remove_{tree_type.lower()}_at_cycle_end', True))
        st.session_state[f'remove_{tree_type.lower()}_at_cycle_end'] = remove_at_cycle_end
        
        if remove_at_cycle_end:
            replanting_delay = st.number_input(f'Years between {tree_type.lower()} tree removal and replanting:', 
                                               min_value=0, 
                                               value=st.session_state.get(f'{tree_type.lower()}_replanting_delay', 0), 
                                               step=1)
            st.session_state[f'{tree_type.lower()}_replanting_delay'] = replanting_delay
    
    if st.button('Update Plantation Timeline'):
        st.session_state['land_prep_year'] = land_prep_year
        st.session_state['shade_planting_year'] = shade_planting_year
        st.session_state['cocoa_planting_year'] = cocoa_planting_year
        st.session_state['timber_planting_year'] = timber_planting_year
        st.session_state.timeline_updated = True
        st.session_state.show_messages = True
        st.experimental_rerun()

    if st.session_state.show_messages:
        st.success('Plantation timeline updated successfully!')
        st.info(f"Land preparation year: {st.session_state['land_prep_year']}")
        st.info(f"Shade tree planting year: {st.session_state['shade_planting_year']}")
        st.info(f"Cocoa tree planting year: {st.session_state['cocoa_planting_year']}")
        st.info(f"Timber tree planting year: {st.session_state['timber_planting_year']}")
        
        for tree_type in tree_types:
            st.info(f"{tree_type} trees removed at cycle end: {st.session_state[f'remove_{tree_type.lower()}_at_cycle_end']}")
            if st.session_state[f'remove_{tree_type.lower()}_at_cycle_end']:
                st.info(f"{tree_type} replanting delay: {st.session_state[f'{tree_type.lower()}_replanting_delay']} years")

    return land_prep_year, shade_planting_year, cocoa_planting_year, timber_planting_year



def create_timeline_dataframe(luc_year, land_prep_year, shade_planting_year, cocoa_planting_year, timber_planting_year, cultivation_cycle_duration):
    start_year = 2000
    end_year = 2100
    years = list(range(start_year, end_year + 1))
    
    df = pd.DataFrame({'Year': years})
    df['LUC'] = ['dot' if year == luc_year else '' for year in years]
    df['Land Preparation'] = ['dot' if year == land_prep_year else '' for year in years]
    
    tree_types = ['Cocoa', 'Shade', 'Timber']
    planting_years = [cocoa_planting_year, shade_planting_year, timber_planting_year]
    
    for tree_type, planting_year in zip(tree_types, planting_years):
        column = []
        current_year = planting_year
        in_delay = False
        
        for year in years:
            if year < planting_year:
                column.append('')
            elif year == current_year:
                column.append('dot')
                in_delay = False
            elif year < current_year + cultivation_cycle_duration - 1 and not in_delay:
                column.append('line')
            elif year == current_year + cultivation_cycle_duration - 1 and not in_delay:
                column.append('end')
                if st.session_state.get(f'remove_{tree_type.lower()}_at_cycle_end', True):
                    in_delay = True
                    current_year = year + 1 + st.session_state.get(f'{tree_type.lower()}_replanting_delay', 0)
                else:
                    current_year = year + 1
            elif in_delay and year < current_year:
                column.append('')
            else:
                current_year = year
                column.append('dot')
        
        df[tree_type] = column
    
    return df



def create_timeline_chart(df):
    # Melt the dataframe for easier plotting
    df_melted = df.melt(id_vars=['Year'], var_name='Event', value_name='Stage')
    
    # Create the base chart
    base = alt.Chart(df_melted).encode(
        x=alt.X('Year:O', 
                axis=alt.Axis(title='Year', labelAngle=0, format='d'),
                scale=alt.Scale(padding=0.5)  # This moves the bars between the year ticks
        ),
        y=alt.Y('Event:N', title=None, sort=['LUC', 'Land Preparation', 'Shade', 'Cocoa', 'Timber'])
    )
    
    # Create gridlines
    gridlines = alt.Chart(pd.DataFrame({'Year': range(2000, 2101)})).mark_rule().encode(
        x='Year:O',
        color=alt.condition(
            alt.datum.Year % 10 == 0,
            alt.value('lightgrey'),  # Major gridlines (every 10 years)
            alt.value('whitesmoke')  # Minor gridlines (every year)
        ),
        size=alt.condition(
            alt.datum.Year % 10 == 0,
            alt.value(1),  # Major gridlines
            alt.value(0.5)  # Minor gridlines
        )
    )
    
    # Create line segments
    lines = base.mark_bar(size=5).encode(
        x='Year:O',
        x2=alt.X2('next_Year:O'),
        color=alt.Color('Event:N', legend=None, scale=alt.Scale(
            domain=['LUC', 'Land Preparation', 'Shade', 'Cocoa', 'Timber'],
            range=['red', 'orange', 'green', 'brown', 'grey']
        ))
    ).transform_window(
        next_Year='lead(Year)',
        next_Stage='lead(Stage)'
    ).transform_filter(
        (alt.datum.Stage == 'line') & (alt.datum.next_Stage == 'line')
    )
    
    # Create circles for 'dot' stages
    dots = base.mark_circle(size=60).encode(
        color=alt.Color('Event:N', legend=None, scale=alt.Scale(
            domain=['LUC', 'Land Preparation', 'Shade', 'Cocoa', 'Timber'],
            range=['red', 'orange', 'green', 'brown', 'grey']
        ))
    ).transform_filter(
        alt.FieldEqualPredicate(field='Stage', equal='dot')
    )
    
    # Create vertical markers for 'end' stages
    ends = base.mark_tick(
        size=30,
        thickness=2,
        orient='vertical'
    ).encode(
        color=alt.Color('Event:N', legend=None, scale=alt.Scale(
            domain=['LUC', 'Land Preparation', 'Shade', 'Cocoa', 'Timber'],
            range=['red', 'orange', 'green', 'brown', 'grey']
        ))
    ).transform_filter(
        alt.FieldEqualPredicate(field='Stage', equal='end')
    )
    
    # Combine all elements
    chart = (gridlines + lines + dots + ends).properties(
        title='Plantation Timeline',
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        labelPadding=10,
        grid=False
    ).configure_view(
        strokeWidth=0
    )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)


