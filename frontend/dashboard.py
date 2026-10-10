import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1.plotting

# Add parent directory to path to import api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import run_telemetry_pipeline
from data_ingestion.fetcher import get_event_list, get_qualifying_session, get_available_drivers_for_q, fetch_q_lap

# --- UI Caching Wrappers ---

@st.cache_data(show_spinner=False)
def fetch_events(year):
    return get_event_list(year)

# Use cache_resource because the FastF1 Session object is complex and not easily serializable
@st.cache_resource(show_spinner=False)
def fetch_session(year, event):
    return get_qualifying_session(year, event)

@st.cache_data(show_spinner=False)
def get_drivers(_session, q_part):
    return get_available_drivers_for_q(_session, q_part)


# --- Plotting Functions ---

def plot_track_map(df, ref_driver, comp_driver, color_ref, color_comp):
    fig = go.Figure()
    
    x = df['X'].values.copy()
    y = df['Y'].values.copy()
    
    # Auto-rotate: If the track is taller than it is wide, rotate it 90 degrees
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    
    if y_span > x_span:
        x, y = y.copy(), -x.copy()
        
    dominance = df['MicroSector_Dominance'].values
    speed_diff = (df['Speed_Comp'] - df['Speed_Ref']).values
    
    # Find indices where dominance changes to group continuous blocks of the same color
    changes = np.where(dominance[:-1] != dominance[1:])[0]
    split_indices = [0] + list(changes + 1) + [len(dominance)]
    
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        # Include the next point (+1) to perfectly connect the boundary to the next block
        end_idx = min(split_indices[i+1] + 1, len(dominance))
            
        x_block = x[start_idx:end_idx]
        y_block = y[start_idx:end_idx]
        dom_val = dominance[start_idx]
        speed_diff_block = speed_diff[start_idx:end_idx]
        
        c = color_ref if dom_val == -1 else color_comp
        
        # Create rich hover text for every single GPS point
        hover_texts = []
        for sd in speed_diff_block:
            faster = comp_driver if sd > 0 else ref_driver
            hover_texts.append(f"<b>Dominant:</b> {faster}<br><b>Speed Diff:</b> {abs(sd):.1f} km/h")
            
        fig.add_trace(go.Scatter(
            x=x_block, y=y_block,
            mode='lines',
            line=dict(color=c, width=8),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=False
        ))
    
    # Lock the aspect ratio to 1:1 so the track geometry is never distorted
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=10),
        height=450,
        dragmode='pan' # Default to pan for maps instead of zoom box
    )
    
    return fig


# --- Main App ---

st.set_page_config(page_title="F1 Telemetry Coach", layout="wide")
st.title("F1 Telemetry Coach \U0001F3CE\uFE0F")

# --- Sidebar UI ---
st.sidebar.title("Telemetry Settings")

year = st.sidebar.selectbox("Year", [2024, 2023, 2022, 2021], index=0)

with st.spinner(f"Loading {year} calendar..."):
    events = fetch_events(year)
event = st.sidebar.selectbox("Grand Prix", events)

q_part = st.sidebar.selectbox("Qualifying Phase", ["Q3", "Q2", "Q1"])

if "current_event" not in st.session_state:
    st.session_state.current_event = None

if st.sidebar.button("Load Session Data"):
    st.session_state.current_event = (year, event)
    # Clear old analysis when loading new session
    if "analysis_data" in st.session_state:
        del st.session_state.analysis_data

if st.session_state.current_event is not None:
    loaded_year, loaded_event = st.session_state.current_event
    with st.spinner(f"Fetching full telemetry for {loaded_event} {loaded_year} (Takes ~10-20s)..."):
        session = fetch_session(loaded_year, loaded_event)
        
    drivers = get_drivers(session, q_part)
    
    st.sidebar.markdown("### Select Drivers")
    driver1 = st.sidebar.selectbox("Driver 1", drivers, index=0)
    driver2 = st.sidebar.selectbox("Driver 2", drivers, index=1 if len(drivers) > 1 else 0)
    
    if st.sidebar.button("Run Telemetry Analysis"):
        if driver1 == driver2:
            st.error("Please select two different drivers.")
        else:
            with st.spinner("Analyzing laps and coaching dynamics..."):
                lap1 = fetch_q_lap(session, driver1, q_part)
                lap2 = fetch_q_lap(session, driver2, q_part)
                
                # Automatically determine reference (who is faster)
                time1 = lap1['LapTime'].total_seconds()
                time2 = lap2['LapTime'].total_seconds()
                
                if time1 <= time2:
                    ref_drv, comp_drv = driver1, driver2
                    ref_lap, comp_lap = lap1, lap2
                else:
                    ref_drv, comp_drv = driver2, driver1
                    ref_lap, comp_lap = lap2, lap1
                    
                st.session_state.analysis_data = {
                    'ref_drv': ref_drv,
                    'comp_drv': comp_drv,
                    'ref_lap': ref_lap,
                    'comp_lap': comp_lap
                }


# --- Render Analysis ---
if "analysis_data" in st.session_state:
    data = st.session_state.analysis_data
    ref_drv = data['ref_drv']
    comp_drv = data['comp_drv']
    
    st.write(f"### {ref_drv} (Reference) vs {comp_drv} (Comparison)")
    
    with st.spinner("Processing Data..."):
        aligned_df, feedbacks = run_telemetry_pipeline(data['ref_lap'], data['comp_lap'])
        
    st.success("Analysis Complete!")
    
    # --- Premium Metrics Row ---
    def format_time(dt):
        sec = dt.total_seconds()
        m = int(sec // 60)
        s = sec % 60
        return f"{m}:{s:06.3f}"
        
    ref_time_str = format_time(data['ref_lap']['LapTime'])
    comp_time_str = format_time(data['comp_lap']['LapTime'])
    time_delta = data['comp_lap']['LapTime'].total_seconds() - data['ref_lap']['LapTime'].total_seconds()
    
    ref_top_speed = aligned_df['Speed_Ref'].max()
    comp_top_speed = aligned_df['Speed_Comp'].max()
    speed_delta = comp_top_speed - ref_top_speed
    
    st.markdown("### Session Overview")
    m1, m2, m3 = st.columns(3)
    
    m1.metric(label=f"🥇 {ref_drv} Lap Time (Reference)", value=ref_time_str)
    
    # delta_color="inverse" ensures that a slower time (+ seconds) is rendered in red, which is correct for racing
    m2.metric(label=f"🥈 {comp_drv} Lap Time", value=comp_time_str, delta=f"+{time_delta:.3f}s", delta_color="inverse")
    
    # delta_color="normal" ensures that more top speed (+ km/h) is rendered in green
    m3.metric(label=f"🚀 {comp_drv} Top Speed", value=f"{comp_top_speed:.0f} km/h", delta=f"{speed_delta:+.0f} km/h vs {ref_drv}", delta_color="normal")
    
    st.divider()
    
    # --- Dynamic Colors with Clash Prevention ---
    def get_driver_color(drv, fallback):
        try:
            # FastF1 3.x requires the session object to correctly map drivers to team colors for that specific year
            if "current_event" in st.session_state and st.session_state.current_event is not None:
                year, event_name = st.session_state.current_event
                s = fetch_session(year, event_name)
                return fastf1.plotting.get_driver_color(drv, session=s)
            return fallback
        except Exception as e:
            return fallback

    color_ref = get_driver_color(ref_drv, "#0000FF")
    color_comp = get_driver_color(comp_drv, "#FF0000")
    
    # Prevent intra-team visual clash (e.g. LEC vs SAI)
    if color_ref == color_comp:
        color_comp = "#00FFFF"  # High contrast Cyan fallback
    
    # Track Map
    st.subheader("Minisector Dominance Map")
    
    # Render colored text dynamically
    st.markdown(f"**<span style='color:{color_ref}'>{ref_drv}</span>** gained time | **<span style='color:{color_comp}'>{comp_drv}</span>** gained time", unsafe_allow_html=True)
    
    fig_map = plot_track_map(aligned_df, ref_drv, comp_drv, color_ref, color_comp)
    st.plotly_chart(fig_map, use_container_width=True)
    
    # --- Plotly 4-Panel Migration ---
    st.subheader(f"Interactive Telemetry ({comp_drv} vs {ref_drv})")
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=("Delta Time", "Speed", "Throttle", "Brake")
    )
    
    # Panel 1: Delta Time
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Delta_Time'], 
                             name=f"Delta ({comp_drv} - {ref_drv})", line=dict(color='white')), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Panel 2: Speed
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Speed_Ref'], 
                             name=f"{ref_drv} Speed", line=dict(color=color_ref)), row=2, col=1)
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Speed_Comp'], 
                             name=f"{comp_drv} Speed", line=dict(color=color_comp)), row=2, col=1)
    
    # Panel 3: Throttle
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Throttle_Ref'], 
                             name=f"{ref_drv} Throttle", line=dict(color=color_ref)), row=3, col=1)
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Throttle_Comp'], 
                             name=f"{comp_drv} Throttle", line=dict(color=color_comp)), row=3, col=1)
    
    # Panel 4: Brake
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Brake_Ref'] * 100, 
                             name=f"{ref_drv} Brake", line=dict(color=color_ref)), row=4, col=1)
    fig.add_trace(go.Scatter(x=aligned_df['Distance'], y=aligned_df['Brake_Comp'] * 100, 
                             name=f"{comp_drv} Brake", line=dict(color=color_comp)), row=4, col=1)
    
    # Highlight action zones
    for fb in feedbacks:
        fig.add_vrect(x0=fb['start_dist'], x1=fb['end_dist'], 
                      fillcolor="gray", opacity=0.2, layer="below", line_width=0)
                      
    fig.update_layout(height=900, template="plotly_dark", hovermode="x unified")
    fig.update_xaxes(title_text="Distance (meters)", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Accordion Feedback Layout ---
    st.subheader("Action Zone Breakdown")
    
    # Session state for expand/collapse toggle
    if "expand_all" not in st.session_state:
        st.session_state.expand_all = True
        
    col_c, col_e, _ = st.columns([2, 2, 8])
    if col_c.button("Collapse All ➖"):
        st.session_state.expand_all = False
        st.rerun()
    if col_e.button("Expand All ➕"):
        st.session_state.expand_all = True
        st.rerun()
        
    for fb in feedbacks:
        if fb.get('type') == 'straight':
            title = f"🏎️ Straight-Line Segment (Dist: {fb['start_dist']:.0f}m - {fb['end_dist']:.0f}m)"
            icon = "info"
        else:
            idx = fb.get('index', '?')
            if "Good execution" in fb['feedback'] or "Gained:" in fb['feedback']:
                title = f"✅ Action Zone {idx} (Dist: {fb['start_dist']:.0f}m)"
                icon = "success"
            elif "didn't trigger a specific pattern" in fb['feedback']:
                title = f"ℹ️ Action Zone {idx} (Dist: {fb['start_dist']:.0f}m)"
                icon = "info"
            else:
                title = f"⚠️ Action Zone {idx} (Dist: {fb['start_dist']:.0f}m)"
                icon = "warning"
                
        with st.expander(title, expanded=st.session_state.expand_all):
            if icon == "success":
                st.success(fb['feedback'])
            elif icon == "warning":
                st.warning(fb['feedback'])
            else:
                st.info(fb['feedback'])
