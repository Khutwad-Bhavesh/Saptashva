import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure SAPTASHVA src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_generator import generate_synthetic_data
from src.data_pipeline.pradan_ingestion import process_pradan_directory, load_solexs_spectrum
from src.data_pipeline.goes_ingestion import parse_goes_timeseries
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.stage0.inference import Stage0Predictor
from src.escalation.inference import EscalationPredictor

# --- UI OVERHAUL: ISRO Aerospace Command CSS ---
st.set_page_config(page_title="ISRO SAPTASHVA | Telemetry Node", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Background - Strict Black */
    .stApp {
        background: #000000;
        color: #00ff00;
    }
    
    /* Monospace Headers & Aerospace Typography */
    h1, h2, h3, p, span, div {
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    h1, h2, h3 {
        font-weight: bold;
        color: #00ff00;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    h1 {
        border-bottom: 2px solid #00ff00;
        padding-bottom: 10px;
        display: inline-block;
    }
    
    /* Metric Cards - Strict Terminal Green/Black */
    div[data-testid="metric-container"] {
        background: #000000;
        border: 1px solid #00ff00;
        border-radius: 0px;
        padding: 1rem;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.6);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #00ff00;
    }
    
    /* Dataframe Styling for Logs */
    .dataframe {
        font-family: 'Courier New', Courier, monospace !important;
        font-size: 0.9rem;
        color: #00ff00 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("ISRO ADITYA-L1 // SAPTASHVA TELEMETRY NODE")
st.markdown("*Operational Solar Flare Detection & Early-Warning Architecture*")

# --- Sidebar Controls ---
st.sidebar.markdown("### MISSION CONFIGURATION")
data_source = st.sidebar.radio(
    "Telemetry Source:",
    ("Synthetic Simulation", "ISRO Aditya-L1 (PRADAN)", "NASA GOES Cycle 24")
)

pradan_dir = st.sidebar.text_input("ISRO PRADAN Target Directory:", value="/home/aizen-sosuke/Study/SAPTASHVA/datasets")
use_goes_model = data_source == "NASA GOES Cycle 24"
model_label = "V5 93.67% Foundation Model" if use_goes_model else "ISRO Base Model"
st.sidebar.info(f"Active Core: **{model_label}**")

run_button = st.sidebar.button("INITIATE UPLINK & INFERENCE", type="primary")

# --- Mappings ---
STATE_MAP = {0: "Quiet", 1: "Active", 2: "Eruptive", 3: "Recovery"}
STATE_COLORS = {0: "rgba(34, 197, 94, 0.2)", 1: "rgba(234, 179, 8, 0.2)", 2: "rgba(249, 115, 22, 0.4)", 3: "rgba(168, 85, 247, 0.2)"}
ALERT_MAP = {0: "None", 1: "Watch", 2: "Warning", 3: "CRITICAL"}
ALERT_COLORS = {0: "lightgrey", 1: "#eab308", 2: "#f97316", 3: "#ef4444"}

if run_button:
    with st.spinner("Establishing Satellite Uplink..."):
        spectrogram_data = None
        spectrogram_times = None
        
        if data_source == "Synthetic Simulation":
            df_raw = generate_synthetic_data(num_samples=300, seed=42)
            spectrogram_times = df_raw['timestamp']
            spectrogram_data = np.random.lognormal(mean=0, sigma=1, size=(300, 10))
            spectrogram_data = spectrogram_data * (df_raw['soft_xray_flux'].values[:, None] * 1e6)
            
        elif data_source == "NASA GOES Cycle 24":
            try:
                df_raw = parse_goes_timeseries()
                df_raw = df_raw.rename(columns={'soft_flux': 'soft_xray_flux', 'hard_flux': 'hard_xray_flux'})
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[-500:] 
            except Exception as e:
                st.error(f"Uplink Error: {e}")
                st.stop()
        else: # ISRO PRADAN
            try:
                df_raw = process_pradan_directory(pradan_dir)
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[:500]
                    
                lc_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(pradan_dir) for f in fn if 'SOLEXS' in f.upper() and f.endswith('.lc.gz')]
                if lc_files:
                    try:
                        times, counts_2d = load_solexs_spectrum(lc_files[0])
                        spectrogram_times = times[:500]
                        spectrogram_data = counts_2d[:500, :]
                    except:
                        pass 
            except Exception as e:
                st.error(f"ISRO Ingestion Error: {e}")
                st.stop()
                
    with st.spinner("Executing Mathematical Core..."):
        df_feat = extract_features(df_raw)
        feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
        
        if 'state' not in df_feat.columns:
            df_feat['state'] = 0 
            
        X_seq, _ = prepare_lstm_sequences(df_feat, feature_cols=feature_cols, target_col='state', lookback=60)
        
        try:
            stage0_pred = Stage0Predictor(use_goes=use_goes_model)
            esc_pred = EscalationPredictor()
        except Exception as e:
            st.error(f"Core Engine Offline. {e}")
            st.stop()
            
        results = []
        for i in range(len(X_seq)):
            seq = X_seq[i]
            probs = stage0_pred.predict(seq)
            predicted_state = np.argmax(probs)
            confidence = np.max(probs) * 100 # Extract Confidence %
            
            last_feats = seq[-1, :]
            xgb_feats = np.concatenate([probs, last_feats]).reshape(1, -1)
            
            applied_alert, raw_alert = esc_pred.predict(xgb_feats)
            ts_idx = df_feat.iloc[i + 60].name if isinstance(df_feat.index, pd.DatetimeIndex) else df_feat.iloc[i + 60]['timestamp']
            
            results.append({
                'Timestamp (UTC)': ts_idx,
                'State': STATE_MAP[predicted_state],
                'Confidence (%)': f"{confidence:.2f}%",
                'Escalation': ALERT_MAP[applied_alert],
                'Soft Flux (W/m²)': last_feats[0],
                'Hard Flux (W/m²)': last_feats[1],
                '_predicted_state': predicted_state, # Hidden for internal mapping
                '_alert_level': applied_alert
            })
            
        df_results = pd.DataFrame(results)
        
    # --- Top Metrics Dashboard ---
    latest = df_results.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ORBITAL STATE", latest['State'])
    with col2:
        st.metric("AI CREDIBILITY", latest['Confidence (%)'])
    with col3:
        st.metric("THREAT ESCALATION", latest['Escalation'])
    with col4:
        st.metric("SOFT FLUX (W/m²)", f"{latest['Soft Flux (W/m²)']:.2e}")
        
    st.markdown("---")
    
    # --- Advanced Aerospace Theming ---
    tab1, tab2, tab3 = st.tabs(["[1] 1D KINEMATICS", "[2] 2D SPECTROGRAM", "[3] TELEMETRY LOG"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_results['Timestamp (UTC)'], y=df_results['Soft Flux (W/m²)'], mode='lines', name='Soft X-ray Flux', line=dict(color='#00ff00', width=2)))
        fig.add_trace(go.Scatter(x=df_results['Timestamp (UTC)'], y=df_results['Hard Flux (W/m²)'], mode='lines', name='Hard X-ray Flux', line=dict(color='#008800', width=2, dash='dot')))
        
        for idx in range(len(df_results) - 1):
            state = df_results.iloc[idx]['_predicted_state']
            if state != 0: 
                fig.add_vrect(
                    x0=df_results.iloc[idx]['Timestamp (UTC)'], x1=df_results.iloc[idx+1]['Timestamp (UTC)'],
                    fillcolor="rgba(0, 255, 0, 0.2)", opacity=1, layer="below", line_width=0,
                )
                
        alerts = df_results[df_results['_alert_level'] > 0]
        fig.add_trace(go.Scatter(
            x=alerts['Timestamp (UTC)'], y=alerts['Soft Flux (W/m²)'], mode='markers', name='CRITICAL ESCALATION',
            marker=dict(color='#00ff00', size=12, symbol='triangle-up', line=dict(color='#000000', width=1)),
            text=[f"Alert: {ALERT_MAP[level]}" for level in alerts['_alert_level']], hoverinfo='text+x+y'
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,1)', plot_bgcolor='rgba(0,0,0,1)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,255,0,0.2)', title="UTC TIME"),
            yaxis=dict(type="log", showgrid=True, gridcolor='rgba(0,255,0,0.2)', title="FLUX (W/m²)"),
            hovermode="x unified",
            font=dict(color='#00ff00', family="Courier New")
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if spectrogram_data is not None:
            fig_spec = go.Figure(data=go.Heatmap(
                z=spectrogram_data.T,
                x=spectrogram_times,
                y=[f"Band {i}" for i in range(spectrogram_data.shape[1])],
                colorscale='Inferno',
                hoverongaps=False
            ))
            fig_spec.update_layout(
                title="SOLEXS High-Fidelity Energy Spectrogram",
                xaxis_title="UTC TIME",
                yaxis_title="ENERGY CHANNELS",
                paper_bgcolor='rgba(0,0,0,1)', plot_bgcolor='rgba(0,0,0,1)',
                font=dict(color='#00ff00', family="Courier New")
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        else:
            st.info("Spectral 2D mapping is currently offline. NASA GOES data provides scalar fluxes. Select ISRO PRADAN or Synthetic for spectrogram generation.")
            
    with tab3:
        st.markdown("### HISTORICAL PREDICTION LOG")
        st.markdown("Real-time mathematical inference validation.")
        
        # Drop internal mapping columns for the clean UI log
        display_df = df_results.drop(columns=['_predicted_state', '_alert_level'])
        # Sort so most recent is at the top
        display_df = display_df.sort_values(by='Timestamp (UTC)', ascending=False).reset_index(drop=True)
        
        # Display as a styled dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("Awaiting command sequence. Configure telemetry in the sidebar and initiate pipeline.")
