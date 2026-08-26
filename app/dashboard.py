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

# --- UI OVERHAUL: Glassmorphism & Cyberpunk CSS ---
st.set_page_config(page_title="SAPTASHVA | Aerospace Command", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Neon Glow Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.2), 0 4px 6px -2px rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.5);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("SAPTASHVA // COMMAND CENTER")
st.markdown("*Advanced Orbital Detection Architecture for Aditya-L1 and GOES Satellites*")

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

run_button = st.sidebar.button("INITIATE INFERENCE PIPELINE", type="primary")

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
            # Generate synthetic 2D spectrogram (Time x Energy Bins)
            spectrogram_times = df_raw['timestamp']
            spectrogram_data = np.random.lognormal(mean=0, sigma=1, size=(300, 10))
            spectrogram_data = spectrogram_data * (df_raw['soft_xray_flux'].values[:, None] * 1e6)
            
        elif data_source == "NASA GOES Cycle 24":
            try:
                df_raw = parse_goes_timeseries()
                df_raw = df_raw.rename(columns={'soft_flux': 'soft_xray_flux', 'hard_flux': 'hard_xray_flux'})
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[-500:] # Grab tail for UI performance
            except Exception as e:
                st.error(f"Uplink Error: {e}")
                st.stop()
        else: # ISRO PRADAN
            try:
                df_raw = process_pradan_directory(pradan_dir)
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[:500]
                    
                # Attempt to extract spectral cube for spectrogram
                lc_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(pradan_dir) for f in fn if 'SOLEXS' in f.upper() and f.endswith('.lc.gz')]
                if lc_files:
                    try:
                        times, counts_2d = load_solexs_spectrum(lc_files[0])
                        spectrogram_times = times[:500]
                        spectrogram_data = counts_2d[:500, :]
                    except:
                        pass # Fallback if spectral extraction fails
            except Exception as e:
                st.error(f"ISRO Ingestion Error: {e}")
                st.stop()
                
    with st.spinner("Extracting Temporal Features & Running Core Engine..."):
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
            
            last_feats = seq[-1, :]
            xgb_feats = np.concatenate([probs, last_feats]).reshape(1, -1)
            
            applied_alert, raw_alert = esc_pred.predict(xgb_feats)
            ts_idx = df_feat.iloc[i + 60].name if isinstance(df_feat.index, pd.DatetimeIndex) else df_feat.iloc[i + 60]['timestamp']
            
            results.append({
                'timestamp': ts_idx,
                'predicted_state': predicted_state,
                'alert_level': applied_alert,
                'soft_flux': last_feats[0],
                'hard_flux': last_feats[1]
            })
            
        df_results = pd.DataFrame(results)
        
    # --- Top Metrics Dashboard ---
    latest = df_results.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CURRENT ORBITAL STATE", STATE_MAP[latest['predicted_state']])
    with col2:
        st.metric("THREAT ESCALATION", ALERT_MAP[latest['alert_level']])
    with col3:
        st.metric("SOFT FLUX (W/m²)", f"{latest['soft_flux']:.2e}")
    with col4:
        st.metric("ACTIVE CORE", model_label)
        
    st.markdown("---")
    
    # --- Advanced Plotly Theming ---
    tab1, tab2 = st.tabs(["1D LIGHTCURVE KINEMATICS", "2D SPECTRAL FOOTPRINT"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_results['timestamp'], y=df_results['soft_flux'], mode='lines', name='Soft X-ray Flux', line=dict(color='#06b6d4', width=2)))
        fig.add_trace(go.Scatter(x=df_results['timestamp'], y=df_results['hard_flux'], mode='lines', name='Hard X-ray Flux', line=dict(color='#ec4899', width=2)))
        
        for idx in range(len(df_results) - 1):
            state = df_results.iloc[idx]['predicted_state']
            if state != 0: 
                fig.add_vrect(
                    x0=df_results.iloc[idx]['timestamp'], x1=df_results.iloc[idx+1]['timestamp'],
                    fillcolor=STATE_COLORS.get(state, "white"), opacity=1, layer="below", line_width=0,
                )
                
        alerts = df_results[df_results['alert_level'] > 0]
        fig.add_trace(go.Scatter(
            x=alerts['timestamp'], y=alerts['soft_flux'], mode='markers', name='CRITICAL ESCALATION',
            marker=dict(color=[ALERT_COLORS[level] for level in alerts['alert_level']], size=12, symbol='triangle-up', line=dict(color='white', width=1)),
            text=[f"Alert: {ALERT_MAP[level]}" for level in alerts['alert_level']], hoverinfo='text+x+y'
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="UTC TIME"),
            yaxis=dict(type="log", showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="FLUX (W/m²)"),
            hovermode="x unified",
            font=dict(color='#94a3b8')
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
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8')
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        else:
            st.info("Spectral 2D mapping is currently offline. NASA GOES data provides scalar fluxes. Select ISRO PRADAN or Synthetic for spectrogram generation.")
else:
    st.info("Awaiting command sequence. Configure telemetry in the sidebar and initiate pipeline.")
