import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Ensure SAPTASHVA src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_generator import generate_synthetic_data
from src.data_pipeline.pradan_ingestion import process_pradan_directory
from src.data_pipeline.goes_ingestion import parse_goes_timeseries
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.stage0.inference import Stage0Predictor
from src.escalation.inference import EscalationPredictor

st.set_page_config(page_title="SAPTASHVA Dashboard", layout="wide")

st.title("SAPTASHVA: Solar Flare Early-Warning System")
st.markdown("Monitoring solar X-ray flux to detect precursors and escalate alerts using Aditya-L1 data.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

data_source = st.sidebar.radio(
    "Select Data Source:",
    ("Synthetic Data", "Real PRADAN Data", "NASA GOES Data")
)

pradan_dir = st.sidebar.text_input("PRADAN Data Directory:", value="/home/aizen-sosuke/Study/SAPTASHVA/datasets")

# Model selection based on data source
use_goes_model = data_source == "NASA GOES Data"
model_label = "GOES Foundation Model" if use_goes_model else "ISRO PRADAN Model"
st.sidebar.info(f"Active Model: **{model_label}**")

run_button = st.sidebar.button("Run Inference Pipeline")

# --- Mappings ---
STATE_MAP = {0: "Quiet", 1: "Active", 2: "Eruptive", 3: "Recovery"}
STATE_COLORS = {0: "green", 1: "yellow", 2: "orange", 3: "purple"}

ALERT_MAP = {0: "None", 1: "Watch", 2: "Warning", 3: "Alert"}
ALERT_COLORS = {0: "lightgrey", 1: "yellow", 2: "orange", 3: "red"}

# --- Main Logic ---
if run_button:
    with st.spinner("Loading Data..."):
        if data_source == "Synthetic Data":
            df_raw = generate_synthetic_data(num_samples=300, seed=42)
        elif data_source == "NASA GOES Data":
            try:
                df_raw = parse_goes_timeseries()
                # Rename columns to match SAPTASHVA standard
                df_raw = df_raw.rename(columns={'soft_flux': 'soft_xray_flux', 'hard_flux': 'hard_xray_flux'})
                # Use a representative window for dashboard demo (last 500 points)
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[-500:]
            except Exception as e:
                st.error(f"Error loading NASA GOES data: {e}")
                st.stop()
        else:
            try:
                df_raw = process_pradan_directory(pradan_dir)
                # For demo purposes, limit to a smaller window if it's too large
                if len(df_raw) > 500:
                    df_raw = df_raw.iloc[:500]
            except Exception as e:
                st.error(f"Error loading PRADAN data: {e}")
                st.stop()
                
    with st.spinner("Extracting Features & Running Inference..."):
        df_feat = extract_features(df_raw)
        feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
        
        # We need a dummy state col for the preparation function
        if 'state' not in df_feat.columns:
            df_feat['state'] = 0 
            
        X_seq, _ = prepare_lstm_sequences(df_feat, feature_cols=feature_cols, target_col='state', lookback=60)
        
        try:
            stage0_pred = Stage0Predictor(use_goes=use_goes_model)
            esc_pred = EscalationPredictor()
        except Exception as e:
            st.error(f"Failed to load models. Have you trained them yet? Error: {e}")
            st.stop()
            
        # Run inference
        results = []
        for i in range(len(X_seq)):
            seq = X_seq[i]
            probs = stage0_pred.predict(seq)
            predicted_state = np.argmax(probs)
            
            last_feats = seq[-1, :]
            xgb_feats = np.concatenate([probs, last_feats]).reshape(1, -1)
            
            applied_alert, raw_alert = esc_pred.predict(xgb_feats)
            
            # The timestamp corresponding to this prediction is at i + 60
            ts_idx = df_feat.iloc[i + 60].name if isinstance(df_feat.index, pd.DatetimeIndex) else df_feat.iloc[i + 60]['timestamp']
            
            results.append({
                'timestamp': ts_idx,
                'predicted_state': predicted_state,
                'alert_level': applied_alert,
                'soft_flux': last_feats[0],
                'hard_flux': last_feats[1]
            })
            
        df_results = pd.DataFrame(results)
        
    st.success("Inference Complete!")
    
    # --- Visualization ---
    st.subheader("Real-Time Solar Flux & Alerts")
    
    fig = go.Figure()

    # Add Soft X-ray
    fig.add_trace(go.Scatter(
        x=df_results['timestamp'], 
        y=df_results['soft_flux'],
        mode='lines',
        name='Soft X-ray Flux',
        line=dict(color='cyan')
    ))
    
    # Add Hard X-ray
    fig.add_trace(go.Scatter(
        x=df_results['timestamp'], 
        y=df_results['hard_flux'],
        mode='lines',
        name='Hard X-ray Flux',
        line=dict(color='magenta')
    ))
    
    # Add background shading for Predicted States
    for idx in range(len(df_results) - 1):
        state = df_results.iloc[idx]['predicted_state']
        if state != 0: # Only shade non-quiet states
            fig.add_vrect(
                x0=df_results.iloc[idx]['timestamp'], 
                x1=df_results.iloc[idx+1]['timestamp'],
                fillcolor=STATE_COLORS.get(state, "white"),
                opacity=0.2,
                layer="below",
                line_width=0,
            )
            
    # Add markers for Alerts
    alerts = df_results[df_results['alert_level'] > 0]
    fig.add_trace(go.Scatter(
        x=alerts['timestamp'],
        y=alerts['soft_flux'],
        mode='markers',
        name='Escalation Alerts',
        marker=dict(
            color=[ALERT_COLORS[level] for level in alerts['alert_level']],
            size=10,
            symbol='triangle-up'
        ),
        text=[f"Alert Level: {ALERT_MAP[level]}" for level in alerts['alert_level']],
        hoverinfo='text+x+y'
    ))

    fig.update_layout(
        title="SAPTASHVA X-ray Flux Monitoring",
        xaxis_title="Time",
        yaxis_title="Flux (W/m^2)",
        yaxis_type="log",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Display current status
    latest = df_results.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest State", STATE_MAP[latest['predicted_state']])
    with col2:
        st.metric("Latest Alert Level", ALERT_MAP[latest['alert_level']])
    with col3:
        st.metric("Active Model", model_label)
        
else:
    st.info("Configure the data source in the sidebar and click 'Run Inference Pipeline' to start.")
