import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_generator import generate_synthetic_data
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.stage0.inference import Stage0Predictor
from src.escalation.xgboost_model import EscalationEngine
from src.data_pipeline.pradan_ingestion import process_pradan_directory, add_pseudo_labels

def generate_escalation_labels(states):
    """
    Synthesize escalation labels based on the 4 solar states.
    States: 0 (Quiet), 1 (Active), 2 (Eruptive), 3 (Recovery)
    Escalation: 0 (None), 1 (Watch), 2 (Warning), 3 (Alert)
    """
    labels = np.zeros_like(states)
    labels[states == 1] = 1 # Active -> Watch
    labels[states == 2] = 3 # Eruptive -> Alert
    # Recovery we might map to Warning or None. Let's map to Warning to see the model learn it.
    labels[states == 3] = 2 
    return labels

def train_escalation(data_dir=None):
    if data_dir and os.path.exists(data_dir):
        print(f"Loading real PRADAN data for Escalation training from {data_dir}...")
        df = process_pradan_directory(data_dir)
        df = add_pseudo_labels(df)
    else:
        print("Generating synthetic data for Escalation training...")
        df = generate_synthetic_data(5000, seed=123) # different seed
    df_features = extract_features(df)
    
    feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
    X_seq, y_state = prepare_lstm_sequences(df_features, feature_cols=feature_cols, target_col='state', lookback=60)
    
    # We need the LSTM predictions to form the input features for XGBoost
    print("Running Stage 0 inference to generate XGBoost features...")
    lstm_predictor = Stage0Predictor()
    
    X_xgb = []
    for seq in X_seq:
        probs = lstm_predictor.predict(seq)
        
        # We can also append the last timestep's raw features
        last_timestep_features = seq[-1, :]
        
        # Combine them: [prob_quiet, prob_active, prob_eruptive, prob_recovery, f1, f2, f3, f4, f5]
        combined = np.concatenate([probs, last_timestep_features])
        X_xgb.append(combined)
        
    X_xgb = np.array(X_xgb)
    
    # Generate labels for XGBoost
    y_esc = generate_escalation_labels(y_state)
    
    print("Training XGBoost Escalation Model...")
    engine = EscalationEngine()
    engine.train(X_xgb, y_esc)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'escalation_xgb.json')
    engine.save_model(model_path)
    print(f"Escalation model saved to {model_path}")

if __name__ == '__main__':
    train_escalation()
