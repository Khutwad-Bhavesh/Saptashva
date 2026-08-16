import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_generator import generate_synthetic_data
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.stage0.train import train_stage0
from src.escalation.train import train_escalation
from src.stage0.inference import Stage0Predictor
from src.escalation.inference import EscalationPredictor

def run_tests():
    print("--- SAPTASHVA ML Pipeline Test ---")
    
    # 1. Train the models (using small epochs for test)
    print("\n1. Training Stage 0 (LSTM)...")
    train_stage0(epochs=2)
    
    print("\n2. Training Escalation (XGBoost)...")
    train_escalation()
    
    # 2. Test End-to-End Inference
    print("\n3. Testing End-to-End Inference...")
    df_test = generate_synthetic_data(num_samples=200, seed=999)
    df_feat = extract_features(df_test)
    
    feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
    
    # Needs 60 minutes lookback
    X_seq, _ = prepare_lstm_sequences(df_feat, feature_cols=feature_cols, target_col='state', lookback=60)
    
    stage0_pred = Stage0Predictor()
    esc_pred = EscalationPredictor()
    
    print(f"\nSimulating {len(X_seq)} timesteps in real-time...")
    for i in range(len(X_seq)):
        seq = X_seq[i]
        
        # Stage 0
        probs = stage0_pred.predict(seq)
        
        # Combine features for XGBoost
        last_feats = seq[-1, :]
        xgb_feats = np.concatenate([probs, last_feats]).reshape(1, -1)
        
        # Escalation
        applied_state, raw_pred = esc_pred.predict(xgb_feats)
        
        if i % 20 == 0:
            print(f"Timestep {i+60}:")
            print(f"  Stage 0 Probs [Quiet, Active, Eruptive, Recovery]: {probs}")
            print(f"  Escalation: Raw Pred = {raw_pred}, Applied = {applied_state}")
            
    print("\nTest completed successfully!")

if __name__ == '__main__':
    run_tests()
