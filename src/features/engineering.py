import pandas as pd
import numpy as np

def extract_features(df, rolling_window=10):
    """
    Extracts the 5 required features per timestep:
    1. soft X-ray flux (raw)
    2. hard X-ray flux (raw)
    3. flux ratio (soft / hard)
    4. derivative (first difference of soft flux)
    5. rolling std (of soft flux)
    
    Expects df with columns: 'timestamp', 'soft_xray_flux', 'hard_xray_flux'
    Returns df with added feature columns.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1 & 2 are already present
    
    # 3. Flux ratio (add small epsilon to avoid division by zero)
    epsilon = 1e-15
    df['flux_ratio'] = df['soft_xray_flux'] / (df['hard_xray_flux'] + epsilon)
    
    # 4. Derivative of soft flux (first difference)
    # Fill first NaN with 0
    df['soft_flux_deriv'] = df['soft_xray_flux'].diff().fillna(0)
    
    # 5. Rolling standard deviation of soft flux
    # Fill initial NaNs with 0
    df['soft_flux_roll_std'] = df['soft_xray_flux'].rolling(window=rolling_window, min_periods=1).std().fillna(0)
    
    return df

def prepare_lstm_sequences(df, feature_cols, target_col='state', lookback=60):
    """
    Converts time-series dataframe into X and y sequences for LSTM.
    lookback: number of previous timesteps to include
    """
    X, y = [], []
    data_values = df[feature_cols].values
    target_values = df[target_col].values
    
    for i in range(len(df) - lookback):
        X.append(data_values[i:(i + lookback)])
        y.append(target_values[i + lookback])
        
    return np.array(X), np.array(y)
