import pandas as pd
import numpy as np

def apply_neupert_enrichment(df):
    """
    Applies the Neupert effect physics principle:
    Hard X-ray flux is proportional to the time derivative of Soft X-ray flux.
    
    If HEL1OS hard X-ray flux is missing (0 or NaN), this fills the gap 
    using the smoothed derivative of the calibrated SoLEXS soft X-ray flux.
    """
    df = df.copy()
    
    # Calculate smoothed derivative of soft flux
    smoothed_soft = df['soft_xray_flux'].rolling(window=5, min_periods=1).mean()
    soft_deriv = smoothed_soft.diff().fillna(0)
    
    # The derivative can be negative during recovery, but hard X-ray flux is non-negative
    # We clip it to 0 and apply a proportional scaling constant k
    k = 0.05 # Baseline estimated proportionality constant
    synthetic_hxr = np.clip(soft_deriv, a_min=0, a_max=None) * k
    
    # Fill gaps in hard_xray_flux where it's 0 or NaN
    mask = (df['hard_xray_flux'].isna()) | (df['hard_xray_flux'] <= 1e-15)
    df.loc[mask, 'hard_xray_flux'] = synthetic_hxr[mask]
    
    return df

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
    
    # Apply Neupert gap-filling before extracting derived features
    if 'hard_xray_flux' in df.columns:
        df = apply_neupert_enrichment(df)
    
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
