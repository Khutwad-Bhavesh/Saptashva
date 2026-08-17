import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor

def calibrate_solexs(solexs_df, goes_df=None):
    """
    Calibrates SoLEXS count rates to standard GOES flux (W/m^2).
    
    If goes_df is provided (must have 'timestamp' index and 'flux' column),
    it fits a robust linear model to map SoLEXS counts to GOES flux.
    
    If goes_df is None, it uses a generic hardcoded scaling factor
    derived from standard solar physics estimates.
    """
    df = solexs_df.copy()
    
    # Generic baseline factor: 1 count/s approx 1e-10 W/m^2 (placeholder scaling)
    # This ensures that even without GOES data, the LSTM sees realistic small magnitudes
    generic_factor = 1e-10
    
    if goes_df is None or len(goes_df) == 0:
        print("No GOES data provided. Applying generic SoLEXS flux calibration.")
        df['soft_xray_flux'] = df['soft_xray_flux'] * generic_factor
        return df
        
    print("Aligning and calibrating SoLEXS against GOES data using Huber Regression...")
    
    # Align data on timestamps using a merge
    aligned = pd.merge(df, goes_df, left_index=True, right_index=True, how='inner')
    
    if len(aligned) < 10:
        print("Warning: Not enough overlapping points to calibrate robustly. Falling back to generic scaling.")
        df['soft_xray_flux'] = df['soft_xray_flux'] * generic_factor
        return df
        
    # X: SoLEXS Counts, Y: GOES Flux
    X = aligned['soft_xray_flux'].values.reshape(-1, 1)
    y = aligned['flux'].values
    
    # Use Huber Regressor to ignore massive flare outliers and get a solid baseline correlation
    model = HuberRegressor()
    model.fit(X, y)
    
    # Apply calibration to all SoLEXS data
    calibrated_flux = model.predict(df['soft_xray_flux'].values.reshape(-1, 1))
    
    # Ensure no negative flux values are produced by the linear model
    calibrated_flux = np.clip(calibrated_flux, a_min=1e-12, a_max=None)
    
    df['soft_xray_flux'] = calibrated_flux
    print(f"Calibration successful. Learned Coefficient: {model.coef_[0]:.2e}")
    
    return df
