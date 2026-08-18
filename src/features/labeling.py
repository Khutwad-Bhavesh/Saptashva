import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

def apply_physics_labels(df, flux_col='soft_flux'):
    """
    Replaces crude thresholds with advanced Scipy Peak Detection.
    Labels:
    0 = Quiet
    1 = Active
    2 = Eruptive (Onset to Peak)
    3 = Recovery (Peak to End)
    """
    print("Applying advanced physics peak detection...")
    
    # Smooth the curve slightly to avoid micro-anomalies
    smoothed = df[flux_col].rolling(window=5, min_periods=1, center=True).mean()
    
    # 1. Find the major flare peaks (height > C-class flare threshold, e.g., 1e-6 W/m^2)
    # distance=60 ensures we don't double count the same flare peak within 60 minutes
    peaks, properties = find_peaks(smoothed, height=1e-6, distance=60, prominence=5e-7)
    
    # 2. Calculate the exact width (start and end) of each peak at half-prominence
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed, peaks, rel_height=0.8)
    
    # Initialize all states to Quiet (0)
    labels = np.zeros(len(df), dtype=int)
    
    # Active baseline (if flux is above a B-class threshold)
    labels[df[flux_col] > 1e-7] = 1
    
    # Loop over every detected flare and mathematically assign Eruptive and Recovery states
    for i in range(len(peaks)):
        peak_idx = peaks[i]
        start_idx = int(left_ips[i])
        end_idx = int(right_ips[i])
        
        # State 2: Eruptive (From the exact start of the slope up to the peak)
        labels[start_idx:peak_idx] = 2
        
        # State 3: Recovery (From the peak down to the calculated end of the flare)
        labels[peak_idx:end_idx] = 3

    df['state'] = labels
    print(f"Detected {len(peaks)} major flare events. Labels applied successfully.")
    return df

def extract_derivatives(df):
    """
    Extracts rolling derivatives to capture the exact velocity of the X-ray curve.
    """
    df['soft_deriv'] = df['soft_flux'].diff()
    df['hard_deriv'] = df['hard_flux'].diff()
    df['flux_ratio'] = df['hard_flux'] / (df['soft_flux'] + 1e-12)
    
    # Fill NaN values created by diff()
    df = df.fillna(method='bfill')
    return df
