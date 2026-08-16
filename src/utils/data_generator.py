import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=10000, start_date='2026-08-01', seed=42):
    """
    Generates synthetic solar X-ray data for SAPTASHVA testing.
    States: 0 (Quiet), 1 (Active), 2 (Eruptive), 3 (Recovery)
    """
    np.random.seed(seed)
    
    # Base timestamps (1 minute resolution)
    start_dt = pd.to_datetime(start_date)
    timestamps = [start_dt + timedelta(minutes=i) for i in range(num_samples)]
    
    # Initialize base series
    # Soft X-ray baseline (approx 1e-8 to 1e-7 W/m^2)
    base_soft = np.random.normal(loc=1e-7, scale=1e-8, size=num_samples)
    base_soft = np.clip(base_soft, a_min=1e-8, a_max=None)
    
    # Hard X-ray baseline (usually much lower, say 1e-9 W/m^2)
    base_hard = np.random.normal(loc=1e-9, scale=1e-10, size=num_samples)
    base_hard = np.clip(base_hard, a_min=1e-10, a_max=None)
    
    states = np.zeros(num_samples, dtype=int) # default 0 (Quiet)
    
    soft_xray = base_soft.copy()
    hard_xray = base_hard.copy()
    
    # Inject flare events
    # We'll inject roughly one flare every 1000 minutes
    num_flares = num_samples // 1000
    flare_centers = np.random.choice(range(100, num_samples - 200), size=num_flares, replace=False)
    
    for center in flare_centers:
        # Define a flare profile using Gaussian-like bumps
        # Active phase (approx 30 mins before peak)
        active_start = center - 30
        
        # Eruptive phase (peak, approx 15 mins)
        eruptive_start = center
        
        # Recovery phase (approx 60 mins)
        recovery_start = center + 15
        recovery_end = center + 75
        
        # Add to state array
        states[active_start:eruptive_start] = 1 # Active
        states[eruptive_start:recovery_start] = 2 # Eruptive
        states[recovery_start:recovery_end] = 3 # Recovery
        
        # Inject flux profiles
        # Active: gradual rise
        soft_xray[active_start:eruptive_start] += np.linspace(0, 1e-6, 30)
        hard_xray[active_start:eruptive_start] += np.linspace(0, 1e-8, 30)
        
        # Eruptive: sharp peak
        soft_xray[eruptive_start:recovery_start] += np.linspace(1e-6, 1e-4, 15)
        hard_xray[eruptive_start:recovery_start] += np.linspace(1e-8, 1e-5, 15)
        
        # Recovery: exponential decay-like
        soft_xray[recovery_start:recovery_end] += np.linspace(1e-4, 0, 60)
        hard_xray[recovery_start:recovery_end] += np.linspace(1e-5, 0, 60)
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'soft_xray_flux': soft_xray,
        'hard_xray_flux': hard_xray,
        'state': states
    })
    
    return df

if __name__ == '__main__':
    df = generate_synthetic_data(1000)
    print(df.head())
    print(f"Generated {len(df)} synthetic samples.")
    print(f"State counts:\n{df['state'].value_counts()}")
