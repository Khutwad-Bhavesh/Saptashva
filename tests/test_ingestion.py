import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.pradan_ingestion import merge_and_resample, process_pradan_directory

def test_merge_logic():
    print("Testing DataFrame merging and resampling logic...")
    
    # Mock SoLEXS (1 second cadence)
    base_time = datetime(2025, 1, 1, 12, 0, 0)
    solexs_times = [base_time + timedelta(seconds=i) for i in range(120)]
    solexs_flux = np.random.uniform(1e-8, 1e-7, size=120)
    solexs_df = pd.DataFrame({'timestamp': solexs_times, 'soft_xray_flux': solexs_flux})
    solexs_df = solexs_df.set_index('timestamp')
    
    # Mock HEL1OS (slightly misaligned cadence, say every 1.5 seconds)
    hel1os_times = [base_time + timedelta(seconds=i*1.5) for i in range(80)]
    hel1os_flux = np.random.uniform(1e-10, 1e-9, size=80)
    hel1os_df = pd.DataFrame({'timestamp': hel1os_times, 'hard_xray_flux': hel1os_flux})
    hel1os_df = hel1os_df.set_index('timestamp')
    
    merged = merge_and_resample(solexs_df, hel1os_df, freq='60S')
    
    print("Merged output shape:", merged.shape)
    print(merged.head())
    
    assert len(merged) == 2, "Should result in 2 minutes of binned data"
    assert 'soft_xray_flux' in merged.columns
    assert 'hard_xray_flux' in merged.columns
    assert not merged.isnull().values.any(), "Should not have NaNs after resampling and filling"
    print("Merge logic test passed!")

def test_real_data(data_dir):
    print(f"\nTesting real PRADAN ingestion from {data_dir}...")
    try:
        df = process_pradan_directory(data_dir)
        print(f"Successfully loaded PRADAN data! Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Failed to load real data: {e}")

if __name__ == '__main__':
    test_merge_logic()
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        if os.path.exists(data_dir):
            test_real_data(data_dir)
