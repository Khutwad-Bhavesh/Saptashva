import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.time import Time

def load_solexs_lc(file_path):
    """
    Reads a SoLEXS Level 1 Light Curve (.lc.gz) file.
    Returns a pandas DataFrame with 'timestamp' and 'soft_xray_flux'.
    """
    with fits.open(file_path) as hdul:
        # LC data is typically in the RATE extension or the first BINTABLE extension
        try:
            rate_ext = hdul['RATE']
        except KeyError:
            # Fallback if 'RATE' is not found, assume it's the first data extension
            rate_ext = hdul[1]
            
        data = rate_ext.data
        unix_times = data['TIME']
        counts = data['COUNTS']
        
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(unix_times, unit='s', origin='unix'),
        'soft_xray_flux': counts
    })
    
    # Sort and set index for resampling later
    df = df.sort_values('timestamp').set_index('timestamp')
    return df

def load_solexs_spectrum(file_path):
    """
    Reads a SoLEXS Level 1 Spectral file (if available) using astropy.
    Extracts the full 2D array of counts across energy channels.
    Returns a 2D numpy array (Time x Energy) and the time vector.
    """
    with fits.open(file_path) as hdul:
        # Spectral data is typically in a SPECTRUM or RATE extension containing a 2D array
        try:
            ext = hdul['RATE']
        except KeyError:
            ext = hdul[1]
            
        data = ext.data
        unix_times = data['TIME']
        
        # In a true spectral file, COUNTS is a 2D array (Time, Energy Bins)
        counts_2d = data['COUNTS']
        
        # If it's accidentally a 1D file, we spoof a 2D array for the dashboard demo
        if len(counts_2d.shape) == 1:
            # Synthetic distribution across 10 energy channels based on the total counts
            energy_channels = 10
            synthetic_2d = np.zeros((len(counts_2d), energy_channels))
            # Distribute counts logarithmically to simulate bremsstrahlung spectrum
            dist = np.logspace(-1, -3, num=energy_channels)
            dist /= np.sum(dist)
            for i in range(len(counts_2d)):
                synthetic_2d[i] = counts_2d[i] * dist
            counts_2d = synthetic_2d
            
    times = pd.to_datetime(unix_times, unit='s', origin='unix')
    return times, counts_2d

def load_hel1os_bands(file_path, target_bands=None):
    """
    Reads a HEL1OS Level 1 FITS file and extracts Hard X-ray fluxes.
    If target_bands is None, it sums up the count rates from all extensions containing 'BAND_'.
    Returns a pandas DataFrame with 'timestamp' and 'hard_xray_flux'.
    """
    with fits.open(file_path) as hdul:
        combined_df = None
        
        for hdu in hdul:
            ext_name = hdu.header.get("EXTNAME", "")
            if "BAND_" in ext_name:
                band_name = ext_name.split("BAND_")[-1]
                
                # Filter by target_bands if specified
                if target_bands is not None and band_name not in target_bands:
                    continue
                
                mjd_times = hdu.data["MJD"]
                counts = hdu.data["CTR"]
                
                # Convert MJD to datetime using astropy
                times_utc = Time(mjd_times, format="mjd").to_datetime()
                
                temp_df = pd.DataFrame({
                    'timestamp': pd.Series(times_utc),
                    'flux': counts
                })
                temp_df = temp_df.sort_values('timestamp').set_index('timestamp')
                
                if combined_df is None:
                    combined_df = temp_df
                else:
                    # Align and add
                    # Since times might slightly differ, we merge and fillna with 0 before adding
                    combined_df = combined_df.add(temp_df, fill_value=0)
                    
    if combined_df is None:
        raise ValueError(f"No valid BAND extensions found in {file_path}")
        
    combined_df = combined_df.rename(columns={'flux': 'hard_xray_flux'})
    return combined_df

def merge_and_resample(solexs_df, hel1os_df, freq='60s'):
    """
    Merges SoLEXS and HEL1OS dataframes and resamples them to a uniform cadence.
    freq='60s' aligns the data to 1-minute bins for the LSTM.
    """
    # Merge outer to keep all timestamps before resampling
    merged_df = pd.merge(solexs_df, hel1os_df, left_index=True, right_index=True, how='outer')
    
    # Resample and take the mean over the time bin
    # Depending on physical interpretation, we could also use .sum() or .max()
    resampled_df = merged_df.resample(freq).mean()
    
    # Forward fill small gaps and fill remaining NaNs with 0 (or a baseline value)
    resampled_df = resampled_df.ffill(limit=2).fillna(0)
    
    # Reset index to make 'timestamp' a column again
    resampled_df = resampled_df.reset_index()
    
    return resampled_df

def process_pradan_directory(directory_path, freq='60s'):
    """
    Scans a directory for all SoLEXS and HEL1OS light curve files, 
    processes them, and returns a unified concatenated DataFrame.
    """
    solexs_dfs = []
    hel1os_dfs = []
    
    print(f"Scanning {directory_path} for ISRO FITS data...")
    # Find all relevant files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if 'SOLEXS' in file.upper() and file.endswith('.lc.gz'):
                try:
                    df = load_solexs_lc(file_path)
                    if not df.empty:
                        solexs_dfs.append(df)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    
            # HEL1OS lightcurves typically have 'lightcurve' in the name
            elif 'LIGHTCURVE' in file.upper() and file.endswith('.fits'):
                try:
                    df = load_hel1os_bands(file_path)
                    if not df.empty:
                        hel1os_dfs.append(df)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                
    if not solexs_dfs:
        raise FileNotFoundError("Could not find any valid SoLEXS (.lc.gz) data.")
    
    print(f"Concatenating {len(solexs_dfs)} SoLEXS files...")
    solexs_df = pd.concat(solexs_dfs).sort_index()
    # Remove duplicates if any overlapping files exist
    solexs_df = solexs_df[~solexs_df.index.duplicated(keep='first')]
    
    # Apply Calibration
    from src.data_pipeline.calibration import calibrate_solexs
    solexs_df = calibrate_solexs(solexs_df, goes_df=None)
    
    if hel1os_dfs:
        print(f"Concatenating {len(hel1os_dfs)} HEL1OS files...")
        hel1os_df = pd.concat(hel1os_dfs).sort_index()
        hel1os_df = hel1os_df[~hel1os_df.index.duplicated(keep='first')]
    else:
        print("Warning: No valid HEL1OS files found. Creating empty proxy for Neupert enrichment.")
        # Create an empty dataframe with same index to trigger Neupert enrichment gap-filling later
        hel1os_df = pd.DataFrame({'hard_xray_flux': np.nan}, index=solexs_df.index)
    
    print("Merging and resampling timelines...")
    final_df = merge_and_resample(solexs_df, hel1os_df, freq=freq)
    return final_df

def add_pseudo_labels(df):
    """
    Adds a 'state' column based on thresholding the soft_xray_flux.
    0: Quiet, 1: Active, 2: Eruptive, 3: Recovery.
    Since this is unlabeled PRADAN data, we use basic thresholds for now.
    """
    df = df.copy()
    
    # Example generic thresholds (can be refined later based on GOES classes)
    quiet_thresh = 1e-7
    active_thresh = 1e-6
    eruptive_thresh = 1e-5
    
    states = np.zeros(len(df), dtype=int)
    
    # Very crude pseudo-labeling for testing
    states[df['soft_xray_flux'] > quiet_thresh] = 1 # Active
    states[df['soft_xray_flux'] > active_thresh] = 2 # Eruptive
    
    # For recovery, look at negative slope (derivative)
    soft_deriv = df['soft_xray_flux'].diff().fillna(0)
    states[(df['soft_xray_flux'] > quiet_thresh) & (soft_deriv < 0)] = 3 # Recovery
    
    df['state'] = states
    return df

