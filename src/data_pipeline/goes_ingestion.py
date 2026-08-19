import os
import glob
import pandas as pd
import numpy as np
from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts

def download_goes_data(data_dir="datasets/goes_2003", start_date='2003-01-01', end_date='2003-06-30'):
    """
    Downloads GOES XRS data for a given date range from the Virtual Solar Observatory.
    Default: 6 months of Solar Cycle 23 (Jan-Jun 2003) covering both quiet and active periods.
    """
    os.makedirs(data_dir, exist_ok=True)
    print(f"Querying Virtual Solar Observatory for GOES XRS data ({start_date} to {end_date})...")
    
    # Query Fido for GOES data
    query = Fido.search(a.Time(start_date, end_date), a.Instrument("XRS"))
    print(f"Found {query.file_num} files.")
    
    # Download files securely without triggering anti-bot throttling
    print("Downloading files cleanly...")
    downloaded_files = Fido.fetch(query, path=data_dir + "/{file}")
    print("Download complete.")
    return downloaded_files

def download_halloween_storms(data_dir="datasets/goes_2003"):
    """
    Legacy function: Downloads only the Halloween Storms subset (Oct-Nov 2003).
    """
    return download_goes_data(data_dir, start_date='2003-10-20', end_date='2003-11-05')

def parse_goes_timeseries(data_dir="datasets/goes_2003", expanded=False):
    """
    Parses all downloaded NetCDF/FITS GOES files into a clean Pandas DataFrame.
    If expanded=True, downloads the full 6-month dataset first.
    """
    files = glob.glob(os.path.join(data_dir, "*"))
    if not files:
        if expanded:
            print("No files found. Downloading expanded 6-month dataset...")
            files = download_goes_data(data_dir)
        else:
            print("No files found. Downloading Halloween Storms subset...")
            files = download_halloween_storms(data_dir)
        
    dfs = []
    for f in sorted(files):
        try:
            # SunPy handles the complexities of FITS/NetCDF parsing automatically
            goes_ts = ts.TimeSeries(f)
            df = goes_ts.to_dataframe()
            # SunPy standardizes columns. Usually: 'xrsa' (hard), 'xrsb' (soft)
            if 'xrsa' in df.columns and 'xrsb' in df.columns:
                df = df.rename(columns={'xrsa': 'hard_flux', 'xrsb': 'soft_flux'})
                # Only keep flux columns
                df = df[['soft_flux', 'hard_flux']]
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {f} due to parsing error: {e}")
            
    if not dfs:
        raise ValueError("Failed to parse any GOES data.")
        
    master_df = pd.concat(dfs).sort_index()
    # Remove duplicates from overlapping files
    master_df = master_df[~master_df.index.duplicated(keep='first')]
    # Resample to 60s to match SAPTASHVA architecture
    master_df = master_df.resample('60s').mean().interpolate(method='linear')
    print(f"Total parsed datapoints: {len(master_df)}")
    return master_df

if __name__ == "__main__":
    df = parse_goes_timeseries()
    print(df.head())
    print(f"Total rows: {len(df)}")
