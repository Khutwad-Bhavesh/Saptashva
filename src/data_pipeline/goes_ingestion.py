import os
import glob
import pandas as pd
import numpy as np
from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts

def download_goes_data_batched(data_dir="datasets/goes_11_year", start_year=2010, end_year=2021):
    """
    Downloads GOES XRS data in 1-year chunks to avoid timeout.
    """
    os.makedirs(data_dir, exist_ok=True)
    all_files = []
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"Querying Virtual Solar Observatory for GOES XRS data ({start_date} to {end_date})...")
        try:
            query = Fido.search(a.Time(start_date, end_date), a.Instrument("XRS"))
            print(f"Found {query.file_num} files for {year}.")
            if query.file_num > 0:
                print(f"Downloading files for {year}...")
                downloaded_files = Fido.fetch(query, path=data_dir + "/{file}")
                all_files.extend(downloaded_files)
                print(f"Year {year} complete.")
            else:
                print(f"No files found for {year}.")
        except Exception as e:
            print(f"Failed to query/download {year}: {e}")
            
    return all_files

def parse_goes_timeseries(data_dir="datasets/goes_11_year", cache_file="datasets/goes_11_year_cache.parquet"):
    """
    Parses all downloaded NetCDF/FITS GOES files into a clean Pandas DataFrame.
    Uses a fast .parquet cache to prevent re-parsing 4,000 files every run.
    """
    if os.path.exists(cache_file):
        print(f"Loading instantly from local cache: {cache_file}")
        return pd.read_parquet(cache_file)
        
    print("Cache not found. Searching for raw FITS/NetCDF files...")
    files = glob.glob(os.path.join(data_dir, "*"))
    if not files:
        print("No raw files found. Triggering batched 11-year download (2010-2021)...")
        files = download_goes_data_batched(data_dir=data_dir)
        
    print(f"Found {len(files)} raw files. Beginning massive parse operation...")
    dfs = []
    
    # Optional: fast-forward indexing for progress bar
    from tqdm import tqdm
    for f in tqdm(sorted(files), desc="Parsing files"):
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
            pass # Skip broken files silently to not spam logs
            
    if not dfs:
        raise ValueError("Failed to parse any GOES data.")
        
    master_df = pd.concat(dfs).sort_index()
    # Remove duplicates from overlapping files
    master_df = master_df[~master_df.index.duplicated(keep='first')]
    # Resample to 60s to match SAPTASHVA architecture
    master_df = master_df.resample('60s').mean().interpolate(method='linear')
    print(f"Total parsed datapoints: {len(master_df)}")
    
    print(f"Saving high-speed cache to {cache_file}...")
    master_df.to_parquet(cache_file)
    print("Cache saved successfully.")
    
    return master_df

if __name__ == "__main__":
    df = parse_goes_timeseries()
    print(df.head())
    print(f"Total rows: {len(df)}")

