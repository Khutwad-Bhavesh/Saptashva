# SAPTASHVA

A solar flare precursor detection and early-warning system built on ISRO Aditya-L1 data (SoLEXS soft X-ray + HEL1OS hard X-ray instruments). 

## Architecture

1. **Stage 0 (State Estimator)**: A 4-state (Quiet → Active → Eruptive → Recovery) stacked LSTM with 60-minute lookback. 
2. **Escalation Layer**: An XGBoost tabular model implementing a 3-tier one-way escalating alert system (Watch → Warning → Alert).
3. **Data Ingestion**: Parses real ISRO PRADAN Level 1 FITS files (SoLEXS `.lc.gz` and HEL1OS `.fits`) and perfectly synchronizes timestamps between MJD and Unix formats.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run end-to-end test with synthetic data:
   ```bash
   python -m tests.test_pipeline
   ```
3. Run ingestion test with real PRADAN data:
   ```bash
   python -m tests.test_ingestion ./data
   ```

## Phase 4 Results
- Peak Validation Accuracy: 88.00%
- Final Validation Accuracy: 85.87%
- Data size: 260,640 samples
- Fix: Implemented Stratified Shuffle Split
- Feature: Live Streamlit UI integrated with NASA data

## Phase 5: Cloud Migration
- Pre-configured for Google Colab Pro and Vertex AI
- Supports full 11-year Solar Cycle 24 download via sunpy
- Automatically detects and utilizes A100/T4 Cloud GPUs
