"""
EEG Stress-Detection Pipeline
Data Ingestion Script

Downloads and processes PhysioNet EEG data (Subject 1, Runs 1 & 2),
selects only channels FP1 and FP2, applies a bandpass filter (0.16 - 40 Hz),
and resamples the data to 128 Hz.
"""

import mne

def process_eeg_data(subject=1, runs=None):
    if runs is None:
        runs = [1, 2]
    
    print(f"Downloading/Loading data for Subject {subject}, Runs {runs}...")
    # This will download the dataset if not already present
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    
    print("Reading raw EDF files...")
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    
    print("Concatenating runs...")
    raw = mne.concatenate_raws(raws)
    
    print("Standardizing channel names...")
    mne.datasets.eegbci.standardize(raw)
    
    print("Selecting FP1 and FP2 channels...")
    # After standardizing, names are typically 'Fp1' and 'Fp2'
    target_channels = ['Fp1', 'Fp2']
    try:
        raw.pick(target_channels)
    except ValueError as e:
        print(f"Standard pick failed: {e}")
        # Fallback to case-insensitive matching
        matched_channels = [ch for ch in raw.ch_names if ch.upper() in ['FP1', 'FP2']]
        print(f"Fallback matched channels: {matched_channels}")
        raw.pick(matched_channels)
        
    print("Filtering data between 0.16 and 40 Hz...")
    raw.filter(l_freq=0.16, h_freq=40.0, fir_design='firwin')
    
    print("Resampling to 128 Hz...")
    raw.resample(128.0)
    
    print("\n--- Pipeline Complete ---")
    print("Final Data Info:")
    print(raw.info)
    
    return raw

if __name__ == "__main__":
    process_eeg_data()
