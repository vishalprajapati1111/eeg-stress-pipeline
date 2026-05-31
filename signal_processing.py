"""
EEG Stress-Detection Pipeline
Signal Processing Script

Imports EEG data from the data ingestion module, calculates a single 
differential signal (Fp1 - Fp2), slices the signal into 2-second windows
with a 0.5-second overlap, and extracts frequency-domain features (Alpha
and Beta band power, and Beta/Alpha ratio) using PSD for machine learning.
"""

import numpy as np
from scipy import signal
from data_ingestion import process_eeg_data

def generate_features(subject=1, runs=None):
    if runs is None:
        runs = [1, 2]
    print(f"--- Starting Signal Processing (Runs: {runs}) ---")
    
    # 1. Import filtered data from the ingestion script
    raw = process_eeg_data(subject=subject, runs=runs)
    
    # Extract data and sampling frequency
    data = raw.get_data()  # Shape: (n_channels, n_times)
    ch_names = raw.ch_names
    fs = int(raw.info['sfreq'])  # Should be 128 Hz
    
    print(f"\nImported data shape: {data.shape} at {fs} Hz")
    
    # 2. Calculate differential signal (FP1 - FP2)
    try:
        fp1_idx = next(i for i, ch in enumerate(ch_names) if ch.upper() == 'FP1')
        fp2_idx = next(i for i, ch in enumerate(ch_names) if ch.upper() == 'FP2')
    except StopIteration:
        raise ValueError(f"Could not find FP1 and/or FP2 in channel names: {ch_names}")

    diff_signal = data[fp1_idx, :] - data[fp2_idx, :]
    print(f"Calculated differential signal (Fp1 - Fp2) with {len(diff_signal)} samples.")
    
    # 3. Slice the signal into 2-second rolling windows with a 0.5-second overlap
    # This implies a stride/step size of 1.5 seconds (2.0 - 0.5)
    window_length_sec = 2.0
    overlap_sec = 0.5
    step_sec = window_length_sec - overlap_sec
    
    window_samples = int(window_length_sec * fs)  # 2.0 * 128 = 256 samples
    step_samples = int(step_sec * fs)             # 1.5 * 128 = 192 samples
    
    windows = []
    # Rolling window extraction
    for start_idx in range(0, len(diff_signal) - window_samples + 1, step_samples):
        window = diff_signal[start_idx : start_idx + window_samples]
        windows.append(window)
        
    # Convert to 2D numpy array: shape is (n_windows, n_samples_per_window)
    X_windows = np.array(windows)
    print(f"Generated {X_windows.shape[0]} windows of shape {X_windows.shape[1]} (2-second windows).")
    
    # 4. Calculate PSD and extract Alpha, Beta, and Beta/Alpha Ratio features
    print("Calculating PSD and extracting Alpha, Beta, and Beta/Alpha Ratio...")
    features = []
    
    for window in windows:
        # Calculate PSD using Welch's method
        freqs, psd = signal.welch(window, fs, nperseg=window_samples)
        
        # Alpha band (8 - 12 Hz)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_power = np.mean(psd[alpha_idx])
        
        # Beta band (12 - 30 Hz)
        beta_idx = np.logical_and(freqs >= 12, freqs <= 30)
        beta_power = np.mean(psd[beta_idx])
        
        # Beta/Alpha Ratio
        beta_alpha_ratio = beta_power / alpha_power if alpha_power != 0 else 0
        
        features.append([alpha_power, beta_power, beta_alpha_ratio])
        
    X_features = np.array(features)
    
    # 5. Output ready for ML classifier
    print(f"Final feature array shape for ML classifier (n_samples, n_features): {X_features.shape}")
    print("--- Signal Processing Complete ---")
    
    return X_features

if __name__ == "__main__":
    # Executing the pipeline directly
    features = generate_features()
