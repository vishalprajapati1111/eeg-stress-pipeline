import numpy as np
import pandas as pd
from signal_processing import generate_features
import os

def build_diversified_dataset(subjects=range(1, 11)): # Using first 10 subjects as a diverse sample
    print(f"Building pure diversified dataset for {len(subjects)} subjects...")
    all_X = []
    all_y = []
    all_subjects = []
    
    for subj in subjects:
        print(f"-> Processing Subject {subj}...")
        try:
            # Baseline (Run 1)
            X_base = generate_features(subject=subj, runs=[1])
            y_base = np.zeros(X_base.shape[0])
            
            # Stressed (Run 2)
            X_stress = generate_features(subject=subj, runs=[2])
            y_stress = np.ones(X_stress.shape[0])
            
            all_X.append(X_base)
            all_X.append(X_stress)
            all_y.append(y_base)
            all_y.append(y_stress)
            
            all_subjects.extend([subj] * (X_base.shape[0] + X_stress.shape[0]))
        except Exception as e:
            print(f"   [!] Error processing subject {subj}: {e}")
            
    # Merge all subjects
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Create a clean, pure DataFrame
    df = pd.DataFrame(X, columns=['Alpha_Power', 'Beta_Power', 'Beta_Alpha_Ratio'])
    df['Label'] = y.astype(int)
    df['Subject_ID'] = all_subjects
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "pure_diversified_eeg_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n=========================================")
    print(f"Success! Dataset exported to: {csv_path}")
    print(f"Total Diversified Samples: {len(df)}")
    print(f"Class Balance: {len(df[df['Label']==0])} Baseline, {len(df[df['Label']==1])} Stressed")
    print(f"=========================================")

if __name__ == "__main__":
    build_diversified_dataset()
