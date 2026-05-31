"""
EEG Stress-Detection Pipeline
Model Training Script

Imports the PCA feature arrays for both baseline and stressed states,
assigns binary labels (0 for Baseline, 1 for Stressed), merges the datasets,
splits them into training and testing sets, trains a Support Vector Machine
(SVM) with a Linear Kernel, and evaluates the model's Accuracy, Precision, 
and Recall.
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from signal_processing import generate_features

def train_and_evaluate():
    print("=== Extracting Features for Baseline State ===")
    # Run 1 is used as the baseline state
    X_baseline = generate_features(subject=1, runs=[1])
    
    print("\n=== Extracting Features for Stressed State ===")
    # Run 2 is used as the stressed state
    X_stressed = generate_features(subject=1, runs=[2])
    
    # 1. Assign binary labels: 0 for Baseline, 1 for Stressed
    y_baseline = np.zeros(X_baseline.shape[0])
    y_stressed = np.ones(X_stressed.shape[0])
    
    # 2. Merge the datasets
    X = np.vstack((X_baseline, X_stressed))
    y = np.concatenate((y_baseline, y_stressed))
    
    print(f"\nTotal Merged Dataset Shape: X={X.shape}, y={y.shape}")
    
    # 3. Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set: {X_test.shape[0]} samples")
    
    # 4. Scale the features using StandardScaler
    print("\nApplying StandardScaler to the features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Renamed to X_test_scaled so the sanity check can find it!
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train a Support Vector Machine (SVM) with a Linear Kernel
    print("\nTraining Linear SVM Model...")
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train, y_train)
    
    # 6. Evaluate the model on the test set
    y_pred = svm_clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print("==========================")
    
    export_cpp_header(svm_clf, scaler, "eeg_weights.h")
    
    # ==========================================
    # LIVE PREDICTION SANITY CHECK
    # ==========================================
    print("\n=== Live Prediction Sanity Check ===")
    # Grab 5 random 2-second windows from the test set
    sample_indices = random.sample(range(len(y_test)), 5)

    for i in sample_indices:
        # Get the exact scaled features (Alpha, Beta, Ratio)
        features = X_test_scaled[i]
        
        # Reshape for a single prediction and ask the SVM to guess
        prediction = svm_clf.predict(features.reshape(1, -1))[0]
        actual = y_test[i]
        
        # Translate binary 0 and 1 back to English
        pred_text = "STRESSED" if prediction == 1 else "RELAXED"
        actual_text = "STRESSED" if actual == 1 else "RELAXED"
        match = "✅ MATCH" if prediction == actual else "❌ FAILED"
        
        print(f"Test Window [{i}]:")
        print(f"  Scaled Inputs (Alpha, Beta, Ratio): [{features[0]:.2f}, {features[1]:.2f}, {features[2]:.2f}]")
        print(f"  AI Predicted : {pred_text}")
        print(f"  Actual State : {actual_text} {match}\n")

    return svm_clf


def export_cpp_header(svm_clf, scaler, filename):
    import os
    
    # Extract weights and intercept
    weights = svm_clf.coef_[0]
    intercept = svm_clf.intercept_[0]
    
    # Extract scaler mean and scale
    means = scaler.mean_
    scales = scaler.scale_
    
    header_content = f"""#ifndef EEG_WEIGHTS_H
#define EEG_WEIGHTS_H

// StandardScaler Parameters
const float scaler_means[{len(means)}] = {{{', '.join(map(str, means))}}};
const float scaler_scales[{len(scales)}] = {{{', '.join(map(str, scales))}}};

// Linear SVM Parameters
const float svm_weights[{len(weights)}] = {{{', '.join(map(str, weights))}}};
const float svm_intercept = {intercept};

#endif // EEG_WEIGHTS_H
"""
    # Write to local file
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w') as f:
        f.write(header_content)
    
    print(f"\nSuccessfully exported weights to {filepath} for ESP32 deployment.")

if __name__ == "__main__":
    train_and_evaluate()