# ==============================================================================
# DATATHON 2025: LIFELINE - FEATURE-ENGINEERED RANDOM FOREST MODEL
#
# Author: Charles Josse
#
# Purpose:
# This script trains and evaluates a Random Forest classifier on a dataset
# enriched with clinically-inspired, engineered features.
#
# The pipeline is as follows:
#   1.  Load the pre-processed and feature-engineered dataset.
#   2.  Perform a stratified train-test split.
#   3.  Apply BorderlineSMOTE to the training set to handle class imbalance.
#   4.  Scale the features using StandardScaler.
#   5.  Train a Random Forest model on the balanced, scaled training data.
#   6.  Evaluate the model on the held-out test set and report performance.
#   7.  Save the trained model, scaler, and a list of used features.
#
# ==============================================================================

import os
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, recall_score
)

# --- Imbalanced Learning ---
from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. MAIN TRAINING AND EVALUATION FUNCTION
# ==============================================================================

def main(args):
    """Main function to run the training and evaluation pipeline."""
    
    print("--- Starting Feature-Engineered Random Forest Pipeline ---")
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/4] Loading feature-engineered data from '{args.data}'...")
    df = pd.read_csv(args.data)
    
    # Define features (X) and target (y)
    y = df['NSP'].values
    X = df.drop('NSP', axis=1)
    feature_names = X.columns.tolist()
    
    print(f"  ✓ Data loaded. Shape: {X.shape}")

    # 2. Train-Test Split
    print("\n[2/4] Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y
    )
    print(f"  ✓ Train size: {len(y_train)}, Test size: {len(y_test)}")

    # 3. SMOTE and Scaling
    print("\n[3/4] Applying SMOTE to training data and scaling features...")
    
    # Balance the training set
    smote = BorderlineSMOTE(random_state=args.seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    print("  ✓ SMOTE and scaling complete.")

    # 4. Train and Evaluate Model
    print("\n[4/4] Training Random Forest and evaluating...")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=args.seed,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_resampled)
    
    y_pred = model.predict(X_test_scaled)
    
    # --- Report Metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    rec3 = recall_score(y_test, y_pred, labels=[3], average='macro', zero_division=0)
    
    print("\n=== Feature-Engineered RF Final Results ===")
    print(f"  Accuracy:            {acc:.4f}")
    print(f"  Macro F1-Score:      {f1m:.4f}")
    print(f"  Pathological Recall: {rec3:.4f}")
    print("\n  Confusion Matrix (Labels: 1, 2, 3):\n", confusion_matrix(y_test, y_pred, labels=[1, 2, 3]))
    print("\n  Classification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

    # --- Save Artifacts ---
    print("\n--- Saving Artifacts ---")
    
    with open(os.path.join(args.outdir, 'fe_rf_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    with open(os.path.join(args.outdir, 'fe_rf_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(os.path.join(args.outdir, 'fe_rf_features.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
        
    print(f"  ✓ Model, scaler, and feature list saved to '{args.outdir}'")
    print("--- Pipeline Complete ---")

# ==============================================================================
# 2. COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a feature-engineered Random Forest model.")
    
    parser.add_argument('--data', type=str, required=True, help="Path to the feature-engineered dataset CSV file.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save the output model and artifacts.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    main(args)
