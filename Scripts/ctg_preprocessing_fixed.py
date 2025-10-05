# ==============================================================================
# DATATHON 2025: LIFELINE - DATA PREPROCESSING PIPELINE
#
# Author: Charles Josse
#
# Purpose:
# This script performs the initial, crucial step of cleaning and preparing the
# raw CTG dataset. It handles missing values, removes duplicates and outliers,
# and saves a set of clean, versioned data files ready for model training.
#
# Rationale:
# A separate, reproducible preprocessing script ensures that all subsequent
# modeling and analysis work is built upon a consistent and high-quality
# data foundation. "Garbage in, garbage out" is a critical risk, and this
# script is the primary safeguard against it.
# ==============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
warnings.filterwarnings('ignore')


class CTGPreprocessor:
    """
    A comprehensive preprocessor for the CTG dataset, designed to ensure
    data quality and prepare the data for machine learning tasks.
    """

    def __init__(self, file_path, output_dir):
        """Initializes the preprocessor with input and output paths."""
        self.file_path = file_path
        self.output_dir = output_dir
        self.required_features = [
            'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB',
            'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax',
            'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
        ]
        self.target_column = 'NSP'
        self.df_raw = None
        self.df_clean = None

    def load_data(self, sheet_name='Data', header=1):
        """Step 1: Load the raw dataset from the specified Excel sheet."""
        print("\n[1/5] Loading raw data...")
        try:
            self.df_raw = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header)
            print(f"  ✓ Successfully loaded data from sheet '{sheet_name}'. Shape: {self.df_raw.shape}")
        except Exception as e:
            print(f"  ✗ ERROR: Could not load data. {e}")
            raise

    def validate_and_clean(self):
        """Step 2: Validate features, remove duplicates, handle missing values and outliers."""
        print("\n[2/5] Validating and cleaning data...")
        
        # --- Feature Validation ---
        available_features = set(self.df_raw.columns)
        missing_features = set(self.required_features) - available_features
        if missing_features:
            print(f"  ✗ ERROR: Missing required features: {missing_features}")
            raise ValueError("Required features are missing from the dataset.")
        print("  ✓ All required features are present.")
        
        # --- Data Cleaning ---
        initial_rows = len(self.df_raw)
        
        # Drop duplicates
        df = self.df_raw.drop_duplicates()
        
        # Drop rows with missing target or feature values
        df = df.dropna(subset=self.required_features + [self.target_column])
        
        # Remove physiological outliers
        df = df[(df['LB'] >= 50) & (df['LB'] <= 200)] # Plausible baseline heart rate
        df = df[df['AC'] >= 0] # Accelerations cannot be negative
        
        # Convert target to integer
        df[self.target_column] = df[self.target_column].astype(int)
        
        self.df_clean = df
        final_rows = len(self.df_clean)
        print(f"  ✓ Data cleaning complete. Rows changed from {initial_rows} to {final_rows}.")

    def prepare_final_dataset(self):
        """Step 3: Select final features and target for saving."""
        print("\n[3/5] Preparing final feature matrix and target vector...")
        self.X = self.df_clean[self.required_features]
        self.y = self.df_clean[self.target_column]
        print(f"  ✓ Final feature matrix shape: {self.X.shape}")
        print(f"  ✓ Final target vector shape: {self.y.shape}")

    def split_and_scale(self, test_size=0.2, random_state=42):
        """Step 4: Perform stratified train-test split and scale the features."""
        print("\n[4/5] Splitting data and scaling features...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features using RobustScaler (less sensitive to outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.split_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
            'scaler': scaler, 'feature_names': self.X.columns.tolist()
        }
        print("  ✓ Data split and scaled successfully.")

    def save_artifacts(self):
        """Step 5: Save all processed data and artifacts for reproducibility."""
        print("\n[5/5] Saving preprocessed artifacts...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save cleaned dataframe
        self.df_clean.to_csv(os.path.join(self.output_dir, 'cleaned_data.csv'), index=False)
        
        # Save split and scaled data arrays
        for name, data in self.split_data.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data.to_csv(os.path.join(self.output_dir, f'{name}.csv'), index=False)
            elif isinstance(data, np.ndarray):
                np.save(os.path.join(self.output_dir, f'{name}.npy'), data)
            else: # For scaler and feature_names list
                with open(os.path.join(self.output_dir, f'{name}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
        
        print(f"  ✓ All artifacts successfully saved to: '{self.output_dir}'")

    def run_pipeline(self):
        """Executes the full preprocessing pipeline in sequence."""
        print("="*80)
        print("STARTING CTG DATA PREPROCESSING PIPELINE")
        print("="*80)
        self.load_data()
        self.validate_and_clean()
        self.prepare_final_dataset()
        self.split_and_scale()
        self.save_artifacts()
        print("\n" + "="*80)
        print("✓ PREPROCESSING PIPELINE COMPLETE!")
        print("="*80)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    # Standardized paths for a GitHub repository structure.
    RAW_DATA_PATH = './data/CTG.xls'
    PREPROCESSED_OUTPUT_DIR = './data/preprocessed/'

    # --- Run Pipeline ---
    preprocessor = CTGPreprocessor(
        file_path=RAW_DATA_PATH,
        output_dir=PREPROCESSED_OUTPUT_DIR
    )
    preprocessor.run_pipeline()

