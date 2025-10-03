# ==============================================================================
# DATATHON 2025: LIFELINE - ENGINEERED TEST SET GENERATOR
#
#
# Purpose:
# This script is a utility to generate the final, engineered test set used for
# model evaluation and SHAP analysis.
#
# Rationale:
# By isolating this process, we ensure that the exact same test data that the
# model was evaluated on can be reliably reproduced for post-hoc analysis,
# such as generating SHAP plots, without re-running the entire training pipeline.
# It guarantees consistency and reproducibility.
# ==============================================================================


import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# STEP 1: CONSISTENT FEATURE ENGINEERING CLASS
#
# Rationale: We re-use the exact same class from the training script to
# guarantee that the feature engineering logic is applied identically.
# ==============================================================================


class CTGFeatureEngineer:
    """
    Applies domain-specific feature engineering to CTG data.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.df_engineered = None

    def engineer_medical_features(self):
        """Creates a set of new features based on clinical guidelines."""
        self.df_engineered = self.df.copy()

        # Variability Features
        if 'ASTV' in self.df.columns and 'MSTV' in self.df.columns:
            self.df_engineered['STV_ratio'] = self.df['ASTV'] / (self.df['MSTV'] + 1e-6)
        if 'ALTV' in self.df.columns and 'MLTV' in self.df.columns:
            self.df_engineered['LTV_ratio'] = self.df['ALTV'] / (self.df['MLTV'] + 1e-6)

        # Deceleration Features
        decel_cols = [col for col in ['DL', 'DS', 'DP', 'DR'] if col in self.df.columns]
        if decel_cols:
            self.df_engineered['total_decelerations'] = self.df[decel_cols].sum(axis=1)
            decel_weights = {'DL': 1, 'DS': 3, 'DP': 2, 'DR': 2}
            self.df_engineered['weighted_deceleration_score'] = sum(
                self.df[col] * decel_weights.get(col, 0) for col in decel_cols
            )
            if 'DS' in self.df.columns:
                self.df_engineered['has_severe_decelerations'] = (self.df['DS'] > 0).astype(int)

        # Deceleration-Acceleration Ratio
        if 'AC' in self.df.columns and 'total_decelerations' in self.df_engineered.columns:
            self.df_engineered['decel_accel_ratio'] = (
                self.df_engineered['total_decelerations'] / (self.df['AC'] + 1)
            )

        # Baseline Heart Rate Categories
        if 'LB' in self.df.columns:
            self.df_engineered['LB_bradycardia'] = (self.df['LB'] < 110).astype(int)
            self.df_engineered['LB_tachycardia'] = (self.df['LB'] > 160).astype(int)

        # Composite Clinical Risk Score
        risk_score = 0
        if 'ASTV' in self.df.columns: risk_score += (self.df['ASTV'] > 50) * 2
        if 'DS' in self.df.columns: risk_score += (self.df['DS'] > 0) * 3
        if 'LB' in self.df.columns:
            risk_score += (self.df['LB'] < 110) * 2
            risk_score += (self.df['LB'] > 160) * 2
        if 'AC' in self.df.columns: risk_score += (self.df['AC'] == 0) * 1
        self.df_engineered['clinical_risk_score'] = risk_score
        
        return self.df_engineered


# ==============================================================================
# STEP 2: MAIN SCRIPT EXECUTION
# ==============================================================================


def generate_test_set():
    """
    Loads data, reproduces the test split, applies feature engineering,
    and saves the final engineered test set to a file.
    """
    CLEANED_DATA_PATH = './data/preprocessed/cleaned_data.csv'
    OUTPUT_PATH = './data/preprocessed/X_test_engineered.csv'

    # --- 1. Load Data ---
    print(f"Loading cleaned data from '{CLEANED_DATA_PATH}'...")
    df_clean = pd.read_csv(CLEANED_DATA_PATH)

    # --- 2. Isolate Original Features and Target ---
    original_features = [
        f for f in [
            'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB', 'ASTV',
            'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
            'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
        ] if f in df_clean.columns
    ]
    X_orig = df_clean[original_features]
    y_orig = df_clean['NSP']
    print("  ✓ Original features and target isolated.")

    # --- 3. Reproduce the Exact Test Set ---
    # Rationale: Using the same random_state and stratify parameters ensures
    # we get the identical test set that the model was evaluated against.
    print("Reproducing the exact 80/20 train-test split...")
    _, X_test_orig, _, _ = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
    )
    print("  ✓ Original test set successfully isolated.")

    # --- 4. Apply Feature Engineering ---
    print("Applying clinical feature engineering to the test set...")
    engineer = CTGFeatureEngineer(X_test_orig)
    X_test_engineered = engineer.engineer_medical_features()
    print("  ✓ Feature engineering complete.")

    # --- 5. Save the Final Engineered Test Set ---
    X_test_engineered.to_csv(OUTPUT_PATH, index=False)
    print("\n" + "="*50)
    print("SUCCESSFULLY CREATED ENGINEERED TEST SET")
    print(f"  - Shape: {X_test_engineered.shape}")
    print(f"  - Saved to: {OUTPUT_PATH}")
    print("="*50)


if __name__ == "__main__":
    generate_test_set()

