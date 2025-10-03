# ==============================================================================
# DATATHON 2025: LIFELINE - FINAL MODEL PIPELINE
#
# This script contains the full, reproducible pipeline for the Lifeline model.
# It performs the following key steps:
#   1. Loads the cleaned CTG dataset.
#   2. Splits data into training and testing sets to prevent data leakage.
#   3. Applies clinical feature engineering to both sets.
#   4. Balances the training data using Borderline-SMOTE.
#   5. Trains a LightGBM classifier.
#   6. Evaluates the model on the unseen test set.
#   7. Generates SHAP explanations for model interpretability.
# ==============================================================================

# --- Core Libraries ---
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# STEP 1: CLINICAL FEATURE ENGINEERING
#
# Rationale: We transform raw features into clinically meaningful metrics
# to help the model learn patterns that align with a doctor's expertise.
# This improves both performance and interpretability.
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

        # Feature 1: Variability Ratios (STV/LTV)
        # Justification: Ratios can capture the relationship between short-term
        # and long-term variability better than raw values alone.
        if 'ASTV' in self.df.columns and 'MSTV' in self.df.columns:
            self.df_engineered['STV_ratio'] = self.df['ASTV'] / (self.df['MSTV'] + 1e-6)
        if 'ALTV' in self.df.columns and 'MLTV' in self.df.columns:
            self.df_engineered['LTV_ratio'] = self.df['ALTV'] / (self.df['MLTV'] + 1e-6)

        # Feature 2: Deceleration Metrics
        # Justification: Aggregating and weighting decelerations provides a
        # single, powerful indicator of fetal stress.
        decel_cols = [col for col in ['DL', 'DS', 'DP', 'DR'] if col in self.df.columns]
        if decel_cols:
            self.df_engineered['total_decelerations'] = self.df[decel_cols].sum(axis=1)
            decel_weights = {'DL': 1, 'DS': 3, 'DP': 2, 'DR': 2}
            self.df_engineered['weighted_deceleration_score'] = sum(
                self.df[col] * decel_weights.get(col, 0) for col in decel_cols
            )
            if 'DS' in self.df.columns:
                self.df_engineered['has_severe_decelerations'] = (self.df['DS'] > 0).astype(int)
        
        # Feature 3: Deceleration-Acceleration Ratio
        # Justification: Captures the balance between reassuring (accelerations)
        # and non-reassuring (decelerations) events.
        if 'AC' in self.df.columns and 'total_decelerations' in self.df_engineered.columns:
            self.df_engineered['decel_accel_ratio'] = (
                self.df_engineered['total_decelerations'] / (self.df['AC'] + 1)
            )

        # Feature 4: Baseline Heart Rate Categories
        # Justification: Converts the continuous baseline into binary flags for
        # bradycardia (<110) and tachycardia (>160), simplifying the pattern for the model.
        if 'LB' in self.df.columns:
            self.df_engineered['LB_bradycardia'] = (self.df['LB'] < 110).astype(int)
            self.df_engineered['LB_tachycardia'] = (self.df['LB'] > 160).astype(int)

        # Feature 5: Composite Clinical Risk Score
        # Justification: A simple heuristic score that mimics how a clinician might
        # quickly assess risk by combining several critical indicators.
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
# STEP 2: MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================

def main():
    """Main function to run the full training and evaluation pipeline."""
    print("="*80)
    print("RUNNING LIFELINE MODEL PIPELINE: FEATURE-ENGINEERING + LIGHTGBM")
    print("="*80)

    CLEANED_DATA_PATH = './data/preprocessed/cleaned_data.csv'
    MODEL_OUTPUT_DIR = 'final_model_artifacts'
    
    # --- 1. Load Data ---
    # Assumes a preprocessing script has already run and saved the cleaned data.
    print(f"\n[1/6] Loading cleaned data from '{CLEANED_DATA_PATH}'...")
    df_clean = pd.read_csv(CLEANED_DATA_PATH)

    original_features = [
        'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB', 'ASTV',
        'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
        'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
    ]
    available_features = [f for f in original_features if f in df_clean.columns]
    X_raw = df_clean[available_features]
    y = df_clean['NSP']
    print("  ✓ Data loaded successfully.")

    # --- 2. Stratified Train-Test Split ---
    # Rationale: We split the data *before* engineering or balancing to prevent
    # data leakage and get a true measure of performance on unseen data.
    print("\n[2/6] Performing stratified 80/20 train-test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print("  ✓ Raw data split complete.")

    # --- 3. Apply Feature Engineering ---
    # Rationale: The same feature engineering logic is applied independently
    # to the training and testing sets.
    print("\n[3/6] Applying clinical feature engineering...")
    fe_train = CTGFeatureEngineer(X_train_raw)
    X_train = fe_train.engineer_medical_features()
    fe_test = CTGFeatureEngineer(X_test_raw)
    X_test = fe_test.engineer_medical_features()
    print("  ✓ Feature engineering complete for both sets.")

    # --- 4. Balance Training Data ---
    # Rationale: Pathological cases are rare. We use Borderline-SMOTE to
    # synthetically create more high-risk examples, forcing the model
    # to learn their patterns and reducing the risk of false negatives.
    print("\n[4/6] Balancing training data with Borderline-SMOTE...")
    smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("  ✓ Training data successfully balanced.")

    # --- 5. Train LightGBM Model ---
    # Rationale: LightGBM is chosen for its speed, efficiency, and high
    # performance on tabular data. 'class_weight=balanced' provides an
    # additional safeguard against class imbalance.
    print("\n[5/6] Training LightGBM classifier...")
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,  # Buffer for classes 1, 2, 3
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced'
    )
    # LightGBM requires 0-indexed labels for multiclass classification.
    lgb_model.fit(X_train_balanced, y_train_balanced - 1)
    print("  ✓ Model training complete.")

    # --- Save the model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'lifeline_lgbm_model.pkl')
    joblib.dump(lgb_model, model_path)
    print(f"  ✓ Model artifact saved to: {model_path}")

    # --- 6. Evaluate and Explain ---
    # Rationale: We evaluate on the original, *unseen* test set. The most
    # critical metric for clinical safety is Pathological Recall.
    print("\n[6/6] Evaluating model on unseen test data and generating explanations...")
    y_pred = lgb_model.predict(X_test) + 1  # Convert predictions back to 1-indexed

    print("\n" + "="*40)
    print("MODEL EVALUATION REPORT")
    print("="*40)
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    path_recall = recall_score(y_test, y_pred, average=None, labels=[1, 2, 3])[2]
    print(f"Pathological Recall (Clinical Safety Focus): {path_recall:.4f}")
    print(f"Macro F1-Score (Balanced Performance): {f1_score(y_test, y_pred, average='macro'):.4f}\n")
    print("Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Generate SHAP plot for interpretability ---
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)
    print("\n  Displaying SHAP summary plot to show global feature importance...")
    shap.summary_plot(shap_values, X_test, class_names=['Normal', 'Suspect', 'Pathological'], show=True)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
