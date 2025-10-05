# ==============================================================================
# DATATHON 2025: LIFELINE - MODEL INFERENCE SCRIPT (FIXED)
#
# This script loads the trained LightGBM model and performs inference on new data.
# It supports:
#   1. Single patient prediction
#   2. Batch prediction from CSV files
#   3. SHAP explanations for individual predictions
#   4. Clinical risk assessment with confidence scores
# ==============================================================================


import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from typing import Union, Dict, List
warnings.filterwarnings('ignore')


# ==============================================================================
# CLINICAL FEATURE ENGINEERING (Must match training pipeline exactly)
# ==============================================================================


class CTGFeatureEngineer:
    """
    Applies domain-specific feature engineering to CTG data.
    This MUST be identical to the training pipeline to ensure consistency.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.df_engineered = None


    def engineer_medical_features(self):
        """Creates a set of new features based on clinical guidelines."""
        self.df_engineered = self.df.copy()


        # Feature 1: Variability Ratios (STV/LTV)
        if 'ASTV' in self.df.columns and 'MSTV' in self.df.columns:
            self.df_engineered['STV_ratio'] = self.df['ASTV'] / (self.df['MSTV'] + 1e-6)
        if 'ALTV' in self.df.columns and 'MLTV' in self.df.columns:
            self.df_engineered['LTV_ratio'] = self.df['ALTV'] / (self.df['MLTV'] + 1e-6)


        # Feature 2: Deceleration Metrics
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
        if 'AC' in self.df.columns and 'total_decelerations' in self.df_engineered.columns:
            self.df_engineered['decel_accel_ratio'] = (
                self.df_engineered['total_decelerations'] / (self.df['AC'] + 1)
            )


        # Feature 4: Baseline Heart Rate Categories
        if 'LB' in self.df.columns:
            self.df_engineered['LB_bradycardia'] = (self.df['LB'] < 110).astype(int)
            self.df_engineered['LB_tachycardia'] = (self.df['LB'] > 160).astype(int)


        # Feature 5: Composite Clinical Risk Score
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
# LIFELINE INFERENCE CLASS
# ==============================================================================


class LifelineInference:
    """
    Handles model loading and inference for the Lifeline CTG classifier.
    """

    # Required features for model input (before engineering)
    REQUIRED_FEATURES = [
        'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB', 'ASTV',
        'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
        'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
    ]

    CLASS_LABELS = {
        1: 'Normal',
        2: 'Suspect',
        3: 'Pathological'
    }


    def __init__(self, model_path: str):
        """
        Initialize the inference class by loading the trained model.

        Args:
            model_path: Path to the saved .pkl model file
        """
        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        self.explainer = None
        print("✓ Model loaded successfully.")


    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates that input data contains all required features.

        Args:
            df: Input dataframe

        Returns:
            Validated dataframe with only required features
        """
        missing_features = [f for f in self.REQUIRED_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return df[self.REQUIRED_FEATURES]


    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering to raw input data.

        Args:
            df: Raw input dataframe

        Returns:
            Engineered feature dataframe
        """
        fe = CTGFeatureEngineer(df)
        return fe.engineer_medical_features()


    def predict_single(self, patient_data: Union[Dict, pd.DataFrame], 
                      explain: bool = False) -> Dict:
        """
        Performs inference on a single patient record.

        Args:
            patient_data: Dictionary or single-row DataFrame with patient features
            explain: Whether to generate SHAP explanations

        Returns:
            Dictionary containing prediction, confidence, and optional explanations
        """
        # Convert dict to DataFrame if needed
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()

        # Validate and preprocess
        df_validated = self._validate_input(df)
        df_processed = self._preprocess(df_validated)

        # Get prediction (model outputs 0-indexed, convert to 1-indexed)
        pred_proba = self.model.predict_proba(df_processed)[0]
        pred_class_idx = np.argmax(pred_proba)
        pred_class = pred_class_idx + 1
        confidence = pred_proba[pred_class_idx]

        result = {
            'prediction': pred_class,
            'prediction_label': self.CLASS_LABELS[pred_class],
            'confidence': float(confidence),
            'probabilities': {
                self.CLASS_LABELS[i+1]: float(pred_proba[i]) 
                for i in range(len(pred_proba))
            },
            'risk_level': self._assess_risk(pred_class, confidence)
        }

        # Add SHAP explanation if requested
        if explain:
            try:
                if self.explainer is None:
                    self.explainer = shap.TreeExplainer(self.model)

                shap_values = self.explainer.shap_values(df_processed)

                # For multiclass, shap_values is a list of arrays [class0, class1, class2]
                # Each array has shape (n_samples, n_features)
                if isinstance(shap_values, list):
                    # Get SHAP values for the predicted class
                    shap_for_pred = shap_values[pred_class_idx][0]
                else:
                    # Binary classification case
                    shap_for_pred = shap_values[0]

                # Ensure all arrays have the same length
                feature_names = list(df_processed.columns)
                feature_values = df_processed.iloc[0].values

                if len(shap_for_pred) == len(feature_names) == len(feature_values):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_for_pred,
                        'feature_value': feature_values
                    }).sort_values('shap_value', key=abs, ascending=False).head(5)

                    result['explanation'] = {
                        'top_features': feature_importance.to_dict('records'),
                        'shap_values_available': True
                    }
                else:
                    result['explanation'] = {
                        'error': 'Feature length mismatch in SHAP calculation',
                        'shap_values_available': False
                    }

            except Exception as e:
                result['explanation'] = {
                    'error': f'SHAP calculation failed: {str(e)}',
                    'shap_values_available': False
                }

        return result


    def predict_batch(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Performs batch inference on a CSV file.

        Args:
            input_path: Path to input CSV file
            output_path: Optional path to save results (default: adds '_predictions' suffix)

        Returns:
            DataFrame with original data plus predictions
        """
        print(f"\nLoading data from: {input_path}")
        df = pd.read_csv(input_path)

        # Validate and preprocess
        df_validated = self._validate_input(df)
        df_processed = self._preprocess(df_validated)

        print("Running batch inference...")
        # Get predictions
        pred_proba = self.model.predict_proba(df_processed)
        pred_classes = np.argmax(pred_proba, axis=1) + 1
        pred_confidence = np.max(pred_proba, axis=1)

        # Add results to original dataframe
        results = df.copy()
        results['predicted_class'] = pred_classes
        results['predicted_label'] = results['predicted_class'].map(self.CLASS_LABELS)
        results['confidence'] = pred_confidence

        for i, label in self.CLASS_LABELS.items():
            results[f'prob_{label}'] = pred_proba[:, i-1]

        results['risk_level'] = results.apply(
            lambda row: self._assess_risk(row['predicted_class'], row['confidence']), 
            axis=1
        )

        # Save results
        if output_path is None:
            output_path = input_path.replace('.csv', '_predictions.csv')

        results.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(results['predicted_label'].value_counts())
        print(f"\nAverage confidence: {results['confidence'].mean():.4f}")
        print(f"High-risk (Pathological) cases: {(results['predicted_class'] == 3).sum()}")

        return results


    def _assess_risk(self, pred_class: int, confidence: float) -> str:
        """
        Provides a clinical risk assessment based on prediction and confidence.

        Args:
            pred_class: Predicted class (1=Normal, 2=Suspect, 3=Pathological)
            confidence: Prediction confidence score

        Returns:
            Risk level description
        """
        if pred_class == 3:
            if confidence > 0.8:
                return "HIGH RISK - Immediate clinical review recommended"
            else:
                return "ELEVATED RISK - Clinical review recommended"
        elif pred_class == 2:
            if confidence > 0.7:
                return "MODERATE RISK - Close monitoring advised"
            else:
                return "UNCERTAIN - Consider additional assessment"
        else:
            if confidence > 0.9:
                return "LOW RISK - Normal monitoring"
            else:
                return "LOW-MODERATE RISK - Continue monitoring"



# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================


def main():
    """
    Demonstrates usage of the inference script.
    """
    # --- Configuration ---
    MODEL_PATH = '/Users/ricardjossemeyer/Documents/python/Projects/Datathon_24/Deliverables/final_model_artifacts/lgbm_hybrid_model_corrected.pkl'

    # Initialize inference class
    lifeline = LifelineInference(MODEL_PATH)


    # --- Example 1: Single Patient Prediction with Explanation ---
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE PATIENT PREDICTION")
    print("="*80)

    # Sample patient data (replace with actual values)
    sample_patient = {
        'b': 120.0, 'e': 0.0, 'AC': 0.0, 'FM': 0.0, 'UC': 0.0,
        'DL': 0.0, 'DS': 0.0, 'DP': 0.0, 'DR': 0.0, 'LB': 120.0,
        'ASTV': 73.0, 'MSTV': 0.5, 'ALTV': 43.0, 'MLTV': 2.4,
        'Width': 64.0, 'Min': 62.0, 'Max': 126.0, 'Nmax': 2.0,
        'Nzeros': 0.0, 'Mode': 120.0, 'Mean': 137.0, 'Median': 121.0,
        'Variance': 73.0, 'Tendency': 1.0
    }

    result = lifeline.predict_single(sample_patient, explain=True)

    print(f"\nPrediction: {result['prediction_label']} (Class {result['prediction']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nClass Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f}")

    if 'explanation' in result and result['explanation'].get('shap_values_available'):
        print(f"\nTop Contributing Features:")
        for feat in result['explanation']['top_features']:
            print(f"  {feat['feature']}: {feat['shap_value']:.4f} (value={feat['feature_value']:.2f})")


    # --- Example 2: Batch Prediction ---
    # Uncomment below to run batch prediction on a CSV file
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*80)

    INPUT_CSV = './data/new_patients.csv'
    OUTPUT_CSV = './data/new_patients_predictions.csv'

    results_df = lifeline.predict_batch(INPUT_CSV, OUTPUT_CSV)
    """

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
