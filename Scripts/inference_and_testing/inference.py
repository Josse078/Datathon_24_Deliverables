#!/usr/bin/env python3
"""
===============================================================================
LIFELINE CTG CLASSIFIER - INFERENCE MODULE
===============================================================================

Datathon 2025 - Team TM-37
Authors: Ricard Josse Meyer, Charles Lukas Chairos Yo, Poon Wei Lok, [Your Name]

This module provides inference capabilities for the Lifeline CTG (Cardiotocography)
classifier, which predicts fetal health status from CTG measurements.

Classes:
    CTGFeatureEngineer: Applies domain-specific feature engineering
    LifelinePredictor: Handles model loading and inference

Usage:
    from inference import LifelinePredictor

    predictor = LifelinePredictor('models/lifeline_model.pkl')
    result = predictor.predict(patient_data)
    print(f"Prediction: {result['label']} ({result['confidence']:.1%})")

Requirements:
    - numpy>=1.21.0
    - pandas>=1.3.0
    - scikit-learn>=0.24.0
    - lightgbm>=3.3.0
    - joblib>=1.0.0

===============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Dict, List, Optional


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

class CTGFeatureEngineer:
    """
    Applies clinical domain knowledge to engineer features from CTG data.

    This class transforms raw CTG measurements into clinically meaningful
    features that improve model performance and interpretability.

    Features created:
    - Variability ratios (STV/LTV relationships)
    - Weighted deceleration scores
    - Heart rate categorizations (bradycardia, tachycardia)
    - Composite clinical risk score

    Attributes:
        df (pd.DataFrame): Input CTG data
        df_engineered (pd.DataFrame): Data with engineered features
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer with CTG data.

        Args:
            df: DataFrame containing raw CTG features
        """
        self.df = df.copy()
        self.df_engineered = None

    def engineer_features(self) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Returns:
            DataFrame with original and engineered features
        """
        self.df_engineered = self.df.copy()

        # Variability ratios
        if 'ASTV' in self.df.columns and 'MSTV' in self.df.columns:
            self.df_engineered['STV_ratio'] = (
                self.df['ASTV'] / (self.df['MSTV'] + 1e-6)
            )

        if 'ALTV' in self.df.columns and 'MLTV' in self.df.columns:
            self.df_engineered['LTV_ratio'] = (
                self.df['ALTV'] / (self.df['MLTV'] + 1e-6)
            )

        # Deceleration metrics
        decel_cols = [col for col in ['DL', 'DS', 'DP', 'DR'] 
                      if col in self.df.columns]

        if decel_cols:
            self.df_engineered['total_decelerations'] = (
                self.df[decel_cols].sum(axis=1)
            )

            # Weighted score (severe decelerations weighted more)
            decel_weights = {'DL': 1, 'DS': 3, 'DP': 2, 'DR': 2}
            self.df_engineered['weighted_deceleration_score'] = sum(
                self.df[col] * decel_weights.get(col, 0) 
                for col in decel_cols
            )

            if 'DS' in self.df.columns:
                self.df_engineered['has_severe_decelerations'] = (
                    (self.df['DS'] > 0).astype(int)
                )

        # Deceleration-acceleration ratio
        if ('AC' in self.df.columns and 
            'total_decelerations' in self.df_engineered.columns):
            self.df_engineered['decel_accel_ratio'] = (
                self.df_engineered['total_decelerations'] / (self.df['AC'] + 1)
            )

        # Baseline heart rate categories
        if 'LB' in self.df.columns:
            self.df_engineered['LB_bradycardia'] = (
                (self.df['LB'] < 110).astype(int)
            )
            self.df_engineered['LB_tachycardia'] = (
                (self.df['LB'] > 160).astype(int)
            )

        # Composite clinical risk score
        risk_score = 0
        if 'ASTV' in self.df.columns:
            risk_score += (self.df['ASTV'] > 50) * 2
        if 'DS' in self.df.columns:
            risk_score += (self.df['DS'] > 0) * 3
        if 'LB' in self.df.columns:
            risk_score += (self.df['LB'] < 110) * 2
            risk_score += (self.df['LB'] > 160) * 2
        if 'AC' in self.df.columns:
            risk_score += (self.df['AC'] == 0) * 1

        self.df_engineered['clinical_risk_score'] = risk_score

        return self.df_engineered


# ==============================================================================
# PREDICTOR CLASS
# ==============================================================================

class LifelinePredictor:
    """
    Main predictor class for the Lifeline CTG classifier.

    This class handles model loading, input validation, feature engineering,
    and prediction generation for CTG data.

    Attributes:
        model: Trained LightGBM model
        model_path: Path to saved model file
        class_labels: Mapping of class indices to names
        required_features: List of required input features

    Example:
        >>> predictor = LifelinePredictor('models/lifeline_model.pkl')
        >>> result = predictor.predict({'LB': 135, 'ASTV': 45, ...})
        >>> print(f"{result['label']}: {result['confidence']:.1%}")
        Normal: 95.3%
    """

    # Required features (must be present in input data)
    REQUIRED_FEATURES = [
        'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB',
        'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
        'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
    ]

    CLASS_LABELS = {
        1: 'Normal',
        2: 'Suspect',
        3: 'Pathological'
    }

    def __init__(self, model_path: Union[str, Path], verbose: bool = True):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model (.pkl file)
            verbose: Whether to print loading messages

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.model_path = Path(model_path)
        self.verbose = verbose

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.verbose:
            print(f"Loading model from: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
            if self.verbose:
                print("✓ Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate that input contains all required features.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If required features are missing
        """
        missing = [f for f in self.REQUIRED_FEATURES if f not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required features: {', '.join(missing)}"
            )

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to input data.

        Args:
            data: Raw input DataFrame

        Returns:
            DataFrame with engineered features
        """
        engineer = CTGFeatureEngineer(data)
        return engineer.engineer_features()

    def predict(
        self, 
        data: Union[Dict, pd.DataFrame],
        return_probabilities: bool = True
    ) -> Dict:
        """
        Generate prediction for single patient or batch.

        Args:
            data: Patient data as dict or DataFrame
            return_probabilities: Include class probabilities in output

        Returns:
            Dictionary containing:
                - prediction: Predicted class (1, 2, or 3)
                - label: Class name (Normal, Suspect, Pathological)
                - confidence: Prediction confidence (0-1)
                - probabilities: Dict of class probabilities (if requested)
                - risk_assessment: Clinical risk description

        Example:
            >>> result = predictor.predict(patient_data)
            >>> result
            {
                'prediction': 1,
                'label': 'Normal',
                'confidence': 0.953,
                'probabilities': {
                    'Normal': 0.953,
                    'Suspect': 0.042,
                    'Pathological': 0.005
                },
                'risk_assessment': 'LOW RISK - Normal monitoring'
            }
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
            single_sample = True
        else:
            df = data.copy()
            single_sample = len(df) == 1

        # Validate and preprocess
        self._validate_input(df)
        df_processed = self._preprocess(df)

        # Generate predictions
        pred_proba = self.model.predict_proba(df_processed)
        pred_classes = np.argmax(pred_proba, axis=1) + 1  # Convert to 1-indexed
        confidences = np.max(pred_proba, axis=1)

        # Format results
        if single_sample:
            result = {
                'prediction': int(pred_classes[0]),
                'label': self.CLASS_LABELS[pred_classes[0]],
                'confidence': float(confidences[0])
            }

            if return_probabilities:
                result['probabilities'] = {
                    self.CLASS_LABELS[i+1]: float(pred_proba[0, i])
                    for i in range(pred_proba.shape[1])
                }

            result['risk_assessment'] = self._assess_risk(
                pred_classes[0], confidences[0]
            )

            return result
        else:
            # Batch prediction - return DataFrame
            results = df.copy()
            results['prediction'] = pred_classes
            results['label'] = [self.CLASS_LABELS[c] for c in pred_classes]
            results['confidence'] = confidences

            if return_probabilities:
                for i, class_name in self.CLASS_LABELS.items():
                    results[f'prob_{class_name}'] = pred_proba[:, i-1]

            results['risk_assessment'] = [
                self._assess_risk(c, conf) 
                for c, conf in zip(pred_classes, confidences)
            ]

            return results

    def predict_batch(
        self,
        input_csv: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Perform batch prediction on CSV file.

        Args:
            input_csv: Path to input CSV file
            output_csv: Path to save results (optional)

        Returns:
            DataFrame with predictions

        Example:
            >>> results = predictor.predict_batch('patients.csv', 'results.csv')
            >>> print(results['label'].value_counts())
        """
        if self.verbose:
            print(f"\nLoading data from: {input_csv}")

        df = pd.read_csv(input_csv)
        results = self.predict(df)

        if output_csv:
            results.to_csv(output_csv, index=False)
            if self.verbose:
                print(f"✓ Results saved to: {output_csv}")

        if self.verbose:
            self._print_summary(results)

        return results

    def _assess_risk(self, prediction: int, confidence: float) -> str:
        """
        Generate clinical risk assessment.

        Args:
            prediction: Predicted class (1, 2, or 3)
            confidence: Prediction confidence

        Returns:
            Risk assessment string
        """
        if prediction == 3:  # Pathological
            if confidence > 0.8:
                return "HIGH RISK - Immediate clinical review recommended"
            else:
                return "ELEVATED RISK - Clinical review recommended"
        elif prediction == 2:  # Suspect
            if confidence > 0.7:
                return "MODERATE RISK - Close monitoring advised"
            else:
                return "UNCERTAIN - Consider additional assessment"
        else:  # Normal
            if confidence > 0.9:
                return "LOW RISK - Normal monitoring"
            else:
                return "LOW-MODERATE RISK - Continue monitoring"

    def _print_summary(self, results: pd.DataFrame) -> None:
        """Print summary statistics for batch predictions."""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(results['label'].value_counts())
        print(f"\nAverage confidence: {results['confidence'].mean():.1%}")
        print(f"High-risk cases: {(results['prediction'] == 3).sum()}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_predictor(model_path: str = 'models/lifeline_model.pkl') -> LifelinePredictor:
    """
    Convenience function to load predictor.

    Args:
        model_path: Path to model file

    Returns:
        Initialized LifelinePredictor instance
    """
    return LifelinePredictor(model_path)


def predict_single(
    model_path: str,
    patient_data: Dict
) -> Dict:
    """
    Convenience function for single prediction.

    Args:
        model_path: Path to model file
        patient_data: Dictionary of patient features

    Returns:
        Prediction results dictionary
    """
    predictor = LifelinePredictor(model_path, verbose=False)
    return predictor.predict(patient_data)


# ==============================================================================
# MAIN (FOR TESTING)
# ==============================================================================

def main():
    """Demo usage of the inference module."""
    print("="*70)
    print("LIFELINE CTG CLASSIFIER - INFERENCE DEMO")
    print("="*70)

    # Example patient data
    sample_patient = {
        'b': 120.0, 'e': 0.0, 'AC': 3.0, 'FM': 0.0, 'UC': 4.0,
        'DL': 0.0, 'DS': 0.0, 'DP': 0.0, 'DR': 0.0, 'LB': 135.0,
        'ASTV': 45.0, 'MSTV': 1.0, 'ALTV': 25.0, 'MLTV': 6.0,
        'Width': 110.0, 'Min': 80.0, 'Max': 190.0, 'Nmax': 4.0,
        'Nzeros': 0.0, 'Mode': 135.0, 'Mean': 136.0, 'Median': 135.0,
        'Variance': 12.0, 'Tendency': 0.0
    }

    # Load model and predict
    predictor = load_predictor('../final_model_artifacts/lgbm_hybrid_model_corrected.pkl')
    result = predictor.predict(sample_patient)

    # Display results
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Prediction: {result['label']} (Class {result['prediction']})")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk Level: {result['risk_assessment']}")

    if 'probabilities' in result:
        print(f"\nClass Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label:15s}: {prob:.1%}")

    print("\n" + "="*70)
    print("✓ Demo complete")
    print("="*70)


if __name__ == "__main__":
    main()
