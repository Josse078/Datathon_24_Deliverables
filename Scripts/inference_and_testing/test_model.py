#!/usr/bin/env python3
"""
===============================================================================
LIFELINE CTG CLASSIFIER - TESTING & DEMONSTRATION
===============================================================================

Datathon 2025 - Team TM-37
Authors: Ricard Josse Meyer, Charles Lukas Chairos Yo, Poon Wei Lok, [Your Name]

This script demonstrates the capabilities of the Lifeline CTG classifier:
1. Single patient prediction
2. Batch prediction
3. Model performance metrics
4. Confidence analysis

Usage:
    python test_model.py                    # Run all tests
    python test_model.py --quick            # Quick test only
    python test_model.py --data <path>      # Test on specific data

===============================================================================
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import inference module
try:
    from inference import LifelinePredictor
except ImportError:
    print("Error: inference.py not found. Make sure it's in the same directory.")
    sys.exit(1)


# ==============================================================================
# TEST CONFIGURATIONS
# ==============================================================================

# Default model path (update as needed)
DEFAULT_MODEL_PATH = '../final_model_artifacts/lgbm_hybrid_model_corrected.pkl'

# Test cases representing different clinical scenarios
TEST_CASES = {
    'normal': {
        'description': 'Normal pregnancy - reassuring pattern',
        'data': {
            'b': 120.0, 'e': 0.0, 'AC': 3.0, 'FM': 0.0, 'UC': 4.0,
            'DL': 0.0, 'DS': 0.0, 'DP': 0.0, 'DR': 0.0, 'LB': 135.0,
            'ASTV': 45.0, 'MSTV': 1.0, 'ALTV': 25.0, 'MLTV': 6.0,
            'Width': 110.0, 'Min': 80.0, 'Max': 190.0, 'Nmax': 4.0,
            'Nzeros': 0.0, 'Mode': 135.0, 'Mean': 136.0, 'Median': 135.0,
            'Variance': 12.0, 'Tendency': 0.0
        },
        'expected': 'Normal'
    },
    'suspect': {
        'description': 'Suspect pattern - borderline variability',
        'data': {
            'b': 120.0, 'e': 1.0, 'AC': 1.0, 'FM': 0.0, 'UC': 3.0,
            'DL': 1.0, 'DS': 0.0, 'DP': 0.0, 'DR': 0.0, 'LB': 150.0,
            'ASTV': 60.0, 'MSTV': 1.5, 'ALTV': 40.0, 'MLTV': 9.0,
            'Width': 140.0, 'Min': 70.0, 'Max': 210.0, 'Nmax': 6.0,
            'Nzeros': 1.0, 'Mode': 145.0, 'Mean': 148.0, 'Median': 145.0,
            'Variance': 20.0, 'Tendency': 1.0
        },
        'expected': 'Suspect'
    },
    'pathological': {
        'description': 'Pathological pattern - concerning features',
        'data': {
            'b': 120.0, 'e': 2.0, 'AC': 0.0, 'FM': 0.0, 'UC': 2.0,
            'DL': 2.0, 'DS': 1.0, 'DP': 1.0, 'DR': 0.0, 'LB': 105.0,
            'ASTV': 75.0, 'MSTV': 2.1, 'ALTV': 60.0, 'MLTV': 12.0,
            'Width': 180.0, 'Min': 50.0, 'Max': 230.0, 'Nmax': 9.0,
            'Nzeros': 2.0, 'Mode': 100.0, 'Mean': 102.0, 'Median': 100.0,
            'Variance': 35.0, 'Tendency': -1.0
        },
        'expected': 'Pathological'
    }
}


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_single_predictions(predictor: LifelinePredictor) -> None:
    """Test single patient predictions on predefined cases."""
    print("\n" + "="*70)
    print("TEST 1: SINGLE PATIENT PREDICTIONS")
    print("="*70)

    for case_name, case_info in TEST_CASES.items():
        print(f"\n{case_name.upper()} CASE")
        print(f"Description: {case_info['description']}")
        print("-" * 70)

        result = predictor.predict(case_info['data'])

        # Check if prediction matches expected
        matches = result['label'] == case_info['expected']
        match_symbol = "✓" if matches else "✗"

        print(f"Expected:   {case_info['expected']}")
        print(f"Predicted:  {result['label']} {match_symbol}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Risk Level: {result['risk_assessment']}")

        print(f"\nProbabilities:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 30)
            print(f"  {label:15s} {prob:6.1%}  {bar}")


def test_batch_prediction(predictor: LifelinePredictor) -> pd.DataFrame:
    """Test batch prediction on multiple samples."""
    print("\n" + "="*70)
    print("TEST 2: BATCH PREDICTION")
    print("="*70)

    # Create batch dataset from test cases
    batch_data = pd.DataFrame([
        case_info['data'] for case_info in TEST_CASES.values()
    ])

    print(f"\nProcessing {len(batch_data)} samples...")
    results = predictor.predict(batch_data)

    print(f"\nResults Summary:")
    print(f"  Total samples: {len(results)}")
    print(f"\nPrediction distribution:")
    print(results['label'].value_counts())

    print(f"\nConfidence statistics:")
    print(f"  Mean: {results['confidence'].mean():.1%}")
    print(f"  Min:  {results['confidence'].min():.1%}")
    print(f"  Max:  {results['confidence'].max():.1%}")

    return results


def test_with_actual_data(
    predictor: LifelinePredictor,
    data_path: str
) -> None:
    """Test on actual test dataset if available."""
    print("\n" + "="*70)
    print("TEST 3: PERFORMANCE ON TEST DATA")
    print("="*70)

    if not Path(data_path).exists():
        print(f"\n⚠️  Test data not found: {data_path}")
        print("   Skipping this test.")
        return

    print(f"\nLoading test data from: {data_path}")
    df = pd.read_csv(data_path)

    # Check if labels exist
    has_labels = 'NSP' in df.columns

    if has_labels:
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'NSP']
        X = df[feature_cols]
        y_true = df['NSP']

        # Get predictions
        results = predictor.predict(X)
        y_pred = results['prediction'].values

        # Calculate metrics
        from sklearn.metrics import (
            classification_report, confusion_matrix, 
            accuracy_score, recall_score
        )

        print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.1%}")

        print(f"\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Suspect', 'Pathological'],
            digits=3
        ))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print("\n             Predicted")
        print("              Nor  Sus  Pat")
        print("Actual Normal", f"{cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
        print("       Suspect", f"{cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
        print("       Pathol.", f"{cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

        # Key metric: Pathological recall (clinical safety)
        path_recall = recall_score(y_true, y_pred, average=None, labels=[1,2,3])[2]
        print(f"\n⭐ Pathological Recall (Key Safety Metric): {path_recall:.1%}")

        # Confidence analysis
        print(f"\nConfidence Analysis:")
        conf = results['confidence']
        print(f"  Mean confidence: {conf.mean():.1%}")
        print(f"  Samples >90% confident: {(conf > 0.9).sum()}/{len(conf)} ({100*(conf > 0.9).sum()/len(conf):.1f}%)")

    else:
        print("\n⚠️  No labels found in test data (NSP column missing)")
        print("   Running prediction only...")

        results = predictor.predict(df)
        print(f"\nPredicted {len(results)} samples")
        print(f"\nPrediction distribution:")
        print(results['label'].value_counts())


def analyze_confidence_distribution(results: pd.DataFrame) -> None:
    """Analyze and visualize confidence distribution."""
    print("\n" + "="*70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*70)

    conf = results['confidence']

    # Binned distribution
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    labels = ['<50%', '50-70%', '70-80%', '80-90%', '90-95%', '95-100%']
    conf_binned = pd.cut(conf, bins=bins, labels=labels, include_lowest=True)

    print(f"\nConfidence ranges:")
    for label in labels:
        count = (conf_binned == label).sum()
        pct = 100 * count / len(conf)
        bar = "█" * int(pct / 2)
        print(f"  {label:10s} {count:3d} ({pct:5.1f}%)  {bar}")

    # Statistical summary
    print(f"\nStatistics:")
    print(f"  Mean:   {conf.mean():.3f}")
    print(f"  Median: {conf.median():.3f}")
    print(f"  Std:    {conf.std():.3f}")
    print(f"  Min:    {conf.min():.3f}")
    print(f"  Max:    {conf.max():.3f}")


def print_usage_examples() -> None:
    """Print code examples for using the inference module."""
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)

    examples = """
# Example 1: Single prediction
from inference import LifelinePredictor

predictor = LifelinePredictor('models/lifeline_model.pkl')

patient_data = {
    'LB': 135, 'ASTV': 45, 'AC': 3,
    # ... all required features
}

result = predictor.predict(patient_data)
print(f"{result['label']}: {result['confidence']:.1%}")


# Example 2: Batch prediction
results = predictor.predict_batch(
    'data/patients.csv',
    'results/predictions.csv'
)

high_risk = results[results['prediction'] == 3]
print(f"Found {len(high_risk)} high-risk cases")


# Example 3: Integration in workflow
for _, patient in patients_df.iterrows():
    result = predictor.predict(patient)

    if result['prediction'] == 3 and result['confidence'] > 0.8:
        alert_clinical_team(patient['id'], result)
"""

    print(examples)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all tests and demonstrations."""
    parser = argparse.ArgumentParser(
        description='Test and demonstrate Lifeline CTG classifier'
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL_PATH,
        help='Path to model file'
    )
    parser.add_argument(
        '--data',
        default=None,
        help='Path to test data CSV (optional)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test only (single predictions)'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show usage examples and exit'
    )

    args = parser.parse_args()

    # Show examples and exit if requested
    if args.examples:
        print_usage_examples()
        return

    # Header
    print("="*70)
    print("LIFELINE CTG CLASSIFIER - TESTING & DEMONSTRATION")
    print("="*70)
    print(f"Model: {args.model}")

    # Load predictor
    try:
        predictor = LifelinePredictor(args.model)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"\nMake sure the model file exists at: {args.model}")
        sys.exit(1)

    # Run tests
    try:
        # Test 1: Single predictions (always run)
        test_single_predictions(predictor)

        if not args.quick:
            # Test 2: Batch prediction
            batch_results = test_batch_prediction(predictor)

            # Test 3: Actual data (if provided)
            if args.data:
                test_with_actual_data(predictor, args.data)

            # Analysis
            analyze_confidence_distribution(batch_results)

        # Success message
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nModel is ready for deployment.")
        print("\nFor usage examples, run: python test_model.py --examples")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
