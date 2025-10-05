# ==============================================================================
# DATATHON 2025: LIFELINE - MODEL EXPLANATION SCRIPT
#
#
# Purpose:
# This script loads the final trained model and the engineered test set
# to generate a SHAP (SHapley Additive exPlanations) summary plot.
#
# Rationale:
# Interpretability is critical for clinical adoption. This script demonstrates
# that our model's decisions are transparent and based on clinically relevant
# features. The SHAP plot provides global feature importance, showing which
# factors most influence the model's predictions across all classes.
# ==============================================================================


import joblib
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')


def generate_shap_summary():
    """
    Loads the trained model and test data to generate and display a
    SHAP summary bar plot for global feature importance.
    """
    print("="*80)
    print("GENERATING SHAP SUMMARY PLOT FOR LIFELINE MODEL")
    print("="*80)

    MODEL_PATH = './final_model_artifacts/lgbm_hybrid_model_corrected.pkl'
    ENGINEERED_TEST_DATA_PATH = './data/preprocessed/X_test_engineered.csv'

    # --- 1. Load Pre-trained Model ---
    print(f"\n[1/3] Loading model from: {MODEL_PATH}...")
    try:
        lgbm_model = joblib.load(MODEL_PATH)
        print("  ✓ Model loaded successfully.")
    except FileNotFoundError:
        print(f"  ✗ ERROR: Model file not found at '{MODEL_PATH}'.")
        print("  Please run the main training script first to create the model artifact.")
        return

    # --- 2. Load Engineered Test Data ---
    print(f"\n[2/3] Loading engineered test data from: {ENGINEERED_TEST_DATA_PATH}...")
    try:
        X_test = pd.read_csv(ENGINEERED_TEST_DATA_PATH)
        print(f"  ✓ Test data with {X_test.shape[1]} features loaded.")
    except FileNotFoundError:
        print(f"  ✗ ERROR: Engineered test data not found at '{ENGINEERED_TEST_DATA_PATH}'.")
        print("  Please run the 'create_engineered_test_set.py' script first.")
        return

    # --- 3. Generate and Display SHAP Plot ---
    # Rationale: A SHAP bar plot provides the clearest and most direct view
    # of feature importance, making it ideal for presentations and reports.
    print("\n[3/3] Creating SHAP explainer and generating summary plot...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_test)

    print("  Displaying SHAP summary plot. This shows the average impact of each feature on the model's output magnitude.")
    shap.summary_plot(
        shap_values,
        X_test,
        class_names=['Normal', 'Suspect', 'Pathological'],
        plot_type="bar",  # "bar" provides a clean ranking of feature importance.
        show=True
    )

    print("\n" + "="*80)
    print("SHAP PLOT GENERATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    generate_shap_summary()
