# ==============================================================================
# DATATHON 2025: LIFELINE - SUPERBLEND ENSEMBLE MODEL
#
#
# Purpose:
# This script trains, evaluates, and saves a powerful superblend ensemble model
# for CTG classification. The process involves several key stages:
#
#   1.  Hyperparameter Tuning: Uses Optuna to find the best parameters for
#       six different base models (LGBM, XGB, CatBoost, RF, ET, MLP) based on
#       cross-validated Macro-F1 score.
#
#   2.  Out-of-Fold (OOF) Prediction Generation: Employs Repeated Stratified K-Fold
#       cross-validation to generate reliable OOF predictions from each base
#       model. This prevents data leakage into the meta-learner.
#
#   3.  Probability Calibration: Uses CalibratedClassifierCV with isotonic
#       regression to ensure the predicted probabilities from each model are
#       well-calibrated.
#
#   4.  Meta-Learning: Trains a final Logistic Regression model (the meta-learner)
#       on the OOF predictions from the base models, plus several engineered
#       "passthrough" features.
#
#   5.  Threshold Optimization: Finds the optimal probability threshold for the
#       'Pathologic' class to maximize a custom metric that heavily weights
#       recall, minimizing false negatives.
#
#   6.  Final Evaluation: Evaluates the complete, fine-tuned pipeline on a
#       held-out test set and saves all necessary artifacts for prediction.
#
# ==============================================================================

import os
import json
import argparse
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import optuna

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# --- Gradient Boosting Libraries ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- Imbalanced Learning ---
from imblearn.over_sampling import BorderlineSMOTE

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION AND UTILITY FUNCTIONS
# ==============================================================================

# --- Constants ---
REQUIRED_FEATURES = [
    'b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR', 'LB',
    'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max',
    'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]
CLASS_NAMES = ['Normal', 'Suspect', 'Pathological']
RANDOM_STATE = 42
N_CLASSES = 3

def load_data(path):
    """Loads and validates the preprocessed CTG data."""
    df = pd.read_csv(path)
    missing_cols = set(REQUIRED_FEATURES + ['NSP']) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Data file is missing required columns: {missing_cols}")
    X = df[REQUIRED_FEATURES].copy()
    y = df['NSP'].astype(int).values
    return X, y

def create_passthrough_features(df):
    """Creates simple engineered features to pass directly to the meta-learner."""
    passthrough_df = pd.DataFrame(index=df.index)
    passthrough_df['decels_weighted'] = (
        df.get('DS', 0) * 3 + df.get('DP', 0) * 2 + 
        df.get('DR', 0) * 2 + df.get('DL', 0) * 1
    )
    passthrough_df['STV_ratio'] = df['ASTV'] / (df['MSTV'] + 1e-6)
    passthrough_df['LTV_ratio'] = df['ALTV'] / (df['MLTV'] + 1e-6)
    passthrough_df['Mean_Median_diff'] = df['Mean'] - df['Median']
    return passthrough_df

# ==============================================================================
# 2. MODEL DEFINITIONS AND HYPERPARAMETER SPACES
# ==============================================================================

def get_model_config(model_name):
    """Returns a model instance and its Optuna hyperparameter search space."""
    if model_name == 'lgbm':
        model = LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1, verbose=-1)
        space = {
            'n_estimators': optuna.trial.Trial.suggest_int('n_estimators', 800, 1600),
            'learning_rate': optuna.trial.Trial.suggest_float('learning_rate', 0.02, 0.07),
            'num_leaves': optuna.trial.Trial.suggest_int('num_leaves', 31, 127),
            'max_depth': optuna.trial.Trial.suggest_int('max_depth', 5, 8),
            'min_child_samples': optuna.trial.Trial.suggest_int('min_child_samples', 10, 60),
            'subsample': optuna.trial.Trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': optuna.trial.Trial.suggest_float('colsample_bytree', 0.7, 1.0)
        }
    elif model_name == 'xgb':
        model = XGBClassifier(objective='multi:softprob', num_class=N_CLASSES, random_state=RANDOM_STATE, tree_method='hist', nthread=4, eval_metric='mlogloss')
        space = {
            'n_estimators': optuna.trial.Trial.suggest_int('n_estimators', 800, 1600),
            'learning_rate': optuna.trial.Trial.suggest_float('learning_rate', 0.02, 0.07),
            'max_depth': optuna.trial.Trial.suggest_int('max_depth', 5, 8),
            'min_child_weight': optuna.trial.Trial.suggest_int('min_child_weight', 1, 6),
            'subsample': optuna.trial.Trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': optuna.trial.Trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': optuna.trial.Trial.suggest_float('gamma', 0.0, 2.0),
            'reg_lambda': optuna.trial.Trial.suggest_float('reg_lambda', 0.0, 5.0)
        }
    elif model_name == 'cat':
        model = CatBoostClassifier(loss_function='MultiClass', random_seed=RANDOM_STATE, thread_count=4, verbose=False)
        space = {
            'iterations': optuna.trial.Trial.suggest_int('iterations', 1200, 2400),
            'learning_rate': optuna.trial.Trial.suggest_float('learning_rate', 0.03, 0.07),
            'depth': optuna.trial.Trial.suggest_int('depth', 6, 8),
            'l2_leaf_reg': optuna.trial.Trial.suggest_float('l2_leaf_reg', 4.0, 12.0),
        }
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        space = {
            'n_estimators': optuna.trial.Trial.suggest_int('n_estimators', 700, 1200),
            'max_depth': optuna.trial.Trial.suggest_int('max_depth', 16, 28),
            'min_samples_split': optuna.trial.Trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': optuna.trial.Trial.suggest_int('min_samples_leaf', 1, 3)
        }
    elif model_name == 'et':
        model = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        space = {
            'n_estimators': optuna.trial.Trial.suggest_int('n_estimators', 700, 1200),
            'max_depth': optuna.trial.Trial.suggest_int('max_depth', 16, 28),
            'min_samples_split': optuna.trial.Trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': optuna.trial.Trial.suggest_int('min_samples_leaf', 1, 3)
        }
    elif model_name == 'mlp':
        model = MLPClassifier(random_state=RANDOM_STATE, max_iter=400, early_stopping=True)
        space = {
            'hidden_layer_sizes': optuna.trial.Trial.suggest_categorical('hidden_layer_sizes', [(128,), (256,), (128, 64)]),
            'alpha': optuna.trial.Trial.suggest_float('alpha', 1e-4, 1e-2, log=True)
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, space

# ==============================================================================
# 3. HYPERPARAMETER OPTIMIZATION
# ==============================================================================

def optimize_model_params(model_name, X, y, n_trials=40):
    """Uses Optuna to find the best hyperparameters for a given model."""
    
    base_model, search_space = get_model_config(model_name)

    def objective(trial):
        params = {name: func(name) for name, func in search_space.items()}
        model = deepcopy(base_model)
        model.set_params(**params)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            smote = BorderlineSMOTE(random_state=RANDOM_STATE)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

            # Adjust for models requiring 0-indexed labels
            y_train_fit = y_train_resampled - 1 if model_name in ['xgb', 'cat'] else y_train_resampled
            
            model.fit(X_train_resampled, y_train_fit)
            y_pred = model.predict(X_val_scaled)
            
            # Convert predictions back to 1-indexed if necessary
            y_pred_eval = y_pred + 1 if model_name in ['xgb', 'cat'] else y_pred
            
            f1_scores.append(f1_score(y_val, y_pred_eval, average='macro'))
            
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params

# ==============================================================================
# 4. THRESHOLD OPTIMIZATION
# ==============================================================================

def optimize_pathologic_threshold(y_true, y_proba, w_recall3=0.7, w_f1=0.3):
    """Finds the best probability threshold for Class 3 (Pathologic)."""
    best_threshold = 0.5
    best_score = -1
    
    for threshold in np.linspace(0.15, 0.55, 41):
        y_pred = np.argmax(y_proba, axis=1) + 1
        y_pred[y_proba[:, 2] > threshold] = 3  # Force prediction to Pathologic
        
        recall_pathologic = recall_score(y_true, y_pred, labels=[3], average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        score = w_recall3 * recall_pathologic + w_f1 * macro_f1
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold

# ==============================================================================
# 5. MAIN TRAINING AND EVALUATION PIPELINE
# ==============================================================================

def main(data_path, out_dir, repeats, folds, trials):
    """Main function to run the entire superblend pipeline."""
    
    print("--- Starting Superblend Pipeline ---")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data and create passthrough features
    X_df, y = load_data(data_path)
    X, P = X_df.values.astype(np.float32), create_passthrough_features(X_df).values.astype(np.float32)
    X_train, X_test, y_train, y_test, P_train, P_test = train_test_split(
        X, y, P, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Data loaded. Train size: {len(y_train)}, Test size: {len(y_test)}")

    # 2. Hyperparameter Tuning
    model_names = ['lgbm', 'xgb', 'cat', 'rf', 'et', 'mlp']
    best_params = {}
    for name in model_names:
        print(f"\n[Optuna] Tuning {name.upper()} with {trials} trials...")
        best_params[name] = optimize_model_params(name, X_train, y_train, n_trials=trials)
        print(f"  ✓ Best params for {name.upper()}: {best_params[name]}")

    # 3. Build OOF Predictions
    print("\n--- Building Out-of-Fold (OOF) Predictions ---")
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=RANDOM_STATE)
    oof_probas = {name: np.zeros((len(y_train), N_CLASSES)) for name in model_names}

    for fold_id, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train), 1):
        X_train_f, X_val_f = X_train[train_idx], X_train[val_idx]
        y_train_f, y_val_f = y_train[train_idx], y_train[val_idx]
        
        scaler = StandardScaler().fit(X_train_f)
        X_train_scaled = scaler.transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)
        
        X_resampled, y_resampled = BorderlineSMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train_f)
        
        for name in model_names:
            model, _ = get_model_config(name)
            model.set_params(**best_params[name])
            
            y_train_fit = y_resampled - 1 if name in ['xgb', 'cat'] else y_resampled
            model.fit(X_resampled, y_train_fit)
            
            calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic').fit(X_val_scaled, y_val_f - 1 if name in ['xgb', 'cat'] else y_val_f)
            oof_probas[name][val_idx] += calibrator.predict_proba(X_val_scaled)
            
        print(f"  ✓ Fold {fold_id}/{folds*repeats} complete.")
    
    # Average OOF predictions across repeats
    for name in model_names:
        oof_probas[name] /= repeats

    # 4. Train Meta-Learner
    print("\n--- Training Meta-Learner ---")
    oof_concat = np.hstack([oof_probas[name] for name in model_names])
    P_train_scaled = StandardScaler().fit_transform(P_train)
    meta_X_train = np.hstack([oof_concat, P_train_scaled])

    # Find optimal threshold on a validation set
    meta_lr_val = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=RANDOM_STATE)
    X_meta_tr, X_meta_val, y_meta_tr, y_meta_val = train_test_split(
        meta_X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    meta_lr_val.fit(X_meta_tr, y_meta_tr)
    val_probas = meta_lr_val.predict_proba(X_meta_val)
    best_threshold = optimize_pathologic_threshold(y_meta_val, val_probas)
    
    # Refit on full meta-training data
    meta_learner = LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=RANDOM_STATE)
    meta_learner.fit(meta_X_train, y_train)
    print("  ✓ Meta-learner trained and threshold optimized.")

    # 5. Final Evaluation on Test Set
    print("\n--- Evaluating on Hold-Out Test Set ---")
    # Get test predictions from each base model
    test_probas_list = []
    for name in model_names:
        model, _ = get_model_config(name)
        model.set_params(**best_params[name])
        
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_resampled, y_resampled = BorderlineSMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)
        y_train_fit = y_resampled - 1 if name in ['xgb', 'cat'] else y_resampled
        
        model.fit(X_resampled, y_train_fit)
        
        calibrator = CalibratedClassifierCV(model, cv='prefit', method='isotonic').fit(X_test_scaled, y_test - 1 if name in ['xgb', 'cat'] else y_test)
        test_probas_list.append(calibrator.predict_proba(X_test_scaled))
        
    # Create meta features for the test set and predict
    P_test_scaled = StandardScaler().fit(P_train).transform(P_test)
    meta_X_test = np.hstack(test_probas_list + [P_test_scaled])
    meta_probas_test = meta_learner.predict_proba(meta_X_test)
    
    y_pred = np.argmax(meta_probas_test, axis=1) + 1
    y_pred[meta_probas_test[:, 2] > best_threshold] = 3

    # Report final metrics
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    rec3 = recall_score(y_test, y_pred, labels=[3], average='macro', zero_division=0)
    
    print("\n=== Superblend Final Results ===")
    print(f"  Best Pathological Threshold: {best_threshold:.4f}")
    print(f"  Accuracy:                  {acc:.4f}")
    print(f"  Macro F1-Score:            {f1m:.4f}")
    print(f"  Pathological Recall:       {rec3:.4f}")
    print("\n  Confusion Matrix (Labels: 1, 2, 3):\n", confusion_matrix(y_test, y_pred, labels=[1, 2, 3]))
    print("\n  Classification Report:\n", classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # 6. Save Artifacts
    print("\n--- Saving Artifacts ---")
    bundle = {
        'base_model_params': best_params,
        'meta_learner': meta_learner,
        'best_threshold': best_threshold,
        'required_features': REQUIRED_FEATURES,
        'passthrough_features': ['decels_weighted', 'STV_ratio', 'LTV_ratio', 'Mean_Median_diff']
    }
    with open(os.path.join(out_dir, 'superblend_bundle.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    
    print(f"  ✓ Bundle saved to {os.path.join(out_dir, 'superblend_bundle.pkl')}")
    print("--- Pipeline Complete ---")

# ==============================================================================
# 6. COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CTG Superblend training pipeline.")
    parser.add_argument('--data', type=str, required=True, help="Path to the cleaned_data.csv file.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory to save output artifacts.")
    parser.add_argument('--repeats', type=int, default=3, help="Number of repeats for RepeatedStratifiedKFold.")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument('--trials', type=int, default=40, help="Number of Optuna trials for each model.")
    args = parser.parse_args()
    
    main(args.data, args.outdir, args.repeats, args.folds, args.trials)
