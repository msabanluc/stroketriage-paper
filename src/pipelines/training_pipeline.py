#%% configuration
from config.config import settings

import os
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from config.reproducibility import enforce_reproducibility
from data import preprocessing
from evaluation.performance import (
    evaluate_model, threshold_sweep_analysis, plot_roc_pr_curves,
    find_best_threshold, compute_operating_metrics
)
from evaluation.shap import plot_shap_summary
from models.calibration import AttendedTemperatureScaling, fit_kfold_calibrated_model
from models.model_train import (
    train_rf_with_optuna, train_xgb_with_optuna, train_snn_with_optuna_and_logits
)

# Set working directory to project root
os.chdir(settings['root'])

#%% Preprocess data 

def run_preprocessing(mode="full", drop_dispatch=False):
    return preprocessing.run_full_preprocessing(mode=mode, drop_dispatch=drop_dispatch)

def run_all():
    LABEL_TYPE = settings['label_type']
    DATA_MODE = settings['data_mode']
    DROP_DISPATCH = settings['drop_dispatch']
    RETUNE_MODELS = settings['retune_models']
    N_TRIALS = settings['n_trials']
    
    enforce_reproducibility(settings['random_seed'])

    # Preprocess
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing(mode=DATA_MODE, drop_dispatch=DROP_DISPATCH)
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    label_column = 'stroke_label' if LABEL_TYPE == 'stroke' else 'critical_label'
    y = y_train_full[label_column]
    y_test_series = y_test[label_column]
    print(f"Using label_column = {label_column}")

    suffix_map = {
        "full": "_all",
        "cpss": "_suspected",
        "stroke_relevant": "_strokeRel"
    }
    suffix = suffix_map[DATA_MODE] + ("_nodisp" if DROP_DISPATCH else "")
    print(f"Using output suffix = {suffix}")

    # Random Forest
    print("\n--- Training and Evaluating Random Forest ---")
    rf_model, rf_best_params = train_rf_with_optuna(X_train_full, y, study_name=f"rf_{LABEL_TYPE}{suffix}", db_file=f"rf_{LABEL_TYPE}{suffix}.db", retune=RETUNE_MODELS)
    rf_calibrated = fit_kfold_calibrated_model(RandomForestClassifier, X_train_full, y, rf_best_params)
    eval_rf = evaluate_model(rf_calibrated, X_test, y_test_series, title="Random Forest (Calibrated)")
    probs_rf_uncal = rf_calibrated.base_model.predict_proba(X_test)[:, 1]
    probs_rf_cal   = rf_calibrated.predict_proba(X_test).values.ravel()
    eval_rf.update({"probs_uncal": probs_rf_uncal, "probs_cal": probs_rf_cal})
    with open(settings['output_dir'] / f"eval_{LABEL_TYPE}{suffix}_rf.pkl", "wb") as f:
        pickle.dump(eval_rf, f)
    print("Random Forest evaluation artifacts saved.")

    # XGBoost
    print("\n--- Training and Evaluating XGBoost ---")
    xgb_model, xgb_best_params = train_xgb_with_optuna(X_train_full, y, study_name=f"xgb_{LABEL_TYPE}{suffix}", db_file=f"xgb_{LABEL_TYPE}{suffix}.db", retune=RETUNE_MODELS)
    xgb_calibrated = fit_kfold_calibrated_model(xgb.XGBClassifier, X_train_full, y, xgb_best_params)
    eval_xgb = evaluate_model(xgb_calibrated, X_test, y_test_series, title="XGBoost (Calibrated)")
    probs_uncal = xgb_calibrated.base_model.predict_proba(X_test)[:, 1]
    probs_cal   = xgb_calibrated.predict_proba(X_test).values.ravel()
    eval_xgb.update({"probs_uncal": probs_uncal, "probs_cal": probs_cal})
    with open(settings['output_dir'] / f"eval_{LABEL_TYPE}{suffix}_xgb.pkl", "wb") as f:
        pickle.dump(eval_xgb, f)
    print("XGBoost evaluation artifacts saved.")

    # SNN
    if DATA_MODE != 'stroke_relevant':
        print("\n--- Training and Evaluating SNN ---")
        snn_models, snn_logits, snn_params = train_snn_with_optuna_and_logits(X_train_full, y, study_name=f"snn_{LABEL_TYPE}{suffix}", db_file=f"snn_{LABEL_TYPE}{suffix}.db", n_trials=N_TRIALS, retune=RETUNE_MODELS)
        ats = AttendedTemperatureScaling(num_bins=5)
        ats.fit(snn_logits, y)
        eval_snn = evaluate_model(snn_models, X_test, y_test_series, ts=ats, title="SNN Calibrated")
        with open(settings['output_dir'] / f"eval_{LABEL_TYPE}{suffix}_snn.pkl", "wb") as f:
            pickle.dump(eval_snn, f)
        print("SNN evaluation artifacts saved.")
    else:
        print("\n--- Skipping SNN for stroke_relevant data mode ---")
        eval_snn = None

    # SHAP Plots
    print("\n--- Generating SHAP Plots ---")
    plot_shap_summary(rf_model, X_test, "Random Forest", f"shap_rf_{LABEL_TYPE}{suffix}.png")
    plot_shap_summary(xgb_model, X_test, "XGBoost", f"shap_xgb_{LABEL_TYPE}{suffix}.png")
    print("SHAP summary plots saved.")
        
    # Threshold Sweeps & Best Thresholds
    print("\n--- Performing Threshold Sweeps (for diagnostics) ---")
    threshold_rf = threshold_sweep_analysis(eval_rf['probabilities'], y_test_series, model_name=f"RF {LABEL_TYPE.title()}", return_full_df=True)
    threshold_xgb = threshold_sweep_analysis(eval_xgb['probabilities'], y_test_series, model_name=f"XGB {LABEL_TYPE.title()}", return_full_df=True)
    
    best_threshold_rf = find_best_threshold(threshold_rf)
    best_threshold_xgb = find_best_threshold(threshold_xgb)

    metrics_rf = compute_operating_metrics(eval_rf['probabilities'], y_test_series, best_threshold_rf)
    metrics_xgb = compute_operating_metrics(eval_xgb['probabilities'], y_test_series, best_threshold_xgb)

    print("Best threshold metrics (Youden's J):")
    print("RF metrics:", metrics_rf)
    print("XGB metrics:", metrics_xgb)

    if eval_snn:
        threshold_snn = threshold_sweep_analysis(eval_snn['probabilities'], y_test_series, model_name=f"SNN {LABEL_TYPE.title()}", return_full_df=True)
        best_threshold_snn = find_best_threshold(threshold_snn)
        metrics_snn = compute_operating_metrics(eval_snn['probabilities'], y_test_series, best_threshold_snn)
        print("SNN metrics:", metrics_snn)

    # ROC/PR Curves for this run
    print("\n--- Generating ROC/PR Curves for this run ---")
    title_map = {
        "stroke": "Stroke Prediction",
        "severe": "Severe Stroke Prediction"
    }
    plot_title = title_map.get(LABEL_TYPE, LABEL_TYPE.title() + " Prediction")

    models_to_plot = [eval_rf, eval_xgb]
    labels_for_plot = ['Random Forest', 'XGBoost']
    if eval_snn:
        models_to_plot.append(eval_snn)
        labels_for_plot.append('SNN')

    plot_roc_pr_curves(
        *models_to_plot,
        baseline=y_test_series.mean(),
        labels=labels_for_plot,
        title=plot_title,
        save_name=f"roc_pr_comparison_{LABEL_TYPE}{suffix}.png"
    )
    print("ROC/PR comparison plot saved.")


if __name__ == '__main__':
    run_all()

# %%
