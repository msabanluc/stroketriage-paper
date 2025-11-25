#%% Imports
import os
import pandas as pd
import pickle
from IPython.display import display

from config.config import settings
from pipelines import training_pipeline as main_runner
from pipelines import preliminary_analysis as prelim
from evaluation import figures, table4

#%%

# Define the tasks and data modes to run
TASKS_TO_RUN = ["stroke", "severe"]
DATA_MODES_TO_RUN = ["full", "cpss", "stroke_relevant"]

os.chdir(settings['root'])
OUTPUT_DIR = settings['output_dir']

#%% Loop through all configurations and run the main evaluation pipeline.

if settings.get('retune_models', False) or settings.get('retrain_models', True):
    if settings.get('retune_models', False):
        print("Running Main Evaluation Pipeline (retune_models is True)")
    else:
        print("Running Main Evaluation Pipeline (retrain_models is True)")
    for task in TASKS_TO_RUN:
        for data_mode in DATA_MODES_TO_RUN:
            print(f"--- Running experiment for task: '{task}', data_mode: '{data_mode}' ---")
            
            original_label_type = settings['label_type']
            original_data_mode = settings['data_mode']
            settings['label_type'] = task
            settings['data_mode'] = data_mode
            
            main_runner.run_all()
            
            settings['label_type'] = original_label_type
            settings['data_mode'] = original_data_mode
            
            print(f"--- Finished experiment for task: '{task}', data_mode: '{data_mode}' ---\n")
    print("All experiments complete. Evaluation .pkl files have been generated.\n")
else:
    print("Skipping Main Evaluation Pipeline (retune_models and retrain_models are False)")


#%% Run Preliminary / Descriptive Analysis
df, df_suspected, df_stroke = prelim.load_and_preprocess_data()

# Generate Descriptive Tables 1 & 2
print("\n--- Generating Table 1 & Table 2 ---")
prelim.generate_table1(df)
prelim.generate_table2(df)

#%% Generate Table 4 (Confusion Matrices)
print("\n--- Generating Table 4 (Confusion Matrices) ---")
try:
    table4_full = table4.generate_table4_confusion_matrices()
    if not table4_full.empty:
        display(table4_full.style.format("{:,.0f}", na_rep="-"))
        table4_save_path = OUTPUT_DIR / "table4_confusion_matrices.csv"
        table4_full.to_csv(table4_save_path)
        print(f"Table 4 saved to {table4_save_path}")
except FileNotFoundError as e:
    print(f"Could not generate Table 4: {e}")
except Exception as e:
    print(f"An unexpected error occurred while generating Table 4: {e}")


#%% Generate Figure 2
print("\n--- Generating Figure 2 ---")
prelim.generate_figure2(df)


#%% Generate Figure 3/SF2 (Combined ROC & PR Curves) for all cohorts
print("\n--- Generating Figure 3/SF2 (Combined ROC & PR Curves) for all cohorts ---")
for suffix in ["_all", "_suspected", "_strokeRel"]:
    try:
        print(f"Generating for cohort: {suffix}")
        stroke_models = figures.load_task("stroke", suffix=suffix)   
        severe_models = figures.load_task("severe", suffix=suffix)
        if not stroke_models or not severe_models:
            print(f"Skipping {suffix} due to missing model files.")
            continue
        figures.plot_combined_roc_pr(
            stroke_models, severe_models,
            save_name=f"fig3_combined_roc_pr{suffix}.png"
        )
    except FileNotFoundError as e:
        print(f"Could not generate Figure 3 for cohort {suffix}: {e}")


#%% Generate Figure 4 (Combined SHAP Plots)
print("\n--- Generating Figure 4 (Combined SHAP Plots) ---")
FIGURE_DIR = settings['figure_dir']
for suffix in ["_all", "_suspected", "_strokeRel"]:
    try:
        print(f"Generating for cohort: {suffix}")
        
        # Top models: XGB for stroke, RF for severe
        left_img_path = FIGURE_DIR / f"shap_xgb_stroke{suffix}.png"
        right_img_path = FIGURE_DIR / f"shap_rf_severe{suffix}.png"
        output_filename = f"fig4_shap_summary{suffix}.png"
        
        if not left_img_path.exists() or not right_img_path.exists():
            print(f"Skipping {suffix} due to missing SHAP plot images.")
            continue

        figures.merge_shap_figures(
            left_img_path,
            right_img_path,
            output_filename,
            title="SHAP Feature Importance",
        )
    except Exception as e:
        print(f"Could not generate combined SHAP figure for cohort {suffix}: {e}")

#%% Generate Supplemental Figure S1 (Sensitivity/Specificity Tradeoff)
print("\n--- Generating Supplemental Figure S1 (Sensitivity/Specificity Tradeoff) ---")
try:
    # Load data for XGB (stroke) and RF (severe) for both "all" and "suspected"
    ev_stroke_all = figures.load_eval("stroke", "_all", "xgb", "XGB")
    ev_severe_all = figures.load_eval("severe", "_all", "rf", "RF")
    ev_stroke_suspected = figures.load_eval("stroke", "_suspected", "xgb", "XGB")
    ev_severe_suspected = figures.load_eval("severe", "_suspected", "rf", "RF")

    figures.plot_threshold_tradeoff(
        ev_stroke_all, ev_stroke_suspected,
        ev_severe_all, ev_severe_suspected
    )
except FileNotFoundError as e:
    print(f"Could not generate sensitivity/specificity tradeoff figure: {e}")


#%% Generate Supplemental Figure S3 (Calibration Plot) for all cohorts
print("\n--- Generating Supplemental Figure S3 (Calibration Plot) for all cohorts ---")
for suffix in ["_all", "_suspected", "_strokeRel"]:
    try:
        print(f"Generating for cohort: {suffix}")
        # Top models: XGB for stroke, RF for severe
        ev_stroke_xgb = figures.load_eval("stroke", suffix, "xgb", "XGB")
        ev_severe_rf = figures.load_eval("severe", suffix, "rf", "RF")
        
        figures.make_calibration_figure(
            ev_stroke_xgb, 
            ev_severe_rf, 
            stroke_model_name="XGB", 
            severe_model_name="RF",
            save_name=f"S3_calibration_figure{suffix}.png"
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not generate calibration figure for cohort {suffix}: {e}")


#%% Generate Supplemental Tables 2-7
print("\n--- Supplemental Tables 2-7 ---")
prelim.generate_supplemental_table2(df)
prelim.generate_supplemental_table3(df)
prelim.generate_supplemental_tables_4_and_5(df)
prelim.generate_supplemental_table6(df)
prelim.generate_supplemental_table7(df, df_suspected)


#%% Generate dispatch performance and threshold metrics
print("\n--- Generating Dispatch Performance and Threshold Metrics ---")
prelim.generate_dispatch_performance(df, df_suspected)
prelim.generate_threshold_metrics()
