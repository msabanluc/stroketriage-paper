#%% Imports
from IPython.display import display
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config.config import settings
from data.preprocessing import run_full_preprocessing
from evaluation.performance import compute_operating_metrics

OUTPUT_DIR = settings['output_dir']

# Set SUFFIX to one of {"_all","_suspected","_strokeRel"}
SUFFIX = "_suspected"

ALLOWED_SUFFIXES = ["_strokeRel", "_suspected", "_all"]

TASKS = {"stroke": "stroke_label", "severe": "critical_label"}
MODELS = ["rf", "xgb", "snn"]
MODEL_PRETTY = {"rf": "Random Forest", "xgb": "XGBoost", "snn": "SNN"}

CUSTOM_THRESHOLDS = {
    ("stroke", "xgb"):  [0.075, 0.08, 0.085, 0.100, 0.150],
    ("severe", "rf"):   [0.005, 0.025, 0.030, 0.100, 0.200],
}

# %% 2) Helpers

def get_adaptive_thresholds(probs, num=100):
    probs = np.asarray(probs)
    min_thr = max(float(np.nanmin(probs)), 1e-6)
    max_thr = min(float(np.nanmax(probs)), 1 - 1e-6)
    if not np.isfinite(min_thr) or not np.isfinite(max_thr) or min_thr >= max_thr:
        return np.linspace(0.001, 0.999, num=num)
    return np.linspace(min_thr, max_thr, num=num)


def _candidate_paths(task, model, suffix=None):
    def _yield_for(sfx):
        if sfx == "_strokeRel":
            yield OUTPUT_DIR / f"eval_{task}_strokeRel_{model}.pkl"
            yield OUTPUT_DIR / f"eval_{task}_stroke_Rel_{model}.pkl"
        else:
            yield OUTPUT_DIR / f"eval_{task}{sfx}_{model}.pkl"

    if suffix is not None:
        for p in _yield_for(suffix):
            yield p
    else:
        for sfx in ALLOWED_SUFFIXES:
            for p in _yield_for(sfx):
                yield p


def _find_eval_path(task, model):
    for p in _candidate_paths(task, model, suffix=SUFFIX):
        if p.exists():
            return p
    return None


def load_eval(task, model):
    p = _find_eval_path(task, model)
    if p is None:
        raise FileNotFoundError(
            f"No eval file found for task='{task}', model='{model}' in {OUTPUT_DIR} "
            f"(searched suffixes: { [SUFFIX] if SUFFIX else ALLOWED_SUFFIXES })."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def select_best_model(evals):
    return max(evals.items(), key=lambda kv: kv[1]["pr_auc"])


def _metrics_table_for(probs, y_true, thresholds=None):
    if thresholds is None:
        thresholds = get_adaptive_thresholds(probs)

    rows = []
    for thr in thresholds:
        m = compute_operating_metrics(probs, y_true, thr)
        rows.append({"Threshold": thr,
                     "Sensitivity": m["sensitivity"],
                     "Specificity": m["specificity"],
                     "PPV": m["ppv"]})
    return pd.DataFrame(rows).set_index("Threshold")


def get_task_metrics(task, model, thresholds=None, num_thresholds=25):
    ev = load_eval(task, model)
    probs, y_true = ev["probabilities"], ev["y_true"]

    if thresholds is None:
        thresholds = get_adaptive_thresholds(probs, num=num_thresholds)

    df = _metrics_table_for(probs, y_true, thresholds).reset_index()
    df.columns = [f"{task}_{model}_{col.lower()}" for col in df.columns]
    return df


def _available_models_for_task(task):
    available = []
    for m in MODELS:
        if _find_eval_path(task, m) is not None:
            available.append(m)
    return available


def build_side_by_side_table(custom_thresholds=None, num_thresholds=25):
    best_models = {}
    task_tables = []

    for task in TASKS:
        candidate_models = _available_models_for_task(task)
        if not candidate_models:
            raise FileNotFoundError(f"No model files found for task='{task}' in {OUTPUT_DIR}.")

        evals = {m: load_eval(task, m) for m in candidate_models}
        best_key, _ = select_best_model(evals)
        best_models[task] = best_key

        thresholds = None
        if custom_thresholds is not None:
            thresholds = custom_thresholds.get((task, best_key), None)

        task_df = get_task_metrics(task, best_key, thresholds=thresholds, num_thresholds=num_thresholds)
        task_tables.append(task_df)

    combined = pd.concat(task_tables, axis=1)

    new_cols = []
    for col in combined.columns:
        parts = col.split("_")
        task_key, model_key = parts[0], parts[1]
        metric = parts[-1].capitalize() 
        pretty_task = "Stroke" if task_key == "stroke" else "Severe"
        pretty_model = MODEL_PRETTY.get(model_key, model_key.upper())
        new_cols.append((f"{pretty_task} – {pretty_model}", metric))
    combined.columns = pd.MultiIndex.from_tuples(new_cols)

    combined = combined[
        sorted(combined.columns, key=lambda x: (x[0], 0 if x[1].lower() == "threshold" else 1))
    ]

    return combined, best_models


def print_best_models(best_models):
    print("Winning models by task:")
    for task, mkey in best_models.items():
        print(f" - {task}: {MODEL_PRETTY.get(mkey, mkey)}")


#%% Display
def generate_table4_confusion_matrices():

    table_specs = {
        "Full Cohort (n=8,221)": {
            "suffix": "_all",
            "Stroke": {"model": "xgb", "threshold": 0.025},
            "Severe Stroke": {"model": "rf", "threshold": 0.010},
        },
        "EMS-Suspected Stroke Cohort (n=1,252)": {
            "suffix": "_suspected",
            "Stroke": {"model": "xgb", "threshold": 0.080},
            "Severe Stroke": {"model": "rf", "threshold": 0.025},
        },
        "Stroke and Mimic Cohort (n=2,573)": {
            "suffix": "_strokeRel",
            "Stroke": {"model": "xgb", "threshold": 0.075},
            "Severe Stroke": {"model": "rf", "threshold": 0.030},
        },
    }

    all_cohort_data = []

    for cohort_name, spec in table_specs.items():
        cohort_results = {}
        for task_name in ["Stroke", "Severe Stroke"]:
            params = spec[task_name]
            task_key = "stroke" if task_name == "Stroke" else "severe"
            
            try:
                global SUFFIX
                original_suffix = SUFFIX
                SUFFIX = spec["suffix"]
                
                ev = load_eval(task_key, params["model"])
                y_true = ev["y_true"]
                probs = ev["probabilities"]
                y_pred = (np.asarray(probs) >= params["threshold"]).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                
                cohort_results[f"Positive: All {task_name}"] = {
                    "Predicted Negative": [tn, fn],
                    "Predicted Positive": [fp, tp]
                }

            except FileNotFoundError as e:
                print(f"Warning: Could not generate CM for {cohort_name} - {task_name}. Reason: {e}")
                cohort_results[f"Positive: All {task_name}"] = {
                    "Predicted Negative": [np.nan, np.nan],
                    "Predicted Positive": [np.nan, np.nan]
                }
            finally:
                SUFFIX = original_suffix
        
        cohort_df = pd.DataFrame([
            cohort_results["Positive: All Stroke"]["Predicted Negative"],
            cohort_results["Positive: All Stroke"]["Predicted Positive"],
            cohort_results["Positive: All Severe Stroke"]["Predicted Negative"],
            cohort_results["Positive: All Severe Stroke"]["Predicted Positive"],
        ], columns=["True Negative", "True Positive"]).T
        
        cohort_df.columns = pd.MultiIndex.from_product([
            ["Positive: All Stroke", "Positive: All Severe Stroke"], 
            ["Predicted Negative", "Predicted Positive"]
        ])
        
        stroke_part = cohort_df["Positive: All Stroke"]
        severe_part = cohort_df["Positive: All Severe Stroke"]
        
        final_cohort_df = pd.concat([stroke_part, severe_part], axis=1, keys=["Positive: All Stroke", "Positive: All Severe Stroke"])
        final_cohort_df.columns = final_cohort_df.columns.droplevel(0)

        all_cohort_data.append((cohort_name, final_cohort_df))


    if not all_cohort_data:
        print("No confusion matrices were generated.")
        return pd.DataFrame()

    final_table = pd.concat([df for _, df in all_cohort_data], keys=[name for name, _ in all_cohort_data])
    
    return final_table


if __name__ == '__main__':
    print("--- Generating Full Table 4 ---")
    table4_full = generate_table4_confusion_matrices()
    if not table4_full.empty:
        display(table4_full.style.format("{:,.0f}", na_rep="-"))

    print("\n\n--- Generating Operating Characteristics Table (Original Script) ---")
    try:
        table4_op, winners = build_side_by_side_table(custom_thresholds=CUSTOM_THRESHOLDS)
        print_best_models(winners)
        display(table4_op.style.format("{:.3f}"))
    except FileNotFoundError as e:
        print(f"Could not generate operating characteristics table: {e}")

# %% b) Confusion matrices for best stroke/severe models at threshold X
THR_CM = 0.025

def confusion_tables_for_best(winners, thr=THR_CM):

    out = {}

    for task, model_key in winners.items():
        ev = load_eval(task, model_key)
        probs = ev["probabilities"]
        y_true = ev["y_true"]

        y_pred = (probs >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=pd.Index(["Actual 0 (No)", "Actual 1 (Yes)"], name=""),
            columns=pd.Index(["Pred 0 (No)", "Pred 1 (Yes)"], name=f"{task} @ {thr:.3f}")
        )

        metrics = {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
            "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "PPV": tp / (tp + fp) if (tp + fp) else float("nan"),
            "NPV": tn / (tn + fn) if (tn + fn) else float("nan"),
            "Prevalence": (tp + fn) / (tp + fp + tn + fn)
        }

        out[task] = {"model": MODEL_PRETTY.get(model_key, model_key),
                     "cm": cm_df, "metrics": metrics}

    return out

if __name__ == '__main__':
    _, winners = build_side_by_side_table(custom_thresholds=CUSTOM_THRESHOLDS)
    best_cms = confusion_tables_for_best(winners, thr=THR_CM)

    for task, bundle in best_cms.items():
        print(f"\n=== {task.capitalize()} — {bundle['model']} @ threshold {THR_CM} ===")
        display(bundle["cm"])
        m = bundle["metrics"]
        print(f"TP={m['TP']}, FP={m['FP']}, TN={m['TN']}, FN={m['FN']}")
        print(f"Sensitivity={m['Sensitivity']:.3f} | Specificity={m['Specificity']:.3f} | "
              f"PPV={m['PPV']:.3f} | NPV={m['NPV']:.3f} | Prevalence={m['Prevalence']:.3f}")


# %% DEBUG
if __name__ == '__main__':
    import pickle
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    OUTPUT_DIR = Path("outputs")
    SUFFIX     = "_all"
    MODELS     = ["rf", "xgb", "snn"]
    TASK       = "stroke"

    #%% Load all eval dicts
    evals = {}
    for m in MODELS:
        with open(OUTPUT_DIR / f"eval_{TASK}{SUFFIX}_{m}.pkl", "rb") as f:
            evals[m] = pickle.load(f)

    #%% Pick the best
    best_name, best_eval = max(evals.items(), key=lambda kv: kv[1]["auc"])

    #%% Grab the arrays
    probs  = best_eval["probabilities"]
    y_true = best_eval["y_true"]

    print("len(probs):", len(probs))
    print("len(y_true):", len(y_true))
    print("positive labels (sum y_true):", y_true.sum())

    df = pd.DataFrame({"y_true": y_true, "prob": probs})
    display(df.head())

    thr = 0.025
    y_pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion at thr={thr:.3f}:")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print("  Actual positives (TP+FN) =", tp+fn)

# %%
