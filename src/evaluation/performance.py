import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc as sk_auc, precision_score, recall_score, matthews_corrcoef, fbeta_score)
import seaborn as sns
from config.paths import FIGURE_DIR


def threshold_sweep_analysis(probs, y_true, model_name="Model", benchmark_sens=None, benchmark_spec=None,
                              plot_benchmark_lines=True, return_full_df=True):
    import matplotlib.pyplot as plt
    thresholds = np.linspace(0, 1, 1000)
    sensitivity_list = []
    specificity_list = []

    for thresh in thresholds:
        predicted = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity_list.append(sens)
        specificity_list.append(spec)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivity_list, label="Sensitivity")
    plt.plot(thresholds, specificity_list, label="Specificity")
    if plot_benchmark_lines and benchmark_sens and benchmark_spec:
        plt.axhline(y=benchmark_sens, color="r", linestyle="--", label="Benchmark Sensitivity")
        plt.axhline(y=benchmark_spec, color="g", linestyle="--", label="Benchmark Specificity")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Sensitivity & Specificity Tradeoff - {model_name}")
    plt.legend()
    plt.grid(True)
    #plt.show()

    sweep_df = pd.DataFrame({"Threshold": thresholds, "Sensitivity": sensitivity_list, "Specificity": specificity_list})
    if return_full_df:
        return sweep_df


def bootstrap_auc_ci(y_true, y_probs, metric="roc", n_bootstraps=5000, ci_level=95, random_state=2023):
    rng = np.random.RandomState(random_state)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        y_true_b = np.array(y_true)[indices]
        y_probs_b = np.array(y_probs)[indices]

        if len(np.unique(y_true_b)) < 2:
            continue

        if metric == "roc":
            aucs.append(roc_auc_score(y_true_b, y_probs_b))
        elif metric == "pr":
            precision, recall, _ = precision_recall_curve(y_true_b, y_probs_b)
            aucs.append(sk_auc(recall, precision))
        else:
            raise ValueError("Metric must be 'roc' or 'pr'")

    lower = np.percentile(aucs, (100 - ci_level) / 2)
    upper = np.percentile(aucs, 100 - (100 - ci_level) / 2)
    return (lower, upper)


def evaluate_model(alg, X_test, y_test, ts=None, title='Model'):
    if isinstance(alg, list):  # SNN 
        logits = np.mean([model.predict(X_test).squeeze() for model in alg], axis=0)
        probabilities = ts.predict_proba(logits) if ts else tf.sigmoid(logits).numpy()
    elif hasattr(alg, 'predict_proba'):
        probs = alg.predict_proba(X_test)
        if isinstance(probs, pd.DataFrame):
            if 1 in probs.columns:
                probabilities = probs[1].values
            else:
                probabilities = probs.iloc[:, -1].values  # fallback
        else:
            probabilities = probs[:, 1]
    else:
        logits = alg.predict(X_test).flatten()
        probabilities = ts.predict_proba(logits) if ts else tf.sigmoid(logits).numpy()

    auc = roc_auc_score(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = sk_auc(recall, precision)
    auc_ci = bootstrap_auc_ci(y_test, probabilities, metric="roc")
    pr_auc_ci = bootstrap_auc_ci(y_test, probabilities, metric="pr")

    fpr, tpr, _ = roc_curve(y_test, probabilities)


    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'auc_ci': auc_ci,
        'pr_auc_ci': pr_auc_ci,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'probabilities': probabilities,
        'y_true': y_test
    }

def plot_roc_pr_curves(*eval_models,
                       baseline,
                       labels=None,
                       title="Stroke Prediction",
                       save_name="roc_pr_comparison_colorblind.png"):

    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(eval_models)
    if labels is None:
        labels = [f"Model {i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError(
            f"labels length ({len(labels)}) must equal number of models ({n})"
        )

    colors = sns.color_palette("colorblind", n)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for res, lbl, c in zip(eval_models, labels, colors):
        ax_roc.plot(
            res['fpr'], res['tpr'], color=c,
            label=(
                f"{lbl} (AUC = {res['auc']:.3f} "
                f"[{res['auc_ci'][0]:.3f}–{res['auc_ci'][1]:.3f}])"
            ),
            linewidth=2
        )
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves')
    ax_roc.legend(fontsize=10, loc='lower right')
    ax_roc.grid(True)

    for res, lbl, c in zip(eval_models, labels, colors):
        ax_pr.plot(
            res['recall'], res['precision'], color=c,
            label=(
                f"{lbl} (PR AUC = {res['pr_auc']:.3f} "
                f"[{res['pr_auc_ci'][0]:.3f}–{res['pr_auc_ci'][1]:.3f}])"
            ),
            linewidth=2
        )
    ax_pr.hlines(
        baseline, 0, 1, colors='black', linestyles='dashed',
        label=f'Baseline (Prev = {baseline:.3f})'
    )
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision–Recall Curves')
    ax_pr.legend(fontsize=10)
    ax_pr.grid(True)

    save_path = FIGURE_DIR / save_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()



def find_best_threshold(sweep_df):
    sweep_df = sweep_df.copy()
    sweep_df["FPR"] = 1 - sweep_df["Specificity"]
    sweep_df["ROC_dist"] = np.sqrt((sweep_df["FPR"])**2 + (1 - sweep_df["Sensitivity"])**2)

    best_idx = sweep_df["ROC_dist"].idxmin()
    return float(sweep_df.loc[best_idx, "Threshold"])


def compute_operating_metrics(probs, y_true, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc  = matthews_corrcoef(y_true, y_pred)
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "mcc": mcc
    }

def plot_combined_roc(eval_dicts, save_name="combined_roc_defaultcolors.png"):
    import matplotlib.pyplot as plt
    from config.paths import FIGURE_DIR

    label_map = {
        ("rf", "stroke"): "RF Stroke",
        ("rf", "severe"): "RF Severe",
        ("xgb", "stroke"): "XGB Stroke",
        ("xgb", "severe"): "XGB Severe",
        ("snn", "stroke"): "SNN Stroke",
        ("snn", "severe"): "SNN Severe",
    }

    plt.figure(figsize=(8, 6))

    for (model_type, label_type), eval_data in eval_dicts.items():
        fpr = eval_data['fpr']
        tpr = eval_data['tpr']
        auc = eval_data['auc']
        ci_low, ci_high = eval_data['auc_ci']
        label = f"{label_map[(model_type, label_type)]} (AUC = {auc:.3f} [{ci_low:.3f}–{ci_high:.3f}])"
        plt.plot(fpr, tpr, label=label, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC-AUC Comparison", fontsize=14)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True)

    save_path = FIGURE_DIR / save_name
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
