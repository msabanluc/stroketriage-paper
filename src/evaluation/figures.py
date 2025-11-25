import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from config.config import settings
from evaluation.shap import RENAME_MAP

OUTPUT_DIR = settings['output_dir']
FIGURE_DIR = settings['figure_dir']

def load_eval(label_type, suffix, model_key, pretty):
    pkl_path = OUTPUT_DIR / f"eval_{label_type}{suffix}_{model_key}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        ev = pickle.load(f)
    ev["label"] = pretty
    return ev

def load_task(label_type, suffix="_suspected"):
    models = []
    for key, pretty_name in [("rf", "Random Forest"), ("xgb", "XGBoost"), ("snn", "SNN")]:
        try:
            models.append(load_eval(label_type, suffix, key, pretty_name))
        except FileNotFoundError as e:
            print(f"Skipping missing model: {e}")
    return models

def plot_combined_roc_pr(task_a, task_b, save_name):
    sns.set_style("whitegrid")
    colors = sns.color_palette("colorblind", max(len(task_a), 3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax_roc_a, ax_pr_a), (ax_roc_b, ax_pr_b) = axes

    def plot_pair(models, ax_roc, ax_pr):
        for ev, c in zip(models, colors):
            ax_roc.plot(ev["fpr"], ev["tpr"], lw=2, color=c,
                        label=f'{ev["label"]} (AUC = {ev["auc"]:.3f} [{ev["auc_ci"][0]:.2f}–{ev["auc_ci"][1]:.2f}])')
            ax_pr.plot(ev["recall"], ev["precision"], lw=2, color=c,
                       label=f'{ev["label"]} (PR AUC = {ev["pr_auc"]:.3f} [{ev["pr_auc_ci"][0]:.2f}–{ev["pr_auc_ci"][1]:.2f}])')

        # ROC
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random Guess")
        ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
        ax_roc.legend(loc="lower right", fontsize=9)

        # PR
        prev = models[0].get("y_true")
        baseline = (prev.mean() if prev is not None else models[0].get("baseline", 0.0))
        ax_pr.hlines(baseline, 0, 1, colors="black", linestyles="dashed",
                     label=f"Baseline (Prev = {baseline:.3f})")
        ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
        ax_pr.legend(fontsize=9)

    plot_pair(task_a, ax_roc_a, ax_pr_a)
    plot_pair(task_b, ax_roc_b, ax_pr_b)

    plt.tight_layout()
    fig.canvas.draw()

    pad = 0.02
    y_row1 = ax_roc_a.get_position().y1 + pad
    y_row2 = ax_roc_b.get_position().y1 + pad

    fig.text(0.5, y_row1, "Stroke Prediction",
             ha="center", va="bottom", fontsize=16, fontweight="bold")
    fig.text(0.5, y_row2, "Severe Stroke Prediction",
             ha="center", va="bottom", fontsize=16, fontweight="bold")

    fig.subplots_adjust(top=min(0.98, y_row1 - 0.01))

    fig.savefig(FIGURE_DIR / save_name, dpi=300, bbox_inches="tight")
    plt.show()


def merge_shap_figures(left_img_path, right_img_path, output_filename,
                       title="SHAP Feature Importance",
                       left_label="a.", right_label="b."):
    
    try:
        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)
    except FileNotFoundError as e:
        print(f"Could not merge SHAP figures: {e}")
        return

    max_height = max(left_img.height, right_img.height)
    scale_left = max_height / left_img.height
    scale_right = max_height / right_img.height

    left_img = left_img.resize((int(left_img.width * scale_left), max_height), Image.Resampling.LANCZOS)
    right_img = right_img.resize((int(right_img.width * scale_right), max_height), Image.Resampling.LANCZOS)

    title_height = 120
    total_width = left_img.width + right_img.width
    combined_img = Image.new("RGB", (total_width, max_height + title_height), "white")

    combined_img.paste(left_img, (0, title_height))
    combined_img.paste(right_img, (left_img.width, title_height))

    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
        title_font = ImageFont.truetype("arial.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    title_width = draw.textlength(title, font=title_font)
    draw.text(((total_width - title_width) / 2, 20), title, fill="black", font=title_font)
    draw.text((25, title_height + 25), left_label, fill="black", font=font)
    draw.text((left_img.width + 25, title_height + 25), right_label, fill="black", font=font)

    # === Save ===
    output_path = FIGURE_DIR / output_filename
    combined_img.save(output_path, dpi=(300, 300))
    print(f"Saved combined figure to {output_path}")

# %% Supplemental Figure S3 (Calibration Plots)

def _calculate_avg_ace(y_true, y_prob, k_percent=20, eta=10):
    l = int(len(y_true) * (k_percent / 100))
    if l <= eta:
        return np.nan 

    ace_values = []
    for k in range(eta, l + 1, eta):
        # Top k% predictions
        sorted_idx = np.argsort(-y_prob)
        top_k_idx = sorted_idx[:k]
        y_top = y_true[top_k_idx]
        p_top = y_prob[top_k_idx]

        # Dynamic binning for ACE@k
        num_bins = max(2, int(np.floor(np.log10(k))))
        bins = np.quantile(p_top, np.linspace(0, 1, num_bins + 1))
        bins[-1] = 1.0  # Ensure last bin edge is 1
        bin_idx = np.digitize(p_top, bins) - 1

        ace_k = 0
        for i in range(num_bins):
            mask = bin_idx == i
            if np.sum(mask) > 0:
                observed_freq = np.mean(y_top[mask])
                predicted_prob = np.mean(p_top[mask])
                ace_k += np.abs(predicted_prob - observed_freq) * (1 / num_bins)
        ace_values.append(ace_k)

    return np.mean(ace_values) if ace_values else np.nan


def plot_reliability_panel(ax, y_trues, model_probs, model_names, n_bins=20, strategy='quantile', title_suffix="", k_percent=20):
    for y_true, y_prob, model_name in zip(y_trues, model_probs, model_names):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        avg_ace = _calculate_avg_ace(y_true, y_prob, k_percent=k_percent)
        ax.plot(
            prob_pred, prob_true, marker='o',
            label=f'{model_name} (Avg-ACE@{k_percent}%: {avg_ace:.4f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(f'Reliability Diagram {title_suffix}')
    ax.legend()
    ax.grid()


def make_calibration_figure(stroke_eval, severe_eval, stroke_model_name="XGB", severe_model_name="RF", save_name="S3_calibration_figure.png"):
    for ev, name in [(stroke_eval, "stroke"), (severe_eval, "severe")]:
        if "probs_uncal" not in ev or "probs_cal" not in ev:
            raise KeyError(
                f"Missing 'probs_uncal' or 'probs_cal' in {name} eval pickle. "
                "Please save these during model evaluation."
            )

    y_true_stroke = np.asarray(stroke_eval["y_true"]).astype(int)
    y_true_severe = np.asarray(severe_eval["y_true"]).astype(int)

    p_stroke_uncal = np.asarray(stroke_eval["probs_uncal"])
    p_severe_uncal = np.asarray(severe_eval["probs_uncal"])
    p_stroke_cal = np.asarray(stroke_eval["probs_cal"])
    p_severe_cal = np.asarray(severe_eval["probs_cal"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Uncalibrated
    plot_reliability_panel(
        axes[0],
        [y_true_stroke, y_true_severe],
        [p_stroke_uncal, p_severe_uncal],
        [f"{stroke_model_name} – Stroke", f"{severe_model_name} – Severe Stroke"],
        title_suffix="(Uncalibrated)"
    )

    # Calibrated
    plot_reliability_panel(
        axes[1],
        [y_true_stroke, y_true_severe],
        [p_stroke_cal, p_severe_cal],
        [f"{stroke_model_name} – Stroke", f"{severe_model_name} – Severe Stroke"],
        title_suffix="(Calibrated)"
    )

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / save_name, dpi=300, bbox_inches="tight")
    print(f"Saved calibration figure to {FIGURE_DIR / save_name}")
    plt.show()


# %% Sensitivity/Specificity tradeoff figure

def _plot_tradeoff_subplot(ax, thresholds, sens, spec, title, ems_sens=None, ems_spec=None, lit_sens=None, lit_spec=None):
    ax.plot(thresholds, sens, label='ML Sensitivity', color='#1f77b4')
    ax.plot(thresholds, spec, label='ML Specificity', color='#ff7f0e')

    if ems_sens is not None:
        ax.axhline(y=ems_sens, color='#1f77b4', linestyle='--', label='EMS Sensitivity (Data)')
    if ems_spec is not None:
        ax.axhline(y=ems_spec, color='#ff7f0e', linestyle='--', label='EMS Specificity (Data)')

    if lit_sens is not None:
        ax.axhline(y=lit_sens, color='#1f77b4', linestyle=':', label='EMS Sensitivity (Literature)')
    if lit_spec is not None:
        ax.axhline(y=lit_spec, color='#ff7f0e', linestyle=':', label='EMS Specificity (Literature)')

    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)


def sens_spec_from_eval(ev, thresholds=None):
    probs = np.asarray(ev.get("probs_cal", ev.get("probabilities")), dtype=float)
    y_true = np.asarray(ev["y_true"], dtype=int)

    if thresholds is None:
        thresholds = np.linspace(0, 1, 400)

    sens, spec = [], []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return thresholds, sens, spec


def plot_threshold_tradeoff(
    stroke_all_ev, stroke_sus_ev, severe_all_ev, severe_sus_ev,
    save_name="S1_tradeoff_figure.png",
    # EMS (Data) values
    stroke_sus_sens=0.650, stroke_sus_spec=0.875,
    stroke_all_sens=0.323, stroke_all_spec=0.982,
    severe_sus_sens=None, severe_sus_spec=None,
    severe_all_sens=None, severe_all_spec=None,
):    
    data = {
        'stroke_all': sens_spec_from_eval(stroke_all_ev),
        'stroke_sus': sens_spec_from_eval(stroke_sus_ev),
        'severe_all': sens_spec_from_eval(severe_all_ev),
        'severe_sus': sens_spec_from_eval(severe_sus_ev),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    
    # Literature lines
    cp_lit_sens, cp_lit_spec = 0.811, 0.517   # CPSS literature (stroke suspected)
    van_lit_sens, van_lit_spec = 0.810, 0.380 # VAN literature (severe suspected)

    # (a) Stroke - All
    _plot_tradeoff_subplot(axes[0, 0], *data['stroke_all'], title="(a) Stroke - All",
                           ems_sens=stroke_all_sens, ems_spec=stroke_all_spec)

    # (b) Severe Stroke - All
    _plot_tradeoff_subplot(axes[0, 1], *data['severe_all'], title="(b) Severe Stroke - All",
                           ems_sens=severe_all_sens, ems_spec=severe_all_spec)

    # (c) Stroke - Suspected Only
    _plot_tradeoff_subplot(axes[1, 0], *data['stroke_sus'], title="(c) Stroke - Suspected Only",
                           ems_sens=stroke_sus_sens, ems_spec=stroke_sus_spec,
                           lit_sens=cp_lit_sens, lit_spec=cp_lit_spec)

    # (d) Severe Stroke - Suspected Only
    _plot_tradeoff_subplot(axes[1, 1], *data['severe_sus'], title="(d) Severe Stroke - Suspected Only",
                           ems_sens=severe_sus_sens, ems_spec=severe_sus_spec,
                           lit_sens=van_lit_sens, lit_spec=van_lit_spec)

    # Unified legend
    custom_lines = [
        plt.Line2D([0], [0], color='#1f77b4', linestyle='-',  label='ML Sensitivity'),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle='-',  label='ML Specificity'),
        plt.Line2D([0], [0], color='#1f77b4', linestyle='--', label='EMS Sensitivity (Data)'),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle='--', label='EMS Specificity (Data)'),
        plt.Line2D([0], [0], color='#1f77b4', linestyle=':',  label='EMS Sensitivity (Literature)'),
        plt.Line2D([0], [0], color='#ff7f0e', linestyle=':',  label='EMS Specificity (Literature)')
    ]
    fig.legend(handles=custom_lines, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Sensitivity & Specificity Tradeoff Curves", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIGURE_DIR / save_name, dpi=300, bbox_inches="tight")
    print(f"Saved tradeoff figure to {FIGURE_DIR / save_name}")
    plt.show()


if __name__ == '__main__':
    try:
        # Top models: Stroke=XGB, Severe=RF
        ev_stroke = load_eval("stroke", "_all", "xgb", "XGB")
        ev_severe = load_eval("severe", "_all", "rf", "RF")
        
        make_calibration_figure(ev_stroke, ev_severe, stroke_model_name="XGB", severe_model_name="RF")

    except (FileNotFoundError, KeyError) as e:
        print(f"Could not generate calibration figure: {e}")

    try:
        ev_stroke_all = load_eval("stroke", "_all", "xgb", "XGB")
        ev_severe_all = load_eval("severe", "_all", "rf", "RF")
        ev_stroke_suspected = load_eval("stroke", "_suspected", "xgb", "XGB")
        ev_severe_suspected = load_eval("severe", "_suspected", "rf", "RF")

        plot_threshold_tradeoff(
            ev_stroke_all, ev_stroke_suspected,
            ev_severe_all, ev_severe_suspected
        )
    except FileNotFoundError as e:
        print(f"Could not generate tradeoff figure: {e}")
