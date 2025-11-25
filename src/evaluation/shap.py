import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config.paths import FIGURE_DIR


RENAME_MAP = {
    'bps1': 'Systolic BP',
    'gcs_total': 'GCS',
    'c_weight': 'Weight',
    'bpd1': 'Diastolic BP',
    'pulse1': 'Pulse',
    'c_age': 'Age',
    'ADI_NATRANK': 'ADI',
    'spo2_1': 'O2 Saturation',
    'resprate1': 'Respiration Rate',
    'race_eth_ed1_NHBlack': 'Non-Hispanic Black',
    'race_eth_ed1_NHWhite': 'Non-Hispanic White',
    'race_eth_ed1_NHOther': 'Non-Hispanic Other',
    'race_eth_ed1_Hispanic': 'Hispanic',
    'c_sex1_Female': 'Female',
    'c_sex1_Male': 'Male',
    'natureofcall_Stroke/CVA': 'Nature of Call: Stroke or CVA',
    'natureofcall_Breathing Problems': 'Nature of Call: Breathing Problems',
    'natureofcall_Other/Medical': 'Nature of Call: Other or Medical',
    'natureofcall_Chest Pain': 'Nature of Call: Chest Pain',
    'natureofcall_Transfer/Interfacility/Palliative Care': 'Nature of Call: Transfer or Palliative Care',
    'natureofcall_Headache': 'Nature of Call: Headache',
    'natureofcall_Sick Person': 'Nature of Call: Sick Person',
    'natureofcall_Fall(s)': 'Nature of Call: Fall',
    'natureofcall_MVC': 'Nature of Call: Motor Vehicle Collision',
    'natureofcall_Abdominal Pain': 'Nature of Call: Abdominal Pain',
    'natureofcall_Well Being Check': 'Nature of Call: Well Being Check',
    'natureofcall_Cardiac/Resp Arrest': 'Nature of Call: Cardiac or Respiratory Arrest',
    'natureofcall_Seizures': 'Nature of Call: Seizures',
    'respeffort_Labored': 'Labored Respiratory Effort',
    'respeffort_Normal': 'Normal Respiratory Effort',
    'pulsestrength_Strong': 'Strong Pulse Strength',
    'calltype_2.0': 'Call Type 2',
    'calltype_16.0': 'Call Type 16',
    'calltype_1.0': 'Call Type 1'
}

def plot_shap_summary(model, X_test_enc, model_name: str, filename: str):
    X_test_enc_renamed = X_test_enc.rename(columns=RENAME_MAP)

    low = filename.lower()
    if any(k in low for k in ["severe", "crit", "critical"]):
        task_label = "Severe Stroke Classification"
    else:
        task_label = "Stroke Classification"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_enc)

    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values


    shap.summary_plot(shap_vals_to_plot, X_test_enc_renamed, show=False)
    plt.gcf().suptitle(f'{model_name} {task_label}', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = FIGURE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')