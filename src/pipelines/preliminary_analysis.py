import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, norm
from scipy import stats
import math
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import pickle

from config.config import settings
from data.preprocessing import (get_cpstroke_subset, get_stroke_relevant_subset, load_and_clean_data)
from evaluation.performance import compute_operating_metrics

STROKE_CVA_CODES = {35, 36}
STROKE_RELATED_CODES = {13, 19, 20, 58, 61}
CPSS_NA = {1, 2, 3}
CPSS_ABNORMAL = {4, 16, 5}
CPSS_NORMAL = {6, 14}
CPSS_DOC_VALUES = CPSS_ABNORMAL | CPSS_NORMAL

def cp_status(code):
    if pd.isna(code): return "null"
    try: c = int(code)
    except: return "other"
    if c in CPSS_NA: return "null"
    if c in CPSS_ABNORMAL: return "abnormal"
    if c in CPSS_NORMAL: return "normal"
    return "other"

def dispatch_group(code):
    if pd.isna(code): return "non-stroke"
    try: c = int(code)
    except: return "non-stroke"
    if c in STROKE_CVA_CODES: return "Stroke/CVA"
    if c in STROKE_RELATED_CODES: return "stroke-related"
    return "non-stroke"

def _fmt_number(x, digits=1):
    if x is None or pd.isna(x): return ""
    try:
        fx = float(x)
        if np.isfinite(fx) and abs(fx - round(fx)) < 1e-9:
            return f"{int(round(fx))}"
        else:
            return f"{fx:.{digits}f}"
    except: return str(x)

def _series_is_integerish(s):
    vals = pd.to_numeric(s.dropna(), errors='coerce').values
    if vals.size == 0: return True
    return np.all(np.isfinite(vals) & (np.mod(vals, 1) == 0))

def _percent(n, d, digits=1):
    if d == 0: return "0.0"
    return f"{100.0 * n / d:.{digits}f}"

def build_continuous_table(df, cols, name_map=None, default_digits=1):
    rows = []
    total = len(df)
    for col in cols:
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors='coerce')
        n_nonmiss = s.notna().sum()
        miss_pct = _percent(total - n_nonmiss, total, digits=1)
        clean = s.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            med = q1 = q3 = minv = maxv = None
        else:
            med = clean.median()
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            minv = clean.min()
            maxv = clean.max()
        digits = 0 if _series_is_integerish(clean) else default_digits
        med_q = f"{_fmt_number(med, digits)} [{_fmt_number(q1, digits)}, {_fmt_number(q3, digits)}]" if med is not None else ""
        rng = f"{_fmt_number(minv, digits)}–{_fmt_number(maxv, digits)}" if minv is not None else ""
        label = name_map.get(col, col) if name_map else col
        rows.append({
            "Variable": label,
            "N (non-missing)": int(n_nonmiss),
            "Missing (%)": float(miss_pct),
            "Median [Q1, Q3]": med_q,
            "Range": rng
        })
    out = pd.DataFrame(rows)
    order = [name_map.get(c, c) if name_map else c for c in cols]
    out["__ord__"] = out["Variable"].apply(lambda x: order.index(x) if x in order else 1e9)
    out = out.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)
    return out

def get_summary(df, patient_level=False):
    if patient_level:
        df_sorted = df.assign(sort_key=df['any_stroke'].map({1: 0, 0: 1})).sort_values(['mrn', 'sort_key', 'tdate'])
        df = df_sorted.drop_duplicates('mrn', keep='first')
    N = len(df)
    summary = {"N": N}
    for col, name in [('age_ed', 'Age'), ('ht_ed', 'Height (in)'), ('wt_ed', 'Weight (lb)')]:
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        summary[name] = f"{median:.0f} [{q1:.0f}, {q3:.0f}]"
    median = df['bmi_ed'].median()
    q1 = df['bmi_ed'].quantile(0.25)
    q3 = df['bmi_ed'].quantile(0.75)
    summary['BMI (kg/m²)'] = f"{median:.1f} [{q1:.1f}, {q3:.1f}]"
    gender_counts = df['gender_ed'].value_counts()
    summary['Female'] = f"{gender_counts.get('F', 0)} ({gender_counts.get('F', 0)/N*100:.1f}%)"
    summary['Male'] = f"{gender_counts.get('M', 0)} ({gender_counts.get('M', 0)/N*100:.1f}%)"
    race_map = {1: 'Non-Hispanic White', 2: 'Non-Hispanic Black', 3: 'Non-Hispanic Other', 4: 'Hispanic'}
    race_counts = df['race_eth_ed1'].value_counts()
    for race_key, race_name in race_map.items():
        count = race_counts.get(race_key, 0)
        summary[race_name] = f"{count} ({count/N*100:.1f}%)"
    stroke_counts = df['any_stroke'].value_counts()
    summary['Stroke: Yes'] = f"{stroke_counts.get(1, 0)} ({stroke_counts.get(1, 0)/N*100:.1f}%)"
    summary['Stroke: No'] = f"{stroke_counts.get(0, 0)} ({stroke_counts.get(0, 0)/N*100:.1f}%)"
    stroke_type_map = {'Cerebral Infarction': 'Cerebral infarction', 'SAH': 'Subarachnoid hemorrhage', 'ICH': 'Intracerebral hemorrhage'}
    stroke_type_counts = df.loc[df['any_stroke'] == 1, 'stroke_type'].value_counts()
    total_strokes = stroke_type_counts.sum()
    for type_key, type_name in stroke_type_map.items():
        count = stroke_type_counts.get(type_key, 0)
        summary[type_name] = f"{count} ({count/total_strokes*100:.1f}%)" if total_strokes > 0 else "0 (0.0%)"
    return summary

def fmt_med_iqr(series, fmt=".1f"):
    s = series.dropna()
    if len(s) == 0: return "-"
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return f"{med:{fmt}} [{q1:{fmt}}, {q3:{fmt}}]"

def fmt_n_pct(n, total):
    if total == 0: return "0 (0.0%)"
    return f"{n} ({100 * n / total:.1f}%)"

def get_pval_cont(data, var, group_col='any_stroke'):
    g0 = data.loc[data[group_col] == 0, var].dropna()
    g1 = data.loc[data[group_col] == 1, var].dropna()
    if len(g0) == 0 or len(g1) == 0: return "-"
    stat, p = mannwhitneyu(g0, g1, alternative='two-sided')
    return f"{p:.2f}" if p >= 0.01 else "<0.01*"

def get_pval_cat(data, var, group_col='any_stroke'):
    ct = pd.crosstab(data[var], data[group_col], dropna=False)
    if ct.empty: return "-"
    chi2, p, dof, ex = chi2_contingency(ct)
    return f"{p:.2f}" if p >= 0.01 else "<0.01*"

def get_spearman_stats(df, var1, var2, alpha=0.05):
    tmp = df[[var1, var2]].dropna()
    n = len(tmp)
    if n < 3: return n, "-", "-"
    rho, pval = stats.spearmanr(tmp[var1], tmp[var2])
    if abs(rho) < 1.0:
        z = 0.5 * math.log((1 + rho) / (1 - rho))
        se = 1.0 / math.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        lower_z = z - z_crit * se
        upper_z = z + z_crit * se
        lower_rho = (math.exp(2*lower_z) - 1) / (math.exp(2*lower_z) + 1)
        upper_rho = (math.exp(2*upper_z) - 1) / (math.exp(2*upper_z) + 1)
    else:
        lower_rho, upper_rho = rho, rho
    rho_str = f"{rho:.2f} ({lower_rho:.2f}, {upper_rho:.2f})"
    pval_str = "<0.01" if pval < 0.01 else f"{pval:.2f}"
    return n, rho_str, pval_str

def format_pval(p):
    if p < 0.01: return "<0.01"
    return f"{p:.2f}"

def run_gee_model(data, outcome, pre_var, time_var, label, scale_time=False):
    df_mod = data.dropna(subset=[outcome, pre_var, time_var]).copy()
    df_mod['y'] = df_mod[outcome]
    df_mod['x'] = df_mod[pre_var]
    if scale_time: df_mod['time'] = df_mod[time_var] / 10.0
    else: df_mod['time'] = df_mod[time_var]
    model = smf.gee("y ~ x + time", groups="mrn", data=df_mod, family=sm.families.Gaussian())
    result = model.fit()
    params = result.params
    conf_int = result.conf_int()
    pvals = result.pvalues
    def fmt_est_ci(param_name):
        if param_name in params.index:
            est = params[param_name]
            lower, upper = conf_int.loc[param_name]
            return f"{est:.1f} ({lower:.1f}, {upper:.1f})"
        return "NA"
    def get_p(param_name):
        if param_name in pvals.index: return format_pval(pvals[param_name])
        return "NA"
    return {
        "Measure": label,
        "N": f"{int(result.nobs):,}",
        "Intercept estimate (95% CI)": fmt_est_ci('Intercept'),
        "p-value (Int)": get_p('Intercept'),
        "Change in ED estimate per increase in prehospital measurement (95% CI)": fmt_est_ci('x'),
        "p-value (Pre)": get_p('x'),
        "Change in ED estimate per 10-minute increase in transit time (95% CI)": fmt_est_ci('time'),
        "p-value (Time)": get_p('time')
    }

def categorize_cp(x):
    if pd.isna(x): return 'NaN'
    if x in CPSS_NORMAL: return 'Normal'
    if x in CPSS_ABNORMAL: return 'Abnormal'
    if x in [5, 15]: return 'Non-Conclusive'
    return 'Not Applicable/Known/Available'

def save_table(df, filename, index=True):
    out_path = settings['output_dir'] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if filename.endswith('.csv'):
        df.to_csv(out_path, index=index)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(df.to_string(index=index))
    print(f"Saved table to {out_path}")

# Main Functions

def load_and_preprocess_data():
    print("Loading and preprocessing data for preliminary analysis...")
    
    df, _, _, _ = load_and_clean_data(mode="full")
    
    # Set memberships
    df = df.copy()
    df["is_stroke"]  = df["critical_stroke_cat"].isin([1, 2])
    df["is_severe"]  = df["critical_stroke_cat"].eq(1)
    df["cp_status"]       = df["stroke_scale"].apply(cp_status)
    df["cp_documented"]   = df["cp_status"].isin(["abnormal", "normal"])
    df["cp_abnormal"]     = df["cp_status"].eq("abnormal")
    df["cp_normal"]       = df["cp_status"].eq("normal")
    df["dispatch_group"]      = df["natureofcall"].apply(dispatch_group)
    df["dispatch_strokelike"] = df["dispatch_group"].isin(["Stroke/CVA", "stroke-related"])
    df["dispatch_cva"]        = df["dispatch_group"].eq("Stroke/CVA")
    df["dispatch_related"]    = df["dispatch_group"].eq("stroke-related")
    df["in_cpss_subset"]        = df["cp_documented"]
    df["in_stroke_mimic_subset"] = df["is_stroke"] | df["cp_documented"] | df["dispatch_strokelike"]
    
    cat_map = {1: 'Severe Stroke', 2: 'Non-Severe Stroke', 3: 'Non-Stroke'}
    df['true_label'] = df['critical_stroke_cat'].map(cat_map)
    df['true_stroke'] = df['true_label'] != 'Non-Stroke'
    
    df_suspected = get_cpstroke_subset(df)
    df_stroke = get_stroke_relevant_subset(df)
    
    return df, df_suspected, df_stroke

def generate_table1(df):
    print("\n--- Generating Table 1 ---")
    patient_summary = get_summary(df, patient_level=True)
    encounter_summary = get_summary(df, patient_level=False)
    table1_data = {
        "Characteristic": [
            "Age, median [Q1, Q3]", "Gender, N (%)", "  Female", "  Male",
            "Race/Ethnicity, N (%)", "  Non-Hispanic White", "  Non-Hispanic Black",
            "  Non-Hispanic Other", "  Hispanic", "Height (in), median [Q1, Q3]",
            "Weight (lb), median [Q1, Q3]", "BMI (kg/m²), median [Q1, Q3]",
            "Stroke, N (%)", "  Yes", "  No", "Type of stroke, N (%)",
            "  Cerebral infarction", "  Subarachnoid hemorrhage", "  Intracerebral hemorrhage"
        ],
        f"All patients (N={patient_summary['N']})": [
            patient_summary.get('Age'), "", patient_summary.get('Female'), patient_summary.get('Male'),
            "", patient_summary.get('Non-Hispanic White'), patient_summary.get('Non-Hispanic Black'),
            patient_summary.get('Non-Hispanic Other'), patient_summary.get('Hispanic'),
            patient_summary.get('Height (in)'), patient_summary.get('Weight (lb)'), patient_summary.get('BMI (kg/m²)'),
            "", patient_summary.get('Stroke: Yes'), patient_summary.get('Stroke: No'),
            "", patient_summary.get('Cerebral infarction'), patient_summary.get('Subarachnoid hemorrhage'),
            patient_summary.get('Intracerebral hemorrhage')
        ],
        f"All encounters (N={encounter_summary['N']})": [
            encounter_summary.get('Age'), "", encounter_summary.get('Female'), encounter_summary.get('Male'),
            "", encounter_summary.get('Non-Hispanic White'), encounter_summary.get('Non-Hispanic Black'),
            encounter_summary.get('Non-Hispanic Other'), encounter_summary.get('Hispanic'),
            encounter_summary.get('Height (in)'), encounter_summary.get('Weight (lb)'), encounter_summary.get('BMI (kg/m²)'),
            "", encounter_summary.get('Stroke: Yes'), encounter_summary.get('Stroke: No'),
            "", encounter_summary.get('Cerebral infarction'), encounter_summary.get('Subarachnoid hemorrhage'),
            encounter_summary.get('Intracerebral hemorrhage')
        ]
    }
    table1 = pd.DataFrame(table1_data).set_index("Characteristic")
    print(table1)
    save_table(table1, "table1_demographics.csv")
    return table1

def generate_table2(df):
    print("\n--- Generating Table 2 ---")
    display_name = {
        'c_age': 'Age (years)', 'c_weight': 'Weight (kg)', 'pulse1': 'Heart rate (bpm)',
        'resprate1': 'Respiratory rate', 'bps1': 'Systolic blood pressure', 'bpd1': 'Diastolic blood pressure',
        'spo2_1': 'SpO₂ (%)', 'temperature1': 'Temperature (°F)', 'glucose1': 'Glucose (mg/dL)',
        'gcs_total': 'Glasgow Coma Score', 'ADI_NATRANK': 'Area Deprivation Index (national rank)',
        'c_sex1': 'Sex', 'race_eth_ed1': 'Race/Ethnicity (ED)', 'pulsestrength': 'Pulse strength',
        'respeffort': 'Respiratory effort', 'calltype': 'Call type', 'priority': 'EMS priority',
        'transpriority': 'Transport priority', 'natureofcall': 'Dispatch nature of call'
    }
    table2_pairs = [
        ("pulse1", "hr_ed", display_name.get("pulse1", "Heart rate (bpm)")),
        ("bps1", "sbp_ed", display_name.get("bps1", "Systolic blood pressure")),
        ("bpd1", "dbp_ed", display_name.get("bpd1", "Diastolic blood pressure")),
        ("resprate1", "rr_ed", display_name.get("resprate1", "Respiratory rate")),
        ("spo2_1", "spo2_ed", display_name.get("spo2_1", "SpO₂ (%)")),
        ("temperature1", "temp_ed", display_name.get("temperature1", "Temperature (°F)")),
        ("glucose1", "glucose_ed", display_name.get("glucose1", "Glucose (mg/dL)")),
        ("gcs_total", "gcs_ed", display_name.get("gcs_total", "Glasgow Coma Score")),
    ]
    pre_cols = [p for (p, _, _) in table2_pairs]
    ed_cols  = [e for (_, e, _) in table2_pairs]
    ed_display_map = {e: label for (_, e, label) in table2_pairs}
    pre_cont = build_continuous_table(df, pre_cols, name_map=display_name, default_digits=1)
    ed_cont  = build_continuous_table(df, ed_cols,  name_map=ed_display_map, default_digits=1)
    pre_view = pre_cont[["Variable", "N (non-missing)", "Median [Q1, Q3]", "Range"]].set_index("Variable")
    ed_view  = ed_cont [["Variable", "N (non-missing)", "Median [Q1, Q3]", "Range"]].set_index("Variable")
    pre_view.columns = pd.MultiIndex.from_product([["Initial prehospital measures"], pre_view.columns])
    ed_view.columns  = pd.MultiIndex.from_product([["ED measures"], ed_view.columns])
    row_order = [label for (_, _, label) in table2_pairs]
    table2 = (pd.concat([pre_view, ed_view], axis=1).reindex(row_order))
    print(table2)
    save_table(table2, "table2_prehospital_vs_ed.csv")
    return table2

def generate_supplemental_table2(df):
    print("\n--- Generating Supplemental Table 2 ---")
    df_sorted = df.assign(sort_key=df['any_stroke'].map({1: 0, 0: 1})).sort_values(['mrn', 'sort_key', 'tdate'])
    demo = df_sorted.drop_duplicates('mrn', keep='first').copy()
    pat_stroke = demo[demo['any_stroke'] == 1]
    pat_nonstroke = demo[demo['any_stroke'] == 0]
    enc_stroke = df[df['any_stroke'] == 1]
    enc_nonstroke = df[df['any_stroke'] == 0]
    
    rows = []
    rows.append({
        "Demographics": "Age",
        "Stroke patients": fmt_med_iqr(pat_stroke['age_ed'], ".0f"),
        "Non-stroke patients": fmt_med_iqr(pat_nonstroke['age_ed'], ".0f"),
        "p-val (Pat)": get_pval_cont(demo, 'age_ed'),
        "Stroke encounters": fmt_med_iqr(enc_stroke['age_ed'], ".0f"),
        "Non-stroke encounters": fmt_med_iqr(enc_nonstroke['age_ed'], ".0f"),
        "p-val (Enc)": get_pval_cont(df, 'age_ed')
    })
    p_gender_pat = get_pval_cat(demo, 'gender_ed')
    p_gender_enc = get_pval_cat(df, 'gender_ed')
    rows.append({"Demographics": "Gender, N (%)", "p-val (Pat)": p_gender_pat, "p-val (Enc)": p_gender_enc})
    gender_map = [('Female', 'F'), ('Male', 'M')]
    for label, val in gender_map:
        n_pat_s = (pat_stroke['gender_ed'] == val).sum()
        n_pat_ns = (pat_nonstroke['gender_ed'] == val).sum()
        n_enc_s = (enc_stroke['gender_ed'] == val).sum()
        n_enc_ns = (enc_nonstroke['gender_ed'] == val).sum()
        rows.append({
            "Demographics": f"  {label}",
            "Stroke patients": fmt_n_pct(n_pat_s, len(pat_stroke)),
            "Non-stroke patients": fmt_n_pct(n_pat_ns, len(pat_nonstroke)),
            "Stroke encounters": fmt_n_pct(n_enc_s, len(enc_stroke)),
            "Non-stroke encounters": fmt_n_pct(n_enc_ns, len(enc_nonstroke)),
        })
    p_race_pat = get_pval_cat(demo, 'race_eth_ed1')
    p_race_enc = get_pval_cat(df, 'race_eth_ed1')
    rows.append({"Demographics": "Race/Ethnicity, N (%)", "p-val (Pat)": p_race_pat, "p-val (Enc)": p_race_enc})
    race_map = [('Non-Hispanic White', 1), ('Non-Hispanic Black', 2), ('Non-Hispanic Other', 3), ('Hispanic', 4)]
    for label, val in race_map:
        n_pat_s = (pat_stroke['race_eth_ed1'] == val).sum()
        n_pat_ns = (pat_nonstroke['race_eth_ed1'] == val).sum()
        n_enc_s = (enc_stroke['race_eth_ed1'] == val).sum()
        n_enc_ns = (enc_nonstroke['race_eth_ed1'] == val).sum()
        rows.append({
            "Demographics": f"  {label}",
            "Stroke patients": fmt_n_pct(n_pat_s, len(pat_stroke)),
            "Non-stroke patients": fmt_n_pct(n_pat_ns, len(pat_nonstroke)),
            "Stroke encounters": fmt_n_pct(n_enc_s, len(enc_stroke)),
            "Non-stroke encounters": fmt_n_pct(n_enc_ns, len(enc_nonstroke)),
        })
    vars_config = [("Height (in)", "ht_ed", ".0f"), ("Weight (lb)", "wt_ed", ".1f"), ("BMI (kg/m²)", "bmi_ed", ".1f")]
    for label, col, fmt in vars_config:
        rows.append({
            "Demographics": label,
            "Stroke patients": fmt_med_iqr(pat_stroke[col], fmt),
            "Non-stroke patients": fmt_med_iqr(pat_nonstroke[col], fmt),
            "p-val (Pat)": get_pval_cont(demo, col),
            "Stroke encounters": fmt_med_iqr(enc_stroke[col], fmt),
            "Non-stroke encounters": fmt_med_iqr(enc_nonstroke[col], fmt),
            "p-val (Enc)": get_pval_cont(df, col)
        })
    rows.append({
        "Demographics": "Prehospital measurements",
        "Stroke patients": fmt_med_iqr(pat_stroke['prehosp_vitals_tot_count'], ".0f"),
        "Non-stroke patients": fmt_med_iqr(pat_nonstroke['prehosp_vitals_tot_count'], ".0f"),
        "p-val (Pat)": get_pval_cont(demo, 'prehosp_vitals_tot_count'),
        "Stroke encounters": "-", "Non-stroke encounters": "-", "p-val (Enc)": "-"
    })
    table_s2 = pd.DataFrame(rows)
    cols = ["Demographics", "Stroke patients", "Non-stroke patients", "p-val (Pat)", "Stroke encounters", "Non-stroke encounters", "p-val (Enc)"]
    table_s2 = table_s2[cols].fillna("")
    table_s2.columns = ["Demographics", f"Stroke patients (N={len(pat_stroke)})", f"Non-stroke patients (N={len(pat_nonstroke)})", "p-value", f"Stroke encounters (N={len(enc_stroke)})", f"Non-stroke encounters (N={len(enc_nonstroke)})", "p-value"]
    print(table_s2.to_string(index=False))
    save_table(table_s2, "supplemental_table2_demographics_expanded.csv", index=False)
    return table_s2

def generate_supplemental_table3(df):
    print("\n--- Generating Supplemental Table 3 ---")
    pairs_map = [
        ('pulse1', 'hr_ed', 'Heart rate'), ('bps1', 'sbp_ed', 'Systolic blood pressure'),
        ('bpd1', 'dbp_ed', 'Diastolic blood pressure'), ('resprate1', 'rr_ed', 'Respiratory rate'),
        ('spo2_1', 'spo2_ed', 'Oxygen saturation'), ('temperature1', 'temp_ed', 'Temperature'),
        ('glucose1', 'glucose_ed', 'Glucose'), ('gcs_total', 'gcs_ed', 'Glasgow Coma Scale')
    ]
    data_rows = []
    for pre, ed, label in pairs_map:
        n_all, rho_all, p_all = get_spearman_stats(df, pre, ed)
        n_str, rho_str, p_str = get_spearman_stats(df[df['any_stroke'] == 1], pre, ed)
        data_rows.append([label, n_all, rho_all, p_all, n_str, rho_str, p_str])
    cols = ["Measure", "N_all", "Rho_all", "P_all", "N_str", "Rho_str", "P_str"]
    table_s3 = pd.DataFrame(data_rows, columns=cols).set_index("Measure")
    header_all = "Prehospital vs. ED measures\n(All encounters)"
    header_str = "Prehospital vs. ED measures\n(Any Stroke)"
    columns_mi = pd.MultiIndex.from_tuples([
        (header_all, "N"), (header_all, "ρ (95% CI)"), (header_all, "p-value"),
        (header_str, "N"), (header_str, "ρ (95% CI)"), (header_str, "p-value"),
    ])
    table_s3.columns = columns_mi
    print(table_s3)
    save_table(table_s3, "supplemental_table3_spearman.csv")
    return table_s3

def generate_supplemental_tables_4_and_5(df):
    print("\n--- Generating Supplemental Tables 4 and 5 ---")
    gee_vars = [
        ("Heart rate", "hr_ed", "pulse1", "hr_timediff"),
        ("Systolic blood pressure", "sbp_ed", "bps1", "bp_timediff"),
        ("Diastolic blood pressure","dbp_ed", "bpd1", "bp_timediff"),
        ("Respiratory rate", "rr_ed", "resprate1", "rr_timediff"),
        ("Oxygen saturation", "spo2_ed", "spo2_1", "spo2_timediff"),
        ("Temperature", "temp_ed", "temperature1", "temp_timediff"),
        ("Glucose", "glucose_ed", "glucose1", "glu_timediff"),
        ("Glasgow Coma Scale", "gcs_ed", "gcs_total", "gcs_timediff")
    ]
    s4_rows = []
    for label, outcome, pre_var, time_var in gee_vars:
        res = run_gee_model(data=df, outcome=outcome, pre_var=pre_var, time_var=time_var, label=label, scale_time=True)
        s4_rows.append(res)
    table_s4 = pd.DataFrame(s4_rows).set_index("Measure")
    print("\nSupplementary Table S4")
    print(table_s4.to_string())
    save_table(table_s4, "supplemental_table4_gee_all.csv")

    stroke_df = df[df['any_stroke'] == 1]
    s5_rows = []
    for label, outcome, pre_var, time_var in gee_vars:
        res = run_gee_model(data=stroke_df, outcome=outcome, pre_var=pre_var, time_var=time_var, label=label, scale_time=True)
        s5_rows.append(res)
    table_s5 = pd.DataFrame(s5_rows).set_index("Measure")
    print("\nSupplementary Table S5")
    print(table_s5.to_string())
    save_table(table_s5, "supplemental_table5_gee_stroke.csv")
    return table_s4, table_s5

def generate_supplemental_table6(df):
    print("\n--- Generating Supplemental Table 6 ---")
    df['cp_category'] = df['stroke_scale'].apply(categorize_cp)
    table_s6 = pd.crosstab(index=df['true_label'], columns=df['cp_category']).reindex(
        index=['Non-Stroke','Non-Severe Stroke','Severe Stroke'],
        columns=['Normal','Abnormal','Non-Conclusive','Not Applicable/Known/Available','NaN'],
        fill_value=0
    )
    print(table_s6)
    save_table(table_s6, "supplemental_table6_cp_category.csv")
    return table_s6

def generate_supplemental_table7(df, df_suspected):
    print("\n--- Generating Supplemental Table 7 ---")
    df['predicted_stroke'] = df['cp_category'].isin(['Abnormal','Non-Conclusive'])
    cm_all = pd.crosstab(index=df['true_stroke'], columns=df['predicted_stroke']).rename(
        index={False:'True Non-Stroke', True:'True Stroke'},
        columns={False:'Predicted Non-Stroke', True:'Predicted Stroke'}
    )
    print(f"\nSupplementary Table S7a — All Encounters (n={len(df)})")
    print(cm_all)
    save_table(cm_all, "supplemental_table7a_confusion_matrix_all.csv")

    df_suspected = df_suspected.copy()
    df_suspected['cp_category'] = df_suspected['stroke_scale'].apply(categorize_cp)
    cat_map = {1: 'Severe Stroke', 2: 'Non-Severe Stroke', 3: 'Non-Stroke'}
    df_suspected['true_label'] = df_suspected['critical_stroke_cat'].map(cat_map)
    df_suspected['true_stroke'] = df_suspected['true_label'] != 'Non-Stroke'
    df_suspected['predicted_stroke'] = df_suspected['cp_category'].isin(['Abnormal','Non-Conclusive'])
    cm_sub = pd.crosstab(index=df_suspected['true_stroke'], columns=df_suspected['predicted_stroke']).rename(
        index={False:'True Non-Stroke', True:'True Stroke'},
        columns={False:'Predicted Non-Stroke (CPSS subset)', True:'Predicted Stroke (CPSS subset)'}
    )
    print(f"\nSupplementary Table S7b — CPSS Subset (n={len(df_suspected)})")
    print(cm_sub)
    save_table(cm_sub, "supplemental_table7b_confusion_matrix_cpss.csv")
    return cm_all, cm_sub

def generate_figure2(df):
    print("\n--- Generating Figure 2 ---")
    var_pairs = [
        ('pulse1', 'hr_ed', 'Heart Rate (bpm), Prehospital', 'Heart Rate (bpm), ED', 'Heart Rate'),
        ('bps1', 'sbp_ed', 'SBP (mmHg), Prehospital', 'SBP (mmHg), ED', 'Systolic Blood Pressure'),
        ('bpd1', 'dbp_ed', 'DBP (mmHg), Prehospital', 'DBP (mmHg), ED', 'Diastolic Blood Pressure'),
        ('glucose1', 'glucose_ed', 'Glucose, Prehospital', 'Glucose, ED', 'Glucose')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, (xcol, ycol, xlabel, ylabel, plot_title) in zip(axes, var_pairs):
        non_stroke = df[df['any_stroke'] == 0].dropna(subset=[xcol, ycol])
        ax.scatter(non_stroke[xcol], non_stroke[ycol], marker='o', facecolors='none', edgecolors='#6c7fa1', alpha=1, zorder=3, s=26)
        if len(non_stroke) > 1:
            slope_ns, intercept_ns = np.polyfit(non_stroke[xcol], non_stroke[ycol], 1)
            x_ns = np.linspace(non_stroke[xcol].min(), non_stroke[xcol].max(), 100)
            y_ns = slope_ns * x_ns + intercept_ns
            ax.plot(x_ns, y_ns, color='#020202', linewidth=2, zorder=4)
        stroke = df[df['any_stroke'] == 1].dropna(subset=[xcol, ycol])
        ax.scatter(stroke[xcol], stroke[ycol], marker='x', color='#a9634d', alpha=1, zorder=3, s=26)
        if len(stroke) > 1:
            slope_s, intercept_s = np.polyfit(stroke[xcol], stroke[ycol], 1)
            x_s = np.linspace(stroke[xcol].min(), stroke[xcol].max(), 100)
            y_s = slope_s * x_s + intercept_s
            ax.plot(x_s, y_s, color='#eb271c', linewidth=2, zorder=4)
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], color='#a9aaa9', linestyle='--', zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{plot_title}: Prehospital vs. ED")
    plt.tight_layout()
    save_path = settings['figure_dir'] / "fig2_prehospital_vs_ed.png"
    fig.savefig(save_path, dpi=300)
    print(f"Figure 2 saved to {save_path}")

def generate_dispatch_performance(df, df_suspected):
    print("\n--- Generating Dispatch Performance Metrics ---")
    
    if 'predicted_stroke' not in df_suspected.columns:
        df_suspected = df_suspected.copy()
        df_suspected['cp_category'] = df_suspected['stroke_scale'].apply(categorize_cp)
        df_suspected['predicted_stroke'] = df_suspected['cp_category'].isin(['Abnormal','Non-Conclusive'])

    df['pred_dispatch_stroke'] = (df['natureofcall'] == 35)
    cm_dispatch = pd.crosstab(index=df['true_stroke'], columns=df['pred_dispatch_stroke']).rename(
        index={False:'True Non-Stroke', True:'True Stroke'},
        columns={False:'Dispatch Pred Non-Stroke', True:'Dispatch Pred Stroke'}
    )
    print(f"\nDispatch Performance — All Encounters (n={len(df)})")
    print(cm_dispatch)
    save_table(cm_dispatch, "dispatch_performance_cm.csv")
    
    def metrics_from_cm(cm):
        tn = cm.iloc[0,0]
        fp = cm.iloc[0,1]
        fn = cm.iloc[1,0]
        tp = cm.iloc[1,1]
        sens = tp/(tp+fn) if (tp+fn)>0 else np.nan
        spec = tn/(tn+fp) if (tn+fp)>0 else np.nan
        ppv  = tp/(tp+fp) if (tp+fp)>0 else np.nan
        return sens, spec, ppv

    disp_sens, disp_spec, disp_ppv = metrics_from_cm(cm_dispatch)
    print(f"Dispatch Sensitivity: {disp_sens:.3f}, Specificity: {disp_spec:.3f}, PPV: {disp_ppv:.3f}")

    df_disp = df[df['pred_dispatch_stroke']]
    cm_ems_disp = pd.crosstab(index=df_disp['true_stroke'], columns=df_disp['predicted_stroke']).rename(
        index={False:'True Non-Stroke', True:'True Stroke'},
        columns={False:'EMS Pred Non-Stroke', True:'EMS Pred Stroke'}
    )
    print(f"\nEMS Performance within Dispatch‐Positive (n={len(df_disp)})")
    print(cm_ems_disp)
    save_table(cm_ems_disp, "ems_performance_within_dispatch_cm.csv")
    ems_sens, ems_spec, ems_ppv = metrics_from_cm(cm_ems_disp)
    print(f"EMS Sensitivity: {ems_sens:.3f}, Specificity: {ems_spec:.3f}, PPV: {ems_ppv:.3f}")

    df_sus_disp = df_suspected[df_suspected['natureofcall'] == 35]
    cm_cpss_disp = pd.crosstab(index=df_sus_disp['true_stroke'], columns=df_sus_disp['predicted_stroke']).rename(
        index={False:'True Non-Stroke', True:'True Stroke'},
        columns={False:'CPSS Pred Non-Stroke', True:'CPSS Pred Stroke'}
    )
    print(f"\nCPSS Performance within Dispatch‐Positive (n={len(df_sus_disp)})")
    print(cm_cpss_disp)
    save_table(cm_cpss_disp, "cpss_performance_within_dispatch_cm.csv")
    cpss_sens, cpss_spec, cpss_ppv = metrics_from_cm(cm_cpss_disp)
    print(f"CPSS Sensitivity: {cpss_sens:.3f}, Specificity: {cpss_spec:.3f}, PPV: {cpss_ppv:.3f}")

def generate_threshold_metrics():
    print("\n--- Generating Threshold-based Model Metrics ---")
    OUTPUT_DIR = settings['output_dir']
    SUFFIX     = "_all"
    MODELS     = ["rf","xgb","snn"]
    TASKS      = {"stroke":"stroke_label","severe":"critical_label"}
    thresholds = np.arange(0.005, 0.5001, 0.005)
    
    for task, label_col in TASKS.items():
        for model in MODELS:
            pkl_path = OUTPUT_DIR / f"eval_{task}{SUFFIX}_{model}.pkl"
            if not pkl_path.exists():
                print(f"Skipping {task} {model} (file not found: {pkl_path})")
                continue
            with open(pkl_path, "rb") as f:
                ev = pickle.load(f)
            probs  = ev["probabilities"]
            y_true = ev["y_true"]
            records = []
            for thr in thresholds:
                m = compute_operating_metrics(probs, y_true, thr)
                records.append({
                    "Threshold":   thr,
                    "Sensitivity": m["sensitivity"],
                    "Specificity": m["specificity"],
                    "PPV":         m["ppv"]
                })
            df = pd.DataFrame(records).round(3)
            print(f"Generated metrics for {task} {model} ({df.shape[0]} rows)")
            save_table(df, f"threshold_metrics_{task}_{model}.csv", index=False)

def run_all_preliminary_analysis():
    df, df_suspected, df_stroke = load_and_preprocess_data()
    generate_table1(df)
    generate_table2(df)
    generate_supplemental_table2(df)
    generate_supplemental_table3(df)
    generate_supplemental_tables_4_and_5(df)
    generate_supplemental_table6(df)
    generate_supplemental_table7(df, df_suspected)
    generate_figure2(df)
    generate_dispatch_performance(df, df_suspected)
    generate_threshold_metrics()

if __name__ == "__main__":
    run_all_preliminary_analysis()
