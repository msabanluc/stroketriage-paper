#%% Imports

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix

from config.config import settings 

from data.preprocessing import run_full_cleaned_data_no_encoding

#%% Functions and definitions

TIME_FORMAT = "%Y/%m/%d:%I:%M:%S %p"

# Regex patterns 
PAT_RELEVANT = re.compile(
    r"\b(nihss|nih stroke scale|nih|last\s*known\s*well|lkw|modified\s*rankin|mrs|aspects|hunt\s*hess|gcs|glasgow|four\s*score)\b",
    flags=re.I,
)

CAT_PATTERNS = {
    "nih_total":  r"(nih.*(total|by\s*formula))|(^|\b)(nihss)\s*(total)?",
    "nih_comp":   r"nih.*(motor|arm|leg|facial|gaze|visual|language|dysarthria|ataxia|sensory|inattention|neglect|level\s*of\s*consciousness|loc|best\s*gaze)",
    "mrs":        r"(modified\s*rankin|rankin\b)",
    "aspects":    r"\baspects\b",
    "hunt_hess":  r"hunt\s*hess",
    "lkw":        r"(last\s*known\s*well|^lkw\b|onset\s*time|onset\s*date)",
    "gcs_total":  r"(gcs.*total|glasgow.*total)",
    "gcs_comp":   r"gcs.*(eye|verbal|motor)|glasgow.*(eye|verbal|motor)",
    "four":       r"\bfour\s*score\b",
}

KEYWORDS_SIMPLE = [
    "last known well","nih","aspects score total","aspects total score","hunt hess","rankin",
    "gcs score","stroke data r neuro glasgow","four score","func score","ruland"
]

# Loading & filtering
def load_flowsheet(path):
    return pd.read_csv(path)

def filter_relevant(df):
    mask = (
        df["FLO_MEAS_NAME"].str.contains(PAT_RELEVANT, na=False) |
        df["disp_name"].str.contains(PAT_RELEVANT, na=False)
    )
    return df.loc[mask].copy()

def subset_keyword_block(df, keywords):
    pat = re.compile("|".join(map(re.escape, keywords)), flags=re.I)
    return df[df["FLO_MEAS_NAME"].str.lower().str.contains(pat)].copy()

# Parsing
def classify_rows(df):
    def _class(name, disp):
        txt = f"{str(name)} || {str(disp)}".lower()
        for cat, pat in CAT_PATTERNS.items():
            if re.search(pat, txt, flags=re.I):
                return cat
        return None
    df = df.copy()
    df["cat"] = [
        _class(n, d) for n, d in zip(df.get("FLO_MEAS_NAME"), df.get("disp_name"))
    ]
    return df[df["cat"].notna()].copy()

def parse_times_and_values(df):
    df = df.copy()
    df["recorded_time_parsed"] = pd.to_datetime(
        df["recorded_time"].astype(str).str.strip("[]"),
        format=TIME_FORMAT,
        errors="coerce",
    )
    df["value_num"] = pd.to_numeric(
        df["meas_value"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    lkw_mask = df["cat"] == "lkw"
    if lkw_mask.any():
        df.loc[lkw_mask, "value_dt"] = pd.to_datetime(
            df.loc[lkw_mask, "meas_value"].astype(str).str.replace(r"[\[\]]", "", regex=True),
            errors="coerce",
        )
    return df

def apply_window_and_ranges(df, start, end):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    win = df[(df["recorded_time_parsed"] >= start_dt) & (df["recorded_time_parsed"] <= end_dt)].copy()
    def _valid(cat, v):
        if pd.isna(v):
            return True
        if cat in ("nih_total", "nih_comp"): return 0 <= v <= 42
        if cat == "mrs": return 0 <= v <= 6
        if cat == "aspects": return 0 <= v <= 10
        if cat == "hunt_hess": return 0 <= v <= 5
        if cat == "gcs_total": return 3 <= v <= 15
        if cat == "gcs_comp": return 0 <= v <= 6
        if cat == "four": return 0 <= v <= 16
        return True
    mask = win.apply(lambda r: _valid(r["cat"], r["value_num"]), axis=1)
    return win[mask].copy()


def summarize_encounters(df):
    if "hsp_account_id" not in df.columns:
        raise KeyError("Missing hsp_account_id in flowsheet data")

    # NIHSS totals
    nih_total = (
        df[df["cat"] == "nih_total"]
        .sort_values("recorded_time_parsed")
        .groupby("hsp_account_id")[
            ["value_num", "recorded_time_parsed"]
        ].first()
        .rename(columns={"value_num": "nihss_total", "recorded_time_parsed": "nihss_total_time"})
    )

    # NIHSS components
    comp = df[df["cat"] == "nih_comp"].copy()
    comp["comp_name"] = comp["FLO_MEAS_NAME"].str.upper().fillna(comp["disp_name"].str.upper())
    comp_earliest = comp.sort_values("recorded_time_parsed").drop_duplicates(["hsp_account_id", "comp_name"])
    nih_comp = comp_earliest.groupby("hsp_account_id").agg(
        nihss_comp_sum=("value_num", "sum"),
        nihss_comp_time=("recorded_time_parsed", "min"),
    )

    nih = nih_total.join(nih_comp, how="outer")
    nih["nihss_final_score"] = nih["nihss_total"].combine_first(nih["nihss_comp_sum"])
    nih["nihss_final_time"] = nih["nihss_total_time"].combine_first(nih["nihss_comp_time"])
    nih["nihss_method"] = np.where(
        nih["nihss_total"].notna(), "total",
        np.where(nih["nihss_comp_sum"].notna(), "components", "missing"),
    )

    # LKW
    lkw = df[df["cat"] == "lkw"].copy()
    lkw["lkw_dt"] = lkw["value_dt"].combine_first(lkw["recorded_time_parsed"])
    lkw = (
        lkw.dropna(subset=["lkw_dt"])
        .sort_values("lkw_dt")
        .drop_duplicates("hsp_account_id")[
            ["hsp_account_id", "lkw_dt"]
        ]
        .set_index("hsp_account_id")
    )

    def earliest_scalar(cat):
        tmp = (
            df[df["cat"] == cat]
            .sort_values("recorded_time_parsed")
            .dropna(subset=["value_num"])
            .drop_duplicates("hsp_account_id")
        )
        if tmp.empty:
            return pd.DataFrame(columns=[f"{cat}_value", f"{cat}_time"])
        return tmp.set_index("hsp_account_id")[
            ["value_num", "recorded_time_parsed"]
        ].rename(columns={"value_num": f"{cat}_value", "recorded_time_parsed": f"{cat}_time"})

    mrs = earliest_scalar("mrs")
    aspects = earliest_scalar("aspects")
    gcs_tot = earliest_scalar("gcs_total")

    enc = (
        nih.join(lkw, how="outer")
        .join(mrs, how="outer")
        .join(aspects, how="outer")
        .join(gcs_tot, how="outer")
        .reset_index()
    )
    return enc

# Pivot table of measures
def build_detail_pivot(df):
    subset = subset_keyword_block(df, KEYWORDS_SIMPLE).copy()
    subset["parsed_time"] = pd.to_datetime(
        subset["recorded_time"].astype(str).str.strip("[]"),
        format=TIME_FORMAT,
        errors="coerce",
    )
    subset["parsed_value"] = pd.to_numeric(
        subset["meas_value"].astype(str).str.extract(r"^(\d+)", expand=False),
        errors="coerce",
    )
    subset["FLO_MEAS_NAME"] = subset["FLO_MEAS_NAME"].str.strip()
    dedup = subset.sort_values("parsed_time").drop_duplicates(["hsp_account_id", "FLO_MEAS_NAME"])
    pv_vals = dedup.pivot(index="hsp_account_id", columns="FLO_MEAS_NAME", values="parsed_value")
    pv_times = dedup.pivot(index="hsp_account_id", columns="FLO_MEAS_NAME", values="parsed_time")
    pv_times.columns = [f"recorded_time_{c}" for c in pv_times.columns]
    pivoted = pd.concat([pv_vals, pv_times], axis=1).reset_index()
    mrn_map = subset[["hsp_account_id", "mrn"]].drop_duplicates()
    out = pivoted.merge(mrn_map, on="hsp_account_id", how="left")
    cols = out.columns.tolist()
    if "mrn" in cols:
        cols.insert(1, cols.pop(cols.index("mrn")))
        out = out[cols]
    return out

# Severity labeling & metrics
def nihss_severity_label(x):
    if pd.isna(x): return np.nan
    x = float(x)
    if x == 0: return "No stroke"
    if 1 <= x <= 4: return "Minor stroke"
    if 5 <= x <= 15: return "Moderate stroke"
    if 16 <= x <= 20: return "Moderate to severe stroke"
    if x >= 21: return "Severe stroke"
    return np.nan

def add_severity(df):
    df = df.copy()
    df["nihss_severity"] = df["nihss_final_score"].apply(nihss_severity_label)
    return df


def confusion_metrics_any_and_severe(df):
    TRUTH = "critical_stroke_cat"  # 1 severe, 2 non-severe, 3 no stroke
    if TRUTH not in df.columns:
        return {}
    d = df.copy()
    d = d[d["nihss_final_score"].notna()]

    def _norm(s):
        return str(s).strip().lower() if pd.notna(s) else np.nan

    sev = d["nihss_severity"].map(_norm)
    NO = {"no stroke"}
    MINOR = {"minor stroke"}
    MOD = {"moderate stroke"}
    MODSEV = {"moderate to severe stroke"}
    SEV = {"severe stroke"}

    # Any stroke task
    y_true_any = d[TRUTH].map(lambda x: 1 if x in (1, 2) else (0 if x == 3 else np.nan))
    y_pred_any = sev.map(lambda z: 1 if z in (MINOR | MOD | MODSEV | SEV) else (0 if z in NO else np.nan))
    mask_any = y_true_any.notna() & y_pred_any.notna()

    # Severe task
    y_true_sev = d[TRUTH].map(lambda x: 1 if x == 1 else (0 if x in (2, 3) else np.nan))
    y_pred_sev = sev.map(lambda z: 1 if z in (MODSEV | SEV) else (0 if z in (NO | MINOR | MOD) else np.nan))
    mask_sev = y_true_sev.notna() & y_pred_sev.notna()

    def _calc(y_t, y_p):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        def _safe(a, b): return float(a) / b if b else np.nan
        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": _safe(tp, tp + fn),
            "Specificity": _safe(tn, tn + fp),
            "PPV": _safe(tp, tp + fp),
            "Rows_used": int(len(y_t)),
        }

    return {
        "any_stroke": _calc(y_true_any[mask_any].astype(int), y_pred_any[mask_any].astype(int)),
        "severe_stroke": _calc(y_true_sev[mask_sev].astype(int), y_pred_sev[mask_sev].astype(int)),
        "dropped_any": int((~mask_any).sum()),
        "dropped_severe": int((~mask_sev).sum()),
    }

# Report
def build_report(orig, merged, pivot_df, metrics, encounters_stats=None):
    lines = []
    lines.append("=== Flowsheet Integration Summary ===")
    lines.append(f"Original encounters: {len(orig):,}")
    lines.append(f"Encounters with any NIHSS score: {merged['nihss_final_score'].notna().sum():,}")
    if "any_stroke" in merged.columns and "mrn" in merged.columns:
        try:
            n_stroke_patients = merged[merged["any_stroke"] == 1]["mrn"].nunique()
            lines.append(f"Number of stroke patients (unique MRN): {n_stroke_patients}")
        except Exception:
            pass
    if "any_stroke" in merged.columns:
        stroke_cov = merged.groupby("any_stroke")["nihss_final_score"].apply(lambda s: s.notna().mean()).round(3)
        lines.append("NIHSS coverage by any_stroke:")
        lines.extend([f"  any_stroke={idx}: {val:.0%}" for idx, val in stroke_cov.items()])

    # method counts
    if "nihss_method" in merged.columns:
        mc = merged["nihss_method"].value_counts(dropna=False)
        lines.append("NIHSS derivation method counts:")
        lines.extend([f"  {k}: {v}" for k, v in mc.items()])

    # Severity distribution
    if "nihss_severity" in merged.columns:
        sev_dist = merged["nihss_severity"].value_counts(dropna=False)
        lines.append("NIHSS severity distribution:")
        lines.extend([f"  {k}: {v}" for k, v in sev_dist.items()])
        if "critical_stroke_cat" in merged.columns:
            try:
                ct = pd.crosstab(merged["critical_stroke_cat"], merged["nihss_severity"], dropna=False)
                lines.append("\nTrue label vs NIHSS severity (counts):")
                # Compact print
                lines.append(ct.to_string())
            except Exception:
                pass
            try:
                med = merged.groupby("critical_stroke_cat")["nihss_final_score"].median()
                lines.append("Median NIHSS by true label:")
                for idx, val in med.items():
                    lines.append(f"  {idx}: {val}")
            except Exception:
                pass

    # Flowsheet match distribution
    if "has_flowsheet_match" in merged.columns:
        vc = merged["has_flowsheet_match"].value_counts(dropna=False)
        lines.append("\nFlowsheet match flag distribution (0/1):")
        lines.extend([f"  {idx}: {val}" for idx, val in vc.items()])
        if "any_stroke" in merged.columns:
            pct = merged.groupby("any_stroke")["has_flowsheet_match"].mean().round(3)
            lines.append("  By any_stroke (mean of match flag):")
            lines.extend([f"    any_stroke={k}: {v:.0%}" for k, v in pct.items()])

    # Metrics
    if metrics:
        lines.append("\n=== Performance Metrics ===")
        for task, vals in metrics.items():
            if task.startswith("dropped"): continue
            lines.append(f"[{task}] rows used={vals['Rows_used']}, TP={vals['TP']} FP={vals['FP']} FN={vals['FN']} TN={vals['TN']}")
            lines.append(f"  Sensitivity={vals['Sensitivity']:.3f} Specificity={vals['Specificity']:.3f} PPV={vals['PPV']:.3f}")
        lines.append(f"Dropped (any stroke task): {metrics.get('dropped_any', 0)}")
        lines.append(f"Dropped (severe stroke task): {metrics.get('dropped_severe', 0)}")

    # Encounters overlap
    if encounters_stats:
        lines.append("\n=== Encounters overlap ===")
        for k, v in encounters_stats.items():
            lines.append(f"{k}: {v}")

    # Pivot snapshot
    lines.append("\nPivot (detail) columns count: {}".format(len(pivot_df.columns)))
    lines.append("Sample pivot columns (first 12):")
    lines.append("  " + ", ".join(pivot_df.columns[:12]))
    try:
        if "mrn" in pivot_df.columns:
            cov = pivot_df.drop(columns=["mrn"]).notna().mean().sort_values(ascending=False)
        else:
            cov = pivot_df.notna().mean().sort_values(ascending=False)
        top = cov.head(10)
        lines.append("Top 10 most populated pivot measures:")
        for k, v in top.items():
            lines.append(f"  {k}: {v:.0%}")
    except Exception:
        pass

    return "\n".join(lines)

#%% Main
def main():
    # Load
    orig = run_full_cleaned_data_no_encoding()
    orig["hsp_account_id"] = orig["hsp_account_id"].astype(str)

    # Settings
    flowsheet_path = settings.get("flowsheet_data_path") or r"R:\mtootooni\213809\DataGiven_08152024\irb_213809_data_given_flowsheet_08152024.csv"
    date_cfg = settings.get("analysis_settings", {})
    start_date = date_cfg.get("start_date", "2015-01-01")
    end_date = date_cfg.get("end_date", "2020-12-31 23:59:59")

    # Load flowsheet
    flowsheet = load_flowsheet(flowsheet_path)
    flowsheet["hsp_account_id"] = flowsheet["hsp_account_id"].astype(str)

    # Pivot table
    pivot_df = build_detail_pivot(flowsheet)

    # Merge pivot with orig for flowsheet match
    merged_pivot = orig.merge(pivot_df, on="hsp_account_id", how="left", suffixes=("", "_flowsheet"))
    merged_pivot["has_flowsheet_match"] = merged_pivot.get("mrn_flowsheet").notna().astype(int) if "mrn_flowsheet" in merged_pivot.columns else merged_pivot.get("mrn").notna().astype(int)

    # Classification/summarization
    relevant = filter_relevant(flowsheet)
    classified = classify_rows(relevant)
    parsed = parse_times_and_values(classified)
    cleaned = apply_window_and_ranges(parsed, start_date, end_date)
    enc_summary = summarize_encounters(cleaned)

    enc_summary["hsp_account_id"] = enc_summary["hsp_account_id"].astype(str)
    merged = orig.merge(enc_summary, on="hsp_account_id", how="left")
    merged = add_severity(merged)

    # Metrics
    metrics = confusion_metrics_any_and_severe(merged)

    # Load encounters.csv for overlap stats
    encounters_stats = {}
    enc_path = settings.get("encounters_path") or r"R:\mtootooni\213809\msaban\encounters.csv"
    try:
        enc = pd.read_csv(enc_path)
        enc["hsp_account_id"] = enc["hsp_account_id"].astype(str)
        ids_enc = set(enc["hsp_account_id"].unique())
        ids_ems = set(orig["hsp_account_id"].unique())
        ids_flow = set(flowsheet["hsp_account_id"].astype(str).unique())
        encounters_stats["orig in encounters"] = f"{len(ids_ems & ids_enc)} / {len(ids_ems)}"
        encounters_stats["encounters in flowsheet"] = f"{len(ids_enc & ids_flow)} / {len(ids_enc)}"
        encounters_stats["ems in flowsheet"] = f"{len(ids_ems & ids_flow)} / {len(ids_ems)}"
    except Exception:
        encounters_stats = None

    # Report
    report = build_report(orig, merged, pivot_df, metrics, encounters_stats)
    print(report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = settings["output_dir"] / f"ed_measures_report_{ts}.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")

if __name__ == "__main__":
    main()

