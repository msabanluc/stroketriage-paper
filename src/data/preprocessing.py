import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from config.config import settings
from config.paths import SPLIT_DIR, DATA_DIR
from data.load_data import load_splits


DISPATCH_VARS = ["calltype", "priority", "transpriority", "natureofcall"]

def load_raw_data(paths):
    ph2 = pd.read_sas(paths['ph2'], encoding='latin1')
    geo = pd.read_excel(paths['geo'], dtype=object)
    adi = pd.read_csv(paths['adi'])
    fdc_desc = pd.read_csv(paths['fdc_desc'])
    noc = pd.read_excel(paths['noc'])
    prior = pd.read_excel(paths['prior'])
    return ph2, geo, adi, fdc_desc, noc, prior

def clean_geodata(geo, adi):
    geo_s = geo[['puaddr','census_block_group_id_2010','census_tract_id_2010']].drop_duplicates(subset='puaddr', keep='first')
    adi_s = adi[['FIPS','ADI_NATRANK']].copy()
    adi_s.rename(columns={'FIPS': 'census_block_group_id_2010'}, inplace=True)
    adi_s['ADI_NATRANK'] = pd.to_numeric(adi_s['ADI_NATRANK'], errors='coerce')
    return geo_s, adi_s

def basic_cleaning(ph2):
    ph2 = ph2[ph2['Loyola_Destination'] == 'Loyola']
    ph2.replace('NULL', np.nan, inplace=True)
    ph2['arrival_time_diff'] = (ph2['admit_time'] - ph2['atdtime']) / 60
    ph2['hsp_account_id'] = ph2['hsp_account_id'].astype('int64')
    ph2['custno'] = ph2['custno'].astype('int64')
    ph2 = ph2[(ph2['transfer_patient'] != 1) & (ph2['natureofcall'] != 37)]
    for col in ['stroke_scale', 'pulsestrength', 'respeffort', 'calltype']:
        ph2[col] = ph2[col].astype(float)
    return ph2.reset_index(drop=True)



def remove_outliers(df):
    mapping = {
        'hr':      {'timediff': 'hr_timediff',     'prehosp': ['pulse1'],           'ed': ['hr_ed']},
        'rr':      {'timediff': 'rr_timediff',     'prehosp': ['resprate1'],        'ed': ['rr_ed']},
        'bp':      {'timediff': 'bp_timediff',     'prehosp': ['bps1', 'bpd1'],     'ed': ['sbp_ed', 'dbp_ed']},
        'temp':    {'timediff': 'temp_timediff',   'prehosp': ['temperature1'],     'ed': ['temp_ed']},
        'glu':     {'timediff': 'glu_timediff',    'prehosp': ['glucose1'],         'ed': ['glucose_ed']},
        'spo2':    {'timediff': 'spo2_timediff',   'prehosp': ['spo2_1'],           'ed': ['spo2_ed']},
        'gcs':     {'timediff': 'gcs_timediff',    'prehosp': [],                   'ed': ['gcs_ed']},
    }

    for var, spec in mapping.items():
        tdiff = spec['timediff']
        if tdiff in df.columns:
            mask = (df[tdiff] > 240) | (df[tdiff] < 0)

            for col in spec['prehosp']:
                if col in df.columns:
                    df.loc[mask, col] = np.nan

            for col in spec['ed']:
                if col in df.columns:
                    df.loc[mask, col] = np.nan

    # Recompute gcs_total
    gcs_components = ['gcseyes', 'gcsmotor', 'gcsverbal']
    if all(col in df.columns for col in gcs_components):
        df['gcs_total'] = df[gcs_components].sum(axis=1, skipna=False)

    return df


def deduplicate_and_merge(ph2):
    ph2 = ph2.sort_values(by=['hsp_account_id', 'prehosp_vitals_count']).reset_index(drop=True)
    ph2 = ph2.drop_duplicates(subset='hsp_account_id', keep='first')
    fill_cols =list(ph2.filter(regex='_ed\d*$').columns)
    def fill_ed(df):
        for uid in df['custno'].unique():
            sub = df[df['custno'] == uid]
            if len(sub) > 1:
                for col in fill_cols:
                    if pd.isnull(sub.iloc[0][col]):
                        val = sub[col].dropna().iloc[0] if not sub[col].dropna().empty else np.nan
                        df.loc[sub.index[0], col] = val
        return df
    ph2 = fill_ed(ph2)
    ph2 = ph2.drop_duplicates(subset='custno', keep='first')
    return ph2.reset_index(drop=True)

def remove_internal_addresses(df):
    df2 = df[~df['puaddr'].isin(['2160 S. FIRST AVENUE','2160 S 1ST AVE','2160 1st ave','2160 S. 1st Ave','2160 S 1ST AVE Bldg 150 3332'])]
    return df2.reset_index(drop=True)

def merge_geo_adi(df, geo_s, adi_s):
    df = df.merge(geo_s, how='left', on='puaddr')
    df = df.merge(adi_s, how='left', on='census_block_group_id_2010')
    return df

def derive_labels(df):
    df.loc[(df.stroke_type == 'ICH') | (df.stroke_type == 'SAH'), ['critical_stroke','critical_stroke_cat']] = 1, 1
    df['gcs_total'] = df[['gcseyes', 'gcsmotor', 'gcsverbal']].sum(axis=1, skipna=False)
    df['stroke'] = df['stroke_type'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
    return df


def get_cpstroke_subset(df, label_column=False):
    na_codes = [1, 2, 3]
    abnormal_codes = [4, 16]
    normal_codes = [6, 14]
    non_conclusive_codes = [5, 15]

    # Keep only rows where CPSS was performed
    df_cp = df[df['stroke_scale'].notnull() & ~df['stroke_scale'].isin(na_codes)].copy()

    if label_column:
        df_cp['cpstroke_label'] = df_cp['stroke_scale'].apply(
            lambda x: 1 if x in abnormal_codes + non_conclusive_codes else
                      0 if x in normal_codes else
                      np.nan
        )

    return df_cp.reset_index(drop=True)


def get_stroke_relevant_subset(
    df,
    noc_col="natureofcall",        # integer code matching noc.xlsx 'code'
    label_col="stroke",      # 1 = stroke, 0 = no stroke
    cp_col="stroke_scale"          # CPSS code column
):
    # valid CPSS
    na_codes = [1, 2, 3]
    mask_stroke = df[label_col] == 1
    mask_cpss   = df[cp_col].notna() & ~df[cp_col].isin(na_codes)

    # stroke-relevant dispatch codes (from noc.xlsx)
    stroke_like_noc = {13, 19, 20, 35, 36, 58, 61} #Convulsions, Fall(s), Headache, Stroke/CVA, Syncope/Unconcious, Dizziness, Seizures
    mask_noc = df[noc_col].isin(stroke_like_noc)

    subset = df[mask_stroke | mask_cpss | mask_noc].copy()
    return subset.reset_index(drop=True)


def extract_ml_dataset(df, include_dispatch = True):
    base_cols = [
        'mrn','hsp_account_id','custno','tdate','job','seq','vdate','vtime',
        'prehosp_vitals_count','pat_name','stroke_scale','stroke_type',
        'critical_stroke','critical_stroke_cat','c_age','c_weight','pulse1',
        'resprate1','bps1','bpd1','spo2_1','ADI_NATRANK','c_sex1',
        'race_eth_ed1','pulsestrength','respeffort','gcs_total'
    ]
    dispatch_cols = ["calltype","priority","transpriority","natureofcall"]
    if include_dispatch:
        use_cols = base_cols + dispatch_cols + ['narrative']
    else:
        use_cols = base_cols + ['narrative']

    df = df[use_cols].copy()
    df['ADI_NATRANK'] = df['ADI_NATRANK'].astype(float)

    # Drop labels + narrative + stroke_scale
    drop_cols = [
        'mrn','hsp_account_id','custno','tdate','job','seq','vdate','vtime',
        'prehosp_vitals_count','pat_name','critical_stroke','critical_stroke_cat',
        'stroke_type','stroke_scale','narrative'
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[['critical_stroke_cat']].copy()
    return X, y


def impute_and_encode(X, y, do_impute=True, do_encode=True, fdc_desc=None, noc=None, prior=None):
    X = X.copy()

    if do_impute:
        impfeats = ['c_age', 'c_weight', 'pulse1', 'resprate1', 'bps1', 'bpd1', 'spo2_1', 'ADI_NATRANK', 'gcs_total']
        avail_impfeats = [c for c in impfeats if c in X.columns]
        if avail_impfeats:
            imp = IterativeImputer(random_state=0)
            X_imp = pd.DataFrame(imp.fit_transform(X[avail_impfeats]), columns=avail_impfeats, index=X.index).round(0)
        else:
            X_imp = pd.DataFrame(index=X.index)

        non_num_cols = ['respeffort', 'pulsestrength', 'race_eth_ed1', 'c_sex1',
                        'natureofcall', 'calltype', 'priority', 'transpriority']
        passthrough = [c for c in non_num_cols if c in X.columns]
        other = X[passthrough].copy() if passthrough else pd.DataFrame(index=X.index)

        X = pd.concat([X_imp, other], axis=1)

        if 'respeffort' in X.columns:
            X['respeffort'] = X['respeffort'].fillna(14)
        if 'natureofcall' in X.columns:
            X['natureofcall'] = X['natureofcall'].fillna(1)
        X = X.fillna(0)

        if 'c_sex1' in X.columns:
            X['c_sex1'] = X['c_sex1'].replace(['Female', 'Male'], [0, 1])

    if do_encode:
        temp = X.copy()

        # Map coded fields to strings only if those columns exist
        if fdc_desc is not None:
            descr = fdc_desc[fdc_desc['type'].isin([17, 19])]
            ps = descr[descr['type'] == 17].set_index('code')['descr'].to_dict()
            rf = descr[descr['type'] == 19].set_index('code')['descr'].to_dict()
            if 'pulsestrength' in temp.columns:
                temp['pulsestrength'] = temp['pulsestrength'].replace(ps)
            if 'respeffort' in temp.columns:
                temp['respeffort'] = temp['respeffort'].replace(rf)

        if noc is not None and 'natureofcall' in temp.columns:
            temp['natureofcall'] = temp['natureofcall'].replace(noc.set_index('code')['descr'].to_dict())
            temp['natureofcall'] = temp['natureofcall'].replace('<None>', '(None)')

        if prior is not None and 'transpriority' in temp.columns:
            temp['transpriority'] = temp['transpriority'].replace(prior.set_index('code')['descr'].to_dict())

        if 'c_sex1' in temp.columns:
            temp['c_sex1'] = temp['c_sex1'].replace({0: 'Female', 1: 'Male'})

        if 'race_eth_ed1' in temp.columns:
            temp['race_eth_ed1'] = temp['race_eth_ed1'].replace({
                0: 'Not Available', 1: 'NHWhite', 2: 'NHBlack', 3: 'NHOther', 4: 'Hispanic'
            })

        candidate_enc = ['c_sex1', 'race_eth_ed1', 'pulsestrength', 'respeffort',
                         'priority', 'transpriority', 'calltype', 'natureofcall']
        columns_to_encode = [c for c in candidate_enc if c in temp.columns]
        if columns_to_encode:
            X = pd.get_dummies(temp, columns=columns_to_encode, dtype=float)
        else:
            X = temp

    y = y.copy()
    y['stroke_label'] = y['critical_stroke_cat'].map({1: 1, 2: 1, 3: 0})
    y['critical_label'] = y['critical_stroke_cat'].map({1: 1, 2: 0, 3: 0})

    assert len(X) == len(y), f"impute_and_encode length mismatch: {len(X)} vs {len(y)}"
    assert all(X.index == y.index), "impute_and_encode index misalignment"

    return X, y

def split_data(X, y, stratify_col='critical_stroke_cat'):

    assert len(X) == len(y), f"split_data pre-split length mismatch: {len(X)} vs {len(y)}"
    assert all(X.index == y.index), "split_data pre-split index mismatch"

    random_state = 2023

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y[stratify_col]
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp[stratify_col]
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )



def load_and_clean_data(mode="full"):

    path_dict = {key: DATA_DIR / filename for key, filename in settings['paths'].items()}

    ph2, geo, adi, fdc_desc, noc, prior = load_raw_data(path_dict)
    geo_s, adi_s   = clean_geodata(geo, adi)
    ph2            = basic_cleaning(ph2)
    ph2            = remove_outliers(ph2)
    ph2            = deduplicate_and_merge(ph2)
    ph2            = remove_internal_addresses(ph2)
    ph2            = merge_geo_adi(ph2, geo_s, adi_s)
    ph2            = derive_labels(ph2)

    if mode == "cpss":
        ph2 = get_cpstroke_subset(ph2)
    elif mode == "stroke_relevant":
        ph2 = get_stroke_relevant_subset(ph2)

    return ph2, fdc_desc, noc, prior


def run_full_preprocessing(mode = "full", drop_dispatch = False):

    suffix_map = {
        "full": "",
        "cpss": "_sub",
        "stroke_relevant": "_sr"
    }
    if mode not in suffix_map:
        raise ValueError(f"Unknown mode: {mode}. Use 'full', 'cpss', or 'stroke_relevant'.")
    suffix = suffix_map[mode]
    if drop_dispatch:
        suffix=f"{suffix}_nodisp"

    # check for existing files
    expected = [
        f"X_train_enc{suffix}.csv", f"y_train{suffix}.csv",
        f"X_val_enc{suffix}.csv",   f"y_val{suffix}.csv",
        f"X_test_enc{suffix}.csv",  f"y_test{suffix}.csv",
    ]
    if all((SPLIT_DIR / fn).exists() for fn in expected):
        print(f"Loading preprocessed splits from disk (mode={mode}, drop_dispatch={drop_dispatch})")
        splits = load_splits(mode=mode, extra_suffix=("_nodisp" if drop_dispatch else ""))
        # load_splits returns: X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test
        return splits[0], splits[2], splits[4], splits[1], splits[3], splits[5]


    # otherwise: full pipeline
    ph2, fdc_desc, noc, prior = load_and_clean_data(mode=mode)

    # feature-engineering
    X_raw, y_raw = extract_ml_dataset(ph2, include_dispatch=(not drop_dispatch))
    assert len(X_raw) == len(y_raw), f"extract_ml_dataset length mismatch: {len(X_raw)} vs {len(y_raw)}"


    X_proc, y_proc = impute_and_encode(
        X_raw, y_raw,
        do_impute=True, do_encode=True,
        fdc_desc=fdc_desc, noc=noc, prior=prior
    )

    # train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_proc, y_proc, stratify_col="critical_stroke_cat"
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(SPLIT_DIR / f"X_train_enc{suffix}.csv", index=False)
    y_train.to_csv(SPLIT_DIR / f"y_train{suffix}.csv",       index=False)
    X_val.to_csv( SPLIT_DIR / f"X_val_enc{suffix}.csv",   index=False)
    y_val.to_csv( SPLIT_DIR / f"y_val{suffix}.csv",       index=False)
    X_test.to_csv(SPLIT_DIR / f"X_test_enc{suffix}.csv",  index=False)
    y_test.to_csv(SPLIT_DIR / f"y_test{suffix}.csv",      index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test







def run_full_cleaned_data_no_encoding():

    ph2, _, _, _ = load_and_clean_data(mode="full")

    impfeats = ['c_age', 'c_weight', 'pulse1', 'resprate1', 'bps1', 'bpd1', 'spo2_1', 'ADI_NATRANK', 'gcs_total']
    imp = IterativeImputer(random_state=0)
    available_impfeats = [col for col in impfeats if col in ph2.columns]
    ph2[available_impfeats] = imp.fit_transform(ph2[available_impfeats]).round(0)

    ph2['respeffort'] = ph2['respeffort'].fillna(14)
    ph2['natureofcall'] = ph2['natureofcall'].fillna(1)
    ph2['c_sex1'] = ph2['c_sex1'].replace(['Female', 'Male'], [0, 1])

    return ph2
