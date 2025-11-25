# data/load_data.py
import pandas as pd
from config.paths import DATA_DIR, SPLIT_DIR

def _suffix_from_mode(mode):
    suffix_map = {"full": "", "cpss": "_sub", "stroke_relevant": "_sr"}
    if mode not in suffix_map:
        raise ValueError(f"Unknown mode: {mode}. Use 'full', 'cpss', or 'stroke_relevant'.")
    return suffix_map[mode]

def load_splits(mode = "full", extra_suffix = ""):
    suffix = _suffix_from_mode(mode) + (extra_suffix or "")

    X_train_enc = pd.read_csv(SPLIT_DIR / f"X_train_enc{suffix}.csv")
    y_train     = pd.read_csv(SPLIT_DIR / f"y_train{suffix}.csv")
    X_val_enc   = pd.read_csv(SPLIT_DIR / f"X_val_enc{suffix}.csv")
    y_val       = pd.read_csv(SPLIT_DIR / f"y_val{suffix}.csv")
    X_test_enc  = pd.read_csv(SPLIT_DIR / f"X_test_enc{suffix}.csv")
    y_test      = pd.read_csv(SPLIT_DIR / f"y_test{suffix}.csv")
    return X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test
