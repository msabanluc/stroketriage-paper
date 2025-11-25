import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import cvxpy as cp

def cap_proba(p):
    if p > 1:
        return 1
    elif p < 0:
        return 0
    else:
        return p



def get_inc_poly(x, y, deg):
    p = cp.Variable(deg)
    V = np.vander(x, deg+1, increasing=True)[:, 1:]
    axis = np.linspace(0, 1, 1000)
    V_axis_1 = np.vander(axis, deg, increasing=True) * np.array([i for i in range(1, deg+1)])
    
    objective = cp.Minimize(cp.sum_squares(V @ p - y))
    constraints = [V_axis_1 @ p >=0]
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")
    
    return p.value

class KFoldCalibratedModel:
    def __init__(self, base_model, P, deg):
        self.base_model = base_model
        self.P = P
        self.deg = deg

    def predict_proba(self, X):
        proba_uncal = self.base_model.predict_proba(X)[:, 1]
        V = np.vander(proba_uncal, self.deg + 1, increasing=True)[:, 1:]
        return pd.DataFrame(V @ self.P, index=X.index).applymap(cap_proba)

    def calibrate_proba(self, proba_uncal):
        V = np.vander(proba_uncal, self.deg + 1, increasing=True)[:, 1:]
        return pd.DataFrame(V @ self.P).applymap(cap_proba)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def fit_kfold_calibrated_model(base_model_class, X, y, params, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=2023)
    oof_preds = np.zeros(len(X))
    oof_index = np.zeros(len(X), dtype=bool)
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = base_model_class(random_state=2023, **params)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = probs
        oof_index[val_idx] = True
    assert oof_index.all(), "Not all rows were covered in CV"
    deg = 10
    P = get_inc_poly(oof_preds, y, deg)
    model_final = base_model_class(random_state=2023, **params)
    model_final.fit(X, y)
    return KFoldCalibratedModel(model_final, P, deg)


class AttendedTemperatureScaling:
    def __init__(self, num_bins=5):
        self.temperatures = None
        self.num_bins = num_bins

    def loss_fn(self, temp, logits, labels, sample_weights):
        import tensorflow as tf
        scaled_logits = logits / temp
        loss = tf.keras.losses.binary_crossentropy(labels, tf.sigmoid(scaled_logits), from_logits=False)
        return tf.reduce_mean(loss * sample_weights)

    def fit(self, logits, labels):
        import tensorflow as tf
        from scipy.optimize import minimize
        logits = np.array(logits).flatten()
        labels = np.array(labels).flatten()
        prob_preds = tf.sigmoid(logits).numpy()
        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(prob_preds, bins) - 1
        bin_counts = np.array([np.sum(bin_indices == i) for i in range(self.num_bins)])
        bin_weights = 1.0 / (bin_counts + 1e-6)
        bin_weights /= np.sum(bin_weights)
        sample_weights = bin_weights[bin_indices]
        res = minimize(lambda t: self.loss_fn(t, logits, labels, sample_weights).numpy(),
                       x0=np.array([1.0]), method='L-BFGS-B', bounds=[(0.1, 10.0)])
        self.temperatures = res.x[0]

    def scale_logits(self, logits):
        import tensorflow as tf
        return logits / self.temperatures

    def predict_proba(self, logits):
        import tensorflow as tf
        return tf.sigmoid(self.scale_logits(logits)).numpy()