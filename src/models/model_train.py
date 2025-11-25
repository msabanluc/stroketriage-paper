#%% 
import optuna
from datetime import datetime
from pathlib import Path
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score 
import xgboost as xgb
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config.paths import MODEL_DIR
import json

def _ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

#%% Random Forest

def objective_rf(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_depth': trial.suggest_categorical('max_depth', [None] + list(range(2, 32, 2))),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
        'random_state': 2023
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return score.mean()


def train_rf_with_optuna(X_train, y_train, study_name="rf_study", n_trials=3000, db_file="rf_stroke_all.db", retune=False):
    _ensure_model_dir()
    
    params_file_base = Path(db_file).stem
    params_path = MODEL_DIR / f"{params_file_base}_best_params.json"
    study = None

    if retune:
        timestamp = datetime.now().strftime("%m%d%y_%H")
        storage_path = MODEL_DIR / f"{params_file_base}_{timestamp}.db"
        
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{storage_path}",
            study_name=study_name,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=n_trials)
        best_params = study.best_trial.params
        
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        if not params_path.exists():
            raise FileNotFoundError(
                f"Best parameters file not found at: {params_path}. "
                "Run with retune=True to generate it."
            )
        with open(params_path, 'r') as f:
            best_params = json.load(f)

    best_model = RandomForestClassifier(random_state=2023, **best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params

#%% XGBoost

def objective_xgb(trial, X_train, y_train):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 2023,
        'n_estimators': trial.suggest_int('n_estimators', 50, 750, step=25),
        'learning_rate': trial.suggest_float('eta', 0.0001, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0, 15, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    return score.mean()


def train_xgb_with_optuna(X_train, y_train, study_name="xgb_study", n_trials=3000, db_file="xgb_stroke_all.db", retune=False):
    _ensure_model_dir()
    
    params_file_base = Path(db_file).stem
    params_path = MODEL_DIR / f"{params_file_base}_best_params.json"
    study = None

    if retune:
        timestamp = datetime.now().strftime("%m%d%y_%H")
        storage_path = MODEL_DIR / f"{params_file_base}_{timestamp}.db"
        
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{storage_path}",
            study_name=study_name,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=n_trials)
        best_params = study.best_trial.params
        
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        if not params_path.exists():
            raise FileNotFoundError(
                f"Best parameters file not found at: {params_path}. "
                "Run with retune=True to generate it."
            )
        with open(params_path, 'r') as f:
            best_params = json.load(f)

    best_model = xgb.XGBClassifier(random_state=2023, **best_params)
    best_model.fit(X_train, y_train)
    return best_model, best_params

#%% Sequential Neural Network

def build_model_logits(shape, n_hidden, n_neurons, dropout, eta, output_bias):
    bias_initializer = keras.initializers.Constant(output_bias) if output_bias is not None else None
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(shape,)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(units=n_neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation=None, bias_initializer=bias_initializer))
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=eta),
        metrics=[keras.metrics.AUC(name='auc')]
    )
    return model

class WarmUpLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, initial_lr=1e-5, target_lr=0.001):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            new_lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(new_lr)


def kfold_train_snn_with_logits(X, y, params, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
    all_logits = np.zeros(len(X))
    all_indices = np.zeros(len(X), dtype=bool)
    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        output_bias = np.log(y_tr.value_counts()[1] / y_tr.value_counts()[0])
        model = build_model_logits(
            shape=X.shape[-1], n_hidden=params['n_hidden'], n_neurons=params['n_neurons'],
            dropout=params['dropout'], eta=params['eta'], output_bias=output_bias
        )
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            class_weight={0: params['class_weight_0'], 1: params['class_weight_1']},
            batch_size=params['batch_size'], epochs=250,
            callbacks=[
                WarmUpLearningRate(warmup_epochs=10, initial_lr=1e-5, target_lr=params['eta']),
                EarlyStopping(monitor='val_auc', patience=30, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=5e-6)
            ], verbose=0
        )
        logits = model.predict(X_val).squeeze()
        all_logits[val_idx] = logits
        all_indices[val_idx] = True
        models.append(model)
        tf.keras.utils.set_random_seed(2023)
    assert all_indices.all(), "Not all rows were covered during CV"
    return all_logits, models


def objective_snn(trial, X_train_full, y_train_full):
    # Hyperparameter search space
    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_neurons = trial.suggest_int('n_neurons', 50, 500, step=25)
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    eta = trial.suggest_float('eta', 0.0001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    class_weight_0 = trial.suggest_float('class_weight_0', 0.1, 1.0, step=0.1)
    class_weight_1 = trial.suggest_float('class_weight_1', 2.0, 60.0, step=2.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
    aucs = []
    input_shape = X_train_full.shape[1]
    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_tr, X_val = X_train_full.iloc[train_idx].values, X_train_full.iloc[val_idx].values
        y_tr = y_train_full.iloc[train_idx].astype(float)
        y_val = y_train_full.iloc[val_idx].astype(float)
        output_bias = np.log(y_tr.mean() / (1 - y_tr.mean()))
        model = build_model_logits(input_shape, n_hidden, n_neurons, dropout, eta, output_bias)
        callbacks = [
            WarmUpLearningRate(warmup_epochs=10, initial_lr=1e-5, target_lr=eta),
            EarlyStopping(monitor='val_auc', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=5e-6)
        ]
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), class_weight={0: class_weight_0, 1: class_weight_1},
                  batch_size=batch_size, epochs=250, callbacks=callbacks, verbose=0)
        logits = model.predict(X_val).flatten()
        probs = tf.sigmoid(logits).numpy()
        aucs.append(roc_auc_score(y_val, probs))
    return np.mean(aucs)


def train_snn_with_optuna_and_logits(X_train, y_train, study_name="snn_study", db_file="snn_stroke_all.db", n_trials=1000, retune=False):  
    _ensure_model_dir()
    
    params_file_base = Path(db_file).stem
    params_path = MODEL_DIR / f"{params_file_base}_best_params.json"

    if retune:
        timestamp = datetime.now().strftime("%m%d%y_%H%M")
        storage_path = MODEL_DIR / f"{params_file_base}_{timestamp}.db"
        
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{storage_path}",
            study_name=study_name,
            load_if_exists=True
        )
        study.optimize(lambda trial: objective_snn(trial, X_train, y_train), n_trials=n_trials)
        best_params = study.best_trial.params
        
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
    else:
        if not params_path.exists():
            raise FileNotFoundError(
                f"Best parameters file not found at: {params_path}. "
                "Run with retune=True to generate it."
            )
        with open(params_path, 'r') as f:
            best_params = json.load(f)

    logits, models = kfold_train_snn_with_logits(X_train, y_train, best_params)
    return models, logits, best_params