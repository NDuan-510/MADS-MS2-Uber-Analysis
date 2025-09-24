# project_package/modeling.py — lean + tuning + speed-ups

from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# ---- leakage guard ----
EXCLUDE_ALWAYS = {
    "Booking Status",
    "Reason for cancelling by Customer_fill",
    "Driver Cancellation Reason_fill",
    "Incomplete Rides Reason_fill",
    "Incomplete Rides_fill",
    "Cancelled Rides by Customer_fill",
    "Cancelled Rides by Driver_fill",
}

# ---- helpers ----
def load_csv_dedup(p: str) -> pd.DataFrame:
    df = pd.read_csv(p, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return _downcast_numeric(df)

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s): df[c] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s): df[c] = pd.to_numeric(s, downcast="integer")
    return df

def make_binary_target(df: pd.DataFrame, col="Booking Status") -> Tuple[pd.DataFrame, str]:
    if col not in df.columns: raise KeyError(f"{col} missing")
    out = df.copy()
    out["_target_completed_"] = out[col].astype(str).str.lower().str.contains("completed").astype(int)
    return out, "_target_completed_"

def time_or_random_split(df: pd.DataFrame, target: str, valid_ratio=0.2, seed=42):
    if "booking_datetime" in df.columns:
        d = df.copy(); d["booking_datetime"] = pd.to_datetime(d["booking_datetime"], errors="coerce")
        d = d.sort_values("booking_datetime").reset_index(drop=True)
        cut = int(len(d)*(1-valid_ratio)); return d.iloc[:cut].copy(), d.iloc[cut:].copy()
    strat = df[target] if set(pd.unique(df[target].dropna())) <= {0,1} else None
    tr, va = train_test_split(df, test_size=valid_ratio, random_state=seed, stratify=strat)
    return tr.copy(), va.copy()

def feat_types(X: pd.DataFrame, max_cat=200):
    num, cat = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s): num.append(c)
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if s.nunique(dropna=False) <= max_cat: cat.append(c)
    return num, cat

def pre_ohe_scaled(X: pd.DataFrame) -> ColumnTransformer:
    num, cat = feat_types(X, max_cat=200)
    return ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)],
        remainder="drop", sparse_threshold=1.0
    )

def pre_ordinal(X: pd.DataFrame) -> ColumnTransformer:
    num, cat = feat_types(X, max_cat=200)
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat)],
        remainder="drop"
    )

def _param_grid_iter(grid: Dict[str, Iterable]) -> Iterable[Dict[str, object]]:
    # small Cartesian product generator
    if not grid: yield {}
    else:
        keys = list(grid.keys())
        vals = [list(grid[k]) for k in keys]
        idx = [0]*len(keys)
        def rec(i):
            if i==len(keys):
                yield {keys[j]: vals[j][idx[j]] for j in range(len(keys))}
                return
            for t in range(len(vals[i])):
                idx[i]=t
                yield from rec(i+1)
        yield from rec(0)

# ---- results ----
@dataclass
class ClassificationResult:
    best_model_name: str
    report: Dict[str, float]
    model_path: str
    artifacts_dir: str

@dataclass
class RegressionResult:
    best_model_name: str
    report: Dict[str, float]
    model_path: str
    artifacts_dir: str

# ---- exporters (per-family best only) ----
def export_cls_preds(name: str, y_true, y_pred, proba, valid_df, id_cols, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = name.replace(" ", "_")
    cols = {c: valid_df[c].values for c in id_cols if c in valid_df.columns}
    data = {"y_true": y_true, "y_pred": y_pred}
    if proba is not None: data["proba"] = proba
    pd.DataFrame({**cols, **data}).to_csv(os.path.join(outdir, f"cls_{base}_preds.csv"), index=False)

def export_reg_preds(name: str, target_col: str, y_true, y_pred, valid_df, id_cols, outdir):
    os.makedirs(outdir, exist_ok=True)
    base = name.replace(" ", "_"); tgt = target_col.replace(" ", "_")
    cols = {c: valid_df[c].values for c in id_cols if c in valid_df.columns}
    pd.DataFrame({**cols, "y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred}).to_csv(
        os.path.join(outdir, f"reg_{tgt}_{base}_preds.csv"), index=False
    )

# ---- training APIs with lightweight tuning ----
def train_classification_from_csv(
    csv_path: str, target_col: Optional[str] = None, id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "artifacts/supervised", valid_ratio: float = 0.2, random_state: int = 42,
    tune_row_cap: Optional[int] = 50000,
) -> ClassificationResult:
    os.makedirs(artifacts_dir, exist_ok=True)
    df = load_csv_dedup(csv_path)
    if target_col is None or target_col not in df.columns: df, target_col = make_binary_target(df)
    id_cols = id_cols or []
    if id_cols: df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

    tr, va = time_or_random_split(df, target_col, valid_ratio, random_state)
    y_tr, y_va = tr[target_col].astype(int).values, va[target_col].astype(int).values
    drop_cols = [c for c in ({target_col} | EXCLUDE_ALWAYS | set(id_cols)) if c in tr.columns]
    X_tr, X_va = tr.drop(columns=drop_cols, errors="ignore"), va.drop(columns=drop_cols, errors="ignore")

    # optional cap for tuning (refit best on full training later)
    if tune_row_cap is not None and len(X_tr) > tune_row_cap:
        X_tune, y_tune = X_tr.iloc[:tune_row_cap], y_tr[:tune_row_cap]
    else:
        X_tune, y_tune = X_tr, y_tr

    families = {
        "logreg": {
            "make": lambda: Pipeline([("pre", pre_ohe_scaled(X_tr)),
                                      ("clf", LogisticRegression(solver="saga", max_iter=2000,
                                                                 class_weight="balanced",
                                                                 random_state=random_state))]),
            "grid": {"clf__C": [0.5, 1.0, 2.0]}
        },
        "rf": {
            "make": lambda: Pipeline([("pre", pre_ordinal(X_tr)),
                                      ("clf", RandomForestClassifier(n_estimators=150,
                                                                     class_weight="balanced",
                                                                     n_jobs=-1, random_state=random_state))]),
            "grid": {"clf__n_estimators": [150, 250], "clf__max_depth": [None, 12]}
        },
        "knn": {
            "make": lambda: Pipeline([("pre", pre_ohe_scaled(X_tr)),
                                      ("clf", KNeighborsClassifier())]),
            "grid": {"clf__n_neighbors": [7, 21], "clf__weights": ["uniform", "distance"]}
        },
    }

    comp_rows, fam_best = [], {}
    # per-family tuning on capped set
    for name, spec in families.items():
        best_params, best_f1, best_pipe = None, -np.inf, None
        for params in _param_grid_iter(spec["grid"]):
            pipe = spec["make"]()
            pipe.set_params(**params)
            pipe.fit(X_tune, y_tune)
            pred = pipe.predict(X_va)
            f1 = f1_score(y_va, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_params, best_pipe = f1, params, pipe
        # refit on full training with best params
        final_pipe = families[name]["make"]()
        if best_params: final_pipe.set_params(**best_params)
        final_pipe.fit(X_tr, y_tr)
        y_pred = final_pipe.predict(X_va)
        row = {
            "model": name,
            "params": best_params if best_params else {},
            "accuracy": accuracy_score(y_va, y_pred),
            "precision": precision_score(y_va, y_pred, zero_division=0),
            "recall": recall_score(y_va, y_pred, zero_division=0),
            "f1": f1_score(y_va, y_pred, zero_division=0),
        }
        try:
            proba = final_pipe.predict_proba(X_va)[:, 1]
            row["roc_auc"] = roc_auc_score(y_va, proba)
        except Exception:
            proba = None
        comp_rows.append(row)
        export_cls_preds(name, y_va, y_pred, proba, va, id_cols, artifacts_dir)
        fam_best[name] = {"pipe": final_pipe, "row": row}

    pd.DataFrame(comp_rows).to_csv(os.path.join(artifacts_dir, "classification_model_comparison.csv"), index=False)

    # pick global best by F1
    best_name = max(fam_best.keys(), key=lambda k: fam_best[k]["row"]["f1"])
    best_pipe, best_report = fam_best[best_name]["pipe"], fam_best[best_name]["row"]
    model_path = os.path.join(artifacts_dir, f"best_cls_{best_name}_{target_col}.joblib")
    joblib.dump(best_pipe, model_path)
    return ClassificationResult(best_name, best_report, model_path, artifacts_dir)

def train_regression_from_csv(
    csv_path: str, target_col: str, id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "artifacts/supervised", valid_ratio: float = 0.2, random_state: int = 42,
    tune_row_cap: Optional[int] = 50000,
) -> RegressionResult:
    os.makedirs(artifacts_dir, exist_ok=True)
    df = load_csv_dedup(csv_path)
    id_cols = id_cols or []
    tr, va = time_or_random_split(df, target_col, valid_ratio, random_state)

    drop_cols = [c for c in (set(id_cols) | EXCLUDE_ALWAYS | {target_col}) if c in tr.columns]
    y_tr, y_va = tr[target_col].values, va[target_col].values
    X_tr, X_va = tr.drop(columns=drop_cols, errors="ignore"), va.drop(columns=drop_cols, errors="ignore")

    if tune_row_cap is not None and len(X_tr) > tune_row_cap:
        X_tune, y_tune = X_tr.iloc[:tune_row_cap], y_tr[:tune_row_cap]
    else:
        X_tune, y_tune = X_tr, y_tr

    families = {
        "ridge": {
            "make": lambda: Pipeline([("pre", pre_ohe_scaled(X_tr)), ("reg", Ridge())]),
            "grid": {"reg__alpha": [0.3, 1.0, 3.0]}
        },
        "tree": {
            "make": lambda: Pipeline([("pre", pre_ordinal(X_tr)),
                                      ("reg", DecisionTreeRegressor(random_state=random_state))]),
            "grid": {"reg__max_depth": [3, 5, 9], "reg__min_samples_leaf": [1, 5]}
        },
        "knn": {
            "make": lambda: Pipeline([("pre", pre_ohe_scaled(X_tr)), ("reg", KNeighborsRegressor())]),
            "grid": {"reg__n_neighbors": [7, 15, 25], "reg__weights": ["uniform", "distance"]}
        },
    }

    comp_rows, fam_best = [], {}
    for name, spec in families.items():
        best_params, best_rmse, best_pipe = None, np.inf, None
        for params in _param_grid_iter(spec["grid"]):
            pipe = spec["make"]()
            pipe.set_params(**params)
            pipe.fit(X_tune, y_tune)
            pred = pipe.predict(X_va)
            rmse = mean_squared_error(y_va, pred, squared=False)
            if rmse < best_rmse:
                best_rmse, best_params, best_pipe = rmse, params, pipe
        final_pipe = families[name]["make"]()
        if best_params: final_pipe.set_params(**best_params)
        final_pipe.fit(X_tr, y_tr)
        y_pred = final_pipe.predict(X_va)
        row = {"model": name, "params": best_params if best_params else {},
               "MAE": mean_absolute_error(y_va, y_pred),
               "RMSE": mean_squared_error(y_va, y_pred, squared=False),
               "R2": r2_score(y_va, y_pred)}
        comp_rows.append(row)
        export_reg_preds(name, target_col, y_va, y_pred, va, id_cols, artifacts_dir)
        fam_best[name] = {"pipe": final_pipe, "row": row}

    pd.DataFrame(comp_rows).to_csv(os.path.join(artifacts_dir, f"regression_model_comparison_{target_col}.csv"), index=False)

    best_name = min(fam_best.keys(), key=lambda k: fam_best[k]["row"]["RMSE"])
    best_pipe, best_report = fam_best[best_name]["pipe"], fam_best[best_name]["row"]
    model_path = os.path.join(artifacts_dir, f"best_reg_{best_name}_{target_col}.joblib")
    joblib.dump(best_pipe, model_path)
    return RegressionResult(best_name, best_report, model_path, artifacts_dir)

# ---- inference ----
def predict(df_new: pd.DataFrame, model_or_path) -> np.ndarray:
    model = joblib.load(model_or_path) if isinstance(model_or_path, str) else model_or_path
    return model.predict(df_new)
