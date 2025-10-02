# project_package/modeling.py
# Goal:
# - Supervised: classification/regression with leakage guard + time-based split, tiny grids,
#   pick ONE best model, export: (1) per-row validation preds CSV, (2) split-assignments CSV,
#   (3) best pipeline .pkl (pickle only) + a one-line report CSV.
# - Unsupervised: k-means (auto-pick best k by silhouette), isolation forest, PCA embeddings;
#   export labels/scores/embeddings CSV + saved pipeline .pkl (pickle only).
# - Robust paths: auto-detect project root (folder that contains `project_package`),
#   auto-locate CSV in root or ./datasets, artifacts saved under <project_root>/Model/{supervised|unsupervised}.

from __future__ import annotations
import os, sys, pickle, fnmatch, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable

# ---------------- Project root & path helpers (works even if notebooks live under /Model) -----
def _find_project_root(marker_folder: str = "project_package", max_up: int = 8) -> str:
    cur = os.path.abspath(os.getcwd())
    for _ in range(max_up + 1):
        if os.path.isdir(os.path.join(cur, marker_folder)):
            return cur
        cur = os.path.abspath(os.path.join(cur, ".."))
    return os.path.abspath(os.getcwd())

_PROJECT_ROOT = _find_project_root()

def _resolve_csv_path(csv_path_or_name: str) -> str:
    """Accept absolute path, relative path, or bare filename.
    Try CWD, <project_root>, <project_root>/datasets, else recursive search."""
    if os.path.isabs(csv_path_or_name):
        if os.path.exists(csv_path_or_name):
            return csv_path_or_name
        raise FileNotFoundError(csv_path_or_name)

    cand = os.path.abspath(csv_path_or_name)
    if os.path.exists(cand):
        return cand

    cand2 = os.path.join(_PROJECT_ROOT, csv_path_or_name)
    if os.path.exists(cand2):
        return cand2

    fname = os.path.basename(csv_path_or_name)
    cand3 = os.path.join(_PROJECT_ROOT, "datasets", fname)
    if os.path.exists(cand3):
        return cand3

    matches = []
    for root, _, files in os.walk(_PROJECT_ROOT):
        for f in files:
            if f.lower() == fname.lower() or fnmatch.fnmatch(f.lower(), fname.lower()):
                matches.append(os.path.join(root, f))
    if matches:
        matches.sort(key=lambda p: len(os.path.relpath(p, _PROJECT_ROOT).split(os.sep)))
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate CSV '{csv_path_or_name}'. "
        f"Tried CWD, project root, and project_root/datasets. Project root = {_PROJECT_ROOT}"
    )

def _resolve_artifacts_dir(subdir: str) -> str:
    """Save outputs under <project_root>/<subdir>. If 'subdir' is relative, join to project root."""
    out = subdir
    if not os.path.isabs(out):
        out = os.path.join(_PROJECT_ROOT, subdir)
    os.makedirs(out, exist_ok=True)
    return out

# ---------------- scikit-learn core ----------------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Supervised models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Supervised metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)

# Unsupervised models & metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ========================== Leakage guard ==========================
# Post-outcome or tightly outcome-linked fields must never be used as features.
EXCLUDE_ALWAYS = {
    # Outcome / post-outcome or tightly coupled fields
    "Booking Status",

    # Cancellation / incomplete reasons
    "Reason for cancelling by Customer_fill",
    "Driver Cancellation Reason_fill",
    "Incomplete Rides Reason_fill",
    "Incomplete Rides_fill",
    "Cancelled Rides by Customer_fill",
    "Cancelled Rides by Driver_fill",

    # VTAT / CTAT (post-outcome-ish timing)
    "Avg VTAT_fill_scaled", "Avg VTAT_fill", "Avg VTAT",
    "Avg CTAT_fill_scaled", "Avg CTAT_fill", "Avg CTAT",
    "VTAT_missing_flag", "CTAT_missing_flag",

    # Booking value / ride distance (only certain after start/finish)
    "Booking Value_fill_scaled", "Booking Value_fill", "Booking Value",
    "Ride Distance_fill_scaled", "Ride Distance_fill", "Ride Distance",
    "BookingValue_missing_flag",

    # Ratings (likely observed after or strongly correlated with outcome)
    "Customer Rating_fill", "Customer Rating",
    "Driver Ratings_fill",  "Driver Ratings",

    # Destination geometry (typically confirmed after trip start)
    "drop_longitude", "drop_latitude", "drop_address", "drop_region", "drop_locality",
    "drop_station_latitude", "drop_station_longitude",

    # Dropoff-side weather (future information relative to booking time)
    "temperature_2m_dropoff_scaled", "relative_humidity_2m_dropoff_scaled",
    "dew_point_2m_dropoff_scaled", "apparent_temperature_dropoff_scaled",
    "wind_speed_10m_dropoff_scaled",
    "precipitation_dropoff_log_scaled", "rain_dropoff_log_scaled", "snowfall_dropoff_log_scaled",
}


# ============================== Utils ==============================
def load_csv_dedup(path: str) -> pd.DataFrame:
    """Read CSV, drop duplicate-named columns, downcast numerics for speed/memory."""
    real = _resolve_csv_path(path)
    df = pd.read_csv(real, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            df[c] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
    return df

def make_binary_target(df: pd.DataFrame, status_col="Booking Status") -> Tuple[pd.DataFrame, str]:
    """Derive binary target: contains 'completed' -> 1 else 0."""
    if status_col not in df.columns:
        raise KeyError(f"{status_col} not found")
    out = df.copy()
    out["_target_completed_"] = out[status_col].astype(str).str.lower().str.contains("completed").astype(int)
    return out, "_target_completed_"

def time_or_random_split(df: pd.DataFrame, target: str, valid_ratio=0.2, seed=42):
    """Prefer time split (booking_datetime). Else random; stratify for binary classification."""
    if "booking_datetime" in df.columns:
        d = df.copy()
        d["booking_datetime"] = pd.to_datetime(d["booking_datetime"], errors="coerce")
        d = d.sort_values("booking_datetime").reset_index(drop=True)
        cut = int(len(d) * (1 - valid_ratio))
        return d.iloc[:cut].copy(), d.iloc[cut:].copy()
    strat = None
    if target in df.columns and set(pd.unique(df[target].dropna())) <= {0, 1}:
        strat = df[target]
    tr, va = train_test_split(df, test_size=valid_ratio, random_state=seed, stratify=strat)
    return tr.copy(), va.copy()

def feat_types(X: pd.DataFrame, max_cat=200):
    """Return numeric and categorical columns. Cap high-cardinality to avoid huge OHE."""
    num, cat = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if s.nunique(dropna=False) <= max_cat:
                cat.append(c)
    return num, cat

def pre_ohe_scaled(X: pd.DataFrame) -> ColumnTransformer:
    """For linear/distance models: median-impute + standardize numerics; OHE cats."""
    num, cat = feat_types(X, max_cat=200)
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

def pre_ordinal(X: pd.DataFrame) -> ColumnTransformer:
    """For trees: median-impute numerics; most-freq + Ordinal for cats (unknown -> -1)."""
    num, cat = feat_types(X, max_cat=200)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
        ],
        remainder="drop",
    )

def _product(grid: Dict[str, Iterable]) -> Iterable[Dict[str, object]]:
    """Tiny cartesian product generator for very small grids."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    vals = [list(grid[k]) for k in keys]
    idx = [0] * len(keys)
    def rec(i):
        if i == len(keys):
            yield {keys[j]: vals[j][idx[j]] for j in range(len(keys))}
            return
        for t in range(len(vals[i])):
            idx[i] = t
            yield from rec(i + 1)
    yield from rec(0)

# =========================== Result holders =========================
@dataclass
class ClassificationResult:
    best_model_name: str
    report: Dict[str, float]
    model_path: str
    artifacts_dir: str
    preds_csv_path: str
    split_csv_path: str

@dataclass
class RegressionResult:
    best_model_name: str
    report: Dict[str, float]
    model_path: str
    artifacts_dir: str
    preds_csv_path: str
    split_csv_path: str

@dataclass
class ClusteringResult:
    best_k: int
    report: Dict[int, Dict[str, float]]   # k -> {silhouette, dbi, ch}
    model_path: str
    artifacts_dir: str
    labels_csv_path: str
    pca2_csv_path: str

@dataclass
class AnomalyResult:
    contamination: float
    model_path: str
    artifacts_dir: str
    scores_csv_path: str

@dataclass
class EmbeddingResult:
    method: str
    components: int
    artifacts_dir: str
    embed_csv_path: str

# ============================ Export helpers =========================
def _export_split_assignments(csv_path: str, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                              id_cols: List[str], target_col: str):
    """Write a simple file telling which rows were train vs. validation (for viz & audit)."""
    rows = []
    ids = id_cols or []
    for _, r in train_df.iterrows():
        row = {"split": "train", target_col: r[target_col]}
        for c in ids:
            if c in train_df.columns:
                row[c] = r.get(c, None)
        rows.append(row)
    for _, r in valid_df.iterrows():
        row = {"split": "valid", target_col: r[target_col]}
        for c in ids:
            if c in valid_df.columns:
                row[c] = r.get(c, None)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

def _export_cls_preds(csv_path: str, valid_df: pd.DataFrame, id_cols: List[str],
                      y_true: np.ndarray, y_pred: np.ndarray, proba: Optional[np.ndarray]):
    cols = {c: valid_df[c].values for c in (id_cols or []) if c in valid_df.columns}
    data = {"y_true": y_true, "y_pred": y_pred}
    if proba is not None:
        data["proba"] = proba
    pd.DataFrame({**cols, **data}).to_csv(csv_path, index=False)

def _export_reg_preds(csv_path: str, valid_df: pd.DataFrame, id_cols: List[str],
                      y_true: np.ndarray, y_pred: np.ndarray):
    cols = {c: valid_df[c].values for c in (id_cols or []) if c in valid_df.columns}
    out = pd.DataFrame({**cols, "y_true": y_true, "y_pred": y_pred})
    out["residual"] = out["y_true"] - out["y_pred"]
    out.to_csv(csv_path, index=False)

# ========================= Supervised: Train =========================
def train_classification_from_csv(
    csv_path: str,
    target_col: Optional[str] = None,     # None -> derive from Booking Status
    id_cols: Optional[List[str]] = None,  # IDs removed from X but kept in CSV exports
    artifacts_dir: str = "Model/supervised",
    valid_ratio: float = 0.2,
    random_state: int = 42,
    tune_row_cap: Optional[int] = 40000,
) -> ClassificationResult:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)

    df = load_csv_dedup(csv_path)
    if target_col is None or target_col not in df.columns:
        df, target_col = make_binary_target(df)

    ids = id_cols or []
    if ids:
        df = df.drop(columns=[c for c in ids if c in df.columns], errors="ignore")

    tr, va = time_or_random_split(df, target_col, valid_ratio, random_state)
    y_tr, y_va = tr[target_col].astype(int).values, va[target_col].astype(int).values
    drop_cols = [c for c in ({target_col} | EXCLUDE_ALWAYS | set(ids)) if c in tr.columns]
    X_tr, X_va = tr.drop(columns=drop_cols, errors="ignore"), va.drop(columns=drop_cols, errors="ignore")

    # Subset for quick tuning
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
                                      ("clf", RandomForestClassifier(n_estimators=200,
                                                                     class_weight="balanced",
                                                                     n_jobs=-1, random_state=random_state))]),
            "grid": {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 12]}
        },
        "knn": {
            "make": lambda: Pipeline([("pre", pre_ohe_scaled(X_tr)),
                                      ("clf", KNeighborsClassifier())]),
            "grid": {"clf__n_neighbors": [9, 21], "clf__weights": ["uniform", "distance"]}
        },
    }

    best_name, best_pipe, best_report, best_pred, best_proba = None, None, None, None, None
    for name, spec in families.items():
        # Tune on subset
        best_params, best_f1 = None, -np.inf
        for params in _product(spec["grid"]):
            p = spec["make"](); p.set_params(**params); p.fit(X_tune, y_tune)
            f1 = f1_score(y_va, p.predict(X_va), zero_division=0)
            if f1 > best_f1:
                best_f1, best_params = f1, params
        # Refit on full train and evaluate on validation
        final_p = spec["make"]()
        if best_params: final_p.set_params(**best_params)
        final_p.fit(X_tr, y_tr)
        pred = final_p.predict(X_va)
        rep = {
            "model": name,
            "accuracy": accuracy_score(y_va, pred),
            "precision": precision_score(y_va, pred, zero_division=0),
            "recall": recall_score(y_va, pred, zero_division=0),
            "f1": f1_score(y_va, pred, zero_division=0),
        }
        try:
            proba = final_p.predict_proba(X_va)[:, 1]
            rep["roc_auc"] = roc_auc_score(y_va, proba)
        except Exception:
            proba = None
        if best_report is None or rep["f1"] > best_report["f1"]:
            best_name, best_pipe, best_report, best_pred, best_proba = name, final_p, rep, pred, proba

    # Export split assignments
    split_csv = os.path.join(artifacts_dir, "split_assignments_classification.csv")
    _export_split_assignments(split_csv, tr, va, ids, target_col)

    # Export per-row validation predictions
    preds_csv = os.path.join(artifacts_dir, f"cls_best_{best_name}_preds.csv")
    _export_cls_preds(preds_csv, va, ids, y_va, best_pred, best_proba)

    # Save model + one-line report
    model_path = os.path.join(artifacts_dir, f"best_cls_{best_name}_{target_col}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    pd.DataFrame([best_report]).to_csv(os.path.join(artifacts_dir, f"cls_best_{best_name}_report.csv"), index=False)

    return ClassificationResult(best_name, best_report, model_path, artifacts_dir, preds_csv, split_csv)

def train_regression_from_csv(
    csv_path: str,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "Model/supervised",
    valid_ratio: float = 0.2,
    random_state: int = 42,
    tune_row_cap: Optional[int] = 40000,
) -> RegressionResult:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)

    df = load_csv_dedup(csv_path)
    ids = id_cols or []

    tr, va = time_or_random_split(df, target_col, valid_ratio, random_state)
    y_tr, y_va = tr[target_col].values, va[target_col].values
    drop_cols = [c for c in (set(ids) | EXCLUDE_ALWAYS | {target_col}) if c in tr.columns]
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

    best_name, best_pipe, best_report, best_pred = None, None, None, None
    for name, spec in families.items():
        best_params, best_rmse = None, np.inf
        for params in _product(spec["grid"]):
            p = spec["make"](); p.set_params(**params); p.fit(X_tune, y_tune)
            rmse = mean_squared_error(y_va, p.predict(X_va), squared=False)
            if rmse < best_rmse:
                best_rmse, best_params = rmse, params
        final_p = spec["make"]()
        if best_params: final_p.set_params(**best_params)
        final_p.fit(X_tr, y_tr)
        pred = final_p.predict(X_va)
        rep = {
            "model": name,
            "MAE": mean_absolute_error(y_va, pred),
            "RMSE": mean_squared_error(y_va, pred, squared=False),
            "R2": r2_score(y_va, pred),
        }
        if best_report is None or rep["RMSE"] < best_report["RMSE"]:
            best_name, best_pipe, best_report, best_pred = name, final_p, rep, pred

    safe_tgt = target_col.replace(' ', '_')
    split_csv = os.path.join(artifacts_dir, f"split_assignments_regression_{safe_tgt}.csv")
    _export_split_assignments(split_csv, tr, va, ids, target_col)

    preds_csv = os.path.join(artifacts_dir, f"reg_best_{safe_tgt}_{best_name}_preds.csv")
    _export_reg_preds(preds_csv, va, ids, y_va, best_pred)

    model_path = os.path.join(artifacts_dir, f"best_reg_{best_name}_{target_col}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)

    return RegressionResult(best_name, best_report, model_path, artifacts_dir, preds_csv, split_csv)

# ========================== Supervised: Inference ====================
def predict(df_new: pd.DataFrame, model_or_path) -> np.ndarray:
    """Load a saved Pipeline (or use a fitted one) and predict on raw DataFrame."""
    if isinstance(model_or_path, str):
        with open(model_or_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = model_or_path
    return model.predict(df_new)

# ========================= Unsupervised: Utils =======================
def _unsup_feat_types(X: pd.DataFrame, max_cat=200):
    num, cat = [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if s.nunique(dropna=False) <= max_cat:
                cat.append(c)
    return num, cat

def _unsup_pre_ordinal(X: pd.DataFrame) -> ColumnTransformer:
    """Dense-friendly preprocessing for unsupervised (KMeans, IsoForest): median/most_frequent + Ordinal."""
    num, cat = _unsup_feat_types(X, max_cat=200)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat),
        ],
        remainder="drop",
    )

def _build_unsup_X(df: pd.DataFrame, drop_cols: List[str]) -> Tuple[pd.DataFrame, ColumnTransformer, np.ndarray]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    pre = _unsup_pre_ordinal(X)
    X_t = pre.fit_transform(X)
    return X, pre, X_t

# ========================= Unsupervised: KMeans ======================
def run_kmeans_from_csv(
    csv_path: str,
    id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "Model/unsupervised",
    k_list: List[int] = (3, 5, 8, 10),
    random_state: int = 42,
) -> ClusteringResult:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)
    df = load_csv_dedup(csv_path)

    # reuse leakage guard columns to avoid outcome leakage in clusters
    drops = list(EXCLUDE_ALWAYS | set(id_cols or []))
    X_raw, pre, X = _build_unsup_X(df, drops)

    report: Dict[int, Dict[str, float]] = {}
    best_k, best_sil, best_model = None, -np.inf, None

    for k in k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)  # int for broad sklearn compatibility
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        report[k] = {"silhouette": float(sil), "dbi": float(dbi), "ch": float(ch)}
        if sil > best_sil:
            best_sil, best_k, best_model = sil, k, km

    labels = best_model.predict(X)
    lab_df = pd.DataFrame({"cluster": labels})
    for c in (id_cols or []):
        if c in df.columns:
            lab_df[c] = df[c].values
    labels_csv = os.path.join(artifacts_dir, f"kmeans_k{best_k}_labels.csv")
    lab_df.to_csv(labels_csv, index=False)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    pca2 = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels})
    for c in (id_cols or []):
        if c in df.columns:
            pca2[c] = df[c].values
    pca2_csv = os.path.join(artifacts_dir, f"kmeans_k{best_k}_pca2.csv")
    pca2.to_csv(pca2_csv, index=False)

    pipe = Pipeline([("pre", pre), ("kmeans", best_model)])
    model_path = os.path.join(artifacts_dir, f"kmeans_k{best_k}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    return ClusteringResult(best_k, report, model_path, artifacts_dir, labels_csv, pca2_csv)

# ===================== Unsupervised: Isolation Forest =================
def run_isolation_forest_from_csv(
    csv_path: str,
    id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "Model/unsupervised",
    contamination: float = 0.02,
    random_state: int = 42,
) -> AnomalyResult:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)
    df = load_csv_dedup(csv_path)

    drops = list(EXCLUDE_ALWAYS | set(id_cols or []))
    X_raw, pre, X = _build_unsup_X(df, drops)

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    scores = iso.fit_predict(X)  # -1 = outlier, 1 = inlier
    score_val = iso.score_samples(X)  # lower = more anomalous

    out = pd.DataFrame({"is_outlier": (scores == -1).astype(int), "anomaly_score": score_val})
    for c in (id_cols or []):
        if c in df.columns:
            out[c] = df[c].values
    scores_csv = os.path.join(artifacts_dir, "isoforest_scores.csv")
    out.to_csv(scores_csv, index=False)

    pipe = Pipeline([("pre", pre), ("iso", iso)])
    model_path = os.path.join(artifacts_dir, "isoforest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    return AnomalyResult(contamination, model_path, artifacts_dir, scores_csv)

# ======================== Unsupervised: PCA ==========================
def run_pca_embeddings_from_csv(
    csv_path: str,
    id_cols: Optional[List[str]] = None,
    artifacts_dir: str = "Model/unsupervised",
    n_components: int = 2,
    random_state: int = 42,
) -> EmbeddingResult:
    artifacts_dir = _resolve_artifacts_dir(artifacts_dir)
    df = load_csv_dedup(csv_path)
    drops = list(EXCLUDE_ALWAYS | set(id_cols or []))
    X_raw, pre, X = _build_unsup_X(df, drops)

    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)

    cols = {f"pc{i+1}": Z[:, i] for i in range(n_components)}
    out = pd.DataFrame(cols)
    for c in (id_cols or []):
        if c in df.columns:
            out[c] = df[c].values

    embed_csv = os.path.join(artifacts_dir, f"pca_{n_components}d.csv")
    out.to_csv(embed_csv, index=False)

    pipe = Pipeline([("pre", pre), ("pca", pca)])
    model_pkl = os.path.join(artifacts_dir, f"pca_{n_components}d.pkl")
    with open(model_pkl, "wb") as f:
        pickle.dump(pipe, f)

    return EmbeddingResult("pca", n_components, artifacts_dir, embed_csv)
