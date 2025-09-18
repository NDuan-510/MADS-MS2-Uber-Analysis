Owner: Howard
Goal:Train baseline classifiers to predict ride completion vs. cancellation, and export evaluation metrics for visualization.

TL;DR

- Input datasetsprocessedncr_ride_bookings_with_weather_filled_scaled_short.csv
- Target Booking Status → binary mapping via explicit label set
- Models Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- CV Stratified K-Fold (default 5 folds, auto-safe when minority is small)
- Output (for Viz) artifactsmetricsmetrics.csv — longtidy metrics table (≈ models × folds × metrics)
- Repro Entire pipeline lives in data_modelling.ipynb (Cells 1–7)

1) Data Contract (from Data Processing)

We assume the processed table follows these column suffix conventions

- _scaled, _log_scaled → numeric features already scaled (e.g., Avg CTAT_fill_scaled, rain_log_scaled)
- _fill → imputed feature (may be numeric or categorical)
- _missing_flag → 01 indicator that the original value was missing
- Vehicle Type → categorical (kept as raw, one-hot later)

Additional whitelisted raw numerics (no suffix but included)
booking_hour, pick_longitudelatitude, drop_longitudelatitude, pick_station_, drop_station_

Excluded from modeling (free text, IDs, raw date strings)
Booking ID, Customer ID, Date, Time, booking_datetime, addresses & regionlocality columns.

2) What Each Cell Does (Cells 1–7)
Cell 1 — Imports & Folders
-Imports pandas/sklearn; creates artifact folders:
-artifacts/metrics, artifacts/models, artifacts/plots
-Purpose: consistent output locations for downstream use.

Cell 2 — Feature Inference (with whitelist & safety)

-Deduplicates duplicate column names.
-Automatically builds three lists:
-numeric: all *_scaled/*_log_scaled + numeric *_fill + whitelisted raw numerics
-categorical: non-numeric *_fill + Vehicle Type
-flags: *_missing_flag
-Guards against accidental inclusion of non-numeric columns in the numeric list.

Cell 3 — Preprocessor

-Builds a ColumnTransformer:
-One-Hot for categorical features (handle_unknown="ignore")
-Passthrough for numeric + missing flags (already scaled / 0-1)

Cell 4 — Model Registry

Defines baseline models:
-logreg_l2: LogisticRegression (class_weight="balanced")
-dtree: DecisionTreeClassifier (class_weight="balanced")
-rf_300: RandomForestClassifier (300 trees, class_weight="balanced")
-gbdt: GradientBoostingClassifier
-Keeps a consistent dictionary {model_name: estimator}.

Cell 5 — Exact Label Mapping & Safe Stratified CV

-Converts Booking Status → binary with explicit label set:
    -Positive (1): {Cancelled by Customer, Cancelled by Driver, Incomplete, No Driver Found}
    -Negative (0): {Completed}
-Safe CV helper:
    -Uses StratifiedKFold
    -Automatically reduces folds to ≤ minority count; skips degenerate folds (single class).