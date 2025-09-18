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
