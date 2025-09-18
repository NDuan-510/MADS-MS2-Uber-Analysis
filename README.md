Owner: Howard
Goal:Train baseline classifiers to predict ride completion vs. cancellation, and export evaluation metrics for visualization.

TL;DR

- Input datasetsprocessedncr_ride_bookings_with_weather_filled_scaled_short.csv
- Target Booking Status → binary mapping via explicit label set
- Models Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- CV Stratified K-Fold (default 5 folds, auto-safe when minority is small)
- Output (for Viz) artifactsmetricsmetrics.csv — longtidy metrics table (≈ models × folds × metrics)
- Repro Entire pipeline lives in data_modelling.ipynb (Cells 1–7)
# Data Modeling — Milestone II

---

## How It Works

**Input:**  
- `datasets/processed/ncr_ride_bookings_with_weather_filled_scaled_short.csv`

**Target:**  
- `Booking Status`
  - Positive = `Cancelled by Customer`, `Cancelled by Driver`, `Incomplete`, `No Driver Found`
  - Negative = `Completed`

**Features:**  
- Numeric: `*_scaled`, `*_log_scaled`, numeric `*_fill`, plus `booking_hour`, coordinates
- Categorical: text `*_fill`, `Vehicle Type`
- Missing flags: `*_missing_flag`

**Models (5-fold CV):**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting

---

## How to Run

1. Open `data_modelling.ipynb`
2. Run Cells **1 → 7**  
3. Outputs will be saved to:

