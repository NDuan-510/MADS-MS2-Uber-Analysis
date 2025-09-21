# 📘 Milestone II · Data Modeling Results - Howard Lin

## ⚡ TL;DR
- 📁 Input: processed Uber 2024 dataset → model training done  
- 📊 Output: `artifacts/metrics/metrics.csv` (125 rows × 7 columns)  
- 🖼️ Use it to plot model performance (leaderboards, boxplots, heatmaps)

---

## 🎯 What This Is
This notebook trains several machine learning models to predict **ride completion vs. cancellation** using the processed Uber 2024 dataset.

---

## 📂 Input Dataset
- File: `datasets/processed/ncr_ride_bookings_with_weather_filled_scaled_short.csv`
- Target: **`Booking Status`**
  - Positive = `Cancelled by Customer`, `Cancelled by Driver`, `Incomplete`, `No Driver Found`
  - Negative = `Completed`
---

## ⚙️ What I Did
I trained and evaluated the following models using 5-fold stratified cross-validation:

| Category               | Models included            |
|------------------------|----------------------------|
| Logistic Regression    | `logreg_l2`                |
| Decision Tree          | `dtree`                    |
| Ensemble Methods       | `rf_300` (Random Forest), `gbdt` (Gradient Boosted Trees) |
| Baseline (extra)       | `dummy_mf` (most frequent) |

Each model was evaluated on these metrics:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

---

## 📤 What You Get
- **`artifacts/metrics/metrics.csv`**  
  - Tidy long format (ready for plotting)  
  - Columns:  
    `run_id, timestamp, model_name, fold, metric, value, params`
  - Shape: **125 rows × 7 columns**  
    (5 models × 5 folds × 5 metrics)
- Each row = one metric value from one fold of one model

---


