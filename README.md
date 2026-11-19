# Montreal Metro Incident Analysis & Delay Prediction (DS Challenge)

This project cleans, visualizes, and models Montreal Metro (STM) incident and planned kilometrage data (`2019–2025`) to predict whether a train service incident will cause a **Major Delay (≥ 5 minutes)** or a **Minor Delay (≤ 4 minutes)**.

## Project Structure

- `01_clean_data.py` — Cleans the two raw `.xls` (CSV text) files, fixes formatting/duplicate issues, and saves clean datasets to `outputs/data/`.
- `02_visualize_data.py` — Generates 4 exploratory charts in `outputs/figures/` covering annual/seasonal trends, causes & rolling stock, hourly rush-hour patterns, and hotspot stations.
- `03_predictive_model.py` — Trains and evaluates **Logistic Regression (Baseline)** and **Random Forest (Main Model)** on train incidents (`Type T`) using a chronological split (Train: `2019–2023` [72%], Test: `2024–2025` [28%]), saving metrics to `outputs/models/` and evaluation plots to `outputs/figures/`.

## Key Results

- **Random Forest (Out-of-Time Test Set 2024–2025):**
  - **Accuracy:** 80.0% | **ROC-AUC:** 0.832 | **PR-AUC:** 0.611 | **F1-Score:** 0.577
  - **Top-10% Risk Alert Accuracy:** 71.0% actual major delays (**3.1x lift** over the 22.9% baseline rate).
- **Top Drivers of Major Delays:** Traction power cut (`CAT`), identified train run (`Numéro de tournée`), station historical severity rate, initial symptom (`Symptome`), secondary cause (`Cause secondaire`), and emergency response (`Urgence métro`).

## How to Run

```bash
python3 01_clean_data.py
python3 02_visualize_data.py
python3 03_predictive_model.py
```
