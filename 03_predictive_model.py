"""
Step 3: Predictive Modeling (03_predictive_model.py)
----------------------------------------------------
Trains, evaluates, and visualizes the Real-Time Incident Severity Classifier
using a strict chronological (out-of-time) train/test split:

  Real-Time Incident Severity Classifier:
    - Unit: Individual Train Incident (Type d'incident == 'T')
    - Target: Major Disruption >= 5 minutes (is_major_delay_5m: 1 vs 0)
    - Split: Train 2019–2023 (n=21,058) -> Test 2024–2025 (n=8,109)
             (Also evaluates 2019–2023 subset with Planned KM feature)
    - Algorithms: Logistic Regression (Baseline), Random Forest (Main)

Outputs metrics to outputs/models/ and diagnostic plots to outputs/figures/:
  - outputs/figures/05_modelA_severity_evaluation.png
"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = os.path.join("outputs", "data")
FIG_DIR = os.path.join("outputs", "figures")
MODEL_DIR = os.path.join("outputs", "models")


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


# ==============================================================================
# MODEL A: REAL-TIME INCIDENT SEVERITY CLASSIFIER (MAJOR DELAY >= 5 MIN)
# ==============================================================================
def run_model_a_severity_classifier(df_inc: pd.DataFrame, df_panel: pd.DataFrame):
    print("=" * 75)
    print("MODEL A: REAL-TIME INCIDENT SEVERITY CLASSIFIER (Major Delay >= 5 min)")
    print("=" * 75)

    # Filter strictly to Type d'incident == 'T' (since 'S' rows have hardcoded duration)
    t_df = df_inc[df_inc["Type d'incident"] == "T"].copy()
    t_df = t_df.sort_values(["Jour calendaire", "heure_incident_min"]).reset_index(
        drop=True
    )

    # Merge strictly lagged daily line context from daily_line_panel (no leakage)
    panel_lags = df_panel[
        ["Ligne", "Jour calendaire", "roll7d_incidents_T", "roll7d_major_delays"]
    ].rename(columns={"Ligne": "ligne_clean"})
    t_df = t_df.merge(panel_lags, on=["ligne_clean", "Jour calendaire"], how="left")
    t_df["roll7d_incidents_T"] = t_df["roll7d_incidents_T"].fillna(
        t_df["roll7d_incidents_T"].median()
    )
    t_df["roll7d_major_delays"] = t_df["roll7d_major_delays"].fillna(
        t_df["roll7d_major_delays"].median()
    )

    # Additional pre-onset operational features
    t_df["has_vehicule_id"] = (t_df["vehicule_clean"] != "Inconnu").astype(int)
    t_df["has_tournee_id"] = (t_df["tournee_clean"] != "Inconnu").astype(int)

    # Keep top 50 stations as explicit categories, group rare locations into 'Autre_Lieu'
    top_stations = set(t_df["Code de lieu"].value_counts().head(50).index)
    t_df["lieu_groupe"] = t_df["Code de lieu"].where(
        t_df["Code de lieu"].isin(top_stations), "Autre_Lieu"
    )

    # Out-of-time split: Train 2019-2023, Test 2024-2025
    train_mask = t_df["Année civile"] <= 2023
    test_mask = t_df["Année civile"] >= 2024

    train_df = t_df[train_mask].copy()
    test_df = t_df[test_mask].copy()

    # Compute out-of-fold / training-only smoothed historical station major-delay rate
    global_prior = train_df["is_major_delay_5m"].mean()
    loc_stats = train_df.groupby("Code de lieu")["is_major_delay_5m"].agg(
        ["count", "mean"]
    )
    smoothing_m = 20.0
    loc_smoothed = (
        (loc_stats["count"] * loc_stats["mean"] + smoothing_m * global_prior)
        / (loc_stats["count"] + smoothing_m)
    ).to_dict()

    train_df["station_hist_major_rate"] = (
        train_df["Code de lieu"].map(loc_smoothed).fillna(global_prior)
    )
    test_df["station_hist_major_rate"] = (
        test_df["Code de lieu"].map(loc_smoothed).fillna(global_prior)
    )

    cat_cols = [
        "ligne_clean",
        "Cause primaire",
        "Cause secondaire",
        "Symptome",
        "flotte_train",
        "porte_voiture_clean",
        "lieu_groupe",
    ]
    num_cols = [
        "CAT",
        "Porte",
        "KFS",
        "Urgence métro",
        "Dommage matériel",
        "has_vehicule_id",
        "has_tournee_id",
        "is_code_technique_lieu",
        "heure_operationnelle",
        "is_pointe_am",
        "is_pointe_pm",
        "is_apres_minuit",
        "Mois calendrier",
        "Jour de la semaine",
        "station_hist_major_rate",
        "roll7d_incidents_T",
        "roll7d_major_delays",
    ]
    feature_cols = cat_cols + num_cols
    target_col = "is_major_delay_5m"

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    print(
        f"Train Period: 2019–2023 (n={len(X_train):,}, Major delay rate={y_train.mean() * 100:.2f}%)"
    )
    print(
        f"Test Period : 2024–2025 (n={len(X_test):,}, Major delay rate={y_test.mean() * 100:.2f}%)"
    )

    # Preprocessors
    ohe_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ]
    )

    models = {
        "Logistic Regression (Baseline)": Pipeline(
            [
                ("prep", ohe_preprocessor),
                ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
            ]
        ),
        "Random Forest (Main)": Pipeline(
            [
                ("prep", ohe_preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=10,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    metrics_rows = []
    preds_dict = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob_test = pipe.predict_proba(X_test)[:, 1]
        preds_dict[name] = prob_test

        roc_auc = roc_auc_score(y_test, prob_test)
        pr_auc = average_precision_score(y_test, prob_test)
        brier = brier_score_loss(y_test, prob_test)

        # Optimal F1 threshold tuned on training set
        prob_train = pipe.predict_proba(X_train)[:, 1]
        prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_train, prob_train)
        f1_tr = 2 * prec_tr * rec_tr / (prec_tr + rec_tr + 1e-9)
        best_th = thresh_tr[np.argmax(f1_tr[:-1])]

        pred_label = (prob_test >= best_th).astype(int)
        acc = accuracy_score(y_test, pred_label)
        prec = precision_score(y_test, pred_label, zero_division=0)
        rec = recall_score(y_test, pred_label, zero_division=0)
        f1 = f1_score(y_test, pred_label)

        # Top-decile lift on test set
        top_10pct_cutoff = np.quantile(prob_test, 0.90)
        top_decile_actual_rate = y_test[prob_test >= top_10pct_cutoff].mean()
        lift_top10 = top_decile_actual_rate / y_test.mean()

        metrics_rows.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4),
                "PR_AUC": round(pr_auc, 4),
                "Brier_Score": round(brier, 4),
                "Best_Threshold": round(float(best_th), 3),
                "Top_10pct_Actual_Major_Rate": round(float(top_decile_actual_rate), 4),
                "Top_10pct_Lift": round(float(lift_top10), 2),
            }
        )

    df_metrics_a = pd.DataFrame(metrics_rows)
    print("\n--- Model A Test Set Performance (2024–2025 Out-of-Time) ---")
    print(df_metrics_a.to_string(index=False))

    # Also test adding Planned KM on 2019-2023 subset (Train 2019-2022 -> Test 2023)
    sub_km = train_df[train_df["km_planifie"].notna()].copy()
    sub_train = sub_km[sub_km["Année civile"] <= 2022].copy()
    sub_test = sub_km[sub_km["Année civile"] == 2023].copy()
    sub_prior = sub_train["is_major_delay_5m"].mean()
    sub_loc = sub_train.groupby("Code de lieu")["is_major_delay_5m"].agg(
        ["count", "mean"]
    )
    sub_map = (
        (sub_loc["count"] * sub_loc["mean"] + 20.0 * sub_prior)
        / (sub_loc["count"] + 20.0)
    ).to_dict()
    sub_train["station_hist_major_rate"] = (
        sub_train["Code de lieu"].map(sub_map).fillna(sub_prior)
    )
    sub_test["station_hist_major_rate"] = (
        sub_test["Code de lieu"].map(sub_map).fillna(sub_prior)
    )
    km_cols = feature_cols + ["km_planifie"]
    rf_km = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            cat_cols,
                        ),
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            num_cols + ["km_planifie"],
                        ),
                    ]
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    rf_km.fit(sub_train[km_cols], sub_train[target_col])
    prob_2023 = rf_km.predict_proba(sub_test[km_cols])[:, 1]
    print(
        f"\n[Sub-experiment 2019–2022 -> 2023 with Planned KM feature] "
        f"Test 2023 ROC-AUC: {roc_auc_score(sub_test[target_col], prob_2023):.4f}, "
        f"PR-AUC: {average_precision_score(sub_test[target_col], prob_2023):.4f}"
    )

    # Permutation Importance for Main Model (Random Forest)
    main_pipe = models["Random Forest (Main)"]
    perm = permutation_importance(
        main_pipe,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=42,
        n_jobs=-1,
    )
    df_imp_a = (
        pd.DataFrame(
            {
                "Feature": feature_cols,
                "Importance_Mean": perm.importances_mean,
                "Importance_Std": perm.importances_std,
            }
        )
        .sort_values("Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )
    print(
        "\n--- Top 10 Features Driving Incident Severity (Permutation ROC-AUC Drop) ---"
    )
    print(df_imp_a.head(10).to_string(index=False))

    # Save metrics & feature importances
    df_metrics_a.to_csv(
        os.path.join(MODEL_DIR, "modelA_severity_metrics.csv"), index=False
    )
    df_imp_a.to_csv(
        os.path.join(MODEL_DIR, "modelA_feature_importance.csv"), index=False
    )

    # Plot Model A Evaluation Figure (4 panels)
    plot_model_a_diagnostics(y_test, preds_dict, df_imp_a)


def plot_model_a_diagnostics(
    y_test: np.ndarray, preds_dict: dict, df_imp: pd.DataFrame
):
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    model_colors = {
        "Logistic Regression (Baseline)": "#5b9bd5",
        "Random Forest (Main)": "#c00000",
    }

    # Panel A: ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, probs in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        lw = 2.6 if "Main" in name else 1.8
        ax1.plot(
            fpr,
            tpr,
            linewidth=lw,
            color=model_colors[name],
            label=f"{name} (AUC = {auc_val:.3f})",
        )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier (AUC = 0.500)")
    ax1.set_title("A. ROC Curves (Out-of-Time Test Set: 2024–2025)")
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR / Recall)")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    # Panel B: Precision-Recall Curves
    ax2 = fig.add_subplot(gs[0, 1])
    base_rate = y_test.mean()
    for name, probs in preds_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap_val = average_precision_score(y_test, probs)
        lw = 2.6 if "Main" in name else 1.8
        ax2.plot(
            rec,
            prec,
            linewidth=lw,
            color=model_colors[name],
            label=f"{name} (PR-AUC = {ap_val:.3f})",
        )
    ax2.axhline(
        base_rate,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline Positive Rate ({base_rate * 100:.1f}%)",
    )
    ax2.set_title("B. Precision-Recall Curves (Test Set: 2024–2025)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel C: Top 12 Permutation Feature Importances
    ax4 = fig.add_subplot(gs[1, :])
    top12 = df_imp.head(12).iloc[::-1]
    readable_names = {
        "CAT": "Traction Power Cut (CAT)",
        "Symptome": "Initial Symptom (Symptome)",
        "Cause secondaire": "Secondary Cause (Cause secondaire)",
        "Urgence métro": "Metro Emergency (Urgence métro)",
        "Cause primaire": "Primary Cause (Cause primaire)",
        "has_tournee_id": "Train Run Identified (Numéro de tournée)",
        "Porte": "Door Fault Indicator (Porte)",
        "heure_operationnelle": "Operational Hour (Heure de l'incident)",
        "KFS": "Emergency Track Switch (KFS)",
        "has_vehicule_id": "Train Car Identified (Véhicule)",
        "station_hist_major_rate": "Station Historical Major-Delay Rate",
        "lieu_groupe": "Station / Location (Code de lieu)",
        "ligne_clean": "Metro Line (Ligne)",
        "flotte_train": "Train Fleet (MR-73 vs. Azur)",
        "porte_voiture_clean": "Car Door Number (Porte de voiture)",
        "Mois calendrier": "Calendar Month (Mois calendrier)",
        "roll7d_incidents_T": "7-Day Rolling Line Incidents (Type T)",
        "roll7d_major_delays": "7-Day Rolling Major Delays",
    }
    labels = [readable_names.get(f, f) for f in top12["Feature"]]
    y_pos = np.arange(len(top12))
    ax4.barh(
        y_pos,
        top12["Importance_Mean"],
        xerr=top12["Importance_Std"],
        color="#1f4e78",
        alpha=0.88,
        edgecolor="black",
        capsize=3,
    )
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_title("C. Top 12 Feature Importances (Permutation ROC-AUC Drop)")
    ax4.set_xlabel("Mean Decrease in Test ROC-AUC When Permuted")

    fig.suptitle(
        "Model A Evaluation: Real-Time Prediction of Major Delays (≥ 5 min)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "05_modelA_severity_evaluation.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    setup_style()

    print("Loading cleaned datasets from outputs/data/...")
    df_inc = pd.read_csv(
        os.path.join(DATA_DIR, "incidents_clean.csv"), low_memory=False
    )
    df_panel = pd.read_csv(
        os.path.join(DATA_DIR, "daily_line_panel.csv"), low_memory=False
    )

    run_model_a_severity_classifier(df_inc, df_panel)
    print("\nModeling task completed successfully.")


if __name__ == "__main__":
    main()
