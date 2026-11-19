"""
Step 1: Data Cleaning & Feature Engineering (01_clean_data.py)
--------------------------------------------------------------
Reads the two raw STM Metro files (.xls extension, actually CSV text files):
  1. Incidents métro_20260707.xls (semicolon-delimited UTF-8 with BOM)
  2. Kilométrage métro planifé_20260707.xls (comma-delimited UTF-8 with French decimals)

Fixes data quality issues and outputs cleaned datasets to outputs/data/:
  - outputs/data/km_clean.csv
  - outputs/data/incidents_clean.csv
  - outputs/data/daily_line_panel.csv
"""

import glob
import os
import re
import numpy as np
import pandas as pd


OUTPUT_DIR = os.path.join("outputs", "data")
MAIN_LINES = ["Ligne verte", "Ligne orange", "Ligne bleue", "Ligne jaune"]

# STM Line number to canonical name mapping
LINE_NUM_TO_NAME = {
    "1": "Ligne verte",
    "2": "Ligne orange",
    "4": "Ligne jaune",
    "5": "Ligne bleue",
}

DURATION_ORDER = [
    "02 min et moins",
    "03 à 04 min",
    "05 à 09 min",
    "10 à 14 min",
    "15 à 19 min",
    "20 à 24 min",
    "25 à 29 min",
    "30 min et plus",
]

DURATION_MIDPOINT_MIN = {
    "02 min et moins": 1.0,
    "03 à 04 min": 3.5,
    "05 à 09 min": 7.0,
    "10 à 14 min": 12.0,
    "15 à 19 min": 17.0,
    "20 à 24 min": 22.0,
    "25 à 29 min": 27.0,
    "30 min et plus": 45.0,
}


def clean_kilometrage(filepath: str) -> pd.DataFrame:
    """
    Cleans the Planned Kilometrage dataset.
    Resolves the 1,536 duplicate/split (Ligne, Jour calendaire) records in 2019-2020
    where one row holds metadata (Période Exploitation, Type de Jour) with KM=0
    and the paired row holds '#' / 'Non affecté' with KM > 0.
    """
    print(f"[1/3] Loading raw kilometrage file: {filepath}")
    df_raw = pd.read_csv(filepath, sep=",", encoding="utf-8-sig", low_memory=False)
    print(f"      Raw shape: {df_raw.shape}")

    # Convert French decimal comma to float
    df_raw["km_planifie"] = (
        df_raw["KM planifié"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # Helper to pick the valid (non-'#', non-'Non affecté') value per group
    def pick_valid(series: pd.Series, invalid_tokens: set) -> str:
        valid = [v for v in series if pd.notna(v) and str(v) not in invalid_tokens]
        return valid[0] if valid else series.iloc[0]

    df_clean = (
        df_raw.groupby(["Ligne", "Jour calendaire"], as_index=False)
        .agg(
            periode_exploitation=("Période Exploitation", lambda s: pick_valid(s, {"#"})),
            type_de_jour=("Type de Jour", lambda s: pick_valid(s, {"Non affecté", "#"})),
            annee_civile=("Année civile", "first"),
            annee_mois=("Année civile/mois", "first"),
            annee_semaine=("Année civile/semaine", "first"),
            jour_semaine_nom=("Jour de la semaine", "first"),
            mois_calendrier=("Mois calendrier", "first"),
            km_planifie=("km_planifie", "sum"),
            n_raw_rows=("km_planifie", "size"),
        )
        .sort_values(["Jour calendaire", "Ligne"])
        .reset_index(drop=True)
    )

    # Validate cleaning
    n_zero_km = (df_clean["km_planifie"] <= 0).sum()
    n_hash_periode = (df_clean["periode_exploitation"] == "#").sum()
    n_na_type = (df_clean["type_de_jour"] == "Non affecté").sum()
    print(
        f"      Cleaned shape: {df_clean.shape} "
        f"(merged {len(df_raw) - len(df_clean)} split/duplicate rows)"
    )
    print(
        f"      Validation -> Zero KM rows: {n_zero_km}, "
        f"'#' Période rows: {n_hash_periode}, 'Non affecté' Type de Jour rows: {n_na_type}"
    )
    return df_clean


def standardize_ligne(val: str):
    """
    Standardizes the 'Ligne' column in Incidents:
      - Fixes Excel date corruption ('02-avr' -> 'Ligne 2, 4')
      - Canonicalizes multi-line strings ('Ligne 2, 1' -> 'Ligne 1, 2')
      - Returns (canonical_ligne, primary_category, impacted_lines_list)
    """
    val_str = str(val).strip()
    if val_str in MAIN_LINES:
        return val_str, val_str, [val_str]
    if val_str == "Non affecté" or pd.isna(val):
        return "Inconnu", "Inconnu", []
    if val_str == "02-avr":
        # Excel converted "2, 4" or "2-4" into April 2 ("02-avr")
        nums = ["2", "4"]
    else:
        nums = sorted(set(re.findall(r"[1245]", val_str)))

    if nums:
        canonical = "Ligne " + ", ".join(nums)
        impacted = [LINE_NUM_TO_NAME[n] for n in nums if n in LINE_NUM_TO_NAME]
        return canonical, "Multi-lignes", impacted
    return "Inconnu", "Inconnu", []


def parse_28h_time_to_minutes(t_str: str) -> float:
    """
    Parses STM's 28-hour clock strings ('HH:MM' or 'HH:MM:SS', e.g., '24:58:00')
    into total minutes from 00:00 of the operational day.
    """
    if pd.isna(t_str):
        return np.nan
    parts = str(t_str).strip().split(":")
    try:
        h = int(parts[0])
        m = int(parts[1])
        return float(h * 60 + m)
    except (ValueError, IndexError):
        return np.nan


def clean_incidents(filepath: str, df_km: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Incidents dataset and engineers features for EDA and modeling.
    """
    print(f"\n[2/3] Loading raw incidents file: {filepath}")
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig", low_memory=False)
    print(f"      Raw shape: {df.shape}")

    # 1. Standardize Ligne and extract impacted lines
    ligne_parsed = df["Ligne"].map(standardize_ligne)
    df["ligne_clean"] = [x[0] for x in ligne_parsed]
    df["ligne_categorie"] = [x[1] for x in ligne_parsed]
    impacted_lists = [x[2] for x in ligne_parsed]

    for line_name in MAIN_LINES:
        col_flag = "impacts_" + line_name.split()[1]
        df[col_flag] = [int(line_name in lst) for lst in impacted_lists]

    # 2. Clean missing value tokens ('#', 'Non affecté', NaN) & trailing semicolons
    df["Cause primaire"] = df["Cause primaire"].fillna("Inconnu").replace({"#": "Inconnu"})
    df["Cause secondaire"] = df["Cause secondaire"].fillna("Inconnu").replace({"#": "Inconnu"})
    df["Symptome"] = df["Symptome"].fillna("Inconnu").replace({"#": "Inconnu"})

    df["porte_voiture_clean"] = (
        df["Porte de voiture"]
        .astype(str)
        .str.rstrip(";")
        .str.strip()
        .replace({"#": "Inconnu", "nan": "Inconnu"})
    )
    df["vehicule_clean"] = df["Véhicule"].astype(str).replace({"#": "Inconnu"})
    df["tournee_clean"] = df["Numéro de tournée"].astype(str).replace({"#": "Inconnu"})
    df["evacuation_clean"] = df["Évacuation"].astype(str).replace({"#": "Aucune"})

    # Map Type de matériel to readable train fleet names
    df["flotte_train"] = df["Type de matériel"].map(
        {"73": "MR-73", "10": "MPM-10 (Azur)", "Non affecté": "Non spécifié"}
    ).fillna("Non spécifié")

    # 3. Parse 28-hour clock timestamps and compute elapsed time
    df["heure_incident_min"] = df["Heure de l'incident"].map(parse_28h_time_to_minutes)
    df["heure_reprise_min"] = df["Heure de reprise"].map(parse_28h_time_to_minutes)

    # Elapsed minutes between incident and recovery (handle rare wrap-arounds)
    elapsed = df["heure_reprise_min"] - df["heure_incident_min"]
    elapsed = np.where(elapsed < 0, elapsed + 1440.0, elapsed)
    df["duree_reelle_min"] = elapsed

    df["heure_operationnelle"] = (df["heure_incident_min"] // 60).astype(int)
    df["heure_24h"] = df["heure_operationnelle"] % 24
    df["is_apres_minuit"] = (df["heure_operationnelle"] >= 24).astype(int)

    # Rush hour indicators (AM: 06:00-08:59, PM: 15:00-18:29)
    df["is_pointe_am"] = df["heure_incident_min"].between(360, 539).astype(int)
    df["is_pointe_pm"] = df["heure_incident_min"].between(900, 1109).astype(int)
    df["is_heure_pointe"] = ((df["is_pointe_am"] == 1) | (df["is_pointe_pm"] == 1)).astype(int)

    # 4. Operational date alignment (fix post-midnight calendar vs weekday offset)
    cal_dt = pd.to_datetime(df["Jour calendaire"])
    cal_dow = cal_dt.dt.dayofweek + 1  # 1=Monday .. 7=Sunday
    df["is_decalage_jour_exploitation"] = (df["Jour de la semaine"] != cal_dow).astype(int)
    df["jour_exploitation"] = np.where(
        df["is_decalage_jour_exploitation"] == 1,
        (cal_dt - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
        df["Jour calendaire"],
    )

    # 5. Severity targets (meaningful for Type d'incident == 'T')
    duration_rank_map = {label: idx for idx, label in enumerate(DURATION_ORDER)}
    df["severite_rang"] = df["Incident en minutes"].map(duration_rank_map)
    df["duree_estimee_min"] = df["Incident en minutes"].map(DURATION_MIDPOINT_MIN)

    # Major delay target: >= 5 minutes (1) vs <= 4 minutes (0)
    df["is_major_delay_5m"] = (df["severite_rang"] >= 2).astype(int)
    # Any delay > 2 minutes: (1) vs <= 2 minutes (0)
    df["is_delay_gt2m"] = (df["severite_rang"] >= 1).astype(int)

    # Classify station vs inter-station/track code in Code de lieu
    # Track/switch codes typically look like 'BR42', 'Interstation', etc.
    df["is_code_technique_lieu"] = (
        df["Code de lieu"].str.match(r"^[A-Z]{2,3}\d+", na=False).astype(int)
    )

    # 6. Merge planned KM & Type de Jour where single-line & within 2019-2023
    km_lookup = df_km[
        ["Ligne", "Jour calendaire", "km_planifie", "type_de_jour", "periode_exploitation"]
    ].rename(columns={"Ligne": "ligne_clean"})
    df = df.merge(km_lookup, on=["ligne_clean", "Jour calendaire"], how="left")

    print(f"      Cleaned incidents shape: {df.shape}")
    print(
        f"      Type T count: {(df['Type d\'incident'] == 'T').sum():,}, "
        f"Type S count: {(df['Type d\'incident'] == 'S').sum():,}"
    )
    t_df = df[df["Type d'incident"] == "T"]
    print(
        f"      Among Type T -> Major delays (>=5 min): {t_df['is_major_delay_5m'].sum():,} "
        f"({t_df['is_major_delay_5m'].mean()*100:.1f}%), "
        f"Delays >2 min: {t_df['is_delay_gt2m'].sum():,} ({t_df['is_delay_gt2m'].mean()*100:.1f}%)"
    )
    return df


def build_daily_line_panel(df_inc: pd.DataFrame, df_km: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a complete daily panel per (Ligne, Jour calendaire) from 2019-01-01 to 2025-09-30
    with daily incident counts (by type, severity, and cause), planned KM (2019-2023),
    normalized incident rates per 100k KM, and strictly lagged rolling features for forecasting.
    """
    print("\n[3/3] Building daily line panel dataset (daily_line_panel.csv)...")
    all_dates = pd.date_range(
        df_inc["Jour calendaire"].min(), df_inc["Jour calendaire"].max(), freq="D"
    ).strftime("%Y-%m-%d")

    grid = pd.MultiIndex.from_product(
        [MAIN_LINES, all_dates], names=["Ligne", "Jour calendaire"]
    ).to_frame().reset_index(drop=True)

    # Aggregate single-line + multi-line incidents for each impacted main line
    records = []
    for line_name in MAIN_LINES:
        flag_col = "impacts_" + line_name.split()[1]
        sub = df_inc[df_inc[flag_col] == 1]

        daily_agg = (
            sub.groupby("Jour calendaire")
            .agg(
                n_incidents_total=("Numero d'incident", "size"),
                n_incidents_T=("Type d'incident", lambda s: (s == "T").sum()),
                n_incidents_S=("Type d'incident", lambda s: (s == "S").sum()),
                n_major_delays_5m=(
                    "is_major_delay_5m",
                    lambda s: s[sub.loc[s.index, "Type d'incident"] == "T"].sum(),
                ),
                n_delays_gt2m=(
                    "is_delay_gt2m",
                    lambda s: s[sub.loc[s.index, "Type d'incident"] == "T"].sum(),
                ),
                total_est_delay_min_T=(
                    "duree_estimee_min",
                    lambda s: s[sub.loc[s.index, "Type d'incident"] == "T"].sum(),
                ),
                n_cause_clientele=(
                    "Cause primaire",
                    lambda s: ((s == "Clientèle") & (sub.loc[s.index, "Type d'incident"] == "T")).sum(),
                ),
                n_cause_materiel_roulant=(
                    "Cause primaire",
                    lambda s: ((s == "Matériel roulant") & (sub.loc[s.index, "Type d'incident"] == "T")).sum(),
                ),
                n_cause_equipements_fixes=(
                    "Cause primaire",
                    lambda s: ((s == "Équipements fixes") & (sub.loc[s.index, "Type d'incident"] == "T")).sum(),
                ),
                n_cause_exploitation=(
                    "Cause primaire",
                    lambda s: ((s == "Exploitation trains") & (sub.loc[s.index, "Type d'incident"] == "T")).sum(),
                ),
                n_cat=("CAT", "sum"),
                n_porte=("Porte", "sum"),
                n_kfs=("KFS", "sum"),
                n_urgence=("Urgence métro", "sum"),
            )
            .reset_index()
        )
        daily_agg["Ligne"] = line_name
        records.append(daily_agg)

    df_daily_inc = pd.concat(records, ignore_index=True)
    panel = grid.merge(df_daily_inc, on=["Ligne", "Jour calendaire"], how="left").fillna(0)

    # Merge planned KM data (available 2019-01-01 to 2023-09-30)
    panel = panel.merge(
        df_km[["Ligne", "Jour calendaire", "km_planifie", "periode_exploitation", "type_de_jour"]],
        on=["Ligne", "Jour calendaire"],
        how="left",
    )

    # Calendar features
    dt = pd.to_datetime(panel["Jour calendaire"])
    panel["annee_civile"] = dt.dt.year
    panel["mois_calendrier"] = dt.dt.month
    panel["jour_du_mois"] = dt.dt.day
    panel["jour_de_la_semaine"] = dt.dt.dayofweek + 1  # 1=Mon .. 7=Sun
    panel["jour_de_lannee"] = dt.dt.dayofyear
    panel["is_weekend"] = (panel["jour_de_la_semaine"] >= 6).astype(int)

    # Cyclical seasonal encodings
    panel["month_sin"] = np.sin(2 * np.pi * panel["mois_calendrier"] / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["mois_calendrier"] / 12.0)
    panel["dow_sin"] = np.sin(2 * np.pi * panel["jour_de_la_semaine"] / 7.0)
    panel["dow_cos"] = np.cos(2 * np.pi * panel["jour_de_la_semaine"] / 7.0)

    # Normalized incident rates per 100,000 planned KM
    has_km = panel["km_planifie"] > 0
    panel["rate_T_per_100k_km"] = np.where(
        has_km, panel["n_incidents_T"] / (panel["km_planifie"] / 100_000.0), np.nan
    )
    panel["rate_S_per_100k_km"] = np.where(
        has_km, panel["n_incidents_S"] / (panel["km_planifie"] / 100_000.0), np.nan
    )
    panel["rate_total_per_100k_km"] = np.where(
        has_km, panel["n_incidents_total"] / (panel["km_planifie"] / 100_000.0), np.nan
    )
    panel["rate_major_per_100k_km"] = np.where(
        has_km, panel["n_major_delays_5m"] / (panel["km_planifie"] / 100_000.0), np.nan
    )

    # Sort chronologically within each line and compute strictly lagged features (no leakage)
    panel = panel.sort_values(["Ligne", "Jour calendaire"]).reset_index(drop=True)
    grp = panel.groupby("Ligne")

    panel["lag1_incidents_T"] = grp["n_incidents_T"].shift(1)
    panel["lag2_incidents_T"] = grp["n_incidents_T"].shift(2)
    panel["lag7_incidents_T"] = grp["n_incidents_T"].shift(7)

    for window in [7, 14, 30]:
        panel[f"roll{window}d_incidents_T"] = grp["n_incidents_T"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        panel[f"roll{window}d_major_delays"] = grp["n_major_delays_5m"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        panel[f"roll{window}d_equipements"] = grp["n_cause_equipements_fixes"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        panel[f"roll{window}d_materiel"] = grp["n_cause_materiel_roulant"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        panel[f"roll{window}d_clientele"] = grp["n_cause_clientele"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    print(
        f"      Daily line panel shape: {panel.shape} "
        f"({panel['Jour calendaire'].nunique()} days x {panel['Ligne'].nunique()} lines)"
    )
    print(f"      Rows with valid KM planifié (2019-2023): {has_km.sum():,}")
    return panel


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    f_inc = sorted(glob.glob("Incidents*.xls"))[0]
    f_km = sorted(glob.glob("Kilom*.xls"))[0]

    df_km_clean = clean_kilometrage(f_km)
    df_inc_clean = clean_incidents(f_inc, df_km_clean)
    df_panel = build_daily_line_panel(df_inc_clean, df_km_clean)

    km_out = os.path.join(OUTPUT_DIR, "km_clean.csv")
    inc_out = os.path.join(OUTPUT_DIR, "incidents_clean.csv")
    panel_out = os.path.join(OUTPUT_DIR, "daily_line_panel.csv")

    df_km_clean.to_csv(km_out, index=False, encoding="utf-8")
    df_inc_clean.to_csv(inc_out, index=False, encoding="utf-8")
    df_panel.to_csv(panel_out, index=False, encoding="utf-8")

    print("\nSaved cleaned datasets:")
    print(f"  - {km_out} ({len(df_km_clean):,} rows)")
    print(f"  - {inc_out} ({len(df_inc_clean):,} rows)")
    print(f"  - {panel_out} ({len(df_panel):,} rows)")


if __name__ == "__main__":
    main()
