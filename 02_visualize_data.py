"""
Step 2: Exploratory Data Visualization (02_visualize_data.py)
-------------------------------------------------------------
Reads cleaned datasets from outputs/data/ and produces 4 publication-ready
figures in outputs/figures/ with English titles, axis labels, and explanations
(while preserving original French data categories where applicable):
  1. 01_yearly_monthly_trends.png
  2. 02_causes_and_rolling_stock.png
  3. 03_hourly_and_weekday_heatmap.png
  4. 04_top_stations_and_severity_drivers.png
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_DIR = os.path.join("outputs", "data")
FIG_DIR = os.path.join("outputs", "figures")

LINE_COLORS = {
    "Ligne verte": "#008E4F",
    "Ligne orange": "#F07D1A",
    "Ligne bleue": "#0078C1",
    "Ligne jaune": "#D9A400",
}

MONTH_NAMES_EN = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

DOW_NAMES_EN = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_01_yearly_monthly_trends(df_inc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Yearly counts by Type (T vs S)
    ax = axes[0]
    yearly = (
        df_inc.groupby(["Année civile", "Type d'incident"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["T", "S"])
    )
    years = yearly.index.astype(str).tolist()
    years[-1] = "2025*\n(Jan–Sep)"
    x = np.arange(len(years))

    ax.bar(
        x,
        yearly["T"],
        width=0.6,
        label="Type T (Train / Service Interruption)",
        color="#1f4e78",
    )
    ax.bar(
        x,
        yearly["S"],
        width=0.6,
        bottom=yearly["T"],
        label="Type S (Station / Other)",
        color="#6baed6",
    )

    totals = yearly["T"] + yearly["S"]
    for idx, tot in enumerate(totals):
        ax.text(
            idx,
            tot + 130,
            f"{tot:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            idx,
            yearly["T"].iloc[idx] / 2,
            f"{yearly['T'].iloc[idx]:,}",
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, totals.max() * 1.25)
    ax.set_title("A. Annual Incident Volume by Type (2019–2025*)")
    ax.set_ylabel("Number of Incidents")
    ax.legend(frameon=True, loc="upper right")

    # Panel B: Monthly seasonality across full years (2019-2024)
    ax2 = axes[1]
    full_years = df_inc[df_inc["Année civile"] <= 2024]
    monthly = (
        full_years.groupby(["Année civile", "Mois calendrier", "Type d'incident"])
        .size()
        .unstack(fill_value=0)
        .groupby("Mois calendrier")[["T", "S"]]
        .mean()
    )
    m_x = np.arange(12)
    ax2.plot(
        m_x,
        monthly["T"],
        marker="o",
        linewidth=2.5,
        color="#1f4e78",
        label="Type T (Monthly Avg 2019–2024)",
    )
    ax2.plot(
        m_x,
        monthly["S"],
        marker="s",
        linewidth=2.0,
        color="#6baed6",
        label="Type S (Monthly Avg 2019–2024)",
    )
    ax2.plot(
        m_x,
        monthly["T"] + monthly["S"],
        marker="^",
        linewidth=2.0,
        linestyle="--",
        color="#d95f02",
        label="Total Monthly Average",
    )

    ax2.axvspan(-0.3, 2.3, color="#cce5ff", alpha=0.35, label="Winter Peak (Jan–Mar)")
    ax2.set_xticks(m_x)
    ax2.set_xticklabels(MONTH_NAMES_EN)
    ax2.set_title("B. Average Monthly Seasonality (2019–2024)")
    ax2.set_ylabel("Average Monthly Incidents")
    ax2.legend(frameon=True, loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Macro Trends and Monthly Seasonality of Montreal Metro Incidents",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "01_yearly_monthly_trends.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_03_causes_and_rolling_stock(df_inc: pd.DataFrame):
    t_inc = df_inc[df_inc["Type d'incident"] == "T"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Evolution of Primary Causes (2019-2024 full years)
    ax1 = axes[0]
    t_full = t_inc[t_inc["Année civile"] <= 2024]
    causes_yr = (
        t_full.groupby(["Année civile", "Cause primaire"]).size().unstack(fill_value=0)
    )
    top_causes = [
        "Clientèle",
        "Équipements fixes",
        "Matériel roulant",
        "Exploitation trains",
        "Autres",
    ]
    cause_labels_en = {
        "Clientèle": "Clientèle (Passengers)",
        "Équipements fixes": "Équipements fixes (Fixed Equipment)",
        "Matériel roulant": "Matériel roulant (Rolling Stock)",
        "Exploitation trains": "Exploitation trains (Train Ops)",
        "Autres": "Autres (Other)",
    }
    cause_colors = {
        "Clientèle": "#1f77b4",
        "Équipements fixes": "#d62728",
        "Matériel roulant": "#2ca02c",
        "Exploitation trains": "#9467bd",
        "Autres": "#7f7f7f",
    }
    for c in top_causes:
        if c in causes_yr.columns:
            ax1.plot(
                causes_yr.index,
                causes_yr[c],
                marker="o",
                linewidth=2.3,
                label=cause_labels_en[c],
                color=cause_colors[c],
            )
            y_offset = (
                28 if c == "Autres" else (-28 if c == "Exploitation trains" else 0)
            )
            ax1.text(
                2024.08,
                causes_yr.loc[2024, c] + y_offset,
                f"{causes_yr.loc[2024, c]}",
                va="center",
                fontsize=8.5,
                color=cause_colors[c],
                fontweight="bold",
            )

    ax1.set_xticks([2019, 2020, 2021, 2022, 2023, 2024])
    ax1.set_xlim(2018.8, 2024.5)
    ax1.set_title("A. Evolution of Primary Causes (Type T Incidents, 2019–2024)")
    ax1.set_ylabel("Number of Type T Incidents")
    ax1.legend(frameon=True, loc="upper left", fontsize=8.5)

    # Panel B: Rolling stock faults (MR-73 vs MPM-10 Azur) by year
    ax2 = axes[1]
    fleet_yr = (
        t_full[t_full["flotte_train"].isin(["MR-73", "MPM-10 (Azur)"])]
        .groupby(["Année civile", "flotte_train"])
        .size()
        .unstack(fill_value=0)
    )
    x = np.arange(len(fleet_yr.index))
    w = 0.36
    b1 = ax2.bar(
        x - w / 2,
        fleet_yr["MR-73"],
        width=w,
        color="#2b5c8f",
        label="MR-73 (Older Generation Fleet)",
    )
    b2 = ax2.bar(
        x + w / 2,
        fleet_yr["MPM-10 (Azur)"],
        width=w,
        color="#f07d1a",
        label="MPM-10 Azur (Newer Generation Fleet)",
    )

    for bar in b1:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(bar.get_height())}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    for bar in b2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(bar.get_height())}",
            ha="center",
            fontsize=8.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(fleet_yr.index.astype(str))
    ax2.set_ylim(0, fleet_yr["MR-73"].max() * 1.18)
    ax2.set_title("B. Rolling Stock Faults: MR-73 vs. MPM-10 Azur (2019–2024)")
    ax2.set_ylabel("Number of Incidents")
    ax2.legend(frameon=True)

    fig.suptitle(
        "Service Interruption Causes and Rolling Stock Comparison",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "03_causes_and_rolling_stock.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_04_hourly_and_weekday_heatmap(df_inc: pd.DataFrame):
    t_inc = df_inc[df_inc["Type d'incident"] == "T"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: Heatmap of Day of Week x Operational Hour (05h to 25h)
    ax1 = axes[0]
    valid_hours = t_inc[t_inc["heure_operationnelle"].between(5, 25)]
    piv = (
        valid_hours.groupby(["Jour de la semaine", "heure_operationnelle"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[1, 2, 3, 4, 5, 6, 7])
    )
    piv.index = DOW_NAMES_EN
    piv.columns = [f"{h:02d}h" if h < 24 else f"0{h - 24}h+" for h in piv.columns]

    sns.heatmap(
        piv,
        cmap="YlOrRd",
        ax=ax1,
        cbar_kws={"label": "Number of Type T Incidents"},
        linewidths=0.3,
    )
    ax1.set_title("A. Type T Incident Density by Day and Operational Hour")
    ax1.set_xlabel("Operational Hour (05:00 to 01:59 After Midnight)")
    ax1.set_ylabel("")

    # Panel B: Hourly Volume vs Major Delay Rate (>=5 min)
    ax2 = axes[1]
    hourly_stats = valid_hours.groupby("heure_operationnelle").agg(
        count=("is_major_delay_5m", "size"),
        major_rate=("is_major_delay_5m", lambda s: s.mean() * 100.0),
    )
    h_labels = [f"{h:02d}h" if h < 24 else f"0{h - 24}h+" for h in hourly_stats.index]
    h_x = np.arange(len(h_labels))

    ax2.bar(
        h_x,
        hourly_stats["count"],
        color="#4a86e8",
        alpha=0.75,
        label="Type T Incident Volume",
    )
    ax2.set_xticks(h_x)
    ax2.set_xticklabels(h_labels, rotation=45, fontsize=8.5)
    ax2.set_ylim(0, hourly_stats["count"].max() * 1.22)
    ax2.set_ylabel("Type T Incident Volume", color="#1f4e78")

    ax2_twin = ax2.twinx()
    ax2_twin.spines["top"].set_visible(False)
    ax2_twin.grid(False)
    ax2_twin.plot(
        h_x,
        hourly_stats["major_rate"],
        color="#c00000",
        marker="o",
        linewidth=2.2,
        label="% Major Delays (≥5 min)",
    )
    ax2_twin.set_ylabel("Major Delay Rate ≥5 min (%)", color="#c00000")
    ax2_twin.set_ylim(10, 38)

    ax2.set_title("B. Hourly Profile: Incident Volume vs. Major Delay Rate (≥5 min)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    fig.suptitle(
        "Temporal Distribution: Rush-Hour Peaks and Day-of-Week Patterns",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "04_hourly_and_weekday_heatmap.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_05_top_stations_and_severity_drivers(df_inc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Top 15 Hotspot Stations
    ax1 = axes[0]
    top_locs = df_inc["Code de lieu"].value_counts().head(15).index.tolist()
    loc_piv = (
        df_inc[df_inc["Code de lieu"].isin(top_locs)]
        .groupby(["Code de lieu", "Type d'incident"])
        .size()
        .unstack(fill_value=0)
        .reindex(top_locs)[["T", "S"]]
        .iloc[::-1]
    )
    y = np.arange(len(loc_piv))
    ax1.barh(y, loc_piv["T"], color="#1f4e78", label="Type T (Train)")
    ax1.barh(
        y, loc_piv["S"], left=loc_piv["T"], color="#6baed6", label="Type S (Station)"
    )

    totals = loc_piv["T"] + loc_piv["S"]
    for idx, tot in enumerate(totals):
        ax1.text(
            tot + 35, idx, f"{tot:,}", va="center", fontsize=8.5, fontweight="bold"
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(loc_piv.index)
    ax1.set_xlim(0, totals.max() * 1.15)
    ax1.set_title("A. Top 15 Hotspot Stations / Locations (2019–2025)")
    ax1.set_xlabel("Total Number of Incidents")
    ax1.legend(frameon=True, loc="lower right")

    # Panel B: Severity Drivers (% Major Delay >= 5 min among Type T)
    ax2 = axes[1]
    t_inc = df_inc[df_inc["Type d'incident"] == "T"]
    base_rate = t_inc["is_major_delay_5m"].mean() * 100.0

    drivers = [
        (
            "Metro Emergency (Urgence métro = 1, n=479)",
            t_inc[t_inc["Urgence métro"] == 1]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Emergency Track Switch (KFS = 1, n=1,054)",
            t_inc[t_inc["KFS"] == 1]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Traction Power Cut (CAT = 1, n=7,045)",
            t_inc[t_inc["CAT"] == 1]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Material Damage (Dommage matériel = 1, n=353)",
            t_inc[t_inc["Dommage matériel"] == 1]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Symptom: Fixed Equip. (Équipements fixes, n=3,877)",
            t_inc[t_inc["Symptome"] == "Équipements fixes"]["is_major_delay_5m"].mean()
            * 100,
        ),
        (
            "Door Fault Indicator (Porte = 1, n=2,911)",
            t_inc[t_inc["Porte"] == 1]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Symptom: Rolling Stock (Matériel roulant, n=7,750)",
            t_inc[t_inc["Symptome"] == "Matériel roulant"]["is_major_delay_5m"].mean()
            * 100,
        ),
        (
            "Symptom: Passengers (Clientèle, n=13,095)",
            t_inc[t_inc["Symptome"] == "Clientèle"]["is_major_delay_5m"].mean() * 100,
        ),
        (
            "Symptom: Train Ops (Exploitation trains, n=2,071)",
            t_inc[t_inc["Symptome"] == "Exploitation trains"][
                "is_major_delay_5m"
            ].mean()
            * 100,
        ),
        (
            "Symptom: Station Ops (Exploitation stations, n=944)",
            t_inc[t_inc["Symptome"] == "Exploitation stations"][
                "is_major_delay_5m"
            ].mean()
            * 100,
        ),
    ][::-1]

    d_labels = [d[0] for d in drivers]
    d_vals = [d[1] for d in drivers]
    d_colors = ["#c00000" if v > base_rate else "#4a86e8" for v in d_vals]

    y2 = np.arange(len(drivers))
    ax2.barh(y2, d_vals, color=d_colors, alpha=0.85, edgecolor="black")
    ax2.axvline(
        base_rate,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Overall Type T Baseline ({base_rate:.1f}%)",
    )

    for idx, val in enumerate(d_vals):
        ax2.text(
            val + 1.2, idx, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold"
        )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(d_labels)
    ax2.set_xlim(0, 95)
    ax2.set_title("B. Severity Drivers: % of Type T Incidents Lasting ≥ 5 Minutes")
    ax2.set_xlabel("Probability of Major Delay ≥ 5 min (%)")
    ax2.legend(frameon=True, loc="lower right")

    fig.suptitle(
        "Geographic Hotspots (Stations) and Key Drivers of Major Delays",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "05_top_stations_and_severity_drivers.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    print("Loading cleaned datasets from outputs/data/...")
    df_inc = pd.read_csv(
        os.path.join(DATA_DIR, "incidents_clean.csv"), low_memory=False
    )

    print("Generating visualizations in English...")
    plot_01_yearly_monthly_trends(df_inc)
    plot_03_causes_and_rolling_stock(df_inc)
    plot_04_hourly_and_weekday_heatmap(df_inc)
    plot_05_top_stations_and_severity_drivers(df_inc)
    print("All 4 exploratory figures generated successfully in outputs/figures/.")


if __name__ == "__main__":
    main()
