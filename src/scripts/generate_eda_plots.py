"""Generate EDA and model evaluation plots for docs/."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)

PALETTE = ["#1a56db", "#0e9f6e", "#e3a008", "#e02424", "#7e3af2", "#ff5a1f", "#0694a2", "#6b7280"]
PRODUCT_NAMES = {
    1: "Microcrédit 3 mois", 2: "Microcrédit 12 mois", 3: "Assurance agricole",
    4: "Leasing équipement", 5: "Épargne groupe", 6: "Paiement mobile",
    7: "Financement facture", 8: "Prêt récolte"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.dpi": 150,
})


def load():
    profiles = pd.read_csv("data/raw/sme_profiles.csv")
    financial = pd.read_csv("data/raw/sme_financial_profile.csv")
    interactions = pd.read_csv("data/raw/product_interactions.csv")
    return profiles, financial, interactions


# ── Plot 1 : SME Distribution by Country ─────────────────────────────────────
def plot_country_distribution(df):
    counts = df["country"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[0], alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 15, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color="#374151")
    ax.set_xlabel("Nombre de PME", fontsize=11)
    ax.set_title("Distribution des PME par Pays", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, counts.max() * 1.15)
    plt.tight_layout()
    fig.savefig(DOCS / "eda_01_countries.png")
    plt.close()
    print("✓ eda_01_countries.png")


# ── Plot 2 : Sector Distribution ─────────────────────────────────────────────
def plot_sector_distribution(df):
    counts = df["sector"].value_counts()
    labels = counts.index.str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=PALETTE[:len(counts)], alpha=0.85, width=0.65)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{int(bar.get_height()):,}", ha="center", fontsize=8.5)
    ax.set_ylabel("Nombre de PME", fontsize=11)
    ax.set_title("Répartition Sectorielle des PME", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(DOCS / "eda_02_sectors.png")
    plt.close()
    print("✓ eda_02_sectors.png")


# ── Plot 3 : Revenue Distribution (log scale) ─────────────────────────────────
def plot_revenue_distribution(df, fi):
    merged = df.merge(fi[["sme_id", "has_bank_account"]], on="sme_id", how="left")
    rev = df["annual_revenue_usd"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log histogram
    ax = axes[0]
    ax.hist(np.log10(rev.clip(lower=1)), bins=40, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.set_xlabel("log₁₀(Revenu annuel USD)", fontsize=11)
    ax.set_ylabel("Fréquence", fontsize=11)
    ax.set_title("Distribution des Revenus (échelle log)", fontsize=12, fontweight="bold")
    quartiles = np.percentile(rev, [25, 50, 75])
    for q, label, color in zip(quartiles, ["Q1", "Médiane", "Q3"], ["#e3a008", "#e02424", "#0e9f6e"]):
        ax.axvline(np.log10(q), color=color, linestyle="--", linewidth=1.5, label=f"{label}: ${q:,.0f}")
    ax.legend(fontsize=9)

    # Right: box by sector
    ax = axes[1]
    top_sectors = df["sector"].value_counts().head(6).index
    data = [np.log10(df[df["sector"] == s]["annual_revenue_usd"].dropna().clip(lower=1))
            for s in top_sectors]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels([s.replace("_", "\n").title() for s in top_sectors], fontsize=8)
    ax.set_ylabel("log₁₀(Revenu USD)", fontsize=11)
    ax.set_title("Revenus par Secteur", fontsize=12, fontweight="bold")

    plt.suptitle("Analyse des Revenus des PME", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(DOCS / "eda_03_revenue.png")
    plt.close()
    print("✓ eda_03_revenue.png")


# ── Plot 4 : Financial Inclusion ─────────────────────────────────────────────
def plot_financial_inclusion(fi, profiles):
    merged = fi.merge(profiles[["sme_id", "country"]], on="sme_id", how="left")

    def pct_have(series):
        return (series.fillna("").str.lower() == "have now").mean() * 100

    countries = merged["country"].value_counts().head(8).index
    bank_rates = [pct_have(merged[merged["country"] == c]["has_bank_account"]) for c in countries]
    mm_rates   = [pct_have(merged[merged["country"] == c]["has_mobile_money"]) for c in countries]

    x = np.arange(len(countries))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, bank_rates, w, label="Compte bancaire", color=PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, mm_rates,   w, label="Mobile money",    color=PALETTE[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=20, ha="right")
    ax.set_ylabel("Taux d'inclusion (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Inclusion Financière par Pays", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    for i, (b, m) in enumerate(zip(bank_rates, mm_rates)):
        ax.text(i - w/2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=7.5, color="#374151")
        ax.text(i + w/2, m + 1.5, f"{m:.0f}%", ha="center", fontsize=7.5, color="#374151")
    plt.tight_layout()
    fig.savefig(DOCS / "eda_04_financial_inclusion.png")
    plt.close()
    print("✓ eda_04_financial_inclusion.png")


# ── Plot 5 : Interaction Funnel ───────────────────────────────────────────────
def plot_interaction_funnel(interactions):
    order = ["inquiry", "application", "approved", "rejected", "completed", "defaulted"]
    colors = [PALETTE[6], PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[3]]
    counts = interactions["interaction_type"].value_counts().reindex(order, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bar chart
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.85, width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", fontsize=9)
    ax.set_ylabel("Nombre d'interactions", fontsize=11)
    ax.set_title("Volume par Type d'Interaction", fontsize=12, fontweight="bold")
    plt.sca(ax)
    plt.xticks(rotation=20, ha="right")

    # Right: product adoption rates
    ax = axes[1]
    adopted = interactions[interactions["interaction_type"].isin(["approved", "completed", "active"])]
    prod_counts = adopted["product_id"].value_counts().sort_index()
    prod_labels = [PRODUCT_NAMES.get(pid, f"P{pid}") for pid in prod_counts.index]
    bars = ax.barh(prod_labels, prod_counts.values, color=PALETTE[:len(prod_counts)], alpha=0.85)
    for bar, val in zip(bars, prod_counts.values):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8.5)
    ax.set_xlabel("Adoptions (approved + completed)", fontsize=11)
    ax.set_title("Adoption par Produit", fontsize=12, fontweight="bold")

    plt.suptitle("Analyse des Interactions Produit", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(DOCS / "eda_05_interactions.png")
    plt.close()
    print("✓ eda_05_interactions.png")


# ── Plot 6 : Model CV Results ─────────────────────────────────────────────────
def plot_model_results():
    results = {
        "model":    ["Baseline", "SVD", "NMF"],
        "RMSE":     [0.444, 0.447, 0.726],
        "RMSE_std": [0.005, 0.006, 0.052],
        "MAE":      [0.394, 0.393, 0.580],
        "MAE_std":  [0.003, 0.004, 0.040],
    }
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = [PALETTE[1] if r == df["RMSE"].min() else PALETTE[0] for r in df["RMSE"]]

    for ax, metric, std_col in zip(axes, ["RMSE", "MAE"], ["RMSE_std", "MAE_std"]):
        bars = ax.bar(df["model"], df[metric], color=colors, alpha=0.85, width=0.5,
                      yerr=df[std_col], capsize=5, error_kw=dict(elinewidth=1.5, ecolor="#6b7280"))
        for bar, val, std in zip(bars, df[metric], df[std_col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + std + 0.005,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} — 5-Fold CV", fontsize=12, fontweight="bold")
        ax.set_ylim(0, df[metric].max() * 1.35)

    best_patch = mpatches.Patch(color=PALETTE[1], label="Meilleur modèle")
    other_patch = mpatches.Patch(color=PALETTE[0], label="Autres modèles")
    fig.legend(handles=[best_patch, other_patch], loc="upper right", fontsize=9)
    plt.suptitle("Comparaison des Modèles — Validation Croisée", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(DOCS / "model_01_cv_results.png")
    plt.close()
    print("✓ model_01_cv_results.png")


# ── Plot 7 : Sparsity & Coverage ─────────────────────────────────────────────
def plot_matrix_stats(interactions):
    adopted = interactions[interactions["interaction_type"].isin(["approved", "completed", "active", "defaulted"])]
    n_smes = adopted["sme_id"].nunique()
    n_products = 8
    n_interactions = len(adopted.groupby(["sme_id", "product_id"]))
    sparsity = 1 - n_interactions / (n_smes * n_products)

    # Co-adoption heatmap
    ratings = adopted.groupby(["sme_id", "product_id"]).size().reset_index(name="v")
    matrix = ratings.pivot(index="sme_id", columns="product_id", values="v").fillna(0)
    for pid in range(1, 9):
        if pid not in matrix.columns:
            matrix[pid] = 0
    matrix = (matrix > 0).astype(int)
    coadoption = matrix.T.dot(matrix).astype(float)
    np.fill_diagonal(coadoption.values, 0)
    coadoption = coadoption / n_smes

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: interaction count per SME histogram
    ax = axes[0]
    counts_per_sme = adopted.groupby("sme_id")["product_id"].nunique()
    ax.hist(counts_per_sme.values, bins=range(1, 10), color=PALETTE[0], alpha=0.85,
            edgecolor="white", rwidth=0.8)
    ax.axvline(counts_per_sme.mean(), color=PALETTE[3], linestyle="--",
               linewidth=2, label=f"Moyenne: {counts_per_sme.mean():.1f}")
    ax.set_xlabel("Nombre de produits adoptés par PME", fontsize=11)
    ax.set_ylabel("Nombre de PME", fontsize=11)
    ax.set_title(f"Distribution des Adoptions\n(Matrice {n_smes}×{n_products}, sparsité {sparsity:.1%})",
                 fontsize=11, fontweight="bold")
    ax.legend()

    # Right: co-adoption heatmap
    ax = axes[1]
    labels = [PRODUCT_NAMES.get(i, f"P{i}") for i in range(1, 9)]
    short = [n.split()[0] + "\n" + n.split()[1] if len(n.split()) > 1 else n for n in labels]
    sns.heatmap(coadoption.values, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=ax, linewidths=0.4,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Co-adoption des Produits\n(proportion des PME)", fontsize=11, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.suptitle("Structure de la Matrice User-Item", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(DOCS / "eda_06_matrix.png")
    plt.close()
    print("✓ eda_06_matrix.png")


if __name__ == "__main__":
    print("Chargement des données...")
    profiles, financial, interactions = load()

    plot_country_distribution(profiles)
    plot_sector_distribution(profiles)
    plot_revenue_distribution(profiles, financial)
    plot_financial_inclusion(financial, profiles)
    plot_interaction_funnel(interactions)
    plot_model_results()
    plot_matrix_stats(interactions)

    print(f"\n7 figures sauvegardées dans docs/")
