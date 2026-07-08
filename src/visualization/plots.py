import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "viridis"


def plot_adoption_rates(ratings_long: pd.DataFrame, product_names: dict = None) -> plt.Figure:
    counts = ratings_long.groupby("product_id")["sme_id"].nunique().reset_index()
    counts.columns = ["product_id", "adopters"]
    n_smes = ratings_long["sme_id"].nunique()
    counts["rate"] = counts["adopters"] / n_smes
    if product_names:
        counts["name"] = counts["product_id"].map(product_names)
    else:
        counts["name"] = counts["product_id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=counts, x="name", y="rate", palette=PALETTE, ax=ax)
    ax.set_title("Product Adoption Rates")
    ax.set_ylabel("Adoption Rate")
    ax.set_xlabel("Product")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_sector_country_distribution(sme_features: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sme_features["country"].value_counts().plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("SMEs by Country")
    axes[0].set_xlabel("Count")

    sme_features["sector"].value_counts().plot(kind="barh", ax=axes[1], color="coral")
    axes[1].set_title("SMEs by Sector")
    axes[1].set_xlabel("Count")
    plt.tight_layout()
    return fig


def plot_revenue_by_sector(sme_features: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    order = sme_features.groupby("sector")["annual_revenue_usd"].median().sort_values(ascending=False).index
    sns.boxplot(data=sme_features, x="sector", y="annual_revenue_usd",
                order=order, palette=PALETTE, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Annual Revenue by Sector (USD)")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Revenue (log scale)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_coadoption_heatmap(user_item_matrix: pd.DataFrame, product_names: dict = None) -> plt.Figure:
    binary_matrix = (user_item_matrix > 0).astype(int)
    coadoption = binary_matrix.T.dot(binary_matrix).astype(float)
    n_smes = len(user_item_matrix)
    coadoption = coadoption / n_smes

    labels = [product_names.get(int(c), str(c)) if product_names else str(c)
              for c in coadoption.columns]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(coadoption.values, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Product Co-Adoption Matrix")
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "ndcg@10") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if metric in results_df.columns:
        results_df.sort_values(metric, ascending=False).plot(
            kind="bar", x="model", y=metric, ax=ax, color="teal", legend=False
        )
        ax.set_title(f"Model Comparison — {metric.upper()}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def plot_rating_distribution(ratings_long: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ratings_long["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="mediumpurple")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_latent_space_pca(item_factors: np.ndarray, product_names: dict = None,
                           categories: list = None) -> plt.Figure:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(item_factors)
    n = len(coords)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(coords):
        label = product_names.get(i + 1, str(i + 1)) if product_names else str(i + 1)
        color = colors[i]
        ax.scatter(x, y, c=[color], s=120, zorder=3)
        ax.annotate(label, (x, y), fontsize=8, ha="right")

    ax.set_title("Product Latent Space (PCA of SVD item factors)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()
    return fig
