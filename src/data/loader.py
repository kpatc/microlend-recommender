import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Interaction types that count as product adoption (ordered by quality)
ADOPTION_TYPES = {"completed", "approved", "active", "defaulted"}

# Default rating by interaction type when satisfaction_score is null
_TYPE_RATING = {
    "completed": 5.0,
    "approved": 4.0,
    "active": 3.5,
    "defaulted": 2.0,
}


def _has_to_binary(series: pd.Series) -> pd.Series:
    """Map 'Have now' → 1, anything else → 0."""
    return (series.fillna("").str.strip().str.lower() == "have now").astype(int)


class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config["data"]["raw_path"])

    # ── Raw loaders ───────────────────────────────────────────────────────

    def load_sme_profiles(self) -> pd.DataFrame:
        path = self.raw_path / "sme_profiles.csv"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}. Run `make generate` first.")
        df = pd.read_csv(path)
        logger.info(f"Loaded sme_profiles: {df.shape}")
        return df

    def load_financial_profiles(self) -> pd.DataFrame:
        path = self.raw_path / "sme_financial_profile.csv"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}. Run `make generate` first.")
        df = pd.read_csv(path)
        logger.info(f"Loaded sme_financial_profile: {df.shape}")
        return df

    def load_product_interactions(self) -> pd.DataFrame:
        path = self.raw_path / "product_interactions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}. Run `make generate` first.")
        df = pd.read_csv(path, parse_dates=["interaction_date"], dayfirst=False)
        logger.info(f"Loaded product_interactions: {df.shape}")
        return df

    def load_item_features(self) -> pd.DataFrame:
        path = self.raw_path / "product_catalog.csv"
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(self.config["products"]["catalog"])
        logger.info(f"Loaded item features: {df.shape}")
        return df

    # ── Merged SME view ───────────────────────────────────────────────────

    def load_merged_sme(self) -> pd.DataFrame:
        """Join profiles + financial and derive clean binary columns."""
        profiles = self.load_sme_profiles()
        financial = self.load_financial_profiles()

        # Drop duplicates injected for messiness (keep first occurrence)
        profiles = profiles.drop_duplicates(subset="sme_id", keep="first")

        df = profiles.merge(financial, on="sme_id", how="left")

        # Binary derivations from string columns
        df["has_bank_account_yn"] = _has_to_binary(df.get("has_bank_account", pd.Series()))
        df["has_mobile_money_yn"] = _has_to_binary(df.get("has_mobile_money", pd.Series()))
        df["has_previous_loan"] = (df["n_previous_loans"].fillna(0) > 0).astype(int)
        df["has_default"] = (df["n_defaults"].fillna(0) > 0).astype(int)

        logger.info(f"Merged SME dataset: {df.shape}")
        return df

    # ── User-item matrix construction ─────────────────────────────────────

    def build_ratings_long(self, interactions: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Build a long-form ratings table from the raw interaction log.
        Only adoption-type interactions are included.
        Rating = satisfaction_score if available, else type-based default.
        When an SME has multiple interactions for the same product, keep best rating.
        """
        if interactions is None:
            interactions = self.load_product_interactions()

        adopted = interactions[interactions["interaction_type"].isin(ADOPTION_TYPES)].copy()

        def _rating(row):
            if pd.notna(row["satisfaction_score"]):
                return float(row["satisfaction_score"])
            return _TYPE_RATING.get(row["interaction_type"], 3.0)

        adopted["rating"] = adopted.apply(_rating, axis=1)

        # One rating per (sme_id, product_id) — keep the highest
        ratings = (
            adopted.groupby(["sme_id", "product_id"], as_index=False)["rating"]
            .max()
        )
        ratings["rating"] = ratings["rating"].clip(1.0, 5.0)
        logger.info(f"Built ratings_long: {len(ratings)} rows, "
                    f"{ratings['sme_id'].nunique()} SMEs, "
                    f"{ratings['product_id'].nunique()} products")
        return ratings

    def build_user_item_matrix(self, interactions: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Pivot ratings_long into a sme_id × product_id matrix.
        Missing entries are 0 (not adopted).
        """
        ratings = self.build_ratings_long(interactions)
        all_products = sorted(ratings["product_id"].unique())

        matrix = ratings.pivot(index="sme_id", columns="product_id", values="rating").fillna(0)
        # Ensure all 8 products are columns even if some have no adoption
        for pid in range(1, 9):
            if pid not in matrix.columns:
                matrix[pid] = 0.0
        matrix = matrix[[c for c in sorted(matrix.columns)]]
        matrix.index.name = "sme_id"
        logger.info(f"Built user-item matrix: {matrix.shape}, "
                    f"sparsity={self._sparsity(matrix):.2%}")
        return matrix

    # ── Backward-compatible aliases used by models / dashboard / API ──────

    def load_sme_features(self) -> pd.DataFrame:
        return self.load_merged_sme()

    def load_user_item_matrix(self) -> pd.DataFrame:
        return self.build_user_item_matrix()

    def load_ratings_long(self) -> pd.DataFrame:
        return self.build_ratings_long()

    # ── Utilities ─────────────────────────────────────────────────────────

    def validate_data(self, df: pd.DataFrame, required_cols: list) -> bool:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        return True

    @staticmethod
    def _sparsity(matrix: pd.DataFrame) -> float:
        total = matrix.size
        filled = (matrix > 0).sum().sum()
        return 1 - filled / total
