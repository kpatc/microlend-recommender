import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from loguru import logger


class SMEPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.categorical_cols = ["country", "sector", "urban_rural"]
        # bureau_score replaces credit_score; binary cols are derived in DataLoader
        self.numeric_cols = [
            "annual_revenue_usd", "n_employees", "years_in_business", "bureau_score",
            "n_previous_loans", "n_defaults", "financial_literacy_score",
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))
                self.label_encoders[col] = le
        numeric_present = [c for c in self.numeric_cols if c in df.columns]
        df[numeric_present] = self.scaler.fit_transform(df[numeric_present].fillna(0))
        logger.info(f"Preprocessed SME features: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                vals = df[col].fillna("unknown").astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[f"{col}_enc"] = le.transform(vals)
        numeric_present = [c for c in self.numeric_cols if c in df.columns]
        df[numeric_present] = self.scaler.transform(df[numeric_present].fillna(0))
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        encoded_cats = [f"{c}_enc" for c in self.categorical_cols if f"{c}_enc" in df.columns]
        binary_cols = [
            "has_bank_account_yn", "has_mobile_money_yn",
            "has_previous_loan", "has_default",
        ]
        binary_present = [c for c in binary_cols if c in df.columns]
        feature_cols = self.numeric_cols + encoded_cats + binary_present
        feature_cols = [c for c in feature_cols if c in df.columns]
        return df[feature_cols].fillna(0).values.astype(float)


def build_surprise_dataset(ratings_df: pd.DataFrame):
    from surprise import Dataset, Reader
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["sme_id", "product_id", "rating"]], reader)
    return data


def train_test_split_temporal(ratings_df: pd.DataFrame, test_size: float = 0.2):
    """Split ensuring each SME has interactions in both sets."""
    train_rows, test_rows = [], []
    for sme_id, group in ratings_df.groupby("sme_id"):
        if len(group) == 1:
            train_rows.append(group)
        else:
            n_test = max(1, int(len(group) * test_size))
            shuffled = group.sample(frac=1, random_state=42)
            test_rows.append(shuffled.iloc[:n_test])
            train_rows.append(shuffled.iloc[n_test:])
    return pd.concat(train_rows).reset_index(drop=True), pd.concat(test_rows).reset_index(drop=True)
