import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class HybridRecommender:
    def __init__(self, config: dict):
        cfg = config.get("hybrid", {})
        self.cf_weight = cfg.get("cf_weight", 0.7)
        self.content_weight = cfg.get("content_weight", 0.3)
        self.cf_model = None
        self.sme_features: pd.DataFrame = None
        self.item_features: pd.DataFrame = None
        self.user_item_matrix: pd.DataFrame = None
        self.default_predictor = LogisticRegression(max_iter=500)
        self.sme_feature_matrix: np.ndarray = None
        self.item_feature_matrix: np.ndarray = None
        self.product_ids: list = None

    def fit(self, user_item_matrix: pd.DataFrame, sme_features: pd.DataFrame,
            item_features: pd.DataFrame, cf_model=None):
        from src.models.user_based_cf import UserBasedCF
        self.user_item_matrix = user_item_matrix
        self.sme_features = sme_features.copy()
        self.item_features = item_features.copy()
        self.product_ids = list(user_item_matrix.columns)

        self.cf_model = cf_model or UserBasedCF()
        if not hasattr(self.cf_model, "similarity_matrix") or self.cf_model.similarity_matrix is None:
            self.cf_model.fit(user_item_matrix, sme_features)

        self.sme_feature_matrix = self._build_sme_feature_matrix()
        self.item_feature_matrix = self._build_item_feature_matrix()
        self._fit_default_predictor()
        logger.success("HybridRecommender fitted.")
        return self

    def _build_sme_feature_matrix(self) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        df = self.sme_features.copy()
        for col in ["country", "sector", "urban_rural"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
        num_cols = [
            "annual_revenue_usd", "n_employees", "years_in_business",
            "bureau_score", "has_bank_account_yn", "has_mobile_money_yn",
            "has_previous_loan", "has_default", "n_previous_loans", "n_defaults",
            "country", "sector",
        ]
        num_cols = [c for c in num_cols if c in df.columns]
        scaler = MinMaxScaler()
        return scaler.fit_transform(df[num_cols].fillna(0))

    def _build_item_feature_matrix(self) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        df = self.item_features.copy()
        for col in ["category", "risk_level"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        num_cols = ["category", "risk_level", "min_revenue"]
        num_cols = [c for c in num_cols if c in df.columns]
        scaler = MinMaxScaler()
        return scaler.fit_transform(df[num_cols].fillna(0))

    def _fit_default_predictor(self):
        from sklearn.preprocessing import LabelEncoder
        df = self.sme_features.copy()
        for col in ["country", "sector", "urban_rural"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
        feature_cols = [
            "annual_revenue_usd", "n_employees", "years_in_business",
            "has_bank_account_yn", "has_mobile_money_yn", "has_previous_loan",
            "bureau_score", "country", "sector",
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y = df["has_default"].fillna(0).values.astype(int)
        self.default_predictor.fit(X, y)
        self._default_feature_cols = feature_cols

    def _get_default_prob(self, sme_id) -> float:
        from sklearn.preprocessing import LabelEncoder
        row = self.sme_features[self.sme_features["sme_id"] == sme_id]
        if row.empty:
            return 0.15
        df = row.copy()
        for col in ["country", "sector", "urban_rural"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
        X = df[self._default_feature_cols].fillna(0).values
        return float(self.default_predictor.predict_proba(X)[0, 1])

    def recommend(self, sme_id, n: int = 5) -> list:
        sme_ids = list(self.sme_features["sme_id"])
        sme_idx = sme_ids.index(sme_id) if sme_id in sme_ids else 0

        cf_preds = dict(self.cf_model.predict(sme_id, n_recommendations=len(self.product_ids)))
        sme_vec = self.sme_feature_matrix[sme_idx].reshape(1, -1)
        content_scores = cosine_similarity(sme_vec, self.item_feature_matrix)[0]
        default_prob = self._get_default_prob(sme_id)

        results = []
        for i, pid in enumerate(self.product_ids):
            cf_score = (cf_preds.get(pid, 3.0) - 1) / 4
            content_score = float(content_scores[i])
            id_col = "product_id" if "product_id" in self.item_features.columns else "id"
            risk_row = self.item_features[self.item_features[id_col] == pid]
            risk_level = risk_row.iloc[0]["risk_level"] if not risk_row.empty else "medium"
            risk_mult = {"low": 1.0, "medium": 0.9, "high": 0.7}.get(risk_level, 0.9)
            risk_adjustment = risk_mult * (1 - default_prob)

            final_score = (self.cf_weight * cf_score +
                           self.content_weight * content_score +
                           0.1 * risk_adjustment)

            results.append({
                "product_id": pid,
                "cf_score": round(cf_score, 3),
                "content_score": round(content_score, 3),
                "risk_adjusted_score": round(risk_adjustment, 3),
                "final_score": round(final_score, 3),
                "explanation": self.explain(sme_id, pid, cf_score, content_score, risk_adjustment),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:n]

    def explain(self, sme_id, product_id, cf_score=None, content_score=None, risk_adj=None) -> str:
        return (f"Recommended because: CF score={cf_score:.2f}, "
                f"content match={content_score:.2f}, "
                f"risk adjustment={risk_adj:.2f}")
