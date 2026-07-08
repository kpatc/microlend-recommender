import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from loguru import logger


PRODUCT_NAMES = {
    1: "Microcredit 3 months", 2: "Microcredit 12 months",
    3: "Agricultural insurance", 4: "Equipment leasing",
    5: "Group savings", 6: "Mobile payment setup",
    7: "Invoice financing", 8: "Crop advance loan",
}

ONBOARDING_QUESTIONS = [
    {"field": "sector", "question": "What sector does your business operate in?",
     "options": ["agriculture", "retail", "textile", "food_processing",
                 "transport", "construction", "services", "tech"]},
    {"field": "annual_revenue_usd", "question": "What is your approximate annual revenue (USD)?",
     "options": "numeric"},
    {"field": "has_mobile_money_yn", "question": "Do you use mobile money (M-Pesa, Orange Money, etc.)?",
     "options": [0, 1]},
    {"field": "has_bank_account_yn", "question": "Do you have a bank account?",
     "options": [0, 1]},
    {"field": "country", "question": "Which country are you based in?",
     "options": ["Morocco", "Senegal", "Kenya", "Nigeria",
                 "Ghana", "Ivory Coast", "Tanzania", "Ethiopia"]},
    {"field": "n_employees", "question": "How many employees do you have (including yourself)?",
     "options": "numeric"},
    {"field": "has_previous_loan", "question": "Have you previously taken a loan?",
     "options": [0, 1]},
]


class ColdStartSolver:
    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.sme_features: pd.DataFrame = None
        self.user_item_matrix: pd.DataFrame = None
        self.feature_matrix: np.ndarray = None
        self.label_encoders: dict = {}
        self.scaler = MinMaxScaler()
        self.feature_cols: list = []

    def fit(self, sme_features: pd.DataFrame, user_item_matrix: pd.DataFrame):
        self.sme_features = sme_features.copy()
        self.user_item_matrix = user_item_matrix.copy()
        self.feature_matrix = self._encode_features(sme_features, fit=True)
        logger.success(f"ColdStartSolver fitted on {len(sme_features)} SMEs.")
        return self

    def _encode_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        df = df.copy()
        cat_cols = ["country", "sector", "urban_rural"]

        if fit:
            for col in cat_cols:
                if col not in df.columns:
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
                self.label_encoders[col] = le

            num_cols = [
                "annual_revenue_usd", "n_employees", "years_in_business",
                "bureau_score", "has_bank_account_yn", "has_mobile_money_yn",
                "has_previous_loan", "has_default", "n_previous_loans", "n_defaults",
            ]
            present_num = [c for c in num_cols if c in df.columns]
            present_cat = [c for c in cat_cols if c in df.columns]
            self.feature_cols = present_num + present_cat

            feature_data = df[self.feature_cols].fillna(0).values.astype(float)
            return self.scaler.fit_transform(feature_data)

        # transform mode: use feature_cols from fit; zero-fill any missing cols
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                df[col] = 0
            else:
                known = set(le.classes_)
                vals = df[col].fillna("unknown").astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(vals)

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        feature_data = df[self.feature_cols].fillna(0).values.astype(float)
        return self.scaler.transform(feature_data)

    def recommend_new_sme(self, sme_profile: dict, n: int = 5) -> list:
        defaults = {
            "sme_id": "SME_NEW", "country": "Kenya", "sector": "retail",
            "urban_rural": "urban", "annual_revenue_usd": 1000,
            "n_employees": 3, "years_in_business": 2,
            "has_bank_account_yn": 0, "has_mobile_money_yn": 0,
            "has_previous_loan": 0, "has_default": 0,
            "n_previous_loans": 0, "n_defaults": 0,
            "bureau_score": np.nan,
        }
        defaults.update(sme_profile)
        profile_df = pd.DataFrame([defaults])

        new_vec = self._encode_features(profile_df, fit=False)
        similarities = cosine_similarity(new_vec, self.feature_matrix)[0]
        neighbor_indices = np.argsort(similarities)[::-1][:self.n_neighbors]

        product_scores = {}
        for ni in neighbor_indices:
            neighbor_sme_id = self.sme_features.iloc[ni]["sme_id"]
            sim_score = float(similarities[ni])
            if neighbor_sme_id not in self.user_item_matrix.index:
                continue
            ratings = self.user_item_matrix.loc[neighbor_sme_id]
            for pid, rating in ratings.items():
                if rating > 0:
                    product_scores.setdefault(pid, []).append(sim_score * rating)

        results = []
        for pid, scores_list in product_scores.items():
            avg_score = np.mean(scores_list)
            confidence = min(1.0, len(scores_list) / self.n_neighbors)
            results.append({
                "product_id": int(pid),
                "product_name": PRODUCT_NAMES.get(int(pid), f"Product {pid}"),
                "score": round(float(avg_score), 3),
                "confidence": round(confidence, 3),
                "n_similar_smes": len(scores_list),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n]

    def onboarding_questions(self) -> list:
        return ONBOARDING_QUESTIONS

    def update_on_first_interaction(self, sme_id: str, product_id: int, rating: float):
        if sme_id not in self.user_item_matrix.index:
            logger.warning(f"SME {sme_id} not in matrix — adding new row.")
            new_row = pd.Series(0, index=self.user_item_matrix.columns, name=sme_id)
            self.user_item_matrix = pd.concat([self.user_item_matrix, new_row.to_frame().T])
        self.user_item_matrix.at[sme_id, product_id] = rating
        logger.info(f"Updated SME {sme_id} — product {product_id} rated {rating}.")
