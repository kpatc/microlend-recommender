import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger


class UserBasedCF:
    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.user_item_matrix: np.ndarray = None
        self.sme_features: pd.DataFrame = None
        self.similarity_matrix: np.ndarray = None
        self.sme_ids: list = None
        self.product_ids: list = None

    def fit(self, user_item_matrix: pd.DataFrame, sme_features: pd.DataFrame = None):
        self.sme_ids = list(user_item_matrix.index)
        self.product_ids = list(user_item_matrix.columns)
        self.user_item_matrix = user_item_matrix.values.astype(float)
        self.sme_features = sme_features

        logger.info("Computing user-user cosine similarity...")
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(self.similarity_matrix, 0)

        if sme_features is not None:
            self._apply_sector_bonus()

        logger.success(f"UserBasedCF fitted on {len(self.sme_ids)} SMEs.")
        return self

    def _apply_sector_bonus(self):
        sectors = self.sme_features.set_index("sme_id")["sector"]
        for i, sid_i in enumerate(self.sme_ids):
            for j, sid_j in enumerate(self.sme_ids):
                if i == j:
                    continue
                if sid_i in sectors.index and sid_j in sectors.index:
                    if sectors[sid_i] == sectors[sid_j]:
                        self.similarity_matrix[i, j] *= 1.2
        self.similarity_matrix = np.clip(self.similarity_matrix, 0, 1)

    def predict(self, sme_id, n_recommendations: int = 5) -> list:
        if sme_id not in self.sme_ids:
            raise ValueError(f"SME {sme_id} not in training data.")

        idx = self.sme_ids.index(sme_id)
        user_vector = self.user_item_matrix[idx]
        sim_scores = self.similarity_matrix[idx]

        # Top-K neighbors
        neighbor_indices = np.argsort(sim_scores)[::-1][:self.n_neighbors]

        predictions = {}
        user_mean = user_vector[user_vector > 0].mean() if user_vector.any() else 3.0

        for prod_idx, prod_id in enumerate(self.product_ids):
            if user_vector[prod_idx] > 0:
                continue  # already adopted

            numerator = 0.0
            denominator = 0.0
            for neighbor_idx in neighbor_indices:
                neighbor_rating = self.user_item_matrix[neighbor_idx, prod_idx]
                if neighbor_rating == 0:
                    continue
                neighbor_vec = self.user_item_matrix[neighbor_idx]
                neighbor_mean = neighbor_vec[neighbor_vec > 0].mean() if neighbor_vec.any() else 3.0
                sim = sim_scores[neighbor_idx]
                numerator += sim * (neighbor_rating - neighbor_mean)
                denominator += abs(sim)

            if denominator > 0:
                pred = user_mean + numerator / denominator
                predictions[prod_id] = float(np.clip(pred, 1, 5))

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:n_recommendations]

    def explain(self, sme_id, product_id) -> str:
        idx = self.sme_ids.index(sme_id)
        sim_scores = self.similarity_matrix[idx]
        neighbor_indices = np.argsort(sim_scores)[::-1][:self.n_neighbors]

        prod_idx = self.product_ids.index(product_id)
        ratings = [self.user_item_matrix[ni, prod_idx] for ni in neighbor_indices
                   if self.user_item_matrix[ni, prod_idx] > 0]

        avg_rating = np.mean(ratings) if ratings else 3.0
        sector = "your sector"
        if self.sme_features is not None:
            row = self.sme_features[self.sme_features["sme_id"] == sme_id]
            if not row.empty:
                sector = row.iloc[0]["sector"]

        return (f"SMEs similar to yours in {sector} adopted this product "
                f"with avg satisfaction of {avg_rating:.1f}/5 "
                f"({len(ratings)} similar SMEs).")

    def get_similar_smes(self, sme_id, n: int = 10) -> list:
        idx = self.sme_ids.index(sme_id)
        sim_scores = self.similarity_matrix[idx]
        top_indices = np.argsort(sim_scores)[::-1][:n]
        return [(self.sme_ids[i], float(sim_scores[i])) for i in top_indices]
