import numpy as np
import pandas as pd
from loguru import logger


class ItemBasedCF:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.user_item_matrix: np.ndarray = None
        self.item_similarity: np.ndarray = None
        self.sme_ids: list = None
        self.product_ids: list = None

    def fit(self, user_item_matrix: pd.DataFrame):
        self.sme_ids = list(user_item_matrix.index)
        self.product_ids = list(user_item_matrix.columns)
        self.user_item_matrix = user_item_matrix.values.astype(float)
        logger.info("Computing adjusted cosine item-item similarity...")
        self.item_similarity = self._adjusted_cosine_similarity()
        logger.success(f"ItemBasedCF fitted: {len(self.product_ids)} products.")
        return self

    def _adjusted_cosine_similarity(self) -> np.ndarray:
        matrix = self.user_item_matrix.copy()
        # Mean-center per user (row)
        user_means = np.where(matrix > 0, matrix, np.nan)
        user_means = np.nanmean(user_means, axis=1, keepdims=True)
        user_means = np.nan_to_num(user_means, nan=0)
        centered = np.where(matrix > 0, matrix - user_means, 0)

        n_items = centered.shape[1]
        sim = np.zeros((n_items, n_items))
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    sim[i, j] = 1.0
                    continue
                dot = np.dot(centered[:, i], centered[:, j])
                norm_i = np.linalg.norm(centered[:, i])
                norm_j = np.linalg.norm(centered[:, j])
                if norm_i > 0 and norm_j > 0:
                    sim[i, j] = dot / (norm_i * norm_j)
        return sim

    def predict(self, sme_id, n_recommendations: int = 5) -> list:
        if sme_id not in self.sme_ids:
            raise ValueError(f"SME {sme_id} not in training data.")

        idx = self.sme_ids.index(sme_id)
        user_vector = self.user_item_matrix[idx]
        rated_indices = np.where(user_vector > 0)[0]

        predictions = {}
        for prod_idx, prod_id in enumerate(self.product_ids):
            if user_vector[prod_idx] > 0:
                continue
            sims = self.item_similarity[prod_idx, rated_indices]
            top_k = np.argsort(sims)[::-1][:self.n_neighbors]
            top_sims = sims[top_k]
            top_ratings = user_vector[rated_indices[top_k]]

            denom = np.sum(np.abs(top_sims))
            if denom > 0:
                pred = np.dot(top_sims, top_ratings) / denom
                predictions[prod_id] = float(np.clip(pred, 1, 5))

        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    def get_product_associations(self, product_id, n: int = 5) -> list:
        if product_id not in self.product_ids:
            raise ValueError(f"Product {product_id} not found.")
        prod_idx = self.product_ids.index(product_id)
        sim_row = self.item_similarity[prod_idx].copy()
        sim_row[prod_idx] = -1
        top_indices = np.argsort(sim_row)[::-1][:n]

        # Co-adoption rates
        adopters = self.user_item_matrix[:, prod_idx] > 0
        result = []
        for ti in top_indices:
            co_adopters = (self.user_item_matrix[:, ti] > 0) & adopters
            rate = co_adopters.sum() / max(adopters.sum(), 1)
            result.append((self.product_ids[ti], float(sim_row[ti]), float(rate)))
        return result

    def explain(self, sme_id, product_id) -> str:
        idx = self.sme_ids.index(sme_id)
        user_vector = self.user_item_matrix[idx]
        prod_idx = self.product_ids.index(product_id)
        rated_indices = np.where(user_vector > 0)[0]
        sims = self.item_similarity[prod_idx, rated_indices]
        top_k = np.argsort(sims)[::-1][:2]
        drivers = [self.product_ids[rated_indices[i]] for i in top_k]
        return (f"Because you adopted products {drivers}, "
                f"we recommend product {product_id}.")
