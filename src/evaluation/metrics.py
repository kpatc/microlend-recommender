import numpy as np
import pandas as pd
from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    def dcg(items, rel_set, k):
        score = 0.0
        for i, item in enumerate(items[:k]):
            if item in rel_set:
                score += 1.0 / np.log2(i + 2)
        return score

    rel_set = set(relevant)
    actual_dcg = dcg(recommended, rel_set, k)
    ideal_items = [r for r in recommended if r in rel_set] + [r for r in recommended if r not in rel_set]
    ideal_dcg = dcg(ideal_items, rel_set, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def coverage(all_recommendations: list, all_products: list) -> float:
    recommended_set = set(item for recs in all_recommendations for item in recs)
    return len(recommended_set) / len(all_products) if all_products else 0.0


def novelty(recommendations: list, product_popularity: dict) -> float:
    scores = []
    for pid in recommendations:
        pop = product_popularity.get(pid, 1e-6)
        scores.append(-np.log2(pop))
    return float(np.mean(scores)) if scores else 0.0


def run_full_evaluation(model, test_data: pd.DataFrame, all_product_ids: list,
                        k_values: list = None, model_name: str = "model",
                        config: dict = None, log_mlflow: bool = True) -> pd.DataFrame:
    if k_values is None:
        k_values = [1, 5, 10]

    rows = []
    grouped = test_data.groupby("sme_id")
    all_recommendations_per_k = {k: [] for k in k_values}

    for sme_id, group in grouped:
        relevant = list(group[group["rating"] >= 3]["product_id"])
        if not relevant:
            continue

        try:
            preds = model.predict(sme_id, n_recommendations=max(k_values))
            recommended_ids = [p for p, _ in preds]
        except Exception:
            continue

        for k in k_values:
            p_at_k = precision_at_k(recommended_ids, relevant, k)
            r_at_k = recall_at_k(recommended_ids, relevant, k)
            n_at_k = ndcg_at_k(recommended_ids, relevant, k)
            rows.append({"sme_id": sme_id, "k": k,
                         "precision": p_at_k, "recall": r_at_k, "ndcg": n_at_k})
            all_recommendations_per_k[k].append(recommended_ids[:k])

    results_df = pd.DataFrame(rows)
    summary_rows = []
    for k in k_values:
        subset = results_df[results_df["k"] == k]
        cov = coverage(all_recommendations_per_k[k], all_product_ids)
        row = {
            "model": model_name, "k": k,
            f"precision@{k}": subset["precision"].mean(),
            f"recall@{k}": subset["recall"].mean(),
            f"ndcg@{k}": subset["ndcg"].mean(),
            f"coverage@{k}": cov,
        }
        summary_rows.append(row)
        logger.info(f"{model_name} @{k}: P={row[f'precision@{k}']:.4f}, "
                    f"R={row[f'recall@{k}']:.4f}, NDCG={row[f'ndcg@{k}']:.4f}")

    if log_mlflow and MLFLOW_AVAILABLE:
        from src.tracking import setup_mlflow
        if config:
            setup_mlflow(config)
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("k_values", str(k_values))
            mlflow.log_param("n_test_smes", results_df["sme_id"].nunique())
            for row in summary_rows:
                mlflow.log_metrics({key: val for key, val in row.items()
                                    if isinstance(val, float)})

    return pd.DataFrame(summary_rows)
