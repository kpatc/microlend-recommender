import time
import numpy as np
import pandas as pd
import mlflow
from loguru import logger
from src.tracking import setup_mlflow

try:
    from surprise import SVD, NMF, BaselineOnly, Dataset, Reader
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger.warning("scikit-surprise not installed. MatrixFactorizationBenchmark will be limited.")


class _SurprisePyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around a fitted scikit-surprise algorithm.

    Input DataFrame columns: sme_id (str), product_id (int).
    Output: Series of predicted ratings (float, scale 1–5).
    """
    def __init__(self, algo):
        self.algo = algo

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        preds = [
            float(self.algo.predict(str(row["sme_id"]), str(row["product_id"])).est)
            for _, row in model_input.iterrows()
        ]
        return pd.Series(preds, name="rating_hat")


class MatrixFactorizationBenchmark:
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.results = {}
        self.trained_models = {}

    def build_models(self):
        if not SURPRISE_AVAILABLE:
            raise ImportError("Install scikit-surprise: pip install scikit-surprise")
        cfg = self.config.get("models", {})
        svd_cfg = cfg.get("svd", {})
        nmf_cfg = cfg.get("nmf", {})
        self.models = {
            "SVD": SVD(
                n_factors=svd_cfg.get("n_factors", 50),
                n_epochs=svd_cfg.get("n_epochs", 20),
                lr_all=svd_cfg.get("lr_all", 0.005),
                reg_all=svd_cfg.get("reg_all", 0.02),
            ),
            "NMF": NMF(
                n_factors=nmf_cfg.get("n_factors", 15),
                n_epochs=nmf_cfg.get("n_epochs", 50),
            ),
            "Baseline": BaselineOnly(),
        }
        return self

    def _model_params(self, model_name: str) -> dict:
        cfg = self.config.get("models", {})
        params = {"model": model_name}
        if model_name == "SVD":
            params.update(cfg.get("svd", {}))
        elif model_name == "NMF":
            params.update(cfg.get("nmf", {}))
        return params

    def run_cross_validation(self, data, n_folds: int = 5) -> pd.DataFrame:
        setup_mlflow(self.config)
        rows = []

        for model_name, model in self.models.items():
            logger.info(f"Cross-validating {model_name} ({n_folds} folds)...")
            t0 = time.time()

            with mlflow.start_run(run_name=f"cv_{model_name}"):
                cv_results = cross_validate(model, data, measures=["RMSE", "MAE"],
                                            cv=n_folds, verbose=False)
                elapsed = time.time() - t0

                rmse_mean = float(np.mean(cv_results["test_rmse"]))
                rmse_std  = float(np.std(cv_results["test_rmse"]))
                mae_mean  = float(np.mean(cv_results["test_mae"]))
                mae_std   = float(np.std(cv_results["test_mae"]))

                mlflow.log_params({**self._model_params(model_name), "n_folds": n_folds})
                mlflow.log_metrics({
                    "rmse_mean": rmse_mean,
                    "rmse_std":  rmse_std,
                    "mae_mean":  mae_mean,
                    "mae_std":   mae_std,
                    "fit_time_s": elapsed,
                })
                # Log per-fold metrics
                for fold_i, (rmse_f, mae_f) in enumerate(
                    zip(cv_results["test_rmse"], cv_results["test_mae"])
                ):
                    mlflow.log_metrics({"fold_rmse": float(rmse_f), "fold_mae": float(mae_f)},
                                       step=fold_i)

                rows.append({
                    "model": model_name,
                    "RMSE": rmse_mean,
                    "RMSE_std": rmse_std,
                    "MAE": mae_mean,
                    "MAE_std": mae_std,
                    "fit_time_s": round(elapsed, 2),
                })
                logger.info(f"  {model_name}: RMSE={rmse_mean:.4f}±{rmse_std:.4f}, "
                            f"MAE={mae_mean:.4f}±{mae_std:.4f} [{elapsed:.1f}s]")

        self.results = pd.DataFrame(rows)
        return self.results

    def fit(self, trainset):
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(trainset)
            self.trained_models[name] = model
        return self

    def recommend(self, sme_id, model_name: str = "SVD", n: int = 5,
                  all_product_ids: list = None) -> list:
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not trained.")
        if all_product_ids is None:
            all_product_ids = list(range(1, 9))

        preds = [(pid, model.predict(str(sme_id), str(pid)).est)
                 for pid in all_product_ids]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    def register_best_model(self, data, model_name: str = None) -> str:
        """
        Fit the best CV model (by RMSE) on the full dataset, log it as an mlflow
        pyfunc artifact, and register it in the MLflow Model Registry.
        Returns the registered model name.
        """
        if self.results is None or self.results.empty:
            raise ValueError("Run run_cross_validation() first.")

        best_row = self.results.loc[self.results["RMSE"].idxmin()]
        best_algo_name = best_row["model"]
        algo = self.models[best_algo_name]

        logger.info(f"Fitting {best_algo_name} (RMSE={best_row['RMSE']:.4f}) on full dataset...")
        trainset = data.build_full_trainset()
        algo.fit(trainset)
        self.trained_models[best_algo_name] = algo

        registry_cfg = self.config.get("mlflow", {}).get("registry", {})
        registered_name = model_name or registry_cfg.get("model_name", "microlend_recommender")

        with mlflow.start_run(run_name=f"register_{best_algo_name}"):
            mlflow.log_params({**self._model_params(best_algo_name), "stage": "full_fit"})
            mlflow.log_metrics({"rmse": float(best_row["RMSE"]), "mae": float(best_row["MAE"])})

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_SurprisePyfuncModel(algo),
                registered_model_name=registered_name,
            )
            run_id = mlflow.active_run().info.run_id

        logger.success(
            f"Registered '{registered_name}' (algo={best_algo_name}) in Model Registry — "
            f"run {run_id[:8]}"
        )
        return registered_name

    def get_latent_factors(self, model_name: str = "SVD") -> dict:
        model = self.trained_models.get(model_name)
        if model is None or not hasattr(model, "pu"):
            return {}
        return {"user_factors": model.pu, "item_factors": model.qi,
                "user_biases": model.bu, "item_biases": model.bi}

    def plot_latent_space(self, model_name: str = "SVD", item_features: pd.DataFrame = None):
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        factors = self.get_latent_factors(model_name)
        if not factors:
            logger.warning("No latent factors available.")
            return None

        item_factors = factors["item_factors"]
        pca = PCA(n_components=2)
        coords = pca.fit_transform(item_factors)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(coords)))
        for i, (x, y) in enumerate(coords):
            label = str(i + 1)
            if item_features is not None and not item_features.empty:
                row = item_features[item_features["product_id"] == i + 1]
                if not row.empty:
                    label = row.iloc[0]["name"]
            ax.scatter(x, y, color=colors[i], s=100, zorder=3)
            ax.annotate(label, (x, y), fontsize=8, ha="right")

        ax.set_title(f"{model_name} Product Latent Space (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        return fig
