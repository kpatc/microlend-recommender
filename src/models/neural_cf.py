import time
import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. NeuralCF will not be available.")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class _NullCtx:
    """No-op context manager used when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if MLFLOW_AVAILABLE:
    class _NeuralCFPyfuncModel(mlflow.pyfunc.PythonModel):
        """MLflow pyfunc wrapper around a fitted NeuralCF instance.

        Input DataFrame columns: sme_id (str), product_id (int).
        Output: Series of predicted ratings (float, scale 1–5).
        """
        def __init__(self, ncf):
            self.ncf = ncf

        def predict(self, context, model_input: "pd.DataFrame") -> "pd.Series":
            import pandas as pd
            preds = [
                float(self.ncf.predict(str(row["sme_id"]), [int(row["product_id"])])[0])
                for _, row in model_input.iterrows()
            ]
            return pd.Series(preds, name="rating_hat")


class NeuralCFModel(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32,
                 hidden_layers: list = None, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NeuralCF.")
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        for hidden_dim in hidden_layers:
            mlp_layers += [nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            mlp_input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim + hidden_layers[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i

        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=1))

        concat = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output_layer(concat).squeeze()


class NeuralCF:
    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required.")
        self.full_config = config
        self.config = config.get("models", {}).get("neural_cf", {})
        self.model: NeuralCFModel = None
        self.user_map: dict = {}
        self.item_map: dict = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self, n_users: int, n_items: int):
        emb_dim = self.config.get("embedding_dim", 32)
        hidden = self.config.get("hidden_layers", [64, 32, 16])
        dropout = self.config.get("dropout", 0.2)
        self.model = NeuralCFModel(n_users, n_items, emb_dim, hidden, dropout).to(self.device)
        return self

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        from src.tracking import setup_mlflow
        unique_users = train_df["sme_id"].unique()
        unique_items = train_df["product_id"].unique()
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {p: i for i, p in enumerate(unique_items)}

        self.build_model(len(unique_users), len(unique_items))

        lr = self.config.get("lr", 0.001)
        epochs = self.config.get("epochs", 30)
        batch_size = self.config.get("batch_size", 256)
        emb_dim = self.config.get("embedding_dim", 32)
        hidden = self.config.get("hidden_layers", [64, 32, 16])
        dropout = self.config.get("dropout", 0.2)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        def df_to_tensors(df):
            u = torch.tensor([self.user_map.get(x, 0) for x in df["sme_id"]], dtype=torch.long)
            i = torch.tensor([self.item_map.get(x, 0) for x in df["product_id"]], dtype=torch.long)
            r = torch.tensor((df["rating"].values - 1) / 4, dtype=torch.float32)
            return TensorDataset(u, i, r)

        train_loader = DataLoader(df_to_tensors(train_df), batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience, patience_counter = 5, 0

        if MLFLOW_AVAILABLE:
            setup_mlflow(self.full_config)

        run_ctx = mlflow.start_run(run_name="NeuralCF") if MLFLOW_AVAILABLE else _NullCtx()
        t0 = time.time()

        with run_ctx:
            if MLFLOW_AVAILABLE:
                mlflow.log_params({
                    "model": "NeuralCF",
                    "embedding_dim": emb_dim,
                    "hidden_layers": str(hidden),
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "n_users": len(unique_users),
                    "n_items": len(unique_items),
                    "device": str(self.device),
                })

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for u_batch, i_batch, r_batch in train_loader:
                    u_batch = u_batch.to(self.device)
                    i_batch = i_batch.to(self.device)
                    r_batch = r_batch.to(self.device)
                    optimizer.zero_grad()
                    preds = self.model(u_batch, i_batch)
                    loss = criterion(preds, r_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                val_loss = None
                if val_df is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_tensors = df_to_tensors(val_df)
                        vu, vi, vr = val_tensors.tensors
                        vu, vi, vr = vu.to(self.device), vi.to(self.device), vr.to(self.device)
                        val_preds = self.model(vu, vi)
                        val_loss = criterion(val_preds, vr).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            break

                if MLFLOW_AVAILABLE:
                    metrics = {"train_loss": train_loss}
                    if val_loss is not None:
                        metrics["val_loss"] = val_loss
                    mlflow.log_metrics(metrics, step=epoch)

                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs} — train_loss={train_loss:.4f}" +
                                (f", val_loss={val_loss:.4f}" if val_loss else ""))

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "best_val_loss": best_val_loss,
                    "fit_time_s": time.time() - t0,
                })

                registry_cfg = self.full_config.get("mlflow", {}).get("registry", {})
                if registry_cfg.get("auto_register", True):
                    registered_name = registry_cfg.get("model_name", "microlend_recommender") + "_NeuralCF"
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=_NeuralCFPyfuncModel(self),
                        registered_model_name=registered_name,
                    )
                    logger.success(f"Registered '{registered_name}' in Model Registry")

        return self

    def predict(self, sme_id, product_ids: list) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([self.user_map.get(sme_id, 0)] * len(product_ids), dtype=torch.long).to(self.device)
            i = torch.tensor([self.item_map.get(pid, 0) for pid in product_ids], dtype=torch.long).to(self.device)
            preds = self.model(u, i).cpu().numpy()
        return preds * 4 + 1  # scale back to 1-5

    def recommend(self, sme_id, n: int = 5) -> list:
        all_products = list(self.item_map.keys())
        scores = self.predict(sme_id, all_products)
        ranked = sorted(zip(all_products, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def get_embeddings(self) -> dict:
        return {
            "user_gmf": self.model.gmf_user_emb.weight.detach().cpu().numpy(),
            "item_gmf": self.model.gmf_item_emb.weight.detach().cpu().numpy(),
            "user_mlp": self.model.mlp_user_emb.weight.detach().cpu().numpy(),
            "item_mlp": self.model.mlp_item_emb.weight.detach().cpu().numpy(),
        }
