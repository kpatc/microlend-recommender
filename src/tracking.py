"""Central MLflow configuration — import this before any mlflow call."""

from pathlib import Path
import mlflow
from mlflow import MlflowClient
from loguru import logger


def setup_mlflow(config: dict) -> str:
    """Set tracking URI + experiment from config. Returns experiment_id."""
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "microlend_recommender")

    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = Path(tracking_uri.replace("sqlite:///", "")).resolve()
        abs_uri = f"sqlite:///{db_path}"
    elif not tracking_uri.startswith(("sqlite", "http", "postgresql", "mysql")):
        abs_uri = Path(tracking_uri).resolve().as_uri()
    else:
        abs_uri = tracking_uri

    mlflow.set_tracking_uri(abs_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using MLflow experiment '{experiment_name}' (id={experiment_id}) → {abs_uri}")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_config_params(config: dict, prefix: str = ""):
    """Flatten and log config sections as MLflow params."""
    def _flatten(d, parent=""):
        items = {}
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            elif not isinstance(v, list):
                items[key] = v
        return items

    flat = _flatten(config)
    if prefix:
        flat = {f"{prefix}.{k}": v for k, v in flat.items()}
    mlflow.log_params(flat)


# ── Model Registry ────────────────────────────────────────────────────────────

def register_model(run_id: str, artifact_path: str, model_name: str):
    """Register a logged run artifact to the Model Registry. Returns ModelVersion."""
    uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(uri, model_name)
    logger.info(f"Registered '{model_name}' v{mv.version} ← run {run_id[:8]}")
    return mv


def set_alias(model_name: str, version, alias: str):
    """Assign an alias (e.g. 'champion', 'staging') to a specific model version."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))
    logger.info(f"'{model_name}' v{version} ← alias '{alias}'")


def delete_alias(model_name: str, alias: str):
    """Remove an alias from a registered model."""
    MlflowClient().delete_registered_model_alias(model_name, alias)
    logger.info(f"Removed alias '{alias}' from '{model_name}'")


def set_production(model_name: str, metric: str = "rmse", ascending: bool = True):
    """
    Assign the 'champion' alias to the registered version with the best value for
    `metric`. Falls back to the latest version if no run has that metric logged.
    Returns the promoted ModelVersion.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        logger.warning(f"No versions registered for '{model_name}'")
        return None

    best_version, best_val = None, float("inf") if ascending else float("-inf")
    for mv in versions:
        try:
            run = client.get_run(mv.run_id)
            val = run.data.metrics.get(metric)
            if val is None:
                continue
            if (ascending and val < best_val) or (not ascending and val > best_val):
                best_val, best_version = val, mv
        except Exception:
            continue

    if best_version is None:
        best_version = sorted(versions, key=lambda v: int(v.version))[-1]
        logger.warning(f"Metric '{metric}' not found — promoting latest version.")
        suffix = ""
    else:
        suffix = f"  ({metric}={best_val:.4f})"

    client.set_registered_model_alias(model_name, "champion", best_version.version)
    logger.success(f"'{model_name}' v{best_version.version} → alias 'champion'{suffix}")
    return best_version


def set_staging(model_name: str, version):
    """Assign the 'challenger' alias to a version for staging evaluation."""
    MlflowClient().set_registered_model_alias(model_name, "challenger", str(version))
    logger.info(f"'{model_name}' v{version} → alias 'challenger'")


def load_production_model(model_name: str):
    """Load the 'champion' alias model from the registry as an mlflow.pyfunc model."""
    uri = f"models:/{model_name}@champion"
    model = mlflow.pyfunc.load_model(uri)
    logger.info(f"Loaded '{model_name}@champion' from Model Registry")
    return model


def list_registered_models() -> list[dict]:
    """Return all registered model versions with their aliases as a list of dicts."""
    client = MlflowClient()
    rows = []
    for rm in client.search_registered_models():
        # rm.aliases is {alias_name: version_number} in MLflow 3.x
        aliases = getattr(rm, "aliases", {}) or {}
        alias_map = {str(v): k for k, v in aliases.items()}
        for mv in client.search_model_versions(f"name='{rm.name}'"):
            rows.append({
                "name": mv.name,
                "version": int(mv.version),
                "alias": alias_map.get(str(mv.version), ""),
                "run_id": mv.run_id,
            })
    return rows
