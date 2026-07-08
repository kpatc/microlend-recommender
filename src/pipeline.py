"""
Full training + registration pipeline.
Usage: venv/bin/python src/pipeline.py [--no-register]
"""

import sys
import yaml
from loguru import logger

from src.tracking import setup_mlflow, set_production, list_registered_models
from src.data.loader import DataLoader
from src.data.preprocessing import build_surprise_dataset
from src.models.matrix_factorization import MatrixFactorizationBenchmark


def run(register: bool = True):
    config = yaml.safe_load(open("configs/config.yaml"))
    setup_mlflow(config)

    loader = DataLoader(config)
    ratings = loader.load_ratings_long()
    logger.info(f"Dataset: {len(ratings)} ratings, {ratings['sme_id'].nunique()} SMEs")

    data = build_surprise_dataset(ratings)

    bench = MatrixFactorizationBenchmark(config)
    bench.build_models()
    results = bench.run_cross_validation(data)

    logger.info("\n" + results.to_string())

    if register:
        model_name = bench.register_best_model(data)
        promoted = set_production(model_name, metric="rmse")
        if promoted:
            logger.success(
                f"Model Registry: '{model_name}' v{promoted.version} is now in Production"
            )

        logger.info("\nRegistered models:")
        for row in list_registered_models():
            alias = f"@{row['alias']}" if row['alias'] else ""
            logger.info(f"  {row['name']}  v{row['version']}{alias}  run={row['run_id'][:8]}")

    return results


if __name__ == "__main__":
    no_register = "--no-register" in sys.argv
    run(register=not no_register)
