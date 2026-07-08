import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, coverage


def _make_matrix(n_smes: int = 100, n_products: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # ~70% sparse: only 30% of entries are non-zero ratings 1-5
    data = rng.choice([0, 0, 0, 1, 2, 3, 4, 5], size=(n_smes, n_products))
    sme_ids = [f"SME_{i:05d}" for i in range(1, n_smes + 1)]
    return pd.DataFrame(data.astype(float), index=sme_ids,
                        columns=list(range(1, n_products + 1)))


def _make_sme_features(n_smes: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sme_ids = [f"SME_{i:05d}" for i in range(1, n_smes + 1)]
    return pd.DataFrame({
        "sme_id": sme_ids,
        "country": rng.choice(["Kenya", "Nigeria", "Ghana"], n_smes),
        "sector": rng.choice(["agriculture", "retail", "services"], n_smes),
        "annual_revenue_usd": rng.integers(500, 50000, n_smes).astype(float),
        "n_employees": rng.integers(1, 20, n_smes).astype(float),
        "years_in_business": rng.integers(1, 15, n_smes).astype(float),
        "bureau_score": rng.integers(300, 850, n_smes).astype(float),
        "has_bank_account_yn": rng.integers(0, 2, n_smes).astype(int),
        "has_mobile_money_yn": rng.integers(0, 2, n_smes).astype(int),
        "has_previous_loan": rng.integers(0, 2, n_smes).astype(int),
        "has_default": rng.integers(0, 2, n_smes).astype(int),
        "n_previous_loans": rng.integers(0, 5, n_smes).astype(int),
        "n_defaults": rng.integers(0, 2, n_smes).astype(int),
    })


@pytest.fixture(scope="module")
def matrix():
    return _make_matrix(200, seed=0)


@pytest.fixture(scope="module")
def sme_features():
    return _make_sme_features(200, seed=0)


def test_matrix_shapes(matrix):
    assert matrix.shape == (200, 8)
    assert (matrix >= 0).all().all()
    assert (matrix <= 5).all().all()


def test_user_based_cf(matrix, sme_features):
    model = UserBasedCF(n_neighbors=10)
    model.fit(matrix, sme_features)
    sme_id = matrix.index[0]
    recs = model.predict(sme_id, n_recommendations=5)
    assert len(recs) <= 5
    for pid, score in recs:
        assert 1 <= score <= 5
        assert matrix.loc[sme_id, pid] == 0


def test_item_based_cf(matrix):
    model = ItemBasedCF(n_neighbors=3)
    model.fit(matrix)
    sme_id = matrix.index[0]
    recs = model.predict(sme_id, n_recommendations=3)
    assert len(recs) <= 3


def test_metrics_rmse():
    actual = [3, 4, 5, 2]
    predicted = [3, 4, 5, 2]
    assert rmse(actual, predicted) == pytest.approx(0.0)


def test_metrics_precision_at_k():
    recommended = [1, 2, 3, 4, 5]
    relevant = [2, 4]
    assert precision_at_k(recommended, relevant, k=5) == pytest.approx(0.4)
    assert precision_at_k(recommended, relevant, k=1) == pytest.approx(0.0)


def test_metrics_recall_at_k():
    recommended = [2, 4, 6, 8]
    relevant = [2, 4]
    assert recall_at_k(recommended, relevant, k=2) == pytest.approx(1.0)
    assert recall_at_k(recommended, relevant, k=1) == pytest.approx(0.5)


def test_metrics_ndcg():
    recommended = [1, 2, 3]
    relevant = [1]
    score = ndcg_at_k(recommended, relevant, k=3)
    assert 0 <= score <= 1


def test_coverage():
    recs = [[1, 2], [2, 3], [4, 5]]
    all_products = [1, 2, 3, 4, 5, 6, 7, 8]
    cov = coverage(recs, all_products)
    assert cov == pytest.approx(5 / 8)


def test_cold_start_solver(matrix, sme_features):
    from src.cold_start.solver import ColdStartSolver
    solver = ColdStartSolver(n_neighbors=10)
    solver.fit(sme_features, matrix)
    profile = {
        "sector": "agriculture", "country": "Kenya",
        "annual_revenue_usd": 2000, "n_employees": 5,
        "has_bank_account_yn": 0, "has_mobile_money_yn": 1,
        "has_previous_loan": 0,
    }
    recs = solver.recommend_new_sme(profile, n=5)
    assert len(recs) >= 1
    for r in recs:
        assert "product_id" in r
        assert "confidence" in r
        assert 0 <= r["confidence"] <= 1
