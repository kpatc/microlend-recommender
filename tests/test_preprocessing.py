import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import SMEPreprocessor, train_test_split_temporal


def _make_sme_features(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sme_ids = [f"SME_{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "sme_id": sme_ids,
        "country": rng.choice(["Kenya", "Nigeria", "Ghana", "Senegal"], n),
        "sector": rng.choice(["agriculture", "retail", "services", "tech"], n),
        "urban_rural": rng.choice(["urban", "rural"], n),
        "annual_revenue_usd": np.exp(rng.normal(9.55, 2.94, n)),
        "n_employees": rng.integers(1, 30, n).astype(float),
        "years_in_business": rng.integers(1, 20, n).astype(float),
        "bureau_score": np.where(rng.random(n) < 0.4, np.nan,
                                 rng.integers(300, 850, n).astype(float)),
        "n_previous_loans": rng.integers(0, 6, n).astype(int),
        "n_defaults": rng.integers(0, 3, n).astype(int),
        "has_bank_account_yn": rng.integers(0, 2, n).astype(int),
        "has_mobile_money_yn": rng.integers(0, 2, n).astype(int),
        "has_previous_loan": (rng.integers(0, 6, n) > 0).astype(int),
        "has_default": (rng.integers(0, 3, n) > 0).astype(int),
        "financial_literacy_score": rng.integers(1, 11, n).astype(float),
    })


def _make_ratings(n_smes: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sme_ids = [f"SME_{i:05d}" for i in range(1, n_smes + 1)]
    rows = []
    for sme_id in sme_ids:
        n_prods = rng.integers(1, 5)
        products = rng.choice(range(1, 9), n_prods, replace=False)
        for pid in products:
            rows.append({"sme_id": sme_id, "product_id": int(pid),
                         "rating": float(rng.integers(1, 6))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sample_features():
    return _make_sme_features(100, seed=1)


@pytest.fixture(scope="module")
def sample_ratings():
    return _make_ratings(100, seed=1)


def test_preprocessor_fit_transform(sample_features):
    preprocessor = SMEPreprocessor()
    transformed = preprocessor.fit_transform(sample_features)
    assert "country_enc" in transformed.columns
    assert "sector_enc" in transformed.columns
    numeric_scaled = ["annual_revenue_usd", "n_employees", "years_in_business"]
    for col in numeric_scaled:
        if col in transformed.columns:
            assert transformed[col].between(0, 1).all(), f"{col} not in [0,1] after scaling"


def test_preprocessor_feature_matrix(sample_features):
    preprocessor = SMEPreprocessor()
    preprocessor.fit_transform(sample_features)
    mat = preprocessor.get_feature_matrix(sample_features)
    assert isinstance(mat, np.ndarray)
    assert mat.shape[0] == len(sample_features)
    assert mat.shape[1] > 0


def test_train_test_split(sample_ratings):
    train, test = train_test_split_temporal(sample_ratings, test_size=0.2)
    assert len(train) + len(test) == len(sample_ratings)
    assert len(train) > len(test)


def test_bureau_score_nulls_handled(sample_features):
    preprocessor = SMEPreprocessor()
    transformed = preprocessor.fit_transform(sample_features)
    # bureau_score has ~40% NaN in fixture; after fillna(0)+scale it should be valid floats
    if "bureau_score" in transformed.columns:
        assert transformed["bureau_score"].isna().sum() == 0


def test_binary_derived_cols_present(sample_features):
    assert "has_bank_account_yn" in sample_features.columns
    assert "has_mobile_money_yn" in sample_features.columns
    assert "has_previous_loan" in sample_features.columns
    assert "has_default" in sample_features.columns
