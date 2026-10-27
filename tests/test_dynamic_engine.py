import pytest
import pandas as pd
import numpy as np
from src.dynamic_engine import DynamicDataEngine


@pytest.fixture
def sample_arbitrary_dataset():
    """An arbitrary multi-column synthetic tabular dataset."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        "account_id": [f"ACC_{i}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 120000, size=n),
        "tenure_months": np.random.randint(1, 60, size=n),
        "credit_score": np.random.normal(650, 50, size=n),
        "plan_type": np.random.choice(["Basic", "Standard", "Premium"], size=n),
        "is_active": np.random.choice(["Yes", "No"], size=n),
        "churn": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    })


class TestDynamicDataEngine:
    """Unit and integration tests for DynamicDataEngine."""

    def test_auto_detect_columns(self, sample_arbitrary_dataset):
        engine = DynamicDataEngine(sample_arbitrary_dataset)
        detection = engine.auto_detect_columns()

        assert "churn" in detection["target_candidates"] or detection["suggested_target"] == "churn"
        assert "account_id" in detection["id_candidates"]
        assert "age" in detection["numeric_candidates"]
        assert "income" in detection["numeric_candidates"]

    def test_sanitize_and_prepare(self, sample_arbitrary_dataset):
        engine = DynamicDataEngine(sample_arbitrary_dataset)
        detection = engine.auto_detect_columns()
        
        prep = engine.sanitize_and_prepare(
            target_col="churn",
            id_cols=detection["id_candidates"],
            positive_value="Yes"
        )

        assert prep["total_samples"] == 120
        assert prep["positive_label"] == "Yes"
        assert prep["negative_label"] == "No"
        assert engine.X_encoded is not None
        assert engine.y_clean is not None
        assert "account_id" not in engine.encoded_feature_names

    def test_train_and_benchmark_models(self, sample_arbitrary_dataset):
        engine = DynamicDataEngine(sample_arbitrary_dataset)
        engine.sanitize_and_prepare(target_col="churn", id_cols=["account_id"], positive_value="Yes")
        results = engine.train_and_benchmark_models(test_size=0.25)

        assert len(results) > 0
        for r in results:
            assert "Model" in r
            assert 0.0 <= r["Accuracy"] <= 1.0
            assert 0.0 <= r["Precision"] <= 1.0
            assert 0.0 <= r["Recall"] <= 1.0
            assert 0.0 <= r["F1_Score"] <= 1.0

    def test_compute_clustering_and_anomalies(self, sample_arbitrary_dataset):
        engine = DynamicDataEngine(sample_arbitrary_dataset)
        engine.sanitize_and_prepare(target_col="churn", id_cols=["account_id"], positive_value="Yes")
        engine.train_and_benchmark_models()
        profiles = engine.compute_clustering_and_anomalies(n_clusters=3, contamination=0.05)

        assert len(profiles) == 3
        assert engine.scored_df is not None
        assert "Assigned_Cluster" in engine.scored_df.columns
        assert "Is_Anomaly" in engine.scored_df.columns
        assert "Predicted_Target" in engine.scored_df.columns
        assert "PCA_Dim1" in engine.scored_df.columns
        assert "PCA_Dim2" in engine.scored_df.columns
