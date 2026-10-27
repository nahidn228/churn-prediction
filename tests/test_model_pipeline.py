import pytest
import numpy as np
import pandas as pd
from src.data_processor import DataProcessor


class TestModelPipeline:
    """Unit and integration tests for trained model pipeline and artifacts."""

    EXPECTED_MODELS = [
        "Logistic Regression",
        "Random Forest Classifier",
        "SVM",
        "Decision Tree Classifier",
        "K-Nearest Neighbors (KNN)",
        "Naive Bayes"
    ]

    def test_metadata_structure(self, loaded_artifacts):
        meta = loaded_artifacts["metadata"]

        required_keys = [
            "feature_columns", "outlier_bounds", "evaluation_results",
            "feature_importances", "cluster_profiles", "cm_data", "roc_data",
            "viz_data", "total_samples", "churn_count"
        ]
        for key in required_keys:
            assert key in meta, f"Missing required metadata key: {key}"

        # 30 expected feature columns
        assert len(meta["feature_columns"]) == 30
        assert meta["total_samples"] > 0
        assert meta["churn_count"] > 0

    def test_evaluation_leaderboard_completeness(self, loaded_artifacts):
        eval_results = loaded_artifacts["metadata"]["evaluation_results"]
        assert len(eval_results) == 6

        model_names = [res["Model"] for res in eval_results]
        for expected in self.EXPECTED_MODELS:
            assert expected in model_names

        for res in eval_results:
            assert 0.0 <= res["Accuracy"] <= 1.0
            assert 0.0 <= res["Precision"] <= 1.0
            assert 0.0 <= res["Recall"] <= 1.0
            assert 0.0 <= res["F1_Score"] <= 1.0
            if res["ROC_AUC"] is not None:
                assert 0.5 <= res["ROC_AUC"] <= 1.0

    def test_supervised_models_loaded(self, loaded_artifacts):
        models = loaded_artifacts["models"]
        for expected in self.EXPECTED_MODELS:
            assert expected in models, f"Model {expected} missing from supervised_models.joblib"

    def test_supervised_inference_and_risk_ordering(
        self, loaded_artifacts, sample_high_risk_customer, sample_loyal_customer
    ):
        scaler = loaded_artifacts["scaler"]
        meta = loaded_artifacts["metadata"]
        outlier_bounds = meta.get("outlier_bounds")

        high_feat = DataProcessor.prepare_inference_features(sample_high_risk_customer, outlier_bounds=outlier_bounds)
        loyal_feat = DataProcessor.prepare_inference_features(sample_loyal_customer, outlier_bounds=outlier_bounds)

        high_scaled = scaler.transform(high_feat)
        loyal_scaled = scaler.transform(loyal_feat)

        for model_name, model in loaded_artifacts["models"].items():
            pred_high = model.predict(high_scaled)[0]
            pred_loyal = model.predict(loyal_scaled)[0]
            assert pred_high in [0, 1]
            assert pred_loyal in [0, 1]

            if hasattr(model, "predict_proba"):
                prob_high = model.predict_proba(high_scaled)[0, 1]
                prob_loyal = model.predict_proba(loyal_scaled)[0, 1]
                assert 0.0 <= prob_high <= 1.0
                assert 0.0 <= prob_loyal <= 1.0

                # High-risk profile should score higher probability than loyal profile
                assert prob_high > prob_loyal, (
                    f"Model {model_name} failed risk ordering: high={prob_high:.3f}, loyal={prob_loyal:.3f}"
                )

    def test_kmeans_clustering_inference(self, loaded_artifacts, sample_raw_customer_dict):
        scaler = loaded_artifacts["scaler"]
        kmeans = loaded_artifacts["kmeans"]
        meta = loaded_artifacts["metadata"]

        feat = DataProcessor.prepare_inference_features(
            sample_raw_customer_dict, outlier_bounds=meta.get("outlier_bounds")
        )
        scaled = scaler.transform(feat)
        cluster = kmeans.predict(scaled)[0]

        assert cluster in [0, 1, 2]
        # Cluster should exist in metadata profiles
        assert str(cluster) in meta["cluster_profiles"] or cluster in meta["cluster_profiles"]

    def test_isolation_forest_anomaly_detection(self, loaded_artifacts, sample_loyal_customer):
        scaler = loaded_artifacts["scaler"]
        iso_forest = loaded_artifacts["iso_forest"]

        feat = DataProcessor.prepare_inference_features(sample_loyal_customer)
        scaled = scaler.transform(feat)
        result = iso_forest.predict(scaled)[0]

        # 1 = normal, -1 = anomaly
        assert result in [1, -1]

    def test_pca_projections(self, loaded_artifacts, sample_raw_customer_dict):
        scaler = loaded_artifacts["scaler"]
        pca_2d = loaded_artifacts["pca_2d"]
        pca_3d = loaded_artifacts["pca_3d"]

        feat = DataProcessor.prepare_inference_features(sample_raw_customer_dict)
        scaled = scaler.transform(feat)

        coords_2d = pca_2d.transform(scaled)
        coords_3d = pca_3d.transform(scaled)

        assert coords_2d.shape == (1, 2)
        assert coords_3d.shape == (1, 3)
        assert np.all(np.isfinite(coords_2d))
        assert np.all(np.isfinite(coords_3d))
