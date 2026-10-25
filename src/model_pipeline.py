import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.data_processor import DataProcessor


class ModelPipeline:
    """
    Orchestrates dataset preprocessing, supervised model training,
    unsupervised clustering (K-Means), anomaly detection (Isolation Forest),
    and artifact export.
    """

    CLUSTER_LABELS = {
        0: {"name": "Core Long-Term Subscribers", "risk": "Low", "color": "#10B981",
            "desc": "High tenure, moderate-to-high spend, multi-service users with multi-year contracts."},
        1: {"name": "New / Month-to-Month Budget", "risk": "Medium", "color": "#F59E0B",
            "desc": "Low tenure, low monthly charges, basic phone/DSL, highly price-sensitive."},
        2: {"name": "High-Value At-Risk Streamers", "risk": "High", "color": "#EF4444",
            "desc": "High monthly charges, Fiber optic, streaming services, electronic check payment, vulnerable to churn."}
    }

    def __init__(self, data_path="TelcoCustomerChurn.csv", output_dir="models"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.scaler = None
        self.pca_2d = None
        self.pca_3d = None
        self.kmeans = None
        self.iso_forest = None
        self.outlier_bounds = {}
        self.feature_columns = []
        self.evaluation_results = []
        self.feature_importances = []
        self.cluster_profiles = {}

    def run_training_pipeline(self):
        """Execute complete end-to-end training and evaluation workflow."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[1/6] Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # 1. Clean data using DataProcessor
        print("[2/6] Cleaning and preprocessing dataset...")
        processor = DataProcessor(df)
        df = processor.rename_columns()
        df, _ = processor.drop_null_rows(how="all")
        df, _ = processor.drop_null_rows(subset=["Churn"], how="any")
        df, _ = processor.remove_duplicates()
        df, median_total = processor.clean_total_charges()
        df = processor.convert_data_types()

        # Map target
        df['Churn'] = df['Churn'].astype(str).str.strip().str.lower().map(
            {'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0}
        )
        df = df.dropna(subset=['Churn']).copy()
        df['Churn'] = df['Churn'].astype(int)

        # Separate target
        X = df.drop(columns=[col for col in ['CustomerID', 'Churn'] if col in df.columns])
        y = df['Churn']

        # One hot encode
        X_encoded = DataProcessor.encode_features(X)
        self.feature_columns = X_encoded.columns.tolist()

        # Stratified Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )

        # IQR Outlier Capping learned on training data
        continuous_cols = [c for c in DataProcessor.CONTINUOUS_COLS if c in X_train.columns]
        X_train, X_test, self.outlier_bounds = DataProcessor.cap_iqr_outliers(
            X_train, X_test, continuous_cols
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Full scaled data for unsupervised learning
        full_scaled = np.vstack((X_train_scaled, X_test_scaled))
        full_raw_df = pd.concat([X_train, X_test]).reset_index(drop=True)
        full_y = pd.concat([y_train, y_test]).reset_index(drop=True)

        print("[3/6] Training Supervised Classifiers...")
        defined_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        eval_summary = []
        roc_data = {}
        cm_data = {}

        for name, model in defined_models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            acc = float(accuracy_score(y_test, preds))
            prec = float(precision_score(y_test, preds))
            rec = float(recall_score(y_test, preds))
            f1 = float(f1_score(y_test, preds))
            roc_auc = float(roc_auc_score(y_test, probs)) if probs is not None else float("nan")

            eval_summary.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else None
            })

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            cm_data[name] = cm.tolist()

            # ROC Curve points
            if probs is not None:
                fpr, tpr, _ = roc_curve(y_test, probs)
                # Sample down points for clean JSON export
                idx_step = max(1, len(fpr) // 50)
                roc_data[name] = {
                    "fpr": [round(float(x), 4) for x in fpr[::idx_step]],
                    "tpr": [round(float(x), 4) for x in tpr[::idx_step]],
                    "auc": round(float(auc(fpr, tpr)), 4)
                }

            self.models[name] = model

        eval_df = pd.DataFrame(eval_summary).sort_values(by="Accuracy", ascending=False)
        self.evaluation_results = eval_df.to_dict(orient="records")

        # Feature Importance from Random Forest
        rf_model = self.models["Random Forest Classifier"]
        rf_importances = [
            {"feature": col, "importance": round(float(imp), 5)}
            for col, imp in zip(self.feature_columns, rf_model.feature_importances_)
        ]
        self.feature_importances = sorted(rf_importances, key=lambda x: x["importance"], reverse=True)

        print("[4/6] Training Unsupervised Models (K-Means & Isolation Forest)...")
        # K-Means (3 segments)
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(full_scaled)

        # PCA 2D & 3D for visualization
        self.pca_2d = PCA(n_components=2, random_state=42)
        pca_coords_2d = self.pca_2d.fit_transform(full_scaled)

        self.pca_3d = PCA(n_components=3, random_state=42)
        pca_coords_3d = self.pca_3d.fit_transform(full_scaled)

        # Isolation Forest for Anomaly Detection (contamination=0.03)
        self.iso_forest = IsolationForest(contamination=0.03, random_state=42)
        anomalies = self.iso_forest.fit_predict(full_scaled)  # -1 for anomaly, 1 for normal

        # Compute cluster characteristics
        full_raw_df['Cluster'] = clusters
        full_raw_df['Anomaly'] = anomalies
        full_raw_df['Churn'] = full_y

        cluster_stats = {}
        for c in range(3):
            sub = full_raw_df[full_raw_df['Cluster'] == c]
            cluster_stats[c] = {
                "count": int(len(sub)),
                "pct": round(float(len(sub) / len(full_raw_df) * 100), 1),
                "avg_tenure": round(float(sub['tenure'].mean()), 1) if 'tenure' in sub else 0,
                "avg_monthly_charges": round(float(sub['MonthlyCharges'].mean()), 2) if 'MonthlyCharges' in sub else 0,
                "avg_total_charges": round(float(sub['TotalCharges'].mean()), 2) if 'TotalCharges' in sub else 0,
                "churn_rate": round(float(sub['Churn'].mean() * 100), 1),
                "label": self.CLUSTER_LABELS[c]["name"],
                "risk": self.CLUSTER_LABELS[c]["risk"],
                "desc": self.CLUSTER_LABELS[c]["desc"]
            }
        self.cluster_profiles = cluster_stats

        # Sample subset for fast interactive 2D/3D visualization
        viz_sample_size = min(1500, len(full_scaled))
        np.random.seed(42)
        sample_indices = np.random.choice(len(full_scaled), viz_sample_size, replace=False)

        viz_data = {
            "pc1": [round(float(x), 4) for x in pca_coords_2d[sample_indices, 0]],
            "pc2": [round(float(x), 4) for x in pca_coords_2d[sample_indices, 1]],
            "pc3": [round(float(x), 4) for x in pca_coords_3d[sample_indices, 2]],
            "cluster": [int(clusters[i]) for i in sample_indices],
            "anomaly": [int(anomalies[i]) for i in sample_indices],
            "churn": [int(full_y.iloc[i]) for i in sample_indices],
            "tenure": [float(full_raw_df['tenure'].iloc[i]) for i in sample_indices],
            "monthly_charges": [float(full_raw_df['MonthlyCharges'].iloc[i]) for i in sample_indices]
        }

        print("[5/6] Exporting models and artifacts to disk...")
        joblib.dump(self.models, os.path.join(self.output_dir, "supervised_models.joblib"), compress=3)
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.joblib"), compress=3)
        joblib.dump(self.kmeans, os.path.join(self.output_dir, "kmeans.joblib"), compress=3)
        joblib.dump(self.iso_forest, os.path.join(self.output_dir, "isolation_forest.joblib"), compress=3)
        joblib.dump(self.pca_2d, os.path.join(self.output_dir, "pca_2d.joblib"), compress=3)
        joblib.dump(self.pca_3d, os.path.join(self.output_dir, "pca_3d.joblib"), compress=3)

        metadata = {
            "feature_columns": self.feature_columns,
            "outlier_bounds": self.outlier_bounds,
            "evaluation_results": self.evaluation_results,
            "feature_importances": self.feature_importances[:20],
            "cluster_profiles": self.cluster_profiles,
            "cm_data": cm_data,
            "roc_data": roc_data,
            "viz_data": viz_data,
            "total_samples": len(df),
            "churn_count": int(df['Churn'].sum()),
            "non_churn_count": int(len(df) - df['Churn'].sum()),
            "anomaly_count": int(np.sum(anomalies == -1)),
            "median_total_charges": float(median_total) if median_total is not None else 1397.47
        }

        with open(os.path.join(self.output_dir, "pipeline_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("[6/6] Pipeline training completed successfully!")
        return metadata


if __name__ == "__main__":
    pipeline = ModelPipeline()
    pipeline.run_training_pipeline()
