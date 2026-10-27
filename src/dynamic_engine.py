"""
Dynamic AutoML, Clustering, and Visual Analytics Engine.
Enables uploading any custom tabular dataset (CSV, XLSX, TSV),
auto-detecting target and feature columns, performing data sanitization,
training supervised classifiers, running K-Means clustering,
Isolation Forest anomaly detection, PCA projections, and computing
precision, recall, f1, roc-auc, and interactive violin plots.
"""
import re
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


class DynamicDataEngine:
    """
    Intelligent dynamic preprocessing, multi-model evaluation,
    unsupervised clustering, and visual analytics for arbitrary tabular datasets.
    """

    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()
        self.clean_df = None
        self.target_col = None
        self.id_cols = []
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.label_mapping = {}
        self.inverse_label_mapping = {}
        self.outlier_bounds = {}
        
        # ML Artifacts
        self.scaler = None
        self.X_encoded = None
        self.y_clean = None
        self.encoded_feature_names = []
        self.models = {}
        self.evaluation_results = []
        self.roc_data = {}
        self.cm_data = {}
        self.feature_importances = []
        
        # Unsupervised Artifacts
        self.kmeans = None
        self.iso_forest = None
        self.pca_2d = None
        self.pca_3d = None
        self.cluster_profiles = {}
        self.scored_df = None
        self.pca_coords_2d = None
        self.pca_coords_3d = None

    def auto_detect_columns(self):
        """
        Inspect dataset columns and identify ID candidates,
        potential target candidates, continuous numeric, and categoricals.
        """
        df = self.raw_df
        total_rows = len(df)
        id_candidates = []
        target_name_matches = []
        binary_candidates = []
        numeric_candidates = []
        categorical_candidates = []

        # Common target regex pattern
        target_patterns = re.compile(
            r'^(churn|target|label|status|default|class|attrition|converted|y|outcome|exited)$',
            re.IGNORECASE
        )
        id_patterns = re.compile(
            r'(id|guid|uuid|key|code|account_num|cust_id|index)',
            re.IGNORECASE
        )

        for col in df.columns:
            col_str = str(col).strip()
            num_unique = df[col].nunique(dropna=True)

            # Check ID candidates
            if id_patterns.search(col_str) and (num_unique > total_rows * 0.7 or num_unique == total_rows):
                id_candidates.append(col)
                continue

            # Check target candidates (prioritize exact name matches)
            if target_patterns.search(col_str):
                target_name_matches.append(col)
            elif num_unique == 2:
                # Binary column could be target
                binary_candidates.append(col)

            # Check numeric vs categorical
            # Try converting to numeric if string
            if pd.api.types.is_numeric_dtype(df[col]):
                if num_unique > 2 or not target_patterns.search(col_str):
                    numeric_candidates.append(col)
            else:
                # Check if it's numeric stored as string (like TotalCharges)
                converted = pd.to_numeric(df[col], errors='coerce')
                non_null_ratio = converted.notnull().sum() / max(1, df[col].notnull().sum())
                if non_null_ratio > 0.8 and num_unique > 10:
                    numeric_candidates.append(col)
                else:
                    if col not in id_candidates and not target_patterns.search(col_str):
                        categorical_candidates.append(col)

        # Name matches are prioritized over generic binary columns
        target_candidates = target_name_matches + [c for c in binary_candidates if c not in target_name_matches]
        suggested_target = target_candidates[0] if target_candidates else (
            df.columns[-1] if len(df.columns) > 1 else None
        )

        return {
            "suggested_target": suggested_target,
            "target_candidates": list(dict.fromkeys(target_candidates)),
            "id_candidates": id_candidates,
            "numeric_candidates": numeric_candidates,
            "categorical_candidates": categorical_candidates
        }

    def sanitize_and_prepare(
        self,
        target_col: str,
        id_cols: list = None,
        drop_cols: list = None,
        positive_value: str = None,
        cap_outliers: bool = True
    ):
        """
        Clean missing values, convert data types, encode categorical features,
        and scale numerical features.
        """
        self.target_col = target_col
        self.id_cols = id_cols or []
        df = self.raw_df.copy()

        # Remove entire duplicate rows
        df = df.drop_duplicates()

        # Drop ID and user-selected drop columns
        columns_to_exclude = set(self.id_cols + (drop_cols or []))
        columns_to_exclude.discard(target_col)

        df = df.drop(columns=[c for c in columns_to_exclude if c in df.columns], errors='ignore')

        # Drop rows where target is missing
        df = df.dropna(subset=[target_col]).copy()

        # Normalize target column to 0 / 1 binary classification
        raw_target = df[target_col].astype(str).str.strip()
        unique_targets = list(raw_target.unique())

        if len(unique_targets) < 2:
            raise ValueError(f"Target column '{target_col}' has fewer than 2 distinct classes.")

        if positive_value is not None and str(positive_value) in unique_targets:
            pos_label = str(positive_value)
            neg_label = [u for u in unique_targets if u != pos_label][0]
        else:
            # Auto-detect positive label (e.g., 'Yes', '1', 'True', 'Churn')
            pos_candidates = [u for u in unique_targets if u.lower() in ['yes', '1', 'true', 'churn', 'default', 'positive']]
            if pos_candidates:
                pos_label = pos_candidates[0]
                neg_label = [u for u in unique_targets if u != pos_label][0]
            else:
                pos_label = unique_targets[1]
                neg_label = unique_targets[0]

        self.label_mapping = {neg_label: 0, pos_label: 1}
        self.inverse_label_mapping = {0: neg_label, 1: pos_label}

        y = raw_target.map(lambda v: 1 if v == pos_label else 0).astype(int)
        self.y_clean = y

        # Feature dataframe
        X = df.drop(columns=[target_col]).copy()

        # Classify columns into numeric and categorical
        self.numeric_cols = []
        self.categorical_cols = []

        for col in X.columns:
            # Attempt numeric conversion
            numeric_series = pd.to_numeric(X[col], errors='coerce')
            valid_ratio = numeric_series.notnull().sum() / max(1, len(X))
            
            if valid_ratio >= 0.75 and numeric_series.nunique() > 2:
                # Impute numeric with median
                median_val = numeric_series.median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = numeric_series.fillna(median_val)
                self.numeric_cols.append(col)
            else:
                # Categorical column: impute missing with mode or 'Missing'
                mode_val = X[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
                X[col] = X[col].fillna(fill_val).astype(str).str.strip()
                # Ignore ultra high cardinality categoricals (> 50 unique values) to prevent explosion
                if X[col].nunique() <= 50:
                    self.categorical_cols.append(col)
                else:
                    X = X.drop(columns=[col])

        # Continuous outlier capping via IQR
        if cap_outliers and self.numeric_cols:
            self.outlier_bounds = {}
            for col in self.numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                self.outlier_bounds[col] = (float(lower), float(upper))
                X[col] = X[col].clip(lower=lower, upper=upper)

        # One-Hot Encode Categorical features
        if self.categorical_cols:
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=int)
        else:
            X_encoded = X.copy()

        # Fill any remaining NaN with 0
        X_encoded = X_encoded.fillna(0)

        # Ensure all columns numeric
        for c in X_encoded.columns:
            X_encoded[c] = pd.to_numeric(X_encoded[c], errors='coerce').fillna(0)

        self.encoded_feature_names = X_encoded.columns.tolist()
        self.X_encoded = X_encoded

        # Cleaned dataframe containing properly typed and imputed continuous features plus target
        clean_df = X.copy()
        clean_df[target_col] = df[target_col].values
        self.clean_df = clean_df

        return {
            "total_samples": len(df),
            "target_counts": {neg_label: int((y == 0).sum()), pos_label: int((y == 1).sum())},
            "positive_label": pos_label,
            "negative_label": neg_label,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "num_encoded_features": len(self.encoded_feature_names)
        }

    def train_and_benchmark_models(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train and evaluate 6 supervised classifiers with Accuracy, Precision, Recall,
        F1 Score, ROC-AUC, Confusion Matrix, and ROC Curve data.
        """
        if self.X_encoded is None or self.y_clean is None:
            raise ValueError("Data must be prepared via sanitize_and_prepare before training.")

        X = self.X_encoded
        y = self.y_clean

        # Check stratification capability
        can_stratify = (y.value_counts().min() >= 2)
        stratify_arg = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        # Standard Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        defined_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
            "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5, random_state=random_state),
            "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, random_state=random_state),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        eval_summary = []
        self.roc_data = {}
        self.cm_data = {}
        self.models = {}

        for name, model in defined_models.items():
            try:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

                acc = float(accuracy_score(y_test, preds))
                prec = float(precision_score(y_test, preds, zero_division=0))
                rec = float(recall_score(y_test, preds, zero_division=0))
                f1 = float(f1_score(y_test, preds, zero_division=0))
                
                roc_auc_val = None
                if probs is not None and len(np.unique(y_test)) > 1:
                    try:
                        roc_auc_val = round(float(roc_auc_score(y_test, probs)), 4)
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        idx_step = max(1, len(fpr) // 40)
                        self.roc_data[name] = {
                            "fpr": [round(float(x), 4) for x in fpr[::idx_step]],
                            "tpr": [round(float(x), 4) for x in tpr[::idx_step]],
                            "auc": roc_auc_val
                        }
                    except Exception:
                        roc_auc_val = None

                # Confusion Matrix
                cm = confusion_matrix(y_test, preds)
                self.cm_data[name] = cm.tolist()
                self.models[name] = model

                eval_summary.append({
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1_Score": round(f1, 4),
                    "ROC_AUC": roc_auc_val
                })
            except Exception as e:
                continue

        eval_df = pd.DataFrame(eval_summary).sort_values(by="F1_Score", ascending=False)
        self.evaluation_results = eval_df.to_dict(orient="records")

        # Feature Importances from Random Forest
        if "Random Forest Classifier" in self.models:
            rf = self.models["Random Forest Classifier"]
            if hasattr(rf, "feature_importances_"):
                imps = [
                    {"feature": col, "importance": round(float(val), 5)}
                    for col, val in zip(self.encoded_feature_names, rf.feature_importances_)
                ]
                self.feature_importances = sorted(imps, key=lambda x: x["importance"], reverse=True)

        return self.evaluation_results

    def compute_clustering_and_anomalies(self, n_clusters: int = 3, contamination: float = 0.03):
        """
        Perform dynamic K-Means clustering, Isolation Forest anomaly detection,
        and PCA 2D & 3D projections on the entire dataset.
        """
        if self.X_encoded is None:
            raise ValueError("Data must be prepared before clustering.")

        X = self.X_encoded
        full_scaled = self.scaler.transform(X) if self.scaler is not None else StandardScaler().fit_transform(X)

        # 1. K-Means
        n_clusters = max(2, min(n_clusters, len(X) - 1))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(full_scaled)

        # 2. Isolation Forest
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = self.iso_forest.fit_predict(full_scaled)  # -1 for anomaly, 1 for normal

        # 3. PCA 2D & 3D Projections
        n_components_2d = min(2, X.shape[1])
        n_components_3d = min(3, X.shape[1])

        self.pca_2d = PCA(n_components=n_components_2d, random_state=42)
        self.pca_coords_2d = self.pca_2d.fit_transform(full_scaled)

        self.pca_3d = PCA(n_components=n_components_3d, random_state=42)
        self.pca_coords_3d = self.pca_3d.fit_transform(full_scaled)

        # 4. Generate Scored DataFrame
        scored = self.clean_df.copy()
        scored["Assigned_Cluster"] = [f"Cluster {c}" for c in cluster_labels]
        scored["Is_Anomaly"] = ["Anomaly (Outlier)" if a == -1 else "Normal" for a in anomaly_labels]
        
        # Best model predictions
        best_model_name = self.evaluation_results[0]["Model"] if self.evaluation_results else list(self.models.keys())[0]
        best_model = self.models.get(best_model_name)
        
        if best_model is not None:
            preds = best_model.predict(full_scaled)
            scored["Predicted_Target"] = [self.inverse_label_mapping.get(p, str(p)) for p in preds]
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(full_scaled)[:, 1]
                scored["Prediction_Probability"] = np.round(probs, 4)

        scored["PCA_Dim1"] = np.round(self.pca_coords_2d[:, 0], 4)
        if self.pca_coords_2d.shape[1] > 1:
            scored["PCA_Dim2"] = np.round(self.pca_coords_2d[:, 1], 4)
        if self.pca_coords_3d.shape[1] > 2:
            scored["PCA_Dim3"] = np.round(self.pca_coords_3d[:, 2], 4)

        self.scored_df = scored

        # 5. Cluster Profiles
        profiles = {}
        for c in range(n_clusters):
            c_mask = (cluster_labels == c)
            sub_raw = scored[c_mask]
            
            # Numeric averages
            num_means = {}
            for ncol in self.numeric_cols[:6]:  # top continuous features
                if ncol in sub_raw.columns:
                    series_num = pd.to_numeric(sub_raw[ncol], errors='coerce')
                    mean_val = series_num.mean()
                    if not pd.isna(mean_val):
                        num_means[ncol] = round(float(mean_val), 2)

            if self.y_clean is not None:
                pos_mask = (self.y_clean[c_mask] == 1)
                target_pos_rate = round(float(pos_mask.mean() * 100), 1) if len(pos_mask) > 0 else 0.0
            else:
                target_pos_rate = 0.0

            profiles[c] = {
                "label": f"Cluster {c}",
                "size": int(c_mask.sum()),
                "pct": round(float(c_mask.sum() / len(X) * 100), 1),
                "target_positive_rate": target_pos_rate,
                "numeric_means": num_means
            }

        self.cluster_profiles = profiles
        return profiles


def create_sample_ecommerce_dataset(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic E-Commerce Customer dataset for immediate testing."""
    np.random.seed(seed)
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    age = np.random.randint(18, 72, size=n_samples)
    tenure_months = np.random.randint(1, 60, size=n_samples)
    annual_income = np.random.uniform(22000, 140000, size=n_samples).round(2)
    monthly_spend = np.random.uniform(20.0, 240.0, size=n_samples).round(2)
    web_visits_per_month = np.random.poisson(lam=12, size=n_samples)
    support_tickets = np.random.poisson(lam=2, size=n_samples)
    discount_usage_pct = np.random.uniform(0, 50, size=n_samples).round(1)

    tier_plans = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    payment_methods = np.random.choice(["Credit Card", "PayPal", "Bank Transfer", "Electronic Check"], size=n_samples, p=[0.45, 0.25, 0.15, 0.15])
    device = np.random.choice(["Mobile App", "Desktop Web", "Tablet"], size=n_samples, p=[0.55, 0.35, 0.10])

    # Probabilistic churn based on tickets, spend, and plan
    churn_prob = (
        0.15
        + (support_tickets > 3) * 0.35
        + (tenure_months < 12) * 0.20
        + (payment_methods == "Electronic Check") * 0.15
        - (annual_income > 80000) * 0.10
    ).clip(0.05, 0.90)
    churn = np.where(np.random.rand(n_samples) < churn_prob, "Yes", "No")

    return pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": age,
        "TenureMonths": tenure_months,
        "AnnualIncome": annual_income,
        "MonthlySpend": monthly_spend,
        "WebVisitsPerMonth": web_visits_per_month,
        "CustomerSupportTickets": support_tickets,
        "DiscountUsagePct": discount_usage_pct,
        "SubscriptionTier": tier_plans,
        "PaymentMethod": payment_methods,
        "PrimaryDevice": device,
        "Churn": churn
    })
