import pandas as pd
import numpy as np


class DataProcessor:
    """
    Robust data cleaning and preprocessing pipeline matching the methodology
    from the E-Commerce Customer Segmentation and Churn Prediction study.
    """

    FEATURE_COLUMNS = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'Gender_Male', 'IsSeniorCitizen_Yes',
        'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service',
        'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    CONTINUOUS_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def __init__(self, dataframe=None):
        if dataframe is not None:
            self.df = dataframe.copy()
        else:
            self.df = None

    def rename_columns(self, df=None):
        target_df = self.df if df is None else df.copy()
        rename_map = {col: "CustomerID" for col in target_df.columns if col.strip().lower() == "customerid"}
        rename_map.update({"gender": "Gender", "SeniorCitizen": "IsSeniorCitizen"})
        target_df = target_df.rename(columns=rename_map)
        if df is None:
            self.df = target_df
        return target_df

    def drop_null_rows(self, subset=None, how="all"):
        if self.df is None:
            return None, 0
        rows_before = len(self.df)
        self.df = self.df.dropna(how=how, subset=subset).copy()
        return self.df, rows_before - len(self.df)

    def remove_duplicates(self):
        if self.df is None:
            return None, 0
        duplicate_count = int(self.df.duplicated().sum())
        self.df = self.df.drop_duplicates().copy()
        return self.df, duplicate_count

    def clean_total_charges(self, df=None, fill_value=None):
        target_df = self.df if df is None else df.copy()
        if "TotalCharges" in target_df.columns:
            target_df["TotalCharges"] = pd.to_numeric(target_df["TotalCharges"], errors="coerce")
            if fill_value is None:
                median_value = target_df["TotalCharges"].median()
            else:
                median_value = fill_value
            target_df["TotalCharges"] = target_df["TotalCharges"].fillna(median_value)
            if df is None:
                self.df = target_df
            return target_df, median_value
        return target_df, None

    def convert_data_types(self, df=None):
        target_df = self.df if df is None else df.copy()
        if "IsSeniorCitizen" in target_df.columns:
            # Map numeric 1/0 or boolean to Yes/No
            target_df["IsSeniorCitizen"] = target_df["IsSeniorCitizen"].map(
                lambda v: "Yes" if str(v).strip().lower() in ["1", "yes", "true"] else "No"
            )
        if df is None:
            self.df = target_df
        return target_df

    @staticmethod
    def cap_iqr_outliers(train_df, test_df=None, columns=None, bounds=None):
        """
        Calculate IQR limits on training data only (or use provided bounds)
        and clip features to prevent extreme distortion.
        """
        if columns is None:
            columns = DataProcessor.CONTINUOUS_COLS
        train_df = train_df.copy()
        learned_bounds = {} if bounds is None else bounds.copy()

        for col in columns:
            if col not in train_df.columns:
                continue
            if bounds is None:
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                learned_bounds[col] = (float(lower), float(upper))
            else:
                lower, upper = learned_bounds[col]

            train_df[col] = train_df[col].clip(lower=lower, upper=upper)

        if test_df is not None:
            test_df = test_df.copy()
            for col, (lower, upper) in learned_bounds.items():
                if col in test_df.columns:
                    test_df[col] = test_df[col].clip(lower=lower, upper=upper)
            return train_df, test_df, learned_bounds

        return train_df, learned_bounds

    @staticmethod
    def encode_features(features_df):
        """One-hot encode categorical features for scikit-learn models."""
        return pd.get_dummies(features_df, drop_first=True, dtype=int)

    @classmethod
    def prepare_inference_features(cls, input_dict_or_df, expected_columns=None, outlier_bounds=None):
        """
        Convert a raw user input dict or single-row DataFrame into the exact 30-feature
        vector expected by the trained ML models and scalers.
        """
        if expected_columns is None:
            expected_columns = cls.FEATURE_COLUMNS

        if isinstance(input_dict_or_df, dict):
            raw_df = pd.DataFrame([input_dict_or_df])
        else:
            raw_df = input_dict_or_df.copy()

        # Handle column names
        proc = cls(raw_df)
        raw_df = proc.rename_columns()
        raw_df, _ = proc.clean_total_charges(raw_df, fill_value=0.0)
        raw_df = proc.convert_data_types(raw_df)

        # Drop ID / target if present
        drop_cols = [c for c in raw_df.columns if c.lower() in ['customerid', 'churn', 'target', 'class']]
        raw_df = raw_df.drop(columns=drop_cols, errors='ignore')

        # One hot encode
        encoded_df = pd.get_dummies(raw_df, drop_first=True, dtype=int)

        # Align with expected columns
        aligned_df = pd.DataFrame(0, index=raw_df.index, columns=expected_columns)
        for col in expected_columns:
            if col in encoded_df.columns:
                aligned_df[col] = encoded_df[col]
            else:
                aligned_df[col] = 0

        # Outlier clipping if bounds provided
        if outlier_bounds:
            for col, (lower, upper) in outlier_bounds.items():
                if col in aligned_df.columns:
                    aligned_df[col] = aligned_df[col].clip(lower=lower, upper=upper)

        return aligned_df
