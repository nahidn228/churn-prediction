import pytest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor


class TestDataProcessor:
    """Unit tests for the DataProcessor preprocessing engine."""

    def test_rename_columns(self, sample_raw_dataframe):
        processor = DataProcessor(sample_raw_dataframe)
        renamed_df = processor.rename_columns()

        assert "CustomerID" in renamed_df.columns
        assert "Gender" in renamed_df.columns
        assert "IsSeniorCitizen" in renamed_df.columns
        assert "customerID" not in renamed_df.columns
        assert "SeniorCitizen" not in renamed_df.columns

    def test_clean_total_charges_with_median(self, sample_raw_dataframe):
        processor = DataProcessor(sample_raw_dataframe)
        processor.rename_columns()
        cleaned_df, median_val = processor.clean_total_charges()

        # Row 1 had " ", which should now be imputed with non-null median
        assert cleaned_df["TotalCharges"].isnull().sum() == 0
        assert pd.api.types.is_float_dtype(cleaned_df["TotalCharges"])
        assert median_val is not None
        assert median_val > 0

    def test_clean_total_charges_with_explicit_fill(self):
        df = pd.DataFrame({"TotalCharges": ["10.5", " ", None, "45.0"]})
        processor = DataProcessor(df)
        cleaned_df, _ = processor.clean_total_charges(fill_value=0.0)

        assert cleaned_df["TotalCharges"].iloc[1] == 0.0
        assert cleaned_df["TotalCharges"].iloc[2] == 0.0
        assert cleaned_df["TotalCharges"].iloc[0] == 10.5
        assert cleaned_df["TotalCharges"].iloc[3] == 45.0

    def test_convert_data_types(self):
        df = pd.DataFrame({
            "IsSeniorCitizen": [0, 1, "0", "1", True, False]
        })
        processor = DataProcessor(df)
        converted_df = processor.convert_data_types()

        expected = ["No", "Yes", "No", "Yes", "Yes", "No"]
        assert converted_df["IsSeniorCitizen"].tolist() == expected

    def test_remove_duplicates(self):
        df = pd.DataFrame({
            "tenure": [1, 2, 1],
            "MonthlyCharges": [20.0, 30.0, 20.0]
        })
        processor = DataProcessor(df)
        deduped_df, dup_count = processor.remove_duplicates()

        assert dup_count == 1
        assert len(deduped_df) == 2

    def test_drop_null_rows(self):
        df = pd.DataFrame({
            "A": [1, np.nan, 3, np.nan],
            "B": [10, np.nan, np.nan, 40]
        })
        processor = DataProcessor(df)
        cleaned_df, dropped = processor.drop_null_rows(how="all")

        # Row 1 is completely null
        assert dropped == 1
        assert len(cleaned_df) == 3

    def test_cap_iqr_outliers(self):
        train_df = pd.DataFrame({
            "tenure": [10, 12, 11, 13, 1000],  # 1000 is an extreme outlier
            "MonthlyCharges": [50.0, 52.0, 48.0, 51.0, 50.0]
        })
        test_df = pd.DataFrame({
            "tenure": [12, 2000],
            "MonthlyCharges": [51.0, 500.0]
        })

        train_clipped, test_clipped, bounds = DataProcessor.cap_iqr_outliers(
            train_df, test_df=test_df, columns=["tenure", "MonthlyCharges"]
        )

        assert "tenure" in bounds
        assert "MonthlyCharges" in bounds
        upper_tenure = bounds["tenure"][1]

        # The extreme value in train must be capped at upper bound
        assert train_clipped["tenure"].max() == pytest.approx(upper_tenure)
        # Test extreme value must also be capped at learned train bound
        assert test_clipped["tenure"].max() == pytest.approx(upper_tenure)

    def test_encode_features(self):
        df = pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Numeric": [1, 2, 3]
        })
        encoded = DataProcessor.encode_features(df)

        assert "Contract_One year" in encoded.columns
        assert "Contract_Two year" in encoded.columns
        assert "Contract_Month-to-month" not in encoded.columns  # drop_first=True
        assert "Numeric" in encoded.columns

    def test_prepare_inference_features_from_dict(self, sample_raw_customer_dict):
        features = DataProcessor.prepare_inference_features(sample_raw_customer_dict)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        assert list(features.columns) == DataProcessor.FEATURE_COLUMNS
        assert features.shape[1] == 30

        # Verify values align correctly
        assert features["tenure"].iloc[0] == 12
        assert features["MonthlyCharges"].iloc[0] == 29.85
        assert features["TotalCharges"].iloc[0] == 358.20
        assert features["PaymentMethod_Electronic check"].iloc[0] == 1

    def test_prepare_inference_features_handles_unseen_values(self):
        weird_input = {
            "gender": "Unknown",
            "tenure": 5,
            "MonthlyCharges": 50.0,
            "Contract": "NonExistentContractPlan",
            "InternetService": "Satellite"
        }
        features = DataProcessor.prepare_inference_features(weird_input)

        assert list(features.columns) == DataProcessor.FEATURE_COLUMNS
        assert features["tenure"].iloc[0] == 5
        assert features["MonthlyCharges"].iloc[0] == 50.0
        # Columns that were not present in input should be default 0
        assert features["Contract_One year"].iloc[0] == 0
        assert features["Contract_Two year"].iloc[0] == 0
