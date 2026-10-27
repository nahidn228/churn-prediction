import pytest
from src.utils import evaluate_risk_level, generate_retention_strategy


class TestUtils:
    """Unit tests for risk evaluation and prescriptive retention playbook."""

    @pytest.mark.parametrize("prob,expected_tier,expected_color", [
        (0.85, "Critical Risk", "#EF4444"),
        (0.65, "Critical Risk", "#EF4444"),
        (0.64, "Moderate Risk", "#F59E0B"),
        (0.35, "Moderate Risk", "#F59E0B"),
        (0.34, "Low Risk / Loyal", "#10B981"),
        (0.05, "Low Risk / Loyal", "#10B981"),
    ])
    def test_evaluate_risk_level_thresholds(self, prob, expected_tier, expected_color):
        risk = evaluate_risk_level(prob)
        assert risk["tier"] == expected_tier
        assert risk["badge_color"] == expected_color
        assert "action_urgency" in risk
        assert "icon" in risk

    def test_retention_strategy_contract_optimization(self):
        customer = {
            "Contract": "Month-to-month",
            "PaymentMethod": "Credit card (automatic)",
            "tenure": 24,
            "MonthlyCharges": 40.0
        }
        recs = generate_retention_strategy(customer, churn_prob=0.7)
        titles = [r["title"] for r in recs]
        assert "Incentivize 1-Year or 2-Year Contract" in titles

    def test_retention_strategy_electronic_check_friction(self):
        customer = {
            "Contract": "One year",
            "PaymentMethod": "Electronic check",
            "tenure": 20,
            "MonthlyCharges": 50.0
        }
        recs = generate_retention_strategy(customer, churn_prob=0.5)
        titles = [r["title"] for r in recs]
        assert "Migrate to Automated Payment (ACH / Credit Card)" in titles

    def test_retention_strategy_fiber_optic_unbundled(self):
        customer = {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "tenure": 30,
            "MonthlyCharges": 80.0
        }
        recs = generate_retention_strategy(customer, churn_prob=0.4)
        titles = [r["title"] for r in recs]
        assert "Bundle Free Premium Tech Support & Security" in titles

    def test_retention_strategy_early_onboarding_high_spend(self):
        customer = {
            "Contract": "Two year",
            "PaymentMethod": "Bank transfer (automatic)",
            "InternetService": "DSL",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "tenure": 6,  # < 12 months
            "MonthlyCharges": 85.0  # > $70/mo
        }
        recs = generate_retention_strategy(customer, churn_prob=0.3)
        titles = [r["title"] for r in recs]
        assert "VIP Onboarding & Customer Success Check-in" in titles

    def test_retention_strategy_loyal_fallback(self):
        customer = {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "DSL",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "tenure": 60,
            "MonthlyCharges": 45.0
        }
        recs = generate_retention_strategy(customer, churn_prob=0.1)
        titles = [r["title"] for r in recs]
        assert "Cross-Sell & Advocacy Program" in titles

    def test_retention_strategy_generic_high_risk_fallback(self):
        customer = {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "No",
            "TechSupport": "No internet service",
            "OnlineSecurity": "No internet service",
            "tenure": 24,
            "MonthlyCharges": 20.0
        }
        recs = generate_retention_strategy(customer, churn_prob=0.75)
        titles = [r["title"] for r in recs]
        assert "Targeted Loyalty Discount" in titles
