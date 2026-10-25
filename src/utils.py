"""
Utility functions for customer churn risk analysis, segmentation personas,
and retention strategies.
"""

def evaluate_risk_level(probability):
    """Classify churn probability into actionable risk tiers."""
    prob_pct = probability * 100
    if prob_pct >= 65:
        return {
            "tier": "Critical Risk",
            "badge_color": "#EF4444",
            "action_urgency": "Immediate Intervention Required",
            "icon": "🚨"
        }
    elif prob_pct >= 35:
        return {
            "tier": "Moderate Risk",
            "badge_color": "#F59E0B",
            "action_urgency": "Proactive Engagement Recommended",
            "icon": "⚠️"
        }
    else:
        return {
            "tier": "Low Risk / Loyal",
            "badge_color": "#10B981",
            "action_urgency": "Relationship Maintenance & Upsell",
            "icon": "✅"
        }


def generate_retention_strategy(customer_data, churn_prob):
    """
    Generate tailored, data-driven retention recommendations based on
    the customer's contract, billing method, tenure, and services.
    """
    recommendations = []

    contract = str(customer_data.get('Contract', '')).lower()
    tenure = float(customer_data.get('tenure', 0))
    monthly_charges = float(customer_data.get('MonthlyCharges', 0))
    payment_method = str(customer_data.get('PaymentMethod', '')).lower()
    internet_service = str(customer_data.get('InternetService', '')).lower()
    tech_support = str(customer_data.get('TechSupport', '')).lower()
    online_security = str(customer_data.get('OnlineSecurity', '')).lower()

    if "month" in contract:
        recommendations.append({
            "category": "Contract Optimization",
            "title": "Incentivize 1-Year or 2-Year Contract",
            "detail": "Customer is currently on a Month-to-Month plan, the single strongest churn indicator. Offer an exclusive 15% discount or billing credit on an annual contract renewal."
        })

    if "electronic" in payment_method:
        recommendations.append({
            "category": "Billing & Friction",
            "title": "Migrate to Automated Payment (ACH / Credit Card)",
            "detail": "Electronic check customers exhibit the highest failure and friction rates. Provide a one-time $10 credit to set up automated card or bank draft."
        })

    if "fiber" in internet_service and (tech_support in ['no', 'none'] or online_security in ['no', 'none']):
        recommendations.append({
            "category": "Value-Add Services",
            "title": "Bundle Free Premium Tech Support & Security",
            "detail": "Fiber optic customers with unbundled tech support have 3x higher churn. Offer 3 months complimentary 24/7 Tech Support and Online Security."
        })

    if tenure < 12 and monthly_charges > 70:
        recommendations.append({
            "category": "Early Onboarding Care",
            "title": "VIP Onboarding & Customer Success Check-in",
            "detail": f"New subscriber ({int(tenure)} months) paying premium rate (${monthly_charges:.2f}/mo). Schedule a dedicated customer success check-in call to ensure satisfaction."
        })

    if not recommendations:
        if churn_prob > 0.5:
            recommendations.append({
                "category": "General Retention",
                "title": "Targeted Loyalty Discount",
                "detail": "Deliver a tailored loyalty promotional bundle to reinforce relationship and reduce price sensitivity."
            })
        else:
            recommendations.append({
                "category": "Loyalty & Expansion",
                "title": "Cross-Sell & Advocacy Program",
                "detail": "Customer has solid loyalty metrics. Invite to VIP advocacy program and introduce complementary streaming or backup solutions."
            })

    return recommendations
