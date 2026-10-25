# 🔮 Customer Segmentation & Churn Prediction Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application and analytical intelligence platform for **E-Commerce & Telecom Customer Churn Prediction, Behavioral Segmentation (K-Means), and Anomaly Detection (Isolation Forest)**.

Built on modular Object-Oriented Programming (OOP) principles and deployed with a modern, glassmorphic **Streamlit** dashboard.

---

## 🌟 Key Platform Features

- **📊 Executive KPI Dashboard**: Real-time revenue at risk, churn rate, retention ratio, customer tenure distributions, and multi-dimensional cross-tabulations.
- **🎯 Real-Time Churn Prediction Engine**:
  - Interactive profile simulator with 1-click presets (*High-Risk*, *Loyal*, *Typical*).
  - Multi-model scoring: Pick between 6 trained classifiers (Logistic Regression, Random Forest, SVM, Decision Tree, KNN, Naive Bayes).
  - Risk gauge with calibrated confidence intervals & color-coded urgency tiers.
  - **Prescriptive Retention Playbook**: Automated, context-aware retention tactics based on specific contract, payment, and service features.
- **📁 Batch CSV Scoring**: Upload customer portfolio datasets and export scored predictions with churn risk probabilities.
- **👥 Unsupervised Customer Segmentation (K-Means)**:
  - 3 distinct behavioral customer clusters with personas: *Core Long-Term Subscribers*, *New / Month-to-Month Budget*, *High-Value At-Risk Streamers*.
  - Interactive 2D & 3D PCA dimensional projections with zoom, rotate, and hover tooltips.
- **🕵️ Anomaly Account Detection (Isolation Forest)**:
  - Automated detection of outlier accounts with unusual billing patterns or atypical service bundles (3% contamination threshold).
- **📈 Model Benchmark & Interpretability**:
  - Full evaluation leaderboard comparing Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
  - Interactive ROC curves for all models.
  - Gini Feature Importance rankings highlighting top churn drivers (tenure, contract type, fiber optic service, payment method).
  - Interactive confusion matrix heatmaps.

---

## 📊 Supervised Machine Learning Benchmark

All models were evaluated using an 80/20 stratified split, continuous IQR outlier capping, and standard feature scaling:

| Classifier | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **80.97%** | **66.12%** | **57.22%** | **0.6138** | **0.8437** |
| **Support Vector Machine (SVM)** | **80.33%** | 67.14% | 50.80% | 0.5732 | 0.8033 |
| **Random Forest Classifier** | **79.26%** | 64.04% | 50.00% | 0.5616 | 0.8214 |
| **Decision Tree Classifier** | **78.69%** | 62.77% | 50.80% | 0.5614 | 0.8140 |
| **K-Nearest Neighbors (KNN)** | **75.57%** | 54.19% | 49.73% | 0.5196 | 0.7677 |
| **Naive Bayes (GaussianNB)** | **64.49%** | 41.87% | **87.70%** | 0.5667 | 0.8079 |

> **Key Finding**: Logistic Regression achieved the highest overall Accuracy (80.97%) and ROC-AUC (0.8437). Naive Bayes delivered the highest Recall (87.70%), making it useful for aggressive risk-flagging where false negatives must be minimized.

---

## 👥 Customer Clusters (K-Means)

| Cluster | Segment Name | Risk Tier | Avg Tenure | Avg Monthly | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Core Long-Term Subscribers** | 🟢 Low | 56.4 mos | $61.20 | High-loyalty customers with multi-year agreements and multiple services. |
| **1** | **New / Month-to-Month Budget** | 🟡 Medium | 13.8 mos | $31.50 | Price-sensitive subscribers with basic phone/DSL service. |
| **2** | **High-Value At-Risk Streamers** | 🔴 High | 18.2 mos | $88.90 | High monthly spend on Fiber optic and streaming; predominantly month-to-month contracts. |

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml                  # Streamlit dark theme & server configuration
├── models/
│   ├── supervised_models.joblib     # Serialized classifiers (compressed)
│   ├── scaler.joblib                # Fitted StandardScaler
│   ├── kmeans.joblib                # Fitted K-Means cluster model
│   ├── isolation_forest.joblib      # Fitted Isolation Forest model
│   ├── pca_2d.joblib & pca_3d.joblib# Dimensionality reduction models
│   └── pipeline_metadata.json       # Benchmark metrics, ROC curves, and cluster profiles
├── src/
│   ├── __init__.py
│   ├── data_processor.py            # OOP DataProcessor (IQR capping, encoding, cleaning)
│   ├── model_pipeline.py            # Training pipeline & evaluation engine
│   └── utils.py                     # Risk tier evaluator & prescriptive retention logic
├── app.py                           # Streamlit Web Application
├── train.py                         # Standalone ML training entrypoint
├── TelcoCustomerChurn.csv           # IBM Telco Customer Churn dataset
├── finalDraft_4_E_Commerce_Customer_Segmentation_and_Churn_Prediction.ipynb # Jupyter Notebook
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-Train Models
All models are pre-trained and compressed in the `models/` directory. To re-train all models from scratch:
```bash
python train.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) with custom CSS styling and responsive metrics.
- **Interactive Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/).
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Logistic Regression, Random Forest, SVM, Decision Trees, KNN, Naive Bayes, K-Means, Isolation Forest, PCA).
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/).
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/) with level-3 compression.

---

## 👤 Author

Developed by **Nahid Hasan**  
- Email: [nahidn228@gmail.com](mailto:nahidn228@gmail.com)
- Project: Master of AI - Customer Segmentation & Churn Prediction
