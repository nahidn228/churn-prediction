---
title: Customer Churn & Segmentation Intelligence
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---



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

- **⚡ Dynamic Dataset Ingestion & AutoML Studio**: Upload **any custom tabular database (CSV, Excel, TSV)** or use the built-in sample generator. The system auto-detects targets, cleans missing data, trains 6 classifiers, and generates full visual reports.
- **🎻 Multi-Dimensional Violin & Distribution Plots**: Interactive violin density plots with embedded box plots to visualize multimodal feature distributions across target classes and K-Means clusters.
- **📈 Comprehensive Precision, Recall, F1 & Benchmark Leaderboard**:
  - Full evaluation leaderboard comparing Accuracy, Precision, Recall, F1 Score, and ROC-AUC across 6 classifiers.
  - Interactive ROC curves with AUC scores.
  - Interactive confusion matrix heatmaps.
  - Gini Feature Importance rankings highlighting top drivers.
- **👥 Dynamic K-Means Clustering with 2D & 3D PCA**:
  - Unsupervised behavioral clustering with customizable K (2 to 8 clusters).
  - Interactive 3D and 2D PCA dimensional scatter projections with zoom, rotate, and hover tooltips.
- **🕵️ Anomaly Account Detection (Isolation Forest)**:
  - Automated detection of outlier accounts with unusual feature combinations (customizable contamination threshold).
- **🎯 Dual Intelligence Modes**:
  - **Dynamic Studio**: Upload and analyze any custom dataset.
  - **Pre-Trained IBM Telco Suite**: Instant access to pre-trained classifiers, high-risk presets, and prescriptive retention playbook.
- **📥 Full Scored Dataset Export**: Download predictions, probabilities, cluster tags, and anomaly flags as CSV.

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
│   ├── dynamic_engine.py            # Dynamic AutoML, clustering & visual analytics engine
│   ├── model_pipeline.py            # Training pipeline & evaluation engine
│   └── utils.py                     # Risk tier evaluator & prescriptive retention logic
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures & sample customer profiles
│   ├── test_data_processor.py       # Data cleaning, encoding & feature prep tests
│   ├── test_dynamic_engine.py       # Dynamic AutoML engine unit tests
│   ├── test_model_pipeline.py       # Model loading, inference & metric validation
│   └── test_utils.py                # Risk tiering & prescriptive playbook tests
├── app.py                           # Streamlit Web Application
├── train.py                         # Standalone ML training entrypoint
├── run_tests.py                     # Automated test runner script
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
git clone https://github.com/nahidn228/churn-prediction.git
cd churn-prediction
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

#### Standard Command:
```bash
streamlit run app.py
# or
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Using Pre-configured Quick-Launch Scripts:
If `streamlit` or `python` is not in your system `PATH`:
- **Git Bash**:
  ```bash
  ./run_app.sh
  ```
- **PowerShell**:
  ```powershell
  .\run_app.ps1
  ```
- **Windows File Explorer / CMD**:
  Double-click `run_app.bat`.

> **Troubleshooting (`streamlit: command not found`):**  
> If your terminal cannot find `streamlit`, run using your Python executable directly (e.g. `/c/Users/BLG/AppData/Local/Python/bin/python.exe -m streamlit run app.py`) or add Python and `Scripts\` to your Windows Environment Variables under `Path`.

### 6. Run Automated Tests
Execute the unit and integration test suite:
```bash
pytest -v
# or
python run_tests.py
```

### 7. Applying Changes to Public Links (Streamlit Cloud & Hugging Face)

When you update code or models and want the changes reflected on your public link:

```bash
# 1. Stage and commit your changes
git add .
git commit -m "Describe your updates"

# 2. Update Streamlit Community Cloud (via GitHub)
git push origin main

# 3. Update Hugging Face Spaces
git push space main
```
> **Note:** Always ensure any newly installed dependencies are added to `requirements.txt` before pushing so the cloud server builds successfully.

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
