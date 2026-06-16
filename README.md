# MediGuard AI: Healthcare Insurance Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-orange.svg)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-blueviolet.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff69b4.svg)

An enterprise-grade, full-stack Machine Learning platform designed to detect fraudulent healthcare providers. This system ingests raw claims data, engineers high-density risk features, and deploys an optimized XGBoost model to flag anomalous billing behavior. 

Crucially, it features an **Explainable AI (XAI)** engine using SHAP, translating complex mathematical probabilities into human-readable business audits.

## 🚀 Key Features

* **State-of-the-Art ML Pipeline:** Utilizes **XGBoost** combined with **SMOTE** (Synthetic Minority Over-sampling Technique) to combat the severe class imbalance inherent in real-world fraud data.
* **Automated Hyperparameter Tuning:** Integrates **Optuna** (Tree-structured Parzen Estimator) to algorithmically hunt for the optimal model configuration, preventing overfitting and maximizing recall.
* **Explainable AI (SHAP) with Value Injection:** The AI doesn't just flag a provider; it mathematically proves *why* using SHAP. A custom Natural Language Lexicon translates SHAP outputs and injects raw patient data (e.g., `Overall billed volume exceeds benchmarks [$1,423,600]`) for human auditors.
* **Interactive Risk Sensitivity Slider:** Business users can dynamically adjust the AI's classification threshold (Precision vs. Recall tradeoff) directly from the UI without retraining the backend model.
* **Macro Analytics Dashboard:** Automatically aggregates batch-audit SHAP values into **Chart.js** visualizations, revealing network-wide fraud drivers and risk distributions.
* **High-Performance Backend:** Built on **FastAPI** for asynchronous inference and **SQLite** for persistent audit logging.

## 📊 Model Performance Metrics

Following a 20-trial Optuna optimization focusing on a shallow tree architecture (`max_depth=3`), the model achieved the following elite metrics on unseen test data:

* **ROC-AUC Score:** 85.94%
* **Fraud Recall:** 86.00% *(Successfully catches nearly 9 out of 10 fraudulent providers)*
* **Overall Accuracy:** 74.70%
* **Business Impact:** By prioritizing Recall over raw Precision, the model minimizes false negatives, preventing millions of dollars in missed fraudulent claims, while human auditors quickly clear the false positives.

## 🛠️ Technical Architecture

### 1. Data Engineering & Machine Learning
* `pandas` & `numpy`: Feature engineering and aggregation (e.g., transforming individual claims into 12 distinct provider-level risk profiles).
* `imbalanced-learn` (SMOTE): Synthetic data generation for balanced training.
* `xgboost`: Primary Gradient Boosting classifier.
* `optuna`: Bayesian hyperparameter optimization.
* `shap`: Game-theoretic model explainability.

### 2. Backend API
* `FastAPI`: High-speed REST API routing.
* `uvicorn`: ASGI web server.
* `sqlite3`: Lightweight relational database for logging historical audits.
* `pickle`: Scaler serialization.

### 3. Frontend Dashboard
* `HTML5 / CSS3 / Vanilla JS`: Zero-dependency, lightweight frontend.
* `Chart.js`: Dynamic, theme-responsive data visualization.
* **UI Features:** Drag-and-drop CSV batch auditing, single-provider manual entry, Dark/Light mode toggle, and live analytics.

## 📂 Project Structure

```text
BA-PROJECT/
│
├── data/
│   ├── raw/                  # Original Kaggle datasets
│   └── processed/            # Engineered provider_features.csv
│
├── backend/
│   ├── saved_models/         # Serialized XGBoost JSON & Scaler PKL
│   ├── app.py                # FastAPI server, SHAP Lexicon, API endpoints
│   ├── data_processing.py    # ETL pipeline and feature aggregation
│   ├── train_model.py        # SMOTE, Optuna training, and model evaluation
│   ├── audit_history.db      # SQLite database
│   └── requirements.txt      # Python dependencies
│
├── frontend/
│   └── index.html            # Main dashboard UI, Chart.js logic, API fetching
│
└── README.md                 # Project documentation