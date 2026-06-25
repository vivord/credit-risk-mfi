# Credit Risk Prediction Model

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23FF6F00.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-5C5C5C?style=for-the-badge&logo=xgboost&logoColor=white)

**A machine learning project to predict credit risk (Good/Bad) using the German Credit Dataset.**

This project serves as a foundation for building a real-world Credit Risk Prediction system for **Microfinance institutions** (using internal core banking + CBC data).

---

## 📋 Project Overview

This project demonstrates end-to-end machine learning for credit risk assessment:
- Data exploration and preprocessing
- Model training with imbalance handling
- Model evaluation and interpretability
- Ready to be adapted with real institutional data

**Goal**: Predict whether a loan applicant is likely to default (Bad risk) or repay successfully (Good risk).

---

## 🚀 Features

- Full data preprocessing pipeline (numerical + categorical)
- XGBoost model with class imbalance handling
- Performance metrics (AUC, Precision, Recall, F1)
- Feature importance visualization
- Model saving for deployment
- Easy to extend with new datasets

---

## 📊 Dataset

- **Source**: German Credit Dataset (UCI Repository): https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv
- **Samples**: 1,000
- **Target**: `credit_risk` (1 = Good, 0 = Bad)
- **Features**: 20+ attributes (credit history, loan amount, age, purpose, savings, etc.)

**Note**: This is a public dataset used for learning and portfolio purposes.

---

## 🛠 Technologies Used

- **Python** 3.8+
- **pandas** & **numpy** – Data processing
- **scikit-learn** – Preprocessing & evaluation
- **XGBoost** – Main model
- **matplotlib** & **seaborn** – Visualization
- **joblib** – Model persistence
pip install pandas numpy scikit-learn seaborn matplotlib shap joblib xgboost
---
## 📈 Model Performance (German Credit Data)

| Metric              | Score      |
|---------------------|------------|
| **AUC-ROC**         | **0.7851** |
| Accuracy            | ~76-78%    |
| Recall (Bad Risk)   | Improved   |

*XGBoost model with class imbalance handling*

## 🖥️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/credit-risk-prediction.git
   cd credit-risk-prediction
