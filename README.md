# 🩺 Diabetes Prediction Using Machine Learning

> **Research Paper Implementation**  
> Tanya Batra · Vanshika Kohli  
> Dept. of Computer Science & Engineering, Chitkara University, Punjab, India

---

## 📋 Project Overview

This repository contains the full source code for the research paper **"Diabetes Prediction Using Machine Learning"**. The project trains and compares five supervised classifiers on a 100,001-record Kaggle dataset to predict the onset of diabetes, addressing a critical global public health problem.

Diabetes affects over 463 million people worldwide and is projected to reach 629 million by 2045. Early, automated detection can change patient outcomes — this work builds a reproducible, benchmarked ML pipeline toward that goal.

---

## 🏆 Key Results

| Algorithm | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---|---|
| **Random Forest** | **96.80%** | **0.93** | **0.87** | **0.90** |
| Logistic Regression | 95.94% | 0.91 | 0.80 | 0.85 |
| Decision Tree | 95.16% | 0.84 | 0.85 | 0.84 |
| KNN (k=5) | 94.50% | 0.87 | 0.78 | 0.82 |
| Naive Bayes | 93.10% | 0.82 | 0.75 | 0.78 |

> **Random Forest** achieved the best overall performance.  
> **Decision Tree** achieved the highest minority-class recall (0.73) — clinically important for not missing diabetic patients.

---

## 📊 Dataset

- **Source:** [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Records:** 100,001 patient records
- **Target:** `diabetes` (0 = Non-diabetic, 1 = Diabetic)
- **Class Split:** ~91.5% Non-diabetic / ~8.5% Diabetic

### Features

| Feature | Type | Range / Values | Correlation with Target |
|---|---|---|---|
| gender | Binary | 0 = Female, 1 = Male | −0.038 |
| age | Continuous | 0.08 – 80 yrs | 0.26 |
| hypertension | Binary | 0 or 1 | 0.20 |
| heart_disease | Binary | 0 or 1 | 0.17 |
| smoking_history | Binary | 0 or 1 | 0.003 |
| bmi | Continuous | 10.0 – 95.69 | 0.21 |
| HbA1c_level | Continuous | 3.5 – 9.0 | **0.40** |
| blood_glucose_level | Continuous | 80 – 300 mg/dL | **0.42** |

---

## 🗂️ Repository Structure

```
├── diabetes_prediction.py      # Main source code (training + evaluation + plots)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── diabetes_prediction_dataset.csv   # Dataset (download from Kaggle — see below)
└── outputs/                    # Auto-generated after running the script
    ├── eda_overview.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── results_summary.csv
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Place `diabetes_prediction_dataset.csv` in the root of the repository.

---

## 🚀 Running the Code

```bash
python diabetes_prediction.py
```

Or specify a custom dataset path:

```bash
python diabetes_prediction.py /path/to/your/dataset.csv
```

The script will:
1. Load and validate the dataset
2. Run exploratory data analysis (EDA) and save plots
3. Preprocess data (StandardScaler + stratified 70:30 split)
4. Train and evaluate all 5 classifiers
5. Generate comparison plots and ROC curves
6. Save a `results_summary.csv` with all metrics

---

## 🔬 Methodology

### Preprocessing
- **StandardScaler** normalization applied to continuous features (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`)
- **Stratified 70:30 train-test split** to preserve the class ratio

### Models Implemented
| # | Algorithm | Key Parameter |
|---|---|---|
| 1 | Logistic Regression | `max_iter=1000` |
| 2 | Decision Tree | `criterion='gini'`, no depth limit |
| 3 | Random Forest | `n_estimators=100` |
| 4 | Naive Bayes | Gaussian (GaussianNB) |
| 5 | KNN | `k=5`, Euclidean distance |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class + macro average)
- Confusion Matrix
- AUC-ROC

---

## 📈 Generated Outputs

After running the script, the following files are saved in the working directory:

| File | Description |
|---|---|
| `eda_overview.png` | Class distribution + Pearson correlation heatmap |
| `model_comparison.png` | Bar chart: Accuracy / AUC / F1 across all models |
| `roc_curves.png` | Overlaid ROC curves for all 5 classifiers |
| `confusion_matrices.png` | Grid of confusion matrices |
| `feature_importance.png` | Random Forest feature importances |
| `results_summary.csv` | Full metrics table (importable into Excel / LaTeX) |

---

## 💡 Key Findings

- **Blood glucose (r = 0.42)** and **HbA1c (r = 0.40)** are the strongest predictors — consistent with ADA clinical diagnostic criteria.
- **Overall accuracy is a misleading metric** on this imbalanced dataset. LR (95.94%) and DT (95.16%) look close, but DT's diabetic-class recall is 0.73 vs. LR's 0.62 — a 17-point gap that is clinically significant.
- **Random Forest** wins on all aggregate metrics due to ensemble variance reduction.
- **KNN** is computationally impractical for clinical deployment at scale without approximation methods (e.g., KD-Tree, Ball Tree).

---

## 🔮 Future Work

- Apply **SMOTE** to address class imbalance and improve minority-class recall
- Benchmark **XGBoost**, **LightGBM**, and **CatBoost**
- Implement **SHAP** explainability for clinical transparency
- Tune hyperparameters with **GridSearchCV** or **Optuna**
- Evaluate **TabNet** and **MLP with dropout** for deep learning baseline
- Integrate with electronic health records for longitudinal risk monitoring

---

## 📚 References

Key references from the paper:
1. Islam et al. — Random Forest on real-time data (99.35%)
2. Ahamed et al. — LightGBM on PIDD (95.20%)
3. Breiman — Random Forests, *Machine Learning*, 2001
4. American Diabetes Association — Standards of Medical Care in Diabetes, 2021

Full reference list available in the research paper.

---

## 👩‍💻 Authors

| Name | Email | Institution |
|---|---|---|
| Tanya Batra | tanya907.be22@chitkara.edu.in | Chitkara University, Punjab |
| Vanshika Kohli | vanshika948.be22@chitkara.edu.in | Chitkara University, Punjab |

---

## 📄 License

This project is submitted as an academic research work. All code is open for educational and research use.

---

## 🏷️ Type

**Research Paper** | IPR Submission

---

*For questions or collaboration, feel free to open an issue or reach out via email.*
