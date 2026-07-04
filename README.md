# Bank Fraud Detection 

# Problem Statement
Credit card fraud detection is a classic **imbalanced classification problem**.
Out of 284,807 real transactions, only 492 (0.17%) were fraudulent.
A naive model predicting "all legit" achieves 99.8% accuracy — but catches zero fraud.
This project solves that real-world challenge.

---

## Dataset
- **Source:** [Credit Card Fraud Detection — Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions × 31 columns
- **Fraud rate:** 0.17% (492 fraud vs 284,315 legit)
- **Features:** V1–V28 (PCA-anonymized), Amount, Time, Class

---

## Key Findings

### 1. Extreme Class Imbalance
- 99.83% legitimate vs 0.17% fraud
- Standard accuracy is **misleading** — Precision & Recall used instead

### 2. Fraud Amount Pattern (Non-Obvious Insight)
| | Fraud | Legit |
|--|-------|-------|
| Median Amount | ₹9.25 | ₹22.00 |
| Max Amount | ₹2,125 | ₹25,691 |

> **Insight:** Fraudsters deliberately use **small amounts** to avoid detection — counterintuitive finding from data

### 3. Model Performance
| Metric | Score |
|--------|-------|
| Precision (Fraud) | 96% |
| Recall (Fraud) | 74% |
| Fraud Caught | 73 / 98 |
| False Alarms (Legit blocked) | 3 |

---

## Approach

### Why not just use accuracy?
Dataset has 99.8% legit transactions — if model predicts everything as "legit", accuracy = 99.8% but zero frauds caught. Hence Recall chosen as primary metric over accuracy.

### Handling Class Imbalance
```python
RandomForestClassifier(class_weight='balanced')
```
`class_weight='balanced'` — treats missing a fraud as more costly during training.

### Feature Importance (Top 5)
| Feature | Importance |
|---------|------------|
| V14 | 22.1% |
| V10 | 11.7% |
| V4  | 11.3% |
| V17 | 8.4%  |
| V12 | 7.8%  |

> V14 alone drives 22% of model decisions

---

## Business Impact
- **73 out of 98** fraud transactions detected in test set
- Only **3 genuine customers** incorrectly blocked (0.005% false positive rate)
- Production deployment could prevent estimated **74% of fraud losses**
- Recommendation: Lower decision threshold to improve Recall from 74% → 85%+

---

## Tech Stack
```
Python 3.x
pandas          — data loading, cleaning, analysis
matplotlib      — visualizations
seaborn         — confusion matrix heatmap
scikit-learn    — Random Forest, train-test split, metrics
```

---

##  Visualizations
1. Class Distribution — imbalance visualization
2. Amount Distribution — fraud vs legit comparison
3. Confusion Matrix — TP/FP/TN/FN breakdown
4. Feature Importance — top 10 fraud predictors

---

## Project Structure
```
bank-fraud-detection/
│
├── main.ipynb   ← main notebook
├── README.md                    ← this file
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── requirements.txt
```
