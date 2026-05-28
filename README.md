# 📉 Telco Customer Churn Prediction

A complete end-to-end Machine Learning project that predicts whether a telecom customer will churn, built with Python, Scikit-learn, XGBoost, SHAP, and deployed using Streamlit.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://your-app-name.streamlit.app)  
*(Replace with your Streamlit Cloud URL after deployment)*

---

## 📸 Screenshots

> Add screenshots of your app here after deployment

---

## 📌 Problem Statement

Telecom companies lose millions of dollars every year due to customer churn. This project builds a machine learning system to:
- Predict which customers are likely to churn
- Explain **why** they might churn using SHAP values
- Help businesses take proactive retention actions

---

## 📊 Dataset

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 20 (demographics, services, billing)
- **Target:** `Churn` (Yes / No)

---

## 🧠 Models Used

| Model | ROC AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.86 | 0.64 | 0.52 | 0.84 |
| Random Forest | 0.85 | 0.65 | 0.56 | 0.78 |
| XGBoost | 0.85 | 0.63 | 0.55 | 0.75 |

> **Best Model:** Logistic Regression (highest ROC AUC = 0.86)

---

## 🔑 Key Findings

1. **Tenure** — New customers churn more than long-term ones
2. **Contract Type** — Month-to-month contracts have highest churn risk
3. **Monthly Charges** — Higher charges = higher churn probability
4. **Internet Service** — Fiber optic users churn more than DSL
5. **Value-Added Services** — Customers without security/support churn more

---

## 🗂️ Project Structure

```
customer_churn_pred/
│
├── app.py                      # Streamlit web app
├── ccp.ipynb                   # Full ML notebook (EDA + Modeling)
├── requirements.txt            # Python dependencies
├── Telco-Customer-Churn.csv    # Dataset
│
└── models/
    ├── logistic_model.pkl      # Saved Logistic Regression pipeline
    ├── rf_model.pkl            # Saved Random Forest pipeline
    └── xgb_model.pkl          # Saved XGBoost pipeline
```

---

## ⚙️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
scikit-learn
xgboost
shap
joblib
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Libraries:** Scikit-learn, XGBoost, SHAP
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

---

## 📈 App Features

### Tab 1 — Prediction
- Input customer details via interactive UI
- Choose between 3 ML models
- Get churn probability with risk level (High / Medium / Low)
- SHAP waterfall chart explaining the prediction

### Tab 2 — Model Insights
- Top 15 feature importances (bar chart)
- SHAP beeswarm summary plot
- SHAP feature importance bar plot
- Model performance comparison table
- Business insights and recommendations

---

## 👨‍💻 Author

**Mayur**  
📧 your-email@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile)  
🐙 [GitHub](https://github.com/YOUR_USERNAME)

---

## ⭐ If you found this useful, please star the repo!
