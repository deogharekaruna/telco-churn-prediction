import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap

logistic_model = load("models/logistic_model.pkl")
rf_model = load("models/rf_model.pkl")
xgb_model = load("models/xgb_model.pkl")

preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]
explainer = shap.TreeExplainer(rf_classifier)

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])

with tab1:
    st.title("Customer Churn Prediction App")
    st.write("Enter customer details to predict churn probability")

    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0)
    total = st.number_input("Total Charges", 0.0, 10000.0)

    data = pd.DataFrame({
        "customerID": ["0000-DUMMY"],
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines],
        "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity],
        "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection],
        "TechSupport": [techsupport],
        "StreamingTV": [streamingtv],
        "StreamingMovies": [streamingmovies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if st.button("Predict Churn"):
        try:
            prob = model.predict_proba(data)[0][1]

            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.progress(float(prob))

            if prob > 0.6:
                st.error("🔴 High Risk Customer")
            elif prob > 0.3:
                st.warning("🟡 Medium Risk Customer")
            else:
                st.success("🟢 Low Risk Customer")

            st.write(f"Model Used: **{model_choice}**")

            st.subheader("Prediction Explanation (SHAP)")
            X_transformed = preprocessor.transform(data)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            X_transformed_df = pd.DataFrame(
                X_transformed,
                columns=preprocessor.get_feature_names_out()
            )
            shap_vals = explainer(X_transformed_df)
            fig = plt.figure()
            shap.plots.waterfall(shap_vals[0, :, 1], show=False)
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    feature_names = preprocessor.get_feature_names_out()
    feature_importance = rf_classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    top_features = feat_imp.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Drivers of Customer Churn")
    st.pyplot(fig)

    st.subheader("SHAP Summary Plot")

    sample_data = pd.DataFrame({
        "customerID": ["0000-DUMMY"] * 50,
        "gender": ["Male"] * 50,
        "SeniorCitizen": [0] * 50,
        "Partner": ["Yes"] * 50,
        "Dependents": ["No"] * 50,
        "tenure": np.random.randint(0, 72, size=50),
        "PhoneService": ["Yes"] * 50,
        "MultipleLines": ["No"] * 50,
        "InternetService": ["DSL"] * 50,
        "OnlineSecurity": ["No"] * 50,
        "OnlineBackup": ["No"] * 50,
        "DeviceProtection": ["No"] * 50,
        "TechSupport": ["No"] * 50,
        "StreamingTV": ["No"] * 50,
        "StreamingMovies": ["No"] * 50,
        "Contract": ["Month-to-month"] * 50,
        "PaperlessBilling": ["Yes"] * 50,
        "PaymentMethod": ["Electronic check"] * 50,
        "MonthlyCharges": np.random.uniform(20, 120, size=50),
        "TotalCharges": np.random.uniform(100, 5000, size=50)
    })

    X_sample_transformed = preprocessor.transform(sample_data)
    if hasattr(X_sample_transformed, "toarray"):
        X_sample_transformed = X_sample_transformed.toarray()
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feature_names)

    shap_values = explainer(X_sample_df)

    fig = plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1], max_display=15, show=False)
    st.pyplot(fig)
    plt.close()

    st.subheader("SHAP Feature Importance")
    fig = plt.figure()
    shap.plots.bar(shap_values[:, :, 1], max_display=15, show=False)
    st.pyplot(fig)
    plt.close()

    performance_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC AUC": [0.86, 0.85, 0.85],
        "F1 Score": [0.64, 0.65, 0.63],
        "Precision": [0.52, 0.56, 0.55],
        "Recall": [0.84, 0.78, 0.75]
    })

    st.subheader("Model Performance")
    st.dataframe(performance_df)

    st.subheader("Business Insights")
    st.markdown("""
### Key Drivers of Customer Churn

**1️⃣ Customer Tenure** — New customers have higher churn risk.

**2️⃣ Contract Type** — Month-to-month contracts show highest churn.

**3️⃣ Monthly Charges** — Higher charges correlate with more churn.

**4️⃣ Internet Service** — Fiber optic users churn more than DSL.

**5️⃣ Lack of Value-Added Services** — No security/support = higher churn.

### Business Recommendations
- Encourage long-term contracts through discounts
- Bundle services (security, tech support) to retain customers
- Special offers for high-charge customers
- Focus retention on new low-tenure customers
""")






