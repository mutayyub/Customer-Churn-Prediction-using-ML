import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open("finalized_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:   # 🔹 load your saved encoders
    label_encoders = pickle.load(f)

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn or stay")

# Inputs
Age = st.number_input("Age", 18, 100, 30)
Gender = st.selectbox("Gender", ["Male", "Female"])
Tenure = st.number_input("Tenure (months)", 0, 120, 12)
Usage_Frequency = st.number_input("Usage Frequency", 0, 50, 10)
Support_Calls = st.number_input("Support Calls", 0, 20, 1)
Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
Contract_Length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
Total_Spend = st.number_input("Total Spend ($)", 0, 10000, 500)
Last_Interaction = st.number_input("Last Interaction", 0, 30, 0)

# Encoding categorical
# --- Prepare input as DataFrame (easier for encoders + scaler)
input_dict = {
    "Age": Age,
    "Tenure": Tenure,
    "Usage Frequency": Usage_Frequency,
    "Support Calls": Support_Calls,
    "Total Spend": Total_Spend,
    "Last Interaction": Last_Interaction,
    "Gender": Gender,
    "Subscription Type": Subscription_Type,
    "Contract Length": Contract_Length
}

df_input = pd.DataFrame([input_dict])

# --- Apply LabelEncoders to categorical columns (using saved ones)
for col, le in label_encoders.items():
    df_input[col] = le.transform(df_input[col].astype(str))

# Numeric features
# numeric_features = np.array([[Age, Tenure, Usage_Frequency, Support_Calls, Total_Spend, Last_Interaction]])
# numeric_scaled = scaler.transform(numeric_features)

# Combine scaled numeric + categorical
# sample_input = np.hstack([
#     numeric_scaled,
#     np.array([[gender_map[Gender], contract_map[Contract_Length], subscription_map[Subscription_Type]]])
# ])

# --- Scale numeric features
numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Total Spend", "Last Interaction"]
df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

# --- Convert to numpy for prediction
sample_input = df_input.values

# Predict
if st.button("Predict Churn"):
    pred = model.predict(sample_input)
    prob = model.predict_proba(sample_input)[0]

    if pred[0] == 1:
        st.error(f"⚠️ Likely to Churn (probability: {prob[1]:.2f})")
    else:
        st.success(f"✅ Likely to Stay (probability: {prob[0]:.2f})")
