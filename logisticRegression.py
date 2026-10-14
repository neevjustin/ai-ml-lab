import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. PAGE CONFIG & HEADER ---
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Predictor")
st.write(
    "Predict whether a telecommunications customer is at risk of cancelling their service "
    "using a **Logistic Regression** model trained on subscriber usage and contract data."
)

# --- 2. DATA LOADING & PREPROCESSING (CACHED) ---
@st.cache_resource
def load_and_train_model():
    # Load raw dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Clean 'TotalCharges': convert invalid strings to NaN and impute with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_total = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total)
    
    # Drop non-predictive Customer ID column
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Select key numerical and categorical features
    features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'PaperlessBilling', 'SeniorCitizen', 'Contract']
    target = 'Churn'
    
    df_model = df[features + [target]].copy()
    
    # Binary Encoding
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
    df_model['PaperlessBilling'] = df_model['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
    # One-Hot Encoding for Contract types
    df_model = pd.get_dummies(df_model, columns=['Contract'], drop_first=True)
    
    # Safeguard: ensure no missing values remain
    df_model = df_model.dropna()
    
    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, X.columns.tolist(), accuracy, df

try:
    model, scaler, feature_columns, test_accuracy, raw_df = load_and_train_model()
    st.sidebar.success(f"Model Accuracy: {test_accuracy * 100:.1f}%")
except FileNotFoundError:
    st.error("⚠️ Dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` not found in project directory.")
    st.stop()

# --- 3. SIDEBAR USER INPUTS ---
st.sidebar.header("Subscriber Profile Inputs")

tenure = st.sidebar.slider("Account Tenure (Months)", 0, 72, 12, help="How long the customer has stayed")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
total_charges_estimated = float(tenure * monthly_charges)
total_charges_input = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=total_charges_estimated)

contract_type = st.sidebar.selectbox(
    "Contract Type", 
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.radio("Paperless Billing?", ["Yes", "No"])
senior = st.sidebar.radio("Senior Citizen?", ["No", "Yes"])

# --- 4. MAIN DASHBOARD DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Tenure", f"{tenure} Months")
col2.metric("Monthly Billing", f"${monthly_charges:.2f}")
col3.metric("Contract Type", contract_type)

st.divider()

# --- 5. PREDICTION LOGIC ---
if st.button("Analyze Churn Risk", type="primary", use_container_width=True):
    # Construct raw feature dictionary
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_input,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Contract_One year': 1 if contract_type == "One year" else 0,
        'Contract_Two year': 1 if contract_type == "Two year" else 0,
    }
    
    # Convert to DataFrame matching training feature structure
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Scale feature vector
    scaled_input = scaler.transform(input_df)
    
    # Predict label & probability
    prediction = model.predict(scaled_input)[0]
    churn_probability = model.predict_proba(scaled_input)[0][1]
    
    # Render Output Metrics
    st.subheader("Risk Assessment Result")
    
    if prediction == 1:
        st.error("🚨 **HIGH RISK: Customer Likely to Churn**")
    else:
        st.success("✅ **LOW RISK: Customer Likely Retained**")
        
    st.write(f"**Calculated Probability of Churn:** {churn_probability * 100:.2f}%")
    st.progress(float(churn_probability))
    
    # Retention Insights
    with st.expander("💡 Recommended Retention Actions"):
        if churn_probability > 0.5:
            st.write("- **Incentivize Upgrades:** Offer a discount for transitioning from *Month-to-month* to a *1-Year Contract*.")
            st.write("- **Billing Review:** High monthly charges correlate strongly with early departure.")
        else:
            st.write("- **Maintain Service:** Customer shows high account stability.")