import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time

# Set page config
st.set_page_config(
    page_title="Credit Guard AI (Real Data)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    return model, columns

try:
    model, model_columns = load_resources()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run `python train_model.py` first.")
    st.stop()

# --- Custom CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    h1 {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton > button {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        color: white; border: none; padding: 0.75rem 1.5rem;
        border-radius: 12px; font-weight: 600; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 118, 255, 0.23); }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2344/2344131.png", width=60)
    st.header("Applicant Profile")
    
    # Numerical Vars
    age = st.number_input("Age", 18, 100, 30)
    amount = st.number_input("Credit Amount ($)", 250, 20000, 2000)
    duration = st.slider("Duration (Months)", 4, 72, 24)
    installment_rate = st.slider("Installment Rate %", 1, 4, 3)
    residence = st.slider("Residence Duration (Years)", 1, 4, 2)
    credits = st.slider("Existing Credits", 1, 4, 1)
    maintenance = st.slider("People Liable for", 1, 2, 1)

    st.markdown("---")
    st.header("Financial History")
    
    # Categorical Vars - Mapped to nice labels
    check_status = st.selectbox("Checking Account Status", 
                                ["< 0 DM", "0 - 200 DM", "> 200 DM", "No Checking Account"])
    
    credit_hist = st.selectbox("Credit History", 
                               ["No credits taken / All paid", "All credits at this bank paid", "Existing credits paid duly", "Delay in past", "Critical account / Other credits existing"])
    
    savings = st.selectbox("Savings Account / Bonds",
                           ["< 100 DM", "100 - 500 DM", "500 - 1000 DM", "> 1000 DM", "Unknown / No Savings"])
    
    employment = st.selectbox("Present Employment Since",
                              ["Unemployed", "< 1 year", "1 - 4 years", "4 - 7 years", "> 7 years"])
    
    personal = st.selectbox("Personal Status & Sex",
                            ["Male: Divorced/Separated", "Female: Divorced/Separated/Married", "Male: Single", "Male: Married/Widowed", "Female: Single"])
    
    debtors = st.selectbox("Other Debtors / Guarantors", ["None", "Co-Applicant", "Guarantor"])
    
    property_type = st.selectbox("Property", ["Real Estate", "Building Society Savings/Life Insurance", "Car or Other", "Unknown / No Property"])
    
    other_plans = st.selectbox("Other Installment Plans", ["Bank", "Stores", "None"])
    
    housing = st.selectbox("Housing", ["Rent", "Own", "For Free"])
    
    job = st.selectbox("Job", ["Unemployed / Unskilled (Non-resident)", "Unskilled (Resident)", "Skilled Employee / Official", "Management / Self-Employed / Highly Qualified"])
    
    telephone = st.radio("Telephone", ["None", "Yes, Registered under customer name"])
    foreign_worker = st.radio("Foreign Worker", ["Yes", "No"])
    
    purpose = st.selectbox("Purpose", ["New Car", "Used Car", "Furniture/Equipment", "Radio/TV", "Domestic Appliance", "Repairs", "Education", "Vacation", "Retraining", "Business", "Other"])


# --- Processing Input ---
def process_input():
    # create a dictionary with all model columns initialized to 0
    data = {col: 0 for col in model_columns}
    
    # Numerical
    data['Age'] = age
    data['Amount'] = amount
    data['Duration'] = duration
    data['InstallmentRatePercentage'] = installment_rate
    data['ResidenceDuration'] = residence
    data['NumberExistingCredits'] = credits
    data['NumberPeopleMaintenance'] = maintenance
    
    # Binary
    data['Telephone'] = 1 if "Yes" in telephone else 0
    data['ForeignWorker'] = 1 if foreign_worker == "Yes" else 0
    
    # Categorical Mappings
    # CheckingAccountStatus
    if check_status == "< 0 DM": data['CheckingAccountStatus.lt.0'] = 1
    elif check_status == "0 - 200 DM": data['CheckingAccountStatus.0.to.200'] = 1
    elif check_status == "> 200 DM": data['CheckingAccountStatus.gt.200'] = 1
    elif check_status == "No Checking Account": data['CheckingAccountStatus.none'] = 1
    
    # Credit History
    if "No credits" in credit_hist: data['CreditHistory.NoCredit.AllPaid'] = 1
    elif "this bank" in credit_hist: data['CreditHistory.ThisBank.AllPaid'] = 1
    elif "paid duly" in credit_hist: data['CreditHistory.PaidDuly'] = 1
    elif "Delay" in credit_hist: data['CreditHistory.Delay'] = 1
    elif "Critical" in credit_hist: data['CreditHistory.Critical'] = 1
    
    # Savings
    if "< 100" in savings: data['SavingsAccountBonds.lt.100'] = 1
    elif "100 - 500" in savings: data['SavingsAccountBonds.100.to.500'] = 1
    elif "500 - 1000" in savings: data['SavingsAccountBonds.500.to.1000'] = 1
    elif "> 1000" in savings: data['SavingsAccountBonds.gt.1000'] = 1
    elif "Unknown" in savings: data['SavingsAccountBonds.Unknown'] = 1

    # Employment
    if "Unemployed" in employment: data['EmploymentDuration.Unemployed'] = 1
    elif "< 1" in employment: data['EmploymentDuration.lt.1'] = 1
    elif "1 - 4" in employment: data['EmploymentDuration.1.to.4'] = 1
    elif "4 - 7" in employment: data['EmploymentDuration.4.to.7'] = 1
    elif "> 7" in employment: data['EmploymentDuration.gt.7'] = 1
    
    # Personal
    if "Male: Divorced" in personal: data['Personal.Male.Divorced.Seperated'] = 1
    elif "Female: Divorced" in personal: data['Personal.Female.NotSingle'] = 1 # Mapping check: 'Female.NotSingle' usually covers Div/Sep/Mar
    elif "Male: Single" in personal: data['Personal.Male.Single'] = 1
    elif "Male: Married" in personal: data['Personal.Male.Married.Widowed'] = 1
    elif "Female: Single" in personal: data.get('Personal.Female.Single', 0) # Use get incase it was missing
    
    # Debtors
    if "None" in debtors: data['OtherDebtorsGuarantors.None'] = 1
    elif "Co-Applicant" in debtors: data['OtherDebtorsGuarantors.CoApplicant'] = 1
    elif "Guarantor" in debtors: data['OtherDebtorsGuarantors.Guarantor'] = 1
    
    # Property
    if "Real Estate" in property_type: data['Property.RealEstate'] = 1
    elif "Insurance" in property_type: data['Property.Insurance'] = 1
    elif "Car" in property_type: data['Property.CarOther'] = 1
    elif "Unknown" in property_type: data['Property.Unknown'] = 1
    
    # Other Plans
    if "Bank" in other_plans: data['OtherInstallmentPlans.Bank'] = 1
    elif "Stores" in other_plans: data['OtherInstallmentPlans.Stores'] = 1
    elif "None" in other_plans: data['OtherInstallmentPlans.None'] = 1
    
    # Housing
    if "Rent" in housing: data['Housing.Rent'] = 1
    elif "Own" in housing: data['Housing.Own'] = 1
    elif "For Free" in housing: data['Housing.ForFree'] = 1
    
    # Job
    if "Unemployed" in job: data['Job.UnemployedUnskilled'] = 1
    elif "Unskilled (Resident)" in job: data['Job.UnskilledResident'] = 1
    elif "Skilled" in job: data['Job.SkilledEmployee'] = 1
    elif "Management" in job: data['Job.Management.SelfEmp.HighlyQualified'] = 1
    
    # Purpose
    if "New Car" in purpose: data['Purpose.NewCar'] = 1
    elif "Used Car" in purpose: data['Purpose.UsedCar'] = 1
    elif "Furniture" in purpose: data['Purpose.Furniture.Equipment'] = 1
    elif "Radio" in purpose: data['Purpose.Radio.Television'] = 1
    elif "Domestic" in purpose: data['Purpose.DomesticAppliance'] = 1
    elif "Repairs" in purpose: data['Purpose.Repairs'] = 1
    elif "Education" in purpose: data['Purpose.Education'] = 1
    elif "Vacation" in purpose: data['Purpose.Vacation'] = 1
    elif "Retraining" in purpose: data['Purpose.Retraining'] = 1
    elif "Business" in purpose: data['Purpose.Business'] = 1
    elif "Other" in purpose: data['Purpose.Other'] = 1
    
    return pd.DataFrame([data])

# --- Main Page Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Credit Risk Assessment")
    st.markdown("### Random Forest Model")
    st.markdown("""
        This tool uses your **actual trained model** based on the Credit Dataset.
        Adjust the parameters in the sidebar to assess the creditworthiness of an applicant.
    """)
    
    if st.button("Run Risk Assessment"):
        with st.spinner("Calculating Risk Score..."):
            time.sleep(0.5)
            input_df = process_input()
            
            # Predict
            prediction = model.predict(input_df)[0]
            try:
                probs = model.predict_proba(input_df)[0]
                prob_good = probs[1] # Probability of Class 1 (Good)
            except:
                prob_good = 0.5 # Fallback if model doesn't support proba
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            
            # Check interpretation: 1 = Good, 0 = Bad
            if prediction == 1:
                st.markdown(f"<h2 style='color: #10b981;'>🟢 Credit Approved (Good Risk)</h2>", unsafe_allow_html=True)
                st.markdown(f"The model is **{(prob_good * 100):.1f}%** confident this is a Good credit risk.")
                st.progress(prob_good)
                st.balloons()
            else:
                st.markdown(f"<h2 style='color: #ef4444;'>🔴 Credit Rejected (Bad Risk)</h2>", unsafe_allow_html=True)
                st.markdown(f"The model predicts a high risk of default (Confidence: **{((1-prob_good) * 100):.1f}%**).")
                st.progress(prob_good) # Show the Good percentage (will be low)
                st.toast("High Risk Warning", icon="⚠️")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature Importance (Random Forest)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Key Factors Influencing Prediction")
                coefs = pd.DataFrame({
                    'Feature': model_columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False).head(5)
                
                for index, row in coefs.iterrows():
                    val = input_df.iloc[0][row['Feature']]
                    # RF importances are always positive, so we just list the top ones
                    # We highlight them if they are non-zero in the input
                    st.markdown(f"**{row['Feature']}** (Importance: {row['Importance']:.3f})", unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Type**: Random Forest Classifier
    - **Features**: 61 Input Variables
    - **Training Accuracy**: 76%
    """)
    st.info("Input data is automatically converted to the One-Hot Encoded format expected by the model.")

