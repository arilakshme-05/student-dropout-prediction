import os
import warnings
import streamlit as st
import pandas as pd
import joblib

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Page Configuration
st.set_page_config(
    page_title="Student Retention Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Universal Theme-Adaptive CSS (Works seamlessly in both Light and Dark modes)
st.markdown("""
    <style>
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: var(--background-color-secondary) !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.8;
        font-weight: 500;
    }
    
    /* Risk Status Card Styling */
    .status-card-danger {
        background-color: rgba(220, 53, 69, 0.15) !important;
        color: var(--text-color) !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin-top: 15px;
    }
    .status-card-danger h3, .status-card-danger p, .status-card-danger strong {
        color: var(--text-color) !important;
    }
    
    /* Success Status Card Styling */
    .status-card-success {
        background-color: rgba(40, 167, 69, 0.15) !important;
        color: var(--text-color) !important;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-top: 15px;
    }
    .status-card-success h3, .status-card-success p, .status-card-success strong {
        color: var(--text-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Verify model file exists
if not os.path.exists('dropout_model.pkl'):
    st.error("❌ `dropout_model.pkl` not found in current directory! Ensure the downloaded model file is in the same folder as app.py.")
    st.stop()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load('dropout_model.pkl')

model = load_model()

# Sidebar: System Context & Future Scope
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.title("System Overview")
    st.info("**Stage-1 Early Warning System**\n\nOptimized for 1st-year student retention tracking by synthesizing financial standing, academic momentum, and attendance indicators.")
    
    st.divider()
    st.subheader("🚀 Future Scope")
    st.caption("• **Rolling-Window ML Pipeline:** Multi-semester dynamic evaluation tracking Delta GPA and credit velocity across upperclass years.")
    st.caption("• **Automated Intervention Routing:** Direct integration with academic advising portals.")

# Main Header
st.title("🎓 Student Dropout Risk Prediction Dashboard")
st.caption("Real-time Machine Learning Inference Engine powered by the UCI Student Retention Dataset")

st.divider()

# Interactive Form
with st.form(key="prediction_form"):
    st.subheader("1️⃣ Financial & Demographic Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tuition_fees = st.selectbox("Tuition Fees Paid Up to Date", options=[1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
        debtor = st.selectbox("Outstanding Student Debt", options=[0, 1], format_func=lambda x: "No ✅" if x == 0 else "Yes ⚠️")
    
    with col2:
        scholarship = st.selectbox("Scholarship Recipient", options=[1, 0], format_func=lambda x: "Yes 🎓" if x == 1 else "No")
        gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    
    with col3:
        age = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=20)
        displaced = st.selectbox("Displaced / Relocated Student", options=[1, 0], format_func=lambda x: "Yes 📍" if x == 1 else "No")
    
    st.divider()
    st.subheader("2️⃣ Academic Momentum & Engagement")
    
    attendance_rate = st.slider("Overall Engagement / Attendance Rate (%)", min_value=0, max_value=100, value=85, help="Dynamically maps attendance density to semester evaluation ratios.")
    
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**1st Semester Metrics**")
        c1_approved = st.number_input("1st Sem Approved Course Units", min_value=0, max_value=30, value=5)
        c1_grade = st.number_input("1st Sem Grade Average (0–20 Scale)", min_value=0.0, max_value=20.0, value=12.0)
    
    with col5:
        st.markdown("**2nd Semester Metrics**")
        c2_approved = st.number_input("2nd Sem Approved Course Units", min_value=0, max_value=30, value=5)
        c2_grade = st.number_input("2nd Sem Grade Average (0–20 Scale)", min_value=0.0, max_value=20.0, value=12.0)
        
    submit_button = st.form_submit_button(label="⚡ Run Risk Assessment Model", use_container_width=True, type="primary")

# Prediction Execution & Results
if submit_button:
    # Compute dynamic proxy evaluation units from attendance
    c1_evals = max(1, int(6 * (attendance_rate / 100.0)))
    c2_evals = max(1, int(6 * (attendance_rate / 100.0)))

    # Construct complete 36-feature array for Scikit-Learn compatibility
    input_data = pd.DataFrame([[
        1, 1, 1, 9254, 1, 1, 120.0, 1, 1, 1, 5, 5, 125.0,
        displaced, 0, debtor, tuition_fees, gender, scholarship, age, 0,
        0, 6, c1_evals, c1_approved, c1_grade, 0,
        0, 6, c2_evals, c2_approved, c2_grade, 0,
        10.8, 1.4, 1.74
    ]], columns=[
        'Marital Status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
        'Previous qualification', 'Previous qualification (grade)', 'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International',
        'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
        'Unemployment rate', 'Inflation rate', 'GDP'
    ])
    
    prediction = model.predict(input_data)[0]
    
    st.subheader("📊 Diagnostic Summary")
    m1, m2, m3 = st.columns(3)
    
    # Calculate academic pass efficiency percentage
    pass_rate = round(((c1_approved + c2_approved) / 12.0) * 100, 1)
    
    m1.metric("Attendance Engagement", f"{attendance_rate}%", delta="Low Engagement" if attendance_rate < 50 else "Optimal", delta_color="inverse" if attendance_rate < 50 else "normal")
    m2.metric("Annual Credit Pass Rate", f"{pass_rate}%")
    m3.metric("Tuition Clearance", "Cleared" if tuition_fees == 1 else "Delinquent", delta_color="normal" if tuition_fees == 1 else "inverse")
    
    if prediction == 1:
        st.markdown("""
            <div class="status-card-danger">
                <h3>⚠️ High Dropout Risk Detected</h3>
                <p>The machine learning inference engine has identified structural vulnerabilities based on tuition delinquency, attendance drop-off, or academic pass rates.</p>
                <hr>
                <strong>Recommended Action:</strong> Flag student for immediate academic intervention and financial advising outreach.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-card-success">
                <h3>✅ Low Dropout Risk Identified</h3>
                <p>The student demonstrates stable academic momentum and financial compliance consistent with degree completion patterns.</p>
                <hr>
                <strong>Status:</strong> Clear for regular academic tracking.
            </div>
        """, unsafe_allow_html=True)