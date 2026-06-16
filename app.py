import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Disease Prediction Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f9ff;
}
.title-box {
    background: linear-gradient(90deg, #0f9b8e, #38b6ff);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.success-box {
    background-color: #d4edda;
    color: #155724;
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
}
.warning-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>🩺 Disease Prediction Chatbot</h1>
    <p>Upload a CSV file, train a machine learning model, and predict disease risk</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Instructions")
    st.write("""
    1. Upload a disease dataset in CSV format  
    2. Select the target disease/result column  
    3. Click **Train Model**  
    4. Enter patient values  
    5. Click **Predict**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing Values", int(data.isnull().sum().sum()))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Select Target Column")

    target = st.selectbox(
        "Choose the disease/result column",
        data.columns
    )

    if st.button("🚀 Train Model", use_container_width=True):

        X = data.drop(target, axis=1)
        y = data[target]

        # Convert text columns to numbers
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.factorize(X[col])[0]

        if y.dtype == "object":
            y = pd.factorize(y)[0]

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.session_state["model"] = model
        st.session_state["features"] = X.columns
        st.session_state["accuracy"] = accuracy

        st.success("Model trained successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

if "model" in st.session_state:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    st.metric(
        "Model Accuracy",
        f"{st.session_state['accuracy'] * 100:.2f}%"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Patient Information")

    st.write("Enter patient values below:")

    patient = {}

    cols = st.columns(2)

    for i, feature in enumerate(st.session_state["features"]):
        with cols[i % 2]:
            patient[feature] = st.number_input(
                f"Enter {feature}",
                value=0.0
            )

    if st.button("🔮 Predict Disease", use_container_width=True):

        patient_df = pd.DataFrame([patient])

        prediction = st.session_state["model"].predict(patient_df)[0]

        if hasattr(st.session_state["model"], "predict_proba"):
            probability = st.session_state["model"].predict_proba(patient_df).max()
        else:
            probability = None

        if prediction == 1:
            st.markdown("""
            <div class="success-box" style="background-color:#f8d7da;color:#721c24;">
                ⚠️ Prediction: High Disease Risk
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                ✅ Prediction: Low Disease Risk
            </div>
            """, unsafe_allow_html=True)

        if probability is not None:
            st.info(f"Prediction Confidence: {probability * 100:.2f}%")

        st.markdown("""
        <div class="warning-box">
            ⚠️ Disclaimer: This application is for educational purposes only. 
            It is not a medical diagnosis system. Please consult a qualified doctor.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)