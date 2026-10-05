import streamlit as st
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

# Load the saved model, scaler, and config
model = load_model("wcas_cnn_bigru_model.keras")
scaler = joblib.load("wcas_scaler.pkl")
with open("wcas_model_config.json") as f:
    config = json.load(f)

n_weeks = config["n_weeks"]
n_features = config["n_features"]

st.title("WCAS Early-Warning Predictor")
st.write("Enter a student's  weekly activity to forecast their Total Weighted Cumulative Assessment Score.")

name = st.text_input("Student name")
num_weeks = st.slider("Weeks of data available", min_value=1, max_value=n_weeks, value=8)

st.subheader("Weekly data")
clicks_list, contribution_list, lateness_list = [], [], []

for w in range(num_weeks):
    col1, col2, col3 = st.columns(3)
    with col1:
        c = st.number_input(f"Course Activity (Week {w})", min_value=0, value=50, key=f"clicks_{w}")
    with col2:
        p = st.number_input(f"Assessment Score (Week {w})", min_value=0.0, value=0.0, key=f"points_{w}")
    with col3:
        l = st.number_input(f"Late Submission (Week {w})", value=0.0, key=f"late_{w}")
    clicks_list.append(c)
    contribution_list.append(p)
    lateness_list.append(l)

if st.button("Predict WCAS"):
    input_seq = np.zeros((1, n_weeks, n_features))
    cumulative = 0.0
    for w in range(num_weeks):
        cumulative += contribution_list[w]
        distinct_sites_est = max(1, clicks_list[w] // 10)
        input_seq[0, w, :] = [clicks_list[w], distinct_sites_est, lateness_list[w], contribution_list[w], cumulative]

    input_scaled = scaler.transform(input_seq.reshape(-1, n_features)).reshape(1, n_weeks, n_features)
    pred = model.predict(input_scaled, verbose=0)[0][0]
    pred = np.clip(pred, 0, 100)

    if pred < 40:
        risk, color = f"This Studnet, {name} is At Risk", "red"
    elif pred < 60:
        risk, color = f"This Studet, {name} is on a Borderline", "orange"
    else:
        risk, color = f"This Student, {name} is on Track", "green"

    st.subheader(f"Prediction for {name or 'Student'}")
    st.metric("Predicted WCAS", f"{pred:.2f} / 100")
    st.markdown(f"**Status:** :{color}[{risk}]")