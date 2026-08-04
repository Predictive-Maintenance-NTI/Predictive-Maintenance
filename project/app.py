import streamlit as st

from prediction import predict_failure
from llm_utils import get_recommendation

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧")

st.title("🔧 Predictive Maintenance")
st.write(
    "Predicts whether an industrial machine is likely to fail — and which type of "
    "failure to expect — from live sensor readings. Trained on the "
    "Predictive Maintenance Dataset (AI4I 2020)."
)

st.sidebar.header("Machine Readings")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air Temperature [K]", value=300.0, step=0.1)
process_temp = st.sidebar.number_input("Process Temperature [K]", value=310.0, step=0.1)
rot_speed = st.sidebar.number_input("Rotational Speed [rpm]", value=1500, step=10)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0, step=0.1)
tool_wear = st.sidebar.number_input("Tool Wear [min]", value=100, step=1)

if st.sidebar.button("Predict"):
    prediction, probability = predict_failure(
        air_temp, process_temp, rot_speed, torque, tool_wear, machine_type
    )

    st.subheader("Prediction Result")

    if prediction == "No Failure":
        st.success(f"No failure expected (confidence: {probability:.1%})")
    else:
        st.error(f"Predicted failure type: **{prediction}** (confidence: {probability:.1%})")

    with st.spinner("Getting maintenance recommendation..."):
        recommendation = get_recommendation(
            prediction,
            probability,
            {
                "machine_type": machine_type,
                "air_temperature": air_temp,
                "process_temperature": process_temp,
                "rotational_speed": rot_speed,
                "torque": torque,
                "tool_wear": tool_wear,
            },
        )

    st.subheader("Recommendation")
    st.write(recommendation)
else:
    st.info("Enter machine readings in the sidebar and click **Predict**.")
