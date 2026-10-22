import os
import requests
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

FAILURE_TYPES = [
    ("Tool Wear Failures", "TWF"),
    ("Heat Disspation Failures", "HDF"), 
    ("Power Failures", "PWF"),
    ("Overstrain Failures", "OSF"),
    ("Random Failures", "RNF"),
]


# ----------------------------- helpers -----------------------------
@st.cache_data(ttl=10)
def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health/", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def call_predict(payload: dict):
    """POST the sensor reading; return (ok, data_or_error_message)."""
    try:
        r = requests.post(f"{API_URL}/predict/", json=payload, timeout=100)
    except requests.RequestException as exc:
        return False, f"Could not reach the API: {exc}"

    if r.status_code == 200:
        return True, r.json()

    if r.status_code == 422:  
        try:
            details = r.json().get("detail", [])
            msgs = [f"**{'.'.join(map(str, d['loc'][1:]))}**: {d['msg']}" for d in details]
            return False, "Validation error:\n\n" + "\n\n".join(msgs)
        except Exception:
            pass
    return False, f"API returned {r.status_code}: {r.text}"


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.header("Connection")
    st.caption(f"API: `{API_URL}`")
    if check_health(API_URL):
        st.success("API is healthy")
    else:
        st.error("API unreachable")
        st.caption("Start the FastAPI server, then refresh.")

# ------------------------------- main ------------------------------
st.title("🛠️ Machine Failure Prediction")
st.write(
    "Enter sensor readings. The API predicts whether the machine will fail and, "
    "if so, which failure modes are likely."
)

tab_predict, tab_dashboard = st.tabs(["Predict", "Dashboard"])

# ------------------------------ predict ----------------------------
with tab_predict:
    with st.form("sensor_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_id = st.text_input(
                "Product ID", value="M14860", max_chars=8, help="6–8 characters, e.g. M14860"
            )
            machine_type = st.selectbox("Type", ["L", "M", "H"], index=1)
            air_temp = st.number_input(
                "Air temperature [K]", min_value=288.0, max_value=312.0, value=298.1, step=0.1
            )
            process_temp = st.number_input(
                "Process temperature [K]", min_value=299.0, max_value=322.0, value=308.6, step=0.1
            )

        with col2:
            rpm = st.number_input(
                "Rotational speed [rpm]", min_value=885, max_value=2175, value=1551, step=1
            )
            torque = st.number_input(
                "Torque [Nm]", min_value=2.5, max_value=77.5, value=42.8, step=0.1
            )
            tool_wear = st.number_input(
                "Tool wear [min]", min_value=0, max_value=250, value=0, step=1
            )

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        if len(product_id) < 6:
            st.error("Product ID must be at least 6 characters.")
        else:
            payload = {
                "Product_ID": product_id,
                "Type": machine_type,
                "Air_temperature__K": float(air_temp),
                "Process_temperature__K": float(process_temp),
                "Rotational_speed__rpm": int(rpm),
                "Torque__Nm": float(torque),
                "Tool_wear__min": int(tool_wear),
            }

            with st.spinner("Running prediction..."):
                ok, result = call_predict(payload)

            if not ok:
                st.error(result)
            else:
                st.divider()
                # Branch 1: the API returns MF_Predction only when no failure is predicted
                if "MF_Predction" in result:
                    st.success("✅ No failure predicted — machine looks healthy.")
                # Branch 2: failure predicted, per-type flags are returned
                else:
                    st.error("⚠️ Machine failure predicted")
                    cols = st.columns(len(FAILURE_TYPES))
                    for col, (label, _) in zip(cols, FAILURE_TYPES):
                        flagged = bool(result.get(label, 0))
                        col.metric(label, "Likely" if flagged else "No")

                    explanation = result.get("LLM_Explaination_Failure")

                    if explanation:
                        st.subheader("AI Explanation")
                        if isinstance(explanation, dict):
                            explanation = explanation.get("content") or str(explanation)
                        st.info(explanation)
                    else:
                        st.caption("No explanation available for this prediction.")

                # with st.expander("Raw API response"):
                #     st.json(result)

# ----------------------------- dashboard ---------------------------
with tab_dashboard:
    st.write("Charts rendered by the API's `/all/data` endpoint.")
    if st.button("Load dashboard"):
        try:
            r = requests.get(f"{API_URL}/all/data", timeout=20)
            p = requests.get(f"{API_URL}/all/Predictiondata",timeout=20)
            if r.status_code == 200:
                response_data = r.json()
                response_prediction_data = p.json()

                all_data = response_data.get("data",[])
                all_prediction_data = response_prediction_data.get("predicted_data",[]) 

                toolWear_records = [record['Tool_wear__min'] for record in all_data]
                torque_records = [record['Torque__Nm'] for record in all_data]
                Type_record = [record['Type'] for record in all_data]
                Air_temperature__K = [record['Air_temperature__K'] for record in all_data]
                rotationSpeed = [record['Rotational_speed__rpm'] for record in all_data]
                Machine_predicted = [record['failure_predicted'] for record in all_prediction_data]
                Prediction_Time = [record['predictionTime'] for record in all_prediction_data]
                Predicted_Probability = [record['failure_probability'] for record in all_prediction_data]

                df = pd.DataFrame({
                    "Tool_Wear":toolWear_records,
                    "Torque":torque_records,
                    "Type":Type_record,
                    "AirTemp":Air_temperature__K,
                    "RotationSpeed":rotationSpeed,
                    "Machine_Failure": Machine_predicted,
                    "Predicted_Time" : Prediction_Time,
                    "Predicted_Probability":Predicted_Probability
                })

                color_mapping = {
                     True: "#FF4B4B",  # Red for failure
                     False: "#1C83E1"  # Blue for normal/no failure
                    }
                df["color_code"] = df["Machine_Failure"].map(color_mapping)

                st.scatter_chart(df,x='Tool_Wear',y="RotationSpeed",color="color_code")
                st.line_chart(df,x='Predicted_Time',y='Tool_Wear',x_label="Time",y_label='Toor_Wear[min]',color="color_code")
                st.bar_chart(df,x='Machine_Failure',y='AirTemp',x_label='Machine_Failure',y_label="Air_temperature[K]")
            else:
                st.warning(f"Dashboard returned {r.status_code}: {r.text}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
