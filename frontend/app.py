import streamlit as st
import pandas as pd
import requests
import time
import os

# Constants
BACKEND_URL = "http://backend:8000"
output_path = "shared_folder/output.csv"

# Streamlit settings
st.set_page_config(page_title="🔮 AI-Powered Fraud Detection", layout="wide")
st.title("Real-Time Credit Card Fraud Detection")

# --- Session state setup ---
if "running" not in st.session_state:
    st.session_state["running"] = False

# --- Control Buttons ---
st.sidebar.header("⚙️ Control Panel")
if st.sidebar.button("▶️ Start Predictions"):
    res = requests.post(f"{BACKEND_URL}/start")
    if res.ok:
        st.session_state["running"] = True
        st.sidebar.success("✅ Prediction Started.")

if st.sidebar.button("⏹️ Stop Predictions"):
    res = requests.post(f"{BACKEND_URL}/stop")
    if res.ok:
        st.session_state["running"] = False
        st.sidebar.warning("⛔ Prediction Stopped.")

# --- Status Info ---
status_text = "🟢 Running" if st.session_state["running"] else "🔴 Stopped"
st.sidebar.markdown(f"### Status: {status_text}")

st.markdown("---")
st.subheader("📄 Live Transaction Predictions (Last 20 Shown)")

# --- Placeholders for updates ---
table_placeholder = st.empty()
info_placeholder = st.empty()

# --- Live Update Loop ---
prev_len = 0
last_display_time = time.time() - 60
shown_waiting = False

while st.session_state["running"]:
    try:
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)

            if len(df) > prev_len:
                current_time = time.time()
                if current_time - last_display_time >= 60:
                    prev_len = len(df)
                    latest_df = df.tail(20)

                    # Highlight Fraud
                    def highlight_fraud(row):
                        color = 'background-color: red; color: white;' if row['prediction'] == 1 else ''
                        return [color] * len(row)

                    styled_df = latest_df.style.apply(highlight_fraud, axis=1)

                    table_placeholder.dataframe(styled_df, use_container_width=True)
                    info_placeholder.empty()
                    last_display_time = current_time
                    shown_waiting = False
                else:
                    if not shown_waiting:
                        info_placeholder.info("Waiting for next batch of predictions...")
                        shown_waiting = True
            else:
                if not shown_waiting:
                    info_placeholder.info("Waiting for new samples...")
                    shown_waiting = True
        else:
            if not shown_waiting:
                info_placeholder.info("Waiting for output file to be created...")
                shown_waiting = True

    except Exception as e:
        st.error(f"Error: {e}")

    time.sleep(2)

# --- After stopping, show final full table ---
if os.path.exists(output_path):
    final_df = pd.read_csv(output_path)
    st.subheader("📦 All Predictions Made")
    
    def highlight_fraud_final(row):
        color = 'background-color: red; color: white;' if row['prediction'] == 1 else ''
        return [color] * len(row)

    styled_final_df = final_df.style.apply(highlight_fraud_final, axis=1)
    st.dataframe(styled_final_df, use_container_width=True)

    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
