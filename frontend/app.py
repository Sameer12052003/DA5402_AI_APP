import streamlit as st
import pandas as pd
import requests
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
BACKEND_URL = "http://backend:8000"
output_path = "shared_folder/output.csv"

# Streamlit Settings
st.set_page_config(page_title="🔮 AI-Powered Fraud Detection", layout="wide")
st.title("🔮 AI-Powered Credit Card Fraud Detection")

# --- Session State Setup ---
if "running" not in st.session_state:
    st.session_state["running"] = False

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.header("⚙️ Control Panel")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start"):
            res = requests.post(f"{BACKEND_URL}/start")
            if res.ok:
                st.session_state["running"] = True
                st.success("✅ Predictions Started!")

    with col2:
        if st.button("⏹️ Stop"):
            res = requests.post(f"{BACKEND_URL}/stop")
            if res.ok:
                st.session_state["running"] = False
                st.warning("⛔ Predictions Stopped!")

    # Refresh interval removed per your request
    st.markdown("---")
    status = "🟢 Running" if st.session_state["running"] else "🔴 Stopped"
    st.markdown(f"### Status: {status}")

    # Initialize pie chart with placeholders
    pie_chart_placeholder = st.empty()

    st.markdown("---")
    if os.path.exists(output_path):
        df_sidebar = pd.read_csv(output_path)
        total_txn = len(df_sidebar)
        fraud_txn = (df_sidebar['prediction'] == 1).sum()

        # Update the pie chart with initial data
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            [fraud_txn, total_txn - fraud_txn],
            labels=["Fraud", "Normal"],
            colors=["red", "green"],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax.axis('equal')
        pie_chart_placeholder.pyplot(fig)

# --- Main Section ---
st.markdown("---")
st.subheader("📄 Live Transaction Predictions (Latest 20)")

# --- Placeholders ---
table_placeholder = st.empty()
info_placeholder = st.empty()

# --- Fraud Toast Notification Function ---
def notify_fraudulent_activity(transaction_id):
    st.toast(f"🚨 Fraud Detected! Transaction ID: {transaction_id}", icon="🚨")

# --- Live Update Loop ---
prev_len = 0
last_display_time = time.time() - 60
shown_waiting = False
last_notified_txns = set()

while st.session_state["running"]:
    try:
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)

            if len(df) > prev_len:
                current_time = time.time()
                if current_time - last_display_time >= 60:
                    prev_len = len(df)
                    latest_df = df.tail(20)

                    # Add 'Fraud Alert' column
                    latest_df['Fraud Alert'] = latest_df['prediction'].apply(lambda x: "⚠️ Fraud" if x == 1 else "✅ Normal")

                    # Reorder columns to move 'Fraud Alert' to the leftmost position
                    latest_df = latest_df[['Fraud Alert'] + [col for col in latest_df.columns if col != 'Fraud Alert']]
                    
                    # Show fraud toasts for new frauds
                    for idx, row in latest_df.iterrows():
                        if row['prediction'] == 1 and row.get('transaction_id') not in last_notified_txns:
                            transaction_id = row.get('transaction_id', idx)
                            notify_fraudulent_activity(transaction_id)
                            last_notified_txns.add(transaction_id)

                    # Highlight fraud rows
                    def highlight_fraud(row):
                        color = 'background-color: red; color: white; font-weight: bold;' if row['prediction'] == 1 else ''
                        return [color] * len(row)

                    styled_df = latest_df.style.apply(highlight_fraud, axis=1)
                    table_placeholder.dataframe(styled_df, use_container_width=True)

                    # Update the fraud metrics dynamically
                    total_txn = len(df)
                    fraud_txn = (df['prediction'] == 1).sum()

                    # Update pie chart with new values
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(
                        [fraud_txn, total_txn - fraud_txn],
                        labels=["Fraud", "Normal"],
                        colors=["red", "green"],
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 10}
                    )
                    ax.axis('equal')
                    pie_chart_placeholder.pyplot(fig)

                    info_placeholder.empty()
                    last_display_time = current_time
                    shown_waiting = False
                else:
                    if not shown_waiting:
                        with st.spinner("🔄 Waiting for next batch of predictions..."):
                            time.sleep(2)
                        shown_waiting = True
            else:
                if not shown_waiting:
                    info_placeholder.info("Waiting for new samples...")
                    shown_waiting = True
        else:
            if not shown_waiting:
                with st.spinner("📁 Waiting for output file to be created..."):
                    time.sleep(2)
                shown_waiting = True

    except Exception as e:
        st.error(f"Error: {e}")

    time.sleep(2)

# --- After Stopping: Full Data and Download ---
if os.path.exists(output_path):
    st.subheader("📦 All Predictions Made")

    final_df = pd.read_csv(output_path)
    final_df['Fraud Alert'] = final_df['prediction'].apply(lambda x: "⚠️ Fraud" if x == 1 else "✅ Normal")
    final_df = final_df[['Fraud Alert'] + [col for col in final_df.columns if col != 'Fraud Alert']]

    def highlight_fraud_final(row):
        color = 'background-color: red; color: white; font-weight: bold;' if row['prediction'] == 1 else ''
        return [color] * len(row)

    # Fancy editable table
    st.data_editor(
        final_df,
        column_config={
            "Fraud Alert": st.column_config.TextColumn("Fraud Status", help="Fraud detected or not")
        },
        disabled=["prediction"],
        use_container_width=True,
        hide_index=False
    )

    # Download CSV
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download All Predictions CSV",
        data=csv,
        file_name=f"predictions_{now_str}.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions available to show yet.")
