from fastapi import FastAPI
import pandas as pd
import joblib
import time
import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from data_preprocess import preprocess

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Load ML model
logging.info("Loading model...")
loaded = joblib.load("model.pkl")

model = loaded['model']
threshold = loaded['threshold']

logging.info("✅ Model loaded successfully.")

# Scheduler
scheduler = BackgroundScheduler()
scheduler.start()

job_id = "periodic_prediction_job"
is_running = False
used_indices = set()

def prediction_worker():
    global is_running, used_indices

    if not is_running:
        logging.info("Prediction worker triggered but is currently paused.")
        return

    try:
        data_path = "shared_folder/test.csv"
        output_path = "shared_folder/output.csv"

        test_df = pd.read_csv(data_path)
        total_indices = set(test_df.index)
        available_indices = list(total_indices - used_indices)

        if len(available_indices) < 20:
            logging.warning("⚠️ Not enough new rows to sample 20. Stopping predictions.")
            is_running = False
            scheduler.remove_job(job_id)
            return

        available_df = test_df.loc[available_indices]

        # Ensure 2-3 frauds are included
        fraud_df = available_df[available_df['Class'] == 1]
        nonfraud_df = available_df[available_df['Class'] == 0]

        n_frauds = min(3, len(fraud_df))
        n_nonfrauds = 20 - n_frauds

        if len(fraud_df) < 2:
            logging.warning("⚠️ Not enough fraud samples to ensure minimum frauds.")
            sampled_df = available_df.sample(n=20, random_state=int(time.time()))
        else:
            sampled_frauds = fraud_df.sample(n=n_frauds, random_state=int(time.time()))
            sampled_nonfrauds = nonfraud_df.sample(n=n_nonfrauds, random_state=int(time.time()))
            sampled_df = pd.concat([sampled_frauds, sampled_nonfrauds]).sample(frac=1, random_state=int(time.time()))  # shuffle

        used_indices.update(sampled_df.index)

        logging.info(f"🔁 Sampled {n_frauds} frauds and {n_nonfrauds} non-frauds.")

        if not os.path.exists(output_path):
            open(output_path, "w").close()
            logging.info("🆕 Created output.csv")
            
        # Preprocess all at once
        X_batch, _ = preprocess(sampled_df)

        # Predict all at once
        proba_batch = model.predict_proba(X_batch)[:, 1]
        y_preds = (proba_batch > threshold).astype(int)

        # Add predictions
        sampled_df_out = sampled_df.copy()
        sampled_df_out["prediction"] = y_preds

        # Save batch
        header = not os.path.exists(output_path) or os.stat(output_path).st_size == 0
        sampled_df_out.to_csv(output_path, mode="a", index=False, header=header)

        # Log each prediction
        for idx, pred in enumerate(y_preds):
            logging.info(f"✅ Predicted row {idx}: {pred}")

        logging.info(f"✅ Predicted total {len(y_preds)} rows and saved batch.")

        # for index, row_series in sampled_df.iterrows():
        #     if not is_running:
        #         logging.info("🛑 Prediction stopped mid-batch.")
        #         break

        #     row_df = pd.DataFrame([row_series.to_dict()])
        #     X, _ = preprocess(row_df)
            
        #     proba = model.predict_proba(X)[:, 1]
        #     y_pred = (proba > threshold).astype(int)
        #     # y_pred = model.predict(X)

        #     row_out = row_df.copy()
        #     row_out["prediction"] = y_pred
        #     header = not os.path.exists(output_path) or os.stat(output_path).st_size == 0
        #     row_out.to_csv(output_path, mode="a", index=False, header=header)

        #     logging.info(f"✅ Predicted row {index}: {y_pred[0]}")

    except Exception as e:
        logging.exception(f"❌ Error during prediction worker execution: {e}")

@app.post("/start")
def start_prediction():
    global is_running, used_indices

    if is_running:
        logging.info("⚠️ Prediction already running.")
        return {"status": "already running"}

    is_running = True
    used_indices.clear()
    
    # First prediction immediately at t = 0
    logging.info("⚡ Running immediate first prediction at t = 0")
    prediction_worker()
    
    scheduler.add_job(prediction_worker, "interval", seconds=60, id=job_id, misfire_grace_time=10)
    logging.info("🟢 Scheduled prediction every 60 seconds.")
    return {"status": "started"}

@app.post("/stop")
def stop_prediction():
    global is_running
    is_running = False
    try:
        scheduler.remove_job(job_id)
        logging.info("🔴 Stopped prediction job.")
    except Exception:
        logging.warning("⚠️ Tried to stop a non-existent or already stopped job.")
    return {"status": "stopped"}