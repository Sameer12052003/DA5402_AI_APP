
The original datasst is balanced with 50 % 0s and 50% 1s.

Class 
0    284315
1       492
Name: count, dtype: int64
(myenv) 

Length of original dataset = 284807

Dataset : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

1\) Xgboost is faster to train than nueral network. 


# DVC Setup

1\) git init
dvc init

2\) dvc add credit_card_data/credit_card.csv

3\) git add credit_card_data/.gitignore credit_card_data/creditcard.csv.dvc

4\) git commit -m "Add raw data and track with DVC"

# Add DVC remote (Google Drive)

dvc remote add -d myremote gdrive://1mBZCemtgfnxKj1VE0l6p1ZC7Tp9AsI9H 
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path "D:\IITM academics and courses\IITM courses\MLOPS\AI_App\DA5402_AI_APP\my-project-95161-454306-15827cf726d1.json"

(please specify the absolute path of the my-project-95161-454306-15827cf726d1.json file in your local workspace)

dvc push

# Run the mlpipeline for finetuning :

# Start mlflow server
```bash
mlflow ui
```

```bash
dvc repro
```


# Airflow setup

```bash

curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.10.5/docker-compose.yaml' 

mkdir -p ./dags ./logs ./plugins ./config
echo -e "AIRFLOW_UID=$(id -u)" > .env     

docker compose up airflow-init
```


# Prometheus

Metric | What it Means | Why it’s Useful

HTTP request count 
(http_requests_total) | Number of API calls received (per route) | Check if traffic is coming, detect traffic surges

Request duration 
(request_duration_seconds_sum) | Total time taken to process requests | Detect backend slowness

Request duration histogram 
(request_duration_seconds_bucket) | Distribution of request processing times | See if some requests are taking unusually long

Request status codes 
(http_requests_total{status_code="..."}) | How many requests succeeded (200), failed (400, 500) | Detect backend errors quickly

In-progress requests 
(http_requests_in_progress) | Requests currently being processed | Detect overload in real-time