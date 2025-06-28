#!/bin/bash

# Start MLflow server in the background
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /mlruns \
  --host 0.0.0.0 \
  --port 5000 &

# Wait a few seconds for MLflow server to start
sleep 5

# Run kedro pipeline
kedro run