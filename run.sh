#!/bin/bash

# Start RunPod base services (JupyterLab/SSH) in background
/start.sh &

# Wait for services to initialize
sleep 2

# Run the vllm-bootstrap control plane
exec uvicorn vllm_bootstrap.api:app --host 0.0.0.0 --port 8000
