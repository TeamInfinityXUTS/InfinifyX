#!/bin/bash
# ============================================================
# InfinifyX Inference Server — Deploy / Restart Script
# ============================================================
# Usage (on SageMaker):
#   cd /home/sagemaker-user/InfinifyX
#   bash scripts/deploy.sh
#
# What it does:
#   1. Pulls latest code from main
#   2. Stops any running inference_server process
#   3. Restarts the server in the background on port 8000
# ============================================================

set -e

PROJECT_DIR="/home/sagemaker-user/InfinifyX"
LOG_FILE="${PROJECT_DIR}/server.log"

echo "=== InfinifyX CD Deploy ==="
echo "[1/4] Navigating to project..."
cd "$PROJECT_DIR"

echo "[2/4] Pulling latest code from origin/main..."
git pull origin main

echo "[3/5] Stopping existing server & agent (if any)..."
pkill -f "inference_server" 2>/dev/null || true
pkill -f "clearml-agent" 2>/dev/null || true
sleep 2

echo "[4/5] Starting ClearML Agent in background..."
pip install clearml-agent -q
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=R58U0GS1V7DPMV9POEA3L3E6WHH8EV
export CLEARML_API_SECRET_KEY=XQWoX03bgFcB4eJa6Ux8Kt7zmkaVmVdjg-3xBMa1kFdAdpywfbwfAKb8UzS-Rc2WWXU

AGENT_LOG_FILE="${PROJECT_DIR}/agent.log"
nohup clearml-agent daemon --queue data_engineer > "$AGENT_LOG_FILE" 2>&1 &
AGENT_PID=$!

echo "[5/5] Starting inference server on port 8000..."
nohup python server/inference_server.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

sleep 3
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "=== Deploy SUCCESS ==="
    echo "  Server PID: $SERVER_PID"
    echo "  Agent PID:  $AGENT_PID"
    echo "  Server Log: $LOG_FILE"
    echo "  Agent Log:  $AGENT_LOG_FILE"
    echo "  URL:        http://localhost:8000"
    echo ""
    echo "  To view server logs: tail -f $LOG_FILE"
    echo "  To view agent logs:  tail -f $AGENT_LOG_FILE"
    echo "  To stop server:      kill $SERVER_PID"
    echo "  To stop agent:       kill $AGENT_PID"
else
    echo ""
    echo "=== Deploy FAILED ==="
    echo "  Server did not start. Check logs:"
    echo "  cat $LOG_FILE"
    exit 1
fi
