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

echo "[3/4] Stopping existing server (if any)..."
pkill -f "inference_server" 2>/dev/null || true
sleep 2

echo "[4/4] Starting inference server on port 8000..."
nohup python server/inference_server.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

sleep 3
if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "=== Deploy SUCCESS ==="
    echo "  PID:  $SERVER_PID"
    echo "  Log:  $LOG_FILE"
    echo "  URL:  http://localhost:8000"
    echo ""
    echo "  To view logs:  tail -f $LOG_FILE"
    echo "  To stop:       kill $SERVER_PID"
else
    echo ""
    echo "=== Deploy FAILED ==="
    echo "  Server did not start. Check logs:"
    echo "  cat $LOG_FILE"
    exit 1
fi
