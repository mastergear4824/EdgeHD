#!/bin/bash

BACKEND_PID_FILE="backend.pid"
FRONTEND_PID_FILE="frontend.pid"
BACKEND_LOG="backend.log"
FRONTEND_LOG="frontend.log"
BACKEND_ERROR_LOG="backend_error.log"
FRONTEND_ERROR_LOG="frontend_error.log"

echo "========================================"
echo "EdgeHD 2.0 - Full-Stack Platform"
echo "Starting Background Services"
echo "========================================"
echo

# Check if already running
if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "❌ Backend already running (PID: $BACKEND_PID)"
        echo "   To stop: ./stop.sh"
        exit 1
    else
        echo "🧹 Cleaning up old backend PID file..."
        rm -f "$BACKEND_PID_FILE"
    fi
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "❌ Frontend already running (PID: $FRONTEND_PID)"
        echo "   To stop: ./stop.sh"
        exit 1
    else
        echo "🧹 Cleaning up old frontend PID file..."
        rm -f "$FRONTEND_PID_FILE"
    fi
fi

echo "[1/4] Checking environments..."

# Check Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please run ./install.sh first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' environment not found. Please run ./install.sh first."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please run ./install.sh first."
    exit 1
fi

echo "✅ All environments ready."

echo "[2/4] Initializing log files..."
> "$BACKEND_LOG"
> "$FRONTEND_LOG"
> "$BACKEND_ERROR_LOG"
> "$FRONTEND_ERROR_LOG"

echo "[3/4] Starting backend server..."

# Apple Silicon environment variables
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Project-local model storage environment variables
export HF_HOME="$(pwd)/backend/models"
export TRANSFORMERS_CACHE="$(pwd)/backend/models"
export HUGGINGFACE_HUB_CACHE="$(pwd)/backend/models"

# Start backend in background
cd backend
nohup python app.py > "../$BACKEND_LOG" 2> "../$BACKEND_ERROR_LOG" &
BACKEND_PID=$!
cd ..

# Save backend PID
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

# Wait for backend initialization
sleep 3

# Verify backend is running
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "❌ Failed to start backend server."
    if [ -f "$BACKEND_ERROR_LOG" ]; then
        echo "Backend error log:"
        cat "$BACKEND_ERROR_LOG"
    fi
    rm -f "$BACKEND_PID_FILE"
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"

echo "[4/4] Starting frontend server..."

# Build and start frontend
cd frontend
npm run build > "../$FRONTEND_LOG" 2> "../$FRONTEND_ERROR_LOG"
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed."
    cat "../$FRONTEND_ERROR_LOG"
    exit 1
fi

nohup npm start >> "../$FRONTEND_LOG" 2>> "../$FRONTEND_ERROR_LOG" &
FRONTEND_PID=$!
cd ..

# Save frontend PID
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

# Wait for frontend initialization
sleep 5

# Verify frontend is running
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "❌ Failed to start frontend server."
    if [ -f "$FRONTEND_ERROR_LOG" ]; then
        echo "Frontend error log:"
        cat "$FRONTEND_ERROR_LOG"
    fi
    rm -f "$FRONTEND_PID_FILE"
    exit 1
fi

echo "✅ Frontend started (PID: $FRONTEND_PID)"

# Final verification
sleep 2

echo
echo "🎉 EdgeHD 2.0 Full-Stack Platform started successfully!"
echo
echo "📊 SERVER INFO:"
echo "   • Backend PID: $BACKEND_PID (Python/Flask)"
echo "   • Frontend PID: $FRONTEND_PID (Node.js/Next.js)"
echo "   • Backend Log: $BACKEND_LOG"
echo "   • Frontend Log: $FRONTEND_LOG"
echo
echo "🌐 ACCESS URLs:"
echo "   • Frontend UI: http://localhost:3000"
echo "   • Backend API: http://localhost:8080"
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$LOCAL_IP" ]; then
        echo "   • Network: http://$LOCAL_IP:3000"
    fi
fi
echo
echo "🔧 USEFUL COMMANDS:"
echo "   • Check status: ./status.sh"
echo "   • View backend logs: tail -f $BACKEND_LOG"
echo "   • View frontend logs: tail -f $FRONTEND_LOG"
echo "   • Stop servers: ./stop.sh"
echo
echo "🎨 FEATURES:"
echo "   • Modern React UI with shadcn/ui components"
echo "   • AI-powered image processing (background removal, upscaling)"
echo "   • Real-time progress tracking"
echo "   • Drag & drop file uploads"
echo "   • Video processing capabilities"
echo