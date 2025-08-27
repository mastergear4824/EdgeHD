#!/bin/bash

# Set script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_PID_FILE="$SCRIPT_DIR/backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/frontend.pid"
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"
BACKEND_ERROR_LOG="$SCRIPT_DIR/backend_error.log"
FRONTEND_ERROR_LOG="$SCRIPT_DIR/frontend_error.log"

echo "========================================"
echo "EdgeHD 2.0 - Full-Stack Platform"
echo "Starting Background Services"
echo "========================================"
echo

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to find process by port
find_process_by_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -ti :$port 2>/dev/null | head -n1
    else
        echo ""
    fi
}

# Check if already running
echo "[Pre-check] Checking for existing services..."

if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE" 2>/dev/null)
    if [[ "$BACKEND_PID" =~ ^[0-9]+$ ]] && is_process_running "$BACKEND_PID"; then
        echo "❌ Backend already running (PID: $BACKEND_PID)"
        echo "   To stop: ./stop.sh"
        exit 1
    else
        echo "🧹 Cleaning up stale backend PID file..."
        rm -f "$BACKEND_PID_FILE"
    fi
fi

if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE" 2>/dev/null)
    if [[ "$FRONTEND_PID" =~ ^[0-9]+$ ]] && is_process_running "$FRONTEND_PID"; then
        echo "❌ Frontend already running (PID: $FRONTEND_PID)"
        echo "   To stop: ./stop.sh"
        exit 1
    else
        echo "🧹 Cleaning up stale frontend PID file..."
        rm -f "$FRONTEND_PID_FILE"
    fi
fi

# Check port availability
if command -v lsof &> /dev/null; then
    if lsof -i :8080 2>/dev/null | grep -q LISTEN; then
        echo "❌ Port 8080 is already in use."
        echo "   Another service may be running on this port."
        exit 1
    fi
    
    if lsof -i :3000 2>/dev/null | grep -q LISTEN; then
        echo "❌ Port 3000 is already in use."
        echo "   Another service may be running on this port."
        exit 1
    fi
fi

echo "[1/4] Checking environments..."

# Check Conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found in PATH. Please run ./install.sh first."
    exit 1
fi

# Initialize conda
eval "$(conda shell.bash hook)" 2>/dev/null || {
    CONDA_BASE=$(conda info --base 2>/dev/null) || {
        echo "❌ Cannot determine conda installation path."
        exit 1
    }
    source "$CONDA_BASE/etc/profile.d/conda.sh" || {
        echo "❌ Cannot initialize conda."
        exit 1
    }
}

if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' environment not found. Please run ./install.sh first."
    exit 1
fi

# Check Node.js dependencies
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Please run ./install.sh first."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo "❌ Root dependencies not found. Please run ./install.sh first."
    exit 1
fi

echo "✅ All environments ready."

echo "[2/4] Initializing log files..."
echo "$(date) - Backend server starting..." > "$BACKEND_LOG"
echo "$(date) - Frontend server starting..." > "$FRONTEND_LOG"
> "$BACKEND_ERROR_LOG"
> "$FRONTEND_ERROR_LOG"

echo "[3/4] Starting backend server..."

# Apple Silicon environment variables
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Project-local model storage environment variables
export HF_HOME="$SCRIPT_DIR/backend/models"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/backend/models"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/backend/models"

# Create models directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/backend/models"

# Start backend in background
cd "$BACKEND_DIR"
nohup python app.py > "$BACKEND_LOG" 2> "$BACKEND_ERROR_LOG" &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Save backend PID
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

# Wait for backend initialization
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Verify backend is running and check port
for i in {1..12}; do
    if is_process_running "$BACKEND_PID"; then
        if command -v lsof &> /dev/null && lsof -i :8080 2>/dev/null | grep -q LISTEN; then
            echo "✅ Backend started successfully (PID: $BACKEND_PID)"
            break
        elif [ $i -eq 12 ]; then
            echo "❌ Backend started but port 8080 is not active."
            echo "📄 Backend error log:"
            cat "$BACKEND_ERROR_LOG"
            rm -f "$BACKEND_PID_FILE"
            exit 1
        else
            echo "⏳ Waiting for backend port to become active..."
            sleep 2
        fi
    else
        echo "❌ Backend process died during startup."
        echo "📄 Backend error log:"
        cat "$BACKEND_ERROR_LOG"
        rm -f "$BACKEND_PID_FILE"
        exit 1
    fi
done

echo "[4/4] Starting frontend server..."

# Build and start frontend
cd "$FRONTEND_DIR"
echo "🔨 Building frontend..."
if ! npm run build >> "$FRONTEND_LOG" 2>> "$FRONTEND_ERROR_LOG"; then
    echo "❌ Frontend build failed."
    echo "📄 Frontend error log:"
    cat "$FRONTEND_ERROR_LOG"
    exit 1
fi

echo "🚀 Starting frontend server..."
nohup npm start >> "$FRONTEND_LOG" 2>> "$FRONTEND_ERROR_LOG" &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

# Save frontend PID
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

# Wait for frontend initialization
echo "⏳ Waiting for frontend to initialize..."
sleep 8

# Verify frontend is running and check port
for i in {1..10}; do
    if is_process_running "$FRONTEND_PID"; then
        if command -v lsof &> /dev/null && lsof -i :3000 2>/dev/null | grep -q LISTEN; then
            echo "✅ Frontend started successfully (PID: $FRONTEND_PID)"
            break
        elif [ $i -eq 10 ]; then
            echo "❌ Frontend started but port 3000 is not active."
            echo "📄 Frontend error log:"
            cat "$FRONTEND_ERROR_LOG"
            rm -f "$FRONTEND_PID_FILE"
            exit 1
        else
            echo "⏳ Waiting for frontend port to become active..."
            sleep 3
        fi
    else
        echo "❌ Frontend process died during startup."
        echo "📄 Frontend error log:"
        cat "$FRONTEND_ERROR_LOG"
        rm -f "$FRONTEND_PID_FILE"
        exit 1
    fi
done

# Final verification
echo
echo "🔍 Performing final verification..."
sleep 2

if ! is_process_running "$BACKEND_PID"; then
    echo "❌ Backend process died after startup."
    echo "   Check $BACKEND_ERROR_LOG for details."
    exit 1
fi

if ! is_process_running "$FRONTEND_PID"; then
    echo "❌ Frontend process died after startup."
    echo "   Check $FRONTEND_ERROR_LOG for details."
    exit 1
fi

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