#!/bin/bash

# Enable error handling
set -e

# Set script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "========================================"
echo "EdgeHD 2.0 - Full-Stack Platform"
echo "Starting Development Servers"
echo "========================================"
echo

# Cleanup function
cleanup() {
    echo
    echo "🛑 Stopping development servers..."
    # Kill any existing processes
    pkill -f "python.*app.py" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    echo "✅ Cleanup completed."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Pre-cleanup existing processes
echo "[Pre-check] Cleaning up existing processes..."
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
sleep 2

echo "[1/4] Checking backend environment..."

# Check if conda exists and can be activated
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found in PATH."
    echo "   Please run ./install.sh first."
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)" 2>/dev/null || {
    # Fallback initialization
    CONDA_BASE=$(conda info --base 2>/dev/null) || {
        echo "❌ Cannot determine conda installation path."
        exit 1
    }
    source "$CONDA_BASE/etc/profile.d/conda.sh" || {
        echo "❌ Cannot initialize conda."
        exit 1
    }
}

# Activate environment
if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' conda environment not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

echo "✅ Conda environment 'edgehd' activated."

echo "[2/4] Checking frontend environment..."

# Check if Node.js dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "❌ Frontend dependencies not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    echo "❌ Root dependencies not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

echo "✅ Frontend environment ready."

echo "[3/4] Checking ports availability..."

# Check if ports are available
if command -v lsof &> /dev/null; then
    if lsof -i :8080 2>/dev/null | grep -q LISTEN; then
        echo "❌ Port 8080 is already in use."
        echo "   Please stop any running services on port 8080."
        exit 1
    fi
    
    if lsof -i :3000 2>/dev/null | grep -q LISTEN; then
        echo "❌ Port 3000 is already in use."
        echo "   Please stop any running services on port 3000."
        exit 1
    fi
else
    echo "⚠️  Cannot check port availability (lsof not found)."
fi

echo "✅ Ports 8080 and 3000 are available."

echo "[4/4] Initializing environment and starting servers..."

# Apple Silicon environment variables
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    echo "🍎 Apple Silicon optimizations enabled."
fi

# Project-local model storage environment variables
export HF_HOME="$SCRIPT_DIR/backend/models"
export TRANSFORMERS_CACHE="$SCRIPT_DIR/backend/models"
export HUGGINGFACE_HUB_CACHE="$SCRIPT_DIR/backend/models"

# Create models directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/backend/models"

echo
echo "🔧 ENVIRONMENT CONFIGURED:"
echo "   • Backend: Python $(python --version 2>&1 | cut -d' ' -f2) + PyTorch (conda: edgehd)"
echo "   • Frontend: Node.js $(node --version)"
echo "   • AI Models: $SCRIPT_DIR/backend/models"
echo "   • Working Directory: $SCRIPT_DIR"
echo

echo "🚀 STARTING SERVERS:"
echo "   • Backend API: http://localhost:8080"
echo "   • Frontend UI: http://localhost:3000"
echo
echo "Press Ctrl+C to stop both servers."
echo

# Verify package.json exists
if [ ! -f "$SCRIPT_DIR/package.json" ]; then
    echo "❌ package.json not found in root directory."
    echo "   Please ensure you're running this from the EdgeHD root directory."
    exit 1
fi

# Start both servers using concurrently
cd "$SCRIPT_DIR"
if ! npm run dev; then
    echo
    echo "❌ Failed to start development servers."
    echo "   Check if all dependencies are installed correctly."
    exit 1
fi

echo
echo "Development servers stopped."