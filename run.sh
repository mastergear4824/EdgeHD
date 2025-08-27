#!/bin/bash

echo "========================================"
echo "EdgeHD 2.0 - Full-Stack Platform"
echo "Starting Backend + Frontend Servers"
echo "========================================"
echo

echo "[1/3] Checking backend environment..."

# Check if conda environment exists
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' conda environment not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

echo "✅ Conda environment 'edgehd' found."

echo "[2/3] Checking frontend environment..."

# Check if Node.js dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "❌ Root dependencies not found."
    echo "   Please run ./install.sh first."
    exit 1
fi

echo "✅ Frontend environment ready."

echo "[3/3] Activating environment and starting servers..."

# Apple Silicon environment variables
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Project-local model storage environment variables
export HF_HOME="$(pwd)/backend/models"
export TRANSFORMERS_CACHE="$(pwd)/backend/models"
export HUGGINGFACE_HUB_CACHE="$(pwd)/backend/models"

echo
echo "🔧 ENVIRONMENT CONFIGURED:"
echo "   • Backend: Python $(python --version 2>&1 | cut -d' ' -f2) + PyTorch (conda: edgehd)"
echo "   • Frontend: Node.js $(node --version)"
echo "   • AI Models: $(pwd)/backend/models"
echo

echo "🚀 STARTING SERVERS:"
echo "   • Backend API: http://localhost:8080"
echo "   • Frontend UI: http://localhost:3000"
echo
echo "Press Ctrl+C to stop both servers."
echo

# Start both servers using concurrently
npm run dev