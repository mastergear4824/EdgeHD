#!/bin/bash
# EdgeHD 2.0 - Full-Stack Run Script

echo "🚀 Starting EdgeHD 2.0 Full-Stack Platform..."

# Check if conda environment exists
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda activate edgehd 2>/dev/null; then
        echo "❌ 'edgehd' environment not found."
        echo "   Please run ./install.sh first."
        exit 1
    fi
fi

# Apple Silicon environment variables
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# Project-local model storage environment variables
export HF_HOME="$(pwd)/backend/models"
export TRANSFORMERS_CACHE="$(pwd)/backend/models"
export HUGGINGFACE_HUB_CACHE="$(pwd)/backend/models"

echo "🔧 Environment configured:"
echo "   • Backend: Python $(python --version 2>&1 | cut -d' ' -f2) + PyTorch"
echo "   • Frontend: Node.js $(node --version)"
echo "   • AI Models: $(pwd)/backend/models"
echo ""

# Run both servers
echo "▶️  Starting both servers..."
npm run dev
