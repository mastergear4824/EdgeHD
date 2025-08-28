#!/bin/bash

# EdgeHD 2.0 - AI Image/Video Processing Platform
# Full-Stack Installation Script (Backend + Frontend)
# Usage: chmod +x install.sh && ./install.sh

echo "========================================"
echo "EdgeHD 2.0 - AI Video/Image Processing Platform"
echo "Professional Timeline Editor + AI Processing"
echo "Full-Stack Installation (Backend + Frontend)"
echo "========================================"
echo

# Miniconda installation function
install_miniconda() {
    echo ""
    echo "Downloading Miniconda3 installer..."
    echo "This may take a few minutes depending on your internet connection."
    
    # Architecture and OS-specific installer determination
    case "$OS" in
        Darwin*)
            if [[ "$ARCH" == "arm64" ]]; then
                INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
                INSTALLER_NAME="Miniconda3-latest-MacOSX-arm64.sh"
            else
                INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
                INSTALLER_NAME="Miniconda3-latest-MacOSX-x86_64.sh"
            fi
            ;;
        Linux*)
            if [[ "$ARCH" == "x86_64" ]]; then
                INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                INSTALLER_NAME="Miniconda3-latest-Linux-x86_64.sh"
            else
                INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                INSTALLER_NAME="Miniconda3-latest-Linux-aarch64.sh"
            fi
            ;;
        *)
            echo "ERROR: Unsupported operating system: $OS"
            return 1
            ;;
    esac
    
    # Download
    if command -v curl &> /dev/null; then
        curl -L -o "/tmp/$INSTALLER_NAME" "$INSTALLER_URL"
    elif command -v wget &> /dev/null; then
        wget -O "/tmp/$INSTALLER_NAME" "$INSTALLER_URL"
    else
        echo "ERROR: Neither curl nor wget found. Please install one of them."
        return 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download Miniconda installer."
        echo "Please check your internet connection and try again."
        return 1
    fi
    
    echo ""
    echo "Installing Miniconda3..."
    echo "This will install to: $HOME/miniconda3"
    echo ""
    
    # Silent installation
    bash "/tmp/$INSTALLER_NAME" -b -p "$HOME/miniconda3"
    if [ $? -ne 0 ]; then
        echo "ERROR: Miniconda installation failed."
        return 1
    fi
    
    # Clean up installer
    rm -f "/tmp/$INSTALLER_NAME"
    
    # PATH setup
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # conda initialization
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1
    "$HOME/miniconda3/bin/conda" init zsh >/dev/null 2>&1
    
    echo ""
    echo "Miniconda3 has been installed successfully!"
    echo "Location: $HOME/miniconda3"
    echo ""
    echo "IMPORTANT: Please close this terminal and open a new one for conda to work properly."
    
    return 0
}

# OS and architecture detection
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "[1/5] Checking Node.js installation..."

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed."
    echo ""
    echo "Please install Node.js first:"
    case "$OS" in
        Darwin*)
            echo "   # macOS"
            echo "   brew install node"
            echo "   # Or download from: https://nodejs.org/"
            ;;
        Linux*)
            echo "   # Ubuntu/Debian"
            echo "   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
            echo "   sudo apt-get install -y nodejs"
            echo ""
            echo "   # CentOS/RHEL/Fedora"
            echo "   curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -"
            echo "   sudo yum install -y nodejs"
            echo ""
            echo "   # Or download from: https://nodejs.org/"
            ;;
    esac
    echo ""
    echo "After installation, restart terminal and run this script again."
    exit 1
else
    NODE_VERSION=$(node --version)
    echo "SUCCESS: Node.js $NODE_VERSION found"
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not available."
    exit 1
else
    NPM_VERSION=$(npm --version)
    echo "SUCCESS: npm $NPM_VERSION found"
fi

echo

echo "[2/5] Checking Python/Conda installation..."

# Check Conda installation
if ! command -v conda &> /dev/null; then
    echo "WARNING: Conda is not installed."
    echo ""
    echo "Would you like to automatically install Miniconda? (Recommended)"
    echo "This will download and install Miniconda3 (~100MB)"
    echo ""
    read -p "Install Miniconda automatically? (Y/n): " choice
    case "$choice" in
        n|N ) 
            echo "Please install Conda manually:"
            case "$OS" in
                Darwin*)
                    echo "   # macOS (Miniforge recommended)"
                    echo "   brew install miniforge"
                    echo "   # Or download directly:"
                    echo "   curl -L -O \"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh\""
                    echo "   bash Miniforge3-\$(uname)-\$(uname -m).sh"
                    ;;
                Linux*)
                    echo "   # Linux"
                    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
                    ;;
            esac
            echo ""
            echo "After installation, restart terminal and run this script again."
            exit 1
            ;;
        * )
            install_miniconda
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to install Miniconda."
                echo "Please install manually from: https://docs.conda.io/en/latest/miniconda.html"
                exit 1
            fi
            echo ""
            echo "Miniconda installation completed!"
            echo "Please run this script again to continue with the setup."
            exit 0
            ;;
    esac
fi

echo "SUCCESS: Conda is installed."

echo
echo "[3/5] Setting up backend environment..."

# Create Conda environment
echo "Creating Conda environment 'edgehd'..."
conda create -n edgehd python=3.11 -y

# Activate environment
echo "Activating Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edgehd

# Install PyTorch (system-specific)
echo "Installing PyTorch 2.1.0 (Real-ESRGAN v0.3.0 compatible)..."
case "$OS" in
    Darwin*)
        if [[ "$ARCH" == "arm64" ]]; then
            echo "Apple Silicon detected - Installing MPS-enabled PyTorch 2.1.0"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
            # Apple Silicon environment variables
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
            echo "SUCCESS: Apple Silicon optimization completed"
        else
            echo "Intel Mac detected - Installing CPU PyTorch 2.1.0"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
    Linux*)
        echo "Linux detected"
        if command -v nvidia-smi &> /dev/null; then
            echo "NVIDIA GPU detected - Installing CUDA PyTorch 2.1.0"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        else
            echo "Installing CPU-only PyTorch 2.1.0"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
    *)
        echo "Installing default CPU PyTorch 2.1.0"
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Install transformers compatible version
echo "Installing transformers 4.35.0 (PyTorch 2.1.0 compatible)..."
pip install transformers==4.35.0

# Install backend dependencies
echo "Installing backend dependencies..."
if [ ! -f "backend/requirements.txt" ]; then
    echo "ERROR: Backend requirements.txt not found."
    echo "Please ensure you're running this from the EdgeHD root directory."
    exit 1
fi

cd backend
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install backend dependencies."
    exit 1
fi
cd ..

echo "SUCCESS: Backend environment setup completed."

echo
echo "[4/5] Setting up frontend environment..."

# Install frontend dependencies
if [ ! -f "frontend/package.json" ]; then
    echo "ERROR: Frontend package.json not found."
    echo "Please ensure you're running this from the EdgeHD root directory."
    exit 1
fi

cd frontend
echo "Installing Node.js packages..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install frontend dependencies."
    exit 1
fi
cd ..

echo "SUCCESS: Frontend environment setup completed."

echo
echo "[5/5] Installing root dependencies..."

# Install root dependencies (concurrently)
echo "Installing concurrently for running both servers..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install root dependencies."
    exit 1
fi

echo "SUCCESS: Root dependencies installed."

# Configure project-local model storage
echo
echo "Configuring AI models..."

# Set environment variables for project-local model storage
echo "INFO: Setting up project-local model storage..."
export HF_HOME="$(pwd)/backend/models"
export TRANSFORMERS_CACHE="$(pwd)/backend/models" 
export HUGGINGFACE_HUB_CACHE="$(pwd)/backend/models"

# Test environment variables
echo "INFO: Project-local model storage configured:"
echo "   HF_HOME=$HF_HOME"
echo "   TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "   HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

echo "   All AI models will be stored in backend/models/ directory"
echo "   AI models downloaded on first run:"
echo "      * BiRefNet background removal model (~424MB)"
echo "      * Real-ESRGAN General v3 4x upscaling model (~17MB)"
echo "   Models are managed independently per project"

echo ""
echo "========================================"
echo "    INSTALLATION COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "EdgeHD 2.0 Full-Stack Platform is ready!"
echo ""
echo "ARCHITECTURE:"
echo "   * Backend:  Flask API server (Python 3.11 + PyTorch 2.1.0)"
echo "   * Frontend: Next.js + shadcn/ui (Node.js + React)"
echo "   * Timeline: Professional video editing interface"
echo "   * Database: File-based storage"
echo "   * AI Models: BiRefNet + Real-ESRGAN"
echo ""
echo "KEY FEATURES:"
echo "   * Professional Timeline Editor with frame-level precision"
echo "   * Draggable playhead for instant navigation"
echo "   * Multi-track video/audio editing"
echo "   * AI-powered background removal and upscaling"
echo "   * Responsive design with dynamic layout"
echo ""
echo "HOW TO RUN:"
echo "   npm run dev          - Start both servers (development)"
echo "   npm run dev:backend  - Start backend only (http://localhost:8080)"
echo "   npm run dev:frontend - Start frontend only (http://localhost:3000)"
echo ""
echo "PRODUCTION:"
echo "   npm run build        - Build frontend for production"
echo "   npm run start        - Start both servers (production)"
echo ""
echo "ACCESS URLS:"
echo "   * Frontend UI: http://localhost:3000"
echo "   * Backend API: http://localhost:8080"
echo ""
echo "See README.md for detailed usage instructions."

# Create run script
echo "Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
# EdgeHD 2.0 - Full-Stack Run Script

echo "🚀 Starting EdgeHD 2.0 Full-Stack Platform..."

# Check if conda environment exists
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' environment not found."
    echo "   Please run ./install.sh first."
    exit 1
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
EOF

chmod +x run.sh

echo "SUCCESS: Run script 'run.sh' created"
echo "   You can now run './run.sh' or 'npm run dev' for easy startup."