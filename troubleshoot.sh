#!/bin/bash

# EdgeHD 2.0 - Troubleshooting Script
# Usage: chmod +x troubleshoot.sh && ./troubleshoot.sh

echo "========================================"
echo "EdgeHD 2.0 - Troubleshooting Script"
echo "========================================"
echo

# Error handling function
handle_error() {
    echo "ERROR: Troubleshooting failed at step: $1"
    echo "Please check the error message above and try again."
    exit 1
}

echo "[1/6] Checking system requirements..."
echo

# Check OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Operating System: $OS"
echo "Architecture: $ARCH"
echo

# Check available disk space
echo "Checking available disk space..."
FREE_SPACE=$(df . | awk 'NR==2 {print $4}')
FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024))
echo "Available disk space: ${FREE_SPACE_GB}GB"
if [ $FREE_SPACE_GB -lt 5 ]; then
    echo "WARNING: Less than 5GB free space. Installation may fail."
else
    echo "SUCCESS: Sufficient disk space available."
fi
echo

# Check Node.js
echo "Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found"
    echo "Please install Node.js from: https://nodejs.org/"
else
    NODE_VERSION=$(node --version)
    echo "SUCCESS: Node.js $NODE_VERSION found"
fi
echo

# Check npm
echo "Checking npm..."
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm not found"
else
    NPM_VERSION=$(npm --version)
    echo "SUCCESS: npm $NPM_VERSION found"
fi
echo

echo "[2/6] Checking Python/Conda installation..."
echo

# Check Python
echo "Checking Python..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
else
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        echo "SUCCESS: $PYTHON_VERSION found"
    else
        PYTHON_VERSION=$(python --version 2>&1)
        echo "SUCCESS: $PYTHON_VERSION found"
    fi
fi
echo

# Check conda
echo "Checking conda..."
CONDA_FOUND=false
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "SUCCESS: $CONDA_VERSION found"
    CONDA_FOUND=true
else
    # Check common conda locations
    if [[ -f "$HOME/miniconda3/bin/conda" ]]; then
        echo "SUCCESS: Conda found at $HOME/miniconda3/bin/conda"
        export PATH="$HOME/miniconda3/bin:$PATH"
        CONDA_FOUND=true
    elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
        echo "SUCCESS: Conda found at $HOME/anaconda3/bin/conda"
        export PATH="$HOME/anaconda3/bin:$PATH"
        CONDA_FOUND=true
    elif [[ -f "/opt/conda/bin/conda" ]]; then
        echo "SUCCESS: Conda found at /opt/conda/bin/conda"
        export PATH="/opt/conda/bin:$PATH"
        CONDA_FOUND=true
    else
        echo "ERROR: conda not found"
        echo
        echo "Common conda locations to check:"
        echo "  $HOME/miniconda3/bin/conda"
        echo "  $HOME/anaconda3/bin/conda"
        echo "  /opt/conda/bin/conda"
        echo
        echo "If conda is installed but not found, try:"
        echo "  1. Restart your terminal"
        echo "  2. Add conda to PATH manually"
        echo "  3. Run: conda init"
    fi
fi
echo

echo "[3/6] Checking project structure..."
echo

# Check if we're in the right directory
if [ ! -f "backend/requirements.txt" ]; then
    echo "ERROR: backend/requirements.txt not found"
    echo "Please run this script from the EdgeHD root directory"
    exit 1
else
    echo "SUCCESS: Backend requirements.txt found"
fi

if [ ! -f "frontend/package.json" ]; then
    echo "ERROR: frontend/package.json not found"
    echo "Please run this script from the EdgeHD root directory"
    exit 1
else
    echo "SUCCESS: Frontend package.json found"
fi

if [ ! -f "package.json" ]; then
    echo "ERROR: Root package.json not found"
    echo "Please run this script from the EdgeHD root directory"
    exit 1
else
    echo "SUCCESS: Root package.json found"
fi
echo

echo "[4/6] Checking network connectivity..."
echo

# Test internet connection
echo "Testing internet connection..."
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "SUCCESS: Internet connection available"
else
    echo "ERROR: No internet connection"
    echo "Please check your network connection"
fi

# Test PyPI
echo "Testing PyPI access..."
if python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org')" &> /dev/null 2>&1; then
    echo "SUCCESS: PyPI access available"
else
    echo "WARNING: Cannot access PyPI"
    echo "This may cause package installation issues"
fi

# Test npm registry
echo "Testing npm registry..."
if npm ping &> /dev/null; then
    echo "SUCCESS: npm registry access available"
else
    echo "WARNING: Cannot access npm registry"
    echo "This may cause package installation issues"
fi
echo

echo "[5/6] Checking existing installations..."
echo

# Check if conda environment exists
if command -v conda &> /dev/null; then
    if conda env list | grep -q "edgehd"; then
        echo "SUCCESS: edgehd conda environment exists"
    else
        echo "INFO: edgehd conda environment not found"
    fi
else
    echo "INFO: conda not available, skipping environment check"
fi

# Check if node_modules exist
if [ -d "frontend/node_modules" ]; then
    echo "SUCCESS: Frontend node_modules found"
else
    echo "INFO: Frontend node_modules not found"
fi

if [ -d "node_modules" ]; then
    echo "SUCCESS: Root node_modules found"
else
    echo "INFO: Root node_modules not found"
fi
echo

echo "[6/6] Common fixes..."
echo

echo "Would you like to try automatic fixes? (y/N)"
read -p "Choice: " choice
case "$choice" in
    y|Y|yes|YES)
        echo
        echo "Applying fixes..."
        
        # Clear npm cache
        echo "Clearing npm cache..."
        npm cache clean --force &> /dev/null
        
        # Clear pip cache
        echo "Clearing pip cache..."
        pip cache purge &> /dev/null 2>&1 || pip cache purge &> /dev/null 2>&1
        
        # Remove existing node_modules
        echo "Removing existing node_modules..."
        rm -rf frontend/node_modules &> /dev/null 2>&1
        rm -rf node_modules &> /dev/null 2>&1
        
        # Remove existing conda environment
        if command -v conda &> /dev/null; then
            echo "Removing existing conda environment..."
            conda env remove -n edgehd -y &> /dev/null 2>&1
        fi
        
        echo
        echo "Fixes applied. Please run ./install.sh again."
        ;;
    *)
        echo "Skipping automatic fixes."
        ;;
esac

echo
echo "========================================"
echo "Troubleshooting completed!"
echo "========================================"
echo
echo "If you're still having issues:"
echo "1. Check the error messages above"
echo "2. Ensure you have sufficient disk space"
echo "3. Try running with sudo (Linux/macOS)"
echo "4. Check your firewall isn't blocking downloads"
echo "5. Try using a different network connection"
echo "6. Check system permissions"
echo
echo "For more help, check the README.md file."
