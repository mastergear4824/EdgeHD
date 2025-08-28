#!/bin/bash

echo "========================================"
echo "EdgeHD 2.0 Full-Stack Platform Removal"
echo "Professional Timeline Editor + AI Processing"
echo "========================================"
echo

# Check current directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the EdgeHD project directory"
    exit 1
fi

# User confirmation
echo "⚠️  The following items will be removed:"
echo
echo "🔧 BACKEND:"
echo "   - Conda environment (edgehd)"
echo "   - Python packages and dependencies"
echo "   - AI model files (backend/models/)"
echo "   - Backend uploads/downloads/temp folders"
echo
echo "🎨 FRONTEND:"
echo "   - Node.js dependencies (frontend/node_modules/)"
echo "   - Frontend build files"
echo "   - Next.js cache"
echo "   - Timeline component assets"
echo
echo "📦 ROOT:"
echo "   - Root dependencies (node_modules/)"
echo "   - Log files and PID files"
echo "   - Environment variable settings"
echo
read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Removal cancelled"
    exit 1
fi

echo
echo "🧹 Starting full-stack environment cleanup..."

# 1. Stop any running servers
echo "[1/8] Stopping running servers..."
if [ -f "backend.pid" ] || [ -f "frontend.pid" ] || [ -f "app.pid" ]; then
    echo "🛑 Stopping servers..."
    ./stop.sh >/dev/null 2>&1 || true
fi

# 2. Remove Conda environment
echo "[2/8] Removing backend environment..."
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda info --envs | grep -q "edgehd"; then
        echo "🐍 Removing Conda environment 'edgehd'..."
        conda remove -n edgehd --all -y >/dev/null 2>&1
        echo "✅ Conda environment removed"
    else
        echo "ℹ️  Conda environment 'edgehd' not found"
    fi
else
    echo "ℹ️  Conda not found, skipping environment removal"
fi

# 3. Remove backend files
echo "[3/8] Cleaning backend files..."
if [ -d "backend/models" ]; then
    echo "🤖 Removing AI model files..."
    rm -rf backend/models/
    echo "✅ Model files removed"
fi

if [ -d "backend/uploads" ]; then
    echo "📁 Cleaning backend upload files..."
    find backend/uploads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
    echo "✅ Backend upload files cleaned"
fi

if [ -d "backend/downloads" ]; then
    echo "📂 Cleaning backend download files..."
    find backend/downloads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
    echo "✅ Backend download files cleaned"
fi

if [ -d "backend/temp" ]; then
    echo "🗂️  Removing backend temporary files..."
    rm -rf backend/temp/
    echo "✅ Backend temporary files removed"
fi

# 4. Remove frontend files
echo "[4/8] Cleaning frontend files..."
if [ -d "frontend/node_modules" ]; then
    echo "📦 Removing frontend dependencies..."
    rm -rf frontend/node_modules/
    echo "✅ Frontend dependencies removed"
fi

if [ -d "frontend/.next" ]; then
    echo "⚡ Removing Next.js build cache..."
    rm -rf frontend/.next/
    echo "✅ Next.js cache removed"
fi

if [ -d "frontend/out" ]; then
    echo "📤 Removing Next.js build output..."
    rm -rf frontend/out/
    echo "✅ Next.js build output removed"
fi

# 5. Remove root dependencies
echo "[5/8] Cleaning root dependencies..."
if [ -d "node_modules" ]; then
    echo "📦 Removing root dependencies..."
    rm -rf node_modules/
    echo "✅ Root dependencies removed"
fi

if [ -f "package-lock.json" ]; then
    echo "🔒 Removing package lock file..."
    rm -f package-lock.json
    echo "✅ Package lock file removed"
fi

# 6. Remove Python cache files
echo "[6/8] Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "✅ Python cache files removed"

# 7. Remove log and PID files
echo "[7/8] Removing log and PID files..."
rm -f backend.pid frontend.pid app.pid
rm -f backend.log frontend.log backend_error.log frontend_error.log
rm -f app.log app_error.log
echo "✅ Log and PID files removed"

# 8. Clean legacy files (if any)
echo "[8/8] Cleaning legacy files..."
if [ -d "uploads" ]; then
    echo "📁 Cleaning legacy upload files..."
    find uploads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
fi

if [ -d "downloads" ]; then
    echo "📂 Cleaning legacy download files..."
    find downloads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
fi

if [ -d "temp" ]; then
    echo "🗂️  Removing legacy temporary files..."
    rm -rf temp/
fi

if [ -d "models" ]; then
    echo "🤖 Removing legacy model files..."
    rm -rf models/
fi

echo "✅ Legacy files cleaned"

# Environment variable cleanup guidance
echo
echo "🔧 ENVIRONMENT VARIABLE CLEANUP:"
echo "   Please manually remove the following environment variables if they exist:"
echo "   - HF_HOME"
echo "   - TRANSFORMERS_CACHE"
echo "   - HUGGINGFACE_HUB_CACHE"
echo
echo "   Check and remove from ~/.bashrc or ~/.zshrc:"
echo "   export HF_HOME=..."
echo "   export TRANSFORMERS_CACHE=..."
echo "   export HUGGINGFACE_HUB_CACHE=..."

# System package removal guidance (optional)
echo
echo "📋 ADDITIONAL CLEANUP OPTIONS:"
echo "   To also remove system-installed packages:"
echo
echo "   Python packages (if installed globally):"
echo "   pip uninstall -y flask flask-cors pillow torch torchvision transformers timm realesrgan opencv-python numpy"
echo
echo "   Node.js global packages (if any):"
echo "   npm uninstall -g concurrently"
echo

echo "🎉 EdgeHD 2.0 Full-Stack Platform removal completed!"
echo
echo "📋 WHAT WAS REMOVED:"
echo "   • Backend: Python environment, AI models, dependencies"
echo "   • Frontend: Node.js dependencies, build files, cache, timeline assets"
echo "   • Root: Project dependencies, logs, temporary files"
echo "   • Timeline: Professional editing interface components"
echo
echo "🔄 TO REINSTALL:"
echo "   chmod +x install.sh"
echo "   ./install.sh"
echo
echo "📁 DIRECTORY STRUCTURE PRESERVED:"
echo "   • Source code files remain intact"
echo "   • Configuration files preserved"
echo "   • README and documentation kept"
echo