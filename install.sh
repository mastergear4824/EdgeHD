#!/bin/bash

# AI 이미지/비디오 처리 도구 자동 설치 스크립트
# 사용법: chmod +x install.sh && ./install.sh

# Miniconda 설치 함수
install_miniconda() {
    echo ""
    echo "Downloading Miniconda3 installer..."
    echo "This may take a few minutes depending on your internet connection."
    
    # 아키텍처 및 OS별 설치 파일 결정
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
    
    # 다운로드
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
    
    # 무인 설치 실행
    bash "/tmp/$INSTALLER_NAME" -b -p "$HOME/miniconda3"
    if [ $? -ne 0 ]; then
        echo "ERROR: Miniconda installation failed."
        return 1
    fi
    
    # 설치 파일 정리
    rm -f "/tmp/$INSTALLER_NAME"
    
    # PATH 설정
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # conda 초기화
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1
    "$HOME/miniconda3/bin/conda" init zsh >/dev/null 2>&1
    
    echo ""
    echo "Miniconda3 has been installed successfully!"
    echo "Location: $HOME/miniconda3"
    echo ""
    echo "IMPORTANT: Please close this terminal and open a new one for conda to work properly."
    
    return 0
}

echo "Starting AI Image/Video Processing Tool Installation..."

# 운영체제 및 아키텍처 감지
OS="$(uname -s)"
ARCH="$(uname -m)"

# Conda 설치 확인
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

# Conda 환경 생성
echo "Creating Conda environment 'edgehd'..."
conda create -n edgehd python=3.10 -y

# 환경 활성화
echo "Activating Conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edgehd

# PyTorch 설치 (시스템별)
echo "Installing PyTorch 2.1.0 (Real-ESRGAN v0.3.0 compatible)..."
case "$OS" in
    Darwin*)
        if [[ "$ARCH" == "arm64" ]]; then
            echo "Apple Silicon detected - Installing MPS-enabled PyTorch 2.1.0"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
            # Apple Silicon 환경변수 설정
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

# transformers 호환 버전 설치
echo "Installing transformers 4.35.0 (PyTorch 2.1.0 compatible)..."
pip install transformers==4.35.0

# 프로젝트 디렉토리 구조 생성
echo "Creating project directory structure..."
mkdir -p uploads downloads temp models/hub

# 필수 패키지 설치
echo "Installing required packages..."
pip install -r requirements.txt

# AI 모델 추가 의존성
echo "Installing AI model dependencies..."
pip install einops>=0.6.0 kornia>=0.7.0 timm>=0.9.0 realesrgan==0.3.0

# 프로젝트 내 모델 저장 설정
echo "Configuring AI models..."
echo "   All AI models will be stored in project models/ directory"
echo "   AI models downloaded on first run:"
echo "      * BiRefNet background removal model (~424MB)"
echo "      * Real-ESRGAN General v3 4x upscaling model (~17MB)"
echo "      * v0.3.0 supports only 4x (no 2x dedicated model)"
echo "   Models are managed independently per project"

echo ""
echo "Installation completed successfully!"
echo ""
echo "How to run:"
echo "   conda activate edgehd"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "   export PYTORCH_ENABLE_MPS_FALLBACK=1  # Apple Silicon only"
fi

echo "   python app.py"
echo ""
echo "Server URL: http://localhost:8080"
echo ""
echo "See README.md for detailed usage instructions."

# 실행 스크립트 생성
echo "Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
# AI 이미지/비디오 처리 도구 실행 스크립트

echo "Starting AI Image/Video Processing Tool..."

# Conda 환경 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edgehd

# Apple Silicon 환경변수 설정
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

# 프로젝트 내 모델 저장 환경변수 설정
export HF_HOME="$(pwd)/models"
export TRANSFORMERS_CACHE="$(pwd)/models"

# 애플리케이션 실행
python app.py
EOF

chmod +x run.sh

echo "SUCCESS: Run script 'run.sh' created"
echo "   You can now run './run.sh' for easy startup." 