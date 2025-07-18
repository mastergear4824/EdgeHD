#!/bin/bash

# AI 이미지 처리 도구 자동 설치 스크립트
# 사용법: chmod +x install.sh && ./install.sh

echo "🚀 AI 이미지 처리 도구 설치를 시작합니다..."

# 운영체제 감지
OS="$(uname -s)"
ARCH="$(uname -m)"

# Conda 설치 확인
if ! command -v conda &> /dev/null; then
    echo "❌ Conda가 설치되어 있지 않습니다."
    echo "📥 Conda를 먼저 설치해주세요:"
    
    case "$OS" in
        Darwin*)
            echo "   # macOS (Miniforge 권장)"
            echo "   brew install miniforge"
            echo "   # 또는"
            echo "   curl -L -O \"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh\""
            echo "   bash Miniforge3-\$(uname)-\$(uname -m).sh"
            ;;
        Linux*)
            echo "   # Linux"
            echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            echo "   bash Miniconda3-latest-Linux-x86_64.sh"
            ;;
        *)
            echo "   # Windows"
            echo "   https://docs.conda.io/en/latest/miniconda.html 에서 다운로드"
            ;;
    esac
    exit 1
fi

echo "✅ Conda가 설치되어 있습니다."

# Conda 환경 생성
echo "🔧 Conda 환경 'edgehd' 생성 중..."
conda create -n edgehd python=3.10 -y

# 환경 활성화
echo "🔄 Conda 환경 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edgehd

# PyTorch 설치 (시스템별)
echo "🔥 PyTorch 2.1.0 설치 중 (Real-ESRGAN v0.3.0 호환)..."
case "$OS" in
    Darwin*)
        if [[ "$ARCH" == "arm64" ]]; then
            echo "🍎 Apple Silicon 감지 - MPS 지원 PyTorch 2.1.0 설치"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
            # Apple Silicon 환경변수 설정
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
            echo "✅ Apple Silicon 최적화 완료"
        else
            echo "💻 Intel Mac 감지 - CPU PyTorch 2.1.0 설치"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
    Linux*)
        echo "🐧 Linux 감지"
        if command -v nvidia-smi &> /dev/null; then
            echo "🎮 NVIDIA GPU 감지 - CUDA PyTorch 2.1.0 설치"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        else
            echo "💻 CPU 전용 PyTorch 2.1.0 설치"
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        fi
        ;;
    *)
        echo "💻 기본 CPU PyTorch 2.1.0 설치"
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# transformers 호환 버전 설치
echo "🤖 transformers 4.35.0 설치 중 (PyTorch 2.1.0 호환)..."
pip install transformers==4.35.0

# 프로젝트 디렉토리 구조 생성
echo "📁 프로젝트 디렉토리 구조 생성 중..."
mkdir -p uploads downloads temp models/hub

# 필수 패키지 설치
echo "📦 필수 패키지 설치 중..."
pip install -r requirements.txt

# AI 모델 추가 의존성
echo "🤖 AI 모델 의존성 설치 중..."
pip install einops>=0.6.0 kornia>=0.7.0 timm>=0.9.0 realesrgan==0.3.0

# 프로젝트 내 모델 저장 설정
echo "🤖 AI 모델 설정 중..."
echo "   ⚡ 모든 AI 모델이 프로젝트 내 models/ 디렉토리에 저장됩니다"
echo "   ⚡ 첫 실행 시 자동 다운로드되는 AI 모델들:"
echo "      • BiRefNet 배경제거 모델 (~424MB)"
echo "      • Real-ESRGAN General v3 4x 업스케일링 모델 (~17MB)"
echo "      • ⚠️  v0.3.0에서는 2x 전용 모델이 없어 4x만 지원"
echo "   ⚡ 프로젝트 독립적으로 모델이 관리됩니다"

echo ""
echo "🎉 설치가 완료되었습니다!"
echo ""
echo "🚀 실행 방법:"
echo "   conda activate edgehd"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "   export PYTORCH_ENABLE_MPS_FALLBACK=1  # Apple Silicon 전용"
fi

echo "   python app.py"
echo ""
echo "🌐 서버 주소: http://localhost:8080"
echo ""
echo "📝 자세한 사용법은 README.md를 확인해주세요."

# 실행 스크립트 생성
echo "📜 실행 스크립트 생성 중..."
cat > run.sh << 'EOF'
#!/bin/bash
# AI 이미지 처리 도구 실행 스크립트

echo "🚀 AI 이미지 처리 도구를 시작합니다..."

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

echo "✅ 실행 스크립트 'run.sh' 생성 완료"
echo "   다음부터는 './run.sh'로 간편하게 실행할 수 있습니다." 