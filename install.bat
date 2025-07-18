@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 AI 이미지/비디오 처리 도구 설치를 시작합니다...
echo.

:: Conda 설치 확인
where conda >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda가 설치되어 있지 않습니다.
    echo 📥 Conda를 먼저 설치해주세요:
    echo    ^• Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo    ^• Anaconda: https://www.anaconda.com/products/distribution
    echo.
    echo 설치 후 터미널을 다시 시작하고 이 스크립트를 실행해주세요.
    pause
    exit /b 1
)

echo ✅ Conda가 설치되어 있습니다.

:: Conda 환경 생성
echo 🔧 Conda 환경 'edgehd' 생성 중...
call conda create -n edgehd python=3.10 -y
if errorlevel 1 (
    echo ❌ Conda 환경 생성에 실패했습니다.
    pause
    exit /b 1
)

:: 환경 활성화
echo 🔄 Conda 환경 활성화 중...
call conda activate edgehd
if errorlevel 1 (
    echo ❌ Conda 환경 활성화에 실패했습니다.
    pause
    exit /b 1
)

:: GPU 감지 및 PyTorch 2.1.0 설치 (Real-ESRGAN v0.3.0 호환성)
echo 🔥 PyTorch 2.1.0 설치 중 (Real-ESRGAN v0.3.0 호환성)...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 💻 NVIDIA GPU를 찾을 수 없습니다. CPU 버전 PyTorch 2.1.0을 설치합니다.
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
) else (
    echo 🎮 NVIDIA GPU가 감지되었습니다. CUDA 버전 PyTorch 2.1.0을 설치합니다.
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
)

if errorlevel 1 (
    echo ❌ PyTorch 설치에 실패했습니다.
    pause
    exit /b 1
)

:: transformers 호환 버전 설치
echo 🤖 transformers 4.35.0 설치 중 (Real-ESRGAN v0.3.0 호환)...
pip install transformers==4.35.0
if errorlevel 1 (
    echo ❌ transformers 설치에 실패했습니다.
    pause
    exit /b 1
)

:: 프로젝트 디렉토리 구조 생성
echo 📁 프로젝트 디렉토리 구조 생성 중...
if not exist uploads mkdir uploads
if not exist downloads mkdir downloads
if not exist temp mkdir temp
if not exist models mkdir models
if not exist models\hub mkdir models\hub

:: 필수 패키지 설치
echo 📦 필수 패키지 설치 중...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 패키지 설치에 실패했습니다.
    pause
    exit /b 1
)

:: AI 모델 추가 의존성
echo 🤖 AI 모델 의존성 설치 중...
pip install transformers einops kornia timm realesrgan
if errorlevel 1 (
    echo ❌ AI 모델 의존성 설치에 실패했습니다.
    pause
    exit /b 1
)

:: 프로젝트 내 모델 저장 설정
echo 🤖 AI 모델 설정 중...
echo    ⚡ 모든 AI 모델이 프로젝트 내 models\ 디렉토리에 저장됩니다
echo    ⚡ 첫 실행 시 자동 다운로드되는 AI 모델들:
echo       ^• BiRefNet 배경제거 모델 (~424MB)
echo       ^• Real-ESRGAN General v3 4x 업스케일링 모델 (~17MB)
echo       ^• ⚠️  v0.3.0에서는 2x 전용 모델이 없어 4x만 지원
echo    ⚡ 프로젝트 독립적으로 모델이 관리됩니다

echo.
echo 🎉 설치가 완료되었습니다!
echo.
echo 🚀 실행 방법:
echo    start.bat          - 백그라운드 서버 시작
echo    python app.py      - 개발 서버 시작 (포그라운드)
echo.
echo 🌐 서버 주소: http://localhost:8080
echo.
echo 📝 자세한 사용법은 README.md를 확인해주세요.

:: 실행 스크립트 생성
echo 📜 실행 스크립트 생성 중...
(
echo @echo off
echo chcp 65001 ^>nul
echo echo 🚀 AI 이미지/비디오 처리 도구를 시작합니다...
echo call conda activate edgehd
echo.
echo :: 프로젝트 내 모델 저장 환경변수 설정
echo set HF_HOME=%%cd%%\models
echo set TRANSFORMERS_CACHE=%%cd%%\models
echo.
echo python app.py
echo pause
) > run.bat

echo ✅ 실행 스크립트 'run.bat' 생성 완료
echo    다음부터는 'run.bat'를 더블클릭하여 간편하게 실행할 수 있습니다.
echo.
pause 