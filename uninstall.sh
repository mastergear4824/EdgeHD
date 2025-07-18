#!/bin/bash

echo "🗑️  EdgeHD AI 이미지 처리 애플리케이션 제거를 시작합니다..."

# 현재 디렉토리 확인
if [ ! -f "app.py" ]; then
    echo "❌ EdgeHD 프로젝트 디렉토리에서 실행해주세요"
    exit 1
fi

# 사용자 확인
echo "⚠️  다음 항목들이 제거됩니다:"
echo "   - Python 가상환경 (venv/)"
echo "   - 설치된 Python 패키지들"
echo "   - AI 모델 파일들 (models/)"
echo "   - 업로드된 파일들 (uploads/)"
echo "   - 다운로드된 파일들 (downloads/)"
echo "   - 환경변수 설정"
echo ""
read -p "정말로 제거하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 제거가 취소되었습니다"
    exit 1
fi

echo ""
echo "🧹 환경 정리를 시작합니다..."

# 1. 가상환경 제거
if [ -d "venv" ]; then
    echo "📦 Python 가상환경 제거 중..."
    rm -rf venv/
    echo "✅ 가상환경 제거 완료"
else
    echo "ℹ️  가상환경이 없습니다"
fi

# 2. 모델 파일들 제거
if [ -d "models" ]; then
    echo "🤖 AI 모델 파일들 제거 중..."
    rm -rf models/
    echo "✅ 모델 파일들 제거 완료"
else
    echo "ℹ️  모델 파일들이 없습니다"
fi

# 3. 업로드 파일들 정리
if [ -d "uploads" ]; then
    echo "📁 업로드 파일들 정리 중..."
    find uploads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
    echo "✅ 업로드 파일들 정리 완료"
fi

# 4. 다운로드 파일들 정리
if [ -d "downloads" ]; then
    echo "📂 다운로드 파일들 정리 중..."
    find downloads/ -type f ! -name '.gitkeep' -delete 2>/dev/null
    echo "✅ 다운로드 파일들 정리 완료"
fi

# 5. Python 캐시 파일들 제거
echo "🗂️  Python 캐시 파일들 제거 중..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null
echo "✅ 캐시 파일들 제거 완료"

# 6. 환경변수 설정 안내
echo ""
echo "🔧 환경변수 정리 안내:"
echo "   다음 환경변수들을 수동으로 제거해주세요:"
echo "   - HF_HOME"
echo "   - TRANSFORMERS_CACHE"
echo ""
echo "   ~/.bashrc 또는 ~/.zshrc에서 다음과 같은 라인들을 찾아 제거:"
echo "   export HF_HOME=..."
echo "   export TRANSFORMERS_CACHE=..."

# 7. 시스템 패키지 제거 안내 (선택사항)
echo ""
echo "📋 추가 정리 옵션:"
echo "   시스템에 설치된 Python 패키지들도 제거하려면:"
echo "   pip uninstall -y flask flask-cors pillow torch torchvision transformers timm realesrgan opencv-python numpy"
echo ""

echo "🎉 EdgeHD 제거가 완료되었습니다!"
echo ""
echo "📝 재설치하려면:"
echo "   chmod +x install.sh"
echo "   ./install.sh"
echo "" 