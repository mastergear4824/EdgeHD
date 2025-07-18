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
