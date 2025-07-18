#!/bin/bash

# AI 이미지 처리 도구 시작 스크립트
# 사용법: ./start.sh

PID_FILE="app.pid"
LOG_FILE="app.log"
ERROR_LOG="app_error.log"

echo "🚀 AI 이미지 처리 도구 시작 중..."

# 이미 실행 중인지 확인
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "❌ 이미 실행 중입니다 (PID: $PID)"
        echo "   중지하려면: ./stop.sh"
        exit 1
    else
        echo "🧹 오래된 PID 파일 정리 중..."
        rm -f "$PID_FILE"
    fi
fi

# Conda 환경 확인
if ! command -v conda &> /dev/null; then
    echo "❌ Conda가 설치되어 있지 않습니다."
    echo "   설치 후 다시 시도해주세요."
    exit 1
fi

# Conda 환경 활성화
echo "🔧 Conda 환경 'edgehd' 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda activate edgehd 2>/dev/null; then
    echo "❌ 'edgehd' 환경을 찾을 수 없습니다."
    echo "   먼저 ./install.sh를 실행해주세요."
    exit 1
fi

# Apple Silicon 환경변수 설정
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    echo "🍎 Apple Silicon 최적화 설정 완료"
fi

# 프로젝트 내 모델 저장 환경변수 설정
export HF_HOME="$(pwd)/models"
export TRANSFORMERS_CACHE="$(pwd)/models"
echo "🤖 AI 모델을 프로젝트 내에서 관리합니다 ($(pwd)/models)"

# 로그 파일 초기화
echo "📝 로그 파일 초기화 중..."
> "$LOG_FILE"
> "$ERROR_LOG"

# 백그라운드에서 실행
echo "▶️  서버를 백그라운드에서 시작합니다..."
nohup python app.py > "$LOG_FILE" 2> "$ERROR_LOG" &
PID=$!

# PID 저장
echo "$PID" > "$PID_FILE"

# 서버 시작 대기
echo "⏳ 서버 초기화 대기 중..."
sleep 3

# 프로세스 확인
if kill -0 "$PID" 2>/dev/null; then
    echo ""
    echo "✅ AI 이미지 처리 도구가 성공적으로 시작되었습니다!"
    echo ""
    echo "📊 서버 정보:"
    echo "   • PID: $PID"
    echo "   • 로그 파일: $LOG_FILE"
    echo "   • 에러 로그: $ERROR_LOG"
    echo ""
    echo "🌐 접속 주소:"
    echo "   • 로컬: http://localhost:8080"
    echo "   • 네트워크: http://$(hostname -I | awk '{print $1}'):8080"
    echo ""
    echo "📝 유용한 명령어:"
    echo "   • 상태 확인: ./status.sh"
    echo "   • 로그 보기: tail -f $LOG_FILE"
    echo "   • 서버 중지: ./stop.sh"
    echo ""
else
    echo "❌ 서버 시작에 실패했습니다."
    echo "   에러 로그를 확인해주세요: cat $ERROR_LOG"
    rm -f "$PID_FILE"
    exit 1
fi 