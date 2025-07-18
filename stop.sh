#!/bin/bash

# AI 이미지 처리 도구 중지 스크립트
# 사용법: ./stop.sh

PID_FILE="app.pid"
LOG_FILE="app.log"
ERROR_LOG="app_error.log"

echo "🛑 AI 이미지 처리 도구 중지 중..."

# PID 파일 확인
if [ ! -f "$PID_FILE" ]; then
    echo "❌ 실행 중인 서버를 찾을 수 없습니다."
    echo "   PID 파일이 존재하지 않습니다: $PID_FILE"
    exit 1
fi

# PID 읽기
PID=$(cat "$PID_FILE")

# 프로세스 확인
if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ 프로세스 $PID가 실행 중이지 않습니다."
    echo "🧹 PID 파일 정리 중..."
    rm -f "$PID_FILE"
    exit 1
fi

echo "📋 서버 정보:"
echo "   • PID: $PID"
echo "   • 프로세스 확인: $(ps -p $PID -o comm= 2>/dev/null || echo '프로세스 없음')"

# 정상 종료 시도 (SIGTERM)
echo "⏳ 정상 종료 신호 전송 중... (SIGTERM)"
kill -TERM "$PID"

# 정상 종료 대기
WAIT_TIME=10
for i in $(seq 1 $WAIT_TIME); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "✅ 서버가 정상적으로 종료되었습니다."
        rm -f "$PID_FILE"
        
        # 로그 파일 정리 옵션
        echo ""
        echo "📁 로그 파일 정리:"
        echo "   • 로그 파일: $LOG_FILE ($(du -h $LOG_FILE 2>/dev/null | cut -f1 || echo '0B'))"
        echo "   • 에러 로그: $ERROR_LOG ($(du -h $ERROR_LOG 2>/dev/null | cut -f1 || echo '0B'))"
        echo ""
        read -p "로그 파일을 삭제하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$LOG_FILE" "$ERROR_LOG"
            echo "🗑️  로그 파일이 삭제되었습니다."
        else
            echo "📄 로그 파일이 보존되었습니다."
        fi
        
        echo ""
        echo "🎯 서버 중지 완료!"
        exit 0
    fi
    
    printf "   대기 중... (%d/%d초)\r" "$i" "$WAIT_TIME"
    sleep 1
done

echo ""
echo "⚠️  정상 종료에 실패했습니다. 강제 종료를 시도합니다..."

# 강제 종료 시도 (SIGKILL)
echo "💥 강제 종료 신호 전송 중... (SIGKILL)"
kill -KILL "$PID" 2>/dev/null

# 강제 종료 확인
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "✅ 서버가 강제로 종료되었습니다."
    rm -f "$PID_FILE"
    echo "🎯 서버 중지 완료!"
else
    echo "❌ 프로세스 종료에 실패했습니다."
    echo "   수동으로 종료해야 할 수 있습니다: kill -9 $PID"
    echo "   또는 시스템 관리자에게 문의하세요."
    exit 1
fi 