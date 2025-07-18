#!/bin/bash

# AI 이미지 처리 도구 상태 확인 스크립트
# 사용법: ./status.sh

PID_FILE="app.pid"
LOG_FILE="app.log"
ERROR_LOG="app_error.log"

echo "📊 AI 이미지 처리 도구 상태 확인"
echo "=================================="

# PID 파일 확인
if [ ! -f "$PID_FILE" ]; then
    echo "❌ 서버가 실행되지 않고 있습니다."
    echo "   시작하려면: ./start.sh"
    echo ""
    exit 1
fi

# PID 읽기
PID=$(cat "$PID_FILE")

# 프로세스 확인
if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ 서버가 실행되지 않고 있습니다. (PID 파일은 존재)"
    echo "   PID $PID 프로세스를 찾을 수 없습니다."
    echo "   정리하려면: rm -f $PID_FILE"
    echo "   시작하려면: ./start.sh"
    echo ""
    exit 1
fi

# 서버 상태 표시
echo "✅ 서버가 정상적으로 실행 중입니다!"
echo ""

# 기본 정보
echo "📋 서버 정보:"
echo "   • PID: $PID"
echo "   • 프로세스 이름: $(ps -p $PID -o comm= 2>/dev/null || echo '알 수 없음')"
echo "   • 실행 시간: $(ps -p $PID -o etime= 2>/dev/null | xargs || echo '알 수 없음')"
echo "   • CPU 사용률: $(ps -p $PID -o %cpu= 2>/dev/null | xargs || echo '알 수 없음')%"
echo "   • 메모리 사용률: $(ps -p $PID -o %mem= 2>/dev/null | xargs || echo '알 수 없음')%"
echo ""

# 포트 확인
echo "🌐 네트워크 정보:"
if command -v lsof &> /dev/null; then
    PORT_INFO=$(lsof -p "$PID" -a -i 2>/dev/null | grep LISTEN)
    if [ -n "$PORT_INFO" ]; then
        echo "   • 사용 중인 포트:"
        echo "$PORT_INFO" | while read line; do
            PORT=$(echo "$line" | awk '{print $9}' | cut -d':' -f2)
            echo "     - 포트 $PORT"
        done
    else
        echo "   • 포트 정보를 찾을 수 없습니다."
    fi
else
    echo "   • lsof 명령어를 사용할 수 없습니다."
fi

# 접속 URL
echo "   • 로컬 접속: http://localhost:8080"
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$LOCAL_IP" ]; then
        echo "   • 네트워크 접속: http://$LOCAL_IP:8080"
    fi
fi
echo ""

# 로그 파일 정보
echo "📄 로그 파일 정보:"
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1 || echo '0B')
    LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo '0')
    echo "   • 일반 로그: $LOG_FILE ($LOG_SIZE, $LOG_LINES 줄)"
else
    echo "   • 일반 로그: $LOG_FILE (파일 없음)"
fi

if [ -f "$ERROR_LOG" ]; then
    ERROR_SIZE=$(du -h "$ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
    ERROR_LINES=$(wc -l < "$ERROR_LOG" 2>/dev/null || echo '0')
    echo "   • 에러 로그: $ERROR_LOG ($ERROR_SIZE, $ERROR_LINES 줄)"
    
    # 최근 에러 확인
    if [ -s "$ERROR_LOG" ]; then
        echo "   ⚠️  에러 로그에 내용이 있습니다!"
        echo "      최근 에러 보기: tail -f $ERROR_LOG"
    fi
else
    echo "   • 에러 로그: $ERROR_LOG (파일 없음)"
fi
echo ""

# 시스템 자원 정보 (선택사항)
echo "💻 시스템 자원:"
if command -v free &> /dev/null; then
    MEM_INFO=$(free -h | grep '^Mem:')
    echo "   • 메모리: $(echo $MEM_INFO | awk '{print $3"/"$2" ("$3")"}')"
elif command -v vm_stat &> /dev/null; then
    # macOS
    echo "   • 메모리: macOS (vm_stat 사용)"
fi

if command -v df &> /dev/null; then
    DISK_INFO=$(df -h . | tail -1)
    echo "   • 디스크: $(echo $DISK_INFO | awk '{print $3"/"$2" ("$5" 사용)"}')"
fi
echo ""

# 유용한 명령어
echo "🔧 유용한 명령어:"
echo "   • 로그 실시간 보기: tail -f $LOG_FILE"
echo "   • 에러 로그 보기: tail -f $ERROR_LOG"
echo "   • 서버 중지: ./stop.sh"
echo "   • 서버 재시작: ./stop.sh && ./start.sh"
echo "" 