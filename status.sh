#!/bin/bash

BACKEND_PID_FILE="backend.pid"
FRONTEND_PID_FILE="frontend.pid"
BACKEND_LOG="backend.log"
FRONTEND_LOG="frontend.log"
BACKEND_ERROR_LOG="backend_error.log"
FRONTEND_ERROR_LOG="frontend_error.log"

echo "========================================"
echo "EdgeHD 2.0 Full-Stack Platform Status"
echo "========================================"
echo

# Check backend status
echo "🔧 [BACKEND STATUS]"
if [ ! -f "$BACKEND_PID_FILE" ]; then
    echo "❌ Backend is not running."
    echo "   Backend PID file not found: $BACKEND_PID_FILE"
    BACKEND_RUNNING=false
else
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "❌ Backend is not running. (PID file exists)"
        echo "   Cannot find PID $BACKEND_PID process."
        BACKEND_RUNNING=false
    else
        echo "✅ Backend is running normally!"
        echo "   • PID: $BACKEND_PID"
        echo "   • Process: $(ps -p $BACKEND_PID -o comm= 2>/dev/null || echo 'Unknown')"
        echo "   • Runtime: $(ps -p $BACKEND_PID -o etime= 2>/dev/null | xargs || echo 'Unknown')"
        echo "   • CPU: $(ps -p $BACKEND_PID -o %cpu= 2>/dev/null | xargs || echo 'Unknown')%"
        echo "   • Memory: $(ps -p $BACKEND_PID -o %mem= 2>/dev/null | xargs || echo 'Unknown')%"
        
        # Check port 8080
        if command -v lsof &> /dev/null; then
            if lsof -p "$BACKEND_PID" -a -i :8080 2>/dev/null | grep -q LISTEN; then
                echo "   • Port 8080: Active (LISTENING)"
                echo "   • API URL: http://localhost:8080"
            else
                echo "   • Port 8080: Inactive (may be starting)"
            fi
        fi
        
        BACKEND_RUNNING=true
    fi
fi

echo

# Check frontend status
echo "🎨 [FRONTEND STATUS]"
if [ ! -f "$FRONTEND_PID_FILE" ]; then
    echo "❌ Frontend is not running."
    echo "   Frontend PID file not found: $FRONTEND_PID_FILE"
    FRONTEND_RUNNING=false
else
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "❌ Frontend is not running. (PID file exists)"
        echo "   Cannot find PID $FRONTEND_PID process."
        FRONTEND_RUNNING=false
    else
        echo "✅ Frontend is running normally!"
        echo "   • PID: $FRONTEND_PID"
        echo "   • Process: $(ps -p $FRONTEND_PID -o comm= 2>/dev/null || echo 'Unknown')"
        echo "   • Runtime: $(ps -p $FRONTEND_PID -o etime= 2>/dev/null | xargs || echo 'Unknown')"
        echo "   • CPU: $(ps -p $FRONTEND_PID -o %cpu= 2>/dev/null | xargs || echo 'Unknown')%"
        echo "   • Memory: $(ps -p $FRONTEND_PID -o %mem= 2>/dev/null | xargs || echo 'Unknown')%"
        
        # Check port 3000
        if command -v lsof &> /dev/null; then
            if lsof -p "$FRONTEND_PID" -a -i :3000 2>/dev/null | grep -q LISTEN; then
                echo "   • Port 3000: Active (LISTENING)"
                echo "   • UI URL: http://localhost:3000"
            else
                echo "   • Port 3000: Inactive (may be starting)"
            fi
        fi
        
        FRONTEND_RUNNING=true
    fi
fi

echo

# Overall status
echo "📊 [OVERALL STATUS]"
if [ "$BACKEND_RUNNING" = true ] && [ "$FRONTEND_RUNNING" = true ]; then
    echo "🎉 SUCCESS: Full-Stack Platform is running normally!"
    echo
    echo "🌐 ACCESS URLs:"
    echo "   • Frontend UI: http://localhost:3000"
    echo "   • Backend API: http://localhost:8080"
    
    if command -v hostname &> /dev/null; then
        LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        if [ -n "$LOCAL_IP" ]; then
            echo "   • Network UI: http://$LOCAL_IP:3000"
        fi
    fi
else
    echo "⚠️  WARNING: Platform is not fully operational."
    if [ "$BACKEND_RUNNING" = false ]; then
        echo "   • Backend: Not running"
    fi
    if [ "$FRONTEND_RUNNING" = false ]; then
        echo "   • Frontend: Not running"
    fi
    echo
    echo "To start: ./start.sh"
fi

echo

# Log file information
echo "📄 [LOG FILES]"
if [ -f "$BACKEND_LOG" ]; then
    BACKEND_LOG_SIZE=$(du -h "$BACKEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
    BACKEND_LOG_LINES=$(wc -l < "$BACKEND_LOG" 2>/dev/null || echo '0')
    echo "   • Backend log: $BACKEND_LOG ($BACKEND_LOG_SIZE, $BACKEND_LOG_LINES lines)"
else
    echo "   • Backend log: $BACKEND_LOG (File not found)"
fi

if [ -f "$FRONTEND_LOG" ]; then
    FRONTEND_LOG_SIZE=$(du -h "$FRONTEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
    FRONTEND_LOG_LINES=$(wc -l < "$FRONTEND_LOG" 2>/dev/null || echo '0')
    echo "   • Frontend log: $FRONTEND_LOG ($FRONTEND_LOG_SIZE, $FRONTEND_LOG_LINES lines)"
else
    echo "   • Frontend log: $FRONTEND_LOG (File not found)"
fi

# Check error logs
if [ -f "$BACKEND_ERROR_LOG" ] && [ -s "$BACKEND_ERROR_LOG" ]; then
    BACKEND_ERROR_SIZE=$(du -h "$BACKEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Backend errors: $BACKEND_ERROR_LOG ($BACKEND_ERROR_SIZE)"
    echo "     ⚠️  WARNING: Backend error log contains content!"
fi

if [ -f "$FRONTEND_ERROR_LOG" ] && [ -s "$FRONTEND_ERROR_LOG" ]; then
    FRONTEND_ERROR_SIZE=$(du -h "$FRONTEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Frontend errors: $FRONTEND_ERROR_LOG ($FRONTEND_ERROR_SIZE)"
    echo "     ⚠️  WARNING: Frontend error log contains content!"
fi

echo

# System resources
echo "💻 [SYSTEM RESOURCES]"
if command -v free &> /dev/null; then
    MEM_INFO=$(free -h | grep '^Mem:')
    echo "   • Memory: $(echo $MEM_INFO | awk '{print $3"/"$2" ("$3")"}')"
elif command -v vm_stat &> /dev/null; then
    # macOS
    echo "   • Memory: macOS (use Activity Monitor for details)"
fi

if command -v df &> /dev/null; then
    DISK_INFO=$(df -h . | tail -1)
    echo "   • Disk: $(echo $DISK_INFO | awk '{print $3"/"$2" ("$5" used)"}')"
fi

echo

# Application info
echo "🚀 [APPLICATION INFO]"
echo "   • Architecture: Backend (Flask/Python) + Frontend (Next.js/React)"
echo "   • AI Models: BiRefNet + Real-ESRGAN"
echo "   • Features: Image/Video processing, Background removal, Upscaling"
echo "   • UI: Modern React with shadcn/ui components"
echo

# Useful commands
echo "🔧 [USEFUL COMMANDS]"
echo "   • View backend logs: cat $BACKEND_LOG"
echo "   • View frontend logs: cat $FRONTEND_LOG"
echo "   • Real-time backend logs: tail -f $BACKEND_LOG"
echo "   • Real-time frontend logs: tail -f $FRONTEND_LOG"
echo "   • Stop servers: ./stop.sh"
echo "   • Restart servers: ./stop.sh && ./start.sh"
echo