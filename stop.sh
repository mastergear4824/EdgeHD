#!/bin/bash

BACKEND_PID_FILE="backend.pid"
FRONTEND_PID_FILE="frontend.pid"
BACKEND_LOG="backend.log"
FRONTEND_LOG="frontend.log"
BACKEND_ERROR_LOG="backend_error.log"
FRONTEND_ERROR_LOG="frontend_error.log"

echo "========================================"
echo "EdgeHD 2.0 Full-Stack Platform"
echo "Stopping Backend + Frontend Servers"
echo "========================================"
echo

BACKEND_STOPPED=false
FRONTEND_STOPPED=false

# Stop backend server
echo "[1/2] Stopping backend server..."
if [ ! -f "$BACKEND_PID_FILE" ]; then
    echo "ℹ️  Backend PID file not found. Backend may not be running."
    BACKEND_STOPPED=true
else
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "ℹ️  Backend process $BACKEND_PID not found. Cleaning up PID file."
        rm -f "$BACKEND_PID_FILE"
        BACKEND_STOPPED=true
    else
        echo "🛑 Stopping backend process $BACKEND_PID..."
        
        # Attempt graceful shutdown (SIGTERM)
        kill -TERM "$BACKEND_PID"
        
        # Wait for graceful shutdown
        for i in $(seq 1 5); do
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                echo "✅ Backend stopped gracefully."
                rm -f "$BACKEND_PID_FILE"
                BACKEND_STOPPED=true
                break
            fi
            sleep 1
        done
        
        # Force shutdown if graceful failed
        if [ "$BACKEND_STOPPED" = false ]; then
            echo "⚠️  Graceful shutdown timed out. Force stopping..."
            kill -KILL "$BACKEND_PID" 2>/dev/null
            sleep 2
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                echo "✅ Backend force stopped."
                rm -f "$BACKEND_PID_FILE"
                BACKEND_STOPPED=true
            else
                echo "❌ Failed to stop backend process $BACKEND_PID."
            fi
        fi
    fi
fi

# Stop frontend server
echo "[2/2] Stopping frontend server..."
if [ ! -f "$FRONTEND_PID_FILE" ]; then
    echo "ℹ️  Frontend PID file not found. Frontend may not be running."
    FRONTEND_STOPPED=true
else
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ℹ️  Frontend process $FRONTEND_PID not found. Cleaning up PID file."
        rm -f "$FRONTEND_PID_FILE"
        FRONTEND_STOPPED=true
    else
        echo "🛑 Stopping frontend process $FRONTEND_PID..."
        
        # Attempt graceful shutdown (SIGTERM)
        kill -TERM "$FRONTEND_PID"
        
        # Wait for graceful shutdown
        for i in $(seq 1 5); do
            if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
                echo "✅ Frontend stopped gracefully."
                rm -f "$FRONTEND_PID_FILE"
                FRONTEND_STOPPED=true
                break
            fi
            sleep 1
        done
        
        # Force shutdown if graceful failed
        if [ "$FRONTEND_STOPPED" = false ]; then
            echo "⚠️  Graceful shutdown timed out. Force stopping..."
            kill -KILL "$FRONTEND_PID" 2>/dev/null
            sleep 2
            if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
                echo "✅ Frontend force stopped."
                rm -f "$FRONTEND_PID_FILE"
                FRONTEND_STOPPED=true
            else
                echo "❌ Failed to stop frontend process $FRONTEND_PID."
            fi
        fi
    fi
fi

# Verify ports are freed
echo
echo "🔍 Verifying ports are freed..."
sleep 2

if command -v lsof &> /dev/null; then
    if lsof -i :8080 2>/dev/null | grep -q LISTEN; then
        echo "⚠️  Port 8080 is still in use by another process."
    else
        echo "✅ Port 8080 (backend) is now free."
    fi
    
    if lsof -i :3000 2>/dev/null | grep -q LISTEN; then
        echo "⚠️  Port 3000 is still in use by another process."
    else
        echo "✅ Port 3000 (frontend) is now free."
    fi
fi

# Log file cleanup option
echo
echo "📁 LOG FILE CLEANUP:"
if [ -f "$BACKEND_LOG" ]; then
    BACKEND_LOG_SIZE=$(du -h "$BACKEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Backend log: $BACKEND_LOG ($BACKEND_LOG_SIZE)"
else
    echo "   • Backend log: $BACKEND_LOG (File not found)"
fi

if [ -f "$FRONTEND_LOG" ]; then
    FRONTEND_LOG_SIZE=$(du -h "$FRONTEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Frontend log: $FRONTEND_LOG ($FRONTEND_LOG_SIZE)"
else
    echo "   • Frontend log: $FRONTEND_LOG (File not found)"
fi

# Show error logs if they have content
if [ -f "$BACKEND_ERROR_LOG" ] && [ -s "$BACKEND_ERROR_LOG" ]; then
    BACKEND_ERROR_SIZE=$(du -h "$BACKEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Backend errors: $BACKEND_ERROR_LOG ($BACKEND_ERROR_SIZE)"
    echo
    echo "🚨 BACKEND ERROR LOG PREVIEW (last 5 lines):"
    tail -5 "$BACKEND_ERROR_LOG"
fi

if [ -f "$FRONTEND_ERROR_LOG" ] && [ -s "$FRONTEND_ERROR_LOG" ]; then
    FRONTEND_ERROR_SIZE=$(du -h "$FRONTEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
    echo "   • Frontend errors: $FRONTEND_ERROR_LOG ($FRONTEND_ERROR_SIZE)"
    echo
    echo "🚨 FRONTEND ERROR LOG PREVIEW (last 5 lines):"
    tail -5 "$FRONTEND_ERROR_LOG"
fi

echo
read -p "Delete log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$BACKEND_LOG" "$FRONTEND_LOG" "$BACKEND_ERROR_LOG" "$FRONTEND_ERROR_LOG"
    echo "🗑️  Log files deleted."
else
    echo "📄 Log files preserved."
fi

echo
if [ "$BACKEND_STOPPED" = true ] && [ "$FRONTEND_STOPPED" = true ]; then
    echo "🎉 SUCCESS: EdgeHD 2.0 Full-Stack Platform stopped successfully!"
else
    echo "⚠️  WARNING: Some servers may not have stopped properly."
    echo "Please check the status manually or restart your computer if needed."
fi

echo
echo "🔄 To restart the platform, run: ./start.sh"
echo "🚀 For development mode, run: ./run.sh or npm run dev"