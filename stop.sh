#!/bin/bash

# Set script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID_FILE="$SCRIPT_DIR/backend.pid"
FRONTEND_PID_FILE="$SCRIPT_DIR/frontend.pid"
BACKEND_LOG="$SCRIPT_DIR/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/frontend.log"
BACKEND_ERROR_LOG="$SCRIPT_DIR/backend_error.log"
FRONTEND_ERROR_LOG="$SCRIPT_DIR/frontend_error.log"

echo "========================================"
echo "EdgeHD 2.0 - Full-Stack Platform"
echo "Stopping Backend + Frontend Servers"
echo "========================================"
echo

BACKEND_STOPPED=false
FRONTEND_STOPPED=false

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to safely kill a process
kill_process_safe() {
    local service_name="$1"
    local pid_file="$2"
    local stopped_var="$3"
    
    echo "🛑 [$service_name] Stopping $service_name server..."
    
    if [ ! -f "$pid_file" ]; then
        echo "ℹ️  $service_name PID file not found. $service_name may not be running."
        eval "$stopped_var=true"
        return
    fi
    
    local SERVICE_PID=$(cat "$pid_file" 2>/dev/null)
    
    # Validate PID is numeric
    if ! [[ "$SERVICE_PID" =~ ^[0-9]+$ ]]; then
        echo "❌ Invalid $service_name PID: $SERVICE_PID"
        rm -f "$pid_file"
        eval "$stopped_var=true"
        return
    fi
    
    # Check if process exists
    if ! is_process_running "$SERVICE_PID"; then
        echo "ℹ️  $service_name process $SERVICE_PID not found. Cleaning up PID file."
        rm -f "$pid_file"
        eval "$stopped_var=true"
        return
    fi
    
    echo "🔄 Stopping $service_name process $SERVICE_PID..."
    
    # Attempt graceful shutdown (SIGTERM)
    if kill -TERM "$SERVICE_PID" 2>/dev/null; then
        echo "⏳ Waiting for graceful shutdown..."
        
        # Wait for graceful shutdown (up to 10 seconds)
        for i in $(seq 1 10); do
            if ! is_process_running "$SERVICE_PID"; then
                echo "✅ $service_name stopped gracefully."
                rm -f "$pid_file"
                eval "$stopped_var=true"
                return
            fi
            sleep 1
        done
    fi
    
    # Force shutdown if graceful failed
    echo "⚠️  Graceful shutdown timed out. Force stopping..."
    if kill -KILL "$SERVICE_PID" 2>/dev/null; then
        sleep 2
        if ! is_process_running "$SERVICE_PID"; then
            echo "✅ $service_name force stopped."
            rm -f "$pid_file"
            eval "$stopped_var=true"
        else
            echo "❌ Failed to stop $service_name process $SERVICE_PID."
        fi
    else
        echo "❌ Failed to force stop $service_name process $SERVICE_PID."
    fi
}

# Function to check if port is free
check_port_free() {
    local port=$1
    local service=$2
    
    if command -v lsof &> /dev/null; then
        if lsof -i :$port 2>/dev/null | grep -q LISTEN; then
            echo "⚠️  Port $port is still in use by another process."
            echo "📋 Process using port $port:"
            lsof -i :$port 2>/dev/null | grep LISTEN | while read line; do
                echo "   $line"
            done
        else
            echo "✅ Port $port ($service) is now free."
        fi
    else
        echo "ℹ️  Cannot check port $port (lsof not available)."
    fi
}

# Function to show log information
show_log_info() {
    echo
    echo "📄 [Log Files] Log file information:"
    
    if [ -f "$BACKEND_LOG" ]; then
        BACKEND_LOG_SIZE=$(du -h "$BACKEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
        echo "   • Backend log: $BACKEND_LOG ($BACKEND_LOG_SIZE)"
    else
        echo "   • Backend log: $BACKEND_LOG (Not found)"
    fi
    
    if [ -f "$FRONTEND_LOG" ]; then
        FRONTEND_LOG_SIZE=$(du -h "$FRONTEND_LOG" 2>/dev/null | cut -f1 || echo '0B')
        echo "   • Frontend log: $FRONTEND_LOG ($FRONTEND_LOG_SIZE)"
    else
        echo "   • Frontend log: $FRONTEND_LOG (Not found)"
    fi
    
    # Show error logs if they have content
    if [ -f "$BACKEND_ERROR_LOG" ] && [ -s "$BACKEND_ERROR_LOG" ]; then
        BACKEND_ERROR_SIZE=$(du -h "$BACKEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
        echo "   • Backend errors: $BACKEND_ERROR_LOG ($BACKEND_ERROR_SIZE)"
        echo
        echo "🚨 WARNING: Backend error log contains content!"
        echo "===== Backend Error Log Preview (last 3 lines) ====="
        tail -3 "$BACKEND_ERROR_LOG"
        echo "=================================================="
    fi
    
    if [ -f "$FRONTEND_ERROR_LOG" ] && [ -s "$FRONTEND_ERROR_LOG" ]; then
        FRONTEND_ERROR_SIZE=$(du -h "$FRONTEND_ERROR_LOG" 2>/dev/null | cut -f1 || echo '0B')
        echo "   • Frontend errors: $FRONTEND_ERROR_LOG ($FRONTEND_ERROR_SIZE)"
        echo
        echo "🚨 WARNING: Frontend error log contains content!"
        echo "===== Frontend Error Log Preview (last 3 lines) ====="
        tail -3 "$FRONTEND_ERROR_LOG"
        echo "===================================================="
    fi
}

# Stop services
kill_process_safe "Backend" "$BACKEND_PID_FILE" "BACKEND_STOPPED"
kill_process_safe "Frontend" "$FRONTEND_PID_FILE" "FRONTEND_STOPPED"

# Kill any remaining EdgeHD processes
echo
echo "🧹 [Additional Cleanup] Killing any remaining EdgeHD processes..."
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true

# Verify ports are freed
echo
echo "🔍 [Port Verification] Checking if ports are freed..."
sleep 3

check_port_free 8080 "Backend"
check_port_free 3000 "Frontend"

# Show log information
show_log_info

# Cleanup option
echo
read -p "🗑️  Delete log files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$BACKEND_LOG" "$FRONTEND_LOG" "$BACKEND_ERROR_LOG" "$FRONTEND_ERROR_LOG"
    echo "🗑️  Log files deleted."
else
    echo "📄 Log files preserved."
fi

# Final status
echo
if [ "$BACKEND_STOPPED" = true ] && [ "$FRONTEND_STOPPED" = true ]; then
    echo "🎉 SUCCESS: EdgeHD 2.0 Full-Stack Platform stopped successfully!"
else
    echo "⚠️  WARNING: Some servers may not have stopped properly."
    echo "Please check the processes manually or restart your computer if needed."
fi

echo
echo "🔄 To restart the platform, run: ./start.sh"
echo "🚀 For development mode, run: ./run.sh"