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
echo "EdgeHD 2.0 Full-Stack Platform Status"
echo "========================================"
echo

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to get process uptime
get_process_uptime() {
    local pid=$1
    if command -v ps &> /dev/null; then
        ps -p "$pid" -o etime= 2>/dev/null | xargs || echo "Unknown"
    else
        echo "Unknown"
    fi
}

# Function to get process memory usage
get_process_memory() {
    local pid=$1
    if command -v ps &> /dev/null; then
        ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}' || echo "Unknown"
    else
        echo "Unknown"
    fi
}

# Function to check service status
check_service_status() {
    local service_name="$1"
    local pid_file="$2"
    local port="$3"
    local running_var="$4"
    
    echo "🔧 [$service_name STATUS]"
    
    if [ ! -f "$pid_file" ]; then
        echo "❌ $service_name is not running."
        echo "   $service_name PID file not found: $pid_file"
        eval "$running_var=false"
        return
    fi
    
    local SERVICE_PID=$(cat "$pid_file" 2>/dev/null)
    
    # Validate PID is numeric
    if ! [[ "$SERVICE_PID" =~ ^[0-9]+$ ]]; then
        echo "❌ Invalid $service_name PID: $SERVICE_PID"
        echo "   PID file may be corrupted: $pid_file"
        eval "$running_var=false"
        return
    fi
    
    # Check if process exists
    if ! is_process_running "$SERVICE_PID"; then
        echo "❌ $service_name is not running. (PID file exists but process not found)"
        echo "   Cannot find PID $SERVICE_PID process."
        eval "$running_var=false"
        return
    fi
    
    echo "✅ $service_name is running normally!"
    echo "   • PID: $SERVICE_PID"
    echo "   • Process: $(ps -p $SERVICE_PID -o comm= 2>/dev/null || echo 'Unknown')"
    echo "   • Uptime: $(get_process_uptime $SERVICE_PID)"
    echo "   • Memory: $(get_process_memory $SERVICE_PID)"
    echo "   • CPU: $(ps -p $SERVICE_PID -o %cpu= 2>/dev/null | xargs || echo 'Unknown')%"
    
    # Check port status
    if command -v lsof &> /dev/null; then
        if lsof -p "$SERVICE_PID" -a -i :$port 2>/dev/null | grep -q LISTEN; then
            echo "   • Port $port: Active (LISTENING)"
            if [ "$service_name" = "Backend" ]; then
                echo "   • API URL: http://localhost:$port"
            else
                echo "   • UI URL: http://localhost:$port"
            fi
        else
            echo "   • Port $port: Inactive (service may be starting)"
        fi
    else
        echo "   • Port $port: Cannot check (lsof not available)"
    fi
    
    eval "$running_var=true"
}

# Function to show access URLs
show_access_urls() {
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
}

# Function to show log files information
show_log_files() {
    echo
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
}

# Function to show system resources
show_system_resources() {
    echo
    echo "💻 [SYSTEM RESOURCES]"
    
    # Memory information
    if command -v free &> /dev/null; then
        MEM_INFO=$(free -h | grep '^Mem:')
        TOTAL_MEM=$(echo $MEM_INFO | awk '{print $2}')
        USED_MEM=$(echo $MEM_INFO | awk '{print $3}')
        echo "   • Memory: $USED_MEM / $TOTAL_MEM used"
    elif command -v vm_stat &> /dev/null; then
        # macOS memory info
        PAGE_SIZE=$(vm_stat | grep "page size" | awk '{print $8}')
        if [ -n "$PAGE_SIZE" ]; then
            FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
            ACTIVE_PAGES=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
            INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
            WIRED_PAGES=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | sed 's/\.//')
            
            if [[ "$FREE_PAGES" =~ ^[0-9]+$ ]] && [[ "$ACTIVE_PAGES" =~ ^[0-9]+$ ]]; then
                TOTAL_PAGES=$((FREE_PAGES + ACTIVE_PAGES + INACTIVE_PAGES + WIRED_PAGES))
                USED_PAGES=$((ACTIVE_PAGES + INACTIVE_PAGES + WIRED_PAGES))
                TOTAL_GB=$((TOTAL_PAGES * PAGE_SIZE / 1024 / 1024 / 1024))
                USED_GB=$((USED_PAGES * PAGE_SIZE / 1024 / 1024 / 1024))
                echo "   • Memory: ${USED_GB}GB / ${TOTAL_GB}GB used"
            else
                echo "   • Memory: macOS (use Activity Monitor for details)"
            fi
        else
            echo "   • Memory: macOS (use Activity Monitor for details)"
        fi
    fi
    
    # Disk space
    if command -v df &> /dev/null; then
        DISK_INFO=$(df -h . | tail -1)
        DISK_USED=$(echo $DISK_INFO | awk '{print $3}')
        DISK_TOTAL=$(echo $DISK_INFO | awk '{print $2}')
        DISK_PERCENT=$(echo $DISK_INFO | awk '{print $5}')
        echo "   • Disk: $DISK_USED / $DISK_TOTAL used ($DISK_PERCENT)"
    fi
    
    # CPU load average
    if command -v uptime &> /dev/null; then
        LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1,$2,$3}')
        echo "   • Load Average: $LOAD_AVG (1min, 5min, 15min)"
    fi
}

# Function to show useful commands
show_useful_commands() {
    echo
    echo "🚀 [APPLICATION INFO]"
    echo "   • Architecture: Backend (Flask/Python) + Frontend (Next.js/React)"
    echo "   • AI Models: BiRefNet + Real-ESRGAN"
    echo "   • Features: Image/Video processing, Background removal, Upscaling"
    echo "   • UI: Modern React with shadcn/ui components"
    
    echo
    echo "🔧 [USEFUL COMMANDS]"
    echo "   • View backend logs: cat $BACKEND_LOG"
    echo "   • View frontend logs: cat $FRONTEND_LOG"
    echo "   • Real-time backend logs: tail -f $BACKEND_LOG"
    echo "   • Real-time frontend logs: tail -f $FRONTEND_LOG"
    echo "   • Stop servers: ./stop.sh"
    echo "   • Restart servers: ./stop.sh && ./start.sh"
    echo "   • Development mode: ./run.sh"
}

# Main execution
check_service_status "Backend" "$BACKEND_PID_FILE" "8080" "BACKEND_RUNNING"
echo
check_service_status "Frontend" "$FRONTEND_PID_FILE" "3000" "FRONTEND_RUNNING"

# Overall status
echo
echo "📊 [OVERALL STATUS]"
if [ "$BACKEND_RUNNING" = true ] && [ "$FRONTEND_RUNNING" = true ]; then
    echo "🎉 SUCCESS: Full-Stack Platform is running normally!"
    show_access_urls
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

# Show additional information
show_log_files
show_system_resources
show_useful_commands
echo