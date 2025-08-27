@echo off
setlocal enabledelayedexpansion

:: Set script variables
set "SCRIPT_DIR=%~dp0"
set BACKEND_PID_FILE=%SCRIPT_DIR%backend.pid
set FRONTEND_PID_FILE=%SCRIPT_DIR%frontend.pid
set BACKEND_LOG=%SCRIPT_DIR%backend.log
set FRONTEND_LOG=%SCRIPT_DIR%frontend.log
set BACKEND_ERROR_LOG=%SCRIPT_DIR%backend_error.log
set FRONTEND_ERROR_LOG=%SCRIPT_DIR%frontend_error.log

echo ========================================
echo EdgeHD 2.0 Full-Stack Platform Status
echo ========================================
echo.

:: Check backend status
call :check_service_status "Backend" "%BACKEND_PID_FILE%" "8080" "python.exe" BACKEND_RUNNING

:: Check frontend status
call :check_service_status "Frontend" "%FRONTEND_PID_FILE%" "3000" "node.exe" FRONTEND_RUNNING

:: Overall status
echo.
echo [OVERALL STATUS]
if "%BACKEND_RUNNING%"=="true" if "%FRONTEND_RUNNING%"=="true" (
    echo SUCCESS: Full-Stack Platform is running normally!
    call :show_access_urls
) else (
    echo WARNING: Platform is not fully operational.
    if "%BACKEND_RUNNING%"=="false" (
        echo    * Backend: Not running
    )
    if "%FRONTEND_RUNNING%"=="false" (
        echo    * Frontend: Not running
    )
    echo.
    echo To start: start.bat
)

:: Show log file information
call :show_log_files

:: Show system resources
call :show_system_resources

:: Show useful commands
call :show_useful_commands

pause
goto :eof

:: Function to check service status
:check_service_status
set "service_name=%~1"
set "pid_file=%~2"
set "port=%~3"
set "process_name=%~4"
set "running_var=%~5"

echo [%service_name% STATUS]

if not exist "%pid_file%" (
    echo ERROR: %service_name% is not running.
    echo    %service_name% PID file not found: %pid_file%
    set "%running_var%=false"
    goto :eof
)

set /p SERVICE_PID=<"%pid_file%"
set "SERVICE_PID=!SERVICE_PID:"=!"
set "SERVICE_PID=!SERVICE_PID: =!"

:: Validate PID is numeric
echo !SERVICE_PID!| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo ERROR: Invalid %service_name% PID: !SERVICE_PID!
    echo    PID file may be corrupted: %pid_file%
    set "%running_var%=false"
    goto :eof
)

:: Check if process exists
tasklist /FI "PID eq !SERVICE_PID!" 2>nul | find "!SERVICE_PID!" >nul
if errorlevel 1 (
    echo ERROR: %service_name% is not running. (PID file exists but process not found)
    echo    Cannot find PID !SERVICE_PID! process.
    set "%running_var%=false"
    goto :eof
)

echo SUCCESS: %service_name% is running normally!
echo    * PID: !SERVICE_PID!

:: Get process info
for /f "tokens=1,2,3,4,5" %%a in ('tasklist /FI "PID eq !SERVICE_PID!" /FO CSV /NH 2^>nul') do (
    set "PROCESS_NAME=%%~a"
    set "PROCESS_MEM=%%~e"
    echo    * Process: !PROCESS_NAME!
    echo    * Memory: !PROCESS_MEM!
)

:: Get process uptime (approximation)
for /f "tokens=2" %%a in ('wmic process where "ProcessId=!SERVICE_PID!" get CreationDate /value 2^>nul ^| find "="') do (
    set "CREATION_DATE=%%a"
    if defined CREATION_DATE (
        set "CREATION_TIME=!CREATION_DATE:~8,6!"
        echo    * Started: !CREATION_TIME:~0,2!:!CREATION_TIME:~2,2!:!CREATION_TIME:~4,2!
    )
)

:: Check port status
netstat -an | find ":%port%" | find "LISTENING" >nul
if not errorlevel 1 (
    echo    * Port %port%: Active (LISTENING)
    if "%service_name%"=="Backend" (
        echo    * API URL: http://localhost:%port%
    ) else (
        echo    * UI URL: http://localhost:%port%
    )
) else (
    echo    * Port %port%: Inactive (service may be starting)
)

set "%running_var%=true"
echo.
goto :eof

:: Function to show access URLs
:show_access_urls
echo.
echo ACCESS URLs:
echo    * Frontend UI: http://localhost:3000
echo    * Backend API: http://localhost:8080

:: Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| find "IPv4" 2^>nul') do (
    set "LOCAL_IP=%%a"
    set "LOCAL_IP=!LOCAL_IP: =!"
    if defined LOCAL_IP (
        echo    * Network UI: http://!LOCAL_IP!:3000
        goto :ip_found
    )
)
:ip_found
goto :eof

:: Function to show log files
:show_log_files
echo.
echo [LOG FILES]

if exist "%BACKEND_LOG%" (
    for %%F in ("%BACKEND_LOG%") do set "BACKEND_LOG_SIZE=%%~zF"
    set /a BACKEND_LOG_KB=!BACKEND_LOG_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%BACKEND_LOG%"') do set "BACKEND_LOG_LINES=%%a"
    echo    * Backend log: %BACKEND_LOG% (!BACKEND_LOG_KB! KB, !BACKEND_LOG_LINES! lines)
) else (
    echo    * Backend log: %BACKEND_LOG% (File not found)
)

if exist "%FRONTEND_LOG%" (
    for %%F in ("%FRONTEND_LOG%") do set "FRONTEND_LOG_SIZE=%%~zF"
    set /a FRONTEND_LOG_KB=!FRONTEND_LOG_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%FRONTEND_LOG%"') do set "FRONTEND_LOG_LINES=%%a"
    echo    * Frontend log: %FRONTEND_LOG% (!FRONTEND_LOG_KB! KB, !FRONTEND_LOG_LINES! lines)
) else (
    echo    * Frontend log: %FRONTEND_LOG% (File not found)
)

:: Check error logs
if exist "%BACKEND_ERROR_LOG%" (
    for %%F in ("%BACKEND_ERROR_LOG%") do set "BACKEND_ERROR_SIZE=%%~zF"
    if !BACKEND_ERROR_SIZE! GTR 10 (
        set /a BACKEND_ERROR_KB=!BACKEND_ERROR_SIZE!/1024
        echo    * Backend errors: %BACKEND_ERROR_LOG% (!BACKEND_ERROR_KB! KB)
        echo      WARNING: Backend error log contains content!
    )
)

if exist "%FRONTEND_ERROR_LOG%" (
    for %%F in ("%FRONTEND_ERROR_LOG%") do set "FRONTEND_ERROR_SIZE=%%~zF"
    if !FRONTEND_ERROR_SIZE! GTR 10 (
        set /a FRONTEND_ERROR_KB=!FRONTEND_ERROR_SIZE!/1024
        echo    * Frontend errors: %FRONTEND_ERROR_LOG% (!FRONTEND_ERROR_KB! KB)
        echo      WARNING: Frontend error log contains content!
    )
)
goto :eof

:: Function to show system resources
:show_system_resources
echo.
echo [SYSTEM RESOURCES]

:: Memory information
for /f "tokens=2" %%a in ('wmic computersystem get TotalPhysicalMemory /value 2^>nul ^| find "="') do set "TOTAL_MEM=%%a"
for /f "tokens=2" %%a in ('wmic OS get FreePhysicalMemory /value 2^>nul ^| find "="') do set "FREE_MEM_KB=%%a"

if defined TOTAL_MEM if defined FREE_MEM_KB (
    set /a TOTAL_MEM_GB=!TOTAL_MEM!/1024/1024/1024
    set /a FREE_MEM_MB=!FREE_MEM_KB!/1024
    set /a USED_MEM_MB=(!TOTAL_MEM!/1024/1024) - !FREE_MEM_MB!
    set /a MEM_USAGE_PCT=(!USED_MEM_MB! * 100) / (!TOTAL_MEM!/1024/1024)
    echo    * Memory: !USED_MEM_MB! MB / !TOTAL_MEM_GB! GB used (!MEM_USAGE_PCT!%%)
)

:: Disk space
for /f "tokens=3,4" %%a in ('dir /-c 2^>nul ^| find "bytes free"') do (
    echo    * Disk: %%a %%b free space
    goto :disk_found
)
:disk_found

:: CPU usage (approximation)
for /f "skip=1 tokens=2" %%a in ('wmic cpu get loadpercentage /value 2^>nul ^| find "="') do (
    echo    * CPU: %%a%% usage
)
goto :eof

:: Function to show useful commands
:show_useful_commands
echo.
echo [APPLICATION INFO]
echo    * Architecture: Backend (Flask/Python) + Frontend (Next.js/React)
echo    * AI Models: BiRefNet + Real-ESRGAN
echo    * Features: Image/Video processing, Background removal, Upscaling
echo    * UI: Modern React with shadcn/ui components

echo.
echo [USEFUL COMMANDS]
echo    * View backend logs: type %BACKEND_LOG%
echo    * View frontend logs: type %FRONTEND_LOG%
echo    * Real-time backend logs: powershell "Get-Content %BACKEND_LOG% -Wait"
echo    * Real-time frontend logs: powershell "Get-Content %FRONTEND_LOG% -Wait"
echo    * Stop servers: stop.bat
echo    * Restart servers: stop.bat ^&^& start.bat
echo    * Development mode: run.bat
goto :eof