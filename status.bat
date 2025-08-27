@echo off
setlocal enabledelayedexpansion

set BACKEND_PID_FILE=backend.pid
set FRONTEND_PID_FILE=frontend.pid
set BACKEND_LOG=backend.log
set FRONTEND_LOG=frontend.log
set BACKEND_ERROR_LOG=backend_error.log
set FRONTEND_ERROR_LOG=frontend_error.log

echo ========================================
echo EdgeHD 2.0 Full-Stack Platform Status
echo ========================================
echo.

:: Check backend status
echo [BACKEND STATUS]
if not exist "%BACKEND_PID_FILE%" (
    echo ERROR: Backend is not running.
    echo    Backend PID file not found: %BACKEND_PID_FILE%
    set "BACKEND_RUNNING=false"
) else (
    set /p BACKEND_PID=<"%BACKEND_PID_FILE%"
    tasklist /FI "PID eq !BACKEND_PID!" 2>nul | find "!BACKEND_PID!" >nul
    if errorlevel 1 (
        echo ERROR: Backend is not running. (PID file exists)
        echo    Cannot find PID !BACKEND_PID! process.
        set "BACKEND_RUNNING=false"
    ) else (
        echo SUCCESS: Backend is running normally!
        echo    * PID: !BACKEND_PID!
        
        :: Get process info
        for /f "tokens=1" %%a in ('tasklist /FI "PID eq !BACKEND_PID!" /FO CSV /NH 2^>nul') do set "BACKEND_PROCESS=%%~a"
        echo    * Process: !BACKEND_PROCESS!
        
        :: Check port 8080
        netstat -an | find ":8080" | find "LISTENING" >nul
        if not errorlevel 1 (
            echo    * Port 8080: Active (LISTENING)
            echo    * API URL: http://localhost:8080
        ) else (
            echo    * Port 8080: Inactive (may be starting)
        )
        
        set "BACKEND_RUNNING=true"
    )
)

echo.

:: Check frontend status
echo [FRONTEND STATUS]
if not exist "%FRONTEND_PID_FILE%" (
    echo ERROR: Frontend is not running.
    echo    Frontend PID file not found: %FRONTEND_PID_FILE%
    set "FRONTEND_RUNNING=false"
) else (
    set /p FRONTEND_PID=<"%FRONTEND_PID_FILE%"
    tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | find "!FRONTEND_PID!" >nul
    if errorlevel 1 (
        echo ERROR: Frontend is not running. (PID file exists)
        echo    Cannot find PID !FRONTEND_PID! process.
        set "FRONTEND_RUNNING=false"
    ) else (
        echo SUCCESS: Frontend is running normally!
        echo    * PID: !FRONTEND_PID!
        
        :: Get process info
        for /f "tokens=1" %%a in ('tasklist /FI "PID eq !FRONTEND_PID!" /FO CSV /NH 2^>nul') do set "FRONTEND_PROCESS=%%~a"
        echo    * Process: !FRONTEND_PROCESS!
        
        :: Check port 3000
        netstat -an | find ":3000" | find "LISTENING" >nul
        if not errorlevel 1 (
            echo    * Port 3000: Active (LISTENING)
            echo    * UI URL: http://localhost:3000
        ) else (
            echo    * Port 3000: Inactive (may be starting)
        )
        
        set "FRONTEND_RUNNING=true"
    )
)

echo.

:: Overall status
echo [OVERALL STATUS]
if "%BACKEND_RUNNING%"=="true" if "%FRONTEND_RUNNING%"=="true" (
    echo SUCCESS: Full-Stack Platform is running normally!
    echo.
    echo ACCESS URLs:
    echo    * Frontend UI: http://localhost:3000
    echo    * Backend API: http://localhost:8080
    
    :: Get local IP
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| find "IPv4"') do (
        set "LOCAL_IP=%%a"
        set "LOCAL_IP=!LOCAL_IP: =!"
        echo    * Network UI: http://!LOCAL_IP!:3000
        goto :ip_found
    )
    :ip_found
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

echo.

:: Log file information
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

echo.

:: System resources
echo [SYSTEM RESOURCES]
for /f "tokens=2" %%a in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set "TOTAL_MEM=%%a"
for /f "tokens=2" %%a in ('wmic OS get FreePhysicalMemory /value ^| find "="') do set "FREE_MEM_KB=%%a"

if defined TOTAL_MEM if defined FREE_MEM_KB (
    set /a TOTAL_MEM_GB=!TOTAL_MEM!/1024/1024/1024
    set /a FREE_MEM_MB=!FREE_MEM_KB!/1024
    set /a USED_MEM_MB=(!TOTAL_MEM!/1024/1024) - !FREE_MEM_MB!
    echo    * Memory: !USED_MEM_MB! MB / !TOTAL_MEM_GB! GB in use
)

for /f "tokens=3,4" %%a in ('dir /-c ^| find "bytes free"') do (
    echo    * Disk: %%a %%b free space
    goto :disk_found
)
:disk_found

echo.

:: Application info
echo [APPLICATION INFO]
echo    * Architecture: Backend (Flask/Python) + Frontend (Next.js/React)
echo    * AI Models: BiRefNet + Real-ESRGAN
echo    * Features: Image/Video processing, Background removal, Upscaling
echo    * UI: Modern React with shadcn/ui components
echo.

:: Useful commands
echo [USEFUL COMMANDS]
echo    * View backend logs: type %BACKEND_LOG%
echo    * View frontend logs: type %FRONTEND_LOG%
echo    * Real-time backend logs: powershell "Get-Content %BACKEND_LOG% -Wait"
echo    * Real-time frontend logs: powershell "Get-Content %FRONTEND_LOG% -Wait"
echo    * Stop servers: stop.bat
echo    * Restart servers: stop.bat ^&^& start.bat
echo.

pause