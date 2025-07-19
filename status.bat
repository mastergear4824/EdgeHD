@echo off
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo AI Image/Video Processing Tool Status Check
echo ==========================================

:: Check PID file
if not exist "%PID_FILE%" (
    echo ERROR: Server is not running.
    echo    To start: start.bat
    echo.
    pause
    exit /b 1
)

:: Read PID
set /p PID=<"%PID_FILE%"

:: Check process
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ERROR: Server is not running. (PID file exists)
    echo    Cannot find PID %PID% process.
    echo    To clean up: del %PID_FILE%
    echo    To start: start.bat
    echo.
    pause
    exit /b 1
)

:: Display server status
echo SUCCESS: Server is running normally!
echo.

:: Basic information
echo SERVER INFO:
echo    * PID: %PID%

:: Get process name
for /f "tokens=1" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do set "PROCESS_NAME=%%~a"
echo    * Process name: !PROCESS_NAME!

:: CPU and memory usage (using wmic)
for /f "tokens=2 delims=," %%a in ('wmic process where "ProcessId=%PID%" get PageFileUsage /format:csv 2^>nul ^| find "%PID%"') do set "MEMORY_KB=%%a"
if defined MEMORY_KB (
    set /a MEMORY_MB=!MEMORY_KB!/1024
    echo    * Memory usage: !MEMORY_MB! MB
) else (
    echo    * Memory usage: Unknown
)

:: Start time (approximate)
for /f "tokens=2,3" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do (
    echo    * Process info: %%a %%b
)

echo.

:: Network information
echo NETWORK INFO:
:: Check port 8080
netstat -an | find ":8080" | find "LISTENING" >nul
if not errorlevel 1 (
    echo    * Port 8080: Active (LISTENING)
    echo    * Local access: http://localhost:8080
    
    :: Get local IP address
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| find "IPv4"') do (
        set "LOCAL_IP=%%a"
        set "LOCAL_IP=!LOCAL_IP: =!"
        echo    * Network access: http://!LOCAL_IP!:8080
        goto :ip_found
    )
    :ip_found
) else (
    echo    * Port 8080: Inactive (Server may still be starting)
)
echo.

:: Log file information
echo LOG FILE INFO:
if exist "%LOG_FILE%" (
    for %%F in ("%LOG_FILE%") do set "LOG_SIZE=%%~zF"
    set /a LOG_SIZE_KB=!LOG_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%LOG_FILE%"') do set "LOG_LINES=%%a"
    echo    * General log: %LOG_FILE% (!LOG_SIZE_KB! KB, !LOG_LINES! lines)
) else (
    echo    * General log: %LOG_FILE% (File not found)
)

if exist "%ERROR_LOG%" (
    for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
    set /a ERROR_SIZE_KB=!ERROR_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%ERROR_LOG%"') do set "ERROR_LINES=%%a"
    echo    * Error log: %ERROR_LOG% (!ERROR_SIZE_KB! KB, !ERROR_LINES! lines)
    
    :: Check if error log has content
    if !ERROR_SIZE! GTR 10 (
        echo    WARNING: Error log contains content!
        echo       View recent errors: type %ERROR_LOG%
    )
) else (
    echo    * Error log: %ERROR_LOG% (File not found)
)
echo.

:: System resource information
echo SYSTEM RESOURCES:

:: Memory information
for /f "tokens=2" %%a in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set "TOTAL_MEM=%%a"
for /f "tokens=2" %%a in ('wmic OS get FreePhysicalMemory /value ^| find "="') do set "FREE_MEM_KB=%%a"

if defined TOTAL_MEM if defined FREE_MEM_KB (
    set /a TOTAL_MEM_GB=!TOTAL_MEM!/1024/1024/1024
    set /a FREE_MEM_MB=!FREE_MEM_KB!/1024
    set /a USED_MEM_MB=(!TOTAL_MEM!/1024/1024) - !FREE_MEM_MB!
    echo    * Memory: !USED_MEM_MB! MB / !TOTAL_MEM_GB! GB in use
) else (
    echo    * Memory: Unable to retrieve information
)

:: Disk information (current drive)
for /f "tokens=3,4" %%a in ('dir /-c ^| find "bytes free"') do (
    echo    * Disk: %%a %%b free space
    goto :disk_found
)
for /f "tokens=3,4" %%a in ('dir /-c ^| find "available"') do (
    echo    * Disk: %%a %%b free space
    goto :disk_found
)
:disk_found

echo.

:: Application feature information
echo APPLICATION FEATURES:
echo    * Image Processing:
echo        - High-quality AI background removal (BiRefNet)
echo        - 2x/4x image upscaling
echo        - Sequential processing support
echo    * Video Processing:
echo        - Frame-by-frame background removal
echo        - Video upscaling
echo        - Real-time progress display
echo    * Supported Formats:
echo        - Images: PNG, JPG, JPEG, GIF, BMP, WebP
echo        - Videos: MP4, AVI, MOV, MKV
echo.

:: Useful commands
echo USEFUL COMMANDS:
echo    * Real-time log view: powershell "Get-Content %LOG_FILE% -Wait"
echo    * View error log: type %ERROR_LOG%
echo    * Stop server: stop.bat
echo    * Restart server: stop.bat ^&^& start.bat
echo.

pause 