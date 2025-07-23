@echo off
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo Stopping AI Image/Video Processing Tool...

:: Check PID file
if not exist "%PID_FILE%" (
    echo ERROR: No running server found.
    echo    PID file does not exist: %PID_FILE%
    echo.
    echo INFO: Checking for any running Python processes...
    
    :: Check if any python processes are listening on port 8080
    for /f "tokens=5" %%a in ('netstat -ano ^| find "8080" ^| find "LISTENING"') do (
        set "PORT_PID=%%a"
        echo WARNING: Found process !PORT_PID! using port 8080
        set /p KILL_CHOICE="Force kill this process? (y/N): "
        if /i "!KILL_CHOICE!"=="y" (
            echo INFO: Force killing process !PORT_PID!...
            taskkill /F /PID !PORT_PID! >nul 2>&1
            if not errorlevel 1 (
                echo SUCCESS: Process !PORT_PID! terminated.
            ) else (
                echo ERROR: Failed to kill process !PORT_PID!.
            )
        )
    )
    
    pause
    exit /b 1
)

:: Read PID
set /p PID=<"%PID_FILE%"

:: Remove any quotes or spaces from PID
set "PID=%PID:"=%"
set "PID=%PID: =%"

:: Validate PID is numeric
echo %PID%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo ERROR: Invalid PID in file: %PID%
    echo INFO: Cleaning up invalid PID file...
    del "%PID_FILE%" 2>nul
    pause
    exit /b 1
)

:: Check process
echo INFO: Checking process %PID%...
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ERROR: Process %PID% is not running.
    echo INFO: Cleaning up PID file...
    del "%PID_FILE%" 2>nul
    
    :: Check if port 8080 is still being used by another process
    netstat -ano | find "8080" | find "LISTENING" >nul 2>&1
    if not errorlevel 1 (
        echo WARNING: Port 8080 is still in use by another process.
        for /f "tokens=5" %%a in ('netstat -ano ^| find "8080" ^| find "LISTENING"') do (
            echo    Process using port 8080: %%a
        )
    )
    
    pause
    exit /b 1
)

echo SERVER INFO:
echo    * PID: %PID%
for /f "tokens=1,5" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do (
    set "PROCESS_NAME=%%~a"
    set "MEMORY_USAGE=%%~b"
)
echo    * Process: !PROCESS_NAME!
echo    * Memory: !MEMORY_USAGE!

:: Check if this process is actually using port 8080
netstat -ano | find "8080" | find "%PID%" >nul 2>&1
if not errorlevel 1 (
    echo    * Status: Active (using port 8080)
) else (
    echo    * Status: Running (not using port 8080 - may be starting/stopping)
)

:: Attempt graceful shutdown first
echo.
echo INFO: Attempting graceful shutdown...
taskkill /PID %PID% >nul 2>&1
if errorlevel 1 (
    echo WARNING: Graceful shutdown signal failed. Process may be unresponsive.
    goto :force_shutdown
)

:: Wait for graceful shutdown
set WAIT_TIME=10
echo INFO: Waiting for graceful shutdown (up to %WAIT_TIME% seconds)...
for /L %%i in (1,1,%WAIT_TIME%) do (
    tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
    if errorlevel 1 (
        echo SUCCESS: Server shutdown gracefully in %%i seconds.
        del "%PID_FILE%" 2>nul
        goto :log_cleanup
    )
    
    echo    Waiting... (%%i/%WAIT_TIME%s)
    timeout /t 1 >nul
)

:force_shutdown
echo.
echo WARNING: Graceful shutdown failed or timed out. Attempting force shutdown...

:: Attempt force shutdown
echo INFO: Sending force shutdown signal...
taskkill /F /PID %PID% >nul 2>&1
if errorlevel 1 (
    echo ERROR: Force shutdown command failed.
    echo    This may indicate insufficient permissions or the process is protected.
    echo    You may need to run as administrator.
    pause
    exit /b 1
)

:: Verify force shutdown
echo INFO: Verifying force shutdown...
timeout /t 2 >nul
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo SUCCESS: Server force shutdown completed.
    del "%PID_FILE%" 2>nul
    goto :log_cleanup
) else (
    echo ERROR: Failed to shutdown process after force kill.
    echo    Process %PID% is still running.
    echo.
    echo MANUAL INTERVENTION REQUIRED:
    echo    1. Try running as administrator: Right-click cmd.exe → "Run as administrator"
    echo    2. Manual force kill: taskkill /F /PID %PID%
    echo    3. Use Task Manager: Find python.exe process and end it
    echo    4. Restart computer if process is completely stuck
    pause
    exit /b 1
)

:log_cleanup
:: Verify port 8080 is freed
echo.
echo INFO: Verifying port 8080 is freed...
timeout /t 2 >nul
netstat -an | find "8080" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo SUCCESS: Port 8080 is now free.
) else (
    echo WARNING: Port 8080 is still in use by another process.
    echo    This may be normal if you have multiple instances or other applications using this port.
)

:: Log file cleanup option
echo.
echo LOG FILE CLEANUP:
if exist "%LOG_FILE%" (
    for %%F in ("%LOG_FILE%") do set "LOG_SIZE=%%~zF"
    if !LOG_SIZE! GTR 0 (
        set /a LOG_SIZE_KB=!LOG_SIZE!/1024
        echo    * Log file: %LOG_FILE% (!LOG_SIZE_KB! KB)
    ) else (
        echo    * Log file: %LOG_FILE% (Empty)
    )
) else (
    echo    * Log file: %LOG_FILE% (File not found)
)

if exist "%ERROR_LOG%" (
    for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
    if !ERROR_SIZE! GTR 0 (
        set /a ERROR_SIZE_KB=!ERROR_SIZE!/1024
        echo    * Error log: %ERROR_LOG% (!ERROR_SIZE_KB! KB)
        
        :: Show preview of error log if it has content
        echo.
        echo ERROR LOG PREVIEW (last 5 lines):
        powershell "Get-Content '%ERROR_LOG%' | Select-Object -Last 5"
    ) else (
        echo    * Error log: %ERROR_LOG% (Empty)
    )
) else (
    echo    * Error log: %ERROR_LOG% (File not found)
)

echo.
set /p CLEANUP_CHOICE="Delete log files? (y/N): "
if /i "!CLEANUP_CHOICE!"=="y" (
    if exist "%LOG_FILE%" del "%LOG_FILE%" 2>nul
    if exist "%ERROR_LOG%" del "%ERROR_LOG%" 2>nul
    echo INFO: Log files deleted.
) else (
    echo INFO: Log files preserved.
    if exist "%ERROR_LOG%" (
        for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
        if !ERROR_SIZE! GTR 0 (
            echo    You can review the error log: type %ERROR_LOG%
        )
    )
)

echo.
echo SUCCESS: Server shutdown completed!
echo.
echo INFO: To restart the server, run: start.bat
pause 