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
    pause
    exit /b 1
)

:: Read PID
set /p PID=<"%PID_FILE%"

:: Check process
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ERROR: Process %PID% is not running.
    echo INFO: Cleaning up PID file...
    del "%PID_FILE%" 2>nul
    pause
    exit /b 1
)

echo SERVER INFO:
echo    * PID: %PID%
for /f "tokens=1" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do set "PROCESS_NAME=%%~a"
echo    * Process: !PROCESS_NAME!

:: Attempt graceful shutdown
echo INFO: Sending graceful shutdown signal...
taskkill /PID %PID% >nul 2>&1

:: Wait for graceful shutdown
set WAIT_TIME=10
for /L %%i in (1,1,%WAIT_TIME%) do (
    tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
    if errorlevel 1 (
        echo SUCCESS: Server shutdown gracefully.
        del "%PID_FILE%" 2>nul
        goto :log_cleanup
    )
    
    echo    Waiting... (%%i/%WAIT_TIME%s)
    timeout /t 1 >nul
)

echo.
echo WARNING: Graceful shutdown failed. Attempting force shutdown...

:: Attempt force shutdown
echo INFO: Sending force shutdown signal...
taskkill /F /PID %PID% >nul 2>&1

:: Verify force shutdown
timeout /t 2 >nul
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo SUCCESS: Server force shutdown completed.
    del "%PID_FILE%" 2>nul
    goto :log_cleanup
) else (
    echo ERROR: Failed to shutdown process.
    echo    You may need to shutdown manually: taskkill /F /PID %PID%
    echo    Or use Task Manager.
    pause
    exit /b 1
)

:log_cleanup
:: Log file cleanup option
echo.
echo LOG FILE CLEANUP:
if exist "%LOG_FILE%" (
    for %%F in ("%LOG_FILE%") do set "LOG_SIZE=%%~zF"
    set /a LOG_SIZE_KB=!LOG_SIZE!/1024
    echo    * Log file: %LOG_FILE% (!LOG_SIZE_KB! KB)
) else (
    echo    * Log file: %LOG_FILE% (File not found)
)

if exist "%ERROR_LOG%" (
    for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
    set /a ERROR_SIZE_KB=!ERROR_SIZE!/1024
    echo    * Error log: %ERROR_LOG% (!ERROR_SIZE_KB! KB)
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
)

echo.
echo SUCCESS: Server shutdown completed!
pause 