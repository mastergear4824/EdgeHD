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
echo EdgeHD 2.0 - Full-Stack Platform
echo Stopping Backend + Frontend Servers
echo ========================================
echo.

set "BACKEND_STOPPED=false"
set "FRONTEND_STOPPED=false"

:: Function to safely kill process
call :kill_process_safe "Backend" "%BACKEND_PID_FILE%" BACKEND_STOPPED
call :kill_process_safe "Frontend" "%FRONTEND_PID_FILE%" FRONTEND_STOPPED

:: Kill any remaining EdgeHD processes
echo [Additional Cleanup] Killing any remaining EdgeHD processes...
taskkill /F /IM python.exe /FI "COMMANDLINE eq *app.py*" >nul 2>&1
taskkill /F /IM node.exe /FI "COMMANDLINE eq *npm start*" >nul 2>&1
taskkill /F /IM node.exe /FI "COMMANDLINE eq *next start*" >nul 2>&1

:: Verify ports are freed
echo.
echo [Port Verification] Checking if ports are freed...
timeout /t 3 >nul

call :check_port_free 8080 "Backend"
call :check_port_free 3000 "Frontend"

:: Show log file information
call :show_log_info

:: Final status
echo.
if "%BACKEND_STOPPED%"=="true" if "%FRONTEND_STOPPED%"=="true" (
    echo SUCCESS: EdgeHD 2.0 Full-Stack Platform stopped successfully!
) else (
    echo WARNING: Some servers may not have stopped properly.
    echo Please check the processes manually or restart your computer if needed.
)

echo.
echo INFO: To restart the platform, run: start.bat
echo INFO: For development mode, run: run.bat

pause
goto :eof

:: Function to safely kill a process
:kill_process_safe
set "service_name=%~1"
set "pid_file=%~2"
set "stopped_var=%~3"

echo [%service_name%] Stopping %service_name% server...

if not exist "%pid_file%" (
    echo INFO: %service_name% PID file not found. %service_name% may not be running.
    set "%stopped_var%=true"
    goto :eof
)

set /p SERVICE_PID=<"%pid_file%"
set "SERVICE_PID=!SERVICE_PID:"=!"
set "SERVICE_PID=!SERVICE_PID: =!"

:: Validate PID is numeric
echo !SERVICE_PID!| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo ERROR: Invalid %service_name% PID: !SERVICE_PID!
    del "%pid_file%" 2>nul
    set "%stopped_var%=true"
    goto :eof
)

:: Check if process exists
tasklist /FI "PID eq !SERVICE_PID!" 2>nul | find "!SERVICE_PID!" >nul
if errorlevel 1 (
    echo INFO: %service_name% process !SERVICE_PID! not found. Cleaning up PID file.
    del "%pid_file%" 2>nul
    set "%stopped_var%=true"
    goto :eof
)

echo INFO: Stopping %service_name% process !SERVICE_PID!...

:: Attempt graceful shutdown (SIGTERM equivalent)
taskkill /PID !SERVICE_PID! >nul 2>&1
if not errorlevel 1 (
    :: Wait for graceful shutdown
    echo INFO: Waiting for graceful shutdown...
    timeout /t 5 >nul
    
    :: Check if process is still running
    tasklist /FI "PID eq !SERVICE_PID!" 2>nul | find "!SERVICE_PID!" >nul
    if errorlevel 1 (
        echo SUCCESS: %service_name% stopped gracefully.
        del "%pid_file%" 2>nul
        set "%stopped_var%=true"
        goto :eof
    )
)

:: Force shutdown if graceful failed
echo WARNING: Graceful shutdown failed or timed out. Force stopping...
taskkill /F /PID !SERVICE_PID! >nul 2>&1
if not errorlevel 1 (
    timeout /t 2 >nul
    tasklist /FI "PID eq !SERVICE_PID!" 2>nul | find "!SERVICE_PID!" >nul
    if errorlevel 1 (
        echo SUCCESS: %service_name% force stopped.
        del "%pid_file%" 2>nul
        set "%stopped_var%=true"
    ) else (
        echo ERROR: Failed to stop %service_name% process !SERVICE_PID!.
    )
) else (
    echo ERROR: Failed to force stop %service_name% process !SERVICE_PID!.
)
goto :eof

:: Function to check if port is free
:check_port_free
set "port=%~1"
set "service=%~2"

netstat -an | find ":%port%" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo SUCCESS: Port %port% (%service%) is now free.
) else (
    echo WARNING: Port %port% is still in use by another process.
    echo INFO: Finding process using port %port%...
    for /f "tokens=5" %%a in ('netstat -ano ^| find ":%port%" ^| find "LISTENING"') do (
        echo    Process PID: %%a
        for /f "tokens=1" %%b in ('tasklist /FI "PID eq %%a" /FO CSV /NH 2^>nul') do (
            echo    Process Name: %%~b
        )
    )
)
goto :eof

:: Function to show log information
:show_log_info
echo.
echo [Log Files] Log file information:

if exist "%BACKEND_LOG%" (
    for %%F in ("%BACKEND_LOG%") do set "BACKEND_LOG_SIZE=%%~zF"
    if !BACKEND_LOG_SIZE! GTR 0 (
        set /a BACKEND_LOG_KB=!BACKEND_LOG_SIZE!/1024
        echo    * Backend log: %BACKEND_LOG% (!BACKEND_LOG_KB! KB)
    ) else (
        echo    * Backend log: %BACKEND_LOG% (Empty)
    )
) else (
    echo    * Backend log: %BACKEND_LOG% (Not found)
)

if exist "%FRONTEND_LOG%" (
    for %%F in ("%FRONTEND_LOG%") do set "FRONTEND_LOG_SIZE=%%~zF"
    if !FRONTEND_LOG_SIZE! GTR 0 (
        set /a FRONTEND_LOG_KB=!FRONTEND_LOG_SIZE!/1024
        echo    * Frontend log: %FRONTEND_LOG% (!FRONTEND_LOG_KB! KB)
    ) else (
        echo    * Frontend log: %FRONTEND_LOG% (Empty)
    )
) else (
    echo    * Frontend log: %FRONTEND_LOG% (Not found)
)

:: Show error logs if they have content
if exist "%BACKEND_ERROR_LOG%" (
    for %%F in ("%BACKEND_ERROR_LOG%") do set "BACKEND_ERROR_SIZE=%%~zF"
    if !BACKEND_ERROR_SIZE! GTR 10 (
        set /a BACKEND_ERROR_KB=!BACKEND_ERROR_SIZE!/1024
        echo    * Backend errors: %BACKEND_ERROR_LOG% (!BACKEND_ERROR_KB! KB)
        echo.
        echo WARNING: Backend error log contains content!
        echo ===== Backend Error Log Preview (last 3 lines) =====
        powershell "Get-Content '%BACKEND_ERROR_LOG%' | Select-Object -Last 3"
        echo =====================================================
    )
)

if exist "%FRONTEND_ERROR_LOG%" (
    for %%F in ("%FRONTEND_ERROR_LOG%") do set "FRONTEND_ERROR_SIZE=%%~zF"
    if !FRONTEND_ERROR_SIZE! GTR 10 (
        set /a FRONTEND_ERROR_KB=!FRONTEND_ERROR_SIZE!/1024
        echo    * Frontend errors: %FRONTEND_ERROR_LOG% (!FRONTEND_ERROR_KB! KB)
        echo.
        echo WARNING: Frontend error log contains content!
        echo ===== Frontend Error Log Preview (last 3 lines) =====
        powershell "Get-Content '%FRONTEND_ERROR_LOG%' | Select-Object -Last 3"
        echo ======================================================
    )
)

echo.
set /p CLEANUP_CHOICE="Delete log files? (y/N): "
if /i "!CLEANUP_CHOICE!"=="y" (
    if exist "%BACKEND_LOG%" del "%BACKEND_LOG%" 2>nul
    if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%" 2>nul
    if exist "%BACKEND_ERROR_LOG%" del "%BACKEND_ERROR_LOG%" 2>nul
    if exist "%FRONTEND_ERROR_LOG%" del "%FRONTEND_ERROR_LOG%" 2>nul
    echo INFO: Log files deleted.
) else (
    echo INFO: Log files preserved.
)
goto :eof