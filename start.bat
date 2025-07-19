@echo off
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo Starting AI Image/Video Processing Tool...

:: Check if already running
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /FI "PID eq !PID!" 2>nul | find "!PID!" >nul
    if not errorlevel 1 (
        echo ERROR: Already running (PID: !PID!)
        echo    To stop: stop.bat
        pause
        exit /b 1
    ) else (
        echo INFO: Cleaning up old PID file...
        del "%PID_FILE%" 2>nul
    )
)

:: Check Conda installation
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not installed.
    echo    Please install it first and try again.
    pause
    exit /b 1
)

:: Check Conda environment
echo INFO: Checking Conda environment 'edgehd'...
call conda info --envs | find "edgehd" >nul
if errorlevel 1 (
    echo ERROR: 'edgehd' environment not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

echo INFO: Activating Conda environment...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate Conda environment.
    pause
    exit /b 1
)

:: Set environment variables for project-local model storage
set HF_HOME=%cd%\models
set TRANSFORMERS_CACHE=%cd%\models
echo INFO: AI models will be managed in project directory (%cd%\models)

:: Initialize log files
echo INFO: Initializing log files...
echo. > "%LOG_FILE%"
echo. > "%ERROR_LOG%"

:: Start server in background
echo INFO: Starting server in background...

:: Run Python script in background
start /B python app.py 1>"%LOG_FILE%" 2>"%ERROR_LOG%"

:: Find and save process PID
timeout /t 2 >nul
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| find "python.exe"') do (
    set "PID=%%~a"
    goto :found_pid
)

:found_pid
if not defined PID (
    echo ERROR: Failed to start server.
    echo    Check error log: type %ERROR_LOG%
    pause
    exit /b 1
)

echo %PID% > "%PID_FILE%"

:: Wait for server initialization
echo INFO: Waiting for server initialization...
timeout /t 3 >nul

:: Verify process is running
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if not errorlevel 1 (
    echo.
    echo SUCCESS: AI Image/Video Processing Tool started successfully!
    echo.
    echo SERVER INFO:
    echo    * PID: %PID%
    echo    * Log file: %LOG_FILE%
    echo    * Error log: %ERROR_LOG%
    echo.
    echo ACCESS URLs:
    echo    * Local: http://localhost:8080
    echo    * Network: http://^<computer-ip^>:8080
    echo.
    echo USEFUL COMMANDS:
    echo    * Check status: status.bat
    echo    * View logs: type %LOG_FILE%
    echo    * Stop server: stop.bat
    echo.
    echo FEATURES:
    echo    * Image processing: Background removal, 2x/4x upscaling
    echo    * Video processing: Frame-by-frame processing + upscaling
    echo    * Real-time progress display
    echo.
) else (
    echo ERROR: Failed to start server.
    echo    Check error log: type %ERROR_LOG%
    if exist "%PID_FILE%" del "%PID_FILE%"
    pause
    exit /b 1
)

pause 