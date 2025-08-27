@echo off
setlocal enabledelayedexpansion

:: Set script variables
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
set "LOG_DIR=%SCRIPT_DIR%"

echo ========================================
echo EdgeHD 2.0 - Full-Stack Platform
echo Starting Development Servers
echo ========================================
echo.

:: Kill any existing processes first
echo [Pre-check] Cleaning up existing processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *app.py*" >nul 2>&1
taskkill /F /IM node.exe /FI "COMMANDLINE eq *next dev*" >nul 2>&1
taskkill /F /IM node.exe /FI "COMMANDLINE eq *npm run dev*" >nul 2>&1

:: Wait a moment for cleanup
timeout /t 2 >nul

:: Check if conda environment exists
echo [1/4] Checking backend environment...
where conda >nul 2>&1
if not errorlevel 1 (
    conda info --envs | findstr "edgehd" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: 'edgehd' conda environment not found.
        echo    Please run install.bat first.
        pause
        exit /b 1
    )
    echo SUCCESS: Conda environment 'edgehd' found.
) else (
    echo ERROR: Conda not found in PATH.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

:: Check if Node.js dependencies are installed
echo [2/4] Checking frontend environment...
if not exist "%FRONTEND_DIR%\node_modules" (
    echo ERROR: Frontend dependencies not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%node_modules" (
    echo ERROR: Root dependencies not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

echo SUCCESS: Frontend environment ready.

:: Check ports availability
echo [3/4] Checking ports availability...
netstat -an | find ":8080" | find "LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Port 8080 is already in use.
    echo    Please stop any running services on port 8080.
    pause
    exit /b 1
)

netstat -an | find ":3000" | find "LISTENING" >nul  
if not errorlevel 1 (
    echo ERROR: Port 3000 is already in use.
    echo    Please stop any running services on port 3000.
    pause
    exit /b 1
)

echo SUCCESS: Ports 8080 and 3000 are available.

:: Initialize conda for this session
echo [4/4] Initializing environment and starting servers...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    echo    Try running: conda init cmd.exe
    pause
    exit /b 1
)

:: Set environment variables for project-local model storage
set "HF_HOME=%SCRIPT_DIR%backend\models"
set "TRANSFORMERS_CACHE=%SCRIPT_DIR%backend\models"
set "HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%backend\models"

:: Create models directory if it doesn't exist
if not exist "%SCRIPT_DIR%backend\models" (
    mkdir "%SCRIPT_DIR%backend\models"
)

echo.
echo ENVIRONMENT CONFIGURED:
echo    * Backend: Python + PyTorch (conda: edgehd)
echo    * Frontend: Node.js + Next.js
echo    * AI Models: %SCRIPT_DIR%backend\models
echo    * Working Directory: %SCRIPT_DIR%
echo.

echo STARTING SERVERS:
echo    * Backend API: http://localhost:8080
echo    * Frontend UI: http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers.
echo.

:: Start both servers using concurrently with better error handling
if exist "%SCRIPT_DIR%package.json" (
    npm run dev
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to start development servers.
        echo    Check if all dependencies are installed correctly.
        pause
        exit /b 1
    )
) else (
    echo ERROR: package.json not found in root directory.
    echo    Please ensure you're running this from the EdgeHD root directory.
    pause
    exit /b 1
)

echo.
echo Development servers stopped.
pause