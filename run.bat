@echo off
setlocal enabledelayedexpansion

echo ========================================
echo EdgeHD 2.0 - Full-Stack Platform
echo Starting Backend + Frontend Servers
echo ========================================
echo.

:: Check if conda environment exists
echo [1/3] Checking backend environment...
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
    echo ERROR: Conda not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

:: Check if Node.js dependencies are installed
echo [2/3] Checking frontend environment...
if not exist "frontend\node_modules" (
    echo ERROR: Frontend dependencies not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo ERROR: Root dependencies not found.
    echo    Please run install.bat first.
    pause
    exit /b 1
)

echo SUCCESS: Frontend environment ready.

:: Activate conda environment
echo [3/3] Activating environment and starting servers...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    pause
    exit /b 1
)

:: Set environment variables for project-local model storage
set "HF_HOME=%cd%\backend\models"
set "TRANSFORMERS_CACHE=%cd%\backend\models"
set "HUGGINGFACE_HUB_CACHE=%cd%\backend\models"

echo.
echo ENVIRONMENT CONFIGURED:
echo    * Backend: Python + PyTorch (conda: edgehd)
echo    * Frontend: Node.js + Next.js
echo    * AI Models: %cd%\backend\models
echo.

echo STARTING SERVERS:
echo    * Backend API: http://localhost:8080
echo    * Frontend UI: http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers.
echo.

:: Start both servers using concurrently
npm run dev

pause