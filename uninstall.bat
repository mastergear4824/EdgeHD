@echo off
setlocal enabledelayedexpansion

echo ========================================
echo EdgeHD 2.0 Full-Stack Platform Removal
echo ========================================
echo.

REM Check current directory
if not exist "package.json" (
    echo ERROR: Please run this script from the EdgeHD project directory
    pause
    exit /b 1
)

REM User confirmation
echo WARNING: The following items will be removed:
echo.
echo BACKEND:
echo    - Conda environment (edgehd)
echo    - Python packages and dependencies
echo    - AI model files (backend\models\)
echo    - Backend uploads/downloads/temp folders
echo.
echo FRONTEND:
echo    - Node.js dependencies (frontend\node_modules\)
echo    - Frontend build files
echo    - Next.js cache
echo.
echo ROOT:
echo    - Root dependencies (node_modules\)
echo    - Log files and PID files
echo    - Environment variable settings
echo.
set /p "confirm=Are you sure you want to proceed? (y/N): "
if /i not "%confirm%"=="y" (
    echo INFO: Removal cancelled
    pause
    exit /b 1
)

echo.
echo Starting full-stack environment cleanup...

REM 1. Stop any running servers
echo [1/8] Stopping running servers...
if exist "backend.pid" (
    echo INFO: Stopping backend server...
    call stop.bat >nul 2>&1
) else if exist "app.pid" (
    echo INFO: Stopping legacy server...
    call stop.bat >nul 2>&1
)

REM 2. Remove Conda environment
echo [2/8] Removing backend environment...
where conda >nul 2>&1
if not errorlevel 1 (
    call conda info --envs | find "edgehd" >nul
    if not errorlevel 1 (
        echo INFO: Removing Conda environment 'edgehd'...
        call conda remove -n edgehd --all -y >nul 2>&1
        echo SUCCESS: Conda environment removed
    ) else (
        echo INFO: Conda environment 'edgehd' not found
    )
) else (
    echo INFO: Conda not found, skipping environment removal
)

REM 3. Remove backend files
echo [3/8] Cleaning backend files...
if exist "backend\models" (
    echo INFO: Removing AI model files...
    rmdir /s /q "backend\models" 2>nul
    echo SUCCESS: Model files removed
)

if exist "backend\uploads" (
    echo INFO: Cleaning backend upload files...
    for /f %%i in ('dir /b "backend\uploads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "backend\uploads\%%i" 2>nul
    )
    echo SUCCESS: Backend upload files cleaned
)

if exist "backend\downloads" (
    echo INFO: Cleaning backend download files...
    for /f %%i in ('dir /b "backend\downloads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "backend\downloads\%%i" 2>nul
    )
    echo SUCCESS: Backend download files cleaned
)

if exist "backend\temp" (
    echo INFO: Removing backend temporary files...
    rmdir /s /q "backend\temp" 2>nul
    echo SUCCESS: Backend temporary files removed
)

REM 4. Remove frontend files
echo [4/8] Cleaning frontend files...
if exist "frontend\node_modules" (
    echo INFO: Removing frontend dependencies...
    rmdir /s /q "frontend\node_modules" 2>nul
    echo SUCCESS: Frontend dependencies removed
)

if exist "frontend\.next" (
    echo INFO: Removing Next.js build cache...
    rmdir /s /q "frontend\.next" 2>nul
    echo SUCCESS: Next.js cache removed
)

if exist "frontend\out" (
    echo INFO: Removing Next.js build output...
    rmdir /s /q "frontend\out" 2>nul
    echo SUCCESS: Next.js build output removed
)

REM 5. Remove root dependencies
echo [5/8] Cleaning root dependencies...
if exist "node_modules" (
    echo INFO: Removing root dependencies...
    rmdir /s /q "node_modules" 2>nul
    echo SUCCESS: Root dependencies removed
)

if exist "package-lock.json" (
    echo INFO: Removing package lock file...
    del "package-lock.json" 2>nul
    echo SUCCESS: Package lock file removed
)

REM 6. Remove Python cache files
echo [6/8] Removing Python cache files...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
del /s /q "*.pyc" 2>nul
del /s /q "*.pyo" 2>nul
echo SUCCESS: Python cache files removed

REM 7. Remove log and PID files
echo [7/8] Removing log and PID files...
del "backend.pid" 2>nul
del "frontend.pid" 2>nul
del "app.pid" 2>nul
del "backend.log" 2>nul
del "frontend.log" 2>nul
del "backend_error.log" 2>nul
del "frontend_error.log" 2>nul
del "app.log" 2>nul
del "app_error.log" 2>nul
echo SUCCESS: Log and PID files removed

REM 8. Clean legacy files (if any)
echo [8/8] Cleaning legacy files...
if exist "uploads" (
    echo INFO: Cleaning legacy upload files...
    for /f %%i in ('dir /b "uploads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "uploads\%%i" 2>nul
    )
)

if exist "downloads" (
    echo INFO: Cleaning legacy download files...
    for /f %%i in ('dir /b "downloads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "downloads\%%i" 2>nul
    )
)

if exist "temp" (
    echo INFO: Removing legacy temporary files...
    rmdir /s /q "temp" 2>nul
)

if exist "models" (
    echo INFO: Removing legacy model files...
    rmdir /s /q "models" 2>nul
)

echo SUCCESS: Legacy files cleaned

REM Environment variable cleanup guidance
echo.
echo ENVIRONMENT VARIABLE CLEANUP:
echo    Please manually remove the following environment variables if they exist:
echo    - HF_HOME
echo    - TRANSFORMERS_CACHE
echo    - HUGGINGFACE_HUB_CACHE
echo.
echo    To remove them, you can either:
echo    1. Go to System Properties ^> Advanced ^> Environment Variables
echo    2. Or run these commands:
echo       setx HF_HOME ""
echo       setx TRANSFORMERS_CACHE ""
echo       setx HUGGINGFACE_HUB_CACHE ""

REM System package removal guidance (optional)
echo.
echo ADDITIONAL CLEANUP OPTIONS:
echo    To also remove system-installed packages:
echo.
echo    Python packages (if installed globally):
echo    pip uninstall -y flask flask-cors pillow torch torchvision transformers timm realesrgan opencv-python numpy
echo.
echo    Node.js global packages (if any):
echo    npm uninstall -g concurrently
echo.

echo SUCCESS: EdgeHD 2.0 Full-Stack Platform removal completed!
echo.
echo WHAT WAS REMOVED:
echo    * Backend: Python environment, AI models, dependencies
echo    * Frontend: Node.js dependencies, build files, cache
echo    * Root: Project dependencies, logs, temporary files
echo.
echo TO REINSTALL:
echo    install.bat
echo.
echo DIRECTORY STRUCTURE PRESERVED:
echo    * Source code files remain intact
echo    * Configuration files preserved
echo    * README and documentation kept
echo.

pause