@echo off
setlocal enabledelayedexpansion

echo Starting EdgeHD AI Image Processing Application removal...

REM Check current directory
if not exist "app.py" (
    echo ERROR: Please run this script from the EdgeHD project directory
    pause
    exit /b 1
)

REM User confirmation
echo.
echo WARNING: The following items will be removed:
echo    - Conda environment (edgehd)
echo    - Installed Python packages
echo    - AI model files (models\)
echo    - Uploaded files (uploads\)
echo    - Downloaded files (downloads\)
echo    - Environment variable settings
echo.
set /p "confirm=Are you sure you want to proceed? (y/N): "
if /i not "%confirm%"=="y" (
    echo INFO: Removal cancelled
    pause
    exit /b 1
)

echo.
echo Starting environment cleanup...

REM 1. Remove Conda environment
echo INFO: Checking for Conda environment 'edgehd'...
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

REM 2. Remove model files
if exist "models" (
    echo INFO: Removing AI model files...
    rmdir /s /q "models" 2>nul
    echo SUCCESS: Model files removed
) else (
    echo INFO: No model files found
)

REM 3. Clean upload files
if exist "uploads" (
    echo INFO: Cleaning upload files...
    for /f %%i in ('dir /b "uploads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "uploads\%%i" 2>nul
    )
    echo SUCCESS: Upload files cleaned
)

REM 4. Clean download files
if exist "downloads" (
    echo INFO: Cleaning download files...
    for /f %%i in ('dir /b "downloads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "downloads\%%i" 2>nul
    )
    echo SUCCESS: Download files cleaned
)

REM 5. Remove temporary files
if exist "temp" (
    echo INFO: Removing temporary files...
    rmdir /s /q "temp" 2>nul
    echo SUCCESS: Temporary files removed
)

REM 6. Remove Python cache files
echo INFO: Removing Python cache files...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
del /s /q "*.pyc" 2>nul
del /s /q "*.pyo" 2>nul
echo SUCCESS: Cache files removed

REM 7. Remove log files
if exist "app.log" del "app.log" 2>nul
if exist "app_error.log" del "app_error.log" 2>nul
if exist "app.pid" del "app.pid" 2>nul
echo SUCCESS: Log files removed

REM 8. Environment variable cleanup guidance
echo.
echo ENVIRONMENT VARIABLE CLEANUP:
echo    Please manually remove the following environment variables:
echo    - HF_HOME
echo    - TRANSFORMERS_CACHE
echo.
echo    To remove them, you can either:
echo    1. Go to System Properties ^> Advanced ^> Environment Variables
echo    2. Or run these commands:
echo       setx HF_HOME ""
echo       setx TRANSFORMERS_CACHE ""

REM 9. System package removal guidance (optional)
echo.
echo ADDITIONAL CLEANUP OPTIONS:
echo    To also remove system-installed Python packages:
echo    pip uninstall -y flask flask-cors pillow torch torchvision transformers timm realesrgan opencv-python numpy
echo.

echo SUCCESS: EdgeHD removal completed!
echo.
echo TO REINSTALL:
echo    install.bat
echo.

pause 