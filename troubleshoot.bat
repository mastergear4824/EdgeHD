@echo off
setlocal enabledelayedexpansion

echo ========================================
echo EdgeHD 2.0 - Troubleshooting Script
echo ========================================
echo.

echo [1/6] Checking system requirements...
echo.

:: Check Windows version
echo Checking Windows version...
ver
echo.

:: Check available disk space
echo Checking available disk space...
for /f "tokens=3" %%i in ('dir /-c 2^>nul ^| find "bytes free"') do set "FREE_SPACE=%%i"
echo Available disk space: !FREE_SPACE! bytes
if !FREE_SPACE! LSS 5000000000 (
    echo WARNING: Less than 5GB free space. Installation may fail.
) else (
    echo SUCCESS: Sufficient disk space available.
)
echo.

:: Check Node.js
echo Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found
    echo Please install Node.js from: https://nodejs.org/
) else (
    for /f "tokens=*" %%i in ('node --version 2^>nul') do echo SUCCESS: Node.js %%i found
)
echo.

:: Check npm
echo Checking npm...
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found
) else (
    for /f "tokens=*" %%i in ('npm --version 2^>nul') do echo SUCCESS: npm %%i found
)
echo.

echo [2/6] Checking Python/Conda installation...
echo.

:: Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
) else (
    for /f "tokens=*" %%i in ('python --version 2^>nul') do echo SUCCESS: Python %%i found
)
echo.

:: Check conda
echo Checking conda...
conda --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found
    echo.
    echo Common conda locations to check:
    echo   %USERPROFILE%\miniconda3\Scripts\conda.exe
    echo   %USERPROFILE%\Miniconda3\Scripts\conda.exe
    echo   C:\ProgramData\Miniconda3\Scripts\conda.exe
    echo   %LOCALAPPDATA%\miniconda3\Scripts\conda.exe
    echo   %USERPROFILE%\anaconda3\condabin\conda.bat
    echo.
    echo If conda is installed but not found, try:
    echo   1. Restart your terminal
    echo   2. Add conda to PATH manually
    echo   3. Run: conda init
) else (
    for /f "tokens=*" %%i in ('conda --version 2^>nul') do echo SUCCESS: conda %%i found
)
echo.

echo [3/6] Checking project structure...
echo.

:: Check if we're in the right directory
if not exist "backend\requirements.txt" (
    echo ERROR: backend\requirements.txt not found
    echo Please run this script from the EdgeHD root directory
    pause
    exit /b 1
) else (
    echo SUCCESS: Backend requirements.txt found
)

if not exist "frontend\package.json" (
    echo ERROR: frontend\package.json not found
    echo Please run this script from the EdgeHD root directory
    pause
    exit /b 1
) else (
    echo SUCCESS: Frontend package.json found
)

if not exist "package.json" (
    echo ERROR: Root package.json not found
    echo Please run this script from the EdgeHD root directory
    pause
    exit /b 1
) else (
    echo SUCCESS: Root package.json found
)
echo.

echo [4/6] Checking network connectivity...
echo.

:: Test internet connection
echo Testing internet connection...
ping -n 1 8.8.8.8 >nul 2>&1
if errorlevel 1 (
    echo ERROR: No internet connection
    echo Please check your network connection
) else (
    echo SUCCESS: Internet connection available
)

:: Test PyPI
echo Testing PyPI access...
python -c "import urllib.request; urllib.request.urlopen('https://pypi.org')" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Cannot access PyPI
    echo This may cause package installation issues
) else (
    echo SUCCESS: PyPI access available
)

:: Test npm registry
echo Testing npm registry...
npm ping >nul 2>&1
if errorlevel 1 (
    echo WARNING: Cannot access npm registry
    echo This may cause package installation issues
) else (
    echo SUCCESS: npm registry access available
)
echo.

echo [5/6] Checking existing installations...
echo.

:: Check if conda environment exists
conda info --envs | findstr "edgehd" >nul 2>&1
if errorlevel 1 (
    echo INFO: edgehd conda environment not found
) else (
    echo SUCCESS: edgehd conda environment exists
)

:: Check if node_modules exist
if exist "frontend\node_modules" (
    echo SUCCESS: Frontend node_modules found
) else (
    echo INFO: Frontend node_modules not found
)

if exist "node_modules" (
    echo SUCCESS: Root node_modules found
) else (
    echo INFO: Root node_modules not found
)
echo.

echo [6/6] Common fixes...
echo.

echo Would you like to try automatic fixes? (Y/n)
set /p choice="Choice: "
if /i "!choice!" neq "n" (
    echo.
    echo Applying fixes...
    
    :: Clear npm cache
    echo Clearing npm cache...
    npm cache clean --force >nul 2>&1
    
    :: Clear pip cache
    echo Clearing pip cache...
    pip cache purge >nul 2>&1
    
    :: Remove existing node_modules
    echo Removing existing node_modules...
    if exist "frontend\node_modules" rmdir /s /q "frontend\node_modules" >nul 2>&1
    if exist "node_modules" rmdir /s /q "node_modules" >nul 2>&1
    
    :: Remove existing conda environment
    echo Removing existing conda environment...
    conda env remove -n edgehd -y >nul 2>&1
    
    echo.
    echo Fixes applied. Please run install.bat again.
) else (
    echo Skipping automatic fixes.
)

echo.
echo ========================================
echo Troubleshooting completed!
echo ========================================
echo.
echo If you're still having issues:
echo 1. Check the error messages above
echo 2. Ensure you have sufficient disk space
echo 3. Try running as administrator
echo 4. Check your antivirus isn't blocking downloads
echo 5. Try using a different network connection
echo.
echo For more help, check the README.md file.
pause
