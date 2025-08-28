@echo off
setlocal enabledelayedexpansion

echo ========================================
echo EdgeHD 2.0 - AI Video/Image Processing Platform
echo Professional Timeline Editor + AI Processing
echo Full-Stack Installation (Backend + Frontend)
echo ========================================
echo.

:: Check if Node.js is installed
echo [1/4] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed.
    echo.
    echo Please install Node.js first:
    echo    * Download from: https://nodejs.org/
    echo    * Recommended: LTS version
    echo.
    echo After installation, restart terminal and run this script again.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('node --version 2^>nul') do echo SUCCESS: Node.js %%i found
)

:: Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm is not available.
    pause
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('npm --version 2^>nul') do echo SUCCESS: npm %%i found
)

echo.

:: Check if conda is available and get the actual path
echo [2/4] Checking Python/Conda installation...
for /f "tokens=*" %%i in ('where conda 2^>nul') do (
    echo SUCCESS: Conda found at %%i
    set "CONDA_PATH=%%i"
    goto :start_backend_installation
)

:: Check common Miniconda installation paths
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %USERPROFILE%\miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe"
    goto :start_backend_installation
)

if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %USERPROFILE%\Miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    goto :start_backend_installation
)

if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at C:\ProgramData\Miniconda3\Scripts\conda.exe
    set "CONDA_PATH=C:\ProgramData\Miniconda3\Scripts\conda.exe"
    goto :start_backend_installation
)

if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %LOCALAPPDATA%\miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
    goto :start_backend_installation
)

:: Check Anaconda paths as well
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    echo SUCCESS: Conda found at %USERPROFILE%\anaconda3\condabin\conda.bat
    set "CONDA_PATH=%USERPROFILE%\anaconda3\condabin\conda.bat"
    goto :start_backend_installation
)

:: Conda not found - offer to install
echo WARNING: Conda is not installed.
echo.
echo Would you like to automatically install Miniconda? (Recommended)
echo This will download and install Miniconda3 (~100MB)
echo.
set /p choice="Install Miniconda automatically? (Y/n): "
if /i "%choice%" neq "n" (
    call :install_miniconda
    if errorlevel 1 (
        echo ERROR: Failed to install Miniconda.
        echo Please install manually from: https://docs.conda.io/en/latest/miniconda.html
        pause
        exit /b 1
    )
    echo.
    echo Miniconda installation completed!
    echo Please run this script again to continue with the setup.
    pause
    exit /b 0
) else (
    echo Please install Conda manually:
    echo    * Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo    * Anaconda: https://www.anaconda.com/products/distribution
    echo.
    echo After installation, restart terminal and run this script again.
    pause
    exit /b 1
)

:start_backend_installation
echo INFO: Proceeding with backend installation using conda.

:: Create Conda environment
echo Creating Conda environment 'edgehd'...

:: Check if environment already exists
conda info --envs | findstr "edgehd" >nul 2>&1
if not errorlevel 1 (
    echo INFO: Environment 'edgehd' already exists. Skipping creation.
    goto :activate_env
)

:: Create environment using simple conda command - Use Python 3.11 for better compatibility
conda create -n edgehd python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create Conda environment.
    echo INFO: Please check if conda is properly installed and try again.
    pause
    exit /b 1
)

echo SUCCESS: Conda environment 'edgehd' created successfully.

:activate_env
:: Activate environment
echo Activating Conda environment...
call conda activate edgehd
if errorlevel 1 (
    echo WARNING: Failed to activate environment. Continuing with installation...
    echo NOTE: You may need to manually activate the environment later.
)

:: Install PyTorch 2.1.0 for Real-ESRGAN compatibility
echo.
echo ========================================
echo Installing PyTorch 2.1.0 (Real-ESRGAN v0.3.0 compatible)...
echo ========================================

:: Check if ARM64 Windows and handle specially
echo INFO: Checking system architecture...
wmic os get osarchitecture | findstr "ARM" >nul
if not errorlevel 1 (
    echo.
    echo ========================================
    echo ARM64 Windows Detected!
    echo ========================================
    echo PyTorch ARM64 Windows support requires:
    echo 1. Python 3.12 ARM64
    echo 2. Visual Studio Build Tools with C++
    echo 3. Rust toolchain
    echo.
    echo Please follow these steps:
    echo 1. Install Visual Studio Build Tools with "Desktop development with C++"
    echo 2. Install Rust: rustup-init.exe --default-toolchain stable --default-host aarch64-pc-windows-msvc
    echo 3. Create new environment: conda create -n edgehd-arm python=3.12 -y
    echo 4. Activate: conda activate edgehd-arm
    echo 5. Install PyTorch: pip install --extra-index-url https://download.pytorch.org/whl torch torchvision torchaudio
    echo.
    echo For more info: https://blogs.windows.com/windowsdeveloper/2025/04/23/pytorch-arm-native-builds-now-available-for-windows/
    echo ========================================
    pause
    exit /b 1
)

:: Install specific PyTorch versions for Real-ESRGAN compatibility
echo INFO: Installing PyTorch 2.1.0 for Real-ESRGAN v0.3.0 compatibility...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

:: Remove any existing PyTorch installations first
echo INFO: Removing any existing PyTorch installations...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

:: Install compatible PyTorch versions
echo INFO: Installing PyTorch 2.1.0, torchvision 0.16.0, torchaudio 2.1.0...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch 2.1.0.
    echo INFO: Trying alternative installation method...
    
    :: Try CUDA-specific version if available
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo INFO: NVIDIA GPU not found. Installing CPU version...
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo INFO: NVIDIA GPU detected. Installing CUDA version...
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    )
    
    if errorlevel 1 (
        echo ERROR: All PyTorch installation methods failed.
        echo Please check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo SUCCESS: PyTorch 2.1.0 installed successfully.

:: Install backend dependencies
echo.
echo ========================================
echo Installing backend dependencies...
echo ========================================
cd backend
if not exist requirements.txt (
    echo ERROR: Backend requirements.txt not found.
    echo Please ensure you're running this from the EdgeHD root directory.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install backend dependencies.
    pause
    exit /b 1
)

echo SUCCESS: Backend dependencies installed.
cd ..

:: Install frontend dependencies
echo.
echo ========================================
echo [3/4] Installing frontend dependencies...
echo ========================================
cd frontend
if not exist package.json (
    echo ERROR: Frontend package.json not found.
    echo Please ensure you're running this from the EdgeHD root directory.
    pause
    exit /b 1
)

echo INFO: Installing Node.js packages...
npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies.
    pause
    exit /b 1
)

echo SUCCESS: Frontend dependencies installed.
cd ..

:: Install root dependencies (concurrently)
echo.
echo ========================================
echo [4/4] Installing root dependencies...
echo ========================================
echo INFO: Installing concurrently for running both servers...
npm install
if errorlevel 1 (
    echo ERROR: Failed to install root dependencies.
    pause
    exit /b 1
)

echo SUCCESS: Root dependencies installed.

:: Test package compatibility
echo.
echo ========================================
echo Testing package compatibility...
echo ========================================
call conda activate edgehd
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch import failed.
    pause
    exit /b 1
)

python -c "import flask; print('Flask version:', flask.__version__)" 2>nul
if errorlevel 1 (
    echo ERROR: Flask import failed.
    pause
    exit /b 1
)

echo SUCCESS: All packages are compatible!

:: Configure project-local model storage
echo.
echo ========================================
echo Configuring AI models...
echo ========================================

:: Set environment variables for project-local model storage
echo INFO: Setting up project-local model storage...
set HF_HOME=%cd%\backend\models
set TRANSFORMERS_CACHE=%cd%\backend\models
set HUGGINGFACE_HUB_CACHE=%cd%\backend\models

:: Test environment variables
echo INFO: Project-local model storage configured:
echo    HF_HOME=%HF_HOME%
echo    TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo    HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%

echo    All AI models will be stored in backend\models\ directory
echo    AI models downloaded on first run:
echo       * BiRefNet background removal model (~424MB)
echo       * Real-ESRGAN General v3 4x upscaling model (~17MB)
echo    Models are managed independently per project
echo SUCCESS: AI model configuration completed.

echo.
echo ========================================
echo    INSTALLATION COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo EdgeHD 2.0 Full-Stack Platform is ready!
echo.
echo ARCHITECTURE:
echo    * Backend:  Flask API server (Python 3.11 + PyTorch 2.1.0)
echo    * Frontend: Next.js + shadcn/ui (Node.js + React)
echo    * Timeline: Professional video editing interface
echo    * Database: File-based storage
echo    * AI Models: BiRefNet + Real-ESRGAN
echo.
echo KEY FEATURES:
echo    * Professional Timeline Editor with frame-level precision
echo    * Draggable playhead for instant navigation
echo    * Multi-track video/audio editing
echo    * AI-powered background removal and upscaling
echo    * Responsive design with dynamic layout
echo.
echo HOW TO RUN:
echo    npm run dev          - Start both servers (development)
echo    npm run dev:backend  - Start backend only (http://localhost:8080)
echo    npm run dev:frontend - Start frontend only (http://localhost:3000)
echo.
echo PRODUCTION:
echo    npm run build        - Build frontend for production
echo    npm run start        - Start both servers (production)
echo.
echo ACCESS URLS:
echo    * Frontend UI: http://localhost:3000
echo    * Backend API: http://localhost:8080
echo.
echo See README.md for detailed usage instructions.

pause
goto :eof

:: Miniconda installation function
:install_miniconda
echo.
echo Downloading Miniconda3 installer...
echo This may take a few minutes depending on your internet connection.

:: Detect architecture
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set "INSTALLER_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    set "INSTALLER_NAME=Miniconda3-latest-Windows-x86_64.exe"
) else (
    set "INSTALLER_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe"
    set "INSTALLER_NAME=Miniconda3-latest-Windows-x86.exe"
)

:: Download using PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%TEMP%\%INSTALLER_NAME%'}"
if errorlevel 1 (
    echo ERROR: Failed to download Miniconda installer.
    echo Please check your internet connection and try again.
    exit /b 1
)

echo.
echo Installing Miniconda3...
echo This will install to: %USERPROFILE%\miniconda3
echo.

:: Run installer silently with PATH addition
"%TEMP%\%INSTALLER_NAME%" /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\miniconda3
if errorlevel 1 (
    echo ERROR: Miniconda installation failed.
    exit /b 1
)

:: Clean up installer
del "%TEMP%\%INSTALLER_NAME%" >nul 2>&1

:: Add to PATH for current session
set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3\Library\bin;%PATH%"

:: Initialize conda for both cmd and PowerShell
call "%USERPROFILE%\miniconda3\Scripts\conda.exe" init cmd.exe >nul 2>&1
call "%USERPROFILE%\miniconda3\Scripts\conda.exe" init powershell >nul 2>&1

:: Refresh environment variables for current session
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"

echo.
echo Miniconda3 has been installed successfully!
echo Location: %USERPROFILE%\miniconda3
echo.
echo IMPORTANT: Please close this terminal and open a new one for conda to work properly.

exit /b 0