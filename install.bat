@echo off
setlocal enabledelayedexpansion

echo Starting AI Image/Video Processing Tool Installation...
echo.

:: Check if conda is available and get the actual path
for /f "tokens=*" %%i in ('where conda 2^>nul') do (
    echo SUCCESS: Conda found at %%i
    set "CONDA_PATH=%%i"
    goto :start_installation
)

:: Check common Miniconda installation paths
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %USERPROFILE%\miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe"
    goto :start_installation
)

if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %USERPROFILE%\Miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
    goto :start_installation
)

if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at C:\ProgramData\Miniconda3\Scripts\conda.exe
    set "CONDA_PATH=C:\ProgramData\Miniconda3\Scripts\conda.exe"
    goto :start_installation
)

if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %LOCALAPPDATA%\miniconda3\Scripts\conda.exe
    set "CONDA_PATH=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
    goto :start_installation
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

:start_installation
echo INFO: Proceeding with AI tool installation using conda.

:: Create Conda environment
echo Creating Conda environment 'edgehd'...

:: Check if environment already exists
conda info --envs | findstr "edgehd" >nul 2>&1
if not errorlevel 1 (
    echo INFO: Environment 'edgehd' already exists. Skipping creation.
    goto :activate_env
)

:: Create environment using simple conda command
conda create -n edgehd python=3.10 -y
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

:: Install latest PyTorch (compatible with Real-ESRGAN)
echo.
echo ========================================
echo Installing latest PyTorch (Real-ESRGAN compatible)...
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

:: Direct activation and pip installation (most reliable method)
echo INFO: Installing PyTorch using direct conda activation...

call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    pause
    exit /b 1
)

echo INFO: Trying default PyPI first (most compatible)...
pip install torch torchvision torchaudio
if errorlevel 1 (
    echo INFO: Default PyPI failed. Trying GPU-specific installation...
    
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo INFO: NVIDIA GPU not found. Trying CPU-specific index...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo INFO: NVIDIA GPU detected. Trying CUDA-specific index...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    
    if errorlevel 1 (
        echo ERROR: All PyTorch installation methods failed.
        echo.
        echo ========================================
        echo IMPORTANT: Your system may not be compatible
        echo ========================================
        echo Possible issues:
        echo 1. 32-bit Windows (PyTorch requires 64-bit)
        echo 2. Very old Windows version
        echo 3. Network connectivity issues
        echo.
        echo Manual troubleshooting:
        echo 1. Check if you have 64-bit Windows:
        echo    wmic os get osarchitecture
        echo.
        echo 2. If 64-bit, try in new CMD:
        echo    conda activate edgehd
        echo    pip install torch torchvision torchaudio
        echo.
        echo 3. Check Python version:
        echo    python --version
        echo ========================================
        pause
        exit /b 1
    )
)

echo SUCCESS: PyTorch installed successfully.

:: Install latest transformers version
echo.
echo ========================================
echo Installing latest transformers (Real-ESRGAN compatible)...
echo ========================================
conda run -n edgehd pip install transformers
if errorlevel 1 (
    echo ERROR: Failed to install transformers.
    echo INFO: Trying alternative installation method...
    conda activate edgehd
    pip install transformers
    if errorlevel 1 (
        echo ERROR: All transformers installation methods failed.
        pause
        exit /b 1
    )
)

echo SUCCESS: transformers installed successfully.

:: Create project directory structure
echo.
echo ========================================
echo Creating project directory structure...
echo ========================================
if not exist uploads mkdir uploads
if not exist downloads mkdir downloads
if not exist temp mkdir temp
if not exist models mkdir models
if not exist models\hub mkdir models\hub
echo SUCCESS: Directory structure created.

:: Install required packages from requirements.txt (skip PyTorch/transformers)
echo.
echo ========================================
echo Installing other required packages...
echo ========================================
echo INFO: Skipping PyTorch and transformers (already installed with specific versions)
conda run -n edgehd pip install Flask==3.0.0 Flask-CORS==4.0.0 werkzeug==3.0.1
conda run -n edgehd pip install "numpy>=1.24.0,<2.0.0" Pillow>=10.0.0 opencv-python>=4.8.0 requests>=2.31.0
if errorlevel 1 (
    echo ERROR: Failed to install basic packages.
    pause
    exit /b 1
)
echo SUCCESS: Basic packages installed.

:: Install AI model dependencies
echo.
echo ========================================
echo Installing AI model dependencies...
echo ========================================
conda run -n edgehd pip install einops>=0.6.0 kornia>=0.7.0 timm>=0.9.0 realesrgan==0.3.0
if errorlevel 1 (
    echo ERROR: Failed to install AI model dependencies.
    pause
    exit /b 1
)
echo SUCCESS: AI model dependencies installed.

:: Configure project-local model storage
echo.
echo ========================================
echo Configuring AI models...
echo ========================================
echo    All AI models will be stored in project models\ directory
echo    AI models downloaded on first run:
echo       * BiRefNet background removal model (~424MB)
echo       * Real-ESRGAN General v3 4x upscaling model (~17MB)
echo       * v0.3.0 supports only 4x (no 2x dedicated model)
echo    Models are managed independently per project
echo SUCCESS: AI model configuration completed.

echo.
echo ========================================
echo    INSTALLATION COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo How to run:
echo    start.bat          - Start background server
echo    python app.py      - Start development server (foreground)
echo.
echo Server URL: http://localhost:8080
echo.
echo See README.md for detailed usage instructions.

:: Create run script
echo Creating run script...
(
echo @echo off
echo setlocal enabledelayedexpansion
echo echo Starting AI Image/Video Processing Tool...
echo.
echo :: Find conda installation
echo where conda ^>nul 2^>^&1
echo if not errorlevel 1 ^(
echo     call conda activate edgehd
echo ^) else ^(
echo     if exist "%%USERPROFILE%%\miniconda3\Scripts\activate.bat" ^(
echo         call "%%USERPROFILE%%\miniconda3\Scripts\activate.bat" edgehd
echo     ^) else ^(
echo         echo ERROR: Conda not found. Please run install.bat first.
echo         pause
echo         exit /b 1
echo     ^)
echo ^)
echo.
echo :: Set environment variables for project-local model storage
echo set HF_HOME=%%cd%%\models
echo set TRANSFORMERS_CACHE=%%cd%%\models
echo.
echo python app.py
echo pause
) > run.bat

echo SUCCESS: Run script 'run.bat' created.
echo    You can now simply double-click 'run.bat' to start the application.
echo.
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