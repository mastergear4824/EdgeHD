@echo off
setlocal enabledelayedexpansion

echo Starting AI Image/Video Processing Tool Installation...
echo.

:: Check Conda installation
where conda >nul 2>&1
if errorlevel 1 (
    echo WARNING: Conda is not installed.
    echo.
    echo Would you like to automatically install Miniconda? (Recommended)
    echo This will download and install Miniconda3 (~100MB)
    echo.
    set /p choice="Install Miniconda automatically? (Y/n): "
    if /i "!choice!" neq "n" (
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
)

echo SUCCESS: Conda is installed.

:: Create Conda environment
echo Creating Conda environment 'edgehd'...
call conda create -n edgehd python=3.10 -y
if errorlevel 1 (
    echo ERROR: Failed to create Conda environment.
    pause
    exit /b 1
)

:: Activate environment
echo Activating Conda environment...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate Conda environment.
    pause
    exit /b 1
)

:: Detect GPU and install PyTorch 2.1.0 (Real-ESRGAN v0.3.0 compatibility)
echo Installing PyTorch 2.1.0 (Real-ESRGAN v0.3.0 compatible)...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo INFO: NVIDIA GPU not found. Installing CPU version PyTorch 2.1.0.
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
) else (
    echo INFO: NVIDIA GPU detected. Installing CUDA version PyTorch 2.1.0.
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
)

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    pause
    exit /b 1
)

:: Install compatible transformers version
echo Installing transformers 4.35.0 (Real-ESRGAN v0.3.0 compatible)...
pip install transformers==4.35.0
if errorlevel 1 (
    echo ERROR: Failed to install transformers.
    pause
    exit /b 1
)

:: Create project directory structure
echo Creating project directory structure...
if not exist uploads mkdir uploads
if not exist downloads mkdir downloads
if not exist temp mkdir temp
if not exist models mkdir models
if not exist models\hub mkdir models\hub

:: Install required packages
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages.
    pause
    exit /b 1
)

:: Install AI model dependencies
echo Installing AI model dependencies...
pip install einops>=0.6.0 kornia>=0.7.0 timm>=0.9.0 realesrgan==0.3.0
if errorlevel 1 (
    echo ERROR: Failed to install AI model dependencies.
    pause
    exit /b 1
)

:: Configure project-local model storage
echo Configuring AI models...
echo    All AI models will be stored in project models\ directory
echo    AI models downloaded on first run:
echo       * BiRefNet background removal model (~424MB)
echo       * Real-ESRGAN General v3 4x upscaling model (~17MB)
echo       * v0.3.0 supports only 4x (no 2x dedicated model)
echo    Models are managed independently per project

echo.
echo Installation completed successfully!
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
echo echo Starting AI Image/Video Processing Tool...
echo call conda activate edgehd
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

:: Run installer silently
"%TEMP%\%INSTALLER_NAME%" /InstallationType=JustMe /RegisterPython=1 /S /D=%USERPROFILE%\miniconda3
if errorlevel 1 (
    echo ERROR: Miniconda installation failed.
    exit /b 1
)

:: Clean up installer
del "%TEMP%\%INSTALLER_NAME%" >nul 2>&1

:: Add to PATH for current session
set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3\Library\bin;%PATH%"

:: Initialize conda for cmd
call "%USERPROFILE%\miniconda3\Scripts\conda.exe" init cmd.exe >nul 2>&1

echo.
echo Miniconda3 has been installed successfully!
echo Location: %USERPROFILE%\miniconda3
echo.
echo IMPORTANT: Please close this terminal and open a new one for conda to work properly.

exit /b 0 