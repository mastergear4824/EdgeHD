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

:: Check Conda installation and find correct path
echo INFO: Checking for Conda installation...
where conda >nul 2>&1
if not errorlevel 1 (
    echo SUCCESS: Conda found in PATH
    set "CONDA_FOUND=1"
    goto :check_environment
)

:: Check common Conda paths
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at %USERPROFILE%\miniconda3\Scripts\conda.exe
    set "PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3\Library\bin;%PATH%"
    set "CONDA_FOUND=1"
    goto :check_environment
)

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    echo SUCCESS: Conda found at %USERPROFILE%\anaconda3\condabin\conda.bat
    set "PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Scripts;%USERPROFILE%\anaconda3\condabin;%PATH%"
    set "CONDA_FOUND=1"
    goto :check_environment
)

if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" (
    echo SUCCESS: Conda found at C:\ProgramData\Miniconda3\Scripts\conda.exe
    set "PATH=C:\ProgramData\Miniconda3;C:\ProgramData\Miniconda3\Scripts;C:\ProgramData\Miniconda3\Library\bin;%PATH%"
    set "CONDA_FOUND=1"
    goto :check_environment
)

if not defined CONDA_FOUND (
    echo ERROR: Conda is not installed.
    echo    Please run install.bat first to install conda and the environment.
    pause
    exit /b 1
)

:check_environment
:: Check Conda environment
echo INFO: Checking Conda environment 'edgehd'...
conda info --envs | find "edgehd" >nul
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
    echo INFO: Trying alternative activation method...
    
    :: Try alternative activation for Anaconda
    if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate edgehd
        if errorlevel 1 (
            echo ERROR: All activation methods failed.
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: Could not activate environment.
        pause
        exit /b 1
    )
)

echo SUCCESS: Environment 'edgehd' activated successfully.

:: Set environment variables for project-local model storage
set "HF_HOME=%cd%\models"
set "TRANSFORMERS_CACHE=%cd%\models"
echo INFO: AI models will be managed in project directory (%cd%\models)

:: Initialize log files
echo INFO: Initializing log files...
echo. > "%LOG_FILE%"
echo. > "%ERROR_LOG%"

:: Test Python and packages before starting
echo INFO: Testing Python and packages...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not available in conda environment.
    echo    Please check your installation.
    pause
    exit /b 1
)

python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Flask not installed.
    echo    Please run install.bat again.
    pause
    exit /b 1
)

echo SUCCESS: Python and packages are ready.

:: Start server in background
echo INFO: Starting server in background...

:: Create a temporary batch file to run python with proper conda environment
echo @echo off > temp_start.bat
echo call conda activate edgehd >> temp_start.bat
echo set "HF_HOME=%cd%\models" >> temp_start.bat
echo set "TRANSFORMERS_CACHE=%cd%\models" >> temp_start.bat
echo python app.py >> temp_start.bat

:: Run the temporary batch file in background and capture output
start /B cmd /c "temp_start.bat 1>%LOG_FILE% 2>%ERROR_LOG%"

:: Clean up temporary file
timeout /t 1 >nul
del temp_start.bat >nul 2>&1

:: Wait for server initialization
echo INFO: Waiting for server initialization...
timeout /t 5 >nul

:: Find Python process with more specific criteria
set PID=
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    set "TEMP_PID=%%~a"
    :: Remove quotes if present
    set "TEMP_PID=!TEMP_PID:"=!"
    
    :: Check if this process is actually running our app by checking port usage
    netstat -ano | find "8080" | find "!TEMP_PID!" >nul 2>&1
    if not errorlevel 1 (
        set "PID=!TEMP_PID!"
        goto :found_pid
    )
)

:: If port-based search failed, try most recent python process
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    set "PID=%%~a"
    set "PID=!PID:"=!"
    goto :found_pid
)

:found_pid
if not defined PID (
    echo ERROR: Failed to start server.
    echo INFO: Checking error log for details...
    if exist "%ERROR_LOG%" (
        echo.
        echo ERROR LOG CONTENTS:
        type "%ERROR_LOG%"
    )
    echo.
    echo    Check error log: type %ERROR_LOG%
    pause
    exit /b 1
)

echo %PID% > "%PID_FILE%"

:: Wait a bit more for full initialization
timeout /t 3 >nul

:: Verify server is responding on port 8080
echo INFO: Verifying server is responding...
netstat -an | find "8080" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Port 8080 not found in listening state.
    echo INFO: Server might still be initializing...
    timeout /t 3 >nul
    
    :: Check again
    netstat -an | find "8080" | find "LISTENING" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Server failed to bind to port 8080.
        echo INFO: Checking error log...
        if exist "%ERROR_LOG%" (
            echo.
            echo ERROR LOG CONTENTS:
            type "%ERROR_LOG%"
        )
        
        :: Clean up PID file
        if exist "%PID_FILE%" del "%PID_FILE%"
        pause
        exit /b 1
    )
)

:: Verify process is still running
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if not errorlevel 1 (
    echo.
    echo SUCCESS: AI Image/Video Processing Tool started successfully!
    echo.
    echo SERVER INFO:
    echo    * PID: %PID%
    echo    * Log file: %LOG_FILE%
    echo    * Error log: %ERROR_LOG%
    echo    * Python version: 3.11 with PyTorch 2.1.0
    echo    * Real-ESRGAN: v0.3.0 compatible
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
    echo    * Image processing: Background removal, 4x upscaling
    echo    * Video processing: Frame-by-frame processing + upscaling
    echo    * Real-time progress display
    echo    * Compatible PyTorch 2.1.0 + Real-ESRGAN v0.3.0
    echo.
) else (
    echo ERROR: Server process died unexpectedly.
    echo INFO: Checking error log...
    if exist "%ERROR_LOG%" (
        echo.
        echo ERROR LOG CONTENTS:
        type "%ERROR_LOG%"
    )
    if exist "%PID_FILE%" del "%PID_FILE%"
    pause
    exit /b 1
)

pause 