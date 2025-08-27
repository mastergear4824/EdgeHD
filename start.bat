@echo off
setlocal enabledelayedexpansion

set BACKEND_PID_FILE=backend.pid
set FRONTEND_PID_FILE=frontend.pid
set BACKEND_LOG=backend.log
set FRONTEND_LOG=frontend.log
set BACKEND_ERROR_LOG=backend_error.log
set FRONTEND_ERROR_LOG=frontend_error.log

echo ========================================
echo EdgeHD 2.0 - Full-Stack Platform
echo Starting Background Services
echo ========================================
echo.

:: Check if already running
if exist "%BACKEND_PID_FILE%" (
    set /p BACKEND_PID=<"%BACKEND_PID_FILE%"
    tasklist /FI "PID eq !BACKEND_PID!" 2>nul | find "!BACKEND_PID!" >nul
    if not errorlevel 1 (
        echo ERROR: Backend already running (PID: !BACKEND_PID!)
        echo    To stop: stop.bat
        pause
        exit /b 1
    ) else (
        echo INFO: Cleaning up old backend PID file...
        del "%BACKEND_PID_FILE%" 2>nul
    )
)

if exist "%FRONTEND_PID_FILE%" (
    set /p FRONTEND_PID=<"%FRONTEND_PID_FILE%"
    tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | find "!FRONTEND_PID!" >nul
    if not errorlevel 1 (
        echo ERROR: Frontend already running (PID: !FRONTEND_PID!)
        echo    To stop: stop.bat
        pause
        exit /b 1
    ) else (
        echo INFO: Cleaning up old frontend PID file...
        del "%FRONTEND_PID_FILE%" 2>nul
    )
)

:: Check environments
echo [1/4] Checking environments...
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda not found. Please run install.bat first.
    pause
    exit /b 1
)

conda info --envs | findstr "edgehd" >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'edgehd' environment not found. Please run install.bat first.
    pause
    exit /b 1
)

node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please run install.bat first.
    pause
    exit /b 1
)

echo SUCCESS: All environments ready.

:: Initialize log files
echo [2/4] Initializing log files...
echo. > "%BACKEND_LOG%"
echo. > "%FRONTEND_LOG%"
echo. > "%BACKEND_ERROR_LOG%"
echo. > "%FRONTEND_ERROR_LOG%"

:: Start backend server
echo [3/4] Starting backend server...
call conda activate edgehd
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    pause
    exit /b 1
)

:: Set environment variables for backend
set "HF_HOME=%cd%\backend\models"
set "TRANSFORMERS_CACHE=%cd%\backend\models"
set "HUGGINGFACE_HUB_CACHE=%cd%\backend\models"

:: Create backend startup script
echo @echo off > temp_backend.bat
echo call conda activate edgehd >> temp_backend.bat
echo set "HF_HOME=%cd%\backend\models" >> temp_backend.bat
echo set "TRANSFORMERS_CACHE=%cd%\backend\models" >> temp_backend.bat
echo set "HUGGINGFACE_HUB_CACHE=%cd%\backend\models" >> temp_backend.bat
echo cd backend >> temp_backend.bat
echo python app.py >> temp_backend.bat

:: Start backend in background
start /B cmd /c "temp_backend.bat 1>%BACKEND_LOG% 2>%BACKEND_ERROR_LOG%"

:: Wait for backend initialization
timeout /t 3 >nul

:: Find backend Python process
set BACKEND_PID=
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    set "TEMP_PID=%%~a"
    set "TEMP_PID=!TEMP_PID:"=!"
    
    :: Check if this process is using port 8080
    netstat -ano | find "8080" | find "!TEMP_PID!" >nul 2>&1
    if not errorlevel 1 (
        set "BACKEND_PID=!TEMP_PID!"
        goto :backend_found
    )
)

:backend_found
if not defined BACKEND_PID (
    echo ERROR: Failed to start backend server.
    if exist "%BACKEND_ERROR_LOG%" (
        echo Backend error log:
        type "%BACKEND_ERROR_LOG%"
    )
    pause
    exit /b 1
)

echo %BACKEND_PID% > "%BACKEND_PID_FILE%"
echo SUCCESS: Backend started (PID: %BACKEND_PID%)

:: Start frontend server
echo [4/4] Starting frontend server...
cd frontend

:: Create frontend startup script
echo @echo off > ..\temp_frontend.bat
echo cd frontend >> ..\temp_frontend.bat
echo npm run build >> ..\temp_frontend.bat
echo npm start >> ..\temp_frontend.bat

:: Start frontend in background
start /B cmd /c "..\temp_frontend.bat 1>..\%FRONTEND_LOG% 2>..\%FRONTEND_ERROR_LOG%"

:: Wait for frontend initialization
timeout /t 5 >nul

:: Find frontend Node process
set FRONTEND_PID=
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq node.exe" /FO CSV /NH 2^>nul') do (
    set "TEMP_PID=%%~a"
    set "TEMP_PID=!TEMP_PID:"=!"
    
    :: Check if this process is using port 3000
    netstat -ano | find "3000" | find "!TEMP_PID!" >nul 2>&1
    if not errorlevel 1 (
        set "FRONTEND_PID=!TEMP_PID!"
        goto :frontend_found
    )
)

:frontend_found
if not defined FRONTEND_PID (
    echo ERROR: Failed to start frontend server.
    if exist "%FRONTEND_ERROR_LOG%" (
        echo Frontend error log:
        type "%FRONTEND_ERROR_LOG%"
    )
    pause
    exit /b 1
)

echo %FRONTEND_PID% > "%FRONTEND_PID_FILE%"
echo SUCCESS: Frontend started (PID: %FRONTEND_PID%)

cd ..

:: Clean up temporary files
del temp_backend.bat >nul 2>&1
del temp_frontend.bat >nul 2>&1

:: Verify both servers are running
timeout /t 2 >nul

echo.
echo SUCCESS: EdgeHD 2.0 Full-Stack Platform started successfully!
echo.
echo SERVER INFO:
echo    * Backend PID: %BACKEND_PID% (Python/Flask)
echo    * Frontend PID: %FRONTEND_PID% (Node.js/Next.js)
echo    * Backend Log: %BACKEND_LOG%
echo    * Frontend Log: %FRONTEND_LOG%
echo.
echo ACCESS URLs:
echo    * Frontend UI: http://localhost:3000
echo    * Backend API: http://localhost:8080
echo    * Network: http://^<computer-ip^>:3000
echo.
echo USEFUL COMMANDS:
echo    * Check status: status.bat
echo    * View backend logs: type %BACKEND_LOG%
echo    * View frontend logs: type %FRONTEND_LOG%
echo    * Stop servers: stop.bat
echo.
echo FEATURES:
echo    * Modern React UI with shadcn/ui components
echo    * AI-powered image processing (background removal, upscaling)
echo    * Real-time progress tracking
echo    * Drag & drop file uploads
echo    * Video processing capabilities
echo.

pause