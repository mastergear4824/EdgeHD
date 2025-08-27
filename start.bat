@echo off
setlocal enabledelayedexpansion

:: Set script variables
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"

set BACKEND_PID_FILE=%SCRIPT_DIR%backend.pid
set FRONTEND_PID_FILE=%SCRIPT_DIR%frontend.pid
set BACKEND_LOG=%SCRIPT_DIR%backend.log
set FRONTEND_LOG=%SCRIPT_DIR%frontend.log
set BACKEND_ERROR_LOG=%SCRIPT_DIR%backend_error.log
set FRONTEND_ERROR_LOG=%SCRIPT_DIR%frontend_error.log

echo ========================================
echo EdgeHD 2.0 - Full-Stack Platform
echo Starting Background Services
echo ========================================
echo.

:: Check if already running
echo [Pre-check] Checking for existing services...

if exist "%BACKEND_PID_FILE%" (
    set /p BACKEND_PID=<"%BACKEND_PID_FILE%"
    :: Validate PID is numeric
    echo !BACKEND_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        tasklist /FI "PID eq !BACKEND_PID!" 2>nul | find "!BACKEND_PID!" >nul
        if not errorlevel 1 (
            echo ERROR: Backend already running (PID: !BACKEND_PID!)
            echo    To stop: stop.bat
            pause
            exit /b 1
        )
    )
    echo INFO: Cleaning up stale backend PID file...
    del "%BACKEND_PID_FILE%" 2>nul
)

if exist "%FRONTEND_PID_FILE%" (
    set /p FRONTEND_PID=<"%FRONTEND_PID_FILE%"
    :: Validate PID is numeric  
    echo !FRONTEND_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | find "!FRONTEND_PID!" >nul
        if not errorlevel 1 (
            echo ERROR: Frontend already running (PID: !FRONTEND_PID!)
            echo    To stop: stop.bat
            pause
            exit /b 1
        )
    )
    echo INFO: Cleaning up stale frontend PID file...
    del "%FRONTEND_PID_FILE%" 2>nul
)

:: Check port availability
netstat -an | find ":8080" | find "LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Port 8080 is already in use.
    echo    Another service may be running on this port.
    pause
    exit /b 1
)

netstat -an | find ":3000" | find "LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Port 3000 is already in use.
    echo    Another service may be running on this port.
    pause
    exit /b 1
)

:: Check environments
echo [1/4] Checking environments...
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda not found in PATH. Please run install.bat first.
    pause
    exit /b 1
)

conda info --envs | findstr "edgehd" >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'edgehd' environment not found. Please run install.bat first.
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%\node_modules" (
    echo ERROR: Frontend dependencies not found. Please run install.bat first.
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%node_modules" (
    echo ERROR: Root dependencies not found. Please run install.bat first.
    pause
    exit /b 1
)

echo SUCCESS: All environments ready.

:: Initialize log files
echo [2/4] Initializing log files...
echo %DATE% %TIME% - Backend server starting... > "%BACKEND_LOG%"
echo %DATE% %TIME% - Frontend server starting... > "%FRONTEND_LOG%"
echo. > "%BACKEND_ERROR_LOG%"
echo. > "%FRONTEND_ERROR_LOG%"

:: Start backend server
echo [3/4] Starting backend server...

:: Create backend startup batch file
echo @echo off > "%SCRIPT_DIR%temp_backend.bat"
echo call conda activate edgehd >> "%SCRIPT_DIR%temp_backend.bat"
echo if errorlevel 1 exit /b 1 >> "%SCRIPT_DIR%temp_backend.bat"
echo set "HF_HOME=%SCRIPT_DIR%backend\models" >> "%SCRIPT_DIR%temp_backend.bat"
echo set "TRANSFORMERS_CACHE=%SCRIPT_DIR%backend\models" >> "%SCRIPT_DIR%temp_backend.bat"
echo set "HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%backend\models" >> "%SCRIPT_DIR%temp_backend.bat"
echo cd /d "%BACKEND_DIR%" >> "%SCRIPT_DIR%temp_backend.bat"
echo python app.py >> "%SCRIPT_DIR%temp_backend.bat"

:: Start backend in background
start /B "" cmd /c ""%SCRIPT_DIR%temp_backend.bat" 1>"%BACKEND_LOG%" 2>"%BACKEND_ERROR_LOG%""

:: Wait for backend initialization
echo INFO: Waiting for backend to initialize...
timeout /t 5 >nul

:: Find backend Python process using port
set BACKEND_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8080" ^| find "LISTENING"') do (
    set "TEMP_PID=%%a"
    tasklist /FI "PID eq !TEMP_PID!" 2>nul | find "python.exe" >nul
    if not errorlevel 1 (
        set "BACKEND_PID=!TEMP_PID!"
        goto :backend_found
    )
)

:backend_found
if not defined BACKEND_PID (
    echo ERROR: Failed to start backend server.
    echo INFO: Checking backend error log...
    if exist "%BACKEND_ERROR_LOG%" (
        echo ===== Backend Error Log =====
        type "%BACKEND_ERROR_LOG%"
        echo =============================
    )
    del "%SCRIPT_DIR%temp_backend.bat" 2>nul
    pause
    exit /b 1
)

echo %BACKEND_PID% > "%BACKEND_PID_FILE%"
echo SUCCESS: Backend started (PID: %BACKEND_PID%)

:: Start frontend server
echo [4/4] Starting frontend server...

:: Create frontend startup batch file
echo @echo off > "%SCRIPT_DIR%temp_frontend.bat"
echo cd /d "%FRONTEND_DIR%" >> "%SCRIPT_DIR%temp_frontend.bat"
echo npm run build >> "%SCRIPT_DIR%temp_frontend.bat"
echo if errorlevel 1 exit /b 1 >> "%SCRIPT_DIR%temp_frontend.bat"
echo npm start >> "%SCRIPT_DIR%temp_frontend.bat"

:: Start frontend in background
start /B "" cmd /c ""%SCRIPT_DIR%temp_frontend.bat" 1>"%FRONTEND_LOG%" 2>"%FRONTEND_ERROR_LOG%""

:: Wait for frontend initialization
echo INFO: Waiting for frontend to initialize...
timeout /t 8 >nul

:: Find frontend Node process using port
set FRONTEND_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| find ":3000" ^| find "LISTENING"') do (
    set "TEMP_PID=%%a"
    tasklist /FI "PID eq !TEMP_PID!" 2>nul | find "node.exe" >nul
    if not errorlevel 1 (
        set "FRONTEND_PID=!TEMP_PID!"
        goto :frontend_found
    )
)

:frontend_found
if not defined FRONTEND_PID (
    echo ERROR: Failed to start frontend server.
    echo INFO: Checking frontend error log...
    if exist "%FRONTEND_ERROR_LOG%" (
        echo ===== Frontend Error Log =====
        type "%FRONTEND_ERROR_LOG%"
        echo ===============================
    )
    del "%SCRIPT_DIR%temp_frontend.bat" 2>nul
    pause
    exit /b 1
)

echo %FRONTEND_PID% > "%FRONTEND_PID_FILE%"
echo SUCCESS: Frontend started (PID: %FRONTEND_PID%)

:: Clean up temporary files
del "%SCRIPT_DIR%temp_backend.bat" >nul 2>&1
del "%SCRIPT_DIR%temp_frontend.bat" >nul 2>&1

:: Final verification
echo.
echo INFO: Performing final verification...
timeout /t 3 >nul

:: Verify both processes are still running
tasklist /FI "PID eq %BACKEND_PID%" 2>nul | find "%BACKEND_PID%" >nul
if errorlevel 1 (
    echo ERROR: Backend process died after startup.
    echo    Check %BACKEND_ERROR_LOG% for details.
    pause
    exit /b 1
)

tasklist /FI "PID eq %FRONTEND_PID%" 2>nul | find "%FRONTEND_PID%" >nul
if errorlevel 1 (
    echo ERROR: Frontend process died after startup.
    echo    Check %FRONTEND_ERROR_LOG% for details.
    pause
    exit /b 1
)

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
echo.
echo USEFUL COMMANDS:
echo    * Check status: status.bat
echo    * View backend logs: type %BACKEND_LOG%
echo    * View frontend logs: type %FRONTEND_LOG%
echo    * Stop servers: stop.bat
echo.

pause