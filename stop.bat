@echo off
setlocal enabledelayedexpansion

set BACKEND_PID_FILE=backend.pid
set FRONTEND_PID_FILE=frontend.pid
set BACKEND_LOG=backend.log
set FRONTEND_LOG=frontend.log
set BACKEND_ERROR_LOG=backend_error.log
set FRONTEND_ERROR_LOG=frontend_error.log

echo ========================================
echo EdgeHD 2.0 Full-Stack Platform
echo Stopping Backend + Frontend Servers
echo ========================================
echo.

set "BACKEND_STOPPED=false"
set "FRONTEND_STOPPED=false"

:: Stop backend server
echo [1/2] Stopping backend server...
if not exist "%BACKEND_PID_FILE%" (
    echo INFO: Backend PID file not found. Backend may not be running.
    set "BACKEND_STOPPED=true"
) else (
    set /p BACKEND_PID=<"%BACKEND_PID_FILE%"
    set "BACKEND_PID=!BACKEND_PID:"=!"
    set "BACKEND_PID=!BACKEND_PID: =!"
    
    :: Validate PID is numeric
    echo !BACKEND_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
        echo ERROR: Invalid backend PID: !BACKEND_PID!
        del "%BACKEND_PID_FILE%" 2>nul
        set "BACKEND_STOPPED=true"
    ) else (
        :: Check if process exists
        tasklist /FI "PID eq !BACKEND_PID!" 2>nul | find "!BACKEND_PID!" >nul
        if errorlevel 1 (
            echo INFO: Backend process !BACKEND_PID! not found. Cleaning up PID file.
            del "%BACKEND_PID_FILE%" 2>nul
            set "BACKEND_STOPPED=true"
        ) else (
            echo INFO: Stopping backend process !BACKEND_PID!...
            
            :: Attempt graceful shutdown
            taskkill /PID !BACKEND_PID! >nul 2>&1
            if errorlevel 1 (
                echo WARNING: Graceful shutdown failed. Attempting force shutdown...
                taskkill /F /PID !BACKEND_PID! >nul 2>&1
                if errorlevel 1 (
                    echo ERROR: Failed to stop backend process !BACKEND_PID!.
                ) else (
                    echo SUCCESS: Backend force stopped.
                    del "%BACKEND_PID_FILE%" 2>nul
                    set "BACKEND_STOPPED=true"
                )
            ) else (
                :: Wait for graceful shutdown
                timeout /t 3 >nul
                tasklist /FI "PID eq !BACKEND_PID!" 2>nul | find "!BACKEND_PID!" >nul
                if errorlevel 1 (
                    echo SUCCESS: Backend stopped gracefully.
                    del "%BACKEND_PID_FILE%" 2>nul
                    set "BACKEND_STOPPED=true"
                ) else (
                    echo WARNING: Graceful shutdown timed out. Force stopping...
                    taskkill /F /PID !BACKEND_PID! >nul 2>&1
                    del "%BACKEND_PID_FILE%" 2>nul
                    set "BACKEND_STOPPED=true"
                )
            )
        )
    )
)

:: Stop frontend server
echo [2/2] Stopping frontend server...
if not exist "%FRONTEND_PID_FILE%" (
    echo INFO: Frontend PID file not found. Frontend may not be running.
    set "FRONTEND_STOPPED=true"
) else (
    set /p FRONTEND_PID=<"%FRONTEND_PID_FILE%"
    set "FRONTEND_PID=!FRONTEND_PID:"=!"
    set "FRONTEND_PID=!FRONTEND_PID: =!"
    
    :: Validate PID is numeric
    echo !FRONTEND_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if errorlevel 1 (
        echo ERROR: Invalid frontend PID: !FRONTEND_PID!
        del "%FRONTEND_PID_FILE%" 2>nul
        set "FRONTEND_STOPPED=true"
    ) else (
        :: Check if process exists
        tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | find "!FRONTEND_PID!" >nul
        if errorlevel 1 (
            echo INFO: Frontend process !FRONTEND_PID! not found. Cleaning up PID file.
            del "%FRONTEND_PID_FILE%" 2>nul
            set "FRONTEND_STOPPED=true"
        ) else (
            echo INFO: Stopping frontend process !FRONTEND_PID!...
            
            :: Attempt graceful shutdown
            taskkill /PID !FRONTEND_PID! >nul 2>&1
            if errorlevel 1 (
                echo WARNING: Graceful shutdown failed. Attempting force shutdown...
                taskkill /F /PID !FRONTEND_PID! >nul 2>&1
                if errorlevel 1 (
                    echo ERROR: Failed to stop frontend process !FRONTEND_PID!.
                ) else (
                    echo SUCCESS: Frontend force stopped.
                    del "%FRONTEND_PID_FILE%" 2>nul
                    set "FRONTEND_STOPPED=true"
                )
            ) else (
                :: Wait for graceful shutdown
                timeout /t 3 >nul
                tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | find "!FRONTEND_PID!" >nul
                if errorlevel 1 (
                    echo SUCCESS: Frontend stopped gracefully.
                    del "%FRONTEND_PID_FILE%" 2>nul
                    set "FRONTEND_STOPPED=true"
                ) else (
                    echo WARNING: Graceful shutdown timed out. Force stopping...
                    taskkill /F /PID !FRONTEND_PID! >nul 2>&1
                    del "%FRONTEND_PID_FILE%" 2>nul
                    set "FRONTEND_STOPPED=true"
                )
            )
        )
    )
)

:: Verify ports are freed
echo.
echo INFO: Verifying ports are freed...
timeout /t 2 >nul

netstat -an | find "8080" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo SUCCESS: Port 8080 (backend) is now free.
) else (
    echo WARNING: Port 8080 is still in use by another process.
)

netstat -an | find "3000" | find "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo SUCCESS: Port 3000 (frontend) is now free.
) else (
    echo WARNING: Port 3000 is still in use by another process.
)

:: Log file cleanup option
echo.
echo LOG FILE CLEANUP:
if exist "%BACKEND_LOG%" (
    for %%F in ("%BACKEND_LOG%") do set "BACKEND_LOG_SIZE=%%~zF"
    if !BACKEND_LOG_SIZE! GTR 0 (
        set /a BACKEND_LOG_KB=!BACKEND_LOG_SIZE!/1024
        echo    * Backend log: %BACKEND_LOG% (!BACKEND_LOG_KB! KB)
    ) else (
        echo    * Backend log: %BACKEND_LOG% (Empty)
    )
) else (
    echo    * Backend log: %BACKEND_LOG% (File not found)
)

if exist "%FRONTEND_LOG%" (
    for %%F in ("%FRONTEND_LOG%") do set "FRONTEND_LOG_SIZE=%%~zF"
    if !FRONTEND_LOG_SIZE! GTR 0 (
        set /a FRONTEND_LOG_KB=!FRONTEND_LOG_SIZE!/1024
        echo    * Frontend log: %FRONTEND_LOG% (!FRONTEND_LOG_KB! KB)
    ) else (
        echo    * Frontend log: %FRONTEND_LOG% (Empty)
    )
) else (
    echo    * Frontend log: %FRONTEND_LOG% (File not found)
)

:: Show error logs if they have content
if exist "%BACKEND_ERROR_LOG%" (
    for %%F in ("%BACKEND_ERROR_LOG%") do set "BACKEND_ERROR_SIZE=%%~zF"
    if !BACKEND_ERROR_SIZE! GTR 10 (
        set /a BACKEND_ERROR_KB=!BACKEND_ERROR_SIZE!/1024
        echo    * Backend errors: %BACKEND_ERROR_LOG% (!BACKEND_ERROR_KB! KB)
        echo.
        echo BACKEND ERROR LOG PREVIEW (last 5 lines):
        powershell "Get-Content '%BACKEND_ERROR_LOG%' | Select-Object -Last 5"
    )
)

if exist "%FRONTEND_ERROR_LOG%" (
    for %%F in ("%FRONTEND_ERROR_LOG%") do set "FRONTEND_ERROR_SIZE=%%~zF"
    if !FRONTEND_ERROR_SIZE! GTR 10 (
        set /a FRONTEND_ERROR_KB=!FRONTEND_ERROR_SIZE!/1024
        echo    * Frontend errors: %FRONTEND_ERROR_LOG% (!FRONTEND_ERROR_KB! KB)
        echo.
        echo FRONTEND ERROR LOG PREVIEW (last 5 lines):
        powershell "Get-Content '%FRONTEND_ERROR_LOG%' | Select-Object -Last 5"
    )
)

echo.
set /p CLEANUP_CHOICE="Delete log files? (y/N): "
if /i "!CLEANUP_CHOICE!"=="y" (
    if exist "%BACKEND_LOG%" del "%BACKEND_LOG%" 2>nul
    if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%" 2>nul
    if exist "%BACKEND_ERROR_LOG%" del "%BACKEND_ERROR_LOG%" 2>nul
    if exist "%FRONTEND_ERROR_LOG%" del "%FRONTEND_ERROR_LOG%" 2>nul
    echo INFO: Log files deleted.
) else (
    echo INFO: Log files preserved.
)

echo.
if "%BACKEND_STOPPED%"=="true" if "%FRONTEND_STOPPED%"=="true" (
    echo SUCCESS: EdgeHD 2.0 Full-Stack Platform stopped successfully!
) else (
    echo WARNING: Some servers may not have stopped properly.
    echo Please check the status manually or restart your computer if needed.
)

echo.
echo INFO: To restart the platform, run: start.bat
echo INFO: For development mode, run: run.bat or npm run dev

pause