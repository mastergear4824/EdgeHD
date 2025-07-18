@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo 🛑 AI 이미지/비디오 처리 도구 중지 중...

:: PID 파일 확인
if not exist "%PID_FILE%" (
    echo ❌ 실행 중인 서버를 찾을 수 없습니다.
    echo    PID 파일이 존재하지 않습니다: %PID_FILE%
    pause
    exit /b 1
)

:: PID 읽기
set /p PID=<"%PID_FILE%"

:: 프로세스 확인
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ❌ 프로세스 %PID%가 실행 중이지 않습니다.
    echo 🧹 PID 파일 정리 중...
    del "%PID_FILE%" 2>nul
    pause
    exit /b 1
)

echo 📋 서버 정보:
echo    ^• PID: %PID%
for /f "tokens=1" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do set "PROCESS_NAME=%%~a"
echo    ^• 프로세스: !PROCESS_NAME!

:: 정상 종료 시도
echo ⏳ 정상 종료 신호 전송 중...
taskkill /PID %PID% >nul 2>&1

:: 정상 종료 대기
set WAIT_TIME=10
for /L %%i in (1,1,%WAIT_TIME%) do (
    tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
    if errorlevel 1 (
        echo ✅ 서버가 정상적으로 종료되었습니다.
        del "%PID_FILE%" 2>nul
        goto :log_cleanup
    )
    
    echo    대기 중... (%%i/%WAIT_TIME%초)
    timeout /t 1 >nul
)

echo.
echo ⚠️  정상 종료에 실패했습니다. 강제 종료를 시도합니다...

:: 강제 종료 시도
echo 💥 강제 종료 신호 전송 중...
taskkill /F /PID %PID% >nul 2>&1

:: 강제 종료 확인
timeout /t 2 >nul
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ✅ 서버가 강제로 종료되었습니다.
    del "%PID_FILE%" 2>nul
    goto :log_cleanup
) else (
    echo ❌ 프로세스 종료에 실패했습니다.
    echo    수동으로 종료해야 할 수 있습니다: taskkill /F /PID %PID%
    echo    또는 작업 관리자를 사용하세요.
    pause
    exit /b 1
)

:log_cleanup
:: 로그 파일 정리 옵션
echo.
echo 📁 로그 파일 정리:
if exist "%LOG_FILE%" (
    for %%F in ("%LOG_FILE%") do set "LOG_SIZE=%%~zF"
    set /a LOG_SIZE_KB=!LOG_SIZE!/1024
    echo    ^• 로그 파일: %LOG_FILE% (!LOG_SIZE_KB! KB)
) else (
    echo    ^• 로그 파일: %LOG_FILE% (파일 없음)
)

if exist "%ERROR_LOG%" (
    for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
    set /a ERROR_SIZE_KB=!ERROR_SIZE!/1024
    echo    ^• 에러 로그: %ERROR_LOG% (!ERROR_SIZE_KB! KB)
) else (
    echo    ^• 에러 로그: %ERROR_LOG% (파일 없음)
)

echo.
set /p CLEANUP_CHOICE="로그 파일을 삭제하시겠습니까? (y/N): "
if /i "!CLEANUP_CHOICE!"=="y" (
    if exist "%LOG_FILE%" del "%LOG_FILE%" 2>nul
    if exist "%ERROR_LOG%" del "%ERROR_LOG%" 2>nul
    echo 🗑️  로그 파일이 삭제되었습니다.
) else (
    echo 📄 로그 파일이 보존되었습니다.
)

echo.
echo 🎯 서버 중지 완료!
pause 