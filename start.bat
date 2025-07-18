@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo 🚀 AI 이미지/비디오 처리 도구 시작 중...

:: 이미 실행 중인지 확인
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /FI "PID eq !PID!" 2>nul | find "!PID!" >nul
    if not errorlevel 1 (
        echo ❌ 이미 실행 중입니다 (PID: !PID!)
        echo    중지하려면: stop.bat
        pause
        exit /b 1
    ) else (
        echo 🧹 오래된 PID 파일 정리 중...
        del "%PID_FILE%" 2>nul
    )
)

:: Conda 환경 확인
where conda >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda가 설치되어 있지 않습니다.
    echo    설치 후 다시 시도해주세요.
    pause
    exit /b 1
)

:: Conda 환경 활성화 확인
echo 🔧 Conda 환경 'edgehd' 확인 중...
call conda info --envs | find "edgehd" >nul
if errorlevel 1 (
    echo ❌ 'edgehd' 환경을 찾을 수 없습니다.
    echo    먼저 install.bat를 실행해주세요.
    pause
    exit /b 1
)

echo 🔄 Conda 환경 활성화 중...
call conda activate edgehd
if errorlevel 1 (
    echo ❌ Conda 환경 활성화에 실패했습니다.
    pause
    exit /b 1
)

:: 프로젝트 내 모델 저장 환경변수 설정
set HF_HOME=%cd%\models
set TRANSFORMERS_CACHE=%cd%\models
echo 🤖 AI 모델을 프로젝트 내에서 관리합니다 (%cd%\models)

:: 로그 파일 초기화
echo 📝 로그 파일 초기화 중...
echo. > "%LOG_FILE%"
echo. > "%ERROR_LOG%"

:: 백그라운드에서 실행
echo ▶️  서버를 백그라운드에서 시작합니다...

:: Python 스크립트를 백그라운드에서 실행
start /B python app.py 1>"%LOG_FILE%" 2>"%ERROR_LOG%"

:: 프로세스 PID 찾기 및 저장
timeout /t 2 >nul
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| find "python.exe"') do (
    set "PID=%%~a"
    goto :found_pid
)

:found_pid
if not defined PID (
    echo ❌ 서버 시작에 실패했습니다.
    echo    에러 로그를 확인해주세요: type %ERROR_LOG%
    pause
    exit /b 1
)

echo %PID% > "%PID_FILE%"

:: 서버 시작 대기
echo ⏳ 서버 초기화 대기 중...
timeout /t 3 >nul

:: 프로세스 확인
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if not errorlevel 1 (
    echo.
    echo ✅ AI 이미지/비디오 처리 도구가 성공적으로 시작되었습니다!
    echo.
    echo 📊 서버 정보:
    echo    ^• PID: %PID%
    echo    ^• 로그 파일: %LOG_FILE%
    echo    ^• 에러 로그: %ERROR_LOG%
    echo.
    echo 🌐 접속 주소:
    echo    ^• 로컬: http://localhost:8080
    echo    ^• 네트워크: http://^<컴퓨터IP^>:8080
    echo.
    echo 📝 유용한 명령어:
    echo    ^• 상태 확인: status.bat
    echo    ^• 로그 보기: type %LOG_FILE%
    echo    ^• 서버 중지: stop.bat
    echo.
    echo 📖 기능:
    echo    ^• 🖼️  이미지 처리: 배경제거, 2x/4x 업스케일링
    echo    ^• 🎬 비디오 처리: 프레임별 배경제거 + 업스케일링
    echo    ^• ⚡ 실시간 진행률 표시
    echo.
) else (
    echo ❌ 서버 시작에 실패했습니다.
    echo    에러 로그를 확인해주세요: type %ERROR_LOG%
    if exist "%PID_FILE%" del "%PID_FILE%"
    pause
    exit /b 1
)

pause 