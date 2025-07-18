@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PID_FILE=app.pid
set LOG_FILE=app.log
set ERROR_LOG=app_error.log

echo 📊 AI 이미지/비디오 처리 도구 상태 확인
echo ==================================

:: PID 파일 확인
if not exist "%PID_FILE%" (
    echo ❌ 서버가 실행되지 않고 있습니다.
    echo    시작하려면: start.bat
    echo.
    pause
    exit /b 1
)

:: PID 읽기
set /p PID=<"%PID_FILE%"

:: 프로세스 확인
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo ❌ 서버가 실행되지 않고 있습니다. (PID 파일은 존재)
    echo    PID %PID% 프로세스를 찾을 수 없습니다.
    echo    정리하려면: del %PID_FILE%
    echo    시작하려면: start.bat
    echo.
    pause
    exit /b 1
)

:: 서버 상태 표시
echo ✅ 서버가 정상적으로 실행 중입니다!
echo.

:: 기본 정보
echo 📋 서버 정보:
echo    ^• PID: %PID%

:: 프로세스 이름 가져오기
for /f "tokens=1" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do set "PROCESS_NAME=%%~a"
echo    ^• 프로세스 이름: !PROCESS_NAME!

:: CPU 및 메모리 사용률 (wmic 사용)
for /f "tokens=2 delims=," %%a in ('wmic process where "ProcessId=%PID%" get PageFileUsage /format:csv 2^>nul ^| find "%PID%"') do set "MEMORY_KB=%%a"
if defined MEMORY_KB (
    set /a MEMORY_MB=!MEMORY_KB!/1024
    echo    ^• 메모리 사용량: !MEMORY_MB! MB
) else (
    echo    ^• 메모리 사용량: 알 수 없음
)

:: 시작 시간 (대략적)
for /f "tokens=2,3" %%a in ('tasklist /FI "PID eq %PID%" /FO CSV /NH 2^>nul') do (
    echo    ^• 프로세스 정보: %%a %%b
)

echo.

:: 네트워크 정보
echo 🌐 네트워크 정보:
:: 포트 8080 확인
netstat -an | find ":8080" | find "LISTENING" >nul
if not errorlevel 1 (
    echo    ^• 포트 8080: 활성화됨 (LISTENING)
    echo    ^• 로컬 접속: http://localhost:8080
    
    :: 로컬 IP 주소 가져오기
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| find "IPv4"') do (
        set "LOCAL_IP=%%a"
        set "LOCAL_IP=!LOCAL_IP: =!"
        echo    ^• 네트워크 접속: http://!LOCAL_IP!:8080
        goto :ip_found
    )
    :ip_found
) else (
    echo    ^• 포트 8080: 비활성화 (서버가 아직 시작 중일 수 있음)
)
echo.

:: 로그 파일 정보
echo 📄 로그 파일 정보:
if exist "%LOG_FILE%" (
    for %%F in ("%LOG_FILE%") do set "LOG_SIZE=%%~zF"
    set /a LOG_SIZE_KB=!LOG_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%LOG_FILE%"') do set "LOG_LINES=%%a"
    echo    ^• 일반 로그: %LOG_FILE% (!LOG_SIZE_KB! KB, !LOG_LINES! 줄)
) else (
    echo    ^• 일반 로그: %LOG_FILE% (파일 없음)
)

if exist "%ERROR_LOG%" (
    for %%F in ("%ERROR_LOG%") do set "ERROR_SIZE=%%~zF"
    set /a ERROR_SIZE_KB=!ERROR_SIZE!/1024
    for /f %%a in ('find /C /V "" ^< "%ERROR_LOG%"') do set "ERROR_LINES=%%a"
    echo    ^• 에러 로그: %ERROR_LOG% (!ERROR_SIZE_KB! KB, !ERROR_LINES! 줄)
    
    :: 에러 로그에 내용이 있는지 확인
    if !ERROR_SIZE! GTR 10 (
        echo    ⚠️  에러 로그에 내용이 있습니다!
        echo       최근 에러 보기: type %ERROR_LOG%
    )
) else (
    echo    ^• 에러 로그: %ERROR_LOG% (파일 없음)
)
echo.

:: 시스템 자원 정보
echo 💻 시스템 자원:

:: 메모리 정보
for /f "tokens=2" %%a in ('wmic computersystem get TotalPhysicalMemory /value ^| find "="') do set "TOTAL_MEM=%%a"
for /f "tokens=2" %%a in ('wmic OS get FreePhysicalMemory /value ^| find "="') do set "FREE_MEM_KB=%%a"

if defined TOTAL_MEM if defined FREE_MEM_KB (
    set /a TOTAL_MEM_GB=!TOTAL_MEM!/1024/1024/1024
    set /a FREE_MEM_MB=!FREE_MEM_KB!/1024
    set /a USED_MEM_MB=(!TOTAL_MEM!/1024/1024) - !FREE_MEM_MB!
    echo    ^• 메모리: !USED_MEM_MB! MB / !TOTAL_MEM_GB! GB 사용 중
) else (
    echo    ^• 메모리: 정보를 가져올 수 없음
)

:: 디스크 정보 (현재 드라이브)
for /f "tokens=3,4" %%a in ('dir /-c ^| find "남은 공간"') do (
    echo    ^• 디스크: %%a %%b 여유 공간
    goto :disk_found
)
:disk_found

echo.

:: 애플리케이션 기능 정보
echo 🎯 애플리케이션 기능:
echo    ^• 🖼️  이미지 처리:
echo        - 고품질 AI 배경 제거 (BiRefNet)
echo        - 2x/4x 이미지 업스케일링
echo        - 연속 처리 지원
echo    ^• 🎬 비디오 처리:
echo        - 프레임별 배경 제거
echo        - 비디오 업스케일링
echo        - 실시간 진행률 표시
echo    ^• 📱 지원 형식:
echo        - 이미지: PNG, JPG, JPEG, GIF, BMP, WebP
echo        - 비디오: MP4, AVI, MOV, MKV
echo.

:: 유용한 명령어
echo 🔧 유용한 명령어:
echo    ^• 로그 실시간 보기: powershell "Get-Content %LOG_FILE% -Wait"
echo    ^• 에러 로그 보기: type %ERROR_LOG%
echo    ^• 서버 중지: stop.bat
echo    ^• 서버 재시작: stop.bat ^&^& start.bat
echo.

pause 