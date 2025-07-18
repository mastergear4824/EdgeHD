@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🗑️  EdgeHD AI 이미지 처리 애플리케이션 제거를 시작합니다...

REM 현재 디렉토리 확인
if not exist "app.py" (
    echo ❌ EdgeHD 프로젝트 디렉토리에서 실행해주세요
    pause
    exit /b 1
)

REM 사용자 확인
echo.
echo ⚠️  다음 항목들이 제거됩니다:
echo    - Python 가상환경 (venv\)
echo    - 설치된 Python 패키지들
echo    - AI 모델 파일들 (models\)
echo    - 업로드된 파일들 (uploads\)
echo    - 다운로드된 파일들 (downloads\)
echo    - 환경변수 설정
echo.
set /p "confirm=정말로 제거하시겠습니까? (y/N): "
if /i not "%confirm%"=="y" (
    echo ❌ 제거가 취소되었습니다
    pause
    exit /b 1
)

echo.
echo 🧹 환경 정리를 시작합니다...

REM 1. 가상환경 제거
if exist "venv" (
    echo 📦 Python 가상환경 제거 중...
    rmdir /s /q "venv" 2>nul
    echo ✅ 가상환경 제거 완료
) else (
    echo ℹ️  가상환경이 없습니다
)

REM 2. 모델 파일들 제거
if exist "models" (
    echo 🤖 AI 모델 파일들 제거 중...
    rmdir /s /q "models" 2>nul
    echo ✅ 모델 파일들 제거 완료
) else (
    echo ℹ️  모델 파일들이 없습니다
)

REM 3. 업로드 파일들 정리
if exist "uploads" (
    echo 📁 업로드 파일들 정리 중...
    for /f %%i in ('dir /b "uploads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "uploads\%%i" 2>nul
    )
    echo ✅ 업로드 파일들 정리 완료
)

REM 4. 다운로드 파일들 정리
if exist "downloads" (
    echo 📂 다운로드 파일들 정리 중...
    for /f %%i in ('dir /b "downloads\*" 2^>nul ^| findstr /v ".gitkeep"') do (
        del /q "downloads\%%i" 2>nul
    )
    echo ✅ 다운로드 파일들 정리 완료
)

REM 5. Python 캐시 파일들 제거
echo 🗂️  Python 캐시 파일들 제거 중...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
del /s /q "*.pyc" 2>nul
del /s /q "*.pyo" 2>nul
echo ✅ 캐시 파일들 제거 완료

REM 6. 환경변수 설정 안내
echo.
echo 🔧 환경변수 정리 안내:
echo    다음 환경변수들을 수동으로 제거해주세요:
echo    - HF_HOME
echo    - TRANSFORMERS_CACHE
echo.
echo    시스템 속성 ^> 고급 ^> 환경변수에서 제거하거나
echo    다음 명령어를 실행:
echo    setx HF_HOME ""
echo    setx TRANSFORMERS_CACHE ""

REM 7. 시스템 패키지 제거 안내 (선택사항)
echo.
echo 📋 추가 정리 옵션:
echo    시스템에 설치된 Python 패키지들도 제거하려면:
echo    pip uninstall -y flask flask-cors pillow torch torchvision transformers timm realesrgan opencv-python numpy
echo.

echo 🎉 EdgeHD 제거가 완료되었습니다!
echo.
echo 📝 재설치하려면:
echo    install.bat
echo.

pause 