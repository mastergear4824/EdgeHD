@echo off
setlocal enabledelayedexpansion
echo Starting AI Image/Video Processing Tool...

:: Find conda installation
where conda >nul 2>&1
if not errorlevel 1 (
    call conda activate edgehd
) else (
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat" edgehd
    ) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
        call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate edgehd
    ) else (
        echo ERROR: Conda not found. Please run install.bat first.
        pause
        exit /b 1
    )
)

:: Set environment variables for project-local model storage
set HF_HOME=%cd%\models
set TRANSFORMERS_CACHE=%cd%\models

echo INFO: Starting with PyTorch 2.1.0 and Real-ESRGAN v0.3.0 compatibility
python app.py
pause
