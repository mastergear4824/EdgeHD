# AI 이미지/비디오 처리 도구 🎨🎬

고품질 AI 배경 제거와 이미지/비디오 업스케일링을 제공하는 웹 애플리케이션입니다.

## ✨ 주요 기능

### 🖼️ 이미지 처리

- **🎯 고품질 AI 배경 제거**: BiRefNet 모델을 사용한 85% 성공률의 정밀한 배경 제거
- **📈 AI 업스케일링**: Real-ESRGAN General v3 4x 고품질 이미지 확대 (v0.3.0)
- **🔄 연속 처리**: 배경 제거 → 업스케일 또는 업스케일 → 배경 제거 등 자유로운 조합

### 🎬 비디오 처리 (NEW!)

- **🎞️ 프레임별 AI 처리**: MP4/AVI/MOV/MKV 비디오를 프레임별로 분해하여 처리
- **📸 마지막 프레임 추출**: 비디오에서 마지막 프레임을 이미지로 추출 및 다운로드 (NEW!)
- **🎭 비디오 배경 제거**: 각 프레임에 고품질 AI 배경 제거 적용
- **🔍 비디오 업스케일링**: Real-ESRGAN General v3 4x 비디오 해상도 향상 (v0.3.0)
- **🎪 조합 처리**: 배경제거 + 4x 업스케일링 동시 적용
- **⏰ 실시간 진행률**: 현재 처리 프레임, 남은 시간, 재미있는 메시지 표시

### 🌟 공통 기능

- **📱 반응형 웹 UI**: PC/모바일 최적화된 직관적 인터페이스
- **⚡ 실시간 진행률**: Server-Sent Events를 통한 실시간 처리 상태 확인
- **🍎 Apple Silicon 최적화**: MPS GPU 가속 지원
- **🖥️ 크로스 플랫폼**: Windows, macOS, Linux 지원

## 🛠️ 기술 스택

- **Backend**: Flask 3.0.0, Python 3.10+
- **AI 모델**: BiRefNet (배경 제거), Real-ESRGAN v0.3.0 General v3 (AI 업스케일링)
- **이미지/비디오 처리**: OpenCV, Pillow, NumPy<2.0.0 (호환성)
- **딥러닝**: PyTorch 최신버전, torchvision, torchaudio, Transformers (자동 설치)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **실시간 통신**: Server-Sent Events
- **비디오 코덱**: MP4V (비디오 재조립)

## 📋 시스템 요구사항

- **Python**: 3.10 이상
- **메모리**: 최소 8GB RAM 권장 (비디오 처리 시 16GB 권장)
- **저장공간**: 약 5GB (모델 파일 포함), 비디오 처리 시 추가 임시 공간 필요
- **GPU**: Apple Silicon MPS 또는 CUDA (선택사항, CPU 대체 가능)
- **운영체제**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)

## 🚀 설치 가이드

> **🆕 기존 사용자**: [업데이트 방법](#-기존-설치자-업데이트-권장)을 확인하여 새로운 **📸 마지막 프레임 추출** 기능을 사용해보세요!

### 1. Conda 설치

#### macOS (Miniforge 권장)

```bash
# Homebrew가 없는 경우 먼저 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Miniforge 설치 (Apple Silicon 최적화)
brew install miniforge

# 또는 직접 다운로드
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

#### Windows

```cmd
:: Miniconda 다운로드 및 설치
:: https://docs.conda.io/en/latest/miniconda.html 에서 다운로드

:: 또는 Chocolatey 사용
choco install miniconda3

:: 또는 Winget 사용 (Windows 10+)
winget install Anaconda.Miniconda3
```

#### Linux

```bash
# Miniconda 다운로드 및 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. 프로젝트 클론 및 환경 설정

```bash
# 프로젝트 클론
git clone https://github.com/mastergear4824/EdgeHD
cd EdgeHD
```

### 🚀 간편 설치 (권장)

운영체제에 맞는 설치 스크립트를 사용하여 한 번에 설치할 수 있습니다:

#### 📋 Conda가 이미 설치된 경우 (일반적인 설치)

**Linux/macOS**:

```bash
# 실행 권한 부여 및 설치
chmod +x install.sh
./install.sh
```

**Windows (Intel/AMD)**:

```cmd
# 관리자 권한으로 명령 프롬프트 실행 후
install.bat
```

**ARM64 Windows (Surface Pro X, Copilot+ PC 등)**:

> **🚨 중요**: ARM64 Windows는 별도 설치 과정이 필요합니다.

```cmd
# install.bat 실행 시 ARM64 감지 시 자동 가이드 표시
install.bat
```

또는 [ARM64 Windows 전용 가이드](#arm64-windows-전용-가이드) 참조

#### 🔄 Conda가 설치되어 있지 않은 경우 (자동 설치 - 2단계 필요)

> **⚠️ 중요**: Conda가 없는 경우 **반드시 2번 실행**해야 합니다!

**Windows 사용자**:

```cmd
# 🎯 1단계: Miniconda 자동 설치
install.bat
# 메시지: "Install Miniconda automatically? (Y/n):" → Y 입력
# ✅ 설치 완료 후 "Please run this script again" 메시지 표시
# 🔄 터미널을 완전히 닫고 새로 열기

# 🎯 2단계: AI 도구 설치
install.bat
# ✅ 자동으로 모든 패키지 설치 완료
```

**Linux/macOS 사용자**:

```bash
# 🎯 1단계: Miniconda 자동 설치
./install.sh
# 메시지: "Install Miniconda automatically? (Y/n):" → Y 입력
# ✅ 설치 완료 후 "Please run this script again" 메시지 표시
# 🔄 터미널을 완전히 닫고 새로 열기

# 🎯 2단계: AI 도구 설치
./install.sh
# ✅ 자동으로 모든 패키지 설치 완료
```

#### 🔍 왜 2번 실행해야 하나요?

1. **1차 실행**: Miniconda를 시스템에 설치하고 PATH 환경변수를 설정
2. **터미널 재시작**: 새로운 PATH 설정이 적용되도록 필요
3. **2차 실행**: 이제 `conda` 명령어를 사용할 수 있어서 AI 도구 설치 진행

#### 💡 설치 확인 방법

설치가 완료되면 다음과 같은 메시지를 확인할 수 있습니다:

```
Installation completed successfully!

How to run:
   conda activate edgehd
   python app.py

Server URL: http://localhost:8080
```

### 🔄 기존 설치자 업데이트 (권장)

**📸 마지막 프레임 추출** 등 새로운 기능이 추가되었습니다! 기존 설치를 간단히 업데이트할 수 있습니다:

#### 🚀 간편 업데이트 (권장)

**Linux/macOS**:

```bash
# Git으로 최신 버전 받기
git pull origin main

# 의존성 업데이트 (새로운 패키지가 있는 경우)
conda activate edgehd
pip install -r requirements.txt --upgrade

# 서버 재시작
./stop.sh && ./start.sh
```

**Windows**:

```cmd
:: Git으로 최신 버전 받기
git pull origin main

:: 의존성 업데이트 (새로운 패키지가 있는 경우)
conda activate edgehd
pip install -r requirements.txt --upgrade

:: 서버 재시작
stop.bat && start.bat
```

#### 🔄 자동 업데이트 (Git이 없는 경우)

**Linux/macOS**:

```bash
# 기존 폴더 백업 (중요한 파일이 있는 경우)
cp -r EdgeHD EdgeHD_backup

# 새 버전 다운로드
wget https://github.com/mastergear4824/EdgeHD/archive/main.zip
unzip main.zip
cp -r EdgeHD-main/* EdgeHD/
rm -rf EdgeHD-main main.zip

# 업데이트 적용
cd EdgeHD
./install.sh  # 환경은 유지하고 새 기능만 추가
```

**Windows**:

```cmd
:: 기존 폴더 백업 (중요한 파일이 있는 경우)
xcopy EdgeHD EdgeHD_backup /E /I

:: 새 버전 다운로드 (PowerShell 사용)
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/mastergear4824/EdgeHD/archive/main.zip' -OutFile 'main.zip'"
powershell -Command "Expand-Archive -Path 'main.zip' -DestinationPath '.'"
xcopy EdgeHD-main\* EdgeHD\ /E /Y
rmdir /S /Q EdgeHD-main
del main.zip

:: 업데이트 적용
cd EdgeHD
install.bat
```

#### 📋 업데이트 확인 방법

업데이트가 완료되면 다음을 확인하세요:

```bash
# 서버 실행 후 웹 브라우저에서 확인
# http://localhost:8080

# 새로운 기능 확인:
# 1. 비디오 모드로 전환
# 2. 비디오 업로드
# 3. "📸 마지막 프레임 추출" 버튼 확인
```

#### ⚠️ 업데이트 문제 해결

업데이트 중 문제가 발생한 경우:

```bash
# 1. 기존 환경 정리 후 새로 설치
./uninstall.sh && ./install.sh  # Linux/macOS
# 또는
uninstall.bat && install.bat     # Windows

# 2. 백업에서 중요 파일 복원 (필요한 경우)
cp EdgeHD_backup/downloads/* EdgeHD/downloads/  # Linux/macOS
xcopy EdgeHD_backup\downloads\* EdgeHD\downloads\ /Y  # Windows
```

### 🗑️ 환경 완전 정리 (문제 해결용)

설치 오류나 버전 충돌 문제가 발생한 경우 환경을 완전히 정리할 수 있습니다:

#### Linux/macOS

```bash
# 환경 완전 정리
chmod +x uninstall.sh
./uninstall.sh

# 재설치
./install.sh
```

#### Windows

```cmd
:: 환경 완전 정리
uninstall.bat

:: 재설치
install.bat
```

**정리되는 항목**:

- Python 가상환경 (conda 환경)
- AI 모델 파일들
- 임시 파일 및 캐시
- 업로드/다운로드 파일들

### 🔧 수동 설치

간편 설치 스크립트를 사용하지 않는 경우 다음 단계를 수행하세요:

```bash
# Conda 환경 생성
conda create -n edgehd python=3.10 -y

# 환경 활성화
conda activate edgehd
```

### 3. PyTorch 설치 (최신 버전 - 자동 호환성)

**🚨 중요 업데이트**: PyTorch 2.6부터 공식 Anaconda 채널 지원이 중단되었습니다.

**✅ 권장 방법**: 기본 PyPI에서 최신 버전을 설치합니다. (Real-ESRGAN과 완전 호환)

#### 모든 플랫폼 (권장 - 자동 감지)

```bash
# 기본 설치 (GPU/CPU 자동 감지, 가장 안전)
pip install torch torchvision torchaudio
```

#### 특정 환경별 설치 (필요한 경우만)

```bash
# Apple Silicon (M1/M2/M3/M4 Mac) - MPS 지원
pip install torch torchvision torchaudio

# NVIDIA GPU (CUDA 12.1+) - 최신 CUDA 지원
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 전용 (이전 컴퓨터나 서버용)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### transformers 최신 버전 설치

```bash
# 최신 transformers (모든 Real-ESRGAN 버전과 호환)
pip install transformers
```

### 4. 필수 패키지 설치

```bash
# 최신 요구사항 설치
pip install -r requirements.txt

# AI 모델 의존성 패키지 설치
pip install einops>=0.6.0 kornia>=0.7.0 timm>=0.9.0 realesrgan==0.3.0
```

**📝 참고**: PyTorch와 transformers는 3단계에서 이미 설치했으므로 requirements.txt에서 제외되었습니다.

# 설치 확인

pip list | grep -E "(torch|transformers|einops|kornia|timm|realesrgan)"

````

### 5. 환경 변수 설정 (Apple Silicon)

```bash
# Apple Silicon MPS 폴백 활성화
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 영구 설정을 위해 ~/.zshrc 또는 ~/.bashrc에 추가
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
````

## 🎮 실행 방법

### 🚀 간편 실행 (권장)

#### Linux/macOS

```bash
# 백그라운드 서버 시작
./start.sh

# 서버 상태 확인
./status.sh

# 서버 중지
./stop.sh
```

#### Windows

```cmd
:: 백그라운드 서버 시작
start.bat

:: 서버 상태 확인
status.bat

:: 서버 중지
stop.bat
```

### 개발 서버 실행

```bash
# 환경 활성화
conda activate edgehd

# Apple Silicon에서 환경 변수 설정
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 애플리케이션 실행
python app.py
```

### 원라인 실행 명령어

```bash
# 최신 버전 업데이트 후 실행 (기존 사용자 권장)
git pull origin main && conda activate edgehd && pip install -r requirements.txt --upgrade && ./stop.sh && ./start.sh

# 완전 재설치 후 실행 (문제 해결용)
./uninstall.sh && ./install.sh && ./start.sh

# 자동 설치 후 실행
./install.sh && ./start.sh

# 또는 수동 실행
cd /path/to/EdgeHD && conda activate edgehd && export PYTORCH_ENABLE_MPS_FALLBACK=1 && python app.py
```

### 서버 접속

애플리케이션이 시작되면 다음 주소로 접속할 수 있습니다:

- **로컬**: http://localhost:8080
- **네트워크**: http://192.168.x.x:8080 (터미널에 표시되는 IP)

## 📖 사용법

### 🖼️ 이미지 처리 워크플로우

1. **이미지 업로드**:

   - 지원 형식: PNG, JPG, JPEG, GIF, BMP, WebP
   - 최대 크기: 16MB

2. **처리 선택**:

   - `4x업스케일` (빨간색): Real-ESRGAN General v3으로 이미지를 4배로 확대
   - `배경제거` (보라색): BiRefNet AI를 사용한 고품질 배경 제거
   - ⚠️ v0.3.0에서는 2x 전용 모델이 없어 4x만 지원

3. **연속 처리**:

   - 결과 이미지에서 추가 처리 가능
   - 배경 제거된 이미지는 업스케일만 가능
   - 자유로운 조합으로 최적 결과 달성

4. **결과 다운로드**:
   - 우클릭으로 이미지 저장
   - 투명 배경은 체크보드 패턴으로 표시

### 🎬 비디오 처리 워크플로우 (NEW!)

1. **비디오 업로드**:

   - 지원 형식: MP4, AVI, MOV, MKV
   - 최대 크기: 100MB
   - 비디오 정보 자동 표시 (해상도, FPS, 재생시간)

2. **처리 옵션 선택**:

   - `📸 마지막 프레임 추출`: 비디오의 마지막 프레임을 PNG 이미지로 추출
   - `배경 제거`: 각 프레임에 BiRefNet AI 배경 제거 적용
   - `4x 업스케일`: Real-ESRGAN General v3으로 비디오 해상도를 4배로 향상
   - `배경제거 + 4x`: 배경 제거와 4배 업스케일 동시 적용
   - ⚠️ v0.3.0에서는 2x 전용 모델이 없어 4x만 지원

3. **실시간 처리 상태 확인**:

   - 현재 처리 중인 프레임 번호 표시
   - 남은 시간 실시간 계산
   - 재미있는 처리 메시지와 팁 표시
   - 전체 진행률 바 표시

4. **결과 다운로드**:
   - **마지막 프레임 추출**: PNG 이미지 파일 다운로드 (원본 해상도 유지)
   - **비디오 처리**: 처리 완료 후 MP4 파일 다운로드
   - 원본 FPS 유지
   - 처리 옵션에 따른 해상도 조정

### 📸 마지막 프레임 추출 기능 (NEW!)

비디오에서 마지막 프레임을 이미지로 추출하는 새로운 기능이 추가되었습니다!

#### 주요 특징

- **🎯 정확한 추출**: 비디오의 마지막 유효한 프레임을 안전하게 추출
- **📸 고품질 이미지**: 원본 해상도 유지로 PNG 형식 저장
- **⚡ 빠른 처리**: 전체 비디오 처리 없이 마지막 프레임만 추출
- **🛡️ 안전한 처리**: 손상된 프레임 감지 시 자동으로 이전 유효 프레임 선택

#### 사용법

1. 비디오 파일 업로드 (MP4, AVI, MOV, MKV)
2. "📸 마지막 프레임 추출" 버튼 클릭
3. 처리 완료 후 우측 패널에서 결과 확인
4. PNG 이미지로 다운로드

#### 활용 사례

- 비디오 썸네일 생성
- 영상 종료 화면 캡처
- 스크린샷 추출
- 콘텐츠 미리보기 생성

### 📁 지원 형식

#### 이미지

- **입력**: BMP, GIF, JPEG, JPG, PNG, WebP
- **출력**: PNG (투명 배경 지원)

#### 비디오

- **입력**: MP4, AVI, MOV, MKV
- **출력**:
  - **비디오 처리**: MP4 (H.264 코덱)
  - **마지막 프레임 추출**: PNG (원본 해상도)

## ⚡ 성능 최적화

### GPU 가속

- **Apple Silicon**: MPS 백엔드 자동 사용
- **NVIDIA GPU**: CUDA 자동 감지
- **CPU 폴백**: GPU 미지원시 자동 전환

### 메모리 관리

```python
# 대용량 이미지 처리시 메모리 부족 방지
import torch
torch.cuda.empty_cache()  # CUDA
torch.mps.empty_cache()   # MPS
```

## 🔧 문제 해결

### 일반적인 오류

#### 1. PyTorch 버전 호환성 오류 ⚠️

**증상**: `Real-ESRGAN model loading error` 또는 `SRVGGNetCompact architecture mismatch`

**해결방법**:

```bash
# 환경 완전 정리 후 재설치
./uninstall.sh  # 또는 uninstall.bat
./install.sh    # 또는 install.bat

# 또는 수동 업데이트 (Real-ESRGAN v0.3.0 호환 버전)
pip uninstall torch torchvision torchaudio transformers realesrgan -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install transformers==4.35.0 realesrgan==0.3.0
```

#### 2. PyTorch MPS 오류 (Apple Silicon)

```bash
# 환경 변수 설정 확인
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 3. 메모리 부족 오류

```bash
# 더 작은 이미지로 테스트 또는 시스템 메모리 확인
# 16MB 제한 준수
```

#### 4. 모델 다운로드 실패

```bash
# 인터넷 연결 확인 및 Hugging Face 접근 확인
# VPN 사용시 비활성화 후 재시도
```

#### 5. Import 오류

```bash
# 의존성 재설치 (Real-ESRGAN v0.3.0 호환 버전)
pip uninstall -r requirements.txt -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 transformers==4.35.0
pip install -r requirements.txt
```

### 로그 확인

애플리케이션 실행시 다음과 같은 로그를 확인할 수 있습니다:

```
🚀 고품질 AI 배경 제거 웹 애플리케이션을 시작합니다...
🔥 iOS 수준 고품질 배경 제거 (85% 성공률)
📁 지원 형식: bmp, gif, jpeg, jpg, png, webp
📏 최대 파일 크기: 16MB
🌐 서버 주소: http://localhost:8080
🔄 고품질 AI 모델 미리 로드 중...
🍎 Apple Silicon GPU(MPS) 사용
🚀 고품질 배경 제거 모델 로드 완료!
✅ 고품질 AI 모델 준비 완료!
```

## 🔧 서버 관리

### 관리 스크립트

프로젝트에는 서버 관리를 위한 편리한 스크립트들이 포함되어 있습니다:

#### 🚀 서버 시작

**Linux/macOS**

```bash
./start.sh
```

**Windows**

```cmd
start.bat
```

기능:

- 백그라운드에서 서버 실행
- PID 파일 생성 및 관리
- 로그 파일 자동 생성
- 환경 변수 자동 설정 (Apple Silicon MPS)
- 중복 실행 방지
- 비디오 처리 기능 안내

#### 📊 서버 상태 확인

**Linux/macOS**

```bash
./status.sh
```

**Windows**

```cmd
status.bat
```

기능:

- 서버 실행 상태 확인
- CPU/메모리 사용률 표시
- 포트 정보 확인
- 로그 파일 크기 정보
- 애플리케이션 기능 정보 (이미지/비디오 처리)
- 유용한 명령어 안내

#### 🛑 서버 중지

**Linux/macOS**

```bash
./stop.sh
```

**Windows**

```cmd
stop.bat
```

기능:

- 안전한 서버 종료 (SIGTERM/taskkill)
- 강제 종료 옵션 (SIGKILL/taskkill /F)
- PID 파일 자동 정리
- 로그 파일 정리 옵션

### 로그 관리

#### Linux/macOS

```bash
# 실시간 로그 확인
tail -f app.log

# 에러 로그 확인
tail -f app_error.log

# 로그 파일 크기 확인
ls -lh *.log
```

#### Windows

```cmd
:: 실시간 로그 확인
powershell "Get-Content app.log -Wait"

:: 에러 로그 확인
type app_error.log

:: 로그 파일 크기 확인
dir *.log
```

## 📁 프로젝트 구조

```
EdgeHD/
├── app.py                 # 메인 Flask 애플리케이션
│                         # • /upload (이미지 배경 제거)
│                         # • /upscale (이미지 업스케일링)
│                         # • /process_video (비디오 프레임별 처리)
│                         # • /extract_last_frame (마지막 프레임 추출) NEW!
│                         # • /download/<filename> (파일 다운로드)
├── models/               # 🤖 AI 모델 저장소 (프로젝트 독립)
│   └── hub/             # Hugging Face 모델 캐시
│       └── models--zhengpeng7--BiRefNet/  # 배경 제거 모델 (~424MB)
├── templates/            # HTML 템플릿
│   └── index.html       # 메인 웹 인터페이스 (마지막 프레임 추출 UI 포함)
├── static/              # 정적 파일 (CSS, JS)
├── uploads/             # 업로드된 파일
├── downloads/           # 처리된 결과 파일 (이미지, 비디오, 추출된 프레임)
├── temp/               # 임시 처리 파일 (비디오 프레임 분해용)
├── requirements.txt     # Python 의존성 (PyTorch 2.1.0, transformers 4.35.0)
├── install.sh          # Linux/macOS 설치 스크립트
├── install.bat         # Windows 설치 스크립트
├── uninstall.sh        # Linux/macOS 환경 정리 스크립트
├── uninstall.bat       # Windows 환경 정리 스크립트
├── start.sh/.bat       # 서버 시작 스크립트
├── stop.sh/.bat        # 서버 중지 스크립트
└── status.sh/.bat      # 서버 상태 확인 스크립트
```

## 🤖 AI 모델 관리

### 프로젝트 독립 모델 저장

- **모든 AI 모델이 프로젝트 내 `models/` 디렉토리에 저장됩니다**
- 시스템 전역 캐시(`~/.cache/huggingface/`)가 아닌 프로젝트 로컬 저장
- 프로젝트 이동 시 모델도 함께 이동되어 완전한 독립성 보장

### 첫 실행 시 모델 다운로드

첫 실행 시 자동으로 다음 모델들이 다운로드됩니다:

- **BiRefNet** (zhengpeng7/BiRefNet): 고품질 배경 제거 (~424MB)
- **Real-ESRGAN General v3** (v0.3.0): AI 4배 업스케일링 (~17MB)
- ⚠️ **v0.3.0에서는 2x 전용 모델이 없어 4x만 지원**

### 모델 저장 위치

```bash
models/
├── hub/
│   └── models--zhengpeng7--BiRefNet/
│       ├── snapshots/
│       │   └── [commit-hash]/
│       │       ├── model.safetensors    # 메인 모델 가중치
│       │       ├── config.json          # 모델 설정
│       │       ├── birefnet.py         # 모델 코드
│       │       └── BiRefNet_config.py   # 설정 파일
│       ├── blobs/
│       └── refs/
└── realesrgan/
    └── realesr-general-x4v3.pth    # General v3 4x 업스케일링 모델 (v0.3.0)
```

### Git 버전 관리

- `models/` 디렉토리는 `.gitignore`에 포함되어 Git에서 제외
- 모델 파일들이 GitHub에 업로드되지 않음 (용량 문제 방지)
- 각 환경에서 첫 실행 시 자동 다운로드

## 🔒 보안 고려사항

- 파일 크기 제한: 16MB
- 허용된 파일 형식만 업로드 가능
- 업로드 파일명 보안 처리
- 임시 파일 자동 정리

## 🚀 프로덕션 배포

개발 서버는 프로덕션 환경에 적합하지 않습니다. 프로덕션 배포시:

```bash
# Gunicorn 설치
pip install gunicorn

# 프로덕션 서버 실행
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다.

### 주요 사용 구성요소 라이선스

- **BiRefNet**: MIT 라이선스 ✅
- **PyTorch**: BSD-3-Clause 라이선스 ✅
- **Transformers**: Apache 2.0 라이선스 ✅
- **OpenCV**: Apache 2.0 라이선스 ✅
- **Pillow**: HPND 라이선스 ✅
- **Flask**: BSD-3-Clause 라이선스 ✅

모든 구성요소가 MIT 라이선스와 호환되므로 상업적 사용, 수정, 배포가 자유롭게 가능합니다.

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🛠️ ARM64 Windows 전용 가이드

> **Surface Pro X, Copilot+ PC, ARM64 Windows 사용자 전용**

### 🎯 PyTorch ARM64 Windows 지원 소식

**🎉 중요 업데이트**: PyTorch 2.7부터 ARM64 Windows를 공식 지원합니다! (2025년 4월 발표)

### 📋 전제 조건

#### 1. Visual Studio Build Tools 설치

```cmd
# https://visualstudio.microsoft.com/visual-cpp-build-tools/ 에서 다운로드
# 설치 시 "Desktop development with C++" 워크로드 선택
# ARM64 관련 구성 요소들도 함께 설치됨
```

#### 2. Rust 설치

```cmd
# https://win.rustup.rs/x86_64 에서 다운로드 후 실행:
rustup-init.exe --default-toolchain stable --default-host aarch64-pc-windows-msvc
```

#### 3. Python 3.12 ARM64 설치

```cmd
# https://www.python.org/downloads/ 에서 ARM64 installer 다운로드
# "Windows installer (ARM64)" 선택하여 설치
```

### 🚀 설치 단계

#### 1. 새로운 Conda 환경 생성

```cmd
# Python 3.12로 새 환경 생성
conda create -n edgehd-arm python=3.12 -y
conda activate edgehd-arm
```

#### 2. PyTorch ARM64 설치

```cmd
# ARM64 Windows용 PyTorch 설치
pip install --extra-index-url https://download.pytorch.org/whl torch torchvision torchaudio

# 성공 확인
python -c "import torch; print(f'PyTorch {torch.__version__} on ARM64 Windows')"
```

#### 3. 나머지 패키지 설치

```cmd
# EdgeHD 프로젝트 폴더에서
pip install transformers einops kornia timm realesrgan==0.3.0
pip install Flask Flask-CORS opencv-python Pillow requests
```

#### 4. 실행

```cmd
# 환경 활성화 후 서버 시작
conda activate edgehd-arm
python app.py
```

### 📚 참고 자료

- [Microsoft 공식 발표](https://blogs.windows.com/windowsdeveloper/2025/04/23/pytorch-arm-native-builds-now-available-for-windows/)
- [ARM PyTorch 설치 가이드](https://learn.arm.com/install-guides/pytorch/)
- [ExecuTorch ARM64 최적화](https://pytorch.org/blog/unleashing-ai-mobile/)

## 🤝 기여

버그 리포트와 기능 요청은 이슈를 통해 제출해 주세요.

---

**🌟 특징**

- 🖼️ **이미지 처리**: 고품질 AI 배경 제거 및 업스케일링
- 🎬 **비디오 처리**: 프레임별 AI 처리 및 재조립
- 📸 **마지막 프레임 추출**: 비디오에서 마지막 프레임을 이미지로 빠르게 추출 (NEW!)
- 🖥️ **크로스 플랫폼**: Windows, macOS, Linux 완벽 지원
- ⚡ **실시간 진행률**: 처리 상태 실시간 확인
- 🚀 **간편 설치**: 원클릭 설치 스크립트 제공

**Created with ❤️ using BiRefNet, Flask, and OpenCV**
