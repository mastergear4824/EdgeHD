# EdgeHD 2.0 - AI 이미지/비디오 처리 플랫폼

AI 기반 이미지 및 비디오 처리를 위한 모던 풀스택 웹 애플리케이션입니다.

## 🚀 주요 기능

### 이미지 처리

- **AI 배경 제거**: BiRefNet 모델을 사용한 고품질 배경 제거
- **AI 업스케일링**: Real-ESRGAN을 사용한 2x, 4x 해상도 향상
- **벡터화**: 이미지를 SVG 벡터 형식으로 변환

### 비디오 처리

- **비디오 배경 제거**: 프레임별 배경 제거 처리
- **마지막 프레임 추출**: 비디오에서 마지막 프레임을 이미지로 추출

## 🏗️ 아키텍처

```
EdgeHD/
├── backend/              # Flask API 서버
│   ├── app.py           # 메인 API 서버
│   ├── config.py        # 설정 파일
│   ├── modules/         # AI 모델 모듈들
│   ├── models/          # AI 모델 파일들
│   ├── uploads/         # 업로드된 파일들
│   ├── downloads/       # 처리된 파일들
│   ├── temp/           # 임시 파일들
│   └── requirements.txt # Python 의존성
├── frontend/            # Next.js + shadcn/ui 프론트엔드
│   ├── src/
│   │   ├── app/        # App Router
│   │   ├── components/ # UI 컴포넌트
│   │   └── lib/        # 유틸리티
│   ├── public/         # 정적 파일들
│   └── package.json    # Node.js 의존성
├── package.json         # 루트 스크립트 관리
└── README.md           # 프로젝트 문서
```

## 🛠️ 기술 스택

### 백엔드

- **Python 3.11** - 메인 언어
- **Flask 3.0** - REST API 서버
- **PyTorch 2.1.0** - AI 모델 실행
- **OpenCV** - 이미지/비디오 처리
- **PIL/Pillow** - 이미지 조작
- **BiRefNet** - AI 배경 제거
- **Real-ESRGAN** - AI 업스케일링

### 프론트엔드

- **Next.js 14** - React 프레임워크 (App Router)
- **TypeScript** - 타입 안전성
- **Tailwind CSS** - 유틸리티 기반 스타일링
- **shadcn/ui** - 모던 UI 컴포넌트 라이브러리
- **Lucide React** - 아이콘 라이브러리
- **React Hook Form** - 폼 관리

### 개발 도구

- **concurrently** - 백엔드/프론트엔드 동시 실행
- **ESLint** - 코드 품질 관리
- **Prettier** - 코드 포맷팅

## 📦 설치 및 실행

### 시스템 요구사항

- **Python 3.11+** (Conda/Miniconda 권장)
- **Node.js 18+** (LTS 버전 권장)
- **npm 9+**
- **Git**

### 1. 저장소 클론

```bash
git clone <repository-url>
cd EdgeHD
```

### 2. 자동 설치 (권장)

#### Windows

```bash
# 관리자 권한으로 실행 권장
install.bat
```

#### macOS/Linux

```bash
chmod +x install.sh
./install.sh
```

자동 설치 스크립트는 다음을 수행합니다:

- Conda/Miniconda 설치 (필요시)
- Python 환경 생성 (`edgehd`)
- PyTorch 2.1.0 설치 (시스템별 최적화)
- 백엔드 Python 의존성 설치
- Node.js 의존성 설치
- AI 모델 저장소 설정

### 3. 수동 설치 (고급 사용자)

#### 백엔드 설정

```bash
# Conda 환경 생성
conda create -n edgehd python=3.11 -y
conda activate edgehd

# PyTorch 설치 (시스템에 맞게 선택)
# CPU 버전
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# CUDA 버전 (NVIDIA GPU)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 백엔드 의존성 설치
cd backend
pip install -r requirements.txt
cd ..
```

#### 프론트엔드 설정

```bash
# 프론트엔드 의존성 설치
cd frontend
npm install
cd ..

# 루트 의존성 설치
npm install
```

## 🚀 실행 방법

### 개발 모드

#### 통합 실행 (권장)

```bash
# 백엔드 + 프론트엔드 동시 실행
npm run dev

# 또는 스크립트 사용
# Windows
run.bat

# macOS/Linux
./run.sh
```

#### 개별 실행

```bash
# 백엔드만 실행 (http://localhost:8080)
npm run dev:backend

# 프론트엔드만 실행 (http://localhost:3000)
npm run dev:frontend
```

### 프로덕션 모드

#### 백그라운드 실행

```bash
# Windows
start.bat

# macOS/Linux
./start.sh
```

#### 빌드 및 실행

```bash
# 프론트엔드 빌드
npm run build

# 프로덕션 서버 실행
npm run start
```

## 🔧 환경 설정

### 백엔드 설정 (backend/config.py)

```python
# 서버 설정
HOST = '0.0.0.0'
PORT = 8080
DEBUG = True

# CORS 설정
FRONTEND_URL = 'http://localhost:3000'
ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]

# 파일 크기 제한
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
```

### 프론트엔드 설정

프론트엔드는 `http://localhost:3000`에서 실행되며, 백엔드 API(`http://localhost:8080`)와 통신합니다.

환경변수 설정 (선택사항):

```bash
# .env.local 파일 생성
NEXT_PUBLIC_API_URL=http://localhost:8080
```

## 📡 API 엔드포인트

### 이미지 처리

- `POST /api/upload` - 배경 제거
- `POST /api/upscale` - 이미지 업스케일링 (2x, 4x)
- `POST /api/vectorize` - 이미지 벡터화

### 비디오 처리

- `POST /api/process_video` - 비디오 배경 제거
- `POST /api/extract_last_frame` - 마지막 프레임 추출

### 유틸리티

- `GET /api/health` - 서버 상태 확인
- `GET /api/system-info` - 시스템 정보
- `GET /api/sessions` - 세션 목록
- `GET /api/download/<filename>` - 파일 다운로드

## 🎨 UI 특징

shadcn/ui 기반의 모던한 사용자 인터페이스:

- **드래그 앤 드롭** 파일 업로드
- **실시간 진행률** 표시
- **탭 기반** 이미지/비디오 처리 구분
- **반응형 디자인** (모바일 지원)
- **토스트 알림** 사용자 피드백
- **다크/라이트 모드** 지원
- **접근성** 최적화

## 🔍 관리 및 모니터링

### 상태 확인

```bash
# Windows
status.bat

# macOS/Linux
./status.sh
```

### 서버 중지

```bash
# Windows
stop.bat

# macOS/Linux
./stop.sh
```

### 로그 확인

```bash
# 백엔드 로그
tail -f backend.log

# 프론트엔드 로그
tail -f frontend.log

# 에러 로그
tail -f backend_error.log
tail -f frontend_error.log
```

## 🚀 배포

### Docker (선택사항)

```dockerfile
# Dockerfile 예시
FROM node:18-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ ./backend/
COPY --from=frontend /app/frontend/out ./frontend/out

EXPOSE 8080 3000
CMD ["python", "backend/app.py"]
```

### 클라우드 배포

- **Vercel** (프론트엔드)
- **Railway/Render** (백엔드)
- **AWS/GCP/Azure** (풀스택)

## 🛠️ 개발

### 프로젝트 구조

```
src/
├── app/                 # Next.js App Router
│   ├── layout.tsx      # 루트 레이아웃
│   ├── page.tsx        # 메인 페이지
│   └── globals.css     # 글로벌 스타일
├── components/         # React 컴포넌트
│   └── ui/            # shadcn/ui 컴포넌트
└── lib/               # 유틸리티 함수
    └── utils.ts       # 공통 유틸리티
```

### 개발 가이드라인

- **TypeScript** 사용 필수
- **ESLint** 규칙 준수
- **shadcn/ui** 컴포넌트 활용
- **Tailwind CSS** 스타일링
- **React Hook Form** 폼 관리

## 🔧 문제 해결

### 일반적인 문제

#### 1. Conda 환경 활성화 실패

```bash
# 수동 활성화
conda activate edgehd
```

#### 2. 포트 충돌

```bash
# 포트 사용 확인
netstat -an | grep :8080
netstat -an | grep :3000

# 프로세스 종료
kill -9 <PID>
```

#### 3. AI 모델 다운로드 실패

```bash
# 환경변수 확인
echo $HF_HOME
echo $TRANSFORMERS_CACHE

# 수동 설정
export HF_HOME="$(pwd)/backend/models"
export TRANSFORMERS_CACHE="$(pwd)/backend/models"
```

#### 4. 메모리 부족

- GPU 메모리: 최소 4GB 권장
- 시스템 메모리: 최소 8GB 권장

### 로그 분석

```bash
# 상세 로그 확인
tail -f backend.log | grep ERROR
tail -f frontend.log | grep ERROR
```

## 📊 성능 최적화

### 백엔드

- **GPU 가속** (CUDA/MPS 지원)
- **모델 캐싱** (첫 실행 후 빠른 로딩)
- **배치 처리** (여러 이미지 동시 처리)

### 프론트엔드

- **Next.js 최적화** (이미지, 폰트, 번들)
- **코드 분할** (동적 임포트)
- **캐싱 전략** (SWR/React Query)

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 개발 환경 설정

```bash
# 개발 의존성 설치
npm install --dev

# 린터 실행
npm run lint

# 타입 체크
npm run type-check
```

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 감사의 말

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - 고품질 배경 제거
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI 업스케일링
- [shadcn/ui](https://ui.shadcn.com/) - UI 컴포넌트 라이브러리
- [Next.js](https://nextjs.org/) - React 프레임워크
- [Flask](https://flask.palletsprojects.com/) - Python 웹 프레임워크
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. **GitHub Issues** - 버그 리포트 및 기능 요청
2. **Discussions** - 일반적인 질문 및 토론
3. **Wiki** - 상세한 문서 및 가이드

---

**EdgeHD 2.0** - AI로 더 나은 이미지/비디오 처리 경험을 제공합니다. 🚀
