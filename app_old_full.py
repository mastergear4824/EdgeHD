from flask import Flask, request, render_template, send_file, jsonify, url_for, Response
from flask_cors import CORS

# 경고 메시지 필터링
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# 기본 라이브러리들
from PIL import Image
import os
import uuid
import io
import json
import time
import threading
from werkzeug.utils import secure_filename
from queue import Queue

# 새로운 모듈들 import
from modules.background_removal import BiRefNetModel
from modules.upscaling import RealESRGANUpscaleModel
from modules.vectorization import ImageVectorizerModel
from modules.utils import generate_filename, generate_unique_id, safe_filename, ensure_directory, cleanup_temp_files

# 프로젝트 내 모델 디렉토리 설정
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face 캐시를 프로젝트 내로 설정
os.environ['HF_HOME'] = MODEL_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 실제 배포시에는 환경변수로 설정
CORS(app)

# 설정
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
TEMP_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_VECTOR_EXTENSIONS = {'svg', 'pdf'}  # 벡터 출력 형식
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB (이미지용)
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB (비디오용)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE  # 비디오 파일 크기로 확장

# 업로드, 다운로드, 임시 디렉토리 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# BiRefNetModel 클래스는 modules/background_removal.py로 이동됨

# RealESRGANUpscaleModel 클래스는 modules/upscaling.py로 이동됨
#
    """Real-ESRGAN AI 업스케일링 모델"""
    
    def __init__(self):
        self.loaded = False
        self.model_2x = None
        self.model_4x = None
        print("🔥 Real-ESRGAN AI 업스케일링 활성화!")
        
    def load_model(self, scale=4, progress_callback=None):
        """Real-ESRGAN 모델 로드"""
        try:
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            if scale == 2:
                # v0.3.0에서는 2x 전용 모델이 없으므로 PIL LANCZOS 폴백
                if progress_callback:
                    progress_callback(50, "⚠️ v0.3.0에서는 2x 전용 모델이 없어 기본 방식 사용")
                print("⚠️ Real-ESRGAN v0.3.0에는 2x 전용 모델이 없습니다. PIL LANCZOS 사용")
                return None
                
            elif scale == 4 and self.model_4x is None:
                if progress_callback:
                    progress_callback(20, "🔥 Real-ESRGAN General v3 4x 모델 로딩 중...")
                
                self.model_4x = RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
                    model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=device
                )
                
                if progress_callback:
                    progress_callback(100, "✅ Real-ESRGAN General v3 4x 모델 로드 완료!")
                    
                print("📦 Real-ESRGAN General v3 4x 모델 로드 완료!")
                
            self.loaded = True
            return self.model_2x if scale == 2 else self.model_4x
            
        except Exception as e:
            print(f"❌ Real-ESRGAN 모델 로드 실패, PIL LANCZOS로 폴백: {e}")
            if progress_callback:
                progress_callback(50, f"⚠️ AI 모델 로드 실패, 기본 방식 사용")
            return None
        
    def upscale_image(self, image, scale=4, progress_callback=None):
        """Real-ESRGAN을 사용한 AI 업스케일링 (실패시 PIL 폴백)"""
        try:
            # 먼저 Real-ESRGAN 모델 로드 시도
            model = self.load_model(scale, progress_callback)
            
            if model is not None:
                if progress_callback:
                    progress_callback(50, f"🤖 Real-ESRGAN {scale}x AI 업스케일링 중...")
                
                # PIL Image를 numpy array로 변환
                img_array = np.array(image)
                
                # Real-ESRGAN으로 업스케일링
                output, _ = model.enhance(img_array, outscale=scale)
                
                # numpy array를 PIL Image로 변환
                upscaled_image = Image.fromarray(output)
                
                if progress_callback:
                    progress_callback(100, f"✅ Real-ESRGAN {scale}x AI 업스케일링 완료!")
                    
                original_width, original_height = image.size
                new_width, new_height = upscaled_image.size
                print(f"🤖 Real-ESRGAN {scale}x AI 업스케일링 완료: {original_width}x{original_height} → {new_width}x{new_height}")
                return upscaled_image
            else:
                # Real-ESRGAN 실패시 PIL LANCZOS 폴백
                return self._lanczos_fallback(image, scale, progress_callback)
                
        except Exception as e:
            print(f"❌ Real-ESRGAN 업스케일링 실패, PIL LANCZOS로 폴백: {e}")
            return self._lanczos_fallback(image, scale, progress_callback)
            
    def _lanczos_fallback(self, image, scale, progress_callback=None):
        """PIL LANCZOS 폴백 업스케일링"""
        try:
            if progress_callback:
                progress_callback(20, f"🔧 {scale}x 기본 업스케일링 중...")
            
            original_width, original_height = image.size
            new_width = original_width * scale
            new_height = original_height * scale
            upscaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if progress_callback:
                progress_callback(100, f"✅ {scale}x 업스케일링 완료 (기본 방식)")
            
            print(f"📝 PIL LANCZOS {scale}x 업스케일링 완료: {original_width}x{original_height} → {new_width}x{new_height}")
            return upscaled_image
            
        except Exception as e:
            print(f"❌ 업스케일링 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 업스케일링 실패: {str(e)}")
            raise

# Neural AI 클래스 제거됨 - Potrace만 사용
    
    def neural_vectorize(self, image_path, output_path, **kwargs):
        """신경망 기반 전문 벡터화"""
        try:
            print("🧠 전문 Neural AI 벡터화 시작!")
            # API 체크를 우선 스킵하고 로컬 Neural 알고리즘 사용
            # if not self.api_available:
            #     return False
                
            print("🎯 전문 Neural 벡터화 처리 중...")
            
            # Neural Rasterization 시뮬레이션 (실제로는 API 호출)
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {
                    'output_format': 'svg',
                    'mode': 'neural',
                    'quality': 'high',
                    'colors': kwargs.get('n_colors', 32)
                }
                
                # 실제 API 호출 (현재는 시뮬레이션)
                # response = requests.post('https://api.vectorizer.ai/v1/vectorize', files=files, data=data)
                
                # 시뮬레이션: 고품질 SVG 생성
                return self._create_neural_svg(image_path, output_path, **kwargs)
                
        except Exception as e:
            print(f"❌ 전문 벡터화 실패: {e}")
            return False
    
    def _create_neural_svg(self, image_path, output_path, **kwargs):
        """Neural-inspired 고품질 벡터화 (AI 자동 색상 분석)"""
        try:
            from PIL import Image
            import numpy as np
            
            image = Image.open(image_path)
            width, height = image.size
            
            # AI 자동 색상 분석
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3).astype(np.float32)
            
            # 🌈 원본 색상 보존 모드: K-means 양자화 없이 실제 색상 사용
            use_original_colors = True
            
            if use_original_colors:
                print("🌈 원본 색상 완전 보존 모드 활성화!")
                # 실제 이미지의 모든 고유 색상 추출
                unique_pixels = np.unique(pixels, axis=0)
                actual_colors = len(unique_pixels)
                print(f"🎨 원본 이미지의 실제 고유 색상: {actual_colors:,}개")
                
                # 너무 많은 색상이면 주요 색상만 선별 (성능 고려)
                if actual_colors > 1000:
                    print(f"🔧 성능 최적화: {actual_colors:,}개 → 주요 색상 500개 선별")
                    # 색상별 픽셀 수 계산하여 상위 색상 선택
                    color_counts = {}
                    for pixel in pixels:
                        color_key = tuple(pixel)
                        color_counts[color_key] = color_counts.get(color_key, 0) + 1
                    
                    # 빈도순으로 정렬해서 상위 500개 색상 선택
                    top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:500]
                    centers = np.array([list(color) for color, _ in top_colors])
                    n_colors = len(centers)
                else:
                    # 모든 고유 색상 사용
                    centers = unique_pixels
                    n_colors = actual_colors
                    
                print(f"✨ 최종 사용 색상: {n_colors}개 (원본 보존)")
            else:
                # 기존 K-means 방식 (백업용)
                optimal_colors = self._analyze_optimal_colors(img_array)
                print(f"🎨 AI 분석 결과: {optimal_colors}색상이 최적으로 판단됨")
                
                n_colors = optimal_colors
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
                _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            
            # Neural-style 경로 생성
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    .neural-path {{ fill-opacity: 0.95; stroke-width: 0.5; }}
  </style>
</defs>
<rect width="{width}" height="{height}" fill="white"/>
'''
            
            # 원본 색상 보존 모드에 따른 처리
            if use_original_colors:
                # 원본 색상 그대로 사용
                processed_image = img_array.astype(np.uint8)
                print("🔍 원본 픽셀 색상으로 직접 벡터 경로 생성 중...")
            else:
                # K-means 양자화된 색상 사용
                quantized_image = centers[labels.flatten()].reshape(img_array.shape).astype(np.uint8)
                processed_image = quantized_image
            
            for i, color in enumerate(centers.astype(np.uint8)):
                # 특정 색상 마스크 생성 (원본 보존 모드에 맞게)
                if use_original_colors:
                    # 원본 이미지에서 해당 색상 찾기
                    color_mask = np.all(processed_image == color, axis=2)
                else:
                    # 양자화된 이미지에서 해당 색상 찾기
                    color_mask = np.all(processed_image == color, axis=2)
                
                if np.sum(color_mask) > 100:  # 충분한 픽셀이 있는 경우만
                    # 컨투어 찾기
                    contours, _ = cv2.findContours(color_mask.astype(np.uint8), 
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 50:
                            # Bezier curve 근사 (Neural-style smoothing)
                            epsilon = 0.002 * cv2.arcLength(contour, True)
                            smoothed = cv2.approxPolyDP(contour, epsilon, True)
                            
                            if len(smoothed) > 2:
                                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                                path_data = self._create_smooth_path(smoothed)
                                
                                svg_content += f'<path class="neural-path" d="{path_data}" fill="{hex_color}" stroke="{hex_color}"/>\n'
            
            svg_content += '</svg>'
            
            # SVG 최적화
            svg_content = self._optimize_neural_svg(svg_content)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            print(f"🎯 전문 벡터화 완료: {len(svg_content)} 문자")
            return True
            
        except Exception as e:
            print(f"❌ Neural 벡터화 실패: {e}")
            return False
    
    def _create_smooth_path(self, contour):
        """베지어 곡선 기반 부드러운 경로 생성"""
        if len(contour) < 3:
            return ""
            
        path_data = f"M {contour[0][0][0]},{contour[0][0][1]}"
        
        for i in range(1, len(contour)):
            current = contour[i][0]
            prev = contour[i-1][0]
            
            # 부드러운 곡선을 위한 제어점 계산
            if i < len(contour) - 1:
                next_point = contour[i+1][0]
                cp1_x = prev[0] + (current[0] - prev[0]) * 0.5
                cp1_y = prev[1] + (current[1] - prev[1]) * 0.5
                cp2_x = current[0] + (next_point[0] - current[0]) * 0.3
                cp2_y = current[1] + (next_point[1] - current[1]) * 0.3
                
                path_data += f" Q {cp1_x:.1f},{cp1_y:.1f} {current[0]},{current[1]}"
            else:
                path_data += f" L {current[0]},{current[1]}"
        
        path_data += " Z"
        return path_data
    
    def _analyze_optimal_colors(self, img_array):
        """🧠 AI가 이미지를 분석해서 원본에 가까운 색상 수 자동 결정"""
        try:
            # 1. 실제 고유 색상 수 분석 (더 정확하게)
            pixels = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(pixels, axis=0))
            
            # 2. 색상 히스토그램 분석 (RGB 각 채널)
            hist_r = cv2.calcHist([img_array], [0], None, [256], [0,256])
            hist_g = cv2.calcHist([img_array], [1], None, [256], [0,256]) 
            hist_b = cv2.calcHist([img_array], [2], None, [256], [0,256])
            
            # 활성 색상 구간 계산 (0이 아닌 히스토그램 빈도)
            active_r = np.count_nonzero(hist_r)
            active_g = np.count_nonzero(hist_g)
            active_b = np.count_nonzero(hist_b)
            color_richness = (active_r + active_g + active_b) / (3 * 256)
            
            # 3. 색상 분산 (높을수록 다채로운 이미지)
            color_variance = np.mean(np.var(pixels.astype(np.float32), axis=0))
            
            # 4. 엣지 복잡도
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 🧠 원본 보존형 AI 결정 알고리즘
            # 실제 색상 수를 기반으로 더 관대하게 결정
            base_colors = min(unique_colors // 100, 64)  # 실제 색상의 1%를 기준
            
            # 색상 풍부도 보정
            richness_multiplier = 1.0 + (color_richness * 2.0)
            
            # 분산 보정 (분산이 높으면 더 많은 색상 필요)
            variance_boost = max(1.0, color_variance / 500)
            
            # 엣지 보정
            edge_boost = 1.0 + (edge_density * 1.5)
            
            # 최종 색상 수 계산 (원본을 더 잘 보존)
            optimal_colors = int(base_colors * richness_multiplier * variance_boost * edge_boost)
            
            # 합리적인 범위로 제한 (하지만 더 관대하게)
            optimal_colors = max(8, min(optimal_colors, 128))
            
            # 특별한 경우들
            if unique_colors > 5000:  # 매우 다채로운 이미지
                optimal_colors = max(optimal_colors, 48)
            elif unique_colors > 2000:  # 복잡한 이미지
                optimal_colors = max(optimal_colors, 32)
            elif unique_colors > 500:   # 일반적인 이미지
                optimal_colors = max(optimal_colors, 16)
            
            print(f"🎨 원본 보존형 AI 분석:")
            print(f"   📊 고유색상: {unique_colors:,}개")
            print(f"   🌈 색상풍부도: {color_richness:.3f}")
            print(f"   📈 색상분산: {color_variance:.1f}")
            print(f"   🔍 엣지밀도: {edge_density:.3f}")
            print(f"   🎯 결정된 색상: {optimal_colors}개")
            
            return optimal_colors
            
        except Exception as e:
            print(f"⚠️ AI 색상 분석 실패, 안전한 기본값 사용: {e}")
            return 24  # 더 관대한 기본값
    
    def _optimize_neural_svg(self, svg_content):
        """Neural SVG 최적화"""
        # 소수점 정밀도 최적화
        import re
        svg_content = re.sub(r'(\d+\.\d{1})\d+', r'\1', svg_content)
        # 불필요한 공백 제거
        svg_content = re.sub(r'\s+', ' ', svg_content)
        return svg_content

class ImageVectorizerModel:
    """AI 강화 고품질 이미지 벡터화 모델 (전문 AI + BiRefNet + Real-ESRGAN)"""
    
    def __init__(self, birefnet_model=None, upscaler_model=None):
        self.loaded = False
        self.birefnet_model = birefnet_model  # 배경 제거용
        self.upscaler_model = upscaler_model  # 업스케일링용
        self.professional_ai = ProfessionalVectorizerAI()  # 전문 벡터화 AI
        self.edge_model = None
        print("📐 AI 강화 이미지 벡터화 모듈 활성화!")
        
    def load_model(self, progress_callback=None):
        """벡터화 모델 로드 (AI 전처리 모델)"""
        if self.loaded:
            return
            
        try:
            if progress_callback:
                progress_callback(10, "🧠 AI 전처리 모델 로드 중...")
                
            # OpenCV의 DNN 모듈로 경량 엣지 검출 모델 사용
            # 또는 간단한 CNN 기반 전처리 적용
            self.loaded = True
            
            if progress_callback:
                progress_callback(50, "📐 벡터화 도구 확인 중...")
            
            # Potrace 설치 확인
            if not self._check_potrace():
                print("⚠️ Potrace가 설치되지 않음. 기본 벡터화 방식 사용")
            
            if progress_callback:
                progress_callback(100, "✅ 벡터화 모델 준비 완료!")
                
            print("🚀 이미지 벡터화 모델 로드 완료!")
            
        except Exception as e:
            print(f"❌ 벡터화 모델 로드 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 모델 로드 실패: {str(e)}")
            raise
    
    def _check_potrace(self):
        """Potrace 설치 확인"""
        try:
            # 여러 가능한 경로에서 potrace 확인 (models 디렉토리 우선)
            potrace_paths = [
                'models/potrace',  # models 디렉토리 우선
                './potrace',  # 프로젝트 루트 fallback
                'potrace',
                '/opt/homebrew/bin/potrace',
                '/usr/local/bin/potrace',
                '/usr/bin/potrace'
            ]
            
            for path in potrace_paths:
                try:
                    result = subprocess.run([path, '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"✅ Potrace 발견: {path}")
                        print(f"🎯 Potrace 버전: {result.stdout.strip().split()[1]}")
                        return True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            print("❌ Potrace를 찾을 수 없습니다")
            return False
        except Exception as e:
            print(f"❌ Potrace 확인 중 오류: {e}")
            return False
    
    def _ai_preprocess(self, image, progress_callback=None):
        """AI 기반 이미지 전처리 (BiRefNet + Real-ESRGAN + 엣지 강화)"""
        try:
            if progress_callback:
                progress_callback(20, "🧠 AI가 이미지를 분석하고 최적화 중...")
            
            processed_image = image
            
            # 1. AI 업스케일링으로 품질 향상 (작은 이미지인 경우)
            width, height = image.size
            if width < 512 or height < 512:
                if self.upscaler_model and progress_callback:
                    progress_callback(30, "🔍 AI 업스케일링으로 해상도 향상 중...")
                    try:
                        processed_image = self.upscaler_model.enhance_image(processed_image)
                        print("✨ AI 업스케일링 적용으로 벡터화 품질 향상")
                    except Exception as e:
                        print(f"⚠️ AI 업스케일링 실패: {e}")
            
            if progress_callback:
                progress_callback(50, "🎨 AI 배경 제거 및 주요 객체 추출 중...")
            
            # 2. 벡터화용 배경 제거 스킵 (원본 색상 보존을 위해)
            skip_background_removal = True  # 벡터화에서는 원본 보존이 더 중요
            
            if not skip_background_removal and self.birefnet_model:
                try:
                    # 배경 제거하여 주요 객체만 추출
                    foreground = self.birefnet_model.remove_background(processed_image)
                    
                    # 투명 배경을 흰색으로 변경 (벡터화를 위해)
                    if foreground.mode == 'RGBA':
                        white_bg = Image.new('RGB', foreground.size, (255, 255, 255))
                        white_bg.paste(foreground, mask=foreground.split()[3])  # 알파 채널을 마스크로 사용
                        processed_image = white_bg
                        print("🎯 AI 배경 제거로 주요 객체 추출 완료")
                    else:
                        processed_image = foreground
                except Exception as e:
                    print(f"⚠️ AI 배경 제거 실패: {e}")
            else:
                print("🌈 벡터화용 원본 색상 보존: 배경 제거 스킵")
            
            if progress_callback:
                progress_callback(70, "⚡ AI 엣지 강화 및 벡터화 최적화 중...")
            
            # 3. 원본 색상 보존을 위한 최소한의 전처리
            img_array = np.array(processed_image)
            
            # 벡터화를 위한 경량 처리 (색상 보존 우선)
            preserve_original_colors = True
            
            if preserve_original_colors:
                # 최소한의 노이즈 제거만 (색상 보존)
                enhanced = cv2.medianBlur(img_array, 3)  # 가장 부드러운 노이즈 제거
                print("🌈 원본 색상 보존 모드: 최소한의 전처리만 적용")
            else:
                # 기존 강화 처리 (색상 변경 가능)
                blurred = cv2.GaussianBlur(img_array, (3, 3), 0)
                unsharp_mask = cv2.addWeighted(img_array, 1.8, blurred, -0.8, 0)
                
                if len(unsharp_mask.shape) == 3:
                    lab = cv2.cvtColor(unsharp_mask, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(unsharp_mask)
                
                if len(enhanced.shape) == 3:
                    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
                else:
                    gray = enhanced
                    
                edges = cv2.Canny(gray, 30, 100)
                kernel = np.ones((2,2), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                if len(enhanced.shape) == 3:
                    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    enhanced = cv2.addWeighted(enhanced, 0.7, edges_colored, 0.3, 0)
            
            result = Image.fromarray(enhanced)
            print("🚀 원본 색상 보존 전처리 완료!")
            return result
            
        except Exception as e:
            print(f"⚠️ AI 전처리 실패, 원본 사용: {e}")
            return image
    
    def _quantize_colors(self, image, n_colors=8):
        """색상 양자화 (벡터화 최적화를 위해)"""
        try:
            img_array = np.array(image)
            data = img_array.reshape((-1, 3))
            data = np.float32(data)
            
            # K-means 클러스터링으로 색상 수 줄이기
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 클러스터 중심으로 색상 교체
            centers = np.uint8(centers)
            quantized_data = centers[labels.flatten()]
            quantized_image = quantized_data.reshape(img_array.shape)
            
            return Image.fromarray(quantized_image)
            
        except Exception as e:
            print(f"⚠️ 색상 양자화 실패, 원본 사용: {e}")
            return image
    
    def _bitmap_to_svg_potrace(self, bitmap_path, svg_path):
        """Potrace를 사용한 비트맵 → SVG 변환"""
        try:
            # models 디렉토리의 potrace 우선 사용
            potrace_cmd = 'models/potrace'
            if not os.path.exists(potrace_cmd):
                potrace_cmd = './potrace'  # 프로젝트 루트 fallback
                if not os.path.exists(potrace_cmd):
                    potrace_cmd = 'potrace'  # 시스템 경로 fallback
            
            cmd = [
                potrace_cmd,
                '--svg',
                '--output', svg_path,
                '--turdsize', '2',  # 작은 스펙 제거
                '--alphamax', '1.0',  # 곡선 스무딩
                '--opttolerance', '0.2',  # 최적화 허용 오차
                bitmap_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return True
            else:
                print(f"⚠️ Potrace 실행 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"⚠️ Potrace 변환 실패: {e}")
            return False
    
    def _fallback_vectorize(self, image, svg_path):
        """폴백 벡터화 방식 (Potrace 없을 때)"""
        try:
            # 간단한 SVG 생성 (색상별 경로 생성)
            width, height = image.size
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
<rect width="{width}" height="{height}" fill="white"/>
'''
            
            # 이미지를 그레이스케일로 변환하여 경로 생성
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # 적응형 임계값 또는 Otsu 방법 사용
            # 먼저 Otsu 방법 시도
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"🔍 Otsu 임계값 사용으로 이진화")
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"🔍 디버깅: {len(contours)}개의 컨투어 발견됨")
            
            # 각 컨투어를 SVG 경로로 변환
            valid_contours = 0
            for i, contour in enumerate(contours):
                # 컨투어 면적으로 필터링 (너무 작은 것 제외)
                area = cv2.contourArea(contour)
                if area > 100 and len(contour) > 3:  # 조건 완화
                    valid_contours += 1
                    print(f"🔍 컨투어 {i}: 면적={area:.0f}, 점개수={len(contour)}")
                    # 컨투어를 폴리곤으로 근사화
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # SVG 경로 생성
                    if len(approx) > 2:
                        path_data = f"M {approx[0][0][0]} {approx[0][0][1]}"
                        for point in approx[1:]:
                            path_data += f" L {point[0][0]} {point[0][1]}"
                        path_data += " Z"
                        
                        svg_content += f'<path d="{path_data}" fill="black" stroke="none"/>\n'
            
            svg_content += '</svg>'
            
            print(f"🔍 디버깅: {valid_contours}개의 유효한 컨투어로 SVG 생성")
            print(f"🔍 디버깅: SVG 콘텐츠 길이: {len(svg_content)} 문자")
            
            # 유효한 컨투어가 없다면 단순한 픽셀 기반 접근 시도
            if valid_contours == 0:
                print("🔄 컨투어가 없어서 픽셀 기반 벡터화 시도...")
                svg_content = self._create_pixel_based_svg(image, width, height)
            
            # SVG 파일 저장
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            return True
            
        except Exception as e:
            print(f"⚠️ 폴백 벡터화 실패: {e}")
            return False
    
    def _create_ai_pixel_art_svg(self, image, width, height):
        """AI 강화 픽셀 아트 SVG 생성"""
        try:
            print("🎨 AI 픽셀 아트 모드 시작")
            
            # 1. 이미지 크기 최적화 (너무 크면 성능 문제)
            target_pixels = min(80, max(width//8, 20)), min(80, max(height//8, 20))
            small_image = image.resize(target_pixels, Image.LANCZOS)  # 고품질 리사이징
            
            # 2. 색상 클러스터링으로 팔레트 생성
            img_array = np.array(small_image)
            pixels = img_array.reshape(-1, 3).astype(np.float32)
            
            # K-means로 주요 색상 추출
            n_colors = 8
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 색상 팔레트 생성
            palette = centers.astype(np.uint8)
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
<rect width="{width}" height="{height}" fill="white"/>
'''
            
            # 픽셀별 처리
            pixel_width = width / target_pixels[0]
            pixel_height = height / target_pixels[1]
            
            rendered_pixels = 0
            for y in range(target_pixels[1]):
                for x in range(target_pixels[0]):
                    pixel = small_image.getpixel((x, y))
                    
                    # 밝기 기반 필터링 (더 지능적)
                    brightness = sum(pixel) / 3
                    if brightness < 240:  # 흰색이 아닌 픽셀만
                        # 가장 가까운 팔레트 색상 찾기
                        distances = [sum((pixel[i] - color[i])**2 for i in range(3)) for color in palette]
                        closest_color = palette[np.argmin(distances)]
                        
                        hex_color = f"#{closest_color[0]:02x}{closest_color[1]:02x}{closest_color[2]:02x}"
                        px = x * pixel_width
                        py = y * pixel_height
                        
                        # 크기별 투명도 조절
                        opacity = max(0.7, 1.0 - brightness/300)
                        svg_content += f'<rect x="{px:.1f}" y="{py:.1f}" width="{pixel_width:.1f}" height="{pixel_height:.1f}" fill="{hex_color}" opacity="{opacity:.2f}"/>\n'
                        rendered_pixels += 1
            
            svg_content += '</svg>'
            print(f"🎨 AI 픽셀 아트 완성: {rendered_pixels}개 픽셀, {len(svg_content)} 문자")
            return svg_content
            
        except Exception as e:
            print(f"❌ AI 픽셀 아트 실패: {e}")
            return self._create_simple_fallback_svg(image, width, height)
    
    def _create_simple_fallback_svg(self, image, width, height):
        """단순 폴백 SVG"""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
<rect width="{width}" height="{height}" fill="#f8f9fa"/>
<circle cx="{width//2}" cy="{height//2}" r="{min(width, height)//4}" fill="#e9ecef" opacity="0.5"/>
<text x="{width//2}" y="{height//2}" text-anchor="middle" font-family="Arial" font-size="14" fill="#6c757d">AI 벡터화</text>
</svg>'''
    
    def _optimize_svg(self, svg_content):
        """SVG 최적화 (파일 크기 줄이기)"""
        try:
            import re
            # 소수점 자릿수 줄이기
            svg_content = re.sub(r'(\d+\.\d{2})\d+', r'\1', svg_content)
            # 불필요한 공백 제거
            svg_content = re.sub(r'\s+', ' ', svg_content)
            # 불필요한 속성 제거
            svg_content = re.sub(r' opacity="1\.00"', '', svg_content)
            return svg_content
        except:
            return svg_content
    
    def _create_pixel_based_svg(self, image, width, height):
        """기존 픽셀 기반 SVG (호환성 유지)"""
        return self._create_ai_pixel_art_svg(image, width, height)
    
    def vectorize_image(self, image, output_format='svg', n_colors=8, progress_callback=None):
        """이미지를 벡터 형식으로 변환"""
        if not self.loaded:
            self.load_model(progress_callback)
        
        try:
            if progress_callback:
                progress_callback(10, "🎨 이미지 벡터화 시작...")
            
            # 🌈 벡터화용 완전 원본 보존: 모든 전처리 스킵
            skip_all_preprocessing = True
            
            if skip_all_preprocessing:
                processed_image = image
                print("🚫 벡터화용 전처리 완전 스킵: 원본 그대로 사용")
            else:
                # 기존 AI 전처리 (사용 안함)
                if progress_callback:
                    progress_callback(20, "🧠 AI가 이미지를 최적화하고 있습니다...")
                processed_image = self._ai_preprocess(image, progress_callback)
            
            # 🌈 색상 양자화도 스킵: 원본 색상 그대로 사용
            if skip_all_preprocessing:
                quantized_image = processed_image
                print("🚫 색상 양자화 스킵: 원본 색상 그대로 유지")
            else:
                # 기존 색상 양자화 (사용 안함)
                if progress_callback:
                    progress_callback(40, "🎨 색상 최적화 중...")
                quantized_image = self._quantize_colors(processed_image, n_colors)
            
            # 3. 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp_bmp:
                temp_bmp_path = temp_bmp.name
                quantized_image.save(temp_bmp_path, 'BMP')
            
            # 4. 전문 AI 벡터화 시도
            if progress_callback:
                progress_callback(60, "🧠 전문 AI 벡터화 처리 중...")
            
            temp_svg_path = temp_bmp_path.replace('.bmp', '.svg')
            
            # 🎯 1순위: 전문 Potrace 벡터화 (최고 품질)
            vectorization_success = False
            
            if self._check_potrace():
                print("🚀 전문 Potrace 벡터화 모드 시작!")
                if progress_callback:
                    progress_callback(70, "🔄 Potrace로 고품질 벡터화 중...")
                try:
                    vectorization_success = self._bitmap_to_svg_potrace(temp_bmp_path, temp_svg_path)
                    if vectorization_success:
                        print("🎯 전문 Potrace 벡터화 대성공!")
                        if progress_callback:
                            progress_callback(90, "✅ Potrace 고품질 벡터화 완료!")
                    else:
                        print("⚠️ Potrace 벡터화 실패, 내장 엔진 시도...")
                except Exception as e:
                    print(f"❌ Potrace 벡터화 예외 발생: {e}")
                    vectorization_success = False
            else:
                print("⚠️ Potrace를 찾을 수 없음, 내장 엔진 시도...")
            
            # 3순위: 내장 벡터화 엔진 (모든 방법 실패 시)
            if not vectorization_success:
                if progress_callback:
                    progress_callback(70, "🔄 내장 벡터화 엔진 사용 중...")
                vectorization_success = self._fallback_vectorize(quantized_image, temp_svg_path)
            
            if not vectorization_success:
                raise Exception("모든 벡터화 방법 실패")
            
            # 5. SVG 최적화
            if progress_callback:
                progress_callback(85, "⚡ SVG 최적화 중...")
            self._optimize_svg(temp_svg_path)
            
            # 6. 결과 파일 읽기
            if progress_callback:
                progress_callback(95, "📄 결과 파일 준비 중...")
            
            with open(temp_svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # 임시 파일 정리
            try:
                os.unlink(temp_bmp_path)
                os.unlink(temp_svg_path)
            except:
                pass
            
            if progress_callback:
                progress_callback(100, "🎉 이미지 벡터화 완료!")
            
            print(f"✅ 이미지 벡터화 완료 - {len(svg_content):,}자 SVG 생성")
            
            return svg_content
            
        except Exception as e:
            print(f"❌ 벡터화 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 벡터화 실패: {str(e)}")
            raise

class VideoProcessor:
    """비디오 프레임별 처리 클래스"""
    
    def __init__(self):
        self.temp_dirs = {}
    
    def create_temp_dir(self, session_id):
        """세션별 임시 디렉토리 생성"""
        temp_dir = os.path.join(TEMP_FOLDER, session_id)
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dirs[session_id] = temp_dir
        return temp_dir
    
    def cleanup_temp_dir(self, session_id):
        """임시 디렉토리 정리"""
        if session_id in self.temp_dirs:
            import shutil
            try:
                shutil.rmtree(self.temp_dirs[session_id])
                del self.temp_dirs[session_id]
                print(f"🧹 임시 디렉토리 정리 완료: {session_id}")
            except Exception as e:
                print(f"⚠️ 임시 디렉토리 정리 실패: {e}")
    
    def extract_frames(self, video_path, temp_dir, progress_callback=None):
        """비디오에서 프레임 추출 (강화된 디버깅 및 오류 처리)"""
        try:
            if progress_callback:
                progress_callback(10, "🎬 비디오 정보 분석 중...")
            
            # 비디오 파일 존재 및 크기 확인
            if not os.path.exists(video_path):
                raise ValueError(f"비디오 파일이 존재하지 않습니다: {video_path}")
            
            file_size = os.path.getsize(video_path)
            print(f"📂 비디오 파일 확인: {video_path} (크기: {file_size:,} bytes)")
            
            if file_size == 0:
                raise ValueError("비디오 파일이 비어있습니다")
            
            # OpenCV 버전 및 코덱 지원 확인
            print(f"🔧 OpenCV 버전: {cv2.__version__}")
            
            # 비디오 캡처 객체 생성
            print("📹 VideoCapture 객체 생성 중...")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                # 더 상세한 오류 정보
                print("❌ VideoCapture 초기화 실패")
                print(f"   - 파일 경로: {video_path}")
                print(f"   - 파일 존재: {os.path.exists(video_path)}")
                print(f"   - 파일 크기: {file_size:,} bytes")
                print(f"   - 파일 확장자: {os.path.splitext(video_path)[1]}")
                
                # 다른 방법으로 시도
                print("🔄 절대 경로로 재시도...")
                abs_path = os.path.abspath(video_path)
                cap = cv2.VideoCapture(abs_path)
                
                if not cap.isOpened():
                    raise ValueError(f"비디오 파일을 열 수 없습니다. 지원되지 않는 코덱이거나 손상된 파일일 수 있습니다.\n파일: {video_path}")
            
            print("✅ VideoCapture 초기화 성공")
            
            # 비디오 정보 가져오기 (OpenCV 메타데이터)
            fps = cap.get(cv2.CAP_PROP_FPS)
            estimated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # FOURCC 코덱 정보 디코딩
            codec_chars = [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
            codec_name = ''.join(codec_chars)
            
            print(f"📹 OpenCV 메타데이터:")
            print(f"   - 예상 프레임 수: {estimated_frames}")
            print(f"   - FPS: {fps:.2f}")
            print(f"   - 해상도: {width}x{height}")
            print(f"   - 코덱: {codec_name} (FOURCC: {fourcc})")
            
            # 메타데이터 유효성 검증
            if fps <= 0:
                print("⚠️ 잘못된 FPS 정보, 기본값 25fps 사용")
                fps = 25.0
            
            if width <= 0 or height <= 0:
                print("⚠️ 잘못된 해상도 정보 감지")
                raise ValueError(f"잘못된 비디오 해상도: {width}x{height}")
            
            if progress_callback:
                progress_callback(15, f"🎞️ 프레임 추출 중... (예상 {estimated_frames}개)")
            
            # 프레임 저장 디렉토리 생성
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"📁 프레임 저장 디렉토리: {frames_dir}")
            
            frame_files = []
            frame_count = 0
            consecutive_failures = 0
            max_failures = 10  # 연속 실패 허용 횟수
            
            print("🎬 프레임 추출 시작...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        print(f"⚠️ 연속 {max_failures}회 프레임 읽기 실패, 추출 종료")
                        break
                    continue
                
                # 성공 시 실패 카운터 리셋
                consecutive_failures = 0
                
                # 프레임 유효성 검증
                if frame is None or frame.size == 0:
                    print(f"⚠️ 빈 프레임 감지 (프레임 {frame_count})")
                    continue
                
                try:
                    # BGR to RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # 프레임 저장
                    frame_filename = f"frame_{frame_count:06d}.png"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    frame_pil.save(frame_path, 'PNG')
                    frame_files.append(frame_path)
                    
                    frame_count += 1
                    
                    # 첫 번째 프레임 저장 확인
                    if frame_count == 1:
                        if os.path.exists(frame_path):
                            print(f"✅ 첫 번째 프레임 저장 확인: {frame_path}")
                        else:
                            raise ValueError("첫 번째 프레임 저장 실패")
                    
                    # 진행률 업데이트 (추출은 전체의 20%까지)
                    if progress_callback and frame_count % 5 == 0:
                        # 예상 프레임 수가 있으면 그것 기준으로, 없으면 현재까지 추출된 수로 표시
                        if estimated_frames > 0:
                            progress = 15 + (frame_count / estimated_frames) * 5
                            progress_callback(int(progress), f"🎞️ 프레임 추출 중... ({frame_count}/{estimated_frames})")
                        else:
                            progress_callback(15, f"🎞️ 프레임 추출 중... ({frame_count}개)")
                    
                    # 주기적으로 상태 출력
                    if frame_count % 25 == 0:
                        print(f"   📊 진행 상황: {frame_count}개 프레임 추출됨")
                
                except Exception as frame_error:
                    print(f"⚠️ 프레임 {frame_count} 처리 중 오류: {frame_error}")
                    continue
            
            # 실제 추출된 프레임 수로 정확한 FPS 계산 (cap.release() 전에 수행)
            # 예상 duration 계산
            estimated_duration = estimated_frames / fps if fps > 0 else 0
            
            # 실제 추출된 프레임 수 기준으로 정확한 FPS 계산
            if estimated_duration > 0:
                actual_fps = frame_count / estimated_duration
            else:
                actual_fps = fps if fps > 0 else 25.0  # 기본값
            
            cap.release()
            
            print(f"✅ 프레임 추출 완료:")
            print(f"   - 실제 추출: {frame_count}개 프레임")
            print(f"   - 예상 프레임: {estimated_frames}개")
            print(f"   - 정확한 FPS: {actual_fps:.2f}")
            print(f"   - OpenCV FPS: {fps:.2f}")
            print(f"   - 저장 위치: {frames_dir}")
            
            # 프레임이 하나도 추출되지 않은 경우
            if frame_count == 0:
                raise ValueError("프레임을 하나도 추출할 수 없습니다. 비디오 파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
            
            return {
                'frame_files': frame_files,
                'fps': actual_fps,
                'total_frames': frame_count,  # 실제 추출된 프레임 수 사용
                'original_frames': frame_count,  # 실제 추출된 프레임 수
                'width': width,
                'height': height,
                'estimated_frames': estimated_frames,  # 비교를 위해 예상 프레임 수도 포함
                'codec': codec_name
            }
            
        except Exception as e:
            print(f"❌ 프레임 추출 실패:")
            print(f"   - 오류 메시지: {e}")
            print(f"   - 비디오 경로: {video_path}")
            if 'file_size' in locals():
                print(f"   - 파일 크기: {file_size:,} bytes")
            raise
    
    def process_frames(self, frame_files, remove_bg, upscale, scale_factor, background_color=None, progress_callback=None):
        """각 프레임에 AI 처리 적용 (배경 색상 선택 지원)"""
        try:
            total_frames = len(frame_files)
            processed_files = []
            
            # 배경 색상 설정 (기본값: 흰색)
            if background_color and background_color.startswith('#') and len(background_color) == 7:
                try:
                    # 16진수 색상 코드를 RGB로 변환
                    bg_color = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
                    print(f"🎨 선택된 배경 색상: {background_color} (RGB: {bg_color})")
                except ValueError:
                    bg_color = (255, 255, 255)  # 잘못된 색상 코드 시 흰색 사용
                    print("⚠️ 잘못된 색상 코드, 흰색 배경 사용")
            else:
                bg_color = (255, 255, 255)  # 기본 흰색
            
            if progress_callback:
                progress_callback(20, f"🤖 AI 처리 시작... (총 {total_frames}개 프레임)")
            
            for i, frame_path in enumerate(frame_files):
                # 프레임 로드
                frame_image = Image.open(frame_path).convert('RGB')
                processed_image = frame_image
                
                # 배경 제거 적용
                if remove_bg:
                    processed_image = ai_model.remove_background(processed_image)
                    # RGBA를 RGB로 변환 (비디오는 투명도 지원 안함)
                    if processed_image.mode == 'RGBA':
                        # 선택된 색상으로 배경 합성
                        color_bg = Image.new('RGB', processed_image.size, bg_color)
                        color_bg.paste(processed_image, mask=processed_image.split()[-1])
                        processed_image = color_bg
                
                # 업스케일링 적용
                if upscale:
                    processed_image = upscale_model.upscale_image(processed_image, scale_factor)
                
                # 처리된 프레임 저장
                processed_path = frame_path.replace('frames', 'processed').replace('.png', '_processed.png')
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                processed_image.save(processed_path)
                processed_files.append(processed_path)
                
                # 진행률 업데이트 (프레임 처리는 20%~80%)
                if progress_callback:
                    progress = 20 + (i + 1) / total_frames * 60
                    progress_callback(int(progress), f"🤖 AI가 프레임을 처리하고 있습니다... ({i+1}/{total_frames})")
            
            return processed_files
            
        except Exception as e:
            print(f"❌ 프레임 처리 실패: {e}")
            raise
    
    def extract_last_frame(self, video_path, progress_callback=None):
        """비디오에서 마지막 프레임 추출"""
        try:
            if progress_callback:
                progress_callback(10, "🎬 비디오 마지막 프레임 추출 중...")
            
            # 비디오 파일 존재 확인
            if not os.path.exists(video_path):
                raise ValueError(f"비디오 파일이 존재하지 않습니다: {video_path}")
            
            file_size = os.path.getsize(video_path)
            print(f"📂 비디오 파일 확인: {video_path} (크기: {file_size:,} bytes)")
            
            if file_size == 0:
                raise ValueError("비디오 파일이 비어있습니다")
            
            if progress_callback:
                progress_callback(30, "📹 비디오 정보 분석 중...")
            
            # 비디오 캡처 객체 생성
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
            if progress_callback:
                progress_callback(50, "🎞️ 마지막 프레임 검색 중...")
            
            # 총 프레임 수 가져오기
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 비디오 정보 - 총 프레임: {total_frames}, FPS: {fps:.2f}, 해상도: {width}x{height}")
            
            if total_frames <= 0:
                raise ValueError("비디오의 프레임 수를 확인할 수 없습니다")
            
            # 마지막 프레임으로 이동 (마지막에서 1개 전 프레임을 안전하게 선택)
            last_frame_index = max(0, total_frames - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
            
            if progress_callback:
                progress_callback(70, "🖼️ 마지막 프레임 추출 중...")
            
            # 마지막 프레임 읽기
            ret, frame = cap.read()
            
            if not ret or frame is None:
                # 마지막 프레임 읽기 실패 시 역순으로 프레임 찾기
                print("⚠️ 마지막 프레임 읽기 실패, 역순으로 유효한 프레임 찾는 중...")
                for i in range(min(10, total_frames)):  # 최대 10개 프레임 역순으로 확인
                    frame_index = max(0, total_frames - 3 - i)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ 유효한 프레임 발견: 인덱스 {frame_index}")
                        break
                
                if not ret or frame is None:
                    raise ValueError("비디오에서 유효한 마지막 프레임을 찾을 수 없습니다")
            
            cap.release()
            
            if progress_callback:
                progress_callback(85, "🎨 이미지 변환 중...")
            
            # BGR to RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            if progress_callback:
                progress_callback(100, "✅ 마지막 프레임 추출 완료!")
            
            print(f"🎉 마지막 프레임 추출 완료 - 해상도: {width}x{height}")
            
            return {
                'frame_image': frame_pil,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'fps': fps,
                'frame_index': last_frame_index
            }
            
        except Exception as e:
            print(f"❌ 마지막 프레임 추출 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 추출 실패: {str(e)}")
            raise

    def reassemble_video(self, processed_files, output_path, fps, width, height, progress_callback=None):
        """처리된 프레임들을 비디오로 재조립 (H.264 코덱 사용)"""
        try:
            if progress_callback:
                progress_callback(80, "🎬 H.264 코덱으로 비디오 재조립 중...")
            
            # H.264 코덱 우선 시도, 실패 시 mp4v 폴백
            codecs_to_try = [
                ('h264', 'H.264'),
                ('H264', 'H.264'),
                ('avc1', 'H.264'),
                ('mp4v', 'MP4V')
            ]
            
            out = None
            used_codec = None
            
            for fourcc_code, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        used_codec = codec_name
                        print(f"✅ {codec_name} 코덱으로 비디오 인코딩 시작")
                        break
                    else:
                        out.release()
                        print(f"⚠️ {codec_name} 코덱 사용 실패, 다음 코덱 시도...")
                except Exception as e:
                    print(f"⚠️ {codec_name} 코덱 초기화 실패: {e}")
                    if out:
                        out.release()
                    continue
            
            if not out or not out.isOpened():
                raise ValueError("지원되는 비디오 코덱을 찾을 수 없습니다")
            
            total_frames = len(processed_files)
            
            for i, frame_path in enumerate(processed_files):
                # 프레임 로드 및 크기 조정
                frame_pil = Image.open(frame_path).convert('RGB')
                frame_pil = frame_pil.resize((width, height), Image.Resampling.LANCZOS)
                
                # PIL to OpenCV 변환
                frame_array = np.array(frame_pil)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                # 프레임 쓰기
                out.write(frame_bgr)
                
                # 진행률 업데이트 (재조립은 80%~95%)
                if progress_callback and i % 5 == 0:
                    progress = 80 + (i + 1) / total_frames * 15
                    progress_callback(int(progress), f"🎬 {used_codec} 코덱으로 인코딩 중... ({i+1}/{total_frames})")
            
            out.release()
            
            if progress_callback:
                progress_callback(95, f"✅ {used_codec} 코덱으로 비디오 재조립 완료!")
            
            print(f"🎉 비디오 재조립 완료 - {used_codec} 코덱 사용, 해상도: {width}x{height}, FPS: {fps:.2f}")
            
        except Exception as e:
            print(f"❌ 비디오 재조립 실패: {e}")
            raise

# 전역 모델 인스턴스
ai_model = BiRefNetModel()
upscale_model = RealESRGANUpscaleModel()
vectorizer_model = ImageVectorizerModel(ai_model, upscale_model)
video_processor = VideoProcessor()

# 프로그래스 상태 관리
progress_queues = {}

# UUID 기반 세션 스토리지 (파일 기반 지속성)
SESSION_FILE = os.path.join(DOWNLOAD_FOLDER, 'sessions.json')

def load_session_storage():
    """세션 데이터를 파일에서 로드"""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"⚠️ 세션 데이터 로드 실패: {e}")
        return {}

def save_session_storage(storage):
    """세션 데이터를 파일에 저장"""
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(storage, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 세션 데이터 저장 실패: {e}")

def update_session_data(work_id, data):
    """세션 데이터 업데이트 및 저장"""
    session_storage[work_id] = data
    save_session_storage(session_storage)
    print(f"💾 세션 데이터 저장됨: {work_id}")

# 초기 세션 데이터 로드
session_storage = load_session_storage()
print(f"📂 기존 세션 {len(session_storage)}개 복원됨")

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    """허용된 비디오 파일 확장자인지 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS



def light_improve_mask_quality(image):
    """경량 배경 제거 품질 개선 후처리"""
    try:
        img_array = np.array(image)
        
        if img_array.shape[2] == 4:
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
            
            # 가벼운 가우시안 블러로 부드러운 가장자리
            alpha_smoothed = cv2.GaussianBlur(alpha, (3, 3), 0.5)
            
            # 결합
            img_array[:, :, 3] = alpha_smoothed
            
            return Image.fromarray(img_array, 'RGBA')
        
        return image
        
    except Exception as e:
        print(f"⚠️ 후처리 경고: {e}")
        return image

def smart_guide_processing(image, bounds, original_width, original_height, progress_callback=None):
    """스마트 가이드: 선택된 영역만 보존하고 나머지는 완전 제거"""
    try:
        if progress_callback:
            progress_callback(85, "📐 선택 영역 분석 중...")
        
        # JavaScript에서 이미 원본 이미지 좌표로 변환해서 보냄 - 추가 변환 불필요!
        x = int(bounds['x'])
        y = int(bounds['y'])
        width = int(bounds['width'])
        height = int(bounds['height'])
        
        # 경계 확인만 수행
        x = max(0, min(x, original_width - 1))
        y = max(0, min(y, original_height - 1))
        width = max(1, min(width, original_width - x))
        height = max(1, min(height, original_height - y))
        
        print(f"📍 선택 영역: x:{x}, y:{y}, w:{width}, h:{height}")
        
        if progress_callback:
            progress_callback(88, "✂️ 선택 영역 크롭 중...")
        
        if progress_callback:
            progress_callback(92, "✂️ 선택 영역만 크롭 중...")
        
        # 선택 영역만 크롭
        cropped_region = image.crop((x, y, x + width, y + height))
        
        if progress_callback:
            progress_callback(94, "🎯 선택 영역 내에서만 AI 배경 제거 중...")
        
        # 크롭된 영역에만 AI 적용 (선택 영역 내 맥락으로만 판단)
        print("🔥 선택 영역 내에서만 고품질 AI 배경 제거")
        processed_region = ai_model.remove_background(cropped_region)
        
        if progress_callback:
            progress_callback(96, "🔧 품질 개선 중...")
        

        
        if progress_callback:
            progress_callback(97, "🔧 품질 개선 중...")
        
        # 후처리
        processed_region = light_improve_mask_quality(processed_region)
        
        if progress_callback:
            progress_callback(98, "🎨 투명 캔버스에 합성 중...")
        
        # 완전 투명한 캔버스 생성
        result = Image.new('RGBA', (original_width, original_height), (0, 0, 0, 0))
        
        # 처리된 영역만 원래 위치에 배치 (나머지는 투명)
        result.paste(processed_region, (x, y), processed_region)
        
        if progress_callback:
            progress_callback(100, "✅ 선택 영역 처리 완료!")
        
        print(f"✅ 고품질 AI 스마트 가이드 완료 - 선택된 영역({width}x{height}) 내 객체만 보존, 외부는 투명")
        return result
        
    except Exception as e:
        print(f"❌ 스마트 가이드 실패: {e}")
        if progress_callback:
            progress_callback(0, f"❌ 처리 실패: {str(e)}")
        raise

def send_progress(session_id, progress, message):
    """프로그래스 상태 전송"""
    if session_id in progress_queues:
        progress_queues[session_id].put({'progress': progress, 'message': message})

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/work/<work_id>')
def work_page(work_id):
    """UUID 기반 작업 페이지"""
    # UUID가 세션 스토리지에 있는지 확인
    if work_id in session_storage:
        # 작업 상태가 있으면 해당 상태로 페이지 렌더링
        work_data = session_storage[work_id]
        return render_template('index.html', work_id=work_id, work_data=work_data)
    else:
        # 없으면 일반 페이지로 렌더링 (새 작업 시작 가능)
        return render_template('index.html', work_id=work_id)

@app.route('/progress/<session_id>')
def progress(session_id):
    """Server-Sent Events로 프로그래스 전송"""
    def generate():
        if session_id not in progress_queues:
            progress_queues[session_id] = Queue()
        
        queue = progress_queues[session_id]
        
        try:
            while True:
                try:
                    # 큐에서 프로그래스 데이터 가져오기 (타임아웃 30초)
                    data = queue.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # 완료되면 연결 종료
                    if data['progress'] >= 100:
                        break
                        
                except:
                    # 타임아웃 또는 오류 시 연결 종료
                    break
        finally:
            # 정리
            if session_id in progress_queues:
                del progress_queues[session_id]
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/save_upload_state', methods=['POST'])
def save_upload_state():
    """이미지/비디오 업로드 직후 상태를 즉시 세션에 저장 (메타데이터만)"""
    try:
        work_id = request.form.get('work_id')
        original_filename = request.form.get('original_filename')
        video_type = request.form.get('video_type')  # 비디오인 경우 'uploaded'
        
        if not work_id or not original_filename:
            return jsonify({'error': '필수 정보가 누락되었습니다.'}), 400
        
        # 업로드된 상태로 세션 저장 (이미지/비디오 구분)
        session_data = {
            'type': 'uploaded',
            'original_filename': original_filename,
            'completed': False,
            'timestamp': time.time(),
            'status': 'uploaded'  # 처리 대기 상태
        }
        
        # 비디오인 경우 추가 정보 저장
        if video_type == 'uploaded':
            session_data['video_type'] = 'uploaded'
        
        update_session_data(work_id, session_data)
        
        print(f"📝 업로드 메타데이터 저장 완료: {work_id} - {original_filename} ({'비디오' if video_type else '이미지'})")
        
        return jsonify({
            'success': True,
            'work_id': work_id,
            'message': '업로드 상태가 저장되었습니다.'
        })
        
    except Exception as e:
        print(f"❌ 업로드 상태 저장 실패: {e}")
        return jsonify({'error': f'상태 저장 실패: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 및 배경 제거 처리"""
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 파일 검증 중...")
        
        # 파일 확인
        if 'file' not in request.files:
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        progress_callback(10, "📁 이미지 로드 중...")
        
        # 이미지 로드
        image = Image.open(file.stream).convert('RGB')
        original_width, original_height = image.size
        
        # 처리 모드 확인
        mode = request.form.get('mode', 'auto')
        
        print(f"🔍 처리 모드: {mode}")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        
        if mode == 'auto':
            print("🔥 고품질 AI 자동 배경 제거 모드")
            progress_callback(15, "🚀 고품질 자동 배경 제거 시작...")
            
            # 고품질 AI 모델로 전체 이미지 배경 제거
            result_image = ai_model.remove_background(image, progress_callback)
            
            # 경량 후처리
            print("🔧 경량 후처리: 품질 안정화 중...")
            result_image = light_improve_mask_quality(result_image)
            

        
        else:
            progress_callback(0, "❌ 잘못된 처리 모드입니다.")
            return jsonify({'error': '잘못된 처리 모드입니다.'}), 400
        
        progress_callback(99, "💾 결과 저장 중...")
        
        # 투명 PNG로 결과 저장
        filename = secure_filename(f"removed_{uuid.uuid4().hex}.png")
        filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        
        # RGBA 모드로 투명도 보장
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        # 완전 투명 PNG로 저장 (optimize=True로 파일 크기 최적화)
        result_image.save(filepath, 'PNG', optimize=True, compress_level=6)
        
        progress_callback(100, "🎉 고품질 AI 배경 제거 완료!")
        
        # 세션 데이터 저장
        update_session_data(session_id, {
            'type': 'image',
            'filename': filename,
            'original_filename': file.filename,
            'download_url': url_for('download_file', filename=filename),
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=filename),
            'session_id': session_id,
            'work_id': session_id  # 프론트엔드에서 URL 변경에 사용
        })
        
    except Exception as e:
        print(f"❌ 업로드 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/upscale', methods=['POST'])
def upscale_image():
    """이미지 업스케일링 처리"""
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 파일 검증 중...")
        
        # 파일 확인
        if 'file' not in request.files:
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        # 업스케일 팩터 확인
        scale = request.form.get('scale', '4')
        try:
            scale = int(scale)
            if scale not in [2, 4]:
                return jsonify({'error': '지원하지 않는 업스케일 배율입니다. (2x, 4x만 지원)'}), 400
        except ValueError:
            return jsonify({'error': '잘못된 업스케일 배율입니다.'}), 400
        
        progress_callback(10, f"📁 {scale}x 업스케일링 시작...")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        
        # 파일 저장
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        
        input_filename = f"{name}_{timestamp}{ext}"
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_filepath)
        
        progress_callback(15, "🖼️ 이미지 로드 중...")
        
        # 이미지 로드
        try:
            image = Image.open(input_filepath)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        except Exception as e:
            progress_callback(0, "❌ 이미지를 로드할 수 없습니다.")
            return jsonify({'error': f'이미지 파일이 손상되었습니다: {str(e)}'}), 400
        
        progress_callback(20, f"🔧 {scale}x 업스케일 모델 로드 중...")
        
        # 업스케일링 처리
        try:
            result_image = upscale_model.upscale_image(image, scale, progress_callback)
        except Exception as e:
            progress_callback(0, f"❌ 업스케일링 실패: {str(e)}")
            return jsonify({'error': f'업스케일링 처리 실패: {str(e)}'}), 500
        
        progress_callback(95, "💾 결과 이미지 저장 중...")
        
        # 결과 파일 저장
        unique_id = str(uuid.uuid4())[:8]
        result_filename = f"upscaled_{scale}x_{unique_id}.png"
        result_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], result_filename)
        
        # PNG로 저장 (투명도 유지)
        if result_image.mode == 'RGBA':
            result_image.save(result_filepath, 'PNG', optimize=True, compress_level=6)
        else:
            result_image.save(result_filepath, 'PNG', optimize=True, compress_level=6)
        
        progress_callback(100, f"🎉 {scale}x 업스케일링 완료!")
        
        # 임시 업로드 파일 삭제
        try:
            os.remove(input_filepath)
        except:
            pass
        
        # 세션 데이터 저장
        update_session_data(session_id, {
            'type': 'upscale',
            'filename': result_filename,
            'original_filename': file.filename,
            'download_url': url_for('download_file', filename=result_filename),
            'scale': scale,
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=result_filename),
            'session_id': session_id,
            'work_id': session_id,  # 프론트엔드에서 URL 변경에 사용
            'scale': scale
        })
        
    except Exception as e:
        print(f"❌ 업스케일링 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """비디오 프레임별 처리"""
    session_id = None
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 비디오 파일 검증 중...")
        
        # 파일 확인
        if 'file' not in request.files:
            progress_callback(0, "❌ 비디오 파일이 선택되지 않았습니다.")
            return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            progress_callback(0, "❌ 비디오 파일이 선택되지 않았습니다.")
            return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_video_file(file.filename):
            progress_callback(0, "❌ 지원하지 않는 비디오 형식입니다.")
            return jsonify({'error': '지원하지 않는 비디오 형식입니다. (mp4, avi, mov, mkv만 지원)'}), 400
        
        # 처리 옵션 확인
        remove_bg = request.form.get('remove_bg', 'false').lower() == 'true'
        upscale = request.form.get('upscale', 'false').lower() == 'true'
        scale_factor = int(request.form.get('scale_factor', '2'))
        background_color = request.form.get('background_color', '#FFFFFF')  # 기본값: 흰색
        
        if not remove_bg and not upscale:
            progress_callback(0, "❌ 최소 하나의 처리 옵션을 선택해야 합니다.")
            return jsonify({'error': '배경제거 또는 업스케일링 중 하나는 선택해야 합니다.'}), 400
        
        print(f"🎬 비디오 처리 시작 - 배경제거: {remove_bg}, 업스케일: {upscale} ({scale_factor}x), 배경색: {background_color}")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        
        progress_callback(8, "📁 임시 작업 공간 준비 중...")
        
        # 임시 디렉토리 생성
        temp_dir = video_processor.create_temp_dir(session_id)
        print(f"📁 임시 디렉토리 생성: {temp_dir}")
        
        # 비디오 파일 저장
        video_filename = secure_filename(file.filename)
        video_path = os.path.join(temp_dir, video_filename)
        print(f"💾 비디오 파일 저장 중: {video_path}")
        
        try:
            file.save(video_path)
            saved_size = os.path.getsize(video_path)
            print(f"✅ 비디오 파일 저장 완료: {saved_size:,} bytes")
        except Exception as save_error:
            print(f"❌ 비디오 파일 저장 실패: {save_error}")
            raise ValueError(f"비디오 파일 저장 실패: {save_error}")
        
        # 저장된 파일 검증
        if not os.path.exists(video_path):
            raise ValueError("저장된 비디오 파일이 존재하지 않습니다")
        
        if os.path.getsize(video_path) == 0:
            raise ValueError("저장된 비디오 파일이 비어있습니다")
        
        try:
            # 1. 프레임 추출
            progress_callback(10, "🎞️ 비디오에서 프레임 추출 중...")
            video_info = video_processor.extract_frames(video_path, temp_dir, progress_callback)
            
            # 2. 각 프레임 처리
            progress_callback(20, "🤖 AI가 각 프레임을 처리하고 있습니다...")
            processed_files = video_processor.process_frames(
                video_info['frame_files'], 
                remove_bg, 
                upscale, 
                scale_factor, 
                background_color,  # 배경 색상 전달
                progress_callback
            )
            
            # 3. 비디오 재조립
            progress_callback(80, "🎬 처리된 프레임들을 비디오로 재조립 중...")
            
            # 업스케일링이 적용된 경우 크기 조정
            final_width = video_info['width']
            final_height = video_info['height']
            if upscale:
                final_width *= scale_factor
                final_height *= scale_factor
            
            # 출력 파일명 생성
            output_filename = f"processed_{session_id[:8]}_{video_filename}"
            output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
            
            video_processor.reassemble_video(
                processed_files, 
                output_path, 
                video_info['fps'], 
                final_width, 
                final_height, 
                progress_callback
            )
            
            progress_callback(100, "🎉 비디오 처리가 완료되었습니다!")
            
            # 실제 저장된 파일명 찾기 (downloads 폴더에서 가장 최근 파일)
            download_folder = app.config['DOWNLOAD_FOLDER']
            actual_filename = None
            
            # downloads 폴더의 모든 파일 중 session_id로 시작하는 가장 최근 파일 찾기
            try:
                files = [f for f in os.listdir(download_folder) if f.startswith(f"processed_{session_id[:8]}")]
                if files:
                    # 가장 최근 파일 선택
                    files.sort(key=lambda f: os.path.getmtime(os.path.join(download_folder, f)), reverse=True)
                    actual_filename = files[0]
                    print(f"📁 실제 저장된 파일명: {actual_filename}")
                else:
                    actual_filename = output_filename  # 폴백
                    print(f"⚠️ 파일을 찾을 수 없어 예상 파일명 사용: {actual_filename}")
            except Exception as e:
                actual_filename = output_filename  # 폴백
                print(f"⚠️ 파일 검색 실패, 예상 파일명 사용: {actual_filename} (오류: {e})")
            
            # 처리 결과 정보
            processing_info = []
            if remove_bg:
                processing_info.append("배경제거")
            if upscale:
                processing_info.append(f"{scale_factor}x 업스케일링")
            
            print(f"✅ 비디오 처리 완료 - {', '.join(processing_info)} 적용")
            
            # 세션 데이터 저장 (실제 파일명 사용)
            update_session_data(session_id, {
                'type': 'video',
                'filename': actual_filename,
                'original_filename': file.filename,
                'download_url': url_for('download_file', filename=actual_filename),
                'processing_info': processing_info,
                'original_frames': video_info['total_frames'],
                'fps': video_info['fps'],
                'resolution': f"{final_width}x{final_height}",
                'remove_bg': remove_bg,
                'upscale': upscale,
                'scale_factor': scale_factor,
                'completed': True,
                'timestamp': time.time()
            })
            
            return jsonify({
                'success': True,
                'download_url': url_for('download_file', filename=actual_filename),
                'session_id': session_id,
                'work_id': session_id,  # 프론트엔드에서 URL 변경에 사용
                'processing_info': processing_info,
                'original_frames': video_info['total_frames'],
                'fps': video_info['fps'],
                'resolution': f"{final_width}x{final_height}"
            })
            
        finally:
            # 임시 파일들 정리 (비동기로 실행) - 완료 감지 후 정리되도록 시간 연장
            def cleanup():
                time.sleep(10)  # 10초 대기로 늘림 - 완료 감지 시간 확보
                video_processor.cleanup_temp_dir(session_id)
            
            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
        
    except Exception as e:
        print(f"❌ 비디오 처리 실패: {e}")
        if session_id:
            if session_id in progress_queues:
                send_progress(session_id, 0, f"❌ 오류: {str(e)}")
            # 오류 시에도 정리
            video_processor.cleanup_temp_dir(session_id)
        return jsonify({'error': f'비디오 처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/video_progress_files/<session_id>')
def video_progress_files(session_id):
    """실제 처리된 파일 수 확인 (폴링용)"""
    try:
        temp_dir = os.path.join(app.config['TEMP_FOLDER'], session_id)
        frames_dir = os.path.join(temp_dir, 'frames')
        processed_dir = os.path.join(temp_dir, 'processed')
        
        # 총 프레임 수와 처리된 프레임 수 (temp 디렉토리가 있는 경우만)
        total_frames = 0
        processed_count = 0
        
        if os.path.exists(temp_dir):
            # 총 프레임 수 (frames 디렉토리의 파일 수)
            if os.path.exists(frames_dir):
                total_frames = len([f for f in os.listdir(frames_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # 처리된 프레임 수 (processed 디렉토리의 파일 수)
            if os.path.exists(processed_dir):
                processed_count = len([f for f in os.listdir(processed_dir) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 완료 여부 확인 - downloads 폴더에서 session_id로 시작하는 비디오 파일 찾기
        download_folder = app.config['DOWNLOAD_FOLDER']
        completed = False
        download_url = None
        actual_filename = None
        
        try:
            # downloads 폴더에서 session_id의 첫 8자리로 시작하는 mp4 파일 찾기
            session_prefix = session_id[:8]  # 'd78709e1' 부분만 사용
            video_files = [f for f in os.listdir(download_folder) 
                          if f.startswith(f"processed_{session_prefix}") and f.lower().endswith('.mp4')]
            
            if video_files:
                # 가장 최근 파일 선택
                video_files.sort(key=lambda f: os.path.getmtime(os.path.join(download_folder, f)), reverse=True)
                actual_filename = video_files[0]
                completed = True
                download_url = url_for('download_file', filename=actual_filename)
                print(f"✅ 완료된 비디오 파일 발견: {actual_filename}")
                
                # temp 디렉토리가 없어도 완료로 처리 (이미 정리된 경우)
                if total_frames == 0:
                    total_frames = processed_count = 1  # 완료 표시를 위한 더미 값
                    
            else:
                print(f"⏳ 비디오 파일 아직 준비중: processed_{session_prefix}*.mp4")
                
        except Exception as e:
            print(f"⚠️ 완료 파일 검색 실패: {e}")
        
        # temp 디렉토리도 없고 완료 파일도 없으면 404
        if not os.path.exists(temp_dir) and not completed:
            return jsonify({
                'processed_count': 0, 
                'total_frames': 0, 
                'completed': False,
                'error': '세션을 찾을 수 없습니다.'
            }), 404
        
        print(f"📊 파일 수 확인 - {session_id}: {processed_count}/{total_frames} (완료: {completed})")
        
        return jsonify({
            'processed_count': processed_count,
            'total_frames': total_frames,
            'completed': completed,
            'download_url': download_url,
            'filename': actual_filename,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"❌ 파일 수 확인 실패 ({session_id}): {e}")
        return jsonify({
            'processed_count': 0,
            'total_frames': 0, 
            'completed': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    """처리된 파일 다운로드"""
    try:
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없음: {file_path}")
            # downloads 폴더의 모든 파일 나열 (디버깅용)
            download_folder = app.config['DOWNLOAD_FOLDER']
            if os.path.exists(download_folder):
                existing_files = os.listdir(download_folder)
                print(f"📁 downloads 폴더의 기존 파일들: {existing_files}")
                
                # filename과 유사한 파일이 있는지 확인
                similar_files = [f for f in existing_files if filename.split('_')[0] in f or filename.split('.')[0] in f]
                if similar_files:
                    print(f"🔍 유사한 파일 발견: {similar_files}")
            else:
                print(f"❌ downloads 폴더가 존재하지 않음: {download_folder}")
            
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        print(f"📥 다운로드 시작: {filename} (크기: {file_size:,} bytes)")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print(f"📍 요청된 파일: {filename}")
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

@app.route('/extract_last_frame', methods=['POST'])
def extract_last_frame():
    """비디오에서 마지막 프레임 추출 및 다운로드"""
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 비디오 파일 검증 중...")
        
        # 파일 확인
        if 'file' not in request.files:
            progress_callback(0, "❌ 비디오 파일이 선택되지 않았습니다.")
            return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            progress_callback(0, "❌ 비디오 파일이 선택되지 않았습니다.")
            return jsonify({'error': '비디오 파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_video_file(file.filename):
            progress_callback(0, "❌ 지원하지 않는 비디오 형식입니다.")
            return jsonify({'error': '지원하지 않는 비디오 형식입니다. (mp4, avi, mov, mkv만 지원)'}), 400
        
        print(f"🎬 마지막 프레임 추출 시작")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        
        progress_callback(8, "📁 임시 작업 공간 준비 중...")
        
        # 임시 디렉토리 생성
        temp_dir = video_processor.create_temp_dir(session_id)
        print(f"📁 임시 디렉토리 생성: {temp_dir}")
        
        # 비디오 파일 저장
        video_filename = secure_filename(file.filename)
        video_path = os.path.join(temp_dir, video_filename)
        print(f"💾 비디오 파일 저장 중: {video_path}")
        
        try:
            file.save(video_path)
            saved_size = os.path.getsize(video_path)
            print(f"✅ 비디오 파일 저장 완료: {saved_size:,} bytes")
        except Exception as save_error:
            print(f"❌ 비디오 파일 저장 실패: {save_error}")
            raise ValueError(f"비디오 파일 저장 실패: {save_error}")
        
        # 저장된 파일 검증
        if not os.path.exists(video_path):
            raise ValueError("저장된 비디오 파일이 존재하지 않습니다")
        
        if os.path.getsize(video_path) == 0:
            raise ValueError("저장된 비디오 파일이 비어있습니다")
        
        try:
            # 마지막 프레임 추출
            progress_callback(10, "🎞️ 비디오에서 마지막 프레임 추출 중...")
            frame_info = video_processor.extract_last_frame(video_path, progress_callback)
            
            progress_callback(90, "💾 마지막 프레임 이미지 저장 중...")
            
            # 결과 파일명 생성
            frame_filename = f"last_frame_{session_id[:8]}_{video_filename.rsplit('.', 1)[0]}.png"
            frame_path = os.path.join(app.config['DOWNLOAD_FOLDER'], frame_filename)
            
            # 마지막 프레임을 PNG로 저장
            frame_info['frame_image'].save(frame_path, 'PNG', optimize=True, compress_level=6)
            
            progress_callback(100, "🎉 마지막 프레임 추출 완료!")
            
            # 세션 데이터 저장
            update_session_data(session_id, {
                'type': 'last_frame',
                'filename': frame_filename,
                'original_filename': file.filename,
                'download_url': url_for('download_file', filename=frame_filename),
                'frame_info': {
                    'width': frame_info['width'],
                    'height': frame_info['height'],
                    'total_frames': frame_info['total_frames'],
                    'fps': frame_info['fps'],
                    'frame_index': frame_info['frame_index']
                },
                'completed': True,
                'timestamp': time.time()
            })
            
            print(f"✅ 마지막 프레임 추출 완료:")
            print(f"   - 프레임 인덱스: {frame_info['frame_index']}/{frame_info['total_frames']}")
            print(f"   - 해상도: {frame_info['width']}x{frame_info['height']}")
            print(f"   - 저장 위치: {frame_path}")
            
            return jsonify({
                'success': True,
                'download_url': url_for('download_file', filename=frame_filename),
                'session_id': session_id,
                'work_id': session_id,  # 프론트엔드에서 URL 변경에 사용
                'frame_info': {
                    'width': frame_info['width'],
                    'height': frame_info['height'],
                    'total_frames': frame_info['total_frames'],
                    'fps': frame_info['fps'],
                    'frame_index': frame_info['frame_index']
                }
            })
            
        finally:
            # 임시 파일들 정리 (비동기로 실행)
            def cleanup():
                time.sleep(5)  # 5초 대기 후 정리
                video_processor.cleanup_temp_dir(session_id)
            
            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
        
    except Exception as e:
        print(f"❌ 마지막 프레임 추출 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'마지막 프레임 추출 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_image():
    """이미지 벡터화 처리"""
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 파일 검증 중...")
        
        # 파일 확인
        if 'file' not in request.files:
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        # 벡터화 옵션 확인
        n_colors = request.form.get('n_colors', '8')
        output_format = request.form.get('output_format', 'svg')
        
        try:
            n_colors = int(n_colors)
            if n_colors < 2 or n_colors > 32:
                n_colors = 8  # 기본값
        except ValueError:
            n_colors = 8
        
        if output_format not in ['svg']:  # 현재는 SVG만 지원
            output_format = 'svg'
        
        progress_callback(10, f"📐 {n_colors}색상 벡터화 시작...")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        
        # 이미지 로드
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            progress_callback(0, "❌ 이미지를 로드할 수 없습니다.")
            return jsonify({'error': f'이미지 파일이 손상되었습니다: {str(e)}'}), 400
        
        progress_callback(15, "🧠 AI 벡터화 모델 로드 중...")
        
        # 벡터화 처리
        try:
            svg_content = vectorizer_model.vectorize_image(
                image, 
                output_format=output_format, 
                n_colors=n_colors, 
                progress_callback=progress_callback
            )
        except Exception as e:
            progress_callback(0, f"❌ 벡터화 실패: {str(e)}")
            return jsonify({'error': f'벡터화 처리 실패: {str(e)}'}), 500
        
        progress_callback(98, "💾 SVG 파일 저장 중...")
        
        # 결과 파일 저장
        unique_id = str(uuid.uuid4())[:8]
        result_filename = f"vectorized_{unique_id}.svg"
        result_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], result_filename)
        
        # SVG 파일로 저장
        with open(result_filepath, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        progress_callback(100, "🎉 이미지 벡터화 완료!")
        
        # 세션 데이터 저장
        update_session_data(session_id, {
            'type': 'vectorize',
            'filename': result_filename,
            'original_filename': file.filename,
            'download_url': url_for('download_file', filename=result_filename),
            'n_colors': n_colors,
            'output_format': output_format,
            'file_size': len(svg_content),
            'completed': True,
            'timestamp': time.time()
        })
        
        print(f"✅ 이미지 벡터화 완료 - {n_colors}색상, {len(svg_content):,}자 SVG")
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=result_filename),
            'session_id': session_id,
            'work_id': session_id,  # 프론트엔드에서 URL 변경에 사용
            'n_colors': n_colors,
            'output_format': output_format,
            'file_size': len(svg_content)
        })
        
    except Exception as e:
        print(f"❌ 벡터화 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
@app.route('/reset/<work_id>', methods=['POST'])
def reset_session(work_id=None):
    """세션 리셋 및 정리"""
    try:
        cleaned_items = []
        
        # work_id가 제공된 경우 해당 세션 정리
        if work_id and work_id in session_storage:
            # 세션 데이터 제거
            del session_storage[work_id]
            save_session_storage(session_storage)  # 파일에도 반영
            cleaned_items.append(f"세션 데이터: {work_id}")
            print(f"🧹 세션 데이터 정리 완료: {work_id}")
        
        # 프로그레스 큐 정리
        if work_id and work_id in progress_queues:
            del progress_queues[work_id]
            cleaned_items.append("프로그레스 큐")
            print(f"🧹 프로그레스 큐 정리 완료: {work_id}")
        
        # 비디오 처리 임시 디렉토리 정리
        if work_id:
            video_processor.cleanup_temp_dir(work_id)
            cleaned_items.append("임시 파일")
        
        # 전체 리셋인 경우 (work_id가 없는 경우)
        if not work_id:
            session_storage.clear()
            save_session_storage(session_storage)  # 파일에도 반영
            progress_queues.clear()
            cleaned_items.append("모든 세션 데이터")
            print("🧹 전체 세션 데이터 정리 완료")
        
        return jsonify({
            'success': True,
            'message': '세션이 성공적으로 리셋되었습니다.',
            'cleaned_items': cleaned_items
        })
        
    except Exception as e:
        print(f"❌ 세션 리셋 실패: {e}")
        return jsonify({'error': f'세션 리셋 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 고품질 AI 이미지 처리 웹 애플리케이션을 시작합니다...")
    print("🔥 배경 제거 + 업스케일링 + 벡터화 통합 솔루션")
    print("📐 새로운 기능: 이미지 벡터화 (SVG 변환)")
    
    # 지원 형식 출력
    extensions_list = ', '.join(sorted(ALLOWED_EXTENSIONS))
    print(f"📁 지원 형식: {extensions_list}")
    print(f"📏 최대 파일 크기: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"🌐 서버 주소: http://localhost:8080")
    
    # 모델 미리 로드 (성능 향상)
    try:
        print("🔄 고품질 AI 모델 미리 로드 중...")
        ai_model.load_model()
        print("✅ 고품질 AI 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 모델 미리 로드 실패 (첫 요청 시 로드됩니다): {e}")
    
    try:
        print("🔄 벡터화 모델 미리 로드 중...")
        vectorizer_model.load_model()
        print("✅ 벡터화 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 벡터화 모델 미리 로드 실패 (첫 요청 시 로드됩니다): {e}")
    
    app.run(debug=True, host='0.0.0.0', port=8080) 