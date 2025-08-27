"""
Potrace + VTracer 기반 이미지 벡터화 모듈
"""

import os
import subprocess
import tempfile
import cv2
import numpy as np
import uuid
from PIL import Image


class ImageVectorizerModel:
    """AI 강화 고품질 이미지 벡터화 모델 (Potrace 기반)"""
    
    def __init__(self, birefnet_model=None, upscaler_model=None):
        self.loaded = False
        self.birefnet_model = birefnet_model  # 배경 제거용
        self.upscaler_model = upscaler_model  # 업스케일링용
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
            if self._check_potrace():
                potrace_path = self._get_potrace_path()
                result = subprocess.run([potrace_path, '--version'], 
                                      capture_output=True, text=True)
                print(f"✅ Potrace 발견: {potrace_path}")
                print(f"🎯 Potrace 버전: {result.stdout.strip()}")
            else:
                print("⚠️ Potrace가 설치되지 않음")
            
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
                '/opt/homebrew/bin/potrace',  # macOS Homebrew
                '/usr/local/bin/potrace',     # 일반적인 Linux/macOS
                '/usr/bin/potrace'            # 시스템 기본
            ]
            
            for potrace_path in potrace_paths:
                try:
                    if os.path.exists(potrace_path):
                        # 실행 권한 확인
                        result = subprocess.run([potrace_path, '--version'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            return True
                except (subprocess.SubprocessError, OSError, FileNotFoundError):
                    continue
                    
            return False
        except Exception as e:
            print(f"❌ Potrace 확인 중 오류: {e}")
            return False
    
    def _get_potrace_path(self):
        """Potrace 실행 파일 경로 반환"""
        potrace_paths = [
            'models/potrace',  # models 디렉토리 우선
            './potrace',
            'potrace',
            '/opt/homebrew/bin/potrace',
            '/usr/local/bin/potrace',
            '/usr/bin/potrace'
        ]
        
        for potrace_path in potrace_paths:
            try:
                if os.path.exists(potrace_path):
                    result = subprocess.run([potrace_path, '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return potrace_path
            except (subprocess.SubprocessError, OSError, FileNotFoundError):
                continue
        
        return 'potrace'  # fallback

    def _vtracer_vectorize(self, image, svg_path):
        """전문 VTracer 라이브러리를 사용한 고품질 벡터화"""
        try:
            import vtracer
            
            print("🔥 전문 VTracer 라이브러리 벡터화 시작!")
            
            # @models/ 폴더에 임시 파일 생성 (AI 모델 통합 관리)
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # 임시 입력 파일을 @models/에 생성
            temp_input_path = os.path.join(models_dir, f'vtracer_input_{uuid.uuid4().hex[:8]}.png')
            image.save(temp_input_path, 'PNG')
            
            # VTracer 파라미터 설정 (올바른 API 사용)
            try:
                vtracer.convert_image_to_svg_py(
                    image_path=temp_input_path,  # 올바른 파라미터명
                    out_path=svg_path,  # 올바른 파라미터명
                    colormode='color',  # 컬러 모드
                    hierarchical='stacked',  # 계층 구조
                    mode='spline',  # 스플라인 곡선
                    filter_speckle=4,  # 노이즈 제거
                    color_precision=6,  # 색상 정밀도
                    layer_difference=16,  # 레이어 차이
                    corner_threshold=60,  # 코너 임계값
                    length_threshold=4.0,  # 길이 임계값
                    max_iterations=10,  # 최대 반복
                    splice_threshold=45,  # 연결 임계값
                    path_precision=3  # 경로 정밀도
                )
                
                # 결과 확인
                if os.path.exists(svg_path) and os.path.getsize(svg_path) > 0:
                    print("🎯 VTracer 라이브러리 벡터화 성공!")
                    # @models/ 폴더의 임시 파일 정리
                    try:
                        os.unlink(temp_input_path)
                        print(f"🧹 @models/ 임시 파일 정리 완료: {os.path.basename(temp_input_path)}")
                    except Exception as cleanup_error:
                        print(f"⚠️ 임시 파일 정리 실패: {cleanup_error}")
                    return True
                else:
                    print("⚠️ VTracer 결과 파일이 비어있거나 생성되지 않음")
                    return False
                    
            except Exception as vtracer_error:
                print(f"⚠️ VTracer 처리 오류: {vtracer_error}")
                return False
            
        except ImportError:
            print("⚠️ VTracer 라이브러리가 설치되지 않음")
            return False
        except Exception as e:
            print(f"❌ VTracer 벡터화 실패: {e}")
            return False

    def _ai_preprocess(self, image, progress_callback=None):
        """AI 기반 이미지 전처리 (업스케일링 + 배경제거 + 엣지 강화)"""
        try:
            width, height = image.size
            
            # 전처리 모드 설정
            skip_all_preprocessing = True  # 사용자 요청: 모든 전처리 스킵
            
            if skip_all_preprocessing:
                print("🚫 벡터화용 전처리 완전 스킵: 원본 그대로 사용")
                return image
            
            # AI 업스케일링 (작은 이미지만)
            if self.upscaler_model and (width < 512 or height < 512):
                if progress_callback:
                    progress_callback(20, "🚀 AI 업스케일링으로 품질 향상 중...")
                print("🔍 작은 이미지 감지, AI 업스케일링 적용")
                image = self.upscaler_model.upscale_image(image, scale=2)
                print(f"📈 AI 업스케일링 완료: {width}x{height} → {image.width}x{image.height}")
            
            # AI 배경 제거 (선택적)
            preserve_original_colors = True
            skip_background_removal = True  # 원본 색상 보존을 위해 배경제거 스킵
            
            if self.birefnet_model and not skip_background_removal:
                if progress_callback:
                    progress_callback(40, "🤖 AI 배경 제거로 주요 객체 추출 중...")
                print("🎯 AI 배경 제거 적용")
                foreground = self.birefnet_model.remove_background(image)
                
                # 투명 배경을 흰색으로 변환 (벡터화에 적합)
                white_bg = Image.new('RGB', foreground.size, (255, 255, 255))
                if foreground.mode == 'RGBA':
                    white_bg.paste(foreground, mask=foreground.split()[-1])
                    image = white_bg
                print("✅ AI 배경 제거 완료")
            
            # 이미지를 numpy 배열로 변환
            img_array = np.array(image)
            
            if progress_callback:
                progress_callback(60, "🔧 AI 이미지 최적화 중...")
            
            # 색상 보존 모드에 따른 전처리
            if preserve_original_colors:
                print("🌈 원본 색상 보존 모드: 최소한의 전처리만 적용")
                # 원본 색상 보존: 매우 가벼운 노이즈 제거만
                img_array = cv2.medianBlur(img_array, 3)
            else:
                # 기존 방식: 강력한 전처리
                print("🔧 고품질 전처리 모드")
                # 1. 가우시안 블러로 노이즈 제거
                img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
                
                # 2. 언샤프 마스킹으로 선명화
                gaussian_3 = cv2.GaussianBlur(img_array, (9, 9), 10.0)
                img_array = cv2.addWeighted(img_array, 1.5, gaussian_3, -0.5, 0)
                
                # 3. CLAHE로 명암 대비 향상
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                # 4. 캐니 엣지로 구조 강화
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # 엣지 정보를 원본에 약간 혼합
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                img_array = cv2.addWeighted(img_array, 0.9, edges_colored, 0.1, 0)
                
                # 5. 모폴로지 연산으로 노이즈 제거
                kernel = np.ones((2,2), np.uint8)
                img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            processed_image = Image.fromarray(img_array.astype(np.uint8))
            
            if progress_callback:
                progress_callback(80, "✅ AI 전처리 완료")
            
            print("🎨 AI 기반 이미지 전처리 완료")
            return processed_image
            
        except Exception as e:
            print(f"❌ AI 전처리 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 전처리 실패: {str(e)}")
            return image  # 실패시 원본 반환

    def _quantize_colors(self, image, n_colors=8):
        """K-means를 사용한 색상 양자화"""
        try:
            print(f"🎨 K-means 색상 양자화 시작: {n_colors}색상으로 분리")
            
            # 기존 K-means 양자화 로직
            img_array = np.array(image)
            data = img_array.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            quantized_data = centers[labels.flatten()]
            quantized_image = quantized_data.reshape(img_array.shape)
            
            return Image.fromarray(quantized_image)
        except Exception as e:
            print(f"❌ 색상 양자화 실패: {e}")
            return image

    def vectorize_image(self, image, output_format='svg', n_colors=8, vectorize_mode='color', progress_callback=None):
        """AI 강화 이미지 벡터화
        
        Args:
            vectorize_mode: 'color' for VTracer (컬러), 'bw' for Potrace (흑백)
        """
        if not self.loaded:
            self.load_model()
        
        try:
            # 1. AI 전처리 (업스케일링 + 배경제거 + 엣지강화)
            if progress_callback:
                progress_callback(10, "🧠 AI 전처리 시작...")
            processed_image = self._ai_preprocess(image, progress_callback)
            
            # 2. 색상 양자화 (K-means)
            if progress_callback:
                progress_callback(50, "🎨 색상 최적화 중...")
            quantized_image = self._quantize_colors(processed_image, n_colors)
            
            # 3. 임시 파일 생성
            if progress_callback:
                progress_callback(60, "🧠 전문 AI 벡터화 처리 중...")
            
            temp_bmp_path = tempfile.mktemp(suffix='.bmp')
            quantized_image.save(temp_bmp_path, 'BMP')
            
            temp_svg_path = temp_bmp_path.replace('.bmp', '.svg')
            
            # 사용자 선택에 따른 벡터화 방식 결정
            vectorization_success = False
            
            if vectorize_mode == 'color':
                # 🌈 컬러 모드: VTracer 사용
                print("🎨 컬러 모드 선택: VTracer 라이브러리 시도!")
                if progress_callback:
                    progress_callback(60, "🎨 VTracer 컬러 벡터화 처리 중...")
                vectorization_success = self._vtracer_vectorize(quantized_image, temp_svg_path)
                if vectorization_success:
                    print("🎯 VTracer 컬러 벡터화 성공!")
                else:
                    print("⚠️ VTracer 실패, 내장 컬러 엔진으로 백업...")
                    if progress_callback:
                        progress_callback(70, "🎨 내장 컬러 엔진으로 백업 처리 중...")
                    vectorization_success = self._fallback_vectorize(quantized_image, temp_svg_path)
                    if vectorization_success:
                        print("🎯 내장 컬러 엔진 백업 성공!")
                
            elif vectorize_mode == 'bw':
                # ⚫⚪ 흑백 모드: Potrace 사용
                print("⚫⚪ 흑백 모드 선택: Potrace 라이브러리 시도!")
                if self._check_potrace():
                    if progress_callback:
                        progress_callback(60, "⚫⚪ Potrace 흑백 벡터화 처리 중...")
                    try:
                        vectorization_success = self._bitmap_to_svg_potrace(temp_bmp_path, temp_svg_path)
                        if vectorization_success:
                            print("🎯 Potrace 흑백 벡터화 성공!")
                            if progress_callback:
                                progress_callback(90, "✅ Potrace 흑백 벡터화 완료!")
                        else:
                            print("⚠️ Potrace 실패, 내장 엔진으로 백업...")
                    except Exception as e:
                        print(f"❌ Potrace 벡터화 예외 발생: {e}")
                        vectorization_success = False
                else:
                    print("⚠️ Potrace를 찾을 수 없음")
                    vectorization_success = False
                    
                # Potrace 실패 시 내장 엔진으로 백업
                if not vectorization_success:
                    print("⚠️ 내장 엔진으로 백업...")
                    if progress_callback:
                        progress_callback(70, "🎨 내장 엔진으로 백업 처리 중...")
                    vectorization_success = self._fallback_vectorize(quantized_image, temp_svg_path)
                    if vectorization_success:
                        print("🎯 내장 엔진 백업 성공!")
            
            else:
                # 기본값: 컬러 모드
                print("🎨 기본 컬러 모드: VTracer 라이브러리 시도!")
                if progress_callback:
                    progress_callback(60, "🎨 VTracer 컬러 벡터화 처리 중...")
                vectorization_success = self._vtracer_vectorize(quantized_image, temp_svg_path)
                if vectorization_success:
                    print("🎯 VTracer 컬러 벡터화 성공!")
                else:
                    print("⚠️ VTracer 실패, 내장 컬러 엔진으로 백업...")
                    if progress_callback:
                        progress_callback(70, "🎨 내장 컬러 엔진으로 백업 처리 중...")
                    vectorization_success = self._fallback_vectorize(quantized_image, temp_svg_path)
                    if vectorization_success:
                        print("🎯 내장 컬러 엔진 백업 성공!")
            
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
                progress_callback(100, "🎉 벡터화 완료!")
            
            # 파일 크기 정보 출력
            file_size = len(svg_content)
            print(f"✅ 이미지 벡터화 완료 - {file_size:,}자 SVG 생성")
            
            return svg_content
            
        except Exception as e:
            print(f"❌ 벡터화 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 벡터화 실패: {str(e)}")
            raise

    def _bitmap_to_svg_potrace(self, bitmap_path, svg_path):
        """Potrace를 사용한 비트맵 → SVG 변환"""
        try:
            # 동적으로 potrace 명령어 경로 결정
            potrace_paths = [
                'models/potrace',  # models 디렉토리 우선
                './potrace',
                'potrace'
            ]
            
            potrace_cmd = None
            for path in potrace_paths:
                try:
                    if os.path.exists(path):
                        result = subprocess.run([path, '--version'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            potrace_cmd = path
                            break
                except (subprocess.SubprocessError, OSError, FileNotFoundError):
                    continue
            
            if not potrace_cmd:
                potrace_cmd = 'potrace'  # fallback to system PATH
            
            # Potrace 명령어 실행
            cmd = [
                potrace_cmd,
                '--svg',           # SVG 출력
                '--output', svg_path,
                '--color', '#000000',  # 검은색 선
                '--turnpolicy', 'majority',  # 최적 정책
                '--alphamax', '1.0',  # 부드러운 곡선
                '--opttolerance', '0.2',  # 최적화 관용
                bitmap_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(svg_path):
                print(f"✅ Potrace 벡터화 성공: {svg_path}")
                return True
            else:
                print(f"❌ Potrace 실행 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Potrace 벡터화 오류: {e}")
            return False

    def _fallback_vectorize(self, image, output_path):
        """내장 벡터화 엔진 (Potrace 없을 때 사용)"""
        try:
            print("🔧 내장 벡터화 엔진 사용")
            img_array = np.array(image)
            
            # 그레이스케일 변환
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # 적응적 이진화 (Otsu's method)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 최소 크기 필터링
            min_area = 100
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            print(f"🔍 유효한 컨투어 {len(valid_contours)}개 찾음")
            
            if len(valid_contours) == 0:
                print("⚠️ 컨투어를 찾을 수 없음, 픽셀 기반 SVG 생성")
                return self._create_pixel_based_svg(img_array, output_path)
            
            # SVG 생성
            height, width = img_array.shape[:2]
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
'''
            
            for contour in valid_contours:
                # 컨투어 단순화
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) > 2:
                    path_data = f"M {approx[0][0][0]},{approx[0][0][1]}"
                    for point in approx[1:]:
                        path_data += f" L {point[0][0]},{point[0][1]}"
                    path_data += " Z"
                    
                    svg_content += f'  <path d="{path_data}" fill="black" stroke="black" stroke-width="0.5"/>\n'
            
            svg_content += '</svg>'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            print(f"✅ 내장 엔진 벡터화 완료: {len(svg_content)}자")
            return True
            
        except Exception as e:
            print(f"❌ 내장 벡터화 실패: {e}")
            return False

    def _create_pixel_based_svg(self, img_array, output_path):
        """픽셀 기반 SVG 생성 (마지막 폴백)"""
        try:
            print("🎯 픽셀 기반 SVG 생성 시작")
            height, width = img_array.shape[:2]
            
            # 이미지 크기 축소 (성능을 위해)
            scale_factor = max(1, min(width, height) // 200)
            if scale_factor > 1:
                small_width = width // scale_factor
                small_height = height // scale_factor
                if len(img_array.shape) == 3:
                    resized = cv2.resize(img_array, (small_width, small_height))
                else:
                    resized = cv2.resize(img_array, (small_width, small_height))
            else:
                resized = img_array
                small_width, small_height = width, height
            
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
'''
            
            # 픽셀을 작은 사각형으로 변환
            for y in range(0, small_height, 2):  # 2픽셀씩 건너뛰기
                for x in range(0, small_width, 2):
                    if len(resized.shape) == 3:
                        pixel = resized[y, x]
                        color = f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}"
                    else:
                        pixel_val = resized[y, x]
                        color = f"#{pixel_val:02x}{pixel_val:02x}{pixel_val:02x}"
                    
                    real_x = x * scale_factor
                    real_y = y * scale_factor
                    rect_size = scale_factor * 2
                    
                    svg_content += f'  <rect x="{real_x}" y="{real_y}" width="{rect_size}" height="{rect_size}" fill="{color}"/>\n'
            
            svg_content += '</svg>'
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(svg_content)
            
            print(f"✅ 픽셀 기반 SVG 생성 완료: {len(svg_content)}자")
            return True
            
        except Exception as e:
            print(f"❌ 픽셀 기반 SVG 생성 실패: {e}")
            return False

    def _optimize_svg(self, svg_path):
        """SVG 파일 최적화"""
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 소수점 자릿수 줄이기
            import re
            content = re.sub(r'(\d+\.\d{1})\d+', r'\1', content)
            
            # 불필요한 공백 제거
            content = re.sub(r'\s+', ' ', content)
            content = content.replace('> <', '><')
            
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("⚡ SVG 최적화 완료")
            
        except Exception as e:
            print(f"⚠️ SVG 최적화 실패: {e}")
