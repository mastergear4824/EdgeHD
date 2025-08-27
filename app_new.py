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
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
TEMP_FOLDER = 'temp'

# 업로드, 다운로드, 임시 디렉토리 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# VideoProcessor 클래스 (기존 유지 - 별도 모듈화는 나중에)
class VideoProcessor:
    """비디오 프레임별 처리 클래스"""
    
    def __init__(self, ai_model, upscale_model=None):
        self.ai_model = ai_model
        self.upscale_model = upscale_model
        self.progress = {}
        
    def extract_frames(self, video_path, work_id, max_frames=None):
        """비디오에서 프레임 추출"""
        import cv2
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 최대 프레임 수 제한
            if max_frames and total_frames > max_frames:
                total_frames = max_frames
                
            print(f"📺 비디오 정보: {total_frames}프레임, {fps:.1f}fps")
            
            frames_dir = os.path.join(TEMP_FOLDER, work_id, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            frame_paths = []
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # OpenCV는 BGR이므로 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)
                
                frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
                frame_image.save(frame_path)
                frame_paths.append(frame_path)
                
                # 진행률 업데이트
                progress = int((i + 1) / total_frames * 30)  # 30%까지가 프레임 추출
                self.progress[work_id] = {
                    'progress': progress,
                    'message': f'프레임 추출 중... ({i+1}/{total_frames})'
                }
                
                if max_frames and i >= max_frames - 1:
                    break
                    
            cap.release()
            
            self.progress[work_id] = {
                'progress': 30,
                'message': f'프레임 추출 완료 ({len(frame_paths)}개)'
            }
            
            return frame_paths, fps
            
        except Exception as e:
            print(f"❌ 프레임 추출 실패: {e}")
            self.progress[work_id] = {
                'progress': 0,
                'message': f'프레임 추출 실패: {str(e)}'
            }
            raise

    def process_frames(self, frame_paths, work_id, operation='remove_bg', scale=None):
        """프레임들을 AI로 처리"""
        try:
            processed_dir = os.path.join(TEMP_FOLDER, work_id, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            
            processed_paths = []
            total_frames = len(frame_paths)
            
            for i, frame_path in enumerate(frame_paths):
                try:
                    # 이미지 로드
                    image = Image.open(frame_path)
                    
                    # AI 처리
                    if operation == 'remove_bg':
                        processed_image = self.ai_model.remove_background(image)
                    elif operation == 'upscale' and self.upscale_model:
                        processed_image = self.upscale_model.upscale_image(image, scale=scale)
                    else:
                        processed_image = image
                    
                    # 처리된 이미지 저장
                    frame_filename = os.path.basename(frame_path)
                    processed_path = os.path.join(processed_dir, f'{frame_filename.split(".")[0]}_processed.png')
                    
                    # PNG로 저장 (투명도 보존)
                    if processed_image.mode in ('RGBA', 'LA'):
                        processed_image.save(processed_path, 'PNG')
                    else:
                        processed_image.save(processed_path, 'PNG')
                    
                    processed_paths.append(processed_path)
                    
                    # 진행률 업데이트 (30% ~ 90%)
                    progress = 30 + int((i + 1) / total_frames * 60)
                    self.progress[work_id] = {
                        'progress': progress,
                        'message': f'프레임 처리 중... ({i+1}/{total_frames})'
                    }
                    
                except Exception as e:
                    print(f"⚠️ 프레임 {i} 처리 실패: {e}")
                    # 실패한 프레임은 원본 사용
                    processed_paths.append(frame_path)
            
            self.progress[work_id] = {
                'progress': 90,
                'message': f'프레임 처리 완료 ({len(processed_paths)}개)'
            }
            
            return processed_paths
            
        except Exception as e:
            print(f"❌ 프레임 처리 실패: {e}")
            self.progress[work_id] = {
                'progress': 0,
                'message': f'프레임 처리 실패: {str(e)}'
            }
            raise

    def create_video(self, frame_paths, output_path, fps, work_id):
        """처리된 프레임들로 비디오 생성"""
        import cv2
        
        try:
            if not frame_paths:
                raise ValueError("처리할 프레임이 없습니다")
            
            # 첫 번째 프레임으로 크기 확인
            first_frame = Image.open(frame_paths[0])
            width, height = first_frame.size
            
            # 비디오 라이터 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i, frame_path in enumerate(frame_paths):
                try:
                    # 이미지 로드
                    frame_image = Image.open(frame_path)
                    
                    # RGBA인 경우 RGB로 변환 (비디오는 알파 채널 지원 안함)
                    if frame_image.mode == 'RGBA':
                        # 흰색 배경으로 합성
                        white_bg = Image.new('RGB', frame_image.size, (255, 255, 255))
                        white_bg.paste(frame_image, mask=frame_image.split()[-1])
                        frame_image = white_bg
                    elif frame_image.mode != 'RGB':
                        frame_image = frame_image.convert('RGB')
                    
                    # PIL Image를 numpy 배열로 변환 후 BGR로 변환 (OpenCV 형식)
                    frame_array = np.array(frame_image)
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    
                    out.write(frame_bgr)
                    
                    # 진행률 업데이트 (90% ~ 100%)
                    progress = 90 + int((i + 1) / len(frame_paths) * 10)
                    self.progress[work_id] = {
                        'progress': progress,
                        'message': f'비디오 생성 중... ({i+1}/{len(frame_paths)})'
                    }
                    
                except Exception as e:
                    print(f"⚠️ 프레임 {i} 비디오 쓰기 실패: {e}")
            
            out.release()
            
            self.progress[work_id] = {
                'progress': 100,
                'message': '비디오 생성 완료!'
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 비디오 생성 실패: {e}")
            self.progress[work_id] = {
                'progress': 0,
                'message': f'비디오 생성 실패: {str(e)}'
            }
            raise

# 전역 변수들
progress_data = {}
session_data = {}

# AI 모델 인스턴스 생성
ai_model = BiRefNetModel()
upscale_model = RealESRGANUpscaleModel()
vectorizer_model = ImageVectorizerModel(ai_model, upscale_model)

# 비디오 프로세서 인스턴스
video_processor = VideoProcessor(ai_model, upscale_model)

def load_models():
    """AI 모델들을 백그라운드에서 로드"""
    print("🔄 고품질 AI 모델 미리 로드 중...")
    try:
        ai_model.load_model()
        print("✅ 고품질 AI 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ AI 모델 로드 실패: {e}")
    
    print("🔄 벡터화 모델 미리 로드 중...")
    try:
        vectorizer_model.load_model()
        print("✅ 벡터화 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 벡터화 모델 로드 실패: {e}")

# 기존 세션 데이터 복원
def restore_sessions():
    """애플리케이션 시작시 기존 세션 데이터 복원"""
    try:
        global session_data
        if os.path.exists(TEMP_FOLDER):
            sessions = [d for d in os.listdir(TEMP_FOLDER) if os.path.isdir(os.path.join(TEMP_FOLDER, d)) and d != 'test']
            
            for session_id in sessions:
                session_path = os.path.join(TEMP_FOLDER, session_id)
                # 기본 세션 데이터 생성
                session_data[session_id] = {
                    'created_at': time.time(),
                    'last_activity': time.time(),
                    'files': [],
                    'work_id': session_id
                }
                
                # 업로드된 파일 목록 복원
                uploads_path = os.path.join(session_path, 'uploads')
                if os.path.exists(uploads_path):
                    for file in os.listdir(uploads_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            session_data[session_id]['files'].append({
                                'name': file,
                                'path': os.path.join(uploads_path, file),
                                'type': 'image'
                            })
            
            print(f"📂 기존 세션 {len(session_data)}개 복원됨")
    except Exception as e:
        print(f"⚠️ 세션 복원 실패: {e}")

# 앱 시작시 실행
restore_sessions()

# 모델 로드를 백그라운드에서 실행
model_loading_thread = threading.Thread(target=load_models)
model_loading_thread.daemon = True
model_loading_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """파일 업로드 처리"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        work_id = request.form.get('work_id', str(uuid.uuid4()))
        
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 파일 크기 검사
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE//1024//1024}MB까지 지원됩니다.'}), 400
        
        # 안전한 파일명 생성
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{int(time.time())}.jpg"
        
        # 세션별 업로드 디렉토리
        session_upload_dir = os.path.join(TEMP_FOLDER, work_id, 'uploads')
        os.makedirs(session_upload_dir, exist_ok=True)
        
        # 파일 저장
        file_path = os.path.join(session_upload_dir, filename)
        file.save(file_path)
        
        # 세션 데이터 업데이트
        if work_id not in session_data:
            session_data[work_id] = {
                'created_at': time.time(),
                'last_activity': time.time(),
                'files': [],
                'work_id': work_id
            }
        
        session_data[work_id]['files'].append({
            'name': filename,
            'path': file_path,
            'type': 'image' if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')) else 'video'
        })
        session_data[work_id]['last_activity'] = time.time()
        
        return jsonify({
            'message': '파일 업로드 완료',
            'filename': filename,
            'work_id': work_id,
            'file_size': file_size
        })
        
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        return jsonify({'error': f'업로드 실패: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    """이미지 처리 (배경 제거)"""
    try:
        data = request.get_json()
        work_id = data.get('work_id')
        filename = data.get('filename')
        
        if not work_id or not filename:
            return jsonify({'error': '필수 파라미터가 누락되었습니다.'}), 400
        
        print(f"📍 세션 ID: {work_id} (프론트엔드에서 전달: {bool(work_id)})")
        
        # 파일 경로 찾기
        file_path = os.path.join(TEMP_FOLDER, work_id, 'uploads', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        # 진행률 초기화
        progress_data[work_id] = {'progress': 0, 'message': '처리 시작...'}
        
        def progress_callback(progress, message):
            progress_data[work_id] = {'progress': progress, 'message': message}
        
        # 이미지 로드 및 처리
        image = Image.open(file_path)
        
        progress_callback(10, '🤖 AI 모델 초기화 중...')
        
        # AI 배경 제거
        processed_image = ai_model.remove_background(image, progress_callback)
        
        # 결과 저장
        result_filename = f"processed_{int(time.time())}.png"
        result_path = os.path.join(DOWNLOAD_FOLDER, result_filename)
        processed_image.save(result_path)
        
        # 세션 데이터 저장
        session_data[work_id]['last_activity'] = time.time()
        print(f"💾 세션 데이터 저장됨: {work_id}")
        
        progress_callback(100, '✅ 처리 완료!')
        
        return jsonify({
            'message': '배경 제거 완료',
            'download_url': url_for('download_file', filename=result_filename),
            'work_id': work_id
        })
        
    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        if work_id:
            progress_data[work_id] = {'progress': 0, 'message': f'처리 실패: {str(e)}'}
        return jsonify({'error': f'처리 실패: {str(e)}'}), 500

@app.route('/upscale', methods=['POST'])
def upscale_image():
    """이미지 업스케일링"""
    try:
        data = request.get_json()
        work_id = data.get('work_id')
        filename = data.get('filename')
        scale = int(data.get('scale', 4))
        
        if not work_id or not filename:
            return jsonify({'error': '필수 파라미터가 누락되었습니다.'}), 400
        
        # 파일 경로 찾기
        file_path = os.path.join(TEMP_FOLDER, work_id, 'uploads', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        # 진행률 초기화
        progress_data[work_id] = {'progress': 0, 'message': '업스케일링 시작...'}
        
        def progress_callback(progress, message):
            progress_data[work_id] = {'progress': progress, 'message': message}
        
        # 이미지 로드 및 처리
        image = Image.open(file_path)
        
        progress_callback(10, f'🚀 {scale}x AI 업스케일링 준비 중...')
        
        # AI 업스케일링
        upscaled_image = upscale_model.upscale_image(image, scale=scale, progress_callback=progress_callback)
        
        # 결과 저장
        result_filename = f"upscaled_{scale}x_{int(time.time())}.png"
        result_path = os.path.join(DOWNLOAD_FOLDER, result_filename)
        upscaled_image.save(result_path)
        
        # 세션 데이터 저장
        session_data[work_id]['last_activity'] = time.time()
        
        progress_callback(100, f'✅ {scale}x 업스케일링 완료!')
        
        return jsonify({
            'message': f'{scale}x 업스케일링 완료',
            'download_url': url_for('download_file', filename=result_filename),
            'work_id': work_id
        })
        
    except Exception as e:
        print(f"❌ 업스케일링 실패: {e}")
        if work_id:
            progress_data[work_id] = {'progress': 0, 'message': f'업스케일링 실패: {str(e)}'}
        return jsonify({'error': f'업스케일링 실패: {str(e)}'}), 500

@app.route('/vectorize', methods=['POST'])
def vectorize_image():
    """이미지 벡터화"""
    try:
        work_id = request.form.get('work_id')
        filename = request.form.get('filename')
        output_format = request.form.get('output_format', 'svg')
        ai_auto_colors = request.form.get('ai_auto_colors', 'true') == 'true'
        
        if not work_id or not filename:
            return jsonify({'error': '필수 파라미터가 누락되었습니다.'}), 400
        
        print(f"📍 세션 ID: {work_id} (프론트엔드에서 전달: {bool(work_id)})")
        
        # 파일 경로 찾기
        file_path = os.path.join(TEMP_FOLDER, work_id, 'uploads', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        # 진행률 초기화
        progress_data[work_id] = {'progress': 0, 'message': '벡터화 시작...'}
        
        def progress_callback(progress, message):
            progress_data[work_id] = {'progress': progress, 'message': message}
        
        # 이미지 로드 및 처리
        image = Image.open(file_path)
        
        progress_callback(10, '🎨 AI 벡터화 준비 중...')
        
        # AI 벡터화 (AI 자동 색상 모드)
        n_colors = 8  # AI가 자동으로 조정하므로 기본값
        svg_content = vectorizer_model.vectorize_image(
            image, 
            output_format=output_format, 
            n_colors=n_colors,
            progress_callback=progress_callback
        )
        
        # 결과 저장
        result_filename = f"vectorized_{uuid.uuid4().hex[:8]}.{output_format}"
        result_path = os.path.join(DOWNLOAD_FOLDER, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        # 세션 데이터 저장
        session_data[work_id]['last_activity'] = time.time()
        print(f"💾 세션 데이터 저장됨: {work_id}")
        
        progress_callback(100, '✅ 벡터화 완료!')
        
        # 파일 크기 정보
        file_size = len(svg_content)
        print(f"✅ 이미지 벡터화 완료 - {n_colors}색상, {file_size:,}자 SVG")
        
        return jsonify({
            'message': '벡터화 완료',
            'download_url': url_for('download_file', filename=result_filename),
            'work_id': work_id,
            'file_size': file_size,
            'colors_used': n_colors
        })
        
    except Exception as e:
        print(f"❌ 벡터화 실패: {e}")
        if work_id:
            progress_data[work_id] = {'progress': 0, 'message': f'벡터화 실패: {str(e)}'}
        return jsonify({'error': f'벡터화 실패: {str(e)}'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """비디오 처리"""
    try:
        data = request.get_json()
        work_id = data.get('work_id')
        filename = data.get('filename')
        operation = data.get('operation', 'remove_bg')  # 'remove_bg' 또는 'upscale'
        scale = int(data.get('scale', 4)) if operation == 'upscale' else None
        max_frames = int(data.get('max_frames', 60))  # 최대 프레임 수 제한
        
        if not work_id or not filename:
            return jsonify({'error': '필수 파라미터가 누락되었습니다.'}), 400
        
        # 파일 경로 찾기
        file_path = os.path.join(TEMP_FOLDER, work_id, 'uploads', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        # 진행률 초기화
        video_processor.progress[work_id] = {'progress': 0, 'message': '비디오 처리 시작...'}
        
        def process_video_async():
            try:
                # 1. 프레임 추출
                frame_paths, fps = video_processor.extract_frames(file_path, work_id, max_frames)
                
                # 2. 프레임 처리
                processed_paths = video_processor.process_frames(frame_paths, work_id, operation, scale)
                
                # 3. 비디오 재생성
                output_suffix = f"_{operation}"
                if scale:
                    output_suffix += f"_{scale}x"
                
                result_filename = f"processed{output_suffix}_{int(time.time())}.mp4"
                result_path = os.path.join(DOWNLOAD_FOLDER, result_filename)
                
                video_processor.create_video(processed_paths, result_path, fps, work_id)
                
                # 세션 데이터 업데이트
                session_data[work_id]['last_activity'] = time.time()
                session_data[work_id]['result_video'] = result_filename
                
            except Exception as e:
                print(f"❌ 비디오 처리 실패: {e}")
                video_processor.progress[work_id] = {
                    'progress': 0,
                    'message': f'비디오 처리 실패: {str(e)}'
                }
        
        # 백그라운드에서 비디오 처리
        processing_thread = threading.Thread(target=process_video_async)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            'message': '비디오 처리 시작',
            'work_id': work_id
        })
        
    except Exception as e:
        print(f"❌ 비디오 처리 시작 실패: {e}")
        return jsonify({'error': f'비디오 처리 시작 실패: {str(e)}'}), 500

@app.route('/progress/<work_id>')
def get_progress(work_id):
    """처리 진행률 조회"""
    # 이미지/벡터화 진행률
    if work_id in progress_data:
        return jsonify(progress_data[work_id])
    
    # 비디오 진행률
    if work_id in video_processor.progress:
        return jsonify(video_processor.progress[work_id])
    
    return jsonify({'progress': 0, 'message': '처리 대기 중...'})

@app.route('/download/<filename>')
def download_file(filename):
    """처리된 파일 다운로드"""
    try:
        file_path = os.path.join(DOWNLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            # 파일 크기 로그
            file_size = os.path.getsize(file_path)
            print(f"📥 다운로드 시작: {filename} (크기: {file_size:,} bytes)")
            
            return send_file(file_path, as_attachment=True)
        else:
            return "파일을 찾을 수 없습니다.", 404
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return f"다운로드 실패: {str(e)}", 500

@app.route('/reset/<work_id>', methods=['POST'])
def reset_session(work_id):
    """세션 초기화"""
    try:
        # 세션 데이터 삭제
        if work_id in session_data:
            del session_data[work_id]
        
        # 진행률 데이터 삭제
        if work_id in progress_data:
            del progress_data[work_id]
        
        if work_id in video_processor.progress:
            del video_processor.progress[work_id]
        
        # 임시 파일 삭제
        session_temp_dir = os.path.join(TEMP_FOLDER, work_id)
        if os.path.exists(session_temp_dir):
            import shutil
            shutil.rmtree(session_temp_dir)
        
        print(f"🧹 세션 데이터 정리 완료: {work_id}")
        
        return jsonify({'message': '세션이 초기화되었습니다.'})
        
    except Exception as e:
        print(f"❌ 세션 초기화 실패: {e}")
        return jsonify({'error': f'세션 초기화 실패: {str(e)}'}), 500

# SSE 엔드포인트들 (Server-Sent Events)
@app.route('/events/<work_id>')
def events(work_id):
    """실시간 이벤트 스트림"""
    def event_stream():
        while True:
            # 이미지/벡터화 진행률
            if work_id in progress_data:
                data = progress_data[work_id]
                yield f"data: {json.dumps(data)}\n\n"
                
                # 완료되면 스트림 종료
                if data.get('progress', 0) >= 100:
                    break
            
            # 비디오 진행률
            elif work_id in video_processor.progress:
                data = video_processor.progress[work_id]
                yield f"data: {json.dumps(data)}\n\n"
                
                # 완료되면 스트림 종료
                if data.get('progress', 0) >= 100:
                    break
            
            time.sleep(0.5)  # 0.5초마다 업데이트
    
    return Response(event_stream(), mimetype="text/plain")

if __name__ == '__main__':
    print("🚀 고품질 AI 이미지 처리 웹 애플리케이션을 시작합니다...")
    print("🔥 배경 제거 + 업스케일링 + 벡터화 통합 솔루션")
    print("📐 새로운 기능: 이미지 벡터화 (SVG 변환)")
    print("📁 지원 형식: bmp, gif, jpeg, jpg, png, webp")
    print("📏 최대 파일 크기: 16MB")
    print("🌐 서버 주소: http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
