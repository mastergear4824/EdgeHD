from flask import Flask, request, render_template, send_file, jsonify, url_for, Response
from flask_cors import CORS

# 경고 메시지 필터링
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

from PIL import Image
import os
import uuid
import io
import json
import numpy as np
import cv2
import time
import threading
from werkzeug.utils import secure_filename
from queue import Queue

# 프로젝트 모듈 import
from modules.background_removal import BiRefNetModel
from modules.upscaling import RealESRGANUpscaleModel
from modules.vectorization import ImageVectorizerModel
from modules.video_processing import VideoProcessor
from modules.utils import (
    generate_filename, generate_unique_id, safe_filename, ensure_directory,
    cleanup_temp_files, format_file_size, get_image_info, validate_image_file,
    log_operation, get_system_info, ProgressTracker
)

# 프로젝트 내 모델 디렉토리 설정
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face 캐시를 프로젝트 내로 설정
os.environ['HF_HOME'] = MODEL_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = MODEL_DIR

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
    """파일 업로드만 처리 (AI 처리는 별도 요청 시에만)"""
    try:
        print("📥 업로드 요청 수신됨")
        print(f"📂 request.files keys: {list(request.files.keys())}")
        print(f"📂 request.form keys: {list(request.form.keys())}")
        
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        print(f"📁 파일 업로드 중: {file.filename}")
        
        # 세션 디렉토리 생성
        session_dir = os.path.join('temp', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 원본 파일 저장
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"upload_{int(time.time())}.jpg"
        
        original_path = os.path.join(session_dir, filename)
        file.save(original_path)
        
        print(f"✅ 파일 저장 완료: {original_path}")
        
        # 세션 데이터 저장 (단순 업로드 상태)
        update_session_data(session_id, {
            'type': 'image',
            'original_filename': file.filename,
            'original_path': original_path,
            'uploaded': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'message': '이미지가 업로드되었습니다!',
            'session_id': session_id,
            'work_id': session_id,
            'filename': filename,
            'original_filename': file.filename
        })
        
    except Exception as e:
        print(f"❌ 업로드 처리 실패: {e}")
        return jsonify({'error': f'업로드 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/remove_background', methods=['POST'])
def remove_background():
    """이미지 배경 제거 처리"""
    try:
        # 세션 ID: 프론트엔드에서 전달된 work_id 사용, 없으면 새로 생성
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        
        # 프로그래스 큐 초기화
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)  # UI 업데이트를 위한 약간의 지연
        
        progress_callback(5, "📋 파일 검증 중...")
        
        # 파일 확인 - 새 업로드 또는 기존 세션 파일
        if 'file' in request.files:
            # 새 파일 업로드된 경우
            file = request.files['file']
            if file.filename == '':
                progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
                return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
            if not allowed_file(file.filename):
                progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
                return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
            
            progress_callback(10, "📁 이미지 로드 중...")
            image = Image.open(file.stream).convert('RGB')
            original_filename = file.filename
            
        elif session_id in session_storage and session_storage[session_id].get('uploaded'):
            # 기존 세션에서 업로드된 파일 사용
            progress_callback(10, "📁 기존 업로드 파일 로드 중...")
            session_data = session_storage[session_id]
            original_path = session_data.get('original_path')
            
            if not original_path or not os.path.exists(original_path):
                progress_callback(0, "❌ 기존 업로드 파일을 찾을 수 없습니다.")
                return jsonify({'error': '기존 업로드 파일을 찾을 수 없습니다.'}), 400
            
            image = Image.open(original_path).convert('RGB')
            original_filename = session_data.get('original_filename', 'unknown.jpg')
            
        else:
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        progress_callback(15, "🚀 고품질 자동 배경 제거 시작...")
        
        # 고품질 AI 모델로 전체 이미지 배경 제거
        result_image = ai_model.remove_background(image, progress_callback)
        
        # 경량 후처리
        print("🔧 경량 후처리: 품질 안정화 중...")
        result_image = light_improve_mask_quality(result_image)
        
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
            'original_filename': original_filename,
            'download_url': url_for('download_file', filename=filename),
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': url_for('download_file', filename=filename),
            'session_id': session_id,
            'work_id': session_id
        })
        
    except Exception as e:
        print(f"❌ 배경 제거 처리 실패: {e}")
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
                progress_callback,
                ai_model,  # AI 모델 전달
                upscale_model  # 업스케일 모델 전달
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

@app.route('/temp/<session_id>/<filename>')
def serve_temp_file(session_id, filename):
    """임시 파일 서빙 (업로드된 원본 이미지 등)"""
    try:
        file_path = os.path.join('temp', session_id, filename)
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            print(f"❌ 임시 파일을 찾을 수 없음: {file_path}")
            return jsonify({'error': '파일을 찾을 수 없습니다'}), 404
        
        return send_file(file_path)
        
    except Exception as e:
        print(f"❌ 임시 파일 서빙 실패: {e}")
        return jsonify({'error': f'파일 서빙 실패: {str(e)}'}), 500

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
        
        print(f"📍 벡터화 요청 - session_id: {session_id}")
        print(f"📂 request.files keys: {list(request.files.keys())}")
        print(f"📂 request.form keys: {list(request.form.keys())}")
        
        # 세션 상태 디버깅
        print(f"🔍 session_id in session_storage: {session_id in session_storage}")
        if session_id in session_storage:
            session_data = session_storage[session_id]
            print(f"🔍 session_data: {session_data}")
            print(f"🔍 session_data.get('uploaded'): {session_data.get('uploaded')}")
            print(f"🔍 session_data.get('completed'): {session_data.get('completed')}")
            print(f"🔍 조건 만족 (uploaded 또는 completed): {session_id in session_storage and (session_storage[session_id].get('uploaded') or session_storage[session_id].get('completed'))}")
        else:
            print(f"❌ 세션을 찾을 수 없음: {session_id}")
        
        # 파일 확인 - 새 업로드 또는 기존 세션 파일
        if 'file' in request.files:
            # 새 파일 업로드된 경우
            file = request.files['file']
            if file.filename == '':
                progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
                return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
            if not allowed_file(file.filename):
                progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
                return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
            
            # 새 파일에서 이미지 로드
            image = Image.open(file.stream).convert('RGB')
            filename = file.filename
            print(f"🆕 새 파일 업로드: {filename}")
            
        elif session_id in session_storage and (session_storage[session_id].get('uploaded') or session_storage[session_id].get('completed')):
            # 기존 세션에서 업로드된 파일 또는 처리 완료된 파일 사용
            session_data = session_storage[session_id]
            if session_data.get('uploaded'):
                progress_callback(10, "📁 기존 업로드 파일 로드 중...")
                # 원본 파일 경로 사용
                original_path = session_data.get('original_path')
            else:
                progress_callback(10, "📁 처리 완료된 파일 로드 중...")
                # 벡터화 완료된 경우에는 원본 파일 경로 사용, 다른 처리는 결과 파일 사용
                if session_data.get('type') == 'vectorize':
                    # 벡터화 완료 시: 원본 이미지 경로 사용 (temp 폴더)
                    original_path = os.path.join('temp', session_id, session_data.get('original_filename'))
                else:
                    # 배경제거/업스케일 완료 시: 처리된 파일 경로 사용 (downloads 폴더)
                    original_path = os.path.join('downloads', session_data.get('filename'))
            
            print(f"🎯 사용할 파일 경로: {original_path}")
            
            if not original_path or not os.path.exists(original_path):
                progress_callback(0, f"❌ 파일을 찾을 수 없습니다: {original_path}")
                return jsonify({
                    'error': '파일을 찾을 수 없습니다. 먼저 이미지를 업로드해주세요.',
                    'file_path': original_path,
                    'session_exists': session_id in session_storage
                }), 404
            
            # 기존 파일에서 이미지 로드
            image = Image.open(original_path).convert('RGB')
            filename = session_data.get('original_filename', 'unknown.jpg')
            print(f"📂 기존 세션 파일 사용: {filename} (경로: {original_path})")
            
        else:
            progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        # 벡터화 옵션 확인
        n_colors = request.form.get('n_colors', '8')
        output_format = request.form.get('output_format', 'svg')
        vectorize_mode = request.form.get('vectorize_mode', 'color')  # 'color' 또는 'bw'
        
        try:
            n_colors = int(n_colors)
            if n_colors < 2 or n_colors > 32:
                n_colors = 8  # 기본값
        except ValueError:
            n_colors = 8
        
        if output_format not in ['svg']:  # 현재는 SVG만 지원
            output_format = 'svg'
            
        if vectorize_mode not in ['color', 'bw']:
            vectorize_mode = 'color'  # 기본값
        
        mode_name = "컬러" if vectorize_mode == 'color' else "흑백"
        progress_callback(10, f"📐 {mode_name} 벡터화 시작... ({n_colors}색상)")
        print(f"📍 세션 ID: {session_id} (프론트엔드에서 전달: {'work_id' in request.form})")
        print(f"🎨 벡터화 모드: {mode_name} ({vectorize_mode})")
        
        # 이미지는 이미 위에서 로드됨
        
        progress_callback(15, "🧠 AI 벡터화 모델 로드 중...")
        
        # 벡터화 처리
        try:
            svg_content = vectorizer_model.vectorize_image(
                image, 
                output_format=output_format, 
                n_colors=n_colors, 
                vectorize_mode=vectorize_mode,  # 사용자 선택 모드 전달
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
            'original_filename': filename,
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