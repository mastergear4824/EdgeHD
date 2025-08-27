"""
EdgeHD Backend API Server
AI-powered image and video processing service
"""
from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
import warnings
import os
import uuid
import json
import time
import threading
from werkzeug.utils import secure_filename
from queue import Queue
from PIL import Image

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# 프로젝트 모듈 import
from modules.background_removal import BiRefNetModel
from modules.upscaling import RealESRGANUpscaleModel
from modules.vectorization import ImageVectorizerModel
from modules.video_processing import VideoProcessor
from modules.style_transfer import StyleTransferModel
from modules.utils import (
    generate_filename, generate_unique_id, safe_filename, ensure_directory,
    cleanup_temp_files, format_file_size, get_image_info, validate_image_file,
    log_operation, get_system_info, ProgressTracker
)
from config import *

# Flask 앱 초기화
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# CORS 설정 - 프론트엔드와 통신 허용
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# 앱 설정
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['DOWNLOAD_FOLDER'] = str(DOWNLOAD_FOLDER)
app.config['TEMP_FOLDER'] = str(TEMP_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE

# AI 모델 인스턴스
ai_model = BiRefNetModel()
upscale_model = RealESRGANUpscaleModel()
vectorizer_model = ImageVectorizerModel(ai_model, upscale_model)
video_processor = VideoProcessor()
style_model = StyleTransferModel()

# 프로그래스 상태 관리
progress_queues = {}
session_storage = {}

def load_session_storage():
    """세션 데이터를 파일에서 로드"""
    try:
        if SESSION_FILE.exists():
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                storage = json.load(f)
                
                # 기존 세션들을 새로운 스타일 히스토리 구조로 마이그레이션
                for task_id, session in storage.items():
                    if 'style_results' not in session and 'result_file' in session:
                        result_file = session.get('result_file', '')
                        if result_file.startswith('style_'):
                            # 기존 스타일 변환 결과를 히스토리로 이동
                            session['style_results'] = [{
                                'filename': result_file,
                                'style': session.get('style', 'vangogh'),
                                'strength': session.get('strength', 1.0),
                                'processed_at': session.get('processed_at', time.time())
                            }]
                            print(f"🔄 세션 마이그레이션 완료: {task_id}")
                
                return storage
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

def get_latest_file_path(task_id):
    """task_id에 해당하는 가장 최신 파일 경로 반환 (연속 작업을 위해)"""
    if task_id not in session_storage:
        return None
    
    session = session_storage[task_id]
    
    # 가장 최근 처리 결과 파일이 있으면 그것을 사용
    if 'result_file' in session and session['result_file']:
        result_path = DOWNLOAD_FOLDER / session['result_file']
        if result_path.exists():
            print(f"🔄 연속 작업: 최신 결과 파일 사용 - {session['result_file']}")
            return result_path
    
    # 처리 결과가 없으면 원본 파일 사용
    original_path = Path(session['file_path'])
    if original_path.exists():
        print(f"🔄 연속 작업: 원본 파일 사용 - {original_path.name}")
        return original_path
    
    return None

def update_session_data(work_id, data):
    """세션 데이터 업데이트 및 저장"""
    session_storage[work_id] = data
    save_session_storage(session_storage)
    print(f"💾 세션 데이터 저장됨: {work_id}")

def send_progress(session_id, progress, message):
    """프로그래스 상태 전송"""
    if session_id in progress_queues:
        progress_queues[session_id].put({'progress': progress, 'message': message})

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    """허용된 비디오 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# 초기 세션 데이터 로드
session_storage = load_session_storage()
print(f"📂 기존 세션 {len(session_storage)}개 복원됨")

# ==================== API 엔드포인트 ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'message': 'EdgeHD Backend API is running',
        'version': '2.0.0',
        'timestamp': time.time()
    })

@app.route('/api/system-info', methods=['GET'])
def system_info():
    """시스템 정보 조회"""
    return jsonify(get_system_info())

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """파일 업로드만 담당"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        file = request.files['file']
        task_id = request.form.get('task_id', str(uuid.uuid4()))
        
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 타입 검증
        if not (allowed_file(file.filename) or allowed_video_file(file.filename)):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
            
        # 파일 저장
        filename = secure_filename(f"{task_id}_{file.filename}")
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)
        
        # 비디오 파일인지 확인
        is_video = allowed_video_file(file.filename)
        
        # 세션에 파일 정보 저장
        session_info = {
            'id': task_id,
            'filename': filename,
            'original_name': file.filename,
            'file_path': str(file_path),
            'uploaded_at': time.time(),
            'status': 'uploaded',
            'is_video': is_video
        }
        
        # 비디오 파일인 경우 전체 프레임 추출
        if is_video:
            try:
                print(f"🎬 비디오 파일 감지: {file.filename}")
                print(f"📹 전체 프레임 분해 시작...")
                
                # VideoProcessor 사용하여 전체 프레임 추출
                video_processor = VideoProcessor()
                temp_dir = video_processor.create_temp_dir(task_id)
                
                # 프로그레스 콜백 함수 정의
                def progress_callback(progress, message):
                    print(f"📊 진행률: {progress}% - {message}")
                
                # 프레임 추출
                frame_info = video_processor.extract_frames(str(file_path), temp_dir, progress_callback)
                
                # 프레임 파일들을 다운로드 폴더로 복사하고 목록 저장
                frame_filenames = []
                for i, frame_path in enumerate(frame_info['frame_files']):
                    # 프레임 파일을 다운로드 폴더로 복사
                    frame_filename = f"frame_{task_id}_{i:06d}.png"
                    frame_dest = DOWNLOAD_FOLDER / frame_filename
                    
                    # 파일 복사
                    import shutil
                    shutil.copy2(frame_path, frame_dest)
                    frame_filenames.append(frame_filename)
                
                # 첫 프레임을 미리보기로 설정
                if frame_filenames:
                    session_info['preview_file'] = frame_filenames[0]
                
                # 세션에 프레임 정보 저장
                session_info['frames'] = {
                    'total_frames': frame_info['total_frames'],
                    'fps': frame_info['fps'],
                    'width': frame_info['width'],
                    'height': frame_info['height'],
                    'frame_files': frame_filenames
                }
                
                # 임시 디렉토리 정리
                video_processor.cleanup_temp_dir(task_id)
                
                print(f"✅ 전체 프레임 분해 완료: {len(frame_filenames)}개 프레임")
                
            except Exception as video_error:
                print(f"⚠️ 비디오 프레임 분해 중 오류: {video_error}")
                # 오류가 있어도 업로드는 계속 진행
        
        session_storage[task_id] = session_info
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'filename': filename,
            'message': '파일이 성공적으로 업로드되었습니다.',
            'is_video': is_video,
            'preview_file': session_info.get('preview_file'),
            'frames': session_info.get('frames')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-frames/<task_id>', methods=['GET'])
def get_video_frames(task_id):
    """비디오의 분해된 프레임 목록 조회"""
    try:
        if task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
        
        session = session_storage[task_id]
        if not session.get('is_video'):
            return jsonify({'error': '비디오 파일이 아닙니다.'}), 400
        
        frames_info = session.get('frames', {})
        if not frames_info:
            return jsonify({'error': '프레임 정보가 없습니다.'}), 404
        
        # 프레임 URL 목록 생성
        frame_urls = []
        for frame_file in frames_info.get('frame_files', []):
            frame_urls.append(f'http://localhost:9090/api/download/{frame_file}')
        
        return jsonify({
            'task_id': task_id,
            'total_frames': frames_info.get('total_frames', 0),
            'fps': frames_info.get('fps', 0),
            'width': frames_info.get('width', 0),
            'height': frames_info.get('height', 0),
            'frame_urls': frame_urls
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-frames/<task_id>', methods=['PUT'])
def update_video_frames(task_id):
    """비디오 프레임 편집 (순서 변경, 삭제 등)"""
    try:
        if task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
        
        data = request.get_json()
        operation = data.get('operation')  # 'delete' 또는 'reorder'
        
        session_info = session_storage[task_id]
        frames_info = session_info.get('frames', {})
        
        if operation == 'delete':
            deleted_frames = data.get('deletedFrames', [])
            original_frame_files = frames_info.get('frame_files', [])
            
            # 삭제된 프레임들 제거
            new_frame_files = []
            for i, frame_file in enumerate(original_frame_files):
                if i not in deleted_frames:
                    new_frame_files.append(frame_file)
            
            # 삭제된 프레임 파일들을 실제로 삭제
            for index in deleted_frames:
                if index < len(original_frame_files):
                    frame_file = original_frame_files[index]
                    frame_path = DOWNLOAD_FOLDER / frame_file
                    if frame_path.exists():
                        frame_path.unlink()
                        print(f"프레임 파일 삭제: {frame_file}")
            
            # 세션 정보 업데이트
            frames_info['frame_files'] = new_frame_files
            frames_info['total_frames'] = len(new_frame_files)
            
        elif operation == 'reorder':
            from_index = data.get('fromIndex')
            to_index = data.get('toIndex')
            original_frame_files = frames_info.get('frame_files', [])
            
            # 프레임 순서 변경
            new_frame_files = original_frame_files.copy()
            moved_frame = new_frame_files.pop(from_index)
            new_frame_files.insert(to_index, moved_frame)
            
            # 세션 정보 업데이트
            frames_info['frame_files'] = new_frame_files
            print(f"프레임 순서 변경: {from_index} -> {to_index}")
        
        session_storage[task_id]['frames'] = frames_info
        save_session_storage(session_storage)
        
        return jsonify({
            'success': True,
            'message': f'프레임 {operation} 완료',
            'total_frames': len(frames_info.get('frame_files', [])),
            'operation': operation
        })
        
    except Exception as e:
        print(f"프레임 편집 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    """배경 제거 처리"""
    try:
        task_id = request.form.get('task_id')
        if not task_id or task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
            
        # 최신 파일 경로 가져오기 (연속 작업 지원)
        file_path = get_latest_file_path(task_id)
        if not file_path:
            return jsonify({'error': '처리할 파일을 찾을 수 없습니다.'}), 404
        
        # 이미지 로드
        image = Image.open(file_path).convert('RGB')
        
        # 배경 제거
        result_image = ai_model.remove_background(image)
        
        # 결과 저장
        result_filename = f"removed_{task_id}.png"
        result_path = DOWNLOAD_FOLDER / result_filename
        
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        result_image.save(result_path, 'PNG', optimize=True)
        
        # 세션 업데이트
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time()
        })
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': '배경 제거가 완료되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 비디오 처리 API 추가
@app.route('/api/process-video', methods=['POST'])
def process_video():
    """비디오 배경 제거 처리"""
    try:
        task_id = request.form.get('task_id')
        if not task_id or task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
            
        session = session_storage[task_id]
        file_path = Path(session['file_path'])
        
        # 비디오 처리
        result_filename = f"video_processed_{task_id}.mp4"
        result_path = DOWNLOAD_FOLDER / result_filename
        
        # 실제 비디오 처리는 video_processor 사용
        video_processor.process_video(str(file_path), str(result_path))
        
        # 세션 업데이트
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time()
        })
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': '비디오 처리가 완료되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract-last-frame', methods=['POST'])
def extract_last_frame():
    """비디오 마지막 프레임 추출"""
    try:
        task_id = request.form.get('task_id')
        if not task_id or task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
            
        session = session_storage[task_id]
        file_path = Path(session['file_path'])
        
        # 마지막 프레임 추출
        result_filename = f"last_frame_{task_id}.png"
        result_path = DOWNLOAD_FOLDER / result_filename
        
        # 실제 프레임 추출은 video_processor 사용
        video_processor.extract_last_frame(str(file_path), str(result_path))
        
        # 세션 업데이트
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time()
        })
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': '프레임 추출이 완료되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upscale', methods=['POST'])
def upscale_image():
    """이미지 업스케일링 처리"""
    try:
        task_id = request.form.get('task_id')
        scale = int(request.form.get('scale', '4'))
        
        if not task_id or task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
            
        if scale not in [2, 4]:
            return jsonify({'error': '지원하지 않는 업스케일 배율입니다.'}), 400
            
        # 최신 파일 경로 가져오기 (연속 작업 지원)
        file_path = get_latest_file_path(task_id)
        if not file_path:
            return jsonify({'error': '처리할 파일을 찾을 수 없습니다.'}), 404
        
        # 이미지 로드
        image = Image.open(file_path)
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        # 업스케일링 처리
        result_image = upscale_model.upscale_image(image, scale)
        
        # 결과 저장
        result_filename = f"upscaled_{scale}x_{task_id}.png"
        result_path = DOWNLOAD_FOLDER / result_filename
        
        result_image.save(result_path, 'PNG', optimize=True)
        
        # 세션 업데이트
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time(),
            'scale': scale
        })
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': f'{scale}x 업스케일링이 완료되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        


@app.route('/api/vectorize', methods=['POST'])
def vectorize_image():
    """이미지 벡터화 처리"""
    try:
        task_id = request.form.get('task_id')
        if not task_id or task_id not in session_storage:
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400
            
        # 최신 파일 경로 가져오기 (연속 작업 지원)
        file_path = get_latest_file_path(task_id)
        if not file_path:
            return jsonify({'error': '처리할 파일을 찾을 수 없습니다.'}), 404
        
        n_colors = int(request.form.get('n_colors', '8'))
        if n_colors < 2 or n_colors > 32:
            n_colors = 8
        
        # 이미지 로드
        image = Image.open(file_path).convert('RGB')
        
        # 벡터화 처리
        svg_content = vectorizer_model.vectorize_image(image, n_colors=n_colors)
        
        # 결과 저장
        result_filename = f"vectorized_{task_id}.svg"
        result_path = DOWNLOAD_FOLDER / result_filename
        
        # SVG 내용을 파일로 저장
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        # 세션 업데이트
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time(),
            'n_colors': n_colors
        })
        save_session_storage(session_storage)
        
        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': '벡터화가 완료되었습니다.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 스타일 변환 API
@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    """이미지 스타일 변환 처리"""
    print("🚀 DEBUG: Entering style_transfer API endpoint.")
    try:
        task_id = request.form.get('task_id')
        print(f"📝 DEBUG: Received task_id: {task_id}")
        
        if not task_id or task_id not in session_storage:
            print(f"❌ DEBUG: Invalid task_id: {task_id}")
            return jsonify({'error': '유효하지 않은 task_id입니다.'}), 400

        # 스타일 변환은 항상 원본 파일 사용 (테스트용)
        session = session_storage[task_id]
        file_path = Path(session['file_path'])
        print(f"📁 DEBUG: Using original file path: {file_path}")
        
        if not file_path:
            print(f"❌ DEBUG: 처리할 파일을 찾을 수 없습니다: {task_id}")
            return jsonify({'error': '처리할 파일을 찾을 수 없습니다.'}), 404

        style = request.form.get('style', 'vangogh')
        strength = float(request.form.get('strength', '1.0'))
        print(f"🎨 DEBUG: Style={style}, Strength={strength}")

        # 유효성 검사
        available_styles = ['vangogh', 'oil_painting']
        if style not in available_styles:
            style = 'vangogh'
        if strength < 0.0 or strength > 1.0:
            strength = 1.0

        # 이미지 로드
        print(f"📁 DEBUG: 파일 로드 시도: {file_path}")
        try:
            image = Image.open(file_path)
            print(f"📸 DEBUG: 원본 이미지 로드 성공: 크기={image.size}, 모드={image.mode}")
            image = image.convert('RGB')
            print(f"📸 DEBUG: RGB 변환 후 이미지: 크기={image.size}, 모드={image.mode}")
            
            # 이미지가 검정색인지 확인
            if image.getbbox() is None:
                print("⚠️ DEBUG: 로드된 이미지가 완전히 검정색입니다 (getbbox is None).")
            else:
                print("✅ DEBUG: 로드된 이미지는 검정색이 아닙니다 (getbbox is not None).")

        except Exception as img_e:
            print(f"❌ DEBUG: 이미지 로드 또는 변환 실패: {img_e}")
            import traceback
            print(f"❌ DEBUG: 상세 오류 (이미지 로드): {traceback.format_exc()}")
            return jsonify({'error': f'이미지 로드 또는 변환 실패: {img_e}'}), 500

        # 스타일 변환 처리
        print(f"🎨 DEBUG: 스타일 변환 시작: {style}, 강도={strength}")
        result_image = style_model.transfer_style(image, style=style, strength=strength)
        print(f"📸 DEBUG: 변환 결과: 크기={result_image.size}, 모드={result_image.mode}")
        
        # 결과 이미지가 검정색인지 확인
        if result_image.getbbox() is None:
            print("⚠️ DEBUG: 변환 결과 이미지가 완전히 검정색입니다 (getbbox is None).")
        else:
            print("✅ DEBUG: 변환 결과 이미지는 검정색이 아닙니다 (getbbox is not None).")

        # 결과 저장 (캐시 우회를 위해 타임스탬프 추가)
        import time
        timestamp = int(time.time() * 1000)  # 밀리초 타임스탬프
        result_filename = f"style_{style}_{task_id}_{timestamp}.png"
        result_path = DOWNLOAD_FOLDER / result_filename
        print(f"💾 DEBUG: 파일 저장 경로 (타임스탬프 포함): {result_path}")

        # PNG로 저장
        result_image.save(result_path, 'PNG', optimize=True)
        print(f"✅ DEBUG: 파일 저장 완료: {result_filename}")

        # 세션 업데이트 (스타일 변환 히스토리 관리)
        if 'style_results' not in session_storage[task_id]:
            session_storage[task_id]['style_results'] = []
        
        # 새로운 스타일 변환 결과 추가
        style_result = {
            'filename': result_filename,
            'style': style,
            'strength': strength,
            'processed_at': time.time()
        }
        session_storage[task_id]['style_results'].append(style_result)
        
        # 최신 결과를 result_file에도 저장 (호환성 유지)
        session_storage[task_id].update({
            'status': 'completed',
            'result_file': result_filename,
            'processed_at': time.time(),
            'style': style,
            'strength': strength
        })
        save_session_storage(session_storage)
        print(f"✅ DEBUG: 세션 업데이트 완료")

        return jsonify({
            'task_id': task_id,
            'download_url': f'/api/download/{result_filename}',
            'preview_url': f'http://localhost:9090/api/download/{result_filename}',
            'message': f'{style} 스타일 변환이 완료되었습니다.'
        })

    except Exception as e:
        print(f"❌ DEBUG: 스타일 변환 API 오류: {e}")
        import traceback
        print(f"❌ DEBUG: 상세 오류: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<session_id>')
def progress(session_id):
    """Server-Sent Events로 프로그래스 전송"""
    def generate():
        if session_id not in progress_queues:
            progress_queues[session_id] = Queue()
        
        queue = progress_queues[session_id]
        
        try:
            while True:
                try:
                    data = queue.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if data['progress'] >= 100:
                        break
                except:
                    break
        finally:
            if session_id in progress_queues:
                del progress_queues[session_id]
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/download/<filename>')
def download_file(filename):
    """업로드된 파일 또는 처리된 파일 다운로드/미리보기"""
    try:
        # 먼저 처리된 파일(DOWNLOAD_FOLDER)에서 찾기
        file_path = DOWNLOAD_FOLDER / filename
        
        # 처리된 파일이 없으면 업로드된 파일(UPLOAD_FOLDER)에서 찾기
        if not file_path.exists():
            file_path = UPLOAD_FOLDER / filename
        
        if not file_path.exists():
            print(f"❌ 파일을 찾을 수 없음: {filename}")
            print(f"   확인한 경로: {DOWNLOAD_FOLDER / filename}, {UPLOAD_FOLDER / filename}")
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        file_size = file_path.stat().st_size
        print(f"📥 파일 전송: {filename} (크기: {file_size:,} bytes)")
        
        # URL 파라미터로 다운로드/미리보기 구분
        is_download = request.args.get('download', 'false').lower() == 'true'
        
        return send_file(
            file_path,
            as_attachment=is_download,  # download=true면 강제 다운로드
            download_name=filename
        )
    except Exception as e:
        print(f"❌ 파일 전송 실패: {e}")
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """세션 목록 조회"""
    return jsonify({
        'sessions': session_storage,
        'total': len(session_storage)
    })

@app.route('/api/files', methods=['GET'])
def get_files():
    """업로드된 파일 목록과 처리 결과 조회"""
    try:
        files_list = []
        
        for task_id, session in session_storage.items():
            # 비디오 파일인 경우 미리보기 URL 설정
            preview_url = f'http://localhost:9090/api/download/{session.get("filename", "")}'
            if session.get('is_video') and session.get('preview_file'):
                preview_url = f'http://localhost:9090/api/download/{session.get("preview_file")}'
            
            file_info = {
                'id': task_id,
                'fileName': session.get('filename', ''),
                'originalName': session.get('original_name', ''),
                'fileType': '',  # 파일 확장자로 추정
                'fileSize': 0,   # 실제 파일에서 크기 확인
                'uploadedAt': session.get('uploaded_at', time.time()),
                'status': session.get('status', 'uploaded'),
                'previewUrl': preview_url,
                'processingResults': []
            }
            
            # 파일 타입과 크기 확인
            if 'file_path' in session:
                file_path = Path(session['file_path'])
                if file_path.exists():
                    file_info['fileSize'] = file_path.stat().st_size
                    
                    # 파일 확장자로 타입 추정
                    ext = file_path.suffix.lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                        file_info['fileType'] = f'image/{ext[1:]}'
                    elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
                        file_info['fileType'] = f'video/{ext[1:]}'
            
            # 일반 처리 결과 확인
            if session.get('status') == 'completed' and 'result_file' in session:
                result_filename = session['result_file']
                action_type = ''
                
                # 결과 파일명으로 처리 유형 판단 (스타일 변환 제외)
                if result_filename.startswith('removed_'):
                    action_type = 'remove_bg'
                elif result_filename.startswith('upscaled_'):
                    if '2x' in result_filename:
                        action_type = 'upscale_2x'
                    elif '4x' in result_filename:
                        action_type = 'upscale_4x'
                    else:
                        action_type = 'upscale'
                elif result_filename.startswith('vectorized_'):
                    action_type = 'vectorize'
                elif result_filename.startswith('processed_'):
                    action_type = 'process_video'
                elif result_filename.startswith('last_frame_'):
                    action_type = 'extract_last_frame'
                
                if action_type:
                    processing_result = {
                        'id': f"{task_id}_{action_type}",
                        'actionId': action_type,
                        'actionLabel': get_action_label(action_type),
                        'status': 'completed',
                        'resultUrl': f'/api/download/{result_filename}',
                        'resultPreviewUrl': f'/api/download/{result_filename}',
                        'completedAt': session.get('processed_at', time.time())
                    }
                    file_info['processingResults'].append(processing_result)
            
            # 스타일 변환 히스토리 추가
            if 'style_results' in session:
                for idx, style_result in enumerate(session['style_results']):
                    style_filename = style_result['filename']
                    style = style_result['style']
                    
                    # 스타일에 따른 action_type 결정
                    if style == 'vangogh':
                        action_type = 'style_retro'
                    elif style == 'oil_painting':
                        action_type = 'style_painting'
                    else:
                        action_type = 'style_transfer'
                    
                    processing_result = {
                        'id': f"{task_id}_{action_type}_{idx}",
                        'actionId': action_type,
                        'actionLabel': get_action_label(action_type),
                        'status': 'completed',
                        'resultUrl': f'/api/download/{style_filename}',
                        'resultPreviewUrl': f'/api/download/{style_filename}',
                        'completedAt': style_result['processed_at']
                    }
                    file_info['processingResults'].append(processing_result)
            
            files_list.append(file_info)
        
        # 업로드 시간 순으로 정렬 (최신순)
        files_list.sort(key=lambda x: x['uploadedAt'], reverse=True)
        
        return jsonify({
            'files': files_list,
            'total': len(files_list)
        })
        
    except Exception as e:
        print(f"❌ 파일 목록 조회 실패: {e}")
        return jsonify({'error': str(e)}), 500

def get_action_label(action_id):
    """액션 ID를 한글 라벨로 변환"""
    action_labels = {
        'remove_bg': '배경 제거',
        'upscale_2x': '업스케일링 x2',
        'upscale_4x': '업스케일링 x4',
        'upscale': '업스케일링',
        'vectorize': '벡터화',
        'style_retro': '레트로',
        'style_painting': '페인팅',
        'style_transfer': '스타일 변환',
        'process_video': '비디오 배경 제거',
        'extract_last_frame': '마지막 프레임 추출'
    }
    return action_labels.get(action_id, action_id)

@app.route('/api/sessions/<work_id>', methods=['GET'])
def get_session(work_id):
    """특정 세션 조회"""
    if work_id in session_storage:
        return jsonify(session_storage[work_id])
    return jsonify({'error': '세션을 찾을 수 없습니다.'}), 404

@app.route('/api/sessions/<work_id>', methods=['DELETE'])
def delete_session(work_id):
    """세션 삭제"""
    try:
        if work_id in session_storage:
            del session_storage[work_id]
            save_session_storage(session_storage)
        
        if work_id in progress_queues:
            del progress_queues[work_id]
        
            video_processor.cleanup_temp_dir(work_id)
        
        return jsonify({'message': '세션이 삭제되었습니다.'})
    except Exception as e:
        return jsonify({'error': f'세션 삭제 실패: {str(e)}'}), 500

@app.route('/api/delete-file/<task_id>', methods=['DELETE'])
def delete_file_complete(task_id):
    """파일과 관련 데이터 완전 삭제"""
    try:
        if task_id not in session_storage:
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        session = session_storage[task_id]
        
        # 원본 파일 삭제
        if 'file_path' in session:
            original_file = Path(session['file_path'])
            if original_file.exists():
                original_file.unlink()
                print(f"🗑️ 원본 파일 삭제: {original_file}")
        
        # 처리된 결과 파일들 삭제
        download_files = list(DOWNLOAD_FOLDER.glob(f"*{task_id}*"))
        for file_path in download_files:
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️ 처리 결과 파일 삭제: {file_path}")
        
        # 세션 데이터 삭제
        del session_storage[task_id]
        save_session_storage(session_storage)
        
        # 진행 중인 작업 큐에서도 삭제
        if task_id in progress_queues:
            del progress_queues[task_id]
        
        # 임시 디렉토리 정리
        video_processor.cleanup_temp_dir(task_id)
        
        print(f"✅ 파일 완전 삭제 완료: {task_id}")
        return jsonify({'message': '파일이 성공적으로 삭제되었습니다.'})
        
    except Exception as e:
        print(f"❌ 파일 삭제 실패: {e}")
        return jsonify({'error': str(e)}), 500

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API 엔드포인트를 찾을 수 없습니다.'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    print("🚀 EdgeHD Backend API Server 시작...")
    print("🔥 AI 이미지/비디오 처리 API 서비스")
    print(f"🌐 서버 주소: http://{HOST}:{PORT}")
    print(f"📁 지원 형식: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    print(f"📏 최대 파일 크기: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # 모델 미리 로드
    try:
        print("🔄 AI 모델 미리 로드 중...")
        ai_model.load_model()
        vectorizer_model.load_model()
        print("✅ AI 모델 준비 완료!")
    except Exception as e:
        print(f"⚠️ 모델 미리 로드 실패: {e}")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)