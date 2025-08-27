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

# 프로그래스 상태 관리
progress_queues = {}
session_storage = {}

def load_session_storage():
    """세션 데이터를 파일에서 로드"""
    try:
        if SESSION_FILE.exists():
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
    """파일 업로드 및 배경 제거 처리"""
    try:
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)
        
        progress_callback(5, "📋 파일 검증 중...")
        
        if 'file' not in request.files:
                progress_callback(0, "❌ 파일이 선택되지 않았습니다.")
                return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
                progress_callback(0, "❌ 지원하지 않는 파일 형식입니다.")
                return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
            
            progress_callback(10, "📁 이미지 로드 중...")
        
        image = Image.open(file.stream).convert('RGB')
        mode = request.form.get('mode', 'auto')
        
        print(f"🔍 처리 모드: {mode}")
        print(f"📍 세션 ID: {session_id}")
        
        if mode == 'auto':
            progress_callback(15, "🚀 고품질 자동 배경 제거 시작...")
            result_image = ai_model.remove_background(image, progress_callback)
        else:
            progress_callback(0, "❌ 잘못된 처리 모드입니다.")
            return jsonify({'error': '잘못된 처리 모드입니다.'}), 400
        
        progress_callback(99, "💾 결과 저장 중...")
        
        filename = secure_filename(f"removed_{uuid.uuid4().hex}.png")
        filepath = DOWNLOAD_FOLDER / filename
        
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        result_image.save(filepath, 'PNG', optimize=True, compress_level=6)
        progress_callback(100, "🎉 고품질 AI 배경 제거 완료!")
        
        update_session_data(session_id, {
            'type': 'image',
            'filename': filename,
            'original_filename': file.filename,
            'download_url': f'/api/download/{filename}',
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': f'/api/download/{filename}',
            'session_id': session_id,
            'work_id': session_id
        })
        
    except Exception as e:
        print(f"❌ 업로드 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/upscale', methods=['POST'])
def upscale_image():
    """이미지 업스케일링 처리"""
    try:
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)
        
        progress_callback(5, "📋 파일 검증 중...")
        
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        scale = int(request.form.get('scale', '4'))
            if scale not in [2, 4]:
            return jsonify({'error': '지원하지 않는 업스케일 배율입니다.'}), 400
        
        progress_callback(10, f"📁 {scale}x 업스케일링 시작...")
        
        image = Image.open(file.stream)
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
        
            result_image = upscale_model.upscale_image(image, scale, progress_callback)
        
        progress_callback(95, "💾 결과 이미지 저장 중...")
        
        filename = f"upscaled_{scale}x_{uuid.uuid4().hex[:8]}.png"
        filepath = DOWNLOAD_FOLDER / filename
        result_image.save(filepath, 'PNG', optimize=True, compress_level=6)
        
        progress_callback(100, f"🎉 {scale}x 업스케일링 완료!")
        
        update_session_data(session_id, {
            'type': 'upscale',
            'filename': filename,
            'original_filename': file.filename,
            'download_url': f'/api/download/{filename}',
            'scale': scale,
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': f'/api/download/{filename}',
            'session_id': session_id,
            'work_id': session_id,
            'scale': scale
        })
        
    except Exception as e:
        print(f"❌ 업스케일링 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/api/vectorize', methods=['POST'])
def vectorize_image():
    """이미지 벡터화 처리"""
    try:
        session_id = request.form.get('work_id', str(uuid.uuid4()))
        progress_queues[session_id] = Queue()
        
        def progress_callback(progress, message):
            send_progress(session_id, progress, message)
            time.sleep(0.1)
        
        progress_callback(5, "📋 파일 검증 중...")
        
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
        n_colors = int(request.form.get('n_colors', '8'))
            if n_colors < 2 or n_colors > 32:
            n_colors = 8
        
        progress_callback(10, f"📐 {n_colors}색상 벡터화 시작...")
        
        image = Image.open(file.stream).convert('RGB')
            svg_content = vectorizer_model.vectorize_image(
                image, 
            output_format='svg', 
                n_colors=n_colors, 
                progress_callback=progress_callback
            )
        
        progress_callback(98, "💾 SVG 파일 저장 중...")
        
        filename = f"vectorized_{uuid.uuid4().hex[:8]}.svg"
        filepath = DOWNLOAD_FOLDER / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        progress_callback(100, "🎉 이미지 벡터화 완료!")
        
        update_session_data(session_id, {
            'type': 'vectorize',
            'filename': filename,
            'original_filename': file.filename,
            'download_url': f'/api/download/{filename}',
            'n_colors': n_colors,
            'file_size': len(svg_content),
            'completed': True,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'download_url': f'/api/download/{filename}',
            'session_id': session_id,
            'work_id': session_id,
            'n_colors': n_colors,
            'file_size': len(svg_content)
        })
        
    except Exception as e:
        print(f"❌ 벡터화 처리 실패: {e}")
        if session_id in progress_queues:
            send_progress(session_id, 0, f"❌ 오류: {str(e)}")
        return jsonify({'error': f'처리 중 오류가 발생했습니다: {str(e)}'}), 500

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
    """처리된 파일 다운로드"""
    try:
        file_path = DOWNLOAD_FOLDER / filename
        
        if not file_path.exists():
            return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        
        file_size = file_path.stat().st_size
        print(f"📥 다운로드 시작: {filename} (크기: {file_size:,} bytes)")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """세션 목록 조회"""
    return jsonify({
        'sessions': session_storage,
        'total': len(session_storage)
    })

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