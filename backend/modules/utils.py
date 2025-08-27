"""
공통 유틸리티 함수들
"""

import os
import uuid
import shutil
from datetime import datetime


def generate_filename(prefix, extension):
    """고유한 파일명 생성"""
    timestamp = int(datetime.now().timestamp())
    return f"{prefix}_{timestamp}.{extension}"

def generate_unique_id():
    """고유 ID 생성"""
    return str(uuid.uuid4())

def safe_filename(filename):
    """파일명을 안전하게 정리"""
    import re
    # 안전하지 않은 문자 제거
    safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
    return safe_name

def ensure_directory(directory):
    """디렉토리가 없으면 생성"""
    os.makedirs(directory, exist_ok=True)

def cleanup_temp_files(temp_dir, max_age_hours=24):
    """오래된 임시 파일들 정리"""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path):
                        file_age = current_time - os.path.getmtime(item_path)
                        if file_age > max_age_seconds:
                            os.remove(item_path)
                            print(f"🧹 정리됨: {item}")
                    elif os.path.isdir(item_path):
                        dir_age = current_time - os.path.getmtime(item_path)
                        if dir_age > max_age_seconds:
                            shutil.rmtree(item_path)
                            print(f"🧹 디렉토리 정리됨: {item}")
                except Exception as e:
                    print(f"⚠️ {item} 정리 실패: {e}")
                    
    except Exception as e:
        print(f"❌ 임시 파일 정리 실패: {e}")

def format_file_size(size_bytes):
    """파일 크기를 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_image_info(image):
    """이미지 정보 추출"""
    try:
        width, height = image.size
        mode = image.mode
        format_name = image.format if hasattr(image, 'format') else 'Unknown'
        
        return {
            'width': width,
            'height': height,
            'mode': mode,
            'format': format_name,
            'size_info': f"{width}x{height}",
            'total_pixels': width * height
        }
    except Exception as e:
        print(f"⚠️ 이미지 정보 추출 실패: {e}")
        return None

def validate_image_file(file):
    """업로드된 이미지 파일 유효성 검사"""
    allowed_extensions = {'bmp', 'gif', 'jpeg', 'jpg', 'png', 'webp'}
    max_file_size = 16 * 1024 * 1024  # 16MB
    
    if not file or not file.filename:
        return False, "파일이 선택되지 않았습니다."
    
    # 확장자 검사
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(f'.{ext}') for ext in allowed_extensions):
        return False, f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}"
    
    # 파일 크기 검사 (가능한 경우)
    try:
        file.seek(0, 2)  # 파일 끝으로 이동
        file_size = file.tell()
        file.seek(0)  # 다시 처음으로
        
        if file_size > max_file_size:
            return False, f"파일 크기가 너무 큽니다. 최대 {format_file_size(max_file_size)} 까지 지원됩니다."
    except:
        pass  # 파일 크기 검사 실패시 무시
    
    return True, "유효한 파일입니다."

def log_operation(operation, details=None):
    """작업 로그 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {operation}"
    if details:
        log_message += f" - {details}"
    print(log_message)

def get_system_info():
    """시스템 정보 반환"""
    import platform
    import psutil
    
    try:
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': format_file_size(psutil.virtual_memory().total),
            'memory_available': format_file_size(psutil.virtual_memory().available)
        }
        return info
    except Exception as e:
        print(f"⚠️ 시스템 정보 수집 실패: {e}")
        return {}

class ProgressTracker:
    """진행률 추적 클래스"""
    
    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        
    def update(self, step, message=""):
        """진행률 업데이트"""
        self.current_step = step
        percentage = (step / self.total_steps) * 100
        elapsed_time = datetime.now() - self.start_time
        
        print(f"📊 진행률: {percentage:.1f}% ({step}/{self.total_steps}) - {message}")
        
        if step >= self.total_steps:
            print(f"✅ 완료! 소요시간: {elapsed_time}")
    
    def get_progress(self):
        """현재 진행률 반환"""
        return (self.current_step / self.total_steps) * 100
