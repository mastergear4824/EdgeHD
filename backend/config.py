"""
Backend Configuration
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8080))
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# CORS Configuration
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    'http://localhost:3000',
    'http://127.0.0.1:3000'
]

# File Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
MAX_VIDEO_SIZE = int(os.getenv('MAX_VIDEO_SIZE', 100 * 1024 * 1024))  # 100MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_VECTOR_EXTENSIONS = {'svg', 'pdf'}

# Directory Configuration
UPLOAD_FOLDER = BASE_DIR / 'uploads'
DOWNLOAD_FOLDER = BASE_DIR / 'downloads'
TEMP_FOLDER = BASE_DIR / 'temp'
MODEL_DIR = BASE_DIR / 'models'
STATIC_FOLDER = BASE_DIR / 'static'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, DOWNLOAD_FOLDER, TEMP_FOLDER, MODEL_DIR, STATIC_FOLDER]:
    folder.mkdir(exist_ok=True)

# Model Configuration
os.environ['HF_HOME'] = str(MODEL_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_DIR)

# Session Configuration
SESSION_FILE = DOWNLOAD_FOLDER / 'sessions.json'
