"""
비디오 프레임별 작업 모듈
"""

import os
import time
import threading
import shutil
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename


class VideoProcessor:
    """비디오 프레임별 작업 클래스"""
    
    def __init__(self):
        self.temp_dirs = {}
    
    def create_temp_dir(self, session_id):
        """세션별 임시 디렉토리 생성"""
        temp_dir = os.path.join('temp', session_id)
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dirs[session_id] = temp_dir
        return temp_dir
    
    def cleanup_temp_dir(self, session_id):
        """임시 디렉토리 정리"""
        if session_id in self.temp_dirs:
            try:
                shutil.rmtree(self.temp_dirs[session_id])
                del self.temp_dirs[session_id]
                print(f"🧹 임시 디렉토리 정리 완료: {session_id}")
            except Exception as e:
                print(f"⚠️ 임시 디렉토리 정리 실패: {e}")
    
    def extract_frames(self, video_path, temp_dir, progress_callback=None):
        """비디오에서 프레임 추출 (강화된 디버깅 및 오류 작업)"""
        try:
            if progress_callback:
                progress_callback(10, "🎬 비디오 정보 분석 중...")
            
            # 비디오 작업 존재 및 크기 확인
            if not os.path.exists(video_path):
                raise ValueError(f"비디오 작업이 존재하지 않습니다: {video_path}")
            
            file_size = os.path.getsize(video_path)
            print(f"📂 비디오 작업 확인: {video_path} (크기: {file_size:,} bytes)")
            
            if file_size == 0:
                raise ValueError("비디오 작업이 비어있습니다")
            
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
                    raise ValueError(f"비디오 작업을 열 수 없습니다. 지원되지 않는 코덱이거나 손상된 파일일 수 있습니다.\n파일: {video_path}")
            
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
                    # BGR to RGB 작업
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
                    print(f"⚠️ 프레임 {frame_count} 작업 중 오류: {frame_error}")
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
                raise ValueError("프레임을 하나도 추출할 수 없습니다. 비디오 작업이 손상되었거나 지원되지 않는 형식일 수 있습니다.")
            
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
    
    def process_frames(self, frame_files, remove_bg, upscale, scale_factor, background_color=None, progress_callback=None, ai_model=None, upscale_model=None):
        """각 프레임에 AI 작업 적용 (배경 색상 선택 지원)"""
        try:
            total_frames = len(frame_files)
            processed_files = []
            
            # 배경 색상 설정 (기본값: 흰색)
            if background_color and background_color.startswith('#') and len(background_color) == 7:
                try:
                    # 16진수 색상 코드를 RGB로 작업
                    bg_color = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
                    print(f"🎨 선택된 배경 색상: {background_color} (RGB: {bg_color})")
                except ValueError:
                    bg_color = (255, 255, 255)  # 잘못된 색상 코드 시 흰색 사용
                    print("⚠️ 잘못된 색상 코드, 흰색 배경 사용")
            else:
                bg_color = (255, 255, 255)  # 기본 흰색
            
            if progress_callback:
                progress_callback(20, f"🤖 AI 작업 시작... (총 {total_frames}개 프레임)")
            
            for i, frame_path in enumerate(frame_files):
                # 프레임 로드
                frame_image = Image.open(frame_path).convert('RGB')
                processed_image = frame_image
                
                # 배경 제거 적용
                if remove_bg and ai_model:
                    processed_image = ai_model.remove_background(processed_image)
                    # RGBA를 RGB로 작업 (비디오는 투명도 지원 안함)
                    if processed_image.mode == 'RGBA':
                        # 선택된 색상으로 배경 합성
                        color_bg = Image.new('RGB', processed_image.size, bg_color)
                        color_bg.paste(processed_image, mask=processed_image.split()[-1])
                        processed_image = color_bg
                
                # 업스케일링 적용
                if upscale and upscale_model:
                    processed_image = upscale_model.upscale_image(processed_image, scale_factor)
                
                # 작업된 프레임 저장
                processed_path = frame_path.replace('frames', 'processed').replace('.png', '_processed.png')
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                processed_image.save(processed_path)
                processed_files.append(processed_path)
                
                # 진행률 업데이트 (프레임 작업는 20%~80%)
                if progress_callback:
                    progress = 20 + (i + 1) / total_frames * 60
                    progress_callback(int(progress), f"🤖 AI가 프레임을 작업하고 있습니다... ({i+1}/{total_frames})")
            
            return processed_files
            
        except Exception as e:
            print(f"❌ 프레임 작업 실패: {e}")
            raise
    
    def extract_last_frame(self, video_path, progress_callback=None):
        """비디오에서 마지막 프레임 추출"""
        try:
            if progress_callback:
                progress_callback(10, "🎬 비디오 마지막 프레임 추출 중...")
            
            # 비디오 작업 존재 확인
            if not os.path.exists(video_path):
                raise ValueError(f"비디오 작업이 존재하지 않습니다: {video_path}")
            
            file_size = os.path.getsize(video_path)
            print(f"📂 비디오 작업 확인: {video_path} (크기: {file_size:,} bytes)")
            
            if file_size == 0:
                raise ValueError("비디오 작업이 비어있습니다")
            
            if progress_callback:
                progress_callback(30, "📹 비디오 정보 분석 중...")
            
            # 비디오 캡처 객체 생성
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"비디오 작업을 열 수 없습니다: {video_path}")
            
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
                progress_callback(85, "🎨 이미지 작업 중...")
            
            # BGR to RGB 작업
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
        """작업된 프레임들을 비디오로 재조립 (H.264 코덱 사용)"""
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
                
                # PIL to OpenCV 작업
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
