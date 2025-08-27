"""
Real-ESRGAN 기반 AI 업스케일링 모듈
"""

import os
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact


class RealESRGANUpscaleModel:
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
                
                # @models/ 폴더에 Real-ESRGAN 모델 저장
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'realesrgan')
                os.makedirs(models_dir, exist_ok=True)
                
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
