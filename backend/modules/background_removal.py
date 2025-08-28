"""
BiRefNet 기반 고품질 배경 제거 모듈
"""

import os
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


class BiRefNetModel:
    """고품질 배경 제거 모델"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.loaded = False
        
    def load_model(self, progress_callback=None):
        """모델 로드"""
        if self.loaded:
            return
            
        try:
            if progress_callback:
                progress_callback(10, "🤖 AI 모델 초기화 중...")
            
            # 디바이스 설정
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🍎 Apple Silicon GPU(MPS) 사용")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("🔥 NVIDIA GPU 사용")
            else:
                self.device = torch.device("cpu")
                print("💻 CPU 사용")
            
            if progress_callback:
                progress_callback(30, "📥 고품질 모델 다운로드 중...")
            
            # @models/ 폴더에서 BiRefNet 모델 로드
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            self.model = AutoModelForImageSegmentation.from_pretrained(
                'zhengpeng7/BiRefNet', 
                trust_remote_code=True,
                cache_dir=models_dir  # @models/ 폴더에 캐시 저장
            )
            self.model.to(self.device)
            self.model.eval()
            
            if progress_callback:
                progress_callback(50, "⚙️ 이미지 전처리 설정 중...")
            
            # 이미지 전처리 설정
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            if progress_callback:
                progress_callback(70, "🔧 모델 최적화 중...")
            
            # 모델 최적화
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            
            self.loaded = True
            
            if progress_callback:
                progress_callback(80, "✅ 고품질 AI 모델 준비 완료!")
            
            print("🚀 고품질 배경 제거 모델 로드 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 모델 로드 실패: {str(e)}")
            raise
    
    def remove_background(self, image, progress_callback=None):
        """고품질 배경 제거"""
        if not self.loaded:
            self.load_model(progress_callback)
        
        try:
            if progress_callback:
                progress_callback(85, "🎯 AI가 고정밀 분석하고 있습니다...")
            
            # 원본 크기 저장
            original_size = image.size
            
            # 이미지 전처리
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            if progress_callback:
                progress_callback(90, "🔮 고품질 배경 제거 작업 중...")
            
            # 추론 실행
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()
            
            # 마스크 후작업
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            if progress_callback:
                progress_callback(95, "✨ 최종 이미지 합성 중...")
            
            # 원본 이미지에 마스크 적용
            image_rgba = image.convert("RGBA")
            image_rgba.putalpha(mask)
            
            if progress_callback:
                progress_callback(100, "🎉 고품질 배경 제거 완료!")
            
            return image_rgba
            
        except Exception as e:
            print(f"❌ 배경 제거 실패: {e}")
            if progress_callback:
                progress_callback(0, f"❌ 작업 실패: {str(e)}")
            raise
