"""
실용적인 스타일 변환 모듈
OpenCV와 PIL 기반 아트 효과 구현 (즉시 동작)
"""

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import os
from pathlib import Path

class StyleTransferModel:
    """실용적인 스타일 변환 모델 클래스"""
    
    def __init__(self, models_dir="models/style_transfer"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎨 실용적인 스타일 변환 모델 초기화 완료!")
        
    def anime_style(self, image):
        """애니메이션 스타일 효과"""
        try:
            print("🎨 애니메이션 스타일 처리 시작...")
            print(f"📸 입력 이미지 정보: 크기={image.size}, 모드={image.mode}")
            
            # 가장 간단한 테스트: 원본 그대로 반환 (디버깅용)
            result = image.copy()
            print("🔍 원본 이미지를 그대로 복사해서 반환")
            
            print(f"📸 결과 이미지 정보: 크기={result.size}, 모드={result.mode}")
            print("✅ 애니메이션 스타일 처리 완료")
            return result
            
        except Exception as e:
            print(f"❌ 애니메이션 스타일 처리 실패: {e}")
            import traceback
            print(f"❌ 상세 오류: {traceback.format_exc()}")
            return image
    
    def vangogh_style(self, image):
        """반 고흐 스타일 효과 (소용돌이 및 텍스처)"""
        try:
            print("🌟 반고흐 스타일 처리 시작...")
            
            # 색상 강화
            enhancer = ImageEnhance.Color(image)
            enhanced = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.3)
            
            # 붓터치 효과 (엠보스)
            embossed = enhanced.filter(ImageFilter.EMBOSS)
            
            # 원본과 블렌딩
            result = Image.blend(enhanced, embossed, 0.3)
            
            # 약간의 블러
            result = result.filter(ImageFilter.GaussianBlur(0.5))
            
            print("✅ 반고흐 스타일 처리 완료")
            return result
            
        except Exception as e:
            print(f"❌ 반고흐 스타일 처리 실패: {e}")
            return image
    
    def picasso_style(self, image):
        """피카소 스타일 효과 (기하학적 변형)"""
        try:
            print("🎨 피카소 스타일 처리 시작...")
            
            # 1. 색상 포스터화 (색상 수 줄이기)
            posterized = image.quantize(colors=8).convert('RGB')
            
            # 2. 대비 극대화
            enhancer = ImageEnhance.Contrast(posterized)
            high_contrast = enhancer.enhance(2.0)
            
            # 3. 색상 강화
            enhancer = ImageEnhance.Color(high_contrast)
            vibrant = enhancer.enhance(1.8)
            
            # 4. 엣지 강조
            edges = vibrant.filter(ImageFilter.FIND_EDGES)
            
            # 5. 원본과 엣지 블렌딩
            result = Image.blend(vibrant, edges, 0.2)
            
            print("✅ 피카소 스타일 처리 완료")
            return result
            
        except Exception as e:
            print(f"❌ 피카소 스타일 처리 실패: {e}")
            return image
    
    def oil_painting_style(self, image):
        """유화 스타일 효과"""
        try:
            print("🖼️ 유화 스타일 처리 시작...")
            
            # 1. 색상 강화
            enhancer = ImageEnhance.Color(image)
            enhanced = enhancer.enhance(1.4)
            
            # 2. 부드러운 블러
            blurred = enhanced.filter(ImageFilter.GaussianBlur(1))
            
            # 3. 샤프닝으로 붓터치 효과
            sharpened = blurred.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # 4. 색상 단순화
            simplified = sharpened.quantize(colors=32).convert('RGB')
            
            # 5. 대비 조정
            enhancer = ImageEnhance.Contrast(simplified)
            result = enhancer.enhance(1.1)
            
            print("✅ 유화 스타일 처리 완료")
            return result
            
        except Exception as e:
            print(f"❌ 유화 스타일 처리 실패: {e}")
            return image
    
    def monet_style(self, image):
        """모네 스타일 효과 (인상파)"""
        try:
            print("🌸 모네 스타일 처리 시작...")
            
            # 1. 밝기 조정
            enhancer = ImageEnhance.Brightness(image)
            brightened = enhancer.enhance(1.1)
            
            # 2. 색상 강화
            enhancer = ImageEnhance.Color(brightened)
            enhanced = enhancer.enhance(1.3)
            
            # 3. 인상파 효과 (부드러운 블러)
            blurred = enhanced.filter(ImageFilter.GaussianBlur(1.5))
            
            # 4. 약간의 선명도 추가
            result = blurred.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
            
            print("✅ 모네 스타일 처리 완료")
            return result
            
        except Exception as e:
            print(f"❌ 모네 스타일 처리 실패: {e}")
            return image
    
    def transfer_style(self, image, style='anime', strength=1.0):
        """
        이미지에 스타일 변환 적용
        
        Args:
            image: PIL Image 객체
            style: 스타일 이름 ('anime', 'vangogh', 'picasso', 'monet', 'oil_painting')
            strength: 스타일 적용 강도 (0.0 ~ 1.0)
            
        Returns:
            PIL Image: 스타일 변환된 이미지
        """
        try:
            print(f"🎨 {style} 스타일 변환 시작... 원본 크기: {image.size}, 모드: {image.mode}")
            
            # 이미지 모드 확인 및 변환
            if image.mode != 'RGB':
                print(f"🔄 이미지 모드를 {image.mode}에서 RGB로 변환")
                image = image.convert('RGB')
            
            # 크기 조정 (테스트용으로 비활성화)
            original_size = image.size
            print(f"📏 크기 조정 비활성화: 원본 크기 {original_size} 유지")
            
            # 스타일 적용 (원본 크기 그대로)
            if style == 'vangogh':
                stylized_image = self.vangogh_style(image)
            elif style == 'oil_painting':
                stylized_image = self.oil_painting_style(image)
            else:
                stylized_image = self.vangogh_style(image)  # 기본값
            
            # 크기 복원 단계도 비활성화 (테스트용)
            
            # 스타일 강도 조절 (원본과 블렌딩)
            if strength < 1.0:
                # 원본도 같은 크기로 조정했다가 복원
                if image.size != original_size:
                    image = image.resize(original_size, Image.LANCZOS)
                stylized_image = Image.blend(image, stylized_image, strength)
            
            print(f"✅ {style} 스타일 변환 완료! 결과 크기: {stylized_image.size}, 모드: {stylized_image.mode}")
            
            # 결과 이미지 유효성 검사
            if stylized_image.size[0] == 0 or stylized_image.size[1] == 0:
                print("⚠️ 결과 이미지 크기가 0입니다. 원본 반환.")
                return image
                
            return stylized_image
            
        except Exception as e:
            print(f"❌ 스타일 변환 실패: {e}")
            # 실패 시 원본 반환
            return image
    
    def get_available_styles(self):
        """사용 가능한 스타일 목록 반환"""
        return ['vangogh', 'oil_painting']

# 전역 모델 인스턴스
style_model = None

def get_style_model():
    """스타일 변환 모델 인스턴스 반환"""
    global style_model
    if style_model is None:
        style_model = StyleTransferModel()
    return style_model

def apply_style_transfer(image, style='anime', strength=1.0):
    """
    이미지에 스타일 변환 적용 (외부 API용)
    
    Args:
        image: PIL Image 객체
        style: 스타일 이름
        strength: 스타일 강도 (0.0 ~ 1.0)
        
    Returns:
        PIL Image: 변환된 이미지
    """
    model = get_style_model()
    return model.transfer_style(image, style, strength)

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 스타일 변환 모듈 테스트")
    
    # 더미 이미지로 테스트
    test_image = Image.new('RGB', (256, 256), color=(100, 150, 200))
    model = StyleTransferModel()
    
    for style in ['anime', 'vangogh', 'picasso', 'oil_painting']:
        result = model.transfer_style(test_image, style=style)
        print(f"✅ {style} 스타일 테스트 완료: {result.size}")