import cv2
import os
import glob


# --- 설정 ---
# 기존에 수집했던 버전 1의 원본 이미지 폴더
origin_image_folder = r""

# 새로 저장할 폴더
reprocessed_folder = r""


os.makedirs(reprocessed_folder, exist_ok=True)


image_paths = sorted(glob.glob(os.path.join(origin_image_folder, "*.png")))

print(f"총 {len(image_paths)}개의 이미지를 재처리합니다...")

for img_path in image_paths:
    # 원본 이미지 읽기 (그레이스케일로)
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is not None:
        # 색상 반전하여 이진화 적용        
        binarized = cv2.adaptiveThreshold(gray_image, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, # 색상 반전 옵션
                                          51, -10) # 블록 크기와 상수 값 조정 가능

        filename = os.path.basename(img_path)
        save_path = os.path.join(reprocessed_folder, filename)
        cv2.imwrite(save_path, binarized)

print("완료!")