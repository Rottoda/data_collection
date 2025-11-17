import cv2
import os
import glob

origin_image_folder = r""


reprocessed_folder = r""


os.makedirs(reprocessed_folder, exist_ok=True)


image_paths = sorted(glob.glob(os.path.join(origin_image_folder, "*.png")))

print(f"총 {len(image_paths)}개의 이미지를 재처리합니다...")

for img_path in image_paths:
    
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is not None:
        
        binarized = cv2.adaptiveThreshold(gray_image, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV,
                                          51, -10) 

        filename = os.path.basename(img_path)
        save_path = os.path.join(reprocessed_folder, filename)
        cv2.imwrite(save_path, binarized)

print("완료!")