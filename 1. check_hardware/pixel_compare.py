import cv2
import numpy as np
import time

# ==================== 설정 ====================
CAMERA_INDEX = 0  # 확인할 카메라 인덱스
# ============================================

def capture_image(index, backend=None):
    """지정된 백엔드로 카메라 이미지를 캡처하는 함수"""
    if backend is not None:
        cap = cv2.VideoCapture(index, backend)
    else:
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        return None
    
    # 카메라 안정화를 위해 잠시 대기하고 여러 프레임을 읽어 버퍼를 비움
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

def analyze_difference(img1, img2):
    """두 이미지의 원본 및 이진화 결과 차이를 분석하고 시각화하는 함수"""
    # 1. 원본 이미지를 그레이스케일로 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. 원본 그레이스케일 이미지의 픽셀 값 차이 계산
    diff_image_gray = cv2.absdiff(gray1, gray2)
    enhanced_diff_gray = np.clip(diff_image_gray * 10, 0, 255).astype(np.uint8)

    # 3. 각 이미지를 Adaptive Threshold를 사용하여 이진화
    binarized1 = cv2.adaptiveThreshold(
        gray1, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 10
    )
    binarized2 = cv2.adaptiveThreshold(
        gray2, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 10
    )

    # 4. 이진화된 이미지의 픽셀 값 차이 계산
    diff_image_binary = cv2.absdiff(binarized1, binarized2)

    # 5. 차이점 통계 계산
    print("\n--- [원본] 픽셀 값 비교 분석 ---")
    total_pixels = diff_image_gray.size
    non_zero_pixels_gray = np.count_nonzero(diff_image_gray)
    diff_percentage_gray = (non_zero_pixels_gray / total_pixels) * 100
    mean_diff_gray = np.mean(diff_image_gray)
    print(f"값이 다른 픽셀 비율: {diff_percentage_gray:.4f}%")
    print(f"평균 픽셀 값 차이 (0~255): {mean_diff_gray:.4f}")

    print("\n--- [이진화] 픽셀 값 비교 분석 ---")
    non_zero_pixels_binary = np.count_nonzero(diff_image_binary)
    diff_percentage_binary = (non_zero_pixels_binary / total_pixels) * 100
    print(f"값이 다른 픽셀 수: {non_zero_pixels_binary} / {total_pixels}")
    print(f"차이가 있는 픽셀 비율: {diff_percentage_binary:.4f}%")
    
    # 6. 결과 이미지 표시
    # 원본 및 차이 이미지
    cv2.imshow("1. Original (DSHOW)", img1)
    cv2.imshow("2. Original (Default)", img2)
    cv2.imshow("3. Difference - Grayscale", diff_image_gray)
    cv2.imshow("4. Enhanced Difference - Grayscale (x10)", enhanced_diff_gray)
    
    # 이진화 및 차이 이미지
    cv2.imshow("5. Binarized (DSHOW)", binarized1)
    cv2.imshow("6. Binarized (Default)", binarized2)
    cv2.imshow("7. Difference - Binarized", diff_image_binary)
    
    print("\n결과 창이 표시되었습니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("CAP_DSHOW를 사용하여 이미지 캡처 중...")
    img_dshow = capture_image(CAMERA_INDEX, cv2.CAP_DSHOW)
    if img_dshow is None:
        print("ERROR: CAP_DSHOW로 카메라를 열 수 없습니다.")
        exit()
    print("캡처 성공.")

    print("\n기본 백엔드(MSMF)를 사용하여 이미지 캡처 중...")
    img_default = capture_image(CAMERA_INDEX)
    if img_default is None:
        print("ERROR: 기본 백엔드로 카메라를 열 수 없습니다.")
        exit()
    print("캡처 성공.")

    analyze_difference(img_dshow, img_default)