import sys
import os
import cv2
import numpy as np
from time import sleep

# 이 코드는 로봇 제어 API가 있는 폴더와 같은 상위 폴더에 있다고 가정합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApiMove

# ========================== 실험 설정 ==========================
ROBOT_IP = "192.168.1.6"
CAMERA_INDEX = 0
TARGET_PRESS_POINT = [340.0, 5.0, -107.0, 350.90]
SAFE_HEIGHT_OFFSET = 20.0
NUM_REPETITIONS = 5
# =================================================================

def setup_camera(is_auto_mode=True):
    """카메라를 자동 또는 수동 모드로 설정하는 함수"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다."); return None
    
    if is_auto_mode:
        print("-> 카메라를 [자동 모드]로 설정합니다.")
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        print("-> 카메라를 [수동 모드]로 설정합니다.")
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
        cap.set(cv2.CAP_PROP_GAIN, 0)
    
    sleep(1)
    return cap


def run_press_sequence(move, cap, mode_name):
    """지정된 좌표를 반복해서 누르고 '안정적으로' 사진을 찍는 함수"""
    target_safe_point = TARGET_PRESS_POINT.copy()
    target_safe_point[2] += SAFE_HEIGHT_OFFSET
    
    image_list = []
    for i in range(NUM_REPETITIONS):
        print(f"  [{mode_name.upper()}] {i+1}/{NUM_REPETITIONS}회차 누름 동작 시작...")
        move.MovL(*target_safe_point)
        sleep(1)
        move.MovL(*TARGET_PRESS_POINT)
        sleep(1)

        # --- 이미지 캡처 안정성 강화 로직 ---
        # 1. 버퍼 비우기: 오래된 프레임을 제거
        for _ in range(5):
            cap.read()
        
        # 2. 최신 프레임 캡처 (실패 시 5번 재시도)
        frame = None
        for attempt in range(5):
            ret, captured_frame = cap.read()
            if ret:
                frame = captured_frame
                break # 성공 시 재시도 중단
            print(f"    > 캡처 재시도... ({attempt + 1}/5)")
            sleep(0.1)
        # ------------------------------------

        if frame is not None:
            image_list.append(frame)
            cv2.imwrite(f"{mode_name}_raw_{i}.png", frame)
            print(f"    > 원본 사진 저장: {mode_name}_raw_{i}.png")
        else:
            print(f"    > 캡처 최종 실패: {mode_name}_{i}.png 저장 안 함")

        move.MovL(*target_safe_point)
        sleep(1)
        
    return image_list


def analyze_consistency(raw_image_list, mode_name):
    """촬영된 이미지들을 '이진화'한 후, 그 결과물의 일관성을 분석하는 함수"""
    print(f"\n--- [{mode_name.upper()}] 모드 [이진화 이미지] 일관성 분석 ---")
    if len(raw_image_list) < 2:
        print("비교할 이미지가 2장 미만이라 분석을 건너뜁니다."); return

    binarized_images = []
    for i, frame in enumerate(raw_image_list):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 55, -10)
        binarized_images.append(binarized)
        cv2.imwrite(f"{mode_name}_binarized_{i}.png", binarized)
    print("모든 캡처 이미지를 이진화 처리 완료.")

    # 첫 번째와 두 번째 '이진화 이미지'를 비교
    img1_bin = binarized_images[0]
    img2_bin = binarized_images[1]
    
    diff = cv2.absdiff(img1_bin, img2_bin)
    non_zero_pixels = np.count_nonzero(diff)
    diff_percentage = (non_zero_pixels / diff.size) * 100
    
    print(f"첫 번째와 두 번째 [이진화 이미지]의 차이가 있는 픽셀 비율: {diff_percentage:.4f}%")
    
    cv2.imshow(f"Difference of Binarized in [{mode_name.upper()}] mode", diff)
    cv2.imshow(f"[{mode_name.upper()}] Binarized Image 1", img1_bin)
    cv2.imshow(f"[{mode_name.upper()}] Binarized Image 2", img2_bin)
    print("차이점 이미지를 확인하세요. (완벽히 검은색이면 일관성 높음)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dashboard = DobotApiDashboard(ROBOT_IP, 29999)
    move = DobotApiMove(ROBOT_IP, 30003)
    dashboard.EnableRobot()
    
    print("\n" + "="*50 + "\n[실험 1] 카메라 자동 설정으로 일관성 테스트\n" + "="*50)
    cap_auto = setup_camera(is_auto_mode=True)
    if cap_auto:
        auto_images = run_press_sequence(move, cap_auto, "auto")
        cap_auto.release()
        analyze_consistency(auto_images, "auto")

    print("\n" + "="*50 + "\n[실험 2] 카메라 수동 설정으로 일관성 테스트\n" + "="*50)
    cap_manual = setup_camera(is_auto_mode=False)
    if cap_manual:
        manual_images = run_press_sequence(move, cap_manual, "manual")
        cap_manual.release()
        analyze_consistency(manual_images, "manual")
        
    dashboard.DisableRobot()
    print("\n[SUCCESS] 모든 실험이 완료되었습니다.")