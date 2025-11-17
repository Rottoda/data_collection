import sys
import os
import threading
import pandas as pd
import numpy as np
from time import sleep
import cv2
from datetime import datetime

# ============================== 설정값 ===============================
CONFIG = {
        # 고정 회전 각도
        "radi" : 334.80,

        # 카메라 인덱스
        "camera_index": 0,

        # 로봇 속도 (1~100)
        "robot_speed": 50,

        # 로봇의 안전높이 (눌림의 위치보다 얼마나 위로 이동할지, 단위: mm)
        "safe_height_offset": 20.0,

        # point_from_stl.py에서 생성된 CSV 파일 이름
        "csv_filename": "robot_press_points.csv",

        # 로봇의 ip 주소
        "robot_ip": "192.168.1.6"
}

# =====================================================================


# 프로젝트 루트 경로를 sys.path에 추가하여 모듈을 찾을 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 필요한 모듈 임포트
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile

# 전역 변수
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()

# 로봇 연결 함수 (원본과 동일)
def ConnectRobot():
    """로봇의 IP와 포트에 연결하고 API 인스턴스를 반환합니다."""
    ip = CONFIG['robot_ip']
    dashboardPort = 29999
    movePort = 30003
    feedPort = 30004
    print("Connecting to robot...")
    dashboard = DobotApiDashboard(ip, dashboardPort)
    move = DobotApiMove(ip, movePort)
    feed = DobotApi(ip, feedPort)
    print("Connection successful.")
    return dashboard, move, feed

# 로봇 피드백 수신 쓰레드 (원본과 동일)
def GetFeed(feed: DobotApi):
    """로봇의 현재 상태를 실시간으로 수신하는 쓰레드 함수"""
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState
    hasRead = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0
        feedInfo = np.frombuffer(data, dtype=MyType)
        if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
            with globalLockValue:
                current_actual = feedInfo["tool_vector_actual"][0]
                algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                enableStatus_robot = feedInfo['EnableStatus'][0]
                robotErrorState = feedInfo['ErrorStatus'][0]
        sleep(0.001)

# 목표 지점 도착 대기 함수 (원본과 동일)
def WaitArrive(target_point):
    """로봇이 목표 지점에 도착할 때까지 대기합니다."""
    global current_actual
    while True:
        is_arrive = True
        with globalLockValue:
            if current_actual is not None:
                for i in range(4): # X, Y, Z, R
                    if abs(current_actual[i] - target_point[i]) > 1: # 1mm 오차 허용
                        is_arrive = False
                        break
                if is_arrive:
                    return
        sleep(0.001)

# 이미지 촬영 및 저장 함수 (원본과 동일)
def CaptureImg(cap, origin_dir, bin_dir, index):
    """카메라에서 이미지를 캡처하고 원본 및 이진화 이미지를 저장합니다."""
    # 안정성을 위해 여러 번 읽기 시도
    for _ in range(3): # 최대 3번 시도
        ret, frame = cap.read()
        if ret:
            break # 성공하면 루프 탈출
        sleep(0.1) # 실패 시 잠시 대기 후 재시도

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51, 10
        )
        img_filename = f"{index:03d}.png"
        orig_path = os.path.join(origin_dir, img_filename)
        bin_path = os.path.join(bin_dir, img_filename)
        cv2.imwrite(orig_path, frame)
        cv2.imwrite(bin_path, binarized)
        print(f"  > 이미지 저장 완료: {img_filename}→ ORIG: {orig_path}, BIN: {bin_path}")
    else:
        print("카메라 프레임 캡처에 실패했습니다.")


# ==================================================================
#                           메인 실행 부분
# ==================================================================
if __name__ == "__main__":
    # --- 1. 기본 설정 및 디렉토리 생성 ---
    try:
        # 현재 스크립트가 있는 디렉토리를 기준으로 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(script_dir, "images", f"session_{timestamp}")
    origin_dir = os.path.join(session_dir, "origin")
    bin_dir = os.path.join(session_dir, "bin")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)
    print(f"[INFO] 이미지 저장 디렉토리: {session_dir}")

    # --- 2. STL 기반 좌표 데이터 로드 ---
    # 좌표 생성 스크립트에서 만든 CSV 파일 경로
    csv_path = os.path.join(script_dir, CONFIG['csv_filename'])
    try:
        df = pd.read_csv(csv_path)
        # 로봇이 이동할 절대 좌표만 추출
        absolute_points = df[['x', 'y', 'z']].values
        print(f"[INFO] '{csv_path}'에서 {len(absolute_points)}개의 좌표를 로드했습니다.")
    except FileNotFoundError:
        print(f"ERROR: CSV 파일을 찾을 수 없습니다. 경로: '{csv_path}'")
        print("좌표 생성 스크립트를 먼저 실행해주세요.")
        sys.exit()

    # --- 3. 하드웨어 연결 (카메라 및 로봇) ---
    cap = cv2.VideoCapture(CONFIG["camera_index"]) # 실제 사용하는 카메라 인덱스로 변경
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다.")
        sys.exit()

    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()
    print("[INFO] 로봇 활성화 완료.")

    # 피드백 쓰레드 시작
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()
    
    # --- 4. 데이터 수집 루프 시작 ---
    try:
        # 로봇 이동 파라미터
        user, tool, speed = 1, 1, CONFIG["robot_speed"]

        print("\n[INFO] 데이터 수집을 시작합니다...")
        for i, point in enumerate(absolute_points):
            x, y, z = point
            
            # 안전한 중간 지점으로 먼저 이동
            center_x, center_y = absolute_points[:, 0].mean(), absolute_points[:, 1].mean()
            safe_center_z = absolute_points[:, 2].max() + CONFIG["safe_height_offset"]
            
            if i == 0:
                initial_target = [center_x, center_y, safe_center_z, CONFIG["radi"]]
                move.MovL(initial_target[0], initial_target[1], initial_target[2], initial_target[3], user, tool, speed)
                WaitArrive(initial_target)

            print(f"--- [{i+1}/{len(absolute_points)}] 번째 포인트 처리 중 ---")
            
            # 1. 누를 위치의 상단 (안전 높이)으로 이동
            target_safe = [x, y, z + CONFIG["safe_height_offset"], CONFIG["radi"]]
            print(f"  > 안전 위치로 이동: X={target_safe[0]:.2f}, Y={target_safe[1]:.2f}, Z={target_safe[2]:.2f}")
            move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
            WaitArrive(target_safe)

            # 2. Z축을 내려서 목표 지점 누르기
            target_press = [x, y, z, CONFIG["radi"]]
            print(f"  > 목표 지점 누르기: X={target_press[0]:.2f}, Y={target_press[1]:.2f}, Z={target_press[2]:.2f}")
            move.MovL(target_press[0], target_press[1], target_press[2], target_press[3], user, tool, speed)
            WaitArrive(target_press)
            sleep(0.5) # 누른 후 안정화를 위해 잠시 대기

            # 3. 이미지 촬영
            CaptureImg(cap, origin_dir, bin_dir, i)

            # 4. 다시 안전 높이로 Z축 올리기
            print(f"  > 안전 위치로 복귀 중...")
            move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
            WaitArrive(target_safe)

        print("\n[SUCCESS] 모든 포인트에 대한 데이터 수집을 완료했습니다.")

    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # --- 5. 종료 처리 ---
        print("[INFO] 로봇 및 카메라를 종료합니다.")
        dashboard.DisableRobot()
        print("  > 로봇 비활성화 완료.")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("  > 프로그램 종료.")
