import sys
import os
import threading
import pandas as pd
import numpy as np
from time import sleep
import cv2
from datetime import datetime

sys.path.append(os.path.abspath("./"))

# 필요한 모듈 임포트
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile

# 중심점
# CENTER_3D = np.array([338.04, 4.18, -135.66])
CENTER_3D = np.array([338.05, -19.31, -134.46])

# 전역 변수
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()

# 로봇 연결
def ConnectRobot():
    ip = "192.168.1.6"
    dashboardPort = 29999
    movePort = 30003
    feedPort = 30004
    dashboard = DobotApiDashboard(ip, dashboardPort)
    move = DobotApiMove(ip, movePort)
    feed = DobotApi(ip, feedPort)
    return dashboard, move, feed

# 피드백 쓰레드
def GetFeed(feed: DobotApi):
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

# 도착 대기
def WaitArrive(target):
    global current_actual
    while True:
        is_arrive = True
        with globalLockValue:
            if current_actual is not None:
                for i in range(4):
                    if abs(current_actual[i] - target[i]) > 1:
                        is_arrive = False
                if is_arrive:
                    return
        sleep(0.001)

# 오류 처리 쓰레드
def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()
    while True:
        with globalLockValue:
            if robotErrorState:
                print("로봇 오류 감지됨. 수동으로 초기화 필요.")
            else:
                if int(enableStatus_robot[0]) == 1 and int(algorithm_queue[0]) == 0:
                    dashboard.Continue()
        sleep(5)

# 이동 명령
def RunPoint(move: DobotApiMove, point, user, tool, speed):
    move.MovL(point[0], point[1], point[2], point[3], user, tool, speed)

# 이미지 촬영 및 저장
def CaptureImg(cap, origin_dir, bin_dir, i):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51, 10
        )

        img_filename = f"{i:03d}.png"
        orig_path = os.path.join(origin_dir, img_filename)
        bin_path = os.path.join(bin_dir, img_filename)

        cv2.imwrite(orig_path, frame)
        cv2.imwrite(bin_path, binarized)
        print(f"Img 저장 완료 → ORIG: {orig_path}, BIN: {bin_path}")
    else:
        print("⚠️ 카메라 프레임 캡처 실패")

# MAIN
if __name__ == "__main__":

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    # 스크립트 디렉토리 내에 'images' 폴더 경로를 설정합니다.
    base_path = os.path.join(script_dir, "images")

    # 기본 세션 디렉토리 경로를 타임스탬프로 생성합니다.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    # 1. 이미지 저장 디렉토리 생성 (원본과 동일)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_path, f"session_{timestamp}")
    origin_dir = os.path.join(session_dir, "origin")
    bin_dir = os.path.join(session_dir, "bin")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    print(f"[INFO] 이미지 저장 디렉토리 생성됨: {session_dir}")

    # 카메라 장치 열기
    cap = cv2.VideoCapture(0) # 실제 카메라 인덱스로 변경 필요
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        exit()

    # ==================== 수정된 부분 시작 ====================

    # Z값 계산에 필요한 타원체 정의 로직 추가
    # 3D 타원체 정의에 필요한 포인트 데이터
    points_edge = [
        [326.05, -19.31, -138.44], [328.05, -13.31, -137.34], [328.05, -25.31, -138.64],
        [333.05, -10.31, -137.74], [333.05, -28.31, -137.74], [338.05, -10.31, -137.74],
        [338.05, -28.31, -137.74], [343.05, -10.31, -137.74], [343.05, -28.31, -137.74],
        [348.05, -13.31, -137.34], [348.05, -25.31, -138.64], [350.05, -19.31, -138.44]
    ]
    points_inside = [
        [328.05, -16.31, -135.96], [328.05, -19.31, -135.06], [328.05, -22.31, -136.06],
        [333.05, -13.31, -136.34], [333.05, -16.31, -135.34], [333.05, -19.31, -134.46],
        [333.05, -22.31, -135.06], [333.05, -25.31, -136.06], [338.05, -13.31, -136.34],
        [338.05, -16.31, -135.34], [338.05, -22.31, -135.34], [338.05, -25.31, -136.34],
        [343.05, -13.31, -136.34], [343.05, -16.31, -135.34], [343.05, -19.31, -134.96],
        [343.05, -22.31, -135.34], [343.05, -25.31, -136.34], [348.05, -16.31, -138.64],
        [348.05, -19.31, -136.44], [348.05, -22.31, -137.34]
    ]
    points_center_data = [[338.05, -19.31, -134.46]]

    # 3D 타원체 파라미터 계산
    all_points_for_ellipsoid = points_edge + points_inside + points_center_data
    all_points_np = np.array(all_points_for_ellipsoid)
    ellipsoid_center_3d = np.mean(all_points_np, axis=0)
    ellipsoid_radii_3d = (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0)) / 2.0

    def get_ellipsoid_upper_z(x, y, center, radii):
        """주어진 (x, y)에 해당하는 타원체 상부 표면의 Z좌표를 계산합니다."""
        cx, cy, cz = center
        rx, ry, rz = radii
        term = 1 - ((x - cx) / rx)**2 - ((y - cy) / ry)**2
        if term < 0:
            term = 0 # (x,y)가 타원의 XY 투영 밖에 있는 경우, 가장자리로 간주
        z_offset = rz * np.sqrt(term)
        return cz + z_offset

    # CSV 로딩
    # CSV 파일 경로는 실제 환경에 맞게 수정해야 합니다.
    csv_path = os.path.join(script_dir,"relative_random_points_v2.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{csv_path}'. Please run the point generation script first.")
        sys.exit()


    # 상대 좌표 -> 절대 좌표 변환 (수정된 로직)
    absolute_points_list = []
    for index, row in df.iterrows():
        dx, dy, dz = row["dX"], row["dY"], row["dZ"]
        
        # 절대 X, Y 좌표는 CENTER_3D 기준
        absolute_x = CENTER_3D[0] + dx
        absolute_y = CENTER_3D[1] + dy
        
        # 절대 Z 좌표는 타원체 상부 표면 기준
        surface_z = get_ellipsoid_upper_z(absolute_x, absolute_y, ellipsoid_center_3d, ellipsoid_radii_3d)
        absolute_z = surface_z + dz -1
        
        absolute_points_list.append([absolute_x, absolute_y, absolute_z])

    absolute_points = np.array(absolute_points_list)
    
    # ==================== 수정된 부분 끝 ====================

    # 도봇 연결 (원본과 동일)
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()

    user, tool, speed = "User=1", "Tool=1", "SpeedL=30"

    # 피드백 쓰레드 실행 (원본과 동일)
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()
    threading.Thread(target=ClearRobotError, args=(dashboard,), daemon=True).start()

    # 카운트 (원본과 동일)
    i = 0
    try:
        print("[INFO] CSV 이동 명령 시작...")
        for point in absolute_points:
            # 1. 항상 중앙 값보다 높은 곳으로 이동
            target = [CENTER_3D[0], CENTER_3D[1], CENTER_3D[2]+20, 85.77]
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            # 2. x, y 포지셔닝
            target = [point[0], point[1], CENTER_3D[2]+20, 85.77]
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            # 3. z를 내려서 누르기
            target = [point[0], point[1], point[2], 85.77]
            print(f"[{i}번째] 이동 → {target}")
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)
            # 1초 대기
            sleep(1)

            # 이미지 캡쳐
            if cap.isOpened():
                CaptureImg(cap, origin_dir, bin_dir, i)

            # 4. z를 올리기 누르기
            target = [point[0], point[1], point[2]+20, 85.77]
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            i += 1

        print("[완료] 모든 포인트에 도달했습니다.")

        # 1. 항상 중앙 값보다 높은 곳으로 이동
        target = [CENTER_3D[0], CENTER_3D[1], CENTER_3D[2]+20, 85.77]
        RunPoint(move, target, user, tool, speed)
        WaitArrive(target)

    except KeyboardInterrupt:
        print("사용자 중단")

    finally:
        dashboard.DisableRobot() # 실제 로봇 사용 시 주석 해제
        print("로봇 비활성화 및 종료 완료.")

        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()