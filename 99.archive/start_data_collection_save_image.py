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

    # 1. 이미지 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("images", f"session_{timestamp}")
    origin_dir = os.path.join(session_dir, "origin")
    bin_dir = os.path.join(session_dir, "bin")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(bin_dir, exist_ok=True)

    print(f"[INFO] 이미지 저장 디렉토리 생성됨: {session_dir}")

    # 카메라 장치 열기
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        exit()

    # CSV 로딩 및 상대 좌표 → 절대 좌표 변환
    df = pd.read_csv("./data_collection/relative_random_points_v2.csv")
    absolute_points = df[["dX", "dY", "dZ"]].values + CENTER_3D

    # 도버 연결
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()

    user, tool, speed = "User=1", "Tool=1", "SpeedL=30"


    # 피드백 쓰레드 실행
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()
    threading.Thread(target=ClearRobotError, args=(dashboard,), daemon=True).start()

    # 카운트
    i = 0
    try:
        print("[INFO] CSV 이동 명령 시작...")
        for point in absolute_points:
            # 1. 항상 중앙 값보다 높은 곳으로 이동
            target = [CENTER_3D[0], CENTER_3D[1], CENTER_3D[2]+20, 85.77]  # R값은 예시로 85.77 고정
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            # 2. x, y 포지셔닝
            target = [point[0], point[1], CENTER_3D[2]+20, 85.77]  # R값은 예시로 85.77 고정
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            # 3. z를 내려서 누르기
            target = [point[0], point[1], point[2], 85.77]  # R값은 예시로 85.77 고정
            print(f"[{i}번째] 이동 → {target}")
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)
            # 1초 대기
            sleep(1)

            # 이미지 캡쳐
            CaptureImg(cap, origin_dir, bin_dir, i)

            # 4. z를 올리기 누르기
            target = [point[0], point[1], point[2]+20, 85.77]  # R값은 예시로 85.77 고정
            RunPoint(move, target, user, tool, speed)
            WaitArrive(target)

            i += 1

        print("[완료] 모든 포인트에 도달했습니다.")

        # 1. 항상 중앙 값보다 높은 곳으로 이동
        target = [CENTER_3D[0], CENTER_3D[1], CENTER_3D[2]+20, 85.77]  # R값은 예시로 85.77 고정
        RunPoint(move, target, user, tool, speed)
        WaitArrive(target)

        

    except KeyboardInterrupt:
        print("사용자 중단")

    finally:
        dashboard.DisableRobot()
        print("로봇 비활성화 및 종료 완료.")

        cap.release()
        cv2.destroyAllWindows()
