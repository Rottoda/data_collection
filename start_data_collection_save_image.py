import sys
import os
import threading
import pandas as pd
import numpy as np
from time import sleep, time
import cv2
from datetime import datetime
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
import keyboard

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
        "robot_ip": "192.168.1.6",

        # FT 센서 설정
        "ft_samples": 100,
        "ft_rate": 1000,
        "ft_time": 1.0
}

# =====================================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 필요한 모듈 임포트
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile

# ==================================================================
#                       FT 센서 클래스 (제공된 코드)
# ==================================================================
class FT_NI:
    def __init__(self,**kwargs):
        self.Nsamples = kwargs['samples']
        self.Ratesamples = kwargs['rate']
        self.task=nidaqmx.Task()
        self.offset = np.asarray([.0,.0,.0,.0,.0,.0])
        self.FTsetup()
        # 초기화 시 데이터 한번 읽기 (안정화)
        self.task.read()
        print("FT Sensor Initialized.")

    def FTsetup(self):
        try:
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
            self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=11)
        except nidaqmx.errors.DaqError as e:
            print(f" ERROR: FT 센서 설정 실패. NI-DAQmx 장치가 연결되었는지 확인하세요. ({e})")
            sys.exit()
        '''
        FTsetup 메서드는 NI DAQ 장치의 데이터 수집 작업을 설정
        아날로그 입력 채널을 추가하고, 샘플 클럭 타이밍을 구성

        self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5"):NI DAQ 장치의 아날로그 입력 채널을 추가
        "Dev1/ai0:5"은 장치 'Dev1'(장치 이름)의 아날로그 입력 채널 'ai0'부터 'ai5'까지를 의미
        'add_ai_voltage_chan'메서드는 전압 입력 채널을 추가

        self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=11):
        데이터 수집 작업의 샘플 클럭 타이밍을 설정
        'cfg_samp_clk_timing' 메서드를 사용해 샘플링 속도와 모드를 설정
        'self.Ratesamples'는 샘플링 속도를 설정 / ex) rate=1000으로 설정하면 초당 1000개의 샘플 수집
        'source='는 샘플 클럭의 소스를 설정 / 빈 문자열이므로 내부 클럭 사용을의미
        'active_edge=Edge.RISING'는 샘플 클럭의 활성 에지를 설정 / Edge.RISING은 상승 에지에서 샘플을 수집함을 의미
        'sample_mode=AcquisitionType.FINITE'는 샘플 수집 모드를 설정 / 정해진 수의 샘플을 수집하는 모드
        'samps_per_chan=11'는 채널당 수집할 샘플 수를 설정 / 각 채널에서 11개의 샘플을 수집함을 의미함 / 현재 6개의 채널(Fx~Tz)이므로 총 66개의 샘플을 수집
    '''
        
    def convertingRawData(self):
        # For FT52560
        bias = [0.1663, 1.5724, -0.0462, -0.0193, -0.0029, 0.0093]
        userAxis = [[-1.46365, 0.26208, 1.93786, -34.19271, -1.16799, 32.94542],
                    [-1.30893, 39.73726, -0.37039, -19.60236, 1.84250, -19.16761],
                    [19.26259, -0.08643, 19.17027, -0.25200, 19.58193, -0.01857],
                    [0.46529, 0.12592, -33.14392, 0.31541, 33.76824, -0.14065],
                    [37.74417, -0.26852, -18.62546, 0.43466, -19.49703, 0.01865],
                    [1.16572, -19.72447, 0.49027, -19.51112, 0.54533, -19.30472]]
        offSetCorrection = self.rawData - np.transpose(bias)
        self.forces = np.dot(userAxis, np.transpose(offSetCorrection))

    def readFT(self):
        self.voltages = self.task.read(self.Nsamples)
        self.rawData = np.mean(self.voltages,axis=1)
        self.convertingRawData()
        return self.forces
    
    def calibration(self, second=CONFIG['ft_time']):
        print(f'FT 센서 캘리브레이션을 {second}초 동안 시작합니다...')
        start_time = time()
        count = 0
        stacked_offset = np.asarray([.0,.0,.0,.0,.0,.0])

        while time() - start_time < second:
            stacked_offset += self.readFT()
            count += 1
        
        if count > 0:
            self.offset = stacked_offset / count
        else:
            print(" 경고: 캘리브레이션 중 데이터를 읽지 못했습니다.")
            self.offset = np.zeros(6)

        print(f'캘리브레이션 완료. Offset: {self.offset}')
        return self.offset
    
    def readFT_calibrated(self):
        if self.offset.sum() == 0:
            print("경고: 캘리브레이션이 수행되지 않았습니다. 원본 값을 반환합니다.")
        return self.readFT() - self.offset
    
    def close(self):
        self.task.close()
        print("FT Sensor task closed.")


# 전역 변수
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()


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
        print(f"  > 이미지 저장 완료: {img_filename} -> ORIG: {orig_path}, BIN: {bin_path}")
    else:
        print("카메라 프레임 캡처에 실패했습니다.")


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