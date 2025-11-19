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

# ============================== 설정값 ===============================
CONFIG = {
        # 고정 회전 각도
        "radi" : 350.90,

        # 카메라 인덱스
        "camera_index": 0,

        # 로봇 속도 (1~100)
        "robot_speed": 50,

        # 로봇의 안전높이 (눌림의 위치보다 얼마나 위로 이동할지, 단위: mm)
        "safe_height_offset": 20.0,

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
        self.current_offset = np.asarray([.0,.0,.0,.0,.0,.0])
        self.FTsetup()
        print("FT Sensor Initialized.")

    def FTsetup(self):
        try:
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
            self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=100)
        except nidaqmx.errors.DaqError as e:
            print(f"ERROR: FT 센서 설정 실패. NI-DAQmx 장치가 연결되었는지 확인하세요. ({e})")
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
        
    def calibration(self, second=0.5):
        """지정된 시간 동안의 평균값을 새로운 영점(offset)으로 설정합니다."""
        print(f"  > {second}초 동안 영점 조절...")
        start_time = time()
        count = 0
        stacked_offset = np.zeros(6)
        while time() - start_time < second:
            stacked_offset += self.readFT()
            count += 1
        if count > 0:
            self.current_offset = stacked_offset / count
        else:
            print("경고: 영점 조절 중 데이터를 읽지 못했습니다.")
            self.current_offset = np.zeros(6)
        
    def readFT_calibrated(self):
        """현재 영점을 기준으로 보정된 값을 읽습니다."""
        return self.readFT() - self.current_offset

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

    # --- 2. STL 기반 좌표 데이터 로드 ---
    # 좌표 생성 스크립트에서 만든 CSV 파일 경로
    csv_path = os.path.join(script_dir, CONFIG['csv_filename'])
    try:
        df = pd.read_csv(csv_path)
        # 로봇이 이동할 절대 좌표만 추출
        absolute_points = df[['x', 'y', 'z']].values
        print(f"[INFO] '{csv_path}'에서 {len(absolute_points)}개의 좌표를 로드했습니다.")
    except FileNotFoundError:
        print(f" ERROR: CSV 파일을 찾을 수 없습니다. 경로: '{csv_path}'")
        print("좌표 생성 스크립트를 먼저 실행해주세요.")
        sys.exit()

    # --- 3. 하드웨어 연결 (카메라 및 로봇) ---
    cap = cv2.VideoCapture(CONFIG["camera_index"], cv2.CAP_DSHOW) # 실제 사용하는 카메라 인덱스로 변경
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다.")
        sys.exit()

    dashboard, move, feed = ConnectRobot()
    FT = FT_NI(samples=CONFIG["ft_samples"], rate=CONFIG["ft_rate"])
    dashboard.EnableRobot()
    print("[INFO] 로봇 활성화 완료.")

    # 피드백 쓰레드 시작
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()


    # --- 4. 데이터 수집 루프 시작 ---
    all_data = []  # 수집된 데이터 저장용 리스트
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
                FT.calibration(CONFIG['ft_time'])  # FT 센서 캘리브레이션

            print(f"--- [{i+1}/{len(absolute_points)}] 번째 포인트 처리 중 ---")
            
            # 누를 위치의 상단 (안전 높이)으로 이동
            target_safe = [x, y, z + CONFIG["safe_height_offset"], CONFIG["radi"]]
            print(f"  > 안전 위치로 이동: X={target_safe[0]:.2f}, Y={target_safe[1]:.2f}, Z={target_safe[2]:.2f}")
            move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
            WaitArrive(target_safe)

            # Z축을 내려서 목표 지점 누르기
            target_press = [x, y, z, CONFIG["radi"]]
            print(f"  > 목표 지점 누르기: X={target_press[0]:.2f}, Y={target_press[1]:.2f}, Z={target_press[2]:.2f}")
            move.MovL(target_press[0], target_press[1], target_press[2], target_press[3], user, tool, speed)
            WaitArrive(target_press)
            sleep(0.5) # 누른 후 안정화를 위해 잠시 대기

            # 이미지 촬영
            CaptureImg(cap, origin_dir, bin_dir, i)

            ft_data = FT.readFT_calibrated()
            print(f"  > FT 데이터 측정: [Fx, Fy, Fz] = [{ft_data[0]:.3f}, {ft_data[1]:.3f}, {ft_data[2]:.3f}]")

            # --- 측정된 데이터 기록 ---

            # 원본 CSV의 해당 행 정보 가져오기
            original_row = df.iloc[i].to_dict()
            # FT 데이터 추가
            original_row['Fx'] = ft_data[0]
            original_row['Fy'] = ft_data[1]
            original_row['Fz'] = ft_data[2]
            original_row['Tx'] = ft_data[3]
            original_row['Ty'] = ft_data[4]
            original_row['Tz'] = ft_data[5]
            # 수집된 데이터 리스트에 추가
            all_data.append(original_row)

            # 다시 안전 높이로 Z축 올리기
            print(f"  > 안전 위치로 복귀 중...")
            move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
            WaitArrive(target_safe)

        print("\n[SUCCESS] 모든 포인트에 대한 데이터 수집을 완료했습니다.")

    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # --- 5. 종료 처리 ---
        print("[INFO] 로봇 및 센서를 종료합니다.")
        # 최종 데이터 파일 저장
        if all_data:
            final_df = pd.DataFrame(all_data)
            output_path = os.path.join(session_dir, "full_data.csv")
            final_df.to_csv(output_path, index=False)
            print(f"  > 최종 통합 데이터 저장 완료: {output_path}")

        dashboard.DisableRobot()
        print("  > 로봇 비활성화 완료.")
        FT.close()
        WaitArrive(target_safe)
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("  > 프로그램 종료.")
