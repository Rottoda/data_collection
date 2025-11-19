import sys
import os
import threading
import pandas as pd
import numpy as np
from time import sleep, time
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# ============================== 설정값 ===============================
CONFIG = {
    # 1. 반복해서 누를 단일 목표 지점 [x, y, z, r]
    "target_point_xyzr": [338.0, 5.0, -115.0, 334.80],
    
    # 2. 반복 횟수
    "num_repetitions": 10,
    
    # 3. 로봇 및 FT 센서 설정
    "robot_speed": 50,
    "safe_height_offset": 20.0,
    "ft_samples": 100,
    "ft_rate": 1000,
    "ft_calibration_time_sec": 1.0
}
# =====================================================================
class FT_NI:
    def __init__(self,**kwargs):
        self.Nsamples = kwargs['samples']
        self.Ratesamples = kwargs['rate']
        self.task=nidaqmx.Task()
        self.offset = np.asarray([.0,.0,.0,.0,.0,.0])
        self.FTsetup()
        self.task.read() # 안정화
        print("FT Sensor Initialized.")

    def FTsetup(self):
        try:
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
            self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=11)
        except nidaqmx.errors.DaqError as e:
            print(f"ERROR: FT 센서 설정 실패. NI-DAQmx 장치가 연결되었는지 확인하세요. ({e})")
            sys.exit()

    def convertingRawData(self):
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

    def calibration(self, second=1):
        print(f'FT 센서 캘리브레이션을 {second}초 동안 시작합니다...')
        start_time = time()
        count = 0
        stacked_offset = np.zeros(6)
        while time() - start_time < second:
            stacked_offset += self.readFT()
            count += 1
        if count > 0: self.offset = stacked_offset / count
        else: print("경고: 캘리브레이션 중 데이터를 읽지 못했습니다."); self.offset = np.zeros(6)
        print(f'캘리브레이션 완료. Offset: {np.round(self.offset, 3)}')
        return self.offset

    def readFT_calibrated(self):
        return self.readFT() - self.offset
    
    def close(self):
        self.task.close()
        print("FT Sensor task closed.")

current_actual = None
globalLockValue = threading.Lock()

def ConnectRobot():
    ip = "192.168.1.6"
    dashboardPort = 29999; movePort = 30003; feedPort = 30004
    print("Connecting to robot...")
    try:
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print("Connection successful.")
        return dashboard, move, feed
    except Exception as e:
        print(f"ERROR: 로봇 연결에 실패했습니다. ({e})"); sys.exit()

def GetFeed(feed: DobotApi):
    global current_actual
    hasRead = 0
    while True:
        data = bytes();
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0: hasRead += len(temp); data += temp
        hasRead = 0
        feedInfo = np.frombuffer(data, dtype=MyType)
        if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
            with globalLockValue: current_actual = feedInfo["tool_vector_actual"][0]
        sleep(0.001)

def WaitArrive(target_point):
    global current_actual
    while True:
        with globalLockValue:
            if current_actual is not None and np.all(np.abs(np.subtract(current_actual[:4], target_point[:4])) < 1):
                return
        sleep(0.5)

def plot_ft_data(data_history):
    """수집된 FT 데이터를 6축으로 나누어 그래프로 보여줍니다."""
    if not data_history:
        print("플로팅할 데이터가 없습니다.")
        return
        
    data_array = np.array(data_history)
    labels = ['Fx', 'Fy', 'Fz']
    colors = ['r', 'g', 'b']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('FT Sensor Reliability Test Results', fontsize=16)
    axes = axes.flatten()

    for i in range(3):
        ax = axes[i]
        ax.plot(data_array[:, i], marker='o', linestyle='-', color=colors[i], markersize=4)
        ax.set_title(labels[i])
        ax.set_xlabel('Repetition Count')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # --- 하드웨어 초기화 ---
    dashboard, move, feed = ConnectRobot()
    FT = FT_NI(samples=CONFIG["ft_samples"], rate=CONFIG["ft_rate"])
    
    dashboard.EnableRobot()
    print("[INFO] 로봇 활성화 완료.")
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()

    # --- 데이터 수집 루프 시작 ---
    all_ft_data = []
    try:
        user, tool, speed = 1, 1, CONFIG["robot_speed"]
        target_press = CONFIG["target_point_xyzr"]
        target_safe = target_press.copy()
        target_safe[2] += CONFIG["safe_height_offset"]

        # --- FT 센서 캘리브레이션 ---
        print("\n[INFO] 캘리브레이션을 위해 안전 위치로 이동합니다.")
        move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
        WaitArrive(target_safe)
        FT.calibration(CONFIG["ft_calibration_time_sec"])
        
        print(f"\n[INFO] 총 {CONFIG['num_repetitions']}회 반복 테스트를 시작합니다...")
        for i in range(CONFIG['num_repetitions']):
            print(f"--- [{i+1}/{CONFIG['num_repetitions']}] 번째 측정 중 ---")

            # 1. 목표 지점 누르기
            move.MovL(target_press[0], target_press[1], target_press[2], target_press[3], user, tool, speed)
            WaitArrive(target_press)
            sleep(0.5) # 안정화 대기

            # 2. FT 데이터 측정
            ft_data = FT.readFT_calibrated()
            all_ft_data.append(ft_data[:3])
            print(f"  > FT 데이터 측정: [Fx, Fy, Fz] = [{ft_data[0]:.3f}, {ft_data[1]:.3f}, {ft_data[2]:.3f}]")

            # 3. 안전 위치로 복귀
            move.MovL(target_safe[0], target_safe[1], target_safe[2], target_safe[3], user, tool, speed)
            WaitArrive(target_safe)
            sleep(0.8) # 다음 측정을 위한 대기

        print("\n[SUCCESS] 모든 측정을 완료했습니다.")

    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # --- 종료 처리 ---
        print("[INFO] 로봇 및 센서를 종료합니다.")

        # 최종 데이터 파일 저장
        if all_ft_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ft_reliability_test_{timestamp}.csv"
            df = pd.DataFrame(all_ft_data, columns=['Fx', 'Fy', 'Fz'])
            df.to_csv(filename, index_label="Repetition")
            print(f"  > 테스트 결과 저장 완료: {filename}")

        dashboard.DisableRobot()
        print("  > 로봇 비활성화 완료.")
        FT.close()
        print("  > 프로그램 종료.")

    # --- 최종 결과 플로팅 ---
    plot_ft_data(all_ft_data)
