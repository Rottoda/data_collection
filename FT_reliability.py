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
            print(f"❌ ERROR: FT 센서 설정 실패. NI-DAQmx 장치가 연결되었는지 확인하세요. ({e})")
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
        else: print("⚠️ 경고: 캘리브레이션 중 데이터를 읽지 못했습니다."); self.offset = np.zeros(6)
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
        print(f"❌ ERROR: 로봇 연결에 실패했습니다. ({e})"); sys.exit()