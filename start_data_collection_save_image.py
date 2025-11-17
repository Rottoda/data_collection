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