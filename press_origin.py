# %%
import sys
import os
import threading
import numpy as np
from time import sleep, time
from datetime import datetime
import nidaqmx # Keep FT sensor part if calibration is needed at start
from nidaqmx.constants import AcquisitionType, Edge
import trimesh # STL 로드를 위해 추가
import traceback # 상세 오류 출력을 위해 추가

# ============================== 설정값 ===============================
CONFIG = {
    # --- STL 및 오프셋 설정 ---
    "stl_file_path": r"",
    "robot_origin_offset": np.array([340.0, 5.0, -87.40]), # 로봇 베이스 -> STL 원점
    "press_depth_at_origin_mm": 0.0, # <<< 원점을 얼마나 깊이 누를지 (mm), 0이면 표면까지만 >>>

    # --- 로봇 및 센서 설정 ---
    "radi" : 62.8, # 고정 회전 각도
    "camera_index": 0, # 카메라는 이제 사용하지 않지만 설정은 유지
    "robot_speed": 20, # 안전을 위해 속도 더 낮춤
    "safe_height_offset": 25.0, # 안전 높이 (mm), 조금 더 높게 설정
    "robot_ip": "192.168.1.6",
    # FT 센서는 초기 영점 조절 외에는 사용 안 함
    "ft_samples": 100,
    "ft_rate": 1000,
    "ft_time": 0.5 # 안정화 시간 단축
}
# =====================================================================

try:
    script_dir_abs = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir_abs = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(script_dir_abs, '..')))


try:
    from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
except ImportError:
    print("오류: Dobot API 모듈을 찾을 수 없습니다. 경로 설정을 확인하세요.")
    sys.exit(1)

class FT_NI:
    def __init__(self,**kwargs):
        self.Nsamples = kwargs['samples']
        self.Ratesamples = kwargs['rate']
        self.task=None 
        self.current_offset = np.asarray([.0,.0,.0,.0,.0,.0])
        try: 
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, Edge
            self.task=nidaqmx.Task() 
            self.FTsetup()
            print("FT Sensor Initialized.")
        except ImportError: 
             print("경고: nidaqmx 라이브러리를 찾을 수 없습니다. pip install nidaqmx")
             print("FT 센서 없이 진행합니다 (영점 조절 불가).")
             self.task = None
        except ConnectionError as e:
            print(f"초기화 실패: {e}")
            print("FT 센서 없이 진행합니다 (영점 조절 불가).")
            if self.task:
                 try:
                      self.task.close()
                 except: 
                      pass
            self.task = None 

    def FTsetup(self):
        if self.task is None: return # 초기화 실패 시 설정 건너뜀
        try:
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
            self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.Nsamples)
        except nidaqmx.errors.DaqError as e:
            # 설정 실패 시 task 닫기
            if self.task:
                try:
                    self.task.close()
                except: pass
            self.task = None
            raise ConnectionError(f"FT 센서 설정 실패: {e}")