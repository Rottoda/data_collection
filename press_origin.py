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
        
    def convertingRawData(self):
        if self.task is None: return np.zeros(6)
        # For FT52560 - 필요시 센서 모델에 맞게 수정
        bias = [0.1663, 1.5724, -0.0462, -0.0193, -0.0029, 0.0093]
        userAxis = [[-1.46365, 0.26208, 1.93786, -34.19271, -1.16799, 32.94542],
                    [-1.30893, 39.73726, -0.37039, -19.60236, 1.84250, -19.16761],
                    [19.26259, -0.08643, 19.17027, -0.25200, 19.58193, -0.01857],
                    [0.46529, 0.12592, -33.14392, 0.31541, 33.76824, -0.14065],
                    [37.74417, -0.26852, -18.62546, 0.43466, -19.49703, 0.01865],
                    [1.16572, -19.72447, 0.49027, -19.51112, 0.54533, -19.30472]]
        if self.rawData.ndim == 1:
             offSetCorrection = self.rawData - np.array(bias)
             self.forces = np.dot(userAxis, offSetCorrection)
        elif self.rawData.shape[0] == 6:
             offSetCorrection = self.rawData - np.array(bias)[:, np.newaxis]
             self.forces = np.dot(userAxis, offSetCorrection).mean(axis=1)
        else:
             print(f"경고: rawData 형태({self.rawData.shape})가 예상과 다릅니다.")
             self.forces = np.zeros(6)
        return self.forces
    
    def readFT(self):
        if self.task is None: return np.zeros(6)
        try:
            voltages = np.array(self.task.read(number_of_samples_per_channel=self.Nsamples))
            if voltages.ndim < 2 or voltages.shape[1] == 0: 
                 print("경고: FT 센서에서 유효한 샘플을 읽지 못했습니다.")
                 self.rawData = np.zeros(6)
            else:
                 self.rawData = np.mean(voltages, axis=1)
            return self.convertingRawData()
        except nidaqmx.errors.DaqReadError as e:
            print(f"FT 센서 읽기 오류: {e}")
            return np.zeros(6) 
        
    def calibration(self, second=0.5):
        if self.task is None:
            print("FT 센서가 초기화되지 않아 영점 조절을 건너<0xEB><0x9B><0x84>니다.")
            self.current_offset = np.zeros(6)
            return
        print(f"  > {second}초 동안 영점 조절 시작...")
        start_time = time()
        collected_data = []
        read_count = 0
        while time() - start_time < second:
            ft_value = self.readFT()
            read_count += 1
            collected_data.append(ft_value)
            sleep(max(0.01, float(self.Nsamples) / self.Ratesamples + 0.005))

        print(f"  > 영점 조절 중 {read_count}번 읽기 시도.")
        if collected_data:
            self.current_offset = np.mean(collected_data, axis=0)
            print(f"  > 영점 조절 완료. Offset: {np.round(self.current_offset, 3)}")
        else:
            print("경고: 영점 조절 중 유효 데이터를 얻지 못했습니다. 이전 오프셋 유지.")

    def close(self):
        if self.task is None: return
        try:
            self.task.stop()
            self.task.close()
            print("FT Sensor task closed.")
        except nidaqmx.errors.DaqError as e:
            # 닫기 오류는 경고만 출력하고 계속 진행
            print(f"경고: FT 센서 닫기 오류 (무시): {e}")

current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()
feed_thread_running = True # <<< 스레드 종료 플래그 추가

def ConnectRobot(ip, dashboardPort=29999, movePort=30003, feedPort=30004):
    print(f"Connecting to robot at {ip}...")
    try:
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print("Connection successful.")
        return dashboard, move, feed
    except Exception as e:
        print(f"로봇 연결 실패: {e}")
        return None, None, None