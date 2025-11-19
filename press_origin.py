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
    
def GetFeed(feed: DobotApi):
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState, feed_thread_running # 종료 플래그 사용
    hasRead = 0
    feed.socket_dobot.settimeout(1.0)

    print("피드백 스레드 시작.")
    while feed_thread_running:
        data = bytes()
        try:
            while hasRead < 1440 and feed_thread_running:
                try:
                    temp = feed.socket_dobot.recv(1440 - hasRead)
                except socket.timeout:
                    sleep(0.1) 
                    continue 
                except ConnectionAbortedError: 
                     print("피드백 소켓 연결 종료됨 (Aborted).")
                     with globalLockValue: robotErrorState = True
                     break
                except OSError as oe:
                     print(f"피드 소켓 오류: {oe}")
                     with globalLockValue: robotErrorState = True
                     break

                if len(temp) == 0:
                    print("로봇 피드 연결 끊김.")
                    with globalLockValue:
                        current_actual = None; enableStatus_robot = None; robotErrorState = True
                    break
                hasRead += len(temp)
                data += temp

            if not feed_thread_running or robotErrorState: 
                 break
            if hasRead == 0:
                sleep(0.5)
                continue

            hasRead = 0
            if len(data) == 1440: 
                feedInfo = np.frombuffer(data, dtype=MyType)
                if 'test_value' in feedInfo.dtype.names and \
                   hex(feedInfo['test_value'][0]) == '0x123456789abcdef':
                    with globalLockValue:
                        current_actual = feedInfo["tool_vector_actual"][0]
                        algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                        enableStatus_robot = feedInfo['EnableStatus'][0]
                        
                else:
                    pass 
            else:
                pass 

            sleep(0.02)

        except BlockingIOError:
            sleep(0.05)
        except ConnectionResetError:
            print("로봇 피드 연결 재설정됨.")
            with globalLockValue: current_actual=None; enableStatus_robot=None; robotErrorState=True
            sleep(2)
        except Exception as e:
            print(f"피드 수신 중 예상치 못한 예외 발생: {e}")
            with globalLockValue: current_actual=None; enableStatus_robot=None; robotErrorState=True
            sleep(1)

    print("피드백 스레드 종료.")

def WaitArrive(target_point):
    global current_actual, robotErrorState
    start_wait_time = time()
    max_wait_time = 30
    last_pos_print_time = time()
    print(f"  > 목표 지점 {np.round(target_point[:3], 1)} 도착 대기 시작...")
    while True:
        current_time = time()
        if current_time - start_wait_time > max_wait_time:
             print(f"경고: 목표 {np.round(target_point[:3], 1)} 도착 대기 시간 초과!")
             return False

        with globalLockValue:
            if robotErrorState:
                 error_id = -1 
                 print(f"오류: 로봇 오류 상태 감지됨 (ID: {error_id}). 이동 중단.")
                 return False

            current_pos_local = current_actual

        if current_pos_local is None:
            is_arrive = False
        else:
            current_pos = current_pos_local[:3]
            target_pos = target_point[:3]
            diff = np.abs(current_pos - target_pos)
            is_arrive = np.all(diff <= 1.5)

            if current_time - last_pos_print_time > 1.0:
                print(f"  > 현재: {np.round(current_pos, 1)}, 목표: {np.round(target_pos, 1)}, 오차: {np.round(diff, 1)}")
                last_pos_print_time = current_time

            if is_arrive:
                print(f"도착 확인: {np.round(current_pos, 1)}")
                return True

        sleep(0.1)

def calculate_origin_target(config):
    stl_path = config['stl_file_path']
    robot_offset = config['robot_origin_offset']
    press_depth = config['press_depth_at_origin_mm']

    print(f"\n[INFO] STL 원점 목표 좌표 계산 시작...")
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"오류: STL 파일 '{stl_path}'를 찾을 수 없습니다.")

    stl_origin_local = np.array([0.0, 0.0, 0.0])
    robot_origin_base = stl_origin_local + robot_offset
    robot_target_z = robot_origin_base[2] - press_depth
    robot_target_point = np.array([robot_origin_base[0], robot_origin_base[1], robot_target_z])

    print(f"  > STL 원점 로봇 좌표 (오프셋 적용): {np.round(robot_origin_base, 3)}")
    print(f"  > 누름 깊이: {press_depth:.2f} mm")
    print(f"  > 최종 로봇 목표 좌표 (X, Y, Z): {np.round(robot_target_point, 4)}")
    return robot_target_point

if __name__ == "__main__":
    print("[INFO] 오프셋 검증 모드 시작.")
    dashboard, move, feed, FT = None, None, None, None
    feed_thread = None
    final_position_reached = False
    go_back_to_safe = False
    target_safe = []


    try:
        target_press_xyz = calculate_origin_target(CONFIG)
        target_press = [target_press_xyz[0], target_press_xyz[1], target_press_xyz[2], CONFIG["radi"]]

        dashboard, move, feed = ConnectRobot(CONFIG["robot_ip"])
        if dashboard is None:
            raise ConnectionError("로봇 연결 실패")

        ft_sensor_available = True
        try:
            FT = FT_NI(samples=CONFIG["ft_samples"], rate=CONFIG["ft_rate"])
        except ConnectionError as e:
            print(e); print("경고: FT 센서 없이 진행합니다.")
            ft_sensor_available = False; FT = None