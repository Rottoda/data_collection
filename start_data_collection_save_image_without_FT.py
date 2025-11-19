import sys
import os
import threading
import pandas as pd
import numpy as np
from time import sleep, time
import cv2
from datetime import datetime

# ============================== 설정값 ===============================
CONFIG = {
    "radi" : 350.90,
    "camera_index": 0,
    "robot_speed": 50,
    "safe_height_offset": 20.0,
    "robot_ip": "192.168.1.6",
    "ft_samples": 100,
    "ft_rate": 1000,
    "ft_time": 1.0
}
# =====================================================================

# --- 필요한 모듈 임포트 ---
# nidaqmx가 설치되지 않았을 경우를 대비하여 try...except로 감쌈
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, Edge
    NIDAQMX_AVAILABLE = True
except ImportError:
    NIDAQMX_AVAILABLE = False
    print("경고: nidaqmx 라이브러리를 찾을 수 없습니다. FT 센서 기능이 비활성화됩니다.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TCP_IP_4Axis_Python.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile

# ==================================================================
#                       FT 센서 클래스 (수정 없음)
# ==================================================================
class FT_NI:
    def __init__(self,**kwargs):
        if not NIDAQMX_AVAILABLE:
            raise ImportError("nidaqmx 라이브러리가 없어 FT 센서를 초기화할 수 없습니다.")
        self.Nsamples = kwargs['samples']
        self.Ratesamples = kwargs['rate']
        self.task=nidaqmx.Task()
        self.current_offset = np.zeros(6)
        self.FTsetup()
        print("FT Sensor Initialized.")

    def FTsetup(self):
        # 이 함수는 nidaqmx가 존재할 때만 호출되므로 내부 try-except는 유지
        try:
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:5")
            self.task.timing.cfg_samp_clk_timing(self.Ratesamples, source="", active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE, samps_per_chan=100)
        except nidaqmx.errors.DaqError as e:
            # sys.exit() 대신 예외를 발생시켜 상위에서 처리하도록 변경
            raise ConnectionError(f"FT 센서 설정 실패. 장치 연결을 확인하세요. ({e})")

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
        try:
            self.task.start()
            self.voltages = self.task.read(self.Nsamples)
            self.task.stop()
            self.rawData = np.mean(self.voltages,axis=1)
            self.convertingRawData()
            return self.forces
        except nidaqmx.errors.DaqError:
            self.task.stop()
            return np.full(6, np.nan) # 오류 시 NaN 값으로 채워진 배열 반환

    def calibration(self, second=0.5):
        print(f"  > {second}초 동안 영점 조절..."); start_time = time()
        count = 0; stacked_offset = np.zeros(6)
        while time() - start_time < second:
            stacked_offset += self.readFT(); count += 1
        if count > 0: self.current_offset = stacked_offset / count
        else: print("경고: 영점 조절 중 데이터를 읽지 못했습니다."); self.current_offset = np.zeros(6)
        
    def readFT_calibrated(self):
        return self.readFT() - self.current_offset

    def close(self):
        self.task.close(); print("FT Sensor task closed.")

current_actual = None; algorithm_queue = None; enableStatus_robot = None; robotErrorState = False; globalLockValue = threading.Lock()

def ConnectRobot():
    ip = CONFIG['robot_ip']; dashboardPort = 29999; movePort = 30003; feedPort = 30004
    print("Connecting to robot..."); dashboard = DobotApiDashboard(ip, dashboardPort); move = DobotApiMove(ip, movePort); feed = DobotApi(ip, feedPort); print("Connection successful.")
    return dashboard, move, feed

def GetFeed(feed: DobotApi):
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState; hasRead = 0
    while True:
        data = bytes();
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0: hasRead += len(temp); data += temp
        hasRead = 0; feedInfo = np.frombuffer(data, dtype=MyType)
        if hex(feedInfo['test_value'][0]) == '0x123456789abcdef':
            with globalLockValue:
                current_actual = feedInfo["tool_vector_actual"][0]; algorithm_queue = feedInfo['isRunQueuedCmd'][0]; enableStatus_robot = feedInfo['EnableStatus'][0]; robotErrorState = feedInfo['ErrorStatus'][0]
        sleep(0.001)

def WaitArrive(target_point):
    global current_actual
    while True:
        is_arrive = True
        with globalLockValue:
            if current_actual is not None:
                for i in range(4):
                    if abs(current_actual[i] - target_point[i]) > 1: is_arrive = False; break
                if is_arrive: return
        sleep(0.001)

def CaptureImg(cap, origin_dir, bin_dir, index):
    for _ in range(3):
        ret, frame = cap.read()
        if ret: break
        sleep(0.1)
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        img_filename = f"{index:05d}.png"; orig_path = os.path.join(origin_dir, img_filename); bin_path = os.path.join(bin_dir, img_filename)
        cv2.imwrite(orig_path, frame); cv2.imwrite(bin_path, binarized); print(f"  > 이미지 저장 완료: {img_filename}")
    else: print("카메라 프레임 캡처에 실패했습니다.")


if __name__ == "__main__":
    # --- 1. 기본 설정 및 디렉토리 생성 ---
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); session_dir = os.path.join(script_dir, "collected_images", f"session_{timestamp}"); origin_dir = os.path.join(session_dir, "origin"); bin_dir = os.path.join(session_dir, "bin")
    os.makedirs(origin_dir, exist_ok=True); os.makedirs(bin_dir, exist_ok=True); print(f"[INFO] 이미지 저장 디렉토리: {session_dir}")

    # --- 2. STL 기반 좌표 데이터 로드 ---
    try:
        data_gen_dir = os.path.join(script_dir, "generated_points"); all_sessions = sorted([d for d in os.listdir(data_gen_dir) if os.path.isdir(os.path.join(data_gen_dir, d)) and d.startswith("session_")])
        if not all_sessions: raise FileNotFoundError("세션 폴더 없음")
        latest_session_dir = all_sessions[-1]; print(f"[INFO] 가장 최신 좌표 데이터 세션 로드: {latest_session_dir}")
        csv_path = os.path.join(data_gen_dir, latest_session_dir, "generated_points.csv"); df = pd.read_csv(csv_path); absolute_points = df[['x', 'y', 'z']].values
        print(f"[INFO] '{os.path.basename(csv_path)}'에서 {len(absolute_points)}개의 좌표를 로드했습니다.")
    except Exception as e:
        print(f"ERROR: 좌표 데이터를 로드할 수 없습니다. ({e})"); sys.exit()

    # --- 3. 하드웨어 연결 ---
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened(): print("ERROR: 카메라를 열 수 없습니다."); sys.exit()
    dashboard, move, feed = ConnectRobot()
    
    FT = None # FT 객체를 먼저 None으로 초기화
    try:
        FT = FT_NI(samples=CONFIG["ft_samples"], rate=CONFIG["ft_rate"])
    except (ImportError, ConnectionError, NameError) as e:
        print(f"경고: FT 센서를 초기화할 수 없습니다. Force 데이터 없이 진행합니다. ({e})")


    dashboard.EnableRobot(); print("[INFO] 로봇 활성화 완료.")
    threading.Thread(target=GetFeed, args=(feed,), daemon=True).start()
    
    # --- 4. 데이터 수집 루프 시작 ---
    all_data = []; target_safe = None
    try:
        user, tool, speed = 1, 1, CONFIG["robot_speed"]
        print("\n[INFO] 데이터 수집을 시작합니다...")
        for i, point in enumerate(absolute_points):
            x, y, z = point
            
            if i == 0:
                center_x, center_y = absolute_points[:, 0].mean(), absolute_points[:, 1].mean(); safe_center_z = absolute_points[:, 2].max() + CONFIG["safe_height_offset"]
                initial_target = [center_x, center_y, safe_center_z, CONFIG["radi"]]; move.MovL(*initial_target, user, tool, speed); WaitArrive(initial_target); sleep(CONFIG["ft_time"])
                if FT: FT.calibration() # FT 센서가 있을 때만 캘리브레이션

            print(f"--- [{i+1}/{len(absolute_points)}] 번째 포인트 처리 중 ---")
            
            target_safe = [x, y, z + CONFIG["safe_height_offset"], CONFIG["radi"]]; print("  > 안전 위치로 이동..."); move.MovL(*target_safe, user, tool, speed); WaitArrive(target_safe)
            if FT: FT.calibration() # FT 센서가 있을 때만 캘리브레이션

            target_press = [x, y, z, CONFIG["radi"]]; print("  > 목표 지점 누르기..."); move.MovL(*target_press, user, tool, speed); WaitArrive(target_press); sleep(CONFIG["ft_time"])
            
            CaptureImg(cap, origin_dir, bin_dir, i)

            if FT:
                ft_data = FT.readFT_calibrated()
                print(f"  > FT 데이터 측정: [Fx, Fy, Fz] = [{ft_data[0]:.3f}, {ft_data[1]:.3f}, {ft_data[2]:.3f}]")
            else:
                # 센서가 없으면 NaN 값으로 채움
                ft_data = np.full(3, np.nan)
                print("  > FT 센서 없음: Force 데이터를 NaN으로 기록합니다.")
            
            
            original_row = df.iloc[i].to_dict(); original_row['Fx'] = ft_data[0]; original_row['Fy'] = ft_data[1]; original_row['Fz'] = ft_data[2]; all_data.append(original_row)
            
            print("  > 안전 위치로 복귀 중..."); move.MovL(*target_safe, user, tool, speed); WaitArrive(target_safe)

        print("\n[SUCCESS] 모든 포인트에 대한 데이터 수집을 완료했습니다.")
    except KeyboardInterrupt:
        print("\n[STOP] 사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        # --- 5. 종료 처리 ---
        print("[INFO] 로봇 및 센서를 종료합니다.")
        if all_data:
            final_df = pd.DataFrame(all_data); output_path = os.path.join(session_dir, "full_data.csv")
            final_df.to_csv(output_path, index=False); print(f"  > 최종 통합 데이터 저장 완료: {output_path}")

        dashboard.DisableRobot(); print("  > 로봇 비활성화 완료.")

        if FT: 
            FT.close()

        if target_safe is not None: WaitArrive(target_safe)
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows(); print("  > 프로그램 종료.")