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