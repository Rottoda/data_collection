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
