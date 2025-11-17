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