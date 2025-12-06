import trimesh
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# =========================== 설정값 ============================
CONFIG = {
    # STL 파일의 전체 경로
    "stl_file_path": "tactip.stl",
    
    # 생성할 총 포인트 개수
    "n_points": 100,

    # 로봇 좌표계 오프셋
    "robot_origin_offset": np.array([340, 5, -92.66]),

    # 샘플링할 모델의 축소 비율 (0.8 = 80%)
    "xy_sampling_scale": 0.83375,

    # 중앙 집중 강도
    "central_focus_strength": 0.15,

    # 누르는 깊이 범위 (mm)
    "min_press_depth_mm": 2.0,
    "max_press_depth_mm": 2.5,

    # === 2점 인덴터 설정 ===
    "fixed_indenter_distance_mm": 10.5, 
    "fixed_indenter_angle_deg": 0.0,
    
}
# =================================================================