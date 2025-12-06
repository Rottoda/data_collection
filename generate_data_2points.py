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

# visualize_results 함수는 이전과 동일 (NaN 처리 기능 유지)
def visualize_results(mesh, press_points_A_scaled, press_points_B_scaled, save_path):
    """
    NaN 값을 처리하여 3D 그래프로 시각화하는 함수
    """
    print("결과를 3D로 시각화합니다...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_vertices_local = mesh.vertices

    valid_A = press_points_A_scaled[~np.isnan(press_points_A_scaled).any(axis=1)]
    valid_B = press_points_B_scaled[~np.isnan(press_points_B_scaled).any(axis=1)]

    ax.add_collection3d(plt.tripcolor(
        mesh_vertices_local[:, 0], mesh_vertices_local[:, 1], mesh_vertices_local[:, 2],
        triangles=mesh.faces, facecolor='cyan', alpha=0.1, edgecolor='gray', linewidth=0.1
    ))
    
    ax.scatter(
        valid_A[:, 0], valid_A[:, 1], valid_A[:, 2],
        c='blue', s=10, label=f'Generated Press Points A ({len(valid_A)} valid points)'
    )
    ax.scatter(
        valid_B[:, 0], valid_B[:, 1], valid_B[:, 2],
        c='green', s=10, label=f'Generated Press Points B ({len(valid_B)} valid points)'
    )
    
    ax.scatter(0, 0, 0, c='red', s=150, marker='x', label='STL Origin')
    ax.set_xlabel("X (mm)", fontsize=24); ax.set_ylabel("Y (mm)", fontsize=24); ax.set_zlabel("Z (mm)", fontsize=24)
    ax.set_title(f"Generated 2-Point Pairs on {CONFIG['xy_sampling_scale']*100}% Scaled Model", fontsize=36)
    ax.legend()
    
    axis_limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(axis_limits, axis=1))
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"그래프 이미지 저장 완료: {save_path}")
    plt.show()
