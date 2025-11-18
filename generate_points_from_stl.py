import trimesh
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# ==================== 설정값 (여기만 수정하세요) ====================
CONFIG = {
    # 1. STL 파일의 전체 경로
    "stl_file_path": "tactip.stl",
    
    # 2. 생성할 CSV 파일 이름
    "output_csv_filename": "robot_press_points.csv",
    
    # 3. 생성할 총 포인트 개수
    "n_points": 5000,

    # 4. 로봇 좌표계 오프셋 (STL의 원점(x,y,z)에 해당하는 로봇의 실제 좌표)
    "robot_origin_offset": np.array([339, 6, -112]),

    # 5. 샘플링할 모델의 축소 비율 (0.8 = 80%)
    "xy_sampling_scale": 0.8,

    # 6. 중앙 집중 강도 
    # 값이 작을수록 중앙에 더 강하게 집중 (예: 0.1)
    "central_focus_strength": 0.1,

    # 7. 수동 Z축 보정값 (mm)
    "manual_z_correction": -5.0,

    # 8. 누르는 깊이 범위 (mm)
    "min_press_depth_mm": 4.0, # 보정값과 9 이상 차이를 추천
    "max_press_depth_mm": 6.0  # 보정값과 절대 15이상 벗어나지말 것
}
# =================================================================

def visualize_results(mesh, absolute_points, origin_offset, save_path):
    """생성된 결과를 3D 그래프로 시각화하는 함수"""
    print("결과를 3D로 시각화합니다...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_vertices_local = mesh.vertices
    absolute_points_local = absolute_points - (origin_offset + np.array([0,0,CONFIG['manual_z_correction']]))

    ax.add_collection3d(plt.tripcolor(
        mesh_vertices_local[:, 0], mesh_vertices_local[:, 1], mesh_vertices_local[:, 2],
        triangles=mesh.faces, facecolor='cyan', alpha=0.1, edgecolor='gray', linewidth=0.1
    ))
    ax.scatter(
        absolute_points_local[:, 0], absolute_points_local[:, 1], absolute_points_local[:, 2],
        c='blue', s=10, label=f'Generated Press Points ({len(absolute_points)} points)'
    )
    ax.scatter(0, 0, 0, c='red', s=150, marker='x', label='STL Origin')
    ax.set_xlabel("X (mm)", fontsize=24); ax.set_ylabel("Y (mm)", fontsize=24); ax.set_zlabel("Z (mm)", fontsize=24)
    ax.set_title(f"Generated Points on {CONFIG['xy_sampling_scale']*100}% Scaled STL Model", fontsize=36)
    ax.legend()
    
    axis_limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(axis_limits, axis=1))
    plt.tight_layout()

    # 그래프를 파일로 저장
    plt.savefig(save_path)
    print(f"그래프 이미지 저장 완료: {save_path}")
    plt.show(block=False)
