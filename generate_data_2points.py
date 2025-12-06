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

# Ray Casting을 사용하여 특정 XY 좌표 아래의 표면 Z값을 찾는 함수
def get_surface_z_at_xy(mesh, xy_point, default_z, z_offset=5):
    """
    주어진 XY 좌표 바로 아래의 메쉬 표면 Z값을 Ray Casting으로 찾습니다.
    mesh: trimesh 객체
    xy_point: [x, y] numpy 배열
    default_z: 광선 시작점 계산을 위한 기본 Z 높이 (예: mesh.centroid[2])
    z_offset: 광선 시작점을 default_z보다 얼마나 위에서 시작할지 (충돌 방지)
    반환값: 표면 Z값 (못 찾으면 np.nan)
    """
    ray_origin = np.array([xy_point[0], xy_point[1], default_z + z_offset])
    ray_direction = np.array([0, 0, -1])
    
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin], 
        ray_directions=[ray_direction]
    )
    
    if len(locations) > 0:
        return locations[0][2] 
    else:
        return np.nan 

def main():
    """
    Ray Casting을 사용하여 고정 거리 인덴터의 접촉점을 계산
    """
    try:
        # --- 1. 세션 디렉토리 생성 ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(script_dir, "generated_points", f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        print(f"[INFO] 생성된 데이터 저장 경로: {session_dir}")

        # --- 2. STL 파일 로드 및 축소 모델 생성 ---
        print(f"'{CONFIG['stl_file_path']}' 파일을 로드합니다...")
        mesh_original = trimesh.load_mesh(CONFIG['stl_file_path'])
        mesh_scaled = mesh_original.copy()
        mesh_scaled.apply_scale(CONFIG['xy_sampling_scale'])
        print(f"샘플링용 모델 생성 완료.")

        # Ray Casting을 위한 메쉬 바운딩 박스 정보 미리 계산
        min_bound, max_bound = mesh_scaled.bounds
        ray_start_z = max_bound[2] + 5 # 메쉬 최고점보다 5mm 위에서 레이 시작

        # --- 3. 가우시안 분포를 이용한 중심점 생성 ---
        center = mesh_scaled.centroid
        extents = mesh_scaled.extents
        std_dev_x = extents[0] * CONFIG['central_focus_strength']
        std_dev_y = extents[1] * CONFIG['central_focus_strength']
        rand_x_mid = np.random.normal(loc=center[0], scale=std_dev_x, size=CONFIG['n_points'])
        rand_y_mid = np.random.normal(loc=center[1], scale=std_dev_y, size=CONFIG['n_points'])
        query_midpoints = np.vstack([rand_x_mid, rand_y_mid, np.full(CONFIG['n_points'], center[2])]).T
        surface_midpoints, _, _ = mesh_scaled.nearest.on_surface(query_midpoints)
        print(f"{len(surface_midpoints)}개의 중심점 좌표 생성 완료.")

        # --- 4. Ray Casting으로 2개의 접촉점 및 로봇 좌표 계산 ---
        robot_target_points_list = []
        relative_points_A_list = [] 
        relative_points_B_list = [] 
        visual_press_points_A_list = []
        visual_press_points_B_list = []

        angle_rad = np.deg2rad(CONFIG['fixed_indenter_angle_deg'])
        distance = CONFIG['fixed_indenter_distance_mm']
        half_dist = distance / 2.0