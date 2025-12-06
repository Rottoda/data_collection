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
    "min_press_depth_mm": 0.5,
    "max_press_depth_mm": 1.0,

    # === [수정] 2점 인덴터 설정 ===
    "fixed_indenter_distance_mm": 12.0,
    "fixed_indenter_angle_deg": 0.0,
    
    "bottom_safety_margin_mm": 1.0
}
# =================================================================

# visualize_results 함수는 이전과 동일 (NaN 처리 기능 유지)
def visualize_results(mesh, press_points_A_scaled, press_points_B_scaled, save_path):
    """
    [수정] NaN 값을 처리하여 3D 그래프로 시각화하는 함수
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
        return locations[0][2] # 첫 번째 충돌 지점의 Z 좌표
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

        # for 루프 사용
        for surface_midpoint in surface_midpoints:
            desired_depth = random.uniform(CONFIG['min_press_depth_mm'], CONFIG['max_press_depth_mm'])

            target_x_A = surface_midpoint[0] + half_dist * np.cos(angle_rad)
            target_y_A = surface_midpoint[1] + half_dist * np.sin(angle_rad)
            target_x_B = surface_midpoint[0] - half_dist * np.cos(angle_rad)
            target_y_B = surface_midpoint[1] - half_dist * np.sin(angle_rad)
            
            target_A_xy = np.array([target_x_A, target_y_A])
            target_B_xy = np.array([target_x_B, target_y_B])

            z_surf_A = get_surface_z_at_xy(mesh_scaled, target_A_xy, ray_start_z)
            z_surf_B = get_surface_z_at_xy(mesh_scaled, target_B_xy, ray_start_z)

            is_A_valid = not np.isnan(z_surf_A)
            is_B_valid = not np.isnan(z_surf_B)

            valid_z_values = []
            if is_A_valid: valid_z_values.append(z_surf_A)
            if is_B_valid: valid_z_values.append(z_surf_B)

            else:
            if not valid_z_values:
                z_surface_highest = surface_midpoint[2]
            else:
                z_surface_highest = max(valid_z_values)

            available_depth = z_surface_highest - Z_BASE - CONFIG["bottom_safety_margin_mm"]
            actual_robot_travel_depth = np.maximum(0, np.minimum(desired_depth, available_depth))
            
            robot_z_scaled = z_surface_highest - actual_robot_travel_depth

            if is_A_valid:
                potential_depth_A = z_surf_A - robot_z_scaled
                max_possible_depth_A = z_surf_A - Z_BASE - CONFIG["bottom_safety_margin_mm"]
                actual_depth_A = np.maximum(0, np.minimum(potential_depth_A, max_possible_depth_A))
                
                rel_A = [target_x_A, target_y_A, -actual_depth_A]
                vis_A = np.array([target_x_A, target_y_A, z_surf_A - actual_depth_A])
            else:
                rel_A = [np.nan, np.nan, np.nan]
                vis_A = np.array([np.nan, np.nan, np.nan])

            if is_B_valid:
                potential_depth_B = z_surf_B - robot_z_scaled
                max_possible_depth_B = z_surf_B - Z_BASE - CONFIG["bottom_safety_margin_mm"]
                actual_depth_B = np.maximum(0, np.minimum(potential_depth_B, max_possible_depth_B))
                
                rel_B = [target_x_B, target_y_B, -actual_depth_B]
                vis_B = np.array([target_x_B, target_y_B, z_surf_B - actual_depth_B])

            relative_points_A_list.append(rel_A)
            relative_points_B_list.append(rel_B)
            visual_press_points_A_list.append(vis_A)
            visual_press_points_B_list.append(vis_B)
            
            robot_x_scaled = surface_midpoint[0] # 로봇은 여전히 중심점으로 이동
            robot_y_scaled = surface_midpoint[1]
            
            valid_z_values = []
            if is_A_valid: valid_z_values.append(z_surf_A)
            if is_B_valid: valid_z_values.append(z_surf_B)
            
            if not valid_z_values: # 둘 다 허공이면
                z_surface_highest = surface_midpoint[2] # 그냥 중심점 Z 사용 (어차피 눌리지 않음)
            else:
                z_surface_highest = max(valid_z_values) # 유효한 점 중 높은 Z
            
            robot_z_scaled = z_surface_highest - depth
            
            press_point_scaled = np.array([robot_x_scaled, robot_y_scaled, robot_z_scaled])
            robot_target_points_list.append(press_point_scaled + CONFIG['robot_origin_offset'])

        robot_target_points = np.array(robot_target_points_list)
        relative_points_A = np.array(relative_points_A_list)
        relative_points_B = np.array(relative_points_B_list)
        df_robot = pd.DataFrame(robot_target_points, columns=["x", "y", "z"])
        df_rel_A = pd.DataFrame(relative_points_A, columns=["rel_x1", "rel_y1", "rel_z1"])
        df_rel_B = pd.DataFrame(relative_points_B, columns=["rel_x2", "rel_y2", "rel_z2"])
        final_df = pd.concat([df_robot, df_rel_A, df_rel_B], axis=1)
        output_csv_path = os.path.join(session_dir, "generated_points.csv")
        final_df.to_csv(output_csv_path, index=False)
        print(f"CSV 저장 완료: {output_csv_path}")
        print("저장된 CSV 컬럼:")
        print(final_df.columns.to_list())
        nan_count_A = final_df['rel_x1'].isna().sum()
        nan_count_B = final_df['rel_x2'].isna().sum()
        print(f"[INFO] A 이탈(NaN) 개수: {nan_count_A} / {len(final_df)}")
        print(f"[INFO] B 이탈(NaN) 개수: {nan_count_B} / {len(final_df)}")

        
        visual_press_points_A = np.array(visual_press_points_A_list)
        visual_press_points_B = np.array(visual_press_points_B_list)
        output_graph_path = os.path.join(session_dir, "points_distribution.png")
        visualize_results(mesh_scaled, visual_press_points_A, visual_press_points_B, output_graph_path)

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print(traceback.format_exc()) # 더 자세한 오류 출력
        print("STL 파일 경로, 라이브러리 설치 상태를 확인해주세요.")

if __name__ == '__main__':
    main()