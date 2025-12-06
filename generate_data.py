import trimesh
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from datetime import datetime

# =========================== 설정값 ============================
CONFIG = {
    # STL 파일의 전체 경로
    "stl_file_path": "tactip.stl",
    
    # 생성할 총 포인트 개수
    "n_points": 5000,

    # 로봇 좌표계 오프셋 (STL의 원점(x,y,z)에 해당하는 로봇의 실제 좌표)
    # 구버전은 [339, 6, -112] 이었음
    # 현재는 [340, 5, -109]로 변경됨
    "robot_origin_offset": np.array([340, 5, -109]),

    # 샘플링할 모델의 축소 비율 (0.8 = 80%)
    "xy_sampling_scale": 0.8,

    # 중앙 집중 강도 - 값이 작을수록 중앙에 더 강하게 집중 (예: 0.1)
    "central_focus_strength": 0.15,

    # 누르는 깊이 범위 (mm)
    "min_press_depth_mm": 0.5,
    "max_press_depth_mm": 6.0 
}
# =================================================================

def visualize_results(mesh, absolute_points, origin_offset, save_path):
    """생성된 결과를 3D 그래프로 시각화하는 함수"""
    print("결과를 3D로 시각화합니다...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_vertices_local = mesh.vertices
    absolute_points_local = absolute_points - origin_offset

    ax.add_collection3d(plt.tripcolor(
        mesh_vertices_local[:, 0], mesh_vertices_local[:, 1], mesh_vertices_local[:, 2],
        triangles=mesh.faces, facecolor='cyan', alpha=0.1, edgecolor='gray', linewidth=0.1))
    ax.scatter(
        absolute_points_local[:, 0], absolute_points_local[:, 1], absolute_points_local[:, 2],
        c='blue', s=10, label=f'Generated Press Points ({len(absolute_points)} points)')
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

def main():
    """메인 실행 함수"""
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

        # Tactip의 '바닥' Z좌표를 찾습니다.
        # 모델의 경계 상자(bounds)에서 가장 낮은 Z값 [0][2]를 가져옵니다.
        Z_BASE = mesh_original.bounds[0][2] 
        print(f"[INFO] 모델의 '바닥' Z 높이를 {Z_BASE:.2f} (으)로 설정합니다.")
        
        mesh_scaled = mesh_original.copy()
        mesh_scaled.apply_scale(CONFIG['xy_sampling_scale'])
        print(f"모델 생성 완료.")

        # --- 3. 가우시안 분포를 이용한 중앙 집중형 포인트 생성 ---
        # 모델의 중심점과 크기(표준편차 계산용)를 구함
        center = mesh_scaled.centroid
        extents = mesh_scaled.extents

        # 집중 강도를 반영한 표준편차 계산
        std_dev_x = extents[0] * CONFIG['central_focus_strength']
        std_dev_y = extents[1] * CONFIG['central_focus_strength']

        # 가우시안 분포로 X, Y 좌표 생성
        rand_x = np.random.normal(loc=center[0], scale=std_dev_x, size=CONFIG['n_points'])
        rand_y = np.random.normal(loc=center[1], scale=std_dev_y, size=CONFIG['n_points'])
        
        # Z좌표는 임시로 중심 Z값으로 설정
        query_points = np.vstack([rand_x, rand_y, np.full(CONFIG['n_points'], center[2])]).T

        # 생성된 (X,Y) 점에서 가장 가까운 표면의 (X,Y,Z)를 찾음
        surface_points_scaled, _, _ = mesh_scaled.nearest.on_surface(query_points)
        print(f"{len(surface_points_scaled)}개의 중앙 집중형 표면 좌표 생성 완료.")

        # --- 4. 로봇이 누를 '절대 좌표' 및 학습용 '상대 좌표' 계산 ---
        robot_target_points = []
        relative_points = []

        for surface_pt in surface_points_scaled:
            random_depth = random.uniform(CONFIG['min_press_depth_mm'], CONFIG['max_press_depth_mm'])
            press_point_scaled = surface_pt - np.array([0, 0, random_depth])
            robot_target_points.append(press_point_scaled + CONFIG['robot_origin_offset'])
            relative_points.append([surface_pt[0], surface_pt[1], -random_depth])

        robot_target_points = np.array(robot_target_points)
        print("절대/상대 좌표 계산 완료.")

        # --- 5. CSV 파일로 저장 ---
        df_robot = pd.DataFrame(robot_target_points, columns=["x", "y", "z"])
        df_relative = pd.DataFrame(relative_points, columns=["dX", "dY", "dZ"])
        final_df = pd.concat([df_robot, df_relative], axis=1)

        try:
            script_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_path = os.getcwd()

        output_csv_path = os.path.join(session_dir, "generated_points.csv")
        final_df.to_csv(output_csv_path, index=False)
        print(f"CSV 저장 완료: {output_csv_path}")

        # --- 7. 3D 시각화 ---
        output_graph_path = os.path.join(session_dir, "points_distribution.png")
        visualize_results(mesh_scaled, robot_target_points[:, :3], CONFIG['robot_origin_offset'], output_graph_path)

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print("STL 파일 경로, 라이브러리 설치 상태를 확인해주세요.")

if __name__ == '__main__':
    main()