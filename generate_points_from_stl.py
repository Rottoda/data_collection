import trimesh
import pandas as pd
import numpy as np
import random

# --- 설정값 ---
stl_file_path = 'C:/Users/PC/Desktop/Lee/Rottoda_TacTip/data_collection/tactip.stl'
output_csv_path = 'hybrid_approach_points.csv'
num_points = 500
safe_approach_distance_mm = 3.0
scale_factor_for_xy = 0.75 # X,Y 위치를 샘플링할 모델의 축소 비율

# --------------------------------------------------------------------

print(f"'{stl_file_path}' 파일을 로드합니다...")
try:
    # 1. 원본 100% 크기 모델 로드
    mesh_100 = trimesh.load_mesh(stl_file_path)
    print("원본(100%) 모델 로드 완료.")

    # 2. X, Y 좌표 샘플링을 위한 축소 모델 생성
    mesh_scaled = mesh_100.copy()
    mesh_scaled.apply_scale(scale_factor_for_xy)
    print(f"샘플링용 {scale_factor_for_xy*100}% 축소 모델 생성 완료.")

    # 3. 축소된 모델 표면에서 점 샘플링
    sampled_points_scaled, _ = trimesh.sample.sample_surface(mesh_scaled, num_points)
    print(f"{num_points}개의 위치(X,Y) 샘플링 완료.")

    # 4. 원본 모델과의 교차점을 찾기 위한 Ray-Mesh 교차 검사기 준비
    intersector = trimesh.ray.RayMeshIntersector(mesh_100)

    # 샘플링된 점들의 X, Y 좌표만 사용
    ray_origins = sampled_points_scaled.copy()
    ray_origins[:, 2] = mesh_100.bounds[1, 2] + 10 # 모델의 가장 높은 Z값보다 10mm 위에서 레이저를 쏨
    
    # 레이저 방향은 Z축 아래 방향 (-Z)
    ray_directions = np.array([[0, 0, -1]] * num_points)

    # 5. 레이캐스팅 실행: 광선을 쏴서 원본(100%) 모델 표면과의 교차점(실제 Z값)을 찾음
    locations, _, _ = intersector.intersects_location(ray_origins, ray_directions)
    print("레이캐스팅으로 원본 표면의 실제 Z좌표 계산 완료.")
    
    final_points = locations.copy()

    # 좌표계 오프셋 적용
    offset = np.array([339, 6, -135.3])
    final_points += offset
    print("좌표계 오프셋 적용 완료.")

    # 모든 점을 Z축으로 safe_approach_distance_mm 만큼 위로 이동
    final_points[:, 2] += safe_approach_distance_mm
    print(f"안전 접근점 생성 완료.")

    # CSV 파일로 저장
    df = pd.DataFrame(final_points, columns=['x', 'y', 'z'])
    df.to_csv(output_csv_path, index=False)
    print(f"성공적으로 '{output_csv_path}' 파일에 하이브리드 좌표를 저장했습니다.")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")