import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 데이터 및 상수 정의 ---

# 1. 포인트 생성 및 상대 좌표 계산을 위한 기준점
CENTER_3D = np.array([338.05, -19.31, -134.46])

# 2. 생성할 포인트 개수
N_POINTS = 300

# 3. Z좌표 생성 시 사용할 깊이 (CENTER_3D의 Z좌표로부터 아래 방향으로의 범위)
DEPTH_Z = 2.5

# 4. 2D 타원 및 3D 타원체 정의에 필요한 포인트 데이터
points_edge = [
    [326.05, -19.31, -138.44], [328.05, -13.31, -137.34], [328.05, -25.31, -138.64],
    [333.05, -10.31, -137.74], [333.05, -28.31, -137.74], [338.05, -10.31, -137.74],
    [338.05, -28.31, -137.74], [343.05, -10.31, -137.74], [343.05, -28.31, -137.74],
    [348.05, -13.31, -137.34], [348.05, -25.31, -138.64], [350.05, -19.31, -138.44]
]
points_inside = [
    [328.05, -16.31, -135.96], [328.05, -19.31, -135.06], [328.05, -22.31, -136.06],
    [333.05, -13.31, -136.34], [333.05, -16.31, -135.34], [333.05, -19.31, -134.46],
    [333.05, -22.31, -135.06], [333.05, -25.31, -136.06], [338.05, -13.31, -136.34],
    [338.05, -16.31, -135.34], [338.05, -22.31, -135.34], [338.05, -25.31, -136.34],
    [343.05, -13.31, -136.34], [343.05, -16.31, -135.34], [343.05, -19.31, -134.96],
    [343.05, -22.31, -135.34], [343.05, -25.31, -136.34], [348.05, -16.31, -138.64],
    [348.05, -19.31, -136.44], [348.05, -22.31, -137.34]
]
points_center_data = [[338.05, -19.31, -134.46]]


# --- 1. 포인트 생성을 위한 2D 타원 정의 (define_ellipse_v2.py 로직) ---
xy_points_for_ellipse = np.array([[p[0], p[1]] for p in points_edge])
ellipse_center_2d = np.array([np.mean(xy_points_for_ellipse[:, 0]), -19.31])
x_range_2d = xy_points_for_ellipse[:, 0].max() - xy_points_for_ellipse[:, 0].min()
y_range_2d = xy_points_for_ellipse[:, 1].max() - xy_points_for_ellipse[:, 1].min()
scale_factor_2d = 0.90
ellipse_a = (x_range_2d * 1.1 * scale_factor_2d) / 2 # 타원의 x축 반지름
ellipse_b = (y_range_2d * 1.1 * scale_factor_2d) / 2 # 타원의 y축 반지름

def is_inside_ellipse(x, y, center, a, b):
    """주어진 (x, y)가 타원 내부에 있는지 확인합니다."""
    return ((x - center[0])**2) / a**2 + ((y - center[1])**2) / b**2 <= 1


# --- 2. 상대 Z좌표 계산을 위한 3D 타원체 정의 (define_ellipsoid.py 로직) ---
all_points = points_edge + points_inside + points_center_data
all_points_np = np.array(all_points)
ellipsoid_center_3d = np.mean(all_points_np, axis=0)
ellipsoid_radii_3d = (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0)) / 2.0

def get_ellipsoid_upper_z(x, y, center, radii):
    """주어진 (x, y)에 해당하는 타원체 상부 표면의 Z좌표를 계산합니다."""
    cx, cy, cz = center
    rx, ry, rz = radii
    
    # Z좌표 계산을 위한 방정식의 일부
    term = 1 - ((x - cx) / rx)**2 - ((y - cy) / ry)**2
    
    # (x,y)가 타원체의 XY 평면 투영 밖에 있으면 계산 불가 (음수의 제곱근)
    if term < 0:
        return np.nan 
    
    z_offset = rz * np.sqrt(term)
    return cz + z_offset


# --- 3. 포인트 생성 및 상대 좌표 계산 ---

# 3.1. 2D 타원 내에서 절대 좌표 포인트 생성
absolute_points = []
x_min_bound, x_max_bound = ellipse_center_2d[0] - ellipse_a, ellipse_center_2d[0] + ellipse_a
y_min_bound, y_max_bound = ellipse_center_2d[1] - ellipse_b, ellipse_center_2d[1] + ellipse_b
z_min_abs = CENTER_3D[2] - DEPTH_Z
z_max_abs = CENTER_3D[2]

print(f"{N_POINTS}개의 포인트를 생성합니다...")
while len(absolute_points) < N_POINTS:
    x_rand = random.uniform(x_min_bound, x_max_bound)
    y_rand = random.uniform(y_min_bound, y_max_bound)
    
    # 생성된 (x, y)가 2D 타원 내부에 있을 때만 포인트를 추가
    if is_inside_ellipse(x_rand, y_rand, ellipse_center_2d, ellipse_a, ellipse_b):
        z_rand = random.uniform(z_min_abs, z_max_abs)
        absolute_points.append([x_rand, y_rand, z_rand])

absolute_points = np.array(absolute_points)
print("포인트 생성 완료.")

# 3.2. 상대 좌표 계산
relative_points = []
for point in absolute_points:
    x_abs, y_abs, z_abs = point
    
    # dX, dY: CENTER_3D 기준
    dx = x_abs - CENTER_3D[0]
    dy = y_abs - CENTER_3D[1]
    
    # dZ: 타원체 상부 표면 기준
    z_surface = get_ellipsoid_upper_z(x_abs, y_abs, ellipsoid_center_3d, ellipsoid_radii_3d)
    if np.isnan(z_surface):
        # (x,y)가 타원체 밖에 있는 경우 (정상적인 경우 발생하지 않음)
        continue
    dz = z_abs - z_surface
    
    relative_points.append([dx, dy, dz])

print("상대 좌표 계산 완료.")

# --- 4. CSV 파일로 저장 ---
df_relative = pd.DataFrame(relative_points, columns=["dX", "dY", "dZ"])
# 현재 스크립트 파일이 있는 위치에 CSV 저장
try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 대화형 환경(예: Jupyter)에서 실행 시 현재 작업 디렉토리 사용
    script_path = os.getcwd()

output_filename = "relative_random_points_v2.csv"
output_path = os.path.join(script_path, output_filename)
df_relative.to_csv(output_path, index=False)
print(f"CSV 저장 완료: {output_path}")


# --- 5. 3D 시각화 ---
print("결과를 시각화합니다...")
# 시각화용 타원체 상부 표면 생성
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 50) # 상단 절반만
u, v = np.meshgrid(u, v)
x_surf = ellipsoid_center_3d[0] + ellipsoid_radii_3d[0] * np.cos(u) * np.sin(v)
y_surf = ellipsoid_center_3d[1] + ellipsoid_radii_3d[1] * np.sin(u) * np.sin(v)
z_surf = ellipsoid_center_3d[2] + ellipsoid_radii_3d[2] * np.cos(v)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 생성된 절대 좌표 포인트 플로팅
ax.scatter(absolute_points[:, 0], absolute_points[:, 1], absolute_points[:, 2],
           c='blue', s=10, label='Generated Absolute Points')
# 타원체 상부 표면 플로팅
ax.plot_surface(x_surf, y_surf, z_surf, color='purple', alpha=0.2, label='Ellipsoid Upper Surface')
# CENTER_3D 기준점 플로팅
ax.scatter(CENTER_3D[0], CENTER_3D[1], CENTER_3D[2], c='red', s=100, marker='x', label='CENTER_3D (Reference)')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Generated Points and Ellipsoid Surface")
ax.legend()
plt.tight_layout()
plt.show()