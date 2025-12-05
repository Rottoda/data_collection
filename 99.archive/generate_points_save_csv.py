import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# 중심점
CENTER_3D = np.array([338.04, 4.18, -135.66])
DEPTH_Z = -2.5

# 데이터 개수
N_POINTS = 200

# 타원 정의
xy_points = np.array([
    [338.03, 14.18],
    [338.51, -5.29],
    [347.19, 13.46],
    [346.07, -3.78],
    [325.40, 4.43],
    [330.78, 14.81],
    [332.11, -4.83],
    [350.55, 4.46]
])

# 타원 중심 및 축 길이
x_mean = np.mean(xy_points[:, 0])
y_fixed = 4.18
center = np.array([x_mean, y_fixed])
x_range = xy_points[:, 0].max() - xy_points[:, 0].min()
y_range = xy_points[:, 1].max() - xy_points[:, 1].min()
scale_factor = 0.8
a = (x_range * 1.1 * scale_factor) / 2
b = (y_range * 1.1 * scale_factor) / 2

# 타원 내부 판별 함수
def is_inside_ellipse(x, y):
    return ((x - center[0])**2) / a**2 + ((y - center[1])**2) / b**2 <= 1

# 랜덤 포인트 생성 함수
def generate_random_points_in_ellipse(n_points, z_min=CENTER_3D[2]-DEPTH_Z, z_max=CENTER_3D[2]):
    samples = []
    x_min, x_max = center[0] - a, center[0] + a
    y_min, y_max = center[1] - b, center[1] + b

    while len(samples) < n_points:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if is_inside_ellipse(x, y):
            z = random.uniform(z_min, z_max)
            samples.append([x, y, z])
    
    return np.array(samples)

# 무작위 포인트 및 거리 계산
random_points = generate_random_points_in_ellipse(N_POINTS)
distances = np.linalg.norm(random_points - CENTER_3D, axis=1)

# 상대 좌표 계산 및 저장
relative_points = random_points - CENTER_3D
df_relative = pd.DataFrame(relative_points, columns=["dX", "dY", "dZ"])

# 현재 실행 파일과 같은 경로에 저장
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relative_random_points.csv")
df_relative.to_csv(output_path, index=False)
print(f"CSV 저장 완료: {output_path}")

# 3D 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2],
                c=distances, cmap='viridis')
ax.scatter(CENTER_3D[0], CENTER_3D[1], CENTER_3D[2], c='red', s=50, label='Center')
plt.colorbar(sc, label='Distance from Center')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Random Points Colored by Distance from Center")
ax.legend()
plt.tight_layout()
plt.show()
