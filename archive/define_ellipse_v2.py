import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 좌표 입력 (z 좌표는 무시)
points_edge = [[326.05, -19.31, -138.44],
               
               [328.05, -13.31, -137.34],
               [328.05, -25.31, -138.64],

               [333.05, -10.31, -137.74],
               [333.05, -28.31, -137.74],

               [338.05, -10.31, -137.74],
               [338.05, -28.31, -137.74],

               [343.05, -10.31, -137.74],
               [343.05, -28.31, -137.74],

               [348.05, -13.31, -137.34],
               [348.05, -25.31, -138.64],
               
               [350.05, -19.31, -138.44]]

# x, y 좌표만 추출
xy_points = np.array([[p[0], p[1]] for p in points_edge])

# 중심 설정: x의 평균, y는 고정값 -19.31
# x축에 평행하고 중심 y좌표가 -19.31인 작아진 타원
x_mean = np.mean(xy_points[:, 0])
center_fixed = np.array([x_mean, -19.31])

# 축 길이 계산 및 축소
x_range = xy_points[:, 0].max() - xy_points[:, 0].min()
y_range = xy_points[:, 1].max() - xy_points[:, 1].min()
scale_factor = 0.90
width = x_range * 1.1 * scale_factor
height = y_range * 1.1 * scale_factor

# 시각화
fig, ax = plt.subplots()
ax.scatter(xy_points[:, 0], xy_points[:, 1], label='Original Points')

ellipse = Ellipse(center_fixed, width=width, height=height, angle=0,
                  edgecolor='purple', fc='None', lw=2, label='Fitted Ellipse')
ax.add_patch(ellipse)

ax.set_aspect('equal')
ax.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitted Horizontal Ellipse (Centered at y=-19.31)')
plt.grid(True)
plt.tight_layout()
plt.show()
