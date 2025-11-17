import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# 좌표 입력 (z 좌표는 무시)
points_edge = [
    [338.03, 14.18, -137.66],
    [338.51, -5.29, -139.60],
    [347.19, 13.46, -139.05],
    [346.07, -3.78, -139.50],
    [325.40, 4.43, -136.48],
    [330.78, 14.81, -137.94],
    [332.11, -4.83, -138.80],
    [350.55, 4.46, -138.68]
]

# x, y 좌표만 추출
xy_points = np.array([[p[0], p[1]] for p in points_edge])

# 중심 설정: x의 평균, y는 고정값 4.18
# x축에 평행하고 중심 y좌표가 4.18인 작아진 타원
x_mean = np.mean(xy_points[:, 0])
center_fixed = np.array([x_mean, 4.18])

# 축 길이 계산 및 축소
x_range = xy_points[:, 0].max() - xy_points[:, 0].min()
y_range = xy_points[:, 1].max() - xy_points[:, 1].min()
scale_factor = 0.85
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
plt.title('Fitted Horizontal Ellipse (Centered at y=4.18)')
plt.grid(True)
plt.tight_layout()
plt.show()


points_inside = [[338.03, 11.18, -135.66],
                 [337.72, -3.63, -137.75],
                 [344.54, 3.92, -135.61],
                 [347.36, 9.97, -137.62],
                 [346.48, -1.20, -138.54],
                 [330.77, 4.55, -134.86],
                 [330.21, 10.16, -135.94],
                 [331.64, -1.46, -136.75]]

points_center = [[338.03, 4.18, -134.66]]