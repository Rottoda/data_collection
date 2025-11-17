import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def create_and_plot_tangent_ellipsoid(all_points, edge_points, inside_points, center_points):
    """
    주어진 모든 3D 포인트들을 포함하는 타원체를 생성하고 시각화합니다.

    Args:
        all_points (list): 피팅에 사용될 모든 포인트의 리스트.
        edge_points (list): 경계 포인트.
        inside_points (list): 내부 포인트.
        center_points (list): 중심 포인트.
    """
    # 리스트를 numpy 배열로 변환
    all_points_np = np.array(all_points)
    edge_points_np = np.array(edge_points)
    inside_points_np = np.array(inside_points)
    center_points_np = np.array(center_points)

    if all_points_np.shape[1] != 3:
        raise ValueError("입력 포인트는 3차원이어야 합니다 (x, y, z).")

    # 1. 타원체의 중심과 반지름 계산
    # 중심: 모든 포인트의 각 축 평균값
    center = np.mean(all_points_np, axis=0)
    # 반지름: 모든 포인트의 각 축 (최댓값 - 최솟값) / 2
    # 이렇게 하면 타원체가 모든 포인트들의 경계 상자에 꼭 맞게 됩니다.
    radii = (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0)) / 2.0

    print(f"계산된 타원체 중심: {center}")
    print(f"계산된 타원체 반지름 (x, y, z): {radii}")

    # 2. 타원체 표면의 점들 생성
    # 파라미터 u, v 정의 (구면 좌표계)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # 타원체 파라미터 방정식
    x = center[0] + radii[0] * np.cos(u) * np.sin(v)
    y = center[1] + radii[1] * np.sin(u) * np.sin(v)
    z = center[2] + radii[2] * np.cos(v)

    # 3. 3D 시각화
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 타원체 표면 플로팅 (반투명하게 설정)
    ax.plot_surface(x, y, z, color='purple', alpha=0.15, rstride=5, cstride=5,
                    edgecolor='none')

    # 각 포인트 그룹 플로팅
    ax.scatter(edge_points_np[:, 0], edge_points_np[:, 1], edge_points_np[:, 2],
               color='r', s=25, label='Edge Points')
    ax.scatter(inside_points_np[:, 0], inside_points_np[:, 1], inside_points_np[:, 2],
               color='b', s=25, label='Inside Points')
    ax.scatter(center_points_np[:, 0], center_points_np[:, 1], center_points_np[:, 2],
               color='g', s=100, marker='*', label='Center Point')


    # 계산된 중심점 플로팅
    ax.scatter(center[0], center[1], center[2], color='black', s=100,
               marker='x', label='Calculated Center', depthshade=False)

    # 그래프 설정
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title('Fitted Ellipsoid Enclosing All Points', fontsize=16)

    # 축의 스케일을 비슷하게 맞춰서 타원체가 왜곡되어 보이지 않도록 함
    max_radius = np.max(radii) * 1.1 # 약간의 여백 추가
    ax.set_xlim(center[0] - max_radius, center[0] + max_radius)
    ax.set_ylim(center[1] - max_radius, center[1] + max_radius)
    ax.set_zlim(center[2] - max_radius, center[2] + max_radius)
    
    # 범례 추가
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Edge Points', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Inside Points', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Center Point', markerfacecolor='g', markersize=15),
        plt.Rectangle((0,0), 1, 1, fc="purple", alpha=0.2, label='Fitted Ellipsoid'),
        Line2D([0], [0], marker='x', color='black', label='Calculated Center', markersize=10, markeredgewidth=2, linestyle='None')
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_and_plot_tangent_ellipsoid_upper_half(all_points, edge_points, inside_points, center_points):
    """
    주어진 모든 3D 포인트들을 포함하는 타원체의 상단 절반을 생성하고 시각화합니다.

    Args:
        all_points (list): 피팅에 사용될 모든 포인트의 리스트.
        edge_points (list): 경계 포인트.
        inside_points (list): 내부 포인트.
        center_points (list): 중심 포인트.
    """
    # 리스트를 numpy 배열로 변환
    all_points_np = np.array(all_points)
    edge_points_np = np.array(edge_points)
    inside_points_np = np.array(inside_points)
    center_points_np = np.array(center_points)

    if all_points_np.shape[1] != 3:
        raise ValueError("입력 포인트는 3차원이어야 합니다 (x, y, z).")

    # 1. 타원체의 중심과 반지름 계산
    # 중심: 모든 포인트의 각 축 평균값
    center = np.mean(all_points_np, axis=0)
    # 반지름: 모든 포인트의 각 축 (최댓값 - 최솟값) / 2
    radii = (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0)) / 2.0

    print(f"계산된 타원체 중심: {center}")
    print(f"계산된 타원체 반지름 (x, y, z): {radii}")

    # 2. 타원체 표면의 점들 생성 (상단 절반만)
    # 파라미터 u, v 정의 (구면 좌표계)
    u = np.linspace(0, 2 * np.pi, 100)
    # v의 범위를 0부터 pi/2까지로 수정하여 상단 절반만 생성
    v = np.linspace(0, np.pi / 2, 50)
    u, v = np.meshgrid(u, v)

    # 타원체 파라미터 방정식
    x = center[0] + radii[0] * np.cos(u) * np.sin(v)
    y = center[1] + radii[1] * np.sin(u) * np.sin(v)
    z = center[2] + radii[2] * np.cos(v)

    # 3. 3D 시각화
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 타원체 표면 플로팅 (반투명하게 설정)
    ax.plot_surface(x, y, z, color='purple', alpha=0.2, rstride=5, cstride=5,
                    edgecolor='none')

    # 각 포인트 그룹 플로팅
    ax.scatter(edge_points_np[:, 0], edge_points_np[:, 1], edge_points_np[:, 2],
               color='r', s=25, label='Edge Points')
    ax.scatter(inside_points_np[:, 0], inside_points_np[:, 1], inside_points_np[:, 2],
               color='b', s=25, label='Inside Points')
    ax.scatter(center_points_np[:, 0], center_points_np[:, 1], center_points_np[:, 2],
               color='g', s=100, marker='*', label='Center Point')


    # 계산된 중심점 플로팅
    ax.scatter(center[0], center[1], center[2], color='black', s=100,
               marker='x', label='Calculated Center', depthshade=False)

    # 그래프 설정
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    ax.set_title('Upper Half of Fitted Ellipsoid', fontsize=16)

    # 축의 스케일을 비슷하게 맞춰서 타원체가 왜곡되어 보이지 않도록 함
    max_radius = np.max(radii) * 1.1 # 약간의 여백 추가
    ax.set_xlim(center[0] - max_radius, center[0] + max_radius)
    ax.set_ylim(center[1] - max_radius, center[1] + max_radius)
    ax.set_zlim(center[2] - max_radius, center[2] + max_radius)
    
    # 범례 추가
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Edge Points', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Inside Points', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Center Point', markerfacecolor='g', markersize=15),
        plt.Rectangle((0,0), 1, 1, fc="purple", alpha=0.2, label='Fitted Ellipsoid (Upper Half)'),
        Line2D([0], [0], marker='x', color='black', label='Calculated Center', markersize=10, markeredgewidth=2, linestyle='None')
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 제공된 3D 포인트 데이터
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

    points_center = [[338.05, -19.31, -134.46]]

    # 모든 포인트를 합쳐서 피팅에 사용
    all_points = points_edge + points_inside + points_center

    # 함수 호출
    create_and_plot_tangent_ellipsoid_upper_half(all_points, points_edge, points_inside, points_center)