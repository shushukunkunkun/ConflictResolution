from matplotlib import pyplot as plt
import numpy as np
from other_function import discretize_circle, discretize_square, discretize_line

# 离散化各个障碍物的点
line1_points = discretize_line([(200, 20), (600, 20), (500, 250), (150, 150), (200, 20)])
Square1 = discretize_square([700, 300], 30)
Square2 = discretize_square([620, 450], 80)
Square3 = discretize_square([700, 600], 30)
Triangle = discretize_line([(400, 600), (600, 750), (200, 750), (400, 600)])
Circle1 = discretize_circle([100, 550], 60)
Circle2 = discretize_circle([150, 350], 50)
Wall = discretize_line([(0, 0), (0, 800), (800, 800), (800, 0), (0, 0)])

# 合并所有障碍物点
obstacle_coor = np.vstack((line1_points, Square1, Square2, Square3, Triangle, Circle1, Circle2, Wall))

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 定义圆心和半径
center1 = (50, 750)
radius1 = 30

center2 = (750, 750)
radius2 = 30

center3 = (50, 50)
radius3 = 30

center4 = (750, 50)
radius4 = 30

center5 = (500, 300)
radius5 = 20

center6 = (300, 300)
radius6 = 20

center7 = (500, 500)
radius7 = 20

center8 = (300, 500)
radius8 = 20
centers = [center1, center2, center3, center4, center5, center6, center7, center8]
radii = [radius1, radius2, radius3, radius4, radius5, radius6, radius7, radius8]

# 定义颜色列表，前四个为蓝色，后四个为绿色
colors = ['blue'] * 4 + ['green'] * 4

# 循环创建圆形补丁并添加到坐标轴
for center, radius, color in zip(centers, radii, colors):
    circle = plt.Circle(center, radius, color=color, fill=False, linewidth=2)
    ax.add_patch(circle)
# 绘制障碍物点
ax.scatter(obstacle_coor[:, 0], obstacle_coor[:, 1], color='red', s=20, label='Obstacles')

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Obstacles Plot')
ax.legend()
ax.axis('equal')  # 保证x轴和y轴比例一致

plt.show()
