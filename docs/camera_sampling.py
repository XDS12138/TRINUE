from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

def draw_cone_faces(ax, tip, height, radius, direction, resolution=30, color='royalblue', alpha=0.25):
    direction = direction / np.linalg.norm(direction)
    base_center = tip + direction * height

    # 底面圆构造
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle = np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta)
    ])

    # z轴对齐旋转
    z_axis = np.array([0, 0, -1])
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    s = np.linalg.norm(v)
    R = np.eye(3)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    rotated_circle = R @ circle + base_center[:, np.newaxis]

    # 面片构建
    faces = [[tip, rotated_circle[:, i], rotated_circle[:, (i + 1) % resolution]] for i in range(resolution)]
    poly3d = [[list(p) for p in tri] for tri in faces]
    collection = Poly3DCollection(poly3d, color=color, alpha=alpha, linewidths=0)
    ax.add_collection3d(collection)

# 相机布局参数
x_offsets = np.arange(-3, 3.1, 1.5)
y_offsets = np.arange(0, 5.1, 1.5)
z_height = 2.0
direction = np.array([0, 0, -1])
cone_height = 1.5
cone_radius = 1.2

# 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Camera Sampling Strategy with Overlapping View Cones", fontsize=14)

# 绘制视锥体与相机点
for x in x_offsets:
    for y in y_offsets:
        tip = np.array([x, y, z_height])
        draw_cone_faces(ax, tip, height=cone_height, radius=cone_radius, direction=direction)
        ax.scatter(*tip, color='k', s=20)

# 相机注释（上方远离圆锥）
annotated = np.array([x_offsets[0], y_offsets[0], z_height])
ax.text(annotated[0], annotated[1] - 1.0, annotated[2] + 0.5, "Camera Position", fontsize=10, color='black',
        ha='center', va='bottom')

# 坐标轴标签
ax.set_xlabel("X", labelpad=12)
ax.set_ylabel("Y", labelpad=12)
ax.set_zlabel("Z", labelpad=12)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Z轴线和标签
ax.plot([0, 0], [0, 0], [0, 4], color='gray', linewidth=2, linestyle='--')
ax.text(0, 0, 4.2, "Z", color='gray', fontsize=12, ha='center')

# 设置视角与边界
ax.view_init(elev=30, azim=45)
ax.set_xlim(-4, 4)
ax.set_ylim(-2, 6)
ax.set_zlim(0, 4.5)

plt.tight_layout()
plt.show()
