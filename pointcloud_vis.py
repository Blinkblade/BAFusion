import numpy as np
import open3d as o3d

# 点云文件路径
bin_path = "/home/lm/code/OpenMMlab/mmdetection3d/data/kitti/training/velodyne/000025.bin"

# 读取BIN文件中点云数据
point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
xyz = point_cloud[:, :3]
intensity = point_cloud[:, 3]

# 将反射强度归一化，用于颜色映射 (0~1)
intensity_norm = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)

# 使用颜色映射，将强度映射到一个颜色梯度（例如：从蓝到红）
# 您可根据喜好定义自己的映射，这里简单定义为R通道=强度，G=0, B=1-强度
colors = np.zeros((xyz.shape[0], 3))
colors[:, 0] = intensity_norm    # R: 强度越大越红
colors[:, 1] = 0.0               # G: 固定为0
colors[:, 2] = 1.0 - intensity_norm  # B: 强度越低越蓝

# 构建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 创建坐标系帮助理解点云空间分布
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])

# 使用Visualizer以自定义渲染选项
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="KITTI Point Cloud", width=1200, height=800)
vis.add_geometry(pcd)
vis.add_geometry(axis)

# 获取渲染选项以定制可视化效果
render_option = vis.get_render_option()
render_option.background_color = np.asarray([0.05, 0.05, 0.05])  # 深色背景
render_option.point_size = 2.0  # 增大点的大小
render_option.show_coordinate_frame = False  # 我们已经加了一个坐标系对象，可关闭默认的坐标系展示

# 运行并显示可视化窗口
vis.run()
vis.destroy_window()
