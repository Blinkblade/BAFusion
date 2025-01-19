import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

pc_file = "/home/lm/code/OpenMMlab/mmdetection3d/data/kitti/training/velodyne/000025.bin"

points = np.fromfile(pc_file, dtype=np.float32)
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
# bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900,
#                    -1.5808]]))
# # Draw 3D bboxes
# visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()