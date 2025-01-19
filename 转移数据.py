import os
import shutil

# 定义原始路径和目标根路径
source_base_path = './data/kitti/training/'
target_base_path = '/home/lm/code/Torch-Pruning/develop/data/kitti/training/'

# 定义子目录
sub_dirs = {
    'pcd': 'velodyne_reduced',
    'image': 'image_2',
    'ann': 'calib'
}

# 在目标路径中创建相同的目录结构
for sub_dir in sub_dirs.values():
    os.makedirs(os.path.join(target_base_path, sub_dir), exist_ok=True)

# 复制文件
for i in range(300):
    file_index = str(i).zfill(6)  # 生成类似"000008"的文件编号

    # 定义每个文件的原始路径和目标路径
    pcd_file = os.path.join(source_base_path, sub_dirs['pcd'], f'{file_index}.bin')
    image_file = os.path.join(source_base_path, sub_dirs['image'], f'{file_index}.png')
    ann_file = os.path.join(source_base_path, sub_dirs['ann'], f'{file_index}.txt')

    # 复制到目标路径中的相同结构下
    shutil.copy(pcd_file, os.path.join(target_base_path, sub_dirs['pcd'], f'{file_index}.bin'))
    shutil.copy(image_file, os.path.join(target_base_path, sub_dirs['image'], f'{file_index}.png'))
    shutil.copy(ann_file, os.path.join(target_base_path, sub_dirs['ann'], f'{file_index}.txt'))

print("数据复制完成，路径结构已保持一致")
