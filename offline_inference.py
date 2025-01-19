from mmdet3d.apis import inference_detector, init_model, inference_multi_modality_detector, inference_multi_modality_detector2
from mmdet3d.registry import VISUALIZERS
from mmdet3d.utils import register_all_modules
from torch import nn
import torch

# register all modules in mmdet3d into the registries
register_all_modules()

# # config_file = '../configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
# # # download the checkpoint from model zoo and put it in `checkpoints/`
# # checkpoint_file = '../work_dirs/second/epoch_40.pth'
# config_file = "../configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py"
# checkpoint_file = "../checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"
# # config_file = "../work_dirs/myconfig/myconfig.py"
# # checkpoint_file = "../work_dirs/myconfig/epoch_1000.pth"
# config_file = "../work_dirs/fpp-SGD0005-yoloxpanfpn-kitti-3class-64-bevaug/fpp-SGD0005-yoloxpanfpn-kitti-3class-64-bevaug.py"
# checkpoint_file = "../work_dirs/fpp-SGD0005-yoloxpanfpn-kitti-3class-64-bevaug/epoch_28.pth"
config_file = "./work_dirs/bafpp-SGD-cspfpn-conv33-bn12-nope-aug/bafpp-SGD-cspfpn-conv33-bn12-nope-aug.py"
checkpoint_file = "./work_dirs/bafpp-SGD-cspfpn-conv33-bn12-nope-aug/epoch_3.pth"

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

print(type(model))
print(isinstance(model,nn.Module))
print(model)
torch.save(model,"./torchsave/bafpp_tp_test00.pth")
test_input = torch.randint(384,torch.Size([20210, 4]))
print(test_input.shape)
# model(test_input)/


pcd = './data/kitti/training/velodyne_reduced/000008.bin'
image = "./data/kitti/training/image_2/000008.png"
# ann = './data/kitti/kitti_infos_test.pkl'
# ann = './demo/data/kitti/000008.pkl'
ann = './data/kitti/training/calib/000008.txt'


result, data = inference_multi_modality_detector2(model, pcd, image, ann)
