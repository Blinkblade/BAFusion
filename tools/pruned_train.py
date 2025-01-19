# from mmengine import Config
# from mmengine.runner import load_checkpoint
# from mmdet3d.apis import 
# import torch
# import os
# import argparse


# # def parse_args():
# #     parser = argparse.ArgumentParser(description='Train a 3D detector')
# #     parser.add_argument('config', help='train config file path')
# #     parser.add_argument('--work-dir', help='the dir to save logs and models')
    
# #     parser.add_argument(
# #         '--amp',
# #         action='store_true',
# #         default=False,
# #         help='enable automatic-mixed-precision training')
# #     parser.add_argument(
# #         '--sync_bn',
# #         choices=['none', 'torch', 'mmcv'],
# #         default='none',
# #         help='convert all BatchNorm layers in the model to SyncBatchNorm '
# #         '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
# #     parser.add_argument(
# #         '--auto-scale-lr',
# #         action='store_true',
# #         help='enable automatically scaling LR.')
# #     parser.add_argument(
# #         '--resume',
# #         nargs='?',
# #         type=str,
# #         const='auto',
# #         help='If specify checkpoint path, resume from it, while if not '
# #         'specify, try to auto resume from the latest checkpoint '
# #         'in the work directory.')
# #     parser.add_argument(
# #         '--ceph', action='store_true', help='Use ceph as data storage backend')
# #     # parser.add_argument(
# #     #     '--cfg-options',
# #     #     nargs='+',
# #     #     action=DictAction,
# #     #     help='override some settings in the used config, the key-value pair '
# #     #     'in xxx=yyy format will be merged into config file. If the value to '
# #     #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
# #     #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
# #     #     'Note that the quotation marks are necessary and that no white space '
# #     #     'is allowed.')
# #     parser.add_argument(
# #         '--launcher',
# #         choices=['none', 'pytorch', 'slurm', 'mpi'],
# #         default='none',
# #         help='job launcher')
# #     # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
# #     # will pass the `--local-rank` parameter to `tools/train.py` instead
# #     # of `--local_rank`.
# #     parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
# #     args = parser.parse_args()
# #     if 'LOCAL_RANK' not in os.environ:
# #         os.environ['LOCAL_RANK'] = str(args.local_rank)
# #     return args


# def main():

#     config = Config.fromfile('/home/lm/code/OpenMMlab/mmdetection3d/work_dirs/bafpp-cyc80e-cspfpn-conv33-bn12-nope-aug-prune-add/bafpp-cyc80e-cspfpn-conv33-bn12-nope-aug-prune-add.py')
#     model = build_detector(config.model, test_cfg=config.get('test_cfg'))

#     # # 创建完整模型
#     # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

#     # # 加载剪枝后的state dict
#     # checkpoint = torch.load('pruned_bafpp.pth')

#     # # 只加载需要的部分参数
#     # model.load_state_dict(checkpoint['state_dict'], strict=False)



# if __name__ == '__main__':
#     main()
