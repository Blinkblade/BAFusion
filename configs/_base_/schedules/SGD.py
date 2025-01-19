# training schedule for 1x
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=1)
# train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)

# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

max_epochs=200
# 学习率取自faf_cp_adamwcyc200e tensorboard分析
lr = 0.0005
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# optimizer
# optimizer
# 注意带有动量项时,不可以resume,因为resume之前没有使用动量无法继承
# 考虑重新开始或不继承权重
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)








