#!/bin/bash

# 检查文件是否存在的函数
check_file_exists() {
    if [ ! -f "$1" ]; then
        return 0 # 文件不存在
    else
        return 1 # 文件存在
    fi
}

# 检查是否有训练任务正在运行的函数
check_training_running() {
    count=$(pgrep -f "python tools/train.py" | wc -l)
    if [ $count -gt 0 ]; then
        return 0 # 存在一个或多个训练任务
    else
        return 1 # 没有训练任务
    fi
}

# 指定要检查的文件路径
FILE="work_dirs/BAPartA2S-cyc50e-resnet-4h-auto/epoch_50.pth"

# 无限循环直到训练成功完成
while true; do
    # 首先检查是否有训练任务正在运行
    check_training_running

    # 如果有训练任务正在运行，则等待
    if [ $? -eq 0 ]; then
        echo "检测到训练任务正在运行，等待训练结束..."
        while check_training_running; do
            sleep 60 # 每60秒检查一次
        done
    fi

    # 检查文件是否存在
    check_file_exists $FILE

    # 如果文件不存在，则重新运行训练
    if [ $? -eq 0 ]; then
        echo "epoch_50.pth不存在，开始重新训练..."
        cd ~/LiuMin/code/OpenMMlab/mmdetection3d
        conda activate mmlab
        python tools/train.py configs/BAPartA2/BAPartA2S-cyc50e-resnet-4h-auto.py
    else
        echo "训练已经完成，epoch_50.pth文件存在。"
        break # 退出循环
    fi
done
