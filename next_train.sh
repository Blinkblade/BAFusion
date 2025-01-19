PID=3242853
conda activate mmlab
while kill -0 $PID 2>/dev/null; do
    echo "模型1仍在训练..."
    sleep 60
done
echo "第一个训练任务已完成，启动第二个任务..."
python tools/train.py configs/BAFusionSECONDV2/BAFsecondSV2-resnet-con33-batch3.py 
# echo "第二个训练任务已完成，启动第三个任务..."
# python tools/train.py configs/FusionPointPillars/bafpps-80e-resnet-nopretrain.py 
# echo "第三个训练任务已完成，启动第四个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-SGD10e-resnet-noise-4h-autolr.py
# echo "第四个训练任务已完成，启动第五个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-SGD10e-resnet-noise-4h-lr001.py 
# echo "第5个训练任务已完成，启动第6个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-40e-resnet-noise-4h-autolr.py 
# echo "第6个训练任务已完成，启动第7个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-40e-resnet-noise-2h-autolr.py 
# echo "第7个训练任务已完成，启动第8个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-SGD10e-resnet-noise-autolr.py 
# echo "第8个训练任务已完成，启动第9个任务..."
# python tools/train.py configs/BAFusionSECOND/BAFsecondS-SGD10e-resnet-noise-lr0005.py

