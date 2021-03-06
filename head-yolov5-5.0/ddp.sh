#!/bin/bash
# CUDA_VISIBLE_DEVICES="0,1,2,3" python3 train.py --data data/widerface.yaml --cfg models/yolov5s.yaml --weights 'pretrained models'
python -m torch.distributed.launch \
            --nproc_per_node 4 \
            --master_port 56699 \
        train.py \
            --name head_det_yolo