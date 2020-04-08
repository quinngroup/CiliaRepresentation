#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --model nvp6 --save ../results/exp16/NVP_6_2.h5 --log ../results/exp16/NVP_6_2/ --epochs 30 --source ../../local_data/patches --log_image 2 --lsdim 500 --batch_size 22 --roll 4 --lr 1e-4 --load ../results/exp16/NVP_6.h5 --repeat 
