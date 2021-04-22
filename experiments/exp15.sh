#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --model nvp5 --save ../results/exp15/NVP_5.h5 --log ../results/exp15/NVP_5/ --epochs 90 --source ../../local_data/patches --log_image 2 --lsdim 500 --batch_size 22 --roll 4 --lr 1e-4
