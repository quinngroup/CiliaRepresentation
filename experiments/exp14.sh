#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --model nvp4 --save ../results/exp14/NVP_4.h5 --log ../results/exp14/ --epochs 90 --source ../../local_data/patches --log_image 2 --lsdim 500
