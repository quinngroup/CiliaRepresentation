#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp14/NVP.h5 --log ../results/exp14/NVP --epochs 30 --source ../../local_data/patches --log_image 2
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --model 'nvp1' --save ../results/exp14/NVP_4.h5 --log ../results/exp14/NVP_4 --epochs 30 --source ../../local_data/patches --log_image 2
