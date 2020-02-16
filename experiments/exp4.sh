#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/wd1e-7.h5 --reg2 1e-7 --log ../results/exp2/wd1e-7 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/wd1e-6.h5 --reg2 1e-6 --log ../results/exp2/wd1e-6 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/wd1e-5.h5 --reg2 1e-5 --log ../results/exp2/wd1e-5 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/wd1e-4.h5 --reg2 1e-4 --log ../results/exp2/wd1e-4 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/wd1e-3.h5 --reg2 1e-3 --log ../results/exp2/wd1e-3 --epochs 30 --source ../../local_data/patches --batch_size 160
