#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/g1.h5 --gamma 1.0 --log ../results/exp2/g1.0 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/g5e-1.h5 --gamma 0.5 --log ../results/exp2/g5e-1 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/g1e-1.h5 --gamma 0.1 --log ../results/exp2/g1e-1 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/g5e-2.h5 --gamma 0.05 --log ../results/exp2/g5e-2 --epochs 30 --source ../../local_data/patches --batch_size 160
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/g0.h5 --gamma 0 --log ../results/exp2/g0 --epochs 30 --source ../../local_data/patches --batch_size 160
