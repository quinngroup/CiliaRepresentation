#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr1e-5.h5 --plr 1e-5 --log ../results/exp3/plr1e-5 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr7e-6.h5 --plr 7e-6 --log ../results/exp3/plr7e-6 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr4e-6.h5 --plr 4e-6 --log ../results/exp3/plr4e-6 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr1e-6.h5 --plr 1e-6 --log ../results/exp3/plr1e-6 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr7e-7.h5 --plr 7e-7 --log ../results/exp3/plr7e-7 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr4e-7.h5 --plr 4e-7 --log ../results/exp3/plr4e-7 --epochs 30 --source ../../local_data/patches
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp3/plr1e-7.h5 --plr 1e-7 --log ../results/exp3/plr1e-7 --epochs 30 --source ../../local_data/patches
