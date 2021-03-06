#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/bs10.h5 --log ../results/exp2/bs10 --batch_size 10 --epochs 30 --source ../../local_data/small
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/bs20.h5 --log ../results/exp2/bs20 --batch_size 20 --epochs 30 --source ../../local_data/small
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/bs40.h5 --log ../results/exp2/bs40 --batch_size 40 --epochs 30 --source ../../local_data/small
python -m torch.distributed.launch --nproc_per_node=2 ../VTP/appearance/Driver.py --save ../results/exp2/bs80.h5 --log ../results/exp2/bs80 --batch_size 80 --epochs 30 --source ../../local_data/small
