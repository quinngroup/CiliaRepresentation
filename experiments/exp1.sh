#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python ../Driver.py --save ../results/exp1/lr1e-3.h5 --log ../results/exp1/lr1e-3 --lr 1e-3 --epochs 30 --source ../../local_data/clipcropped
python ../Driver.py --save ../results/exp1/lr1e-4.h5 --log ../results/exp1/lr1e-4 --lr 1e-4 --epochs 30 --source ../../local_data/clipcropped
python ../Driver.py --save ../results/exp1/lr1e-5.h5 --log ../results/exp1/lr1e-5 --lr 1e-5 --epochs 30 --source ../../local_data/clipcropped
python ../Driver.py --save ../results/exp1/lr1e-6.h5 --log ../results/exp1/lr1e-6 --lr 1e-6 --epochs 30 --source ../../local_data/clipcropped
python ../Driver.py --save ../results/exp1/lr1e-7.h5 --log ../results/exp1/lr1e-7 --lr 1e-7 --epochs 30 --source ../../local_data/clipcropped
python ../Driver.py --save ../results/exp1/lr1e-8.h5 --log ../results/exp1/lr1e-8 --lr 1e-8 --epochs 30 --source ../../local_data/clipcropped