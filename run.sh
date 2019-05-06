#!/bin/bash

python train.py \
        --dag_name random \
        --random_dag 3.0 0.4 \
        --intervention_value 1 \
        --use_random True \
        --n_iters 1000 \
        --log_iters 1000 \
        --lr 0.001 \
        --reg_lambda 1 \
        --seed 0