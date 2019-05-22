#!/bin/bash

python train.py \
        --dag_name specified_common_effect \
        --random_weights False \
        --intervention_value 1 \
        --use_random True \
        --n_iters 1000 \
        --log_iters 1000 \
        --lr 0.001 \
        --reg_lambda 1 \
        --predictor repeated \
        --noise_dist bernoulli 0.5 \
        --seed 0