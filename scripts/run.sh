#!/bin/bash

python train.py \
        --intervention_value 5 \
        --n_iters 10000 \
        --use_random True \
        --entr_loss_coeff 0 \
        --log_iters 1000 \
        --lr 0.001 \
        --noise_dist bernoulli 0.5 \
        --predictor two_step \
        --reg_lambda 0 \
        --dag_name random \
        --random_dag 4 0.6 \
        --seed 0 \
        --random_weights True 