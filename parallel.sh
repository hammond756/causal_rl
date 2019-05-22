#!/bin/bash

function train() {
    python train.py \
            --intervention_value 5 \
            --n_iters 15000 \
            --use_random True \
            --entr_loss_coeff 0 \
            --log_iters 1000\
            --lr 0.001 \
            --noise_dist bernoulli 0.5 \
            --predictor repeated \
            --reg_lambda 1 \
            --seed $1 \
            --dag_name $2 \
            --random_weights False  
}

export -f train

experimentName="identify_optimization_problem_chain"

mkdir experiments/$experimentName || exit

parallel --jobs 2 train ::: {0..9} ::: specified_chain

wait

mv experiments/inbox/* experiments/$experimentName
