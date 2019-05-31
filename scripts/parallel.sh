#!/bin/bash

function train() {
    python train.py \
            --intervention_value 5 \
            --n_iters 20000 \
            --use_random True \
            --entr_loss_coeff 0 \
            --log_iters 1000 \
            --lr 0.001 \
            --noise_dist bernoulli 0.5 \
            --predictor two_step \
            --reg_lambda 0 \
            --seed $1 \
            --dag_name $2 \
            --random_weights True  
}

export -f train

experimentName="two_step_end_to_end"

mkdir experiments/$experimentName || exit # check if folder already exists

parallel --jobs 2 train ::: {0..4} ::: chain independent common_effect common_cause classic_confounding

wait

mv experiments/inbox/* experiments/$experimentName