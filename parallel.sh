#!/bin/bash

function train() {
    python train.py \
            --intervention_value 1 \
            --n_iters 50000 \
            --use_random True \
            --entr_loss_coeff 0 \
            --log_iters 50000 \
            --lr 0.001 \
            --noise 0 \
            --reg_lambda 1 \
            --seed $1 \
            --dag_name $2
}

export -f train

experimentName="model_correctness_adam"

mkdir experiments/$experimentName || exit

parallel --jobs 2 train ::: {0..4} ::: independent common_cause common_effect classic_confounding chain

wait

mv experiments/inbox/* experiments/$experimentName
