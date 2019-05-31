#!/bin/bash
for graph in chain common_effect common_cause classic_confounding
do
	python predict.py "@experiments/active_intervention_0/$graph/config.txt" --output_dir "experiments/active_intervention_0/${graph}_check/" --seed 42
done
