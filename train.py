import torch
import argparse
import pickle
import uuid
import os

from causal_rl.sem import StructuralEquationModel, DirectedAcyclicGraph
from causal_rl.sem.utils import draw
from causal_rl.train import train_active, PredictArgumentParser
from causal_rl.environments import causal_models, directed_edges

def save_configuration(config):
    """
    Creates a file in the output directory specified in the _config_ object
    with all the arguments and values. This file can be used to re-run the
    script with the same parameters.
    """
    with open(config.output_dir + '/config.txt', 'w') as f:
        for key, value in vars(config).items():
            f.write('--{}\n'.format(key))

            # if nargs > 1 they all need to be on a new line
            if type(value) == list:
                for val in value:
                    f.write('{}\n'.format(val))
            else:
                f.write('{}\n'.format(value))

if __name__ == '__main__':
    parser = PredictArgumentParser()
    config = parser.parse_args()

    if config.dag_name is 'random':
        assert 'random_dag' in vars(config), 'Size is required for a random graph'

    if not config.output_dir:
        _id = str(uuid.uuid1())
        output_dir = os.path.join('experiments', 'inbox', str(_id))
        os.makedirs(output_dir)
        config.output_dir = output_dir
    
    # save seed for reproducibility
    if config.seed is None:
        config.seed = torch.initial_seed()
    else:
        torch.manual_seed(config.seed)

    # initialize causal model
    if config.dag_name != 'random':
        if config.random_weights:
            graph = directed_edges.get(config.dag_name)
            sem = StructuralEquationModel.random_with_edges(graph, *config.noise_dist)
        else:
            graph = DirectedAcyclicGraph(causal_models.get(config.dag_name))
            sem = StructuralEquationModel(graph, *config.noise_dist)
    else:
        args = config.random_dag + config.noise_dist
        sem = StructuralEquationModel.random(*args)

    stats = train_active(sem, config)

    # # # # #
    # save all the things
    # # # # # 

    save_configuration(config)

    # statistics
    with open(config.output_dir + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)