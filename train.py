import torch
import pandas as pd
import pickle
import uuid
import os

from causal_rl.sem import StructuralEquationModel, DirectedAcyclicGraph
from causal_rl.train import train, PredictArgumentParser
from causal_rl.environments import causal_models, directed_edges


def save_configuration(config, output_dir):
    """
    Creates a file in the output directory specified in the _config_ object
    with all the arguments and values. This file can be used to re-run the
    script with the same parameters.
    """
    with open(output_dir + '/config.txt', 'w') as f:
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


    if config.dag_name == 'random':
        assert 'random_dag' in vars(config), \
            'Size is required for a random graph'

    # save seed for reproducibility
    if config.seed is None:
        config.seed = torch.initial_seed()
    else:
        torch.manual_seed(config.seed)

    # initialize causal model
    if config.dag_name != 'random':
        if config.random_weights:
            graph = directed_edges.get(config.dag_name)
            sem = StructuralEquationModel.random_with_edges(graph,
                                                            *config.noise_dist)
        else:
            graph = DirectedAcyclicGraph(causal_models.get(config.dag_name))
            sem = StructuralEquationModel(graph, *config.noise_dist)
    else:
        args = config.random_dag + config.noise_dist
        sem = StructuralEquationModel.random(*args)

    records = train(sem, config)

    print('----------')
    print('w_true')
    print(records[-1]['w_true'])
    print()
    print('w_model')
    print(records[-1]['w_model'])
    print('----------')

    # # # # #
    # save all the things
    # # # # #

    output_dir = os.path.join(config.output_dir,  str(uuid.uuid1()))
    os.makedirs(output_dir)

    save_configuration(config, output_dir)

    # statistics
    with open(output_dir + '/stats.pkl', 'wb') as f:
        pickle.dump(pd.DataFrame(records), f)
