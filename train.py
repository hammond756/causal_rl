import torch
import argparse
import pickle
import uuid

from causal_rl.sem import StructuralEquationModel
from causal_rl.sem.utils import draw
from causal_rl.train import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--dag_name', type=str, required=True)
    parser.add_argument('--random_dag', type=float, nargs=2, required=False)
    parser.add_argument('--n_iters', type=int, default=50000)
    parser.add_argument('--log_iters', type=int, default=1000)
    parser.add_argument('--use_random', type=str2bool, default=False)
    parser.add_argument('--entr_loss_coeff', type=float, default=0)
    parser.add_argument('--output_dir', type=str, action=readable_dir)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--intervention_value', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reg_lambda', type=float, default=1.)
    parser.add_argument('--noise', type=float, default=0.)

    config = parser.parse_args()

    if config.dag_name is 'random':
        assert 'random_dag' in vars(config), 'Size is required for a random graph'

    graph = causal_models.get(config.dag_name)

    if not config.output_dir:
        _id = str(uuid.uuid1())
        output_dir = os.path.join('experiments', 'inbox', str(_id))
        os.makedirs(output_dir)
        config.output_dir = output_dir

    # save seed for reproducibility
    if config.seed is None:
        config.seed = torch.initial_seed()

    torch.manual_seed(config.seed)

    # initialize causal model
    if config.dag_name != 'random':
        sem = StructuralEquationModel.random_with_edges(graph, std=config.noise)
    else:
        sem = StructuralEquationModel.random(*config.random_dag, std=config.noise)

    stats = predict(sem, config)

    # # # # #
    # save all the things
    # # # # # 

    save_configuration(config)

    # visualization of causal graph
    draw(sem.graph.edges[1,:,:], config.output_dir + '/graph.png')

    # statistics
    with open(config.output_dir + '/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    print('experiment id:', timestamp)