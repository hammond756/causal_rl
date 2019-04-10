"""
This script runs an adverserial RL algorithm on a simple causal
envrionment. The algorithm works as follows:

There are two networks, a predictor and actor. The predictor
is trained to correctly predict the effect of an intervention,
given the state of observed variables X. This network models
the (causal) relation between these variables. The actor
models the agent and picks the intervention.

As a signal, the predictor gets the prediction error.
The actor is rewarded in proportion to the predction loss of the 
predictor. In other words, it is motivated by making the predcitor
slip up. The idea is that the actor proposes interventions that
are optimally informative for the predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import time
import uuid

from causal_rl.graph_utils import bar
from causal_rl.sem.utils import draw
from causal_rl.sem import StructuralEquationModel
from causal_rl.environments import causal_models
from causal_rl.sem.utils import draw

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"

def random_policy(dim):
    return torch.randint(0, dim, (1,)).long().item()

def policy_reward(old, new):
    r = 0
    for param_old, param_new in zip(old, new):
        r += (param_old - param_new).abs().sum().item()
    return r

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = F.log_softmax(self.action_weight, dim=-1)
        return action_logprob

class Predictor(nn.Module):
    def __init__(self, dim, n_hidden):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(dim*2, n_hidden)
        self.linear2 = nn.Linear(n_hidden, dim)
    
    def forward(self, features):
        out = self.linear1(features)
        out = F.relu(out)
        out = self.linear2(out)

        return out

def predict(config):

    torch.manual_seed(config.seed)

    n_iterations = config.n_iters
    log_iters = config.log_iters
    use_random_policy = config.use_random
    entr_loss_coeff = config.entr_loss_coeff
    graph = causal_models.get(config.dag_name)

    # initialize causal model
    if config.dag_name != 'random':
        sem = StructuralEquationModel.random_with_edges(graph)
    else:
        sem = StructuralEquationModel.random(*config.random_dag)

    # save arguments to file
    with open(config.output_dir + '/config.txt', 'w') as f:
        for key, value in vars(config).items():
            f.write('--{}\n'.format(key))
            
            if type(value) == list:
                for val in value:
                    f.write('{}\n'.format(val))
            else:
                f.write('{}\n'.format(value))

    # save visualization of causal graph
    draw(sem.graph.edges[1,:,:], config.output_dir + '/graph.png')

    variables = torch.arange(sem.dim)

    # init predictor. This model takes in sampled values of X and
    # tries to predict X under intervention X_i = 0. The learned
    # weights model the weights on the causal model.
    predictor = Predictor(sem.dim, config.n_hidden)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=config.lr)

    # initialize policy. This model chooses an intervention on one of the nodes in X.
    # This choice is not based on the state of X.
    if not use_random_policy:
        policy = SimplePolicy(sem.dim)
        policy_optim = torch.optim.Adam(policy.parameters(), lr=config.lr)
        # policy_optim = torch.optim.RMSprop(policy.parameters(), lr=0.0143)
        # policy_baseline = 0

    # containers for statistics
    loss_log = []
    action_probs = []
    loss_sum = 0
    iter_log = []
    action_loss_sum = 0
    reward_sum = 0
    reward_log = []

    for iteration in range(n_iterations):
        if use_random_policy:
            action_idx = random_policy(sem.dim)
        else:
            # sample action from policy network
            action_logprob = policy()
            action_prob = action_logprob.exp()
            action_idx = torch.multinomial(action_prob, 1).long().item()
        
        action = variables[action_idx]
        action_vector = torch.ones(sem.dim) \
                              .scatter_(0, torch.tensor(action), 0) \
                              .unsqueeze(0)

        Z_observational = sem(n=1, z_prev=torch.zeros(sem.dim), intervention=None)
        Z_pred_intervention = predictor(torch.cat([Z_observational, action_vector], dim=1))
        Z_true_intervention = sem.counterfactual(z_prev=torch.zeros(sem.dim), intervention=(action, config.intervention_value))

        # Backprop on predictor. Adjust weights s.t. predictions get closer
        # to truth.
        optimizer.zero_grad()

        # compute loss
        pred_loss = (Z_pred_intervention - Z_true_intervention).pow(2).sum()
        loss = pred_loss

        loss_sum += pred_loss.item()
        loss.backward()
        optimizer.step()

        if not use_random_policy:
            policy_optim.zero_grad()

            # policy network gets rewarded for doing interventions
            # that confuse the predictor.
            reward = loss.item() / 10
            reward_sum += reward

            # policy loss makes high reward actions more probable to be intervened on
            # (ie. actions that confuse the predictor)
            log_prob = action_logprob[action_idx]
            action_loss = -reward * log_prob

            # action_loss = -(reward-policy_baseline) * log_prob
            # policy_baseline = policy_baseline * 0.997 + reward * 0.003
            # action_loss_sum += action_loss.item()
            if entr_loss_coeff > 0:
                 entr = -(action_logprob * action_prob).sum()
                 action_loss -= entr_loss_coeff * entr
            action_loss.backward()
            policy_optim.step()

        if (iteration+1) % log_iters == 0:
            print()
            print('{} / {} \t\t loss: {}'.format(iteration+1, n_iterations, loss_sum / log_iters))
            print('obs ', pretty(Z_observational))
            print('int ', pretty(action_vector))
            print('pred', pretty(Z_pred_intervention))
            print('true', pretty(Z_true_intervention))
            print()
            loss_log.append(loss_sum / log_iters)
            iter_log.append(iteration)
            
            loss_sum = 0

            if not use_random_policy:
                action_probs.append(action_prob)
                reward_log.append(reward_sum / log_iters)
                reward_sum = 0


    fig, ax = plt.subplots(2,3)

    ax[0][1].plot(iter_log, loss_log)
    
    if not use_random_policy:
        y_plot = [torch.tensor(y) for y in zip(*action_probs)]
        y_plot = torch.stack(y_plot, dim=0)
        x_plot = torch.arange(y_plot.shape[1])
        ax[1][2].stackplot(x_plot, y_plot, labels=variables.tolist())
        ax[1][2].legend()

        ax[1][0].plot(iter_log, reward_log)
        bar(ax[1][1], action_prob.detach(), labels=variables.tolist())
    
    plt.savefig(config.output_dir + '/stats.png')

    with open(config.output_dir + '/stats.pkl', 'wb') as f:
        data = {
            'loss' : loss_log,
            'action_probs' : action_probs if not config.use_random else None,
            'reward' : reward_log if not config.use_random else None
        }
        pickle.dump(data, f)

class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--dag_name', type=str, required=True)
    parser.add_argument('--random_dag', type=float, nargs=2, required=False)
    parser.add_argument('--n_iters', type=int, default=50000)
    parser.add_argument('--log_iters', type=int, default=1000)
    parser.add_argument('--use_random', type=str2bool, default=False)
    parser.add_argument('--entr_loss_coeff', type=float, default=0)
    parser.add_argument('--output_dir', type=str, action=readable_dir)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--intervention_value', type=int, default=0)
    parser.add_argument('--n_hidden', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0001)

    config = parser.parse_args()

    if config.dag_name is 'random':
        assert 'random_dag' in vars(config), 'Size is required for a random graph'
    
    if not config.output_dir:
        timestamp = str(uuid.uuid1())
        output_dir = os.path.join('experiments', 'inbox', str(timestamp))
        os.makedirs(output_dir)
        config.output_dir = output_dir

    predict(config)
