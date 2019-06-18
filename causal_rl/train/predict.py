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
from causal_rl.models import predictors, policies

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"

def print_pretty(matrix):
    for row in matrix:
        print(pretty(row))

def policy_reward(old, new):
    r = 0
    for param_old, param_new in zip(old, new):
        r += (param_old - param_new).abs().sum().item()
    return r

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

class parse_noise_arg(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dist, param = values
        try:
            param = float(param)
        except:
            raise argparse.ArgumentTypeError('Second argument \'param\' should be a number')
        
        setattr(namespace, self.dest, [dist, param])

class PredictArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(PredictArgumentParser, self).__init__(fromfile_prefix_chars='@')

        self.add_argument('--dag_name', type=str, required=True)
        self.add_argument('--random_weights', type=str2bool, required=False, default=True)
        self.add_argument('--random_dag', type=float, nargs=2, required=False)
        self.add_argument('--predictor', type=str, required=True)
        self.add_argument('--n_iters', type=int, default=50000)
        self.add_argument('--log_iters', type=int, default=1000)
        self.add_argument('--policy', type=str, default='random')
        self.add_argument('--entr_loss_coeff', type=float, default=0)
        self.add_argument('--output_dir', type=str, action=readable_dir)
        self.add_argument('--seed', type=int, default=None)
        self.add_argument('--intervention_value', type=int, default=0)
        self.add_argument('--lr', type=float, default=0.0001)
        self.add_argument('--reg_lambda', type=float, default=1.)
        self.add_argument('--noise_dist', nargs=2, action=parse_noise_arg, default=['gaussian', 1.0])

def train(sem, config):

    n_iterations = config.n_iters
    entr_loss_coeff = config.entr_loss_coeff
    variables = torch.arange(sem.dim)

    # init predictor. This model takes in sampled values of X and
    # tries to predict X under intervention X_i = x. The learned
    # weights model the weights on the causal model.
    predictor = predictors.get(config.predictor)(sem)
    optimizer = torch.optim.SGD([
        {'params' : predictor.predict.parameters(), 'lr' : config.lr},
        {'params' : predictor.abduct.parameters(), 'lr' : 0.1}
    ])

    # initialize policy. This model chooses an intervention on one of the nodes in X.
    # This choice is not based on the state of X.
    policy = policies.get(config.policy)(sem.dim)
    if isinstance(policy, nn.Module):
        policy_optim = torch.optim.Adam(policy.parameters(), lr=config.lr)
        # policy_optim = torch.optim.RMSprop(policy.parameters(), lr=0.0143)
        # policy_baseline = 0

    # containers for statistics
    stats = {
        'loss' : {
            'pred' : [],
            'reg' : [],
            'total' : []
        },
        'action_probs' : [],
        'iterations' : [],
        'reward' : [],
        'causal_err' : [],
        'noise_err' : []
    }

    pred_loss_sum = 0
    reg_loss_sum = 0
    total_loss_sum = 0
    noise_err_sum = 0

    if isinstance(policy, nn.Module):
        action_loss_sum = 0
        reward_sum = 0

    for iteration in range(n_iterations):

        should_log = (iteration+1) % config.log_iters == 0

        Z_observational = sem(n=1, z_prev=torch.zeros(sem.dim), intervention=None)

        # sample action from policy network
        # action_logprob and action_prob are needed elsewhere to calculate the reward
        # and entropy coefficient
        action_logprob = policy(Z_observational)
        action_prob = action_logprob.exp()
        action_idx = torch.multinomial(action_prob, 1).long().item()
        
        # covert action to intervention
        action = variables[action_idx]
        inter_value = config.intervention_value
        intervention = (action, inter_value)

        # sample observation and target
        Z_true_intervention = sem.counterfactual(z_prev=torch.zeros(sem.dim), intervention=intervention)
        
        # # # #
        # Optimze causal model
        # # # #
        Z_pred_intervention = predictor(Z_observational, intervention)

        # Backprop on predictor. Adjust weights s.t. predictions get closer
        # to truth.
        optimizer.zero_grad()

        # compute loss
        pred_loss = (Z_pred_intervention - Z_true_intervention).pow(2).mean()
        reg_loss = torch.norm(predictor.predict.linear1, 1)
        loss = pred_loss + config.reg_lambda * reg_loss

        # accumulate losses
        pred_loss_sum += pred_loss.item()
        reg_loss_sum += reg_loss.item()
        total_loss_sum += loss.item()
        
        noise_err_sum += (predictor.noise - sem.noises).pow(2).mean().item()

        # compute gradients
        loss.backward()

        # heuristic. we know that the true matrix is lower triangular.
        predictor.predict.linear1.grad.tril_(-1)

        optimizer.step()

        if isinstance(policy, nn.Module):
            # # # #
            # Optimze policy network
            # # # #
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

        # print training progress and save statistics
        if should_log:
            print()
            print('{} / {} \t\t loss: {}'.format(iteration+1, n_iterations, total_loss_sum / config.log_iters))
            print('prediction loss:     ', pred_loss_sum / config.log_iters)
            print('regularization loss: ', reg_loss_sum / config.log_iters)
            print('obs  ', pretty(Z_observational))
            print('pred ', pretty(Z_pred_intervention))
            print('true ', pretty(Z_true_intervention))
            print('noise', pretty(sem.noises))
            print('pred noise:', pretty(predictor.noise))
            print()

            w_true = sem.graph.weights[1,:,:]
            w_model = predictor.predict.linear1.detach()
            diff = (w_true - w_model)
            stats['causal_err'].append(diff.abs().sum().item())

            stats['loss']['pred'].append(pred_loss_sum / config.log_iters)
            stats['loss']['reg'].append(reg_loss_sum / config.log_iters)
            stats['loss']['total'].append(total_loss_sum / config.log_iters)
            stats['noise_err'].append(noise_err_sum / config.log_iters)
            stats['iterations'].append(iteration)

            pred_loss_sum = 0
            reg_loss_sum = 0
            total_loss_sum = 0
            noise_err_sum = 0

            if isinstance(policy, nn.Module):
                stats['action_probs'].append(action_prob)
                stats['reward'].append(reward_sum / config.log_iters)
                reward_sum = 0
    
    stats['true_weights'] = w_true
    stats['model_weights'] = w_model
    stats['config'] = config
    
    return stats