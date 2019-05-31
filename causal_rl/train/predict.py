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

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"

def print_pretty(matrix):
    for row in matrix:
        print(pretty(row))

def random_policy(dim):
    return torch.randint(0, dim, (1,)).long().item()

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
        self.add_argument('--use_random', type=str2bool, default=False)
        self.add_argument('--entr_loss_coeff', type=float, default=0)
        self.add_argument('--output_dir', type=str, action=readable_dir)
        self.add_argument('--seed', type=int, default=None)
        self.add_argument('--intervention_value', type=int, default=0)
        self.add_argument('--lr', type=float, default=0.0001)
        self.add_argument('--reg_lambda', type=float, default=1.)
        self.add_argument('--noise_dist', nargs=2, action=parse_noise_arg, default=['gaussian', 1.0])
        self.add_argument('--plot', type=str2bool, default=False)

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = F.log_softmax(self.action_weight, dim=-1)
        return action_logprob

class Predictor(nn.Module):
    def __init__(self, sem):
        super(Predictor, self).__init__()
        self.dim = sem.dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1 to carry noise to prediction
        self.linear1 = nn.Parameter(torch.randn((self.dim,self.dim)).tril_(-1) + sem.roots)

    def _mask(self, vector, intervention):
        if intervention is None:
            return

        target, value = intervention
        vector.scatter_(dim=1, index=torch.tensor([[target]]), value=value)
    
    def forward(self, features, intervention):
        # make a copy of the input, since _mask will modify in-place
        out = torch.tensor(features)
        self._mask(out, intervention)

        for _ in range(self.dim):
            out = out.matmul(self.linear1.t())
            self._mask(out, intervention)

        return out

class OrderedPredictor(nn.Module):
    def __init__(self, dim):
        super(OrderedPredictor, self).__init__()
        self.dim = dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1 to carry noise to prediction
        self.linear1 = nn.Parameter(torch.randn((self.dim,self.dim)).tril_(-1))
    
    def forward(self, noise, intervention):
        target, value = intervention

        output = torch.zeros_like(noise)

        for i in range(self.dim):
            if i == target:
                output[:, i] = value
                continue
            
            output[:, i] = self.linear1[i].matmul(output.clone().t()) + noise[:, i]

        return output

class NoiseInferencer(torch.nn.Module):
    def __init__(self, dim):
        super(NoiseInferencer, self).__init__()
        self.dim = dim
        self.l1 = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        out = self.l1(x)
        return torch.sigmoid(out) # if self.training else (out > 0.7).float()

class TwoStepPredictor(nn.Module):
    def __init__(self, sem):
        super(TwoStepPredictor, self).__init__()
        
        self.dim = sem.dim
        self.noise = None

        self.infer_noise = NoiseInferencer(self.dim)
        self.predictor = OrderedPredictor(self.dim)

    def forward(self, observation, intervention):
        noise = self.infer_noise(observation)
        self.noise = noise
        prediction = self.predictor(noise, intervention)
        return prediction

predictors = {
    'repeated' : Predictor,
    'two_step' : TwoStepPredictor
}

def train_active(sem, config):

    n_iterations = config.n_iters
    use_random_policy = config.use_random
    entr_loss_coeff = config.entr_loss_coeff

    variables = torch.arange(sem.dim)

    # init predictor. This model takes in sampled values of X and
    # tries to predict X under intervention X_i = x. The learned
    # weights model the weights on the causal model.
    predictor = predictors.get(config.predictor)(sem)
    optimizer = torch.optim.SGD([
        {'params' : predictor.predictor.parameters(), 'lr' : config.lr},
        {'params' : predictor.infer_noise.parameters(), 'lr' : 0.1}
    ])

    # initialize policy. This model chooses an intervention on one of the nodes in X.
    # This choice is not based on the state of X.
    policy = SimplePolicy(sem.dim)
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

    action_loss_sum = 0
    reward_sum = 0

    for iteration in range(n_iterations):

        should_log = (iteration+1) % config.log_iters == 0

        # sample action from policy network
        action_logprob = policy()
        action_prob = action_logprob.exp()
        action_idx = torch.multinomial(action_prob, 1).long().item()
        
        # covert action to intervention
        action = variables[action_idx]
        inter_value = config.intervention_value
        intervention = (action, inter_value)

        # sample observation and target
        Z_observational = sem(n=1, z_prev=torch.zeros(sem.dim), intervention=None)
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
        reg_loss = torch.norm(predictor.predictor.linear1, 1)
        loss = pred_loss + config.reg_lambda * reg_loss

        # accumulate losses
        pred_loss_sum += pred_loss.item()
        reg_loss_sum += reg_loss.item()
        total_loss_sum += loss.item()
        
        noise_err_sum += (predictor.noise - sem.noises).pow(2).mean().item()

        # compute gradients
        loss.backward()

        # heuristic. we know that the true matrix is lower triangular.
        predictor.predictor.linear1.grad.tril_(-1)

        optimizer.step()

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
            w_model = predictor.predictor.linear1.detach()
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

            if not use_random_policy:
                stats['action_probs'].append(action_prob)
                stats['reward'].append(reward_sum / config.log_iters)
                reward_sum = 0
    
    stats['true_weights'] = w_true
    stats['model_weights'] = w_model
    stats['config'] = config
    
    return stats
