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

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = F.log_softmax(self.action_weight, dim=-1)
        return action_logprob

class Predictor(nn.Module):
    def __init__(self, dim):
        super(Predictor, self).__init__()
        self.dim = dim

        # heuristic: we know the true weights are lower triangular
        self.linear1 = nn.Parameter(torch.randn((dim,dim)).tril_())

    def _mask(self, vector, intervention):
        target, value = intervention
        vector.scatter_(dim=1, index=torch.tensor([[target]]), value=value)
    
    def forward(self, features, intervention):
        target, value = intervention

        # make a copy of the input, since _mask will modify in-place
        out = torch.tensor(features)
        self._mask(out, intervention)

        for _ in range(self.dim):
            out = out.matmul(self.linear1.t())
            self._mask(out, intervention)

        return out

def predict(sem, config):

    n_iterations = config.n_iters
    log_iters = config.log_iters
    use_random_policy = config.use_random
    entr_loss_coeff = config.entr_loss_coeff

    variables = torch.arange(sem.dim)

    # init predictor. This model takes in sampled values of X and
    # tries to predict X under intervention X_i = 0. The learned
    # weights model the weights on the causal model.
    predictor = Predictor(sem.dim)
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
    causal_err = []

    for iteration in range(n_iterations):

        should_log = (iteration+1) % config.log_iters == 0

        if use_random_policy:
            action_idx = random_policy(sem.dim)
        else:
            # sample action from policy network
            action_logprob = policy()
            action_prob = action_logprob.exp()
            action_idx = torch.multinomial(action_prob, 1).long().item()
        
        action = variables[action_idx]

        Z_observational = sem(n=1, z_prev=torch.zeros(sem.dim), intervention=None)
        Z_pred_intervention = predictor(Z_observational, (action, config.intervention_value))
        Z_true_intervention = sem.counterfactual(z_prev=torch.zeros(sem.dim), intervention=(action, config.intervention_value))

        # Backprop on predictor. Adjust weights s.t. predictions get closer
        # to truth.
        optimizer.zero_grad()

        # compute loss
        pred_loss = (Z_pred_intervention - Z_true_intervention).pow(2).sum()
        reg_loss = torch.norm(predictor.linear1, 1)
        loss = pred_loss + config.reg_lambda * reg_loss

        loss_sum += pred_loss.item()
        loss.backward()

        # heuristic. we know that the true matrix is lower triangular.
        predictor.linear1.grad.tril_()

        
        if should_log:
            print('old model weights')
            print_pretty(predictor.linear1)
            print()

            print('gradients')
            print_pretty(predictor.linear1.grad)
            print()

        optimizer.step()

        if should_log:
            print('new model weights')
            print_pretty(predictor.linear1)
            print()

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

        if should_log:
            print()
            print('{} / {} \t\t loss: {}'.format(iteration+1, n_iterations, loss_sum / config.log_iters))
            print('obs  ', pretty(Z_observational))
            print('pred ', pretty(Z_pred_intervention))
            print('true ', pretty(Z_true_intervention))
            print('noise', pretty(sem.noises))
            print()

            w_true = sem.graph.weights[1,:,:] + sem.roots
            w_model = predictor.linear1.detach()
            diff = (w_true - w_model)
            causal_err.append(diff.abs().sum().item())

            loss_log.append(loss_sum / config.log_iters)
            iter_log.append(iteration)

            
            loss_sum = 0

            if not use_random_policy:
                action_probs.append(action_prob)
                reward_log.append(reward_sum / config.log_iters)
                reward_sum = 0

    print('model', w_model)
    print('true', w_true)

    fig, ax = plt.subplots(2,3)

    im = ax[0][2].matshow(diff.abs(), vmin=0, vmax=1)
    plt.colorbar(mappable=im, ax=ax[0][2])
    ax[0][2].set_title('weight diff', pad=23)

    ax[0][1].plot(iter_log, loss_log)
    ax[0][1].set_title('loss')

    ax[0][0].plot(iter_log, causal_err)
    ax[0][0].set_title('causal_err')
    
    if not use_random_policy:
        y_plot = [torch.tensor(y) for y in zip(*action_probs)]
        y_plot = torch.stack(y_plot, dim=0)
        x_plot = torch.arange(y_plot.shape[1])
        ax[1][2].stackplot(x_plot, y_plot, labels=variables.tolist())
        ax[1][2].legend()
        ax[1][2].set_title('action_probs')

        ax[1][0].plot(iter_log, reward_log)
        ax[1][0].set_title('reward')

        bar(ax[1][1], action_prob.detach(), labels=variables.tolist())
        ax[1][1].set_title('final action prob')
    
    plt.tight_layout()
    plt.savefig(config.output_dir + '/stats.png')
    
    return {
        'true_weights' : w_true,
        'model_weights' : w_model,
        'loss' : loss_log,
        'causal_err' : causal_err,
        'action_probs' : action_probs if not config.use_random else None,
        'reward' : reward_log if not config.use_random else None
    }
