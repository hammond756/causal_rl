"""
This script runs an adverserial RL algorithm on a simple causal
envrionment. The algorithm works as follows:

There are two networks, a predictor and actor. The predictor
is trained to correctly predict the value of target variable Y,
given the state of observed variables X. This network models
the (causal) relation between these sets of variables. The actor
models the agent and picks a variable to intervene on.

As a signal, the predictor gets the prediction error and a regulizer.
The actor is rewarded in proportion to the predction loss of the 
predictor. In other words, it is motivated by making the predcitor
slip up. The idea is that the actor proposes interventions that
are optimally informative for the predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom

from causal_rl.sem.utils import draw
from causal_rl.sem import StructuralEquationModel

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

if __name__ == "__main__":
    torch.manual_seed(42)
    vis = visdom.Visdom()

    dim = 10
    n_targets = 1
    observed = dim - n_targets

    n_iterations = 50000
    log_iters = 1000
    use_random_policy = False
    entr_loss_coeff = 1

    # init predictor. This model takes in sampled values of X and
    # tries to predict Y. (ie. estimating incomming weights on Y)
    predictor = torch.nn.Linear(observed, 1, bias=False)
    optimizer = torch.optim.Adam(predictor.parameters())

    # initialize causal model
    sem = StructuralEquationModel.random(dim, 0.3)

    # initialize policy. This model chooses an intervention on one of the nodes in X.
    # This choice is not based on the state of X.
    if not use_random_policy:
        policy = SimplePolicy(observed)
        policy_optim = torch.optim.Adam(policy.parameters(), lr=0.003)
        # policy_optim = torch.optim.RMSprop(policy.parameters(), lr=0.01)
        # policy_baseline = 0

    loss_log = []
    loss_sum = 0
    iter_log = []
    causal_err = []
    action_loss_sum = 0
    reward_sum = 0
    reward_log = []

    z_prev = torch.randn(dim)
    target = torch.randint(dim, (1,)).long().item()
    observed_variables = torch.tensor([i for i in range(dim) if i is not target])

    for iteration in range(n_iterations):
        if use_random_policy:
            action_idx = random_policy(observed)
        else:
            # sample action from policy network
            action_logprob = policy()
            action_prob = action_logprob.exp()
            action_idx = torch.multinomial(action_prob, 1).long().item()
        
        action = observed_variables[action_idx]
           

        # sample from SEM, using the selected action as intervention
        z = sem(n=1, z_prev=z_prev, intervention=(action, 0))

        X = z[:, observed_variables]
        Y = z[:, target]

        # Backprop on predictor. Adjust weights s.t. predictions get closer
        # to truth.
        optimizer.zero_grad()
        params = next(predictor.parameters())

        # compute loss
        pred_loss = (predictor(X) - Y).abs().sum()
        reg_loss = F.l1_loss(params, torch.zeros_like(params))
        loss = pred_loss + reg_loss

        loss_sum += pred_loss.item()
        loss.backward()
        optimizer.step()

        # go to next state
        z_prev = z

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


        # TODO: adapt logging to time-dependent structure!!!
        if (iteration+1) % log_iters == 0:
            w_true = sem.graph.weights(target)[1, observed_variables].view(-1)
            w_model = predictor.weight.view(-1)
            print("SAMPLE", pretty(X.view(-1)))
            print("TRUTH ", pretty(w_true))
            print("MODEL ", pretty(w_model))
            print()

            loss_log.append(loss_sum / log_iters)
            iter_log.append(iteration)
            causal_err.append((w_true - w_model).abs().sum().item())
            
            loss_sum = 0
            d = torch.stack([w_true, w_model])
            vis.bar(d.t(), win='d')
            vis.line(X=iter_log, Y=loss_log, win='loss', opts={'title': 'pred loss'})
            vis.line(X=iter_log, Y=causal_err, win='causal_err', opts={'title': 'causal_err'})

            if not use_random_policy:
                reward_log.append(reward_sum / log_iters)
                reward_sum = 0
                vis.line(X=iter_log, Y=reward_log, win='reward', opts={'title': 'reward'})
                vis.bar(action_prob, win='action')
