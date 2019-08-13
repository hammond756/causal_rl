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
import argparse
import os
import time
from itertools import chain, combinations
from collections import defaultdict

from causal_rl.models import predictors, policies


def interventions(variables, max, min=0):
    '''
    Generates a byte tensor with all possible combinations of variables
    to intervene on, including the empty set.
    '''

    assert max < len(variables), 'Agent is now allowed to intervene on \
        all nodes'

    def powerset(iterable, min, max):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(min, max+1))

    result = []
    for subset in powerset(variables, min, max):
        mask = torch.tensor(
            [1. if i in subset else 0. for i in range(len(variables))]
        )
        result.append(mask)

    return torch.stack(result).byte()


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
        except Exception:
            raise argparse.ArgumentTypeError(
                'Second argument \'param\' should be a number'
            )

        setattr(namespace, self.dest, [dist, param])


class PredictArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(PredictArgumentParser, self).__init__(fromfile_prefix_chars='@')

        self.add_argument('--dag_name', type=str, required=True)
        self.add_argument('--random_weights', type=str2bool, required=False,
                          default=True)
        self.add_argument('--random_dag', type=float, nargs=2, required=False)
        self.add_argument('--n_iters', type=int, default=50000)
        self.add_argument('--log_iters', type=int, default=1000)
        self.add_argument('--policy', type=str, default='random')
        self.add_argument('--output_dir', type=str,
                          default='experiments/inbox')
        self.add_argument('--seed', type=int, default=None)
        self.add_argument('--intervention_value', type=int, default=0)
        self.add_argument('--min_targets', type=int, default=0)
        self.add_argument('--max_targets', type=int)
        self.add_argument('--lr_apc', type=float, default=0.01)
        self.add_argument('--lr_policy', type=float, default=0.001)
        self.add_argument('--lambda_1', type=float, default=1.0)
        self.add_argument('--lambda_2', type=float, default=1.0)
        self.add_argument('--noise_dist', nargs=2, action=parse_noise_arg,
                          default=['bernoulli', 0.5])
        self.add_argument('--ordered', type=str2bool, default=False)
        self.add_argument('--method', type=str, default='matrix')
        self.add_argument('--predictor', type=str, default='two_step')
        self.add_argument('--clip_grad', type=float, default=float('Inf'))


def train(sem, config):

    n_iterations = config.n_iters

    variables = torch.arange(sem.dim)
    allowed_actions = interventions(variables,
                                    min=config.min_targets,
                                    max=config.max_targets
                                    if config.max_targets is not None
                                    else sem.dim - 1)

    # init APC model. This model takes in sampled values of X and
    # tries to predict X under intervention X_i = x. The learned
    # weights model the weights on the causal model.
    model = predictors.get(config.predictor)(sem.dim,
                                             ordered=config.ordered,
                                             method=config.method)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_apc)

    # gather relevant arguments for policy
    policy_args = {
        'n_actions': len(allowed_actions),
        'dim': sem.dim,
    }

    if config.policy == 'sink':
        # add a bias towards action that contain ONLY sink variables
        bias = allowed_actions[:, sem.non_sink_mask].sum(dim=1) == 0
        policy_args['bias'] = bias.float()

    if config.policy == 'non_sink':
        # add a bias towards action that contain NO sink variables
        bias = allowed_actions[:, sem.sink_mask].sum(dim=1) == 0
        policy_args['bias'] = bias.float()

    if config.policy.split('_')[0] == 'action':
        action_idxs = torch.tensor(
            [int(x) for x in config.policy.split('_')[1:]]
        )
        bias = torch.zeros(len(allowed_actions))
        bias[action_idxs] = 1.
        policy_args['bias'] = bias.float()


    # initialize policy. This model chooses an the intervention to perform
    policy = policies.get(config.policy)(**policy_args)
    if isinstance(policy, nn.Module):
        policy_optim = torch.optim.SGD(policy.parameters(), lr=config.lr_policy)
        # policy_optim = torch.optim.RMSprop(policy.parameters(), lr=0.0143)
        # policy_baseline = 0

    # list to store logs
    records = []

    pred_loss_sum = 0
    lasso_loss_sum = 0
    cycle_loss_sum = 0
    total_loss_sum = 0
    noise_err_sum = 0
    cpu_time_sum = 0
    start_timer = time.time()

    if isinstance(policy, nn.Module):
        action_loss_sum = 0
        reward_sum = 0

    for iteration in range(n_iterations):

        should_log = (iteration + 1) % config.log_iters == 0 or iteration == 0

        observation = sem(n=1, z_prev=torch.zeros(sem.dim), intervention=None)

        # sample action from policy network
        # action_logprob and action_prob are needed elsewhere
        # to calculate the reward and entropy coefficient
        inp = defaultdict(lambda: None, {
            'introspective': model.B.detach(),
            'linear': observation
        })[config.policy]

        action_logprob = policy(inp)
        action_prob = action_logprob.exp()
        action_idx = torch.multinomial(action_prob, 1).long().item()

        # covert action to intervention
        action = allowed_actions[action_idx]
        inter_value = config.intervention_value
        intervention = (action, inter_value)

        # sample observation and target
        target = sem.counterfactual(z_prev=torch.zeros(sem.dim),
                                    intervention=intervention)

        # # # #
        # Optimze causal model
        # # # #
        prediction = model(observation, intervention)

        # Backprop on model. Adjust weights s.t. predictions get closer
        # to truth.
        optimizer.zero_grad()

        # compute loss components of un-intervened nodes
        predicted_vars = 1 - action.unsqueeze(0)
        pred_loss = (prediction - target)[predicted_vars].pow(2).mean()

        lasso_loss = model.B.norm(p=1) / sem.dim ** 2
        cycle_loss = model.B.matrix_power(model.dim).norm(p=1)

        # compute regularization loss
        loss = pred_loss \
            + config.lambda_1 * lasso_loss \
            + config.lambda_2 * cycle_loss

        # accumulate losses
        pred_loss_sum += pred_loss.item()
        lasso_loss_sum += lasso_loss.item()
        cycle_loss_sum += cycle_loss.item()
        total_loss_sum += loss.item()

        noise_err_sum += (model.noise - sem.noise).pow(2).mean().item()

        # compute gradients
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.clip_grad)

        # heuristic. we know that the true matrix is lower triangular.
        if config.ordered:
            model.B.grad.tril_(-1)

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

            # policy loss makes high reward actions more probable to be
            # intervened on (ie. actions that confuse the predictor)
            log_prob = action_logprob[action_idx]
            action_loss = -reward * log_prob
            action_loss_sum += action_loss.item()

            action_loss.backward()
            policy_optim.step()

        # print training progress and save statistics
        if should_log:
            avg_loss = total_loss_sum / config.log_iters
            print()
            print('{} / {} \t\t loss: {}'.format(iteration+1,
                                                 n_iterations,
                                                 avg_loss))
            print('prediction loss:', pred_loss_sum / config.log_iters)
            print('lasso loss:     ', lasso_loss_sum / config.log_iters)
            print('cycle loss:     ', cycle_loss_sum / config.log_iters)
            print('obs  ', pretty(observation))
            print('int  ', pretty(intervention[0]))
            print()
            print('true ', pretty(target))
            print('pred ', pretty(prediction))

            # TODO: make sure this prints the correct noise even when
            # association, intervention and counterfactual are mixed
            print()
            print('noise:', pretty(sem.noise))
            print('pred :', pretty(model.noise))
            print()

            w_true = sem.graph.weights
            w_model = model.B.detach()
            diff = (w_true - w_model)

            def true_postive(sem, diff):
                num_true = sem.graph.edges.sum().item()

                # completely disconnector model, all error is false positive
                if num_true == 0:
                    return 0

                error = (diff.abs() * sem.graph.edges.float()).sum()
                return error.item() / num_true

            def false_positive(sem, diff):
                num_false = sem.dim**2 - sem.graph.edges.sum().item()

                error = (diff.abs() * (1 - sem.graph.edges.float())).sum()
                return error.item() / num_false

            row = {
                'dim': sem.dim,
                'iterations': iteration,
                'causal_err': diff.abs().sum().item(),
                'true_positive': true_postive(sem, diff),
                'false_positive': false_positive(sem, diff),
                'pred_loss': pred_loss_sum / config.log_iters,
                'lasso_loss': lasso_loss_sum / config.log_iters,
                'cycle_loss': cycle_loss_sum / config.log_iters,
                'total_loss': total_loss_sum / config.log_iters,
                'noise_err': noise_err_sum / config.log_iters,
                'cpu_time': cpu_time_sum,
                'action_probs': action_prob.detach().numpy(),
                'w_true': w_true.detach().numpy(),
                'w_model': w_model.clone().detach().numpy(),
            }

            # add entire configuration object to the row
            row = {**row, **config.__dict__}
            records.append(row)

            pred_loss_sum = 0
            lasso_loss_sum = 0
            cycle_loss_sum = 0
            total_loss_sum = 0
            noise_err_sum = 0

            # update cumulative time
            now = time.time()
            cpu_time_sum += now - start_timer
            start_timer = now

            if isinstance(policy, nn.Module):
                row['reward'] = reward_sum / config.log_iters
                reward_sum = 0

    return records
