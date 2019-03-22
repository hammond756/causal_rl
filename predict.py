import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom

def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


class DirectedAcyclicGraph(object):
    def __init__(self, dim, p_sparsity):
        self.g = torch.ones(dim, dim).tril(-1)
        self.g *= torch.zeros(dim, dim).bernoulli_(1 - p_sparsity)
        self.g = self.g.long()

    def parents(self, i):
        return self.g[i].nonzero().view(-1)


class StructuralEquation(object):
    def __init__(self, parents, dim):
        self.w = torch.randn(dim)

        for i in range(dim):
            if i not in parents:
                self.w[i] = 0

    def __call__(self, x):
        return (self.w * x).sum(1, keepdim=True)


class Noise(object):
    def __init__(self, vmin=0.1, vmax=2):
        self.v = torch.zeros(1).uniform_(vmin, vmax).item()

    def __call__(self, n):
        return torch.randn(n) * self.v


class StructuralEquationModel(object):
    def __init__(self, dim, p_sparsity=0.2):
        super(StructuralEquationModel, self).__init__()
        self.d = dim + 1
        self.g = DirectedAcyclicGraph(self.d, p_sparsity)

        self.f = []
        self.n = []
        for i in range(self.d):
            self.f.append(StructuralEquation(self.g.parents(i), self.d))
            self.n.append(Noise())

        i_all = torch.arange(self.d)
        self.y = torch.zeros(1).random_(self.d).int().item()
        self.not_y = torch.cat([i_all[0:self.y], i_all[self.y + 1:]])

    def solution(self):
        return self.f[self.y].w[self.not_y]

    def __call__(self, n=1, intervention=None):
        data = torch.zeros(n, self.d)

        for i in range(self.d):
            if i == intervention and i != self.y:
                data[:, i] = 0
            else:
                data[:, i] = self.f[i](data).view(-1) + self.n[i](n)

        return data[:, self.not_y], data[:, self.y]


def random_policy(dim):
    return torch.randint(0, dim, (1,)).item()

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = F.log_softmax(self.action_weight, dim=-1)
        return action_logprob


if __name__ == "__main__":
    torch.manual_seed(0)
    vis = visdom.Visdom()

    dim = 10
    n_iterations = 50000
    log_iters = 1000
    use_random_policy = False
    entr_loss_coeff = 1

    predictor = torch.nn.Linear(dim, 1, bias=False)
    optimizer = torch.optim.Adam(predictor.parameters())
    sem = StructuralEquationModel(dim)
    if not use_random_policy:
        policy = SimplePolicy(dim)
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
    for iteration in range(n_iterations):
        if use_random_policy:
            action = random_policy(dim)
        else:
            action_logprob = policy()
            action_prob = action_logprob.exp()
            action = torch.multinomial(action_prob, 1).item()

        x, y = sem(intervention=action)

        optimizer.zero_grad()
        loss = (predictor(x) - y).pow(2)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        if not use_random_policy:
            policy_optim.zero_grad()
            reward = loss.item() / 10
            reward_sum += reward
            log_prob = action_logprob[action]
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
            w_true = sem.solution().view(-1)
            w_model = predictor.weight.view(-1)
            print("SAMPLE", pretty(x.view(-1)))
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
