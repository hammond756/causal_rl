import torch
import torch.nn as nn


class SimplePolicy(nn.Module):
    def __init__(self, **kwargs):
        super(SimplePolicy, self).__init__()
        dim = kwargs.get('dim')
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self, *args):
        action_logprob = torch.log_softmax(self.action_weight, dim=-1)
        return action_logprob


class LinearPolicy(nn.Module):
    def __init__(self, **kwargs):
        super(LinearPolicy, self).__init__()
        dim = kwargs.get('dim')
        self.linear = nn.Linear(dim, dim)

    def forward(self, *args):
        x = args[0]
        return torch.log_softmax(self.linear(x), dim=-1).squeeze(0)


class RandomPolicy():
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim')

    def __call__(self, *args):
        action_probs = torch.tensor([1. / self.dim for _ in range(self.dim)])
        return torch.log(action_probs)


class CyclicPolicy():
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim')
        self.position = 0

    def __call__(self, *args):
        action_probs = torch.tensor([1. if i == self.position else 0.
                                     for i in range(self.dim)])

        return torch.log(action_probs)


class ChildPolicy():
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim')
        self.child_idxs = kwargs.get('child_idxs')

    def __call__(self, *args):
        n = len(self.child_idxs)
        action_probs = torch.tensor([1. / n if i in self.child_idxs else 0.
                                     for i in range(self.dim)])

        return torch.log(action_probs)


class RootPolicy():
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim')
        self.root_idxs = kwargs.get('root_idxs')

    def __call__(self, *args):
        n = len(self.root_idxs)
        action_probs = torch.tensor([1. / n if i in self.root_idxs else 0.
                                     for i in range(self.dim)])

        return torch.log(action_probs)


class IntrospectivePolicy(nn.Module):
    def __init__(self, **kwargs):
        super(IntrospectivePolicy, self).__init__()
        self.dim = kwargs.get('dim')
        self.linear = nn.Linear(self.dim*self.dim, self.dim)

    def forward(self, adjecency_matrix):
        inp = adjecency_matrix.view(-1)
        out = self.linear(inp)

        return torch.log_softmax(out, dim=-1)


policies = {
    'simple': SimplePolicy,
    'random': RandomPolicy,
    'linear': LinearPolicy,
    'child': ChildPolicy,
    'root': RootPolicy,
    'introspective': IntrospectivePolicy
}
