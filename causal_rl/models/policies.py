import torch
import torch.nn as nn


class SimplePolicy(nn.Module):
    def __init__(self, **kwargs):
        super(SimplePolicy, self).__init__()
        n_actions = kwargs.get('n_actions')

        self.action_weight = nn.Parameter(torch.randn(n_actions))

    def forward(self, *args):
        action_logprob = torch.log_softmax(self.action_weight, dim=-1)
        return action_logprob


class LinearPolicy(nn.Module):
    def __init__(self, **kwargs):
        super(LinearPolicy, self).__init__()
        dim = kwargs.get('dim')
        n_actions = kwargs.get('n_actions')
        self.linear = nn.Linear(dim, n_actions)

    def forward(self, *args):
        x = args[0]
        return torch.log_softmax(self.linear(x), dim=-1).squeeze(0)


class RandomPolicy():
    def __init__(self, **kwargs):
        n = kwargs.get('n_actions')
        self.weights = torch.ones(n)

        # only sample from the provided subset
        self.subset = kwargs.get('subset', torch.ones(n))
        self.probs = torch.tensor(0.5).repeat(n) * self.subset.float()
        self.dist = torch.distributions.Bernoulli(probs=self.probs)

        self.action_probs = 0.5**torch.sum(self.subset).repeat(n)

    def __call__(self, *args):
        return self.dist.sample().byte()


class IntrospectivePolicy(nn.Module):
    def __init__(self, **kwargs):
        super(IntrospectivePolicy, self).__init__()
        dim = kwargs.get('dim')
        n = kwargs.get('n_actions')

        self.linear = nn.Linear(dim*dim, n)

    def forward(self, adjecency_matrix):
        inp = adjecency_matrix.view(-1)
        out = self.linear(inp)

        return torch.log_softmax(out, dim=-1)


policies = {
    'simple': SimplePolicy,
    'linear': LinearPolicy,
    'random': RandomPolicy,
    'non_sink': RandomPolicy,
    'sink': RandomPolicy,
    'parent': RandomPolicy,
    'non_parent': RandomPolicy,
    'action_0': RandomPolicy,
    'action_1': RandomPolicy,
    'action_2': RandomPolicy,
    'action_3': RandomPolicy,
    'action_4': RandomPolicy,
    'action_5': RandomPolicy,
    'action_6': RandomPolicy,
    'action_1_4': RandomPolicy,
    'introspective': IntrospectivePolicy
}
