import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = torch.nn.functional.log_softmax(self.action_weight, dim=-1)
        return action_logprob

class RandomPolicy():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self):
        return torch.randint(0, self.dim, (1,)).long().item()

policies = {
    'simple' : SimplePolicy,
    'random' : RandomPolicy
}