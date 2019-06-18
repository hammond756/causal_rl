import torch
import torch.nn as nn

class SimplePolicy(nn.Module):
    def __init__(self, dim):
        super(SimplePolicy, self).__init__()
        self.action_weight = nn.Parameter(torch.randn(dim))

    def forward(self):
        action_logprob = torch.log_softmax(self.action_weight, dim=-1)
        return action_logprob

class RandomPolicy():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self):
        action_idx = torch.randint(0, self.dim, (1,)).long().item()
        action_probs = torch.tensor([float(i == action_idx) for i in range(self.dim)])
        return torch.log(action_probs)

policies = {
    'simple' : SimplePolicy,
    'random' : RandomPolicy
}