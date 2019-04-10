import torch

chain = torch.tensor([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0]]
])

common_cause = torch.tensor([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
])

common_effect = torch.tensor([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]]
])

classic_confounding = torch.tensor([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [1, 0, 0],
     [1, 1, 0]]
])

linear_mountaincar = torch.tensor([
    [[1, 1, 0],
     [1, 1, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 1, 0]]
])

independent = torch.tensor([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
])

causal_models = {
    'chain' : chain,
    'common_cause' : common_cause,
    'common_effect' : common_effect,
    'classic_confounding' : classic_confounding,
    'linear_mountaincar' : linear_mountaincar,
    'independent' : independent
}