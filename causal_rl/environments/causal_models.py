import torch

chain = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0]]
)

common_cause = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
)

common_effect = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]]
)

classic_confounding = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [1, 1, 0]]
)

independent = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
)

stacked_chain = torch.tensor(
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 0]]
)

directed_edges = {
    'chain': chain,
    'common_cause': common_cause,
    'common_effect': common_effect,
    'classic_confounding': classic_confounding,
    'independent': independent,
    'stacked_chain': stacked_chain
}

specified_common_effect = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [-0.1316, -0.7984, 0]]
)

specified_stacked_chain = torch.tensor(
    [[-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000],
    [-0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000],
    [ 1.4068,  0.8494, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000],
    [ 0.6812, -0.3108, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
    [ 0.0000, -0.0000, -0.6588, -1.3930,  0.0000,  0.0000, -0.0000],
    [-0.0000,  0.0000, -0.0000, -0.0000,  0.1181, -0.0000,  0.0000],
    [-0.0000,  0.0000, -0.0000, -0.0000, -2.1204, -0.0000, -0.0000]]
)

causal_models = {
    'specified_common_effect': specified_common_effect,
    'specified_chain': chain,
    'specified_stacked_chain': specified_stacked_chain
}