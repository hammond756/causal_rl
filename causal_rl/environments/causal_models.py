import torch

chain = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0]]
)


def n_chain(n):
    return torch.diag(torch.ones(n-1), -1)


confounder = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [1, 0, 0]]
)

collider = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [1, 1, 0]]
)

shielded_collider = torch.tensor(
    [[0, 0, 0],
     [1, 0, 0],
     [1, 1, 0]]
)

independent = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
)

disconnected = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [0, 1, 0]]
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
    'chain': n_chain(3),
    'chain_2': n_chain(2),
    'chain_4': n_chain(4),
    'chain_5': n_chain(5),
    'collider': collider,
    'confounder': confounder,
    'shielded_collider': shielded_collider,
    'independent': independent,
    'disconnected': disconnected,


}

specified_common_effect = torch.tensor(
    [[0, 0, 0],
     [0, 0, 0],
     [-0.1316, -0.7984, 0]]
)

specified_stacked_chain = torch.tensor(
    [[-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000],
     [-0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000],
     [1.4068,  0.8494, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000],
     [0.6812, -0.3108, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
     [0.0000, -0.0000, -0.6588, -1.3930,  0.0000,  0.0000, -0.0000],
     [-0.0000,  0.0000, -0.0000, -0.0000,  0.1181, -0.0000,  0.0000],
     [-0.0000,  0.0000, -0.0000, -0.0000, -2.1204, -0.0000, -0.0000]]
)

causal_models = {
    'specified_common_effect': specified_common_effect,
    'specified_chain': chain,
    'specified_stacked_chain': specified_stacked_chain
}
