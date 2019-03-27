import matplotlib.pyplot as plt
import torch

def bar(ax, X, bar_width=0.25, labels=None):

    if X.dim() == 1:
        n_items = len(X)
        ax.bar(torch.arange(n_items), X, width=bar_width)
    else:

        pos = torch.zeros_like(X)
        n_items = X.shape[1]

        pos[0, :] = torch.arange(n_items)
        for idx in range(1, X.shape[0]):
            pos[idx, :] = torch.tensor([x + bar_width for x in pos[idx - 1, :]])
        
        for idx in range(X.shape[0]):
            ax.bar(pos[idx, :], X[idx, :], width=bar_width)
    
    if labels is not None:
        assert len(labels) == n_items, "Number of labels doens't match data"
        plt.sca(ax)
        plt.xticks([x for x in range(n_items)], labels)
