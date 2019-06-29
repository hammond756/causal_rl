import torch


class DirectedAcyclicGraph(object):
    def __init__(self, weights):
        self._weights = weights.float()
        self._validate()

        self.dim = weights.shape[1]
        self.depth = weights.shape[0]

    def parents(self, i):
        return self._weights[:, i].nonzero()

    def incoming_weights(self, i):
        return self._weights[:, i]

    @property
    def weights(self):
        return self._weights

    @property
    def _roots(self):
        diagonal = torch.zeros(self.dim)
        for i in range(self.dim):
            if self.parents(i).nelement() == 0:
                diagonal[i] = 1

        return torch.diag(diagonal)

    @property
    def _root_indices(self):
        idxs = [i for i in range(self.dim) if self.parents(i).nelement() == 0]
        return torch.tensor(idxs)

    @property
    def _child_indices(self):
        idxs = [i for i in range(self.dim) if self.parents(i).nelement() > 0]
        return torch.tensor(idxs)

    @property
    def edges(self):
        edges = torch.zeros_like(self._weights)
        for i, j, k in self._weights.nonzero():
            edges[i, j, k] = 1

        return edges.long()

    def _validate(self):
        """
        Assure that:
         - adjecency matrix for current time step is lower triangular
        """
        istril = (self.edges[-1].triu(1) == torch.zeros_like(self.edges[-1])) \
            .all()
        assert istril, 'Graph is not lower triangular'


class StructuralEquation(object):
    """
    Represents a structural equation in the structural equation model.

    Args:
     - weights: a 2D tensor of weights. Shape should be TimeSteps x Dimension
    """
    def __init__(self, weights):
        self.w = torch.tensor(weights, dtype=torch.float)

    def __call__(self, z):
        # w   = [ [-- incoming weights t-1 -- ]
        #         [-- incoming weights t   -- ] ]
        # z   = [ [ z_0', z_0 ],
        #         [ z_1', z_1 ]
        #         [ z_2', z_3 ]]
        # where sum(diag(w.T @ z)) is sum(w_prev.T * z_prev + w.T * z)
        return self.w.matmul(z).diag().sum()


class Noise(object):
    def __init__(self, distribution, param):

        if distribution == 'gaussian':
            self.sample_func = lambda: torch.randn(1) * param

        if distribution == 'bernoulli':
            self.sample_func = lambda: torch.zeros(1).bernoulli_(param)

    def __call__(self):
        return self.sample_func()


class StructuralEquationModel(object):
    def __init__(self, graph, noise_dist, noise_param):

        self.graph = graph
        self.dim = graph.dim
        self.depth = graph.depth

        self.funcs = []
        for i in range(self.dim):
            self.funcs.append(StructuralEquation(graph.incoming_weights(i)))

        self.sample_noise = Noise(noise_dist, noise_param)
        self.noises = torch.zeros(self.dim)

    @property
    def roots(self):
        return self.graph._roots

    @property
    def child_idxs(self):
        return self.graph._child_indices

    @property
    def root_idxs(self):
        return self.graph._root_indices

    def counterfactual(self, z_prev=None, intervention=None):
        return self._sample(1, z_prev, intervention, fix_noise=True)

    def _sample(self, n, z_prev=None, intervention=None, fix_noise=False):
        z = torch.zeros(n+1, self.dim)

        # initial state
        z[0] = z_prev if z_prev is not None else torch.randn(self.dim)

        # intervention
        if intervention is not None:
            inter_target, inter_value = intervention
        else:
            inter_target = None

        for t in range(1, n+1):
            for j in range(self.dim):
                if j == inter_target:
                    # variabje j is only determined by action
                    z[t, j] = inter_value
                else:
                    # prepare inputs, see StructuralEquation for details
                    z_ = torch.stack([z[t-1], z[t]], dim=1)
                    z[t, j] = self.funcs[j](z_)

                    # add noise, either from infered values (counterfactual)
                    # or sample new
                    # NB: needs to be done in inner loop (sampling order).
                    if fix_noise:
                        z[t, j] += self.noises[j]
                    else:
                        noise = self.sample_noise()
                        z[t, j] += noise.item()
                        self.noises[j] = noise

        # return states (excluding previous state)
        return z[1:]

    def __call__(self, n, z_prev=None, intervention=None):
        return self._sample(n, z_prev, intervention, fix_noise=False)

    @classmethod
    def random_with_edges(self, edges, noise_dist, noise_param):
        edges = edges.float()
        weights = torch.randn_like(edges)
        for i in range(edges.shape[0]):
            weights[i] = weights[i] * edges[i]

        graph = DirectedAcyclicGraph(weights)
        return StructuralEquationModel(graph, noise_dist, noise_param)

    @classmethod
    def random(self, dim, p_sparsity, noise_dist, noise_param):

        assert dim == int(dim), """Structural equation 'dim' should
            be a whole number."""

        dim = int(dim)

        g_t = torch.ones(dim, dim).tril(-1)
        g_t *= torch.zeros(dim, dim).bernoulli_(1 - p_sparsity)

        # random SEM won't have recurrent connection for now
        # TODO: make this an option
        g_prev = torch.zeros_like(g_t)

        g = torch.stack([g_prev, g_t]).long()

        return self.random_with_edges(g, noise_dist, noise_param)
