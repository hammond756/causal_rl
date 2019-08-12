import torch


class DirectedAcyclicGraph(object):
    def __init__(self, weights):
        self._weights = weights.float()
        self._validate()

        self.dim = weights.shape[1]
        self.depth = weights.shape[0]

    def parents(self, i):
        return self._weights[i, :].nonzero()

    def children(self, i):
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
    def _sink_mask(self):
        mask = [self.children(i).nelement() == 0 for i in range(self.dim)]
        return torch.tensor(mask)

    @property
    def _non_sink_mask(self):
        return 1 - self._sink_mask

    @property
    def edges(self):
        edges = torch.zeros_like(self._weights)
        for j, k in self._weights.nonzero():
            edges[j, k] = 1

        return edges.long()

    def _validate(self):
        """
        Assure that:
         - adjecency matrix for current time step is lower triangular
        """
        istril = (self.edges.triu(1) == torch.zeros_like(self.edges)) \
            .all()
        assert istril, 'Graph is not lower triangular'


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

        self.sample_noise = Noise(noise_dist, noise_param)
        self.noise = torch.zeros(1, self.dim)

    @property
    def roots(self):
        return self.graph._roots

    @property
    def child_idxs(self):
        return self.graph._child_indices

    @property
    def root_idxs(self):
        return self.graph._root_indices

    @property
    def non_sink_mask(self):
        return self.graph._non_sink_mask

    @property
    def sink_mask(self):
        return self.graph._sink_mask

    def counterfactual(self, z_prev=None, intervention=None):
        return self._sample(1, z_prev, intervention, fix_noise=True)

    def _sample(self, n, z_prev=None, intervention=None, fix_noise=False):

        z = torch.cat([torch.zeros(self.dim), torch.tensor([1.])]).unsqueeze(0)

        if fix_noise:
            noise = self.noise
        else:
            noise = torch.stack([self.sample_noise()
                                 for _ in range(self.dim)], dim=1)

        self.noise = noise

        weights = torch.cat([self.graph.weights, noise.t()], dim=1)

        # perform intervention on the model
        if intervention is not None:
            targets, value = intervention
            do_x = torch.tensor([0. for _ in range(self.dim)] + [value])
            weights[targets] = do_x

        model = torch.cat([weights, z], dim=0)

        result = z.clone().t()

        for i in range(self.dim):
            result = model.matmul(result)

        return result.t()[:, :-1]

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
        g_t = g_t.long()

        return self.random_with_edges(g_t, noise_dist, noise_param)
