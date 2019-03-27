import torch

class DirectedAcyclicGraph(object):
    def __init__(self, weights):
        self._weights = weights.float()
        self._validate()

        self.dim = weights.shape[1]
        self.depth = weights.shape[0]

    def parents(self, i):
        return self._weights[:, i].nonzero()
    
    def weights(self, i):
        return self._weights[:, i]

    @property
    def edges(self):
        edges = torch.zeros_like(self._weights)
        for i,j,k in self._weights.nonzero():
            edges[i,j,k] = 1
        
        return edges.long()

    def _validate(self):
        """
        Assure that:
         - adjecency matrix for current time step is lower triangular
        """
        assert (self.edges[-1].triu() == torch.zeros_like(self.edges[-1])).all()

class StructuralEquation(object):
    """
    Represents a structural equation in the structural equation model.

    Args:
     - weights: a 2D tensor of weights. Shape should be TimeSteps x Dimension
    """
    def __init__(self, weights, noise):
        self.w = torch.tensor(weights, dtype=torch.float)
        self.noise = noise

    def __call__(self, z):
        # w   = [ [-- incoming weights t-1 -- ]
        #         [-- incoming weights t   -- ] ]
        # z   = [ [ z_0', z_0 ],
        #         [ z_1', z_1 ]
        #         [ z_2', z_3 ]]
        # where sum(diag(w.T @ z)) is sum(w_prev.T * z_prev + w.T * z)
        if self.noise:
            noise = torch.randn(1)
        else:
            noise = 0
        return self.w.matmul(z).diag().sum() + noise
        
class StructuralEquationModel(object):
    def __init__(self, graph, noise=False):

        self.graph = graph
        self.dim = graph.dim
        self.depth = graph.depth

        self.functions = [StructuralEquation(graph.weights(i), noise) for i in range(self.dim)]

    def __call__(self, n, z_prev=None, intervention=None):

        z = torch.zeros(n+1, self.dim)

        # initial state
        z[0] = z_prev if z_prev is not None else torch.randn(self.dim)

        # invervention
        if intervention is not None:
            inter_target, inter_value = intervention
        else:
            inter_target = None

        for t in range(1, n+1):
            for j in range(self.dim):
                if j == inter_target:
                    z[t, j] = inter_value # variabje j is only determined by action
                else:
                    # prepare inputs, see StructuralEquation.__call__ for details
                    z_ = torch.stack([z[t-1], z[t]], dim=1)
                    z[t, j] = self.functions[j](z_)

        # return states (excluding previous state)
        return z[1:]
    
    @classmethod
    def random_with_edges(self, edges):
        edges = edges.float()
        weights = torch.randn_like(edges)
        for i in range(edges.shape[0]):
            weights[i] = weights[i] * edges[i]
        
        graph = DirectedAcyclicGraph(weights)
        return StructuralEquationModel(graph, noise=True)
    
    @classmethod
    def random(self, dim, p_sparsity):

        assert dim == int(dim), "Structural equation 'dim' should be a whole number."
        dim = int(dim)

        g_t = torch.ones(dim, dim).tril(-1)
        g_t *= torch.zeros(dim, dim).bernoulli_(1 - p_sparsity)

        # random SEM won't have recurrent connection for now
        # TODO: make this an option
        g_prev = torch.zeros_like(g_t)

        g = torch.stack([g_prev, g_t]).long()

        return self.random_with_edges(g)