import gym
import gym.spaces
import torch

class DirectedAcyclicGraph(object):
    def __init__(self, graph):
        # TODO: assert graph is lower triangular. Evaluation of SEM depends on this.
        self.g = graph
        self.g = self.g.long()
        self.dim = graph.shape[1]
        self.depth = graph.shape[0]

    def parents(self, i):
        parents = self.g[:, i].nonzero()
        return parents

class StructuralEquation(object):
    def __init__(self, parents, dim, depth):
        self.w = torch.zeros(dim, depth)
        
        # sample weights for all incomming edges (including trough time)
        for parent in parents:
            t, i = parent
            self.w[i, t] = torch.randn(1)

    def __call__(self, z):
        # w.T = [ [-- incoming weights t-1 -- ]
        #         [-- incoming weights t   -- ] ]
        # z   = [ [ z_0', z_0 ],
        #         [ z_1', z_1 ]
        #         [ z_2', z_3 ]]
        # where sum(diag(w.T @ z)) is sum(w_prev.T * z_prev + w.T * z)
        return self.w.t().matmul(z).diag().sum() + torch.randn(1)

class StructuralEquationModel(object):
    def __init__(self, dag):

        # NB: don't want randomness during development.
        torch.manual_seed(1)
        # # # # #

        self.graph = dag
        self.dim = dag.dim
        self.depth = dag.depth

        self.functions = []
        for i in range(self.dim):
            parents_i = self.graph.parents(i)
            self.functions.append(StructuralEquation(parents_i, self.dim, self.depth))

    def __call__(self, n, z_prev=None, intervention=None):

        z = torch.zeros(n+1, self.dim)

        # initial state
        z[0] = z_prev if z_prev is not None else torch.randn(self.dim)

        for t in range(1, n+1):
            for j in range(self.dim):
                if intervention == j:
                    z[t, j] = 1 # variabje j is only determined by action
                else:
                    # prepare inputs, see StructuralEquation.__call__ for details
                    z_ = torch.stack([z[t-1], z[t]], dim=1)
                    z[t, j] = self.functions[j](z_)

        # return states (excluding previous state)
        return z[1:]

class SemEnv(gym.Env):

    def __init__(self, output_size=5, mode='easy'):

        graph = torch.tensor([
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],

            [[0, 0, 0],
             [1, 0, 0],
             [1, 1, 0]]
        ])
        
        self.time_limit = 1

        # represents causal relations in Z
        self.graph = DirectedAcyclicGraph(graph)
        self.causal_model = StructuralEquationModel(self.graph)

        # transforms Z into S
        self.generative_model = torch.nn.Linear(self.graph.dim, output_size)
        for param in self.generative_model.parameters():
            param.requires_grad = False
    
    def reset(self):
        self.t = 0

        self.state = self.causal_model(1).squeeze()
        self.prev_state = None

        return self._observation()

    def step(self, action):
        self.t = self.t + 1
        
        # update state Z
        self.prev_state = self.state
        self.state = self.causal_model(1, z_prev=self.prev_state, intervention=action).squeeze()
        
        # check deadline
        done = self.t == self.time_limit

        return self._observation(), self._reward(action), done, {}

    def _observation(self):
        """
        Maps internal state Z to observation S
        """
        return self.generative_model(self.state)

    def _reward(self, action):
        """
        Returns reward based on the internal state of the world and the most recent action
        """
        return torch.sum(self.state, dim=0).item()