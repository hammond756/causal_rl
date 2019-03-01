import gym
import gym.spaces
import torch

class DirectedAcyclicGraph(object):
    def __init__(self, graph):
        self.g = graph
        self.g = self.g.long()
        self.dim = graph.shape[0]

    def parents(self, i):
        return self.g[i].nonzero().view(-1).tolist()

class StructuralEquation(object):
    def __init__(self, parents, dim):
        self.w = torch.randn(dim)

        for i in range(dim):
            if i not in parents:
                self.w[i] = 0

    def __call__(self, x):
        return (self.w * x).sum(1, keepdim=True) + torch.randn(1)

class Noise(object):
    def __init__(self, vmin=0.1, vmax=5):
        self.v = torch.zeros(1).uniform_(vmin, vmax).item()

    def __call__(self, n):
        return torch.randn(n) * self.v


class StructuralEquationModel(object):
    def __init__(self, dag):

        torch.manual_seed(1)
        
        self.graph = dag
        self.dim = dag.dim

        self.functions = []
        self.noises = []
        for i in range(self.dim):
            parents_i = self.graph.parents(i)
            self.functions.append(StructuralEquation(parents_i, self.dim))
            self.noises.append(Noise())

    def __call__(self, n, intervention=None):

        data = torch.zeros(n, self.dim)

        for i in range(n):
            for j in range(self.dim):
                if intervention == j:
                    data[i, j] = 1 # variabje j is only determined by action
                else:
                    data[i, j] = self.functions[j](data[i].view(1, -1))

        return data

class SemEnv(gym.Env):

    def __init__(self, output_size=5, mode='easy'):

        graph = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        
        self.dim = graph.shape[0]
        self.time_limit = 30

        # represents causal relations in Z
        self.graph = DirectedAcyclicGraph(graph)
        self.causal_model = StructuralEquationModel(self.graph)

        # transforms Z into S
        self.generative_model = torch.nn.Linear(self.dim, output_size)
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
        self.state = self.causal_model(1, intervention=action).squeeze()
        
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