import gym
import gym.spaces
import torch
from causal_rl.sem import StructuralEquationModel, DirectedAcyclicGraph

class SemEnv(gym.Env):

    def __init__(self, model, actions):
        
        self.time_limit = 100

        # represents causal relations in Z
        self.causal_model = model
        # set environment properties
        # TODO: observation space
        self.action_space = gym.spaces.Discrete(len(actions))
        self.actions = actions

        # transforms Z into S
        self.generative_model = self._generative_model()

    @classmethod
    def with_weights(self, weights, actions):
        graph = DirectedAcyclicGraph(weights)
        model = StructuralEquationModel(graph)
        return SemEnv(model, actions)

    @classmethod
    def with_edges(self, edges, actions):
        model = StructuralEquationModel.random_with_edges(edges)
        return SemEnv(model, actions)
    
    def reset(self):
        self.prev_state = self._initial_state()
        self.state = self.causal_model(1, z_prev=self.prev_state).squeeze()

        return self.generative_model(self.state)

    def step(self, action_idx):

        intervention = self.actions[action_idx]
        
        # update state Z
        self.prev_state = self.state
        self.state = self.causal_model(1, z_prev=self.prev_state, intervention=intervention).squeeze()
        
        return self.generative_model(self.state), self._reward(intervention), self._done(), {}
    
    @classmethod
    def _generative_model(self):
        """
        Returns the generative model. Each subclass should implement its own version.
        """        
        raise NotImplementedError()

    @classmethod
    def _inital_state(self):
        """
        Returns the initial state. Default is None.
        """        
        return None

    def _done(self):
        """
        Returns the value of the termination criterion
        """
        return NotImplementedError()

    def _reward(self, action):
        """
        Returns reward based on the internal state of the world and the most recent action
        """
        return NotImplementedError()