import torch

from causal_rl.environments import SemEnv
from causal_rl.sem import StructuralEquationModel, DirectedAcyclicGraph

class PredatorPrey(SemEnv):

    def __init__(self):

        a = 0.05 # predator population on predator births
        b = 0.05 # predator population on predator deaths
        c = 0.2  # predator popoulation on prey deaths
        d = 0.10 # prey population on prey births
        e = 0.05 # prey population on prey deaths
        l = 0.07 # prey population on predator births
        f = 1 # predator births on predator population (time)
        g = 1 # prey births on prey population (time)
        h = -1 # predator deaths on predator population (time)
        i = -1 # prey deaths on prey population (time)
        j = 1 # predator population on predator population (time)
        k = 1 # prey population on prey population (time)
        
        weights = torch.tensor([
            [[j,0,f,0,h,0,0,0,0,0],  # predator population   t-1
             [0,k,0,g,0,i,0,0,0,0],  # prey population       t-1
             [0,0,0,0,0,0,0,0,0,0],  # predator births       t-1
             [0,0,0,0,0,0,0,0,0,0],  # prey births           t-1
             [0,0,0,0,0,0,0,0,0,0],  # predator deaths       t-1
             [0,0,0,0,0,0,0,0,0,0],  # prey deaths           t-1

             # action variables
             [0,0,1,0,0,0,0,0,0,0],  # predator birth act    t-1
             [0,0,0,1,0,0,0,0,0,0],  # prey birth act        t-1
             [0,0,0,0,1,0,0,0,0,0],  # predator death act    t-1
             [0,0,0,0,0,1,0,0,0,0]], # prey death act        t-1

            [[0,0,0,0,0,0,0,0,0,0],  # predator population   t
             [0,0,0,0,0,0,0,0,0,0],  # prey population       t
             [a,l,0,0,0,0,0,0,0,0],  # predator births       t
             [0,d,0,0,0,0,0,0,0,0],  # prey births           t
             [b,0,0,0,0,0,0,0,0,0],  # predator deaths       t
             [c,e,0,0,0,0,0,0,0,0],  # prey deaths           t

            # action variables
             [0,0,0,0,0,0,0,0,0,0],  # predator birth act    t
             [0,0,0,0,0,0,0,0,0,0],  # prey birth act        t
             [0,0,0,0,0,0,0,0,0,0],  # predator death act    t
             [0,0,0,0,0,0,0,0,0,0]], # prey death act        t
        ])

        graph = DirectedAcyclicGraph(weights)
        model = StructuralEquationModel(graph)
        actions = [(6,1), (7,1), (8,1), (9,1), (6,-1), (7,-1), (8,-1), (9,-1), None]

        super(PredatorPrey, self).__init__(model, actions)

    def _initial_state(self):
        initial = torch.tensor([50., 50., 0., 0., 0., 0., 0., 0., 0., 0.]) # 10 predator, 50 prey

        # calculate birth/deaths
        for i in range(3, self.causal_model.graph.dim):
            initial[i] = self.causal_model.functions[i](initial)
        
        return initial

    @classmethod
    def _generative_model(self):
        return lambda z: z[:2] # only observe population sizes
    
    def _done(self):
        population = self.generative_model(self.state)
        return torch.all(population == 0)
    
    def _reward(self, action):
        intervention_penalty = 0 if action is None else -1
        return -1 + intervention_penalty