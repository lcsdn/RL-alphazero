import numpy as np
import torch

from ..data_structures.agent import Agent
from .network import DEVICE
from .MCTS import NeuralMCTS

class NeuralAgent(Agent):
    """
    Agent using the AlphaZero algorithm with input model to play.
    """
    def __init__(self, env, model, player, num_searches=1000, keep_tree=True, add_noise=True, num_hot_plays=10, **kwargs):
        self.env = env
        self.model = model
        self.player = player
        self.num_searches = num_searches
        self.kwargs = kwargs
        self.MCTS = None
        self.keep_tree = keep_tree
        self.add_noise = add_noise
        self.dirichlet_param = 1 / 9
        self.dirichlet_weight = 0.25
        self.num_hot_plays = num_hot_plays
        self.counter = 10
        self.temperature = self.kwargs.get('temperature', 1)
    
    def add_dirichlet_noise(self):
        if self.MCTS.tree['prior_policy'] is not None: # TODO why does that happen?
            param = [self.dirichlet_param] * len(self.MCTS.tree['prior_policy'])
            noise = np.random.dirichlet(param, size=1)
            noise = torch.from_numpy(noise).squeeze()
            if DEVICE == 'cuda':
                noise = noise.cuda()
            self.MCTS.tree['prior_policy'] *= (1 - self.dirichlet_weight)
            self.MCTS.tree['prior_policy'] += self.dirichlet_weight * noise
    
    def MCTS_init(self):
        self.MCTS = NeuralMCTS(self.env, self.model, self.player, **self.kwargs)
    
    def play(self):
        if self.MCTS is None:
            self.MCTS_init()
        
        if self.add_noise:
            self.add_dirichlet_noise()

        self.MCTS.search(self.num_searches)
        actions, policy = self.MCTS.policy()
        chosen_action = np.random.choice(actions, p=policy)
        state, reward, done, info = self.env.step(chosen_action)
        
        self.register_play(chosen_action)
        
        if self.counter >= self.num_hot_plays:
            self.kwargs['temperature'] = 0.01
            self.MCTS.inv_temp = 1 / 0.01
            
        return chosen_action, actions, policy, state, reward, done, info
    
    def register_play(self, action):
        if self.MCTS is not None:
            if self.keep_tree and action in self.MCTS.tree.children:
                self.MCTS.tree = self.MCTS.tree.children[action]
                self.MCTS.tree.parent = None
            else:
                self.MCTS = None
    
    def reset(self):
        self.counter = 0
        self.kwargs['temperature'] = self.temperature
        self.MCTS = None