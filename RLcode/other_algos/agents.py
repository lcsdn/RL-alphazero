import numpy as np
from random import sample

from ..data_structures.tree import DictTreeNode
from ..data_structures.agent import Agent
from .minimax import minimax, tree_minimax, comprehensive_tree_minimax
from .MCTS import MonteCarloTreeSearch

class HumanAgent(Agent):
    """
    Allow the user to play against an algorithmic agent.
    """
    def __init__(self, env, player):
        self.env = env
        self.player = player
    
    def play(self):
        action = int(input("Your play:"))
        while action not in self.env.available_actions():
            action = int(input("Please select another action:"))
        state, reward, done, info = self.env.step(action)
        self.env.render()
        
        return action, None, None, state, reward, done, info
    
    def register_play(self, action):
        print("Your opponent played:", action)
        self.env.render()

class MinimaxAgent(Agent):
    """
    Play the minimax action.
    """
    def __init__(self, env, player):
        self.env = env
        self.player = player
    
    def play(self):
        _, action_path = minimax(self.env, self.player)
        action = action_path[0]
        state, reward, done, info = self.env.step(action)        
        return action, None, None, state, reward, done, info

class TreeMinimaxAgent(Agent):
    """
    Play the minimax action and store the other minimax actions for efficiency.
    """
    def __init__(self, env, player):
        self.env = env
        self.player = player
        self.tree_init()
        self.root = self.tree
    
    def tree_init(self):
        self.tree = DictTreeNode({
            'value': None,
            'action': None,
            'player': 1,
        })
        tree_minimax(self.env, 1, self.tree)
        
    def play(self):
        if self.tree is None:
            self.tree_init()
        action = self.tree['action']
        state, reward, done, info = self.env.step(action)
        self.register_play(action)
        return action, None, None, state, reward, done, info
    
    def register_play(self, action):
        if self.tree is not None:
            self.tree = self.tree.children[action]
            self.tree.parent = None
    
    def reset(self):
        self.tree = self.root
        
class RandomTreeMinimaxAgent(TreeMinimaxAgent):    
    """
    Play the minimax actions at random and store the other minimax actions for efficiency.
    """
    def tree_init(self):
        self.tree = DictTreeNode({
            'value': None,
            'opt_actions': None,
            'player': 1,
        })
        comprehensive_tree_minimax(self.env, 1, self.tree)
            
    def play(self):
        if self.tree is None:
            self.tree_init()
        opt_actions = self.tree['opt_actions']
        action = sample(opt_actions, k=1)[0]
        state, reward, done, info = self.env.step(action)
        self.register_play(action)
        return action, None, None, state, reward, done, info
        
class MCTSAgent(Agent):
    """
    Play an action after a MCTS search.
    """
    def __init__(self, env, player, num_searches=1000, keep_tree=True, **kwargs):
        self.env = env
        self.player = player
        self.num_searches = num_searches
        self.MCTS = None
        self.keep_tree = keep_tree
        self.kwargs = kwargs
    
    def MCTS_init(self):
        self.MCTS = MonteCarloTreeSearch(self.env, self.player, **self.kwargs)
    
    def play(self):
        if self.MCTS is None:
            self.MCTS_init()
            
        self.MCTS.search(self.num_searches)
        actions, policy = self.MCTS.policy()
        action = np.random.choice(actions, p=policy)
        state, reward, done, info = self.env.step(action)
        
        self.register_play(action)
            
        return action, actions, policy, state, reward, done, info
    
    def register_play(self, action):
        if self.MCTS is not None:
            if self.keep_tree and action in self.MCTS.tree.children:
                self.MCTS.tree = self.MCTS.tree.children[action]
                self.MCTS.tree.parent = None
            else:
                self.MCTS = None
    
    def reset(self):
        self.MCTS = None