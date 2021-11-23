from copy import deepcopy
import numpy as np

from ..data_structures.tree import DictTreeNode

class MonteCarloTreeSearch:
    """
    Base MCTS using Upper Confidence Bound applied to tree (UCT).
    """
    def __init__(self, env, player, exploration_param=1, max_depth=None):
        self.env = env
        self.tree = DictTreeNode({'value': 0, 'num_samples': 0, 'player': player})
        self.exploration_param = exploration_param
        self.max_depth = max_depth
    
    @staticmethod
    def naive_select_child(parent):
        action = next(iter(parent.children))
        return action, parent.children[action]
    
    @staticmethod
    def random_select_child(parent):
        from random import sample
        action = sample(parent.children.keys(), 1)[0]
        return action, parent.children[action]

    def select_child(self, node):
        """
        Select children according to Upper Confidence bound applied to Trees (UCT).
        """
        best_UCT = None
        for action, child in node.children.items():
            # If child has never been visited then UCT is infinite
            if child['num_samples'] == 0:
                best_action, best_child = action, child
                break
            error_margin = np.sqrt(2 * np.log(node['num_samples']) / child['num_samples'])
            UCT = - child['value'] + self.exploration_param * error_margin
            if best_UCT is None or UCT > best_UCT:
                best_UCT, best_action, best_child = UCT, action, child
            
        return best_action, best_child
    
    def selection(self, simulator, node, depth):
        """
        Run down the tree and return a leaf.
        Branches are selected according to criterion defined in self.select_child
        """
        if len(node.children) == 0 or depth == self.max_depth:
            return node, False, None
        
        action, child = self.select_child(node)
        state, reward, done, info = simulator.step(action)
        
        if done:
            return child, done, reward
        
        return self.selection(simulator, child, depth + 1)
    
    @staticmethod
    def expansion(simulator, leaf):
        """
        Expand the leaf found after selection step by adding new children.
        """
        player = leaf['player']
        leaf.add_children_values_dict({
            action: {
                'value': 0,
                'num_samples': 0,
                'player': - player
            }
            for action in simulator.available_actions()
        })
        return leaf.children[simulator.available_actions()[0]]
    
    @staticmethod
    def simulation(simulator):
        """
        Simulate until end of game.
        """
        done = False
        while not done:
            next_action = np.random.choice(simulator.available_actions())
            state, reward, done, info = simulator.step(next_action)
        return reward
    
    @staticmethod
    def backup(leaf, reward):
        """
        Update the values in the tree from leaf to root using obtained reward.
        """
        node = leaf
        while node is not None:
            node['num_samples'] += 1
            node['value'] = node['value'] + (node['player'] * reward - node['value']) / node['num_samples']
            node = node.parent
    
    def search_step(self):
        simulator = deepcopy(self.env)
        leaf, done, reward = self.selection(simulator, self.tree, 0)
        if not done:
            leaf = self.expansion(simulator, leaf)
            reward = self.simulation(simulator)
        self.backup(leaf, reward)
        return simulator, reward
    
    def search(self, num_searches):
        for i in range(num_searches):
            self.search_step()
    
    def policy(self):
        children_num_samples = np.array([child['num_samples'] for child in self.tree.children.values()])
        policy = children_num_samples / children_num_samples.sum()
        return list(self.tree.children.keys()), policy
        
