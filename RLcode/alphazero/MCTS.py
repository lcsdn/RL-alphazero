from copy import deepcopy
import numpy as np
import torch

from ..data_structures.tree import DictTreeNode
from .functional import constrained_softmax
from .network import DEVICE

class NeuralMCTS:
    """
    Monte Carlo Tree Search guided by a neural network.
    """
    def __init__(self, env, model, player, exploration_param=1, temperature=1, max_depth=None):
        self.env = env
        self.model = model
        self.model.eval()
        if DEVICE == 'cuda':
            self.model = self.model.cuda()
        with torch.no_grad():
            state = self.env.current_state.unsqueeze(0)
            if DEVICE == 'cuda':
                state = state.cuda()
            value, scores = self.model(state)
        self.tree = DictTreeNode({
            'value': value.squeeze(),
            'num_visits': 1,
            'prior_policy': constrained_softmax(scores.squeeze(), self.env.available_actions()),
            'player': player
        })
        self.exploration_param = exploration_param
        self.max_depth = max_depth
        self.inv_temp = 1 / temperature

    def select_child(self, node):
        """
        Select children according to some variant of UCT.
        """
        best_UCT = None
        for idx, (action, child) in enumerate(node.children.items()):
            # If child has never been visited then UCT is infinite
            if child['num_visits'] == 0:
                best_action, best_child = action, child
                break
            confidence_margin = np.sqrt(node['num_visits']) / (1 + child['num_visits'])
            confidence_margin *= node['prior_policy'][idx]
            confidence_margin *= self.exploration_param
            UCT = - child['value'] + confidence_margin
            if best_UCT is None or UCT > best_UCT:
                best_UCT, best_action, best_child = UCT, action, child
            
        return best_action, best_child
    
    def selection(self, simulator, node, depth): # output node, reward, done
        """
        Run down the tree and return a leaf.
        Branches are selected according to criterion defined in self.select_child
        """
        if len(node.children) == 0 or depth == self.max_depth or node['value'] is None:
            return node, None, False
        
        action, child = self.select_child(node)
        state, reward, done, info = simulator.step(action)
        
        if done:
            return child, reward, done
        
        return self.selection(simulator, child, depth + 1)
    
    @staticmethod
    def expansion(simulator, leaf):
        """
        Expand the leaf found after selection step by adding a child for 
        each available action. And return the first child.
        """
        player = leaf['player']
        leaf.add_children_values_dict({
            action: {
                'value': None,
                'num_visits': 0,
                'prior_policy': None,
                'player': - player
            }
            for action in simulator.available_actions()
        })
        action = simulator.available_actions()[0]
        state, reward, done, info = simulator.step(action)
        new_leaf = leaf.children[action]
        if done:
            new_leaf['value'] = reward
            new_leaf['num_visits'] = 1
        return new_leaf, reward, done
    
    def evaluation(self, leaf, leaf_state, actions):
        """
        Estimate value of leaf node with neural network.
        """
        with torch.no_grad():
            leaf_state = leaf_state.unsqueeze(0)
            if DEVICE == 'cuda':
                leaf_state = leaf_state.cuda()
            value, scores = self.model(leaf_state)
        leaf['value'] = value.squeeze()
        leaf['prior_policy'] = constrained_softmax(scores.squeeze(), actions)
        leaf['num_visits'] = 1
        return value.squeeze()
    
    @staticmethod
    def backup(leaf, value):
        """
        Update the values in the tree from leaf to root using estimated value.
        """
        node = leaf.parent
        while node is not None:
            node['num_visits'] += 1
            node['value'] = node['value'] + (node['player'] * value - node['value']) / node['num_visits']
            node = node.parent
    
    def search_step(self):
        simulator = deepcopy(self.env)
        leaf, value, done = self.selection(simulator, self.tree, 0)
        # If done, go straight to backup
        # If leaf['value'] is None (expanded but unvisited node), go straight to evaluation
        # Else expand children of leaf node
        if not done and leaf['value'] is not None:
            leaf, value, done = self.expansion(simulator, leaf)
        # If done after expansion, go straight to backup
        if not done:
            value = self.evaluation(leaf, simulator.current_state, simulator.available_actions())
        self.backup(leaf, value)
        return simulator, value
    
    def search(self, num_searches):
        for i in range(num_searches):
            self.search_step()
    
    def policy(self):
        """
        Compute the policy after search.
        """
        unnormalised_probas = np.array([child['num_visits'] for child in self.tree.children.values()])
        unnormalised_probas = unnormalised_probas ** self.inv_temp
        probas = unnormalised_probas / unnormalised_probas.sum()
        return list(self.tree.children.keys()), probas
        
