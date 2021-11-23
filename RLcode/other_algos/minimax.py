from copy import deepcopy

def minimax(env, player):
    """
    Compute the minimax action path from current env and for given player.
    """
    opt_reward = None
    
    for action in env.available_actions():
        simulator = deepcopy(env)
        state, reward, done, info = simulator.step(action)
        
        if done:
            action_path = []
        else:
            reward, action_path = minimax(simulator, - player)
        
        # Select max reward if player = 1, min reward if player = -1
        if opt_reward is None or player * (reward - opt_reward) > 0:
            opt_reward, opt_action_path = reward, [action] + action_path
            
    return opt_reward, opt_action_path

def tree_minimax(env, player, root):
    """
    Compute the minimax value tree from current env and for given player,
    must give the root node of the tree as input.
    """
    root.add_children_values_dict({
        action: {
            'value': None,
            'action': None,
            'player': - player
        }
        for action in env.available_actions()
    })
    
    opt_value = None
    for action, child in root.children.items():
        simulator = deepcopy(env)
        state, reward, done, info = simulator.step(action)
        
        if done:
            child['value'] = reward
        else:
            tree_minimax(simulator, - player, child)
        
        # Select max reward if player = 1, min reward if player = -1
        if opt_value is None or player * (child['value'] - opt_value) > 0:
            opt_value, opt_action = child['value'], action
    root['value'] = opt_value
    root['action'] = opt_action
    
def comprehensive_tree_minimax(env, player, root):
    """
    Compute the minimax value tree from current env and for given player,
    must give the root node of the tree as input.
    """
    root.add_children_values_dict({
        action: {
            'value': None,
            'opt_actions': None,
            'player': - player
        }
        for action in env.available_actions()
    })
    
    opt_value = None
    opt_actions = []
    for action, child in root.children.items():
        simulator = deepcopy(env)
        state, reward, done, info = simulator.step(action)
        
        if done:
            child['value'] = reward
        else:
            comprehensive_tree_minimax(simulator, - player, child)
        
        if opt_value is None or player * (child['value'] - opt_value) > 0:
            opt_value = child['value']
            opt_actions = []
        if child['value'] == opt_value:
            opt_actions.append(action)
    root['value'] = opt_value
    root['opt_actions'] = opt_actions