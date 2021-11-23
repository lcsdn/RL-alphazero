import torch
from tqdm import trange

from .agents import NeuralAgent  
from .data import ReplayMemory  

def selfplay_step(env, model, **kwargs):
    """
    Play one episode between two neural agents and return the transitions and
    the final reward.
    """
    model.eval()
    state = env.reset()
    agent1 = NeuralAgent(env, model, 1, **kwargs)
    agent2 = NeuralAgent(env, model, -1, **kwargs)
    
    transitions = []
    
    done = False
    while not done:
        old_state = state
        chosen_action, actions, policy, state, reward, done, _ = agent1.play()
        agent2.register_play(chosen_action)
        transitions.append([old_state, actions, torch.from_numpy(policy).float()])
        
        if not done:
            old_state = state
            chosen_action, actions, policy, state, reward, done, _ = agent2.play()
            agent1.register_play(chosen_action)
            transitions.append([old_state, actions, torch.from_numpy(policy).float()])
    
    return transitions, reward

def selfplay(env, model, num_episodes, capacity, **kwargs):
    """
    Play several episodes between two neural agents and save the transitions and
    final rewards.
    """
    memory = ReplayMemory(capacity)
    
    t = trange(num_episodes, desc='Self-play', leave=True)
    for episode in t:
        transitions, reward = selfplay_step(env, model, **kwargs)
        memory.save_game(transitions, reward)
        t.set_description("Self-play (last reward %i)" % int(reward))
        t.refresh()

    return memory

def do_episode(agent1, agent2, show=False):
    """
    Play an episode between two agents and return the reward.
    """
    assert agent1.env is agent2.env
    env = agent1.env
    env.reset()
    if show:
        env.render()
    agent1.reset()
    agent2.reset()
    done = False
    while not done:
        chosen_action, _, _, _, reward, done, _ = agent1.play()
        agent2.register_play(chosen_action)
        if show:
            env.render()
        
        if not done:
            chosen_action, _, _, _, reward, done, _ = agent2.play()
            agent1.register_play(chosen_action)
            if show:
                env.render()
    return int(reward)

def compare_agents(agent1, agent2, num_episodes, fair=False): #TODO fair bugs, see minimax notebook
    """
    Play several episodes to compare two agents, return the rewards.
    With fair=True:
        - Even iterations: agent1 starts.
        - Odd iterations: agent2 starts.
    """
    rewards = []
    
    for i, episode in enumerate(range(num_episodes)):
        reward = do_episode(agent1, agent2)
        if fair:
            agent1, agent2 = agent2, agent1
            agent1.player, agent2.player = 1, -1
            if i % 2 == 1:
                reward = - reward
        rewards.append(reward)
    
    return rewards