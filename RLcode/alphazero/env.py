import numpy as np
import torch

class TransformEnv:
    """
    Wrap a gym environment to apply transforms to its state and reward outputs.
    """
    def __init__(self, gym_env, state_transform=lambda x: x, reward_transform=lambda x: x):
        self.gym_env = gym_env
        self.state_transform = state_transform
        self.reward_transform = reward_transform
        self.reset()
    
    def step(self, action):
        state, reward, done, info = self.gym_env.step(action)
        state = self.state_transform(state)
        reward = self.reward_transform(reward)
        self.current_state = state
        return state, reward, done, info
    
    def render(self):
        self.gym_env.render()
    
    def reset(self):
        state = self.state_transform(self.gym_env.reset())
        self.current_state = state
        return state
    
    def available_actions(self):
        if hasattr(self.gym_env, "available_actions"):
            return self.gym_env.available_actions()
        if hasattr(self.gym_env, "get_moves"):
            return self.gym_env.get_moves()

def tic_tac_toe_state_transform(state):
    Os = (np.array(state[0]) == 1).astype(float)
    Xs = (np.array(state[0]) == 2).astype(float)
    state = np.hstack([Os, Xs])
    tensor = torch.tensor(state).float()
    return tensor

def c4_state_transform(state):
    state = [torch.from_numpy(board).float() for board in state]
    state = torch.cat(state)
    return state