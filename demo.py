# Train a model for 20 iterations on a game and then do two episodes against the user

import numpy as np
import torch
import gym
import sys
sys.path.append('gym-tictactoe')
from gym_tictactoe import env as ttt
import gym_connect4

from RLcode.alphazero.env import *
from RLcode.alphazero.network import TTT_model, C4_model
from RLcode.alphazero.policyiteration import policy_iteration_step
from RLcode.alphazero.play import do_episode
from RLcode.alphazero.agents import NeuralAgent
from RLcode.other_algos.agents import HumanAgent

game = int(input('Choose the game (0=Tic Tac Toe, 1=Connect 4):'))
assert game in [0, 1]

if game:
    print('Connect 4')
    env = TransformEnv(
        gym_env=gym.make('Connect4-v0'),
        state_transform=c4_state_transform,
        reward_transform=lambda x: torch.tensor(x[0])
    )
    model = C4_model()
else:
    print('Tic Tac Toe')
    env = TransformEnv(
        gym_env=ttt.TicTacToeEnv(),
        state_transform=tic_tac_toe_state_transform,
        reward_transform=torch.tensor
    )
    model = TTT_model()

print('10 iterations of policy iteration (temperature=1)')
for i in range(10):
    policy_iteration_step(env, model, 32, 10, 500, capacity=10000, num_searches=100, temperature=1, exploration_param=1)
    
print('5 iterations of policy iteration (temperature=0.1)')
for i in range(5):
    policy_iteration_step(env, model, 32, 10, 500, capacity=10000, num_searches=100, temperature=0.1, exploration_param=1)    
    
print('5 iterations of policy iteration (temperature=0.01)')
for i in range(5):
    policy_iteration_step(env, model, 32, 10, 500, capacity=10000, num_searches=100, temperature=0.01, exploration_param=1)
    
print('Play against the model')
agent1 = NeuralAgent(env, model, 1, num_searches=100, temperature=0.01)
agent2 = HumanAgent(env, -1)
do_episode(agent1, agent2)

print('Roles reversed')
agent1 = HumanAgent(env, 1)
agent2 = NeuralAgent(env, model, -1, num_searches=100, temperature=0.01)
do_episode(agent1, agent2)