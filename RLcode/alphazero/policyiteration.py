import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from .play import selfplay
from .training import train

def policy_iteration_step(env, model, batch_size, num_episodes, num_iters, capacity=10000, **kwargs):
    """
    Do one phase of self-play followed by one phase of training based on
    transitions generated in self-play.
    """
    memory = selfplay(env, model, num_episodes, capacity, **kwargs)
    
    optimiser = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=1000, verbose=True)
    #scheduler = None
    train(model, batch_size, num_iters, memory, optimiser, scheduler)