import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from .functional import sparse_cross_entropy
from .network import DEVICE

def train(model, batch_size, num_iters, memory, optimiser, scheduler=None, verbose=True):
    """
    Train a model on transitions stored in memory for given parameters.
    """
    if DEVICE == 'cuda':
        model = model.cuda()
    model.train()
    writer = SummaryWriter()
    t = trange(num_iters, desc='Training', leave=True)
    
    for i in t:        
        optimiser.zero_grad()
        
        states, actions, target_policy, target_values = memory.sample(batch_size)
        if DEVICE == 'cuda':
            states = states.cuda()
            target_values = target_values.cuda()
            target_policy = [probas.cuda() for probas in target_policy]
        
        values, policy_scores = model(states)
        value_loss = F.mse_loss(values, target_values.unsqueeze(1).float())
        policy_loss = sparse_cross_entropy(policy_scores, actions, target_policy)
        loss = value_loss + policy_loss
        loss.backward()
        optimiser.step()
        if scheduler is not None:
            scheduler.step(loss)
        writer.add_scalar('loss', loss, i)