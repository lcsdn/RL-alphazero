import random
import torch

class ReplayMemory:
    """
    Object allowing to save transitions and sample them for training.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.rewrite_pos = 0
    
    def save(self, *transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.rewrite_pos] = transition
            self.rewrite_pos = (self.rewrite_pos + 1) % self.capacity
    
    def save_game(self, transitions, reward):
        for transition in transitions:
            self.save(*transition, reward)
    
    def sample(self, batch_size):
        # If batch size exceeds memory, sample with replacement
        sampler = random.choices if len(self.memory) < batch_size else random.sample
        batch_list = sampler(self.memory, k=batch_size)
        collated_batch = []
        try:
            for j in range(len(batch_list[0])):
                if j==0 or j==len(batch_list[0])-1:
                    collated_batch.append(torch.cat(
                        [batch_list[i][j].unsqueeze(0) for i in range(len(batch_list))]
                    ))
                else:
                    collated_batch.append([batch_list[i][j] for i in range(len(batch_list))])
        except:
            breakpoint()
        return collated_batch
    
    def sample_dataloader(self, batch_size, num_batches=1):
        for i in range(num_batches):
            yield self.sample(batch_size)