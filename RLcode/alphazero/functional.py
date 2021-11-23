import torch.nn.functional as F

def dense_cross_entropy(input, target):
    """
    Compute cross-entropy between a batch of scores (before softmax) and a
    batch of probability vectors.
    """
    log_probas = F.log_softmax(input, dim=1)
    cross_entropy_batch = - (log_probas * target).sum(axis=1)
    loss = cross_entropy_batch.mean()
    return loss

def sparse_cross_entropy(inputs, targets_indices, targets):
    """
    Compute cross-entropy between a batch of scores (before softmax) and a
    batch of probability vectors, but the target probabilities are sparse and
    contain values whose indices are stored in targets_indices. Scores not
    corresponding to these indices are discarded.
    """
    batch_size = len(targets)
    loss = 0
    for i, tensor in enumerate(inputs):
        masked_scores = tensor[targets_indices[i]]
        log_probas = F.log_softmax(masked_scores, dim=0)
        loss -= (log_probas * targets[i]).sum()
    loss /= batch_size
    return loss

def constrained_softmax(scores, indices):
    """
    Compute softmax on a restricted set of indices for given scores.
    """
    masked_scores = scores[indices]
    return masked_scores.softmax(0)
