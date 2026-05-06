import numpy as np
import torch
from ..utils import entropy



def average_prob(log_probs, return_prob=True):
    """
    calculate the arithmetic average of log probs or the geometric average of probs

    input:
    log_probs: List, numpy.ndarray or torch.Tensor. the log probabilitys will take the average along the last dimension.
    return_prob: if True, will return the geometric average of probs, i,e, exp(1/n * sum_i(log p_i))
    """
    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.tensor(log_probs, dtype=torch.double)

    indexes = [0] * len(log_probs.size())
    element = log_probs[tuple(indexes)]
    if element > 0:
        raise ValueError("givn log probs is larger than 0, something is wrong")
    
    average_log_probs = log_probs.mean(dim=-1)
    if return_prob:
        return torch.exp(average_log_probs)
    else:
        return average_log_probs

def unnormalized_prob(log_probs, return_prob=True):
    """
    calculate the sum of log probs or the product of probs, i.e. the logprob of prob of the given sequence.

    input:
    log_probs: List, numpy.ndarray or torch.Tensor. the log probabilitys. will take the average along the last dimension.
    return_prob: if True, will return the geometric average of probs, i,e, exp(1/n * sum_i(log p_i))
    """
    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.tensor(log_probs, dtype=torch.double)

    indexes = [0] * len(log_probs.size())
    element = log_probs[tuple(indexes)]
    if element > 0:
        raise ValueError("givn log probs is larger than 0, something is wrong")
    
    log_probs_sum = log_probs.sum(dim=-1)
    if return_prob:
        return torch.exp(log_probs_sum)
    else:
        return log_probs_sum

def min_prob(log_probs, return_prob=True):
    """
    calculate the miniminum of log probs or probs

    input:
    log_probs: List, numpy.ndarray or torch.Tensor. the log probabilitys. will take the average along the last dimension.
    return_prob: if True, will return the geometric average of probs, i,e, exp(1/n * sum_i(log p_i))
    """
    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.tensor(log_probs, dtype=torch.double)

    indexes = [0] * len(log_probs.size())
    element = log_probs[tuple(indexes)]
    if element > 0:
        raise ValueError("givn log probs is larger than 0, something is wrong")
    
    min_log_probs = log_probs.min(dim=-1).values
    
    if return_prob:
        return torch.exp(min_log_probs)
    else:
        return min_log_probs

def average_entropy(logits, is_logits=True, temperature=1.0, expand_to=None, index=None):
    """
    input:
    logits: List, numpy.ndarray or torch.Tensor. arbitray dims x token_num x vocab_size:  if it is a list of list, it must can be in the nice shape to be transfered to torch.Tensor.
    is_logits: Bool, if True, the logits will pass softmax to get probability
    expand_to: int, if given, the logits vector will be expanded to larger size following the given index, and the unknown value will be filled with min(logits)/10.
    index: List, numpy.ndarray or torch.Tensor, the index used to scatter value.

    return:
    torch.Tensor, arbitrary dims:
    """
    entropys = entropy(logits, is_logits=is_logits, temperature=temperature, expand_to=expand_to, index=index)
    return entropys.mean(dim=-1)

def max_entropy(logits, is_logits=True, temperature=1.0, expand_to=None, index=None):
    """
    input:
    logits: List, numpy.ndarray or torch.Tensor. arbitray dims x token_num x vocab_size:  if it is a list of list, it must can be in the nice shape to be transfered to torch.Tensor.
    is_logits: Bool, if True, the logits will pass softmax to get probability
    expand_to: int, if given, the logits vector will be expanded to larger size following the given index, and the unknown value will be filled with min(logits)/10.
    index: List, numpy.ndarray or torch.Tensor, the index used to scatter value.

    return:
    torch.Tensor, arbitrary dims:
    """
    entropys = entropy(logits, is_logits=is_logits, temperature=temperature, expand_to=expand_to, index=index)
    return entropys.max(dim=-1).values
