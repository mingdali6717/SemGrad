import torch
from ..utils import z_score_normalize, entropy, expand_truncated_logits, kl_div, hellinger_distance, square_hellinger_distance, Bhatacharyya_distance
from loguru import logger

from ...utils import get_gpu_memory
DISTRIBUTION_DISTANCE_FUNC_MAPPING = {
    "kl": kl_div,
    "hellinger": hellinger_distance,
    "square_hellinger": square_hellinger_distance,
    "Bhatacharyya":Bhatacharyya_distance,
}


def DistributionDistance(logits, token_weight,expand_to=None, index=None, distance_measure="kl", temperature=1.0, grid_sizes=[2,4, 5,8, 10]):
    """
    calculate the token-wise distribution distance and pooling to a single variantion score

    input: 
    logits: List, numpy.ndarray or torch.Tensor. if it is a list of list, it must can be in the nice shape to be transfered to torch.Tensor.
    expand_to: int, if expand_to is givn, logits will be expanded to full logits with respect to the index given in 'index', fill the unsaved logits with the min value at each position
    index: List, numpy.ndarray or torch.Tensor. same size as the logits, cached the token_id of the corresponding logit
    distance_measure: str, the method used to calculate distribution distance
        acceptable value:
        "kl": KL-Divergence KL(P|Q)
        "js": Jensen-Shannon Divergence JS(P|Q) = 0.5 * KL(P|(P+Q)/2) + 0.5 * KL(Q|(P+Q)/2)
        "tv": Total Variation Distance TV(P, Q) = SUP_A |P(A)-Q(A)| = 1/2 \sum |p_i - q_i|
        "euclidean": Euclidean Distance ED(P, Q) = sqrt(\sum (p_i - q_i)^2) 
        "emd": Earch Mover Distance (1-d wasserstein distance)
        "hellinger": Hellinger Distance 1\sqrt(2) \sum (sqrt(p_i) - sqrt(q_i))**2
    pooling_method: str the method to pooling the score
        acceptable value:
        "mean": the mean of all distance scores
        "max": the max of all scores
        "meanbatchmax": first calculate the max score of each token across batches, then calculate the average
        "maxbatchmean": first calculate the avergage score of each token across batches, then calculate the max
        "meanseqmax": first calculate the max score along each sequence, then calculate the average
        "maxseqmean": first calculate the avergage score along each sequence, then calculate the max
    centered: bool - if True, only calculate the variation between the original answer and intervened answers, if False, will calculate pairwise variation for all pairs
    """
    
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype = torch.double)
    

    P = logits[0].unsqueeze(0).expand(logits.shape[-3]-1, -1, -1)
    Q = logits[1:]
       
    
    distance_score = DISTRIBUTION_DISTANCE_FUNC_MAPPING[distance_measure](P, Q, temperature=temperature) # 

    
    if not isinstance(token_weight, torch.Tensor):
        token_weight = torch.tensor(token_weight, dtype = distance_score.dtype, device= distance_score.device)
    else:
        token_weight = token_weight.to(distance_score.device)
    
        
    assert token_weight.shape[-1] == distance_score.shape[-1] and len(token_weight.shape) == 1, f"last dimension of weight should be in the same size of the sequence length '{distance_score.shape[-1]}', but the given weight is in the shape of '{token_weight.shape}'"
        
    distance_score = token_weight * distance_score
    
    output_score = {f"mean": distance_score.mean().item()}
    if grid_sizes is not None:
        for grid_size in grid_sizes:
            if grid_size >= distance_score.shape[1]:
                output_score.update({f"grid_size_{grid_size}": distance_score.mean().item()})
                continue
            else:

                num_tokens = distance_score.shape[1]
                if num_tokens % grid_size == 0:
                    grid_mean_score = distance_score.reshape(distance_score.size(0), -1, grid_size).mean(dim=0).mean(dim=1)
                    output_score.update({f"grid_size_{grid_size}": grid_mean_score.max().item()})
                else:
                    first_column_size = ((num_tokens//grid_size) - 1) * grid_size
                    remain_column_size = num_tokens - first_column_size
                    third_column_size = remain_column_size // 2
                    second_column_size = remain_column_size - third_column_size

                    if first_column_size != 0:

                        a_1, a_2, a_3 = torch.split(distance_score, [first_column_size, second_column_size, third_column_size], dim=1 )
                        grid_mean_score = torch.cat((a_1.reshape(distance_score.size(0), -1, grid_size).mean(dim=0).mean(dim=1), a_2.mean().unsqueeze(0), a_3.mean().unsqueeze(0)))
                        output_score.update({f"grid_size_{grid_size}": grid_mean_score.max().item()})
                    else:

                        a_1, a_2 = torch.split(distance_score, [second_column_size, third_column_size], dim=1 )
                        grid_mean_score = torch.cat((a_1.mean().unsqueeze(0), a_2.mean().unsqueeze(0)))
                        output_score.update({f"grid_size_{grid_size}": grid_mean_score.max().item()})

    return output_score

    
    # return distance_score.mean().item()










       

    
    