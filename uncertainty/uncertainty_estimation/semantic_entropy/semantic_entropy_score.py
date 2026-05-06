from collections import defaultdict
import logging
import os
import pickle
import numpy as np
from tqdm import tqdm

from .semantic_entropy_utils import get_semantic_ids
from .semantic_entropy_utils import logsumexp_by_id
from .semantic_entropy_utils import predictive_entropy
from .semantic_entropy_utils import predictive_entropy_rao
from .semantic_entropy_utils import cluster_assignment_entropy
from .semantic_entropy_utils import EntailmentDeberta


import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_semantic_entropy(semantic_ids, log_liks_agg):

    cluster_assignment_entropy_value = cluster_assignment_entropy(semantic_ids)
    regular_entropy_value = predictive_entropy(log_liks_agg)
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    semantic_entropy_value = predictive_entropy_rao(log_likelihood_per_semantic_id)

    return {
        'semantic_ids': semantic_ids,
        'cluster_assignment_entropy': cluster_assignment_entropy_value,
        'regular_entropy': regular_entropy_value,
        'semantic_entropy': semantic_entropy_value,
        'log_liks_agg': log_liks_agg
    }
    # return semantic_entropy_value

def semantic_entropy(sampling_transition_scores, semantic_ids_list):
    """
    input: 
    
        "question" (List[str]): The question being asked.
        sampling_responses (List[List[str]]): A list of list of string
        "logprobs": (List[List[List[float]]]): log prob of each responses
    entailment_model - str, name of the entailment_model.One of deberta, gpt-4, gpt-3.5, gpt-4-turbo, llama.
    """

    # if entailment_model == 'deberta':
    #     entailment_model = EntailmentDeberta()
    # else:
    #     raise ValueError
    semantic_ids_list = [normalize_semantic_ids(i) for i in semantic_ids_list]
    scores = []
    noln_scorers = []
    regular_entropy_scores = []
    cluster_assignment_scores = []
    noln_regular_entropy_scores = []
    # inputs = [[ f'{question} {r}' for r in responses] for question, responses in zip(questions, sampling_responses)]
    # if semantic_cluster_ids is None:
    #     semantic_ids_list = get_semantic_ids(inputs, entailment_model, batch_size=batch_size)
    # else: 
    #     semantic_ids_list = semantic_cluster_ids
    ln_logprobs_list = [[np.mean(transition_score) for transition_score in transition_scores] for transition_scores in sampling_transition_scores]
    logprobs_list = [[np.sum(transition_score) for transition_score in transition_scores] for transition_scores in sampling_transition_scores]

    
    for semantic_ids, logprobs, noln_logprobs in tqdm(zip(semantic_ids_list, ln_logprobs_list, logprobs_list), desc='Calculating semantic_entropy'):
        
        semantic_entropy_value = compute_semantic_entropy(semantic_ids, logprobs)
        noln_semantic_entropy_value = compute_semantic_entropy(semantic_ids, noln_logprobs)
        noln_regular_entropy_scores.append(noln_semantic_entropy_value["regular_entropy"])
        noln_scorers.append(noln_semantic_entropy_value["semantic_entropy"])
        scores.append(semantic_entropy_value["semantic_entropy"])
        regular_entropy_scores.append(semantic_entropy_value["regular_entropy"])
        cluster_assignment_scores.append(semantic_entropy_value["cluster_assignment_entropy"])
   
    
    return {
        "semantic_entropy": scores,
        "semantic_entropy_no_ln": noln_scorers,
        "ln_predictive_entropy": regular_entropy_scores,
        "predictive_entropy": noln_regular_entropy_scores,
        "empirical_entropy": cluster_assignment_scores
    }
def normalize_semantic_ids(semantic_ids):
    distinct_ids = set(semantic_ids)
    id_mapping = {o_id: new_id for o_id, new_id in zip(distinct_ids, range(len(distinct_ids)))}
    return [id_mapping[i] for i in semantic_ids]

