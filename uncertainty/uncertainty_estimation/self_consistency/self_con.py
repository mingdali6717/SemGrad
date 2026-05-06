
import torch
from transformers import AutoModel, AutoTokenizer
import os
from bert_score import bert_cos_score_idf, get_idf_dict, sent_encode
from functools import partial
from itertools import chain
from multiprocessing import Pool, get_context
from collections import Counter, defaultdict
from tqdm import tqdm
from math import log
from ...generation_evaluation import SemSimCalculator
os.environ['TRANSFORMERS_OFFLINE'] = "1"
from ..utils import get_mean_pairwise_sim, get_spectral_eigv, get_eccentricity



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SELF_CON_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "model": "nli/deberta_nli",
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 10,
        "temperature": 0.7,
        "top_p": 1.0,
        "output_scores": False,
        "return_normalized_transition_scores": False,
        "batch_size": 5

    }
}



def self_consistency(original_responses, sampled_responses, sim_model, queries=None, device_name=None, batch_size=256, sim_threshold=0.5):
    num_queries = len(queries)
    if isinstance(original_responses, str):
        original_responses = [original_responses]
        assert isinstance(sampled_responses, list) and isinstance(sampled_responses[0], str), "given original responses is a string, the sampling responses should be a list of string"
        sampled_responses = [sampled_responses]
    else:
        assert isinstance(sampled_responses[0], list) and isinstance(sampled_responses[0][0], str), "given sampling responses should be List[List[str]]"
    
    n = len(sampled_responses[0])
    original_responses_flat = [item for item in original_responses for _ in range(n)]
    queries_flat = [item for item in queries for _ in range(n)]
    sampled_responses_flat = list(chain.from_iterable(sampled_responses))
    similarity_calculator = SemSimCalculator(sim_model, device_name=device_name)
    sim_scores = similarity_calculator(original_responses_flat, sampled_responses_flat, prepend_text=queries_flat, batch_size=batch_size).reshape(num_queries, n)
    identifier = (sim_scores > sim_threshold).int()

    return (1 - identifier.sum(dim=-1)/n).tolist()

def spectral_clustering_metrics(similarity_matrix, eigv_threhold=0.9):
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix)
    return {
        "self_con_average_sim": get_mean_pairwise_sim(similarity_matrix),
        "self_con_spectral_eigv": get_spectral_eigv(similarity_matrix, thres=eigv_threhold),
        "self_con_eccentricity": get_eccentricity(similarity_matrix, thres=eigv_threhold)
    }

def spectral_clustering_metrics_plus_doc(similarity_matrix, eigv_threhold=0.9):
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix)
    return {
        "self_con_plus_doc_average_sim": get_mean_pairwise_sim(similarity_matrix),
        "self_con_plus_doc_spectral_eigv": get_spectral_eigv(similarity_matrix, thres=eigv_threhold),
        "self_con_plus_doc_eccentricity": get_eccentricity(similarity_matrix, thres=eigv_threhold)
    }


