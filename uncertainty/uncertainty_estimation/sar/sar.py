import torch
import numpy as np

DEFAULT_SAR_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "token_importance_model": "cross-encoder/stsb-roberta-large",
    "sentence_similarity_model": "cross-encoder/stsb-roberta-large",
    "sim_batch_size":256,
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 5,
        "temperature": 1.0,
        "output_scores": False,
        "return_normalized_transition_scores": True,
        "top_p": 1.0,
        "batch_size": 10
    }
}

def sar(logprobs, token_wise_importance, sentence_similarity_matrix):
    """
    logprobs: List[List[List[float]]] shape(num_queries x sample_num x response_token_num)
    token_wise_importance: List[List[List[float]]] shape(num_queries x sample_num x response_token_num) same size as logprobs
    sentence_similarity_matrix: List[List[List[float]]] shape(num_queries x sample_num x sample_num)
    """
    token_sar_scores = []
    sentence_sar_scores = []
    sar_scores = []
    for sample_logprob, sample_token_imp, sim_matrix in zip(logprobs, token_wise_importance, sentence_similarity_matrix):
        sample_token_sar_scores = []
        sample_sentence_logprob = []

        for logprob, token_imp in zip(sample_logprob, sample_token_imp):
            logprob = torch.tensor(logprob, dtype=torch.double)
            token_imp = torch.tensor(token_imp, dtype=torch.double)
            ex_token_sar = ((token_imp/token_imp.sum()) * logprob).sum().item()

            sample_token_sar_scores.append(ex_token_sar)
            sample_sentence_logprob.append(logprob.sum().item())
        token_sar_scores.append(-1 * np.mean(sample_token_sar_scores))
        sentence_sar_scores.append(cal_sentence_sar(sim_matrix, sample_sentence_logprob))
        sar_scores.append(cal_sentence_sar(sim_matrix, sample_token_sar_scores))
    

    return {
        "sar": sar_scores,
        "sentence_sar": sentence_sar_scores,
        "token_sar": token_sar_scores
    }

        


def cal_sentence_sar(sim_matrix, sentence_logprobs, t=0.001):
    n = len(sim_matrix)
    sentence_prob = torch.exp(torch.tensor(sentence_logprobs, dtype=torch.double).unsqueeze(0).expand(n, n))

    temperature = (1/t) * torch.ones(n,n, dtype=sentence_prob.dtype)
    temperature.diagonal().fill_(1.0)

    reweighted_sentence_logprob = - torch.log((torch.tensor(sim_matrix, dtype=sentence_prob.dtype) * temperature * sentence_prob).sum(dim=-1))
    
    return reweighted_sentence_logprob.mean().item()







