import os
os.environ['TRANSFORMERS_OFFLINE'] = "1"


import logging
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader



DEFAULT_SEMANTIC_ENTROPY_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "sim_batch_size":256,
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 10,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "output_scores": False,
        "return_normalized_transition_scores": True,
        "batch_size": 5
    }
}



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli", local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli", local_files_only=True).to(DEVICE)

    def check_implication(self, text1, text2, batch_size=256):
        """
        input: 
        text1 - str or List[str]
        text2 - str or List[str]

        output:
        prediction - List[int], 0 present contradiction, 1 present neutral, 2present entailment

        """
        if isinstance(text1, str):
            text1 = [text1]
        
        if isinstance(text2, str):
            text2 = [text2]
        
        
        tokenized_inputs = self.tokenizer(text1, text2, truncation=True)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        logger.info(f"start to calculate semantic similarity score for Semantic Entropy with batch size {batch_size}")
        with torch.no_grad():
            data_collator = DataCollatorWithPadding(self.tokenizer,  pad_to_multiple_of=8 if self.model.dtype==torch.float16 else None)

            dataloader = DataLoader(Dataset.from_dict(tokenized_inputs.data), batch_size=batch_size, collate_fn=data_collator)
            predictions = []
            for batch in tqdm(dataloader, desc="computing responses similarity"):

                outputs = self.model(**batch.to(DEVICE))
                logits = outputs.logits
                # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
                largest_index = torch.argmax(F.softmax(logits, dim=-1), dim=-1)  # pylint: disable=no-member
                prediction = largest_index.tolist()
                predictions.extend(prediction)

        # print(f'{text1} -> {text2}')
        # print(prediction)

        return predictions


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(list_of_strings_list, model, strict_entailment=False, batch_size=256):
    """Group list of predictions into semantic meaning.
    input: 
    list_of_strings_list - List[List[str]] or List[str], each List[str] is the group of strings to clustering semantic meaning

    output:

    semantic_ids: List[List[int]] or List[int].  each List[int] is the clustered semantic ids in the corresponding input group
 

    """

    if isinstance(list_of_strings_list, list) and isinstance(list_of_strings_list[0], str):
        list_of_strings_list = [list_of_strings_list]

    flatten_cands = []
    flatten_refs = []
    num_list_strings = len(list_of_strings_list)
    n = len(list_of_strings_list[0])
    for strings_list in list_of_strings_list:
        assert len(strings_list) == n, "all string list should be in the same size"
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                flatten_cands.append(strings_list[i])
                flatten_refs.append(strings_list[j])
    
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    concated_sents = [hyp + "[$usedforsep$]" + ref for hyp, ref in zip(flatten_cands, flatten_refs)]
    sents = dedup_and_sort(concated_sents)
    compact_hyps, compact_refs = list(zip(*[s.split("[$usedforsep$]") for s in sents]))
        
    compact_scores =  model.check_implication(compact_hyps, compact_refs, batch_size=batch_size)
    assert len(compact_scores) == len(compact_hyps) and len(compact_scores) == len(compact_refs), f"something is wrong, score_len: {len(compact_scores)}, compact_hyps_len: {len(compact_hyps)}, compact_ref_len: {len(compact_refs)}"
    stat_dict = {k:v for k,v in zip(sents, compact_scores)}
    scores = torch.tensor([stat_dict[k] for k in concated_sents], dtype=torch.int64).reshape(num_list_strings, n, n-1)

    index = torch.arange(n).unsqueeze(0).unsqueeze(0).expand(num_list_strings, n, -1)
    non_diagonal_mask = ~torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(num_list_strings,n,n)
    scatter_index = index[non_diagonal_mask].reshape(num_list_strings, n, n-1)
    entailment_matrix = 2 * torch.ones((num_list_strings, n, n), dtype=scores.dtype) # entailment is 2
    entailment_matrix.scatter_(-1, scatter_index, scores)

    identity_matrix = (entailment_matrix * torch.transpose(entailment_matrix,-1,-2)).tolist()
    if strict_entailment:
        threshold = 4 # both direction are 2 i.e. entailment
    else: 
        threshold = 2 # not contradiction for both side and at least one side is 2, i.e. >= 1 * 2 = 2
    semantic_ids = []

    for id_matrix in identity_matrix:
        semantic_set_ids = [-1] * n
        next_id = 0
        for i in range(n):
            if semantic_set_ids[i] == -1:
                semantic_set_ids[i] = next_id
                for j in range(i+1, n):
                    if id_matrix[i][j] >= threshold:
                        semantic_set_ids[j] = next_id
                next_id += 1
        semantic_ids.append(semantic_set_ids)
    


    return semantic_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    #assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm))) # \sum_{i in id_indices}{pi}/(\sum_{all}{p_j})
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
