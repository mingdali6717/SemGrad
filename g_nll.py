import torch
from scipy.stats import wasserstein_distance
from uncertainty.response_generator import LLM_RESULTS, construct_hash
from uncertainty.utils import LLM
import copy
from loguru import logger
import numpy as np
import jsonlines
from uncertainty.uncertainty_evaluation import Uncertainty_Evaluator

EPISILON=1e-15

def entropy(logits, is_logits=True, temperature=1.0, expand_to=None, index=None):
    """
    calculate the entropy of the last dimension of given logits

    input:
    logits: List, numpy.ndarray or torch.Tensor. if it is a list of list, it must can be in the nice shape to be transfered to torch.Tensor.
    is_logits: if True, will softmax before calculating the entropy
    temperature: float, the temperature used in softmax

    return:
    torch.Tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.double, device=device)
    else:
        logits = logits.to(device)
    
    if (expand_to is not None) and (expand_to > logits.shape[-1]):
        assert is_logits==True, "only logits can be expanded"
        assert isinstance(expand_to, int) and expand_to >= logits.shape[-1], f"expand_to should be int with value large than the given logits num"
        assert index is not None, f"expand_to is set to {expand_to}, an index should be given to scatter the value"
        logits = expand_truncated_logits(logits, indexes=index, vocab_size=expand_to)
    
    if is_logits:
        probs = torch.nn.functional.softmax(logits/float(temperature), dim=-1)
    
    return (-torch.log(probs) * probs).sum(dim=-1)

def expand_truncated_logits(truncated_logits, indexes, vocab_size=None, expand_to_max_acceptable_size=False):
    """
    expand the truncated logits to full logits, fill the unsaved logits with the min value at each position

    input:
    truncated_logits: List or torch.tensor, shape(arbitrary_dims,seq_len x truncated_vocab_size) last dimension is the truncated logits num, if it is a List, it must can be in the nice shape to be transfered to torch.Tensor.
    index: the token_id of the corresponding logit, should be in the same size as truncated_logits
    vocab_size: int
    expand_to_max_acceptable_size: bool - if True, will expand to each truncated logits to size N, N is the total number of unique indexs. for example, if indexes are [[1,2], [1,4]], the N will be 3
    return:
    expand_logits: torch.tensor, shape(arbitrary dims x sequence_len x vocab_size)
    """
    


    if not isinstance(truncated_logits, torch.Tensor):
        
        truncated_logits_ts = torch.tensor(truncated_logits, dtype = torch.double)
    else:
        truncated_logits_ts = truncated_logits.to(torch.double)
    if isinstance(indexes, list):
        indexes = torch.tensor(indexes, device=truncated_logits_ts.device)
    assert indexes.shape == truncated_logits_ts.shape, f"indexes shape '{indexes.shape}' should be same as the logits shape '{truncated_logits.shape}"
    max_index = torch.max(indexes).item()

    if expand_to_max_acceptable_size:
        unique_indexes = torch.unique(indexes)
        vocab_size = unique_indexes.shape[0]
        new_index_mapping = {old_i: new_i for new_i, old_i in zip(range(vocab_size), unique_indexes.tolist())}
        indexes = torch.tensor(list(map(lambda x: new_index_mapping[x], indexes.reshape(-1).tolist())), dtype=indexes.dtype).reshape(*indexes.shape)
    else:
        assert vocab_size is not None, "vocab size to expand should be given to expand a truncated logits"
        vocab_size = max(vocab_size, max_index+1) # some model set special token to a number large than vocab size
        


    expand_logits = (torch.min(truncated_logits_ts, dim=-1)[0]/10.0).unsqueeze(-1).expand(*truncated_logits_ts.shape[:-1], vocab_size).clone()
    expand_logits.scatter_(-1, indexes, truncated_logits_ts)
    return expand_logits

correctness_metric = "bem"
datasets = ["truthfulqa", "sciq", "triviaqa"]
models = ["llama3.1-8b-instruct", "qwen3-4b-instruct", "mistral-nemo-instruct"]
for d in datasets:
    for m in models:
        cached_path = f"./output/cached_results/{d}/{m}/results.json"
        cached_results = LLM_RESULTS.load(cached_path)
        # tokenizer = LLM.initial_tokenizer(m)
        logprobs = cached_results.transition_scores
        g_nll = []
        for lp in logprobs:
            nll = np.sum(lp) * -1
            g_nll.append(nll)

        truth_label = (np.array(cached_results.scores[correctness_metric]) < 0.7).astype(int)

        truth_label = (np.array(cached_results.scores[correctness_metric]) < 0.7).astype(int)
        evaluator = Uncertainty_Evaluator(metrics=["auroc", "aucpr", "coverage"])
        evaluator.evaluate(g_nll, truth_label, verbose=False)
    
    
        output_dir = f"./output/cached_results/{d}/{m}/"
        evaluator.to_excel(output_dir, name=f"gnll_{correctness_metric}")

    
        