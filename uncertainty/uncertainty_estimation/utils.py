import torch
from scipy.stats import wasserstein_distance
from uncertainty.response_generator import LLM_RESULTS, construct_hash
import copy
from loguru import logger
import numpy as np
import jsonlines

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

def z_score_normalize(tensor,dim=-1):
    mean = torch.mean(tensor, dim=dim)
    std = torch.std(tensor, dim=dim, unbiased=False)
    return (tensor - mean.unsqueeze(dim)) / torch.clamp(std.unsqueeze(dim), min=1e-12)

def kl_div(P, Q, is_logit=True, temperature=1.0):
    """
    calculate KL(P|Q) of the distribution in the last dimension
    input: 
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)
    Q - torch.tensor (arbitrary dimensions x num_of_outcomes)

    return:
    kl_div: torch.tensor (arbitrary dimensions)
    """
    
    assert P.shape == Q.shape, f"P and Q should be in the same shape, but P is with shape '{list(P.shape)}', while Q is with shape '{list(Q.shape)}'"
    if is_logit:
        P = torch.nn.functional.softmax(P/float(temperature), dim=-1)
        Q = torch.nn.functional.softmax(Q/float(temperature), dim=-1)
    
  
    P = torch.log(torch.clamp(P, min=EPISILON))
    Q = torch.log(torch.clamp(Q, min=EPISILON))
    kl_div_value = torch.nn.functional.kl_div(Q, P, reduction="none", log_target=True).sum(dim=-1)
    
    return kl_div_value



def hellinger_distance(P,Q, is_logit=True, temperature=1.0):
    """
    calculate the Hellinger Distance 1\sqrt(2) \sum (sqrt(p_i) - sqrt(q_i))**2 between the distribution in the last dimension
    input: 
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)

    return:
    score: torch.tensor (arbitrary dimensions)
    """
    assert P.shape == Q.shape, f"P and Q should be in the same shape, but P is with shape '{list(P.shape)}', while Q is with shape '{list(Q.shape)}'"
    if is_logit:
        P_prob = torch.nn.functional.softmax(P/float(temperature), dim=-1)
        Q_prob = torch.nn.functional.softmax(Q/float(temperature), dim=-1)
    else:
        P_prob = P
        Q_prob = Q
    
    return torch.sqrt(((torch.sqrt(P_prob) - torch.sqrt(Q_prob)) ** 2).sum(dim=-1)) / torch.sqrt(torch.tensor(2.0, device=P_prob.device))

def square_hellinger_distance(P,Q, is_logit=True, temperature=1.0):
    """
    calculate the Hellinger Distance 1\sqrt(2) \sum (sqrt(p_i) - sqrt(q_i))**2 between the distribution in the last dimension
    input: 
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)

    return:
    score: torch.tensor (arbitrary dimensions)
    """
    assert P.shape == Q.shape, f"P and Q should be in the same shape, but P is with shape '{list(P.shape)}', while Q is with shape '{list(Q.shape)}'"
    if is_logit:
        P_prob = torch.nn.functional.softmax(P/float(temperature), dim=-1)
        Q_prob = torch.nn.functional.softmax(Q/float(temperature), dim=-1)
    else:
        P_prob = P
        Q_prob = Q
    
    return ((torch.sqrt(P_prob) - torch.sqrt(Q_prob)) ** 2).sum(dim=-1) / torch.tensor(2.0, device=P_prob.device)

def Bhatacharyya_distance(P,Q, is_logit=True, temperature=1.0):
    """
    calculate the Hellinger Distance 1\sqrt(2) \sum (sqrt(p_i) - sqrt(q_i))**2 between the distribution in the last dimension
    input: 
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)
    P - torch.tensor (arbitrary dimensions x num_of_outcomes)

    return:
    score: torch.tensor (arbitrary dimensions)
    """
    assert P.shape == Q.shape, f"P and Q should be in the same shape, but P is with shape '{list(P.shape)}', while Q is with shape '{list(Q.shape)}'"
    if is_logit:
        P_prob = torch.nn.functional.softmax(P/float(temperature), dim=-1)
        Q_prob = torch.nn.functional.softmax(Q/float(temperature), dim=-1)
    else:
        P_prob = P
        Q_prob = Q
    
    return -torch.log(torch.sqrt(P_prob * Q_prob).sum(dim=-1))


def load_sampling_results(cached_path, generation_config, sample_index=None):
    logger.info(f"cached sampling file found, start to load cached results from {cached_path}")
    sampling_outputs = LLM_RESULTS.load(cached_path)
    cached_config = copy.deepcopy(sampling_outputs.config["generation_config"])
    required_config = copy.deepcopy(generation_config)
    cached_n = cached_config.pop("num_responses_per_prompt")
    required_n = required_config.pop("num_responses_per_prompt")
    cached_config.pop("batch_size")
    required_config.pop("batch_size")
    cached_config.pop("num_scores_returned")
    required_config.pop("num_scores_returned")
    cached_transition_scores = cached_config.pop("return_normalized_transition_scores")
    required_transition_scores = required_config.pop("return_normalized_transition_scores")
    cached_output_scores = cached_config.pop("output_scores")
    required_output_scores = required_config.pop("output_scores")
    
    

    if (cached_config != required_config):
        logger.warning(f"the cached_config do not match the given generation config, give wrong cached file, start to sampling from scratch!!!\nrequired config: {required_config}\n given cached config: {cached_config}")
        return None
    
    if cached_n < required_n:
        logger.warning(f"the required sampling num {required_n} is larger than the cached sampling num {cached_n}, start to sampling from scratch")
        return None
    
    if (not cached_transition_scores) and required_transition_scores:
        logger.warning("the cached file do not compute and cache transition scores which is required, start to sampling from scratch")
        return None
    
    if (not cached_output_scores) and required_output_scores:
        logger.warning("the cached file do not compute and cache the logits which is required, start to sampling from scratch.")
        return None


    if required_n < cached_n:
        logger.warning(f"given sample num '{required_n}' is less the cached sample num '{cached_n}'")
        if sample_index is not None:
            assert len(sample_index) == required_n, f"required_n is {required_n}, but only {len(sample_index)} indexes are given"
        important_keys = ["responses", "response_ids", "transition_scores", "scores", "semantic_cluster_ids"]
        output_dict= sampling_outputs.to_dict()
        for k in important_keys:
            if output_dict[k] is not None:
                if sample_index is None:
                    output_dict[k] = [ex[:required_n]for ex in  output_dict[k]]
                else:
                    output_dict[k] = [[ex[i] for i in sample_index] for ex in  output_dict[k]]
        if output_dict["logits"] is not None:
            for k,v in output_dict["logits"].items():
                if sample_index is None:
                    output_dict["logits"][k] = [ex[:required_n] for ex in v]
                else:
                    output_dict["logits"][k] = [[ex[i] for i in sample_index] for ex in v]
        
        if output_dict["token_importance"] is not None:
            for k,v in output_dict["token_importance"].items():
                if sample_index is None:
                    output_dict["token_importance"][k] = [ex[:required_n] for ex in v]
                else:
                    output_dict["token_importance"][k] = [[ex[i] for i in sample_index] for ex in v]
        if output_dict["sim_matrix"] is not None:
            for k,v in output_dict["sim_matrix"].items():
                if sample_index is None:
                    output_dict["sim_matrix"][k] = (torch.tensor(v)[:, :required_n, :required_n]).tolist()
                else:
                    output_dict["sim_matrix"][k] = (torch.tensor(v)[:, torch.tensor(sample_index), :][:, :, torch.tensor(sample_index)]).tolist()
        
        

        
        sampling_outputs = LLM_RESULTS.from_dict(output_dict)
    logger.info("cached sampling results loaded")
    return sampling_outputs

def read_sample_by_index(samples, indexes):
    """
    samples: a python list
    indexes: List[int]

    return: sub_sample - item extracted from samples following the indexes

    """
    return [samples[i] for i in indexes]
def get_D_mat(W):
    """
    compute the degree matrix of the symmetric similarity matrix of sampling responses

    params:
    W: torch_tensor size batch_size x sample_num x sample_num
    """
    D = torch.diag_embed(W.sum(dim=-1))
    return D

def get_degree_uncertainty(degree_matrix):
    
    ret = (1-degree_matrix).mean((-1,-2))
    return ret

def get_L_mat(W):
    """
    compute the Graph Laplacian of the symmetric similarity matrix of sampling responses

    params:
    W: torch_tensor size batch_size x sample_num x sample_num
    """
    
    D = get_D_mat(W)
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
    if len(W.shape) == 3:
        L = torch.bmm(torch.bmm(torch.inverse(torch.sqrt(D)), (D - W)), torch.inverse(torch.sqrt(D)))
    else:
        L = torch.mm(torch.mm(torch.inverse(torch.sqrt(D)), (D - W)), torch.inverse(torch.sqrt(D)))
    return L

def get_mean_pairwise_sim(sim_matrix):
    """
    params:
    sim_matrix: torch_tensor size batch_size x sample_num x sample_num
    """
    return (1 - sim_matrix.mean((-1,-2))).tolist()

def get_eig(L, thres=None, eps=None):
    """
    get the eigen value and eigenvectors of the Graph Laplacian of the symmetric similarity matrix of sampling responses

    params:
    L: torch_tensor size batch_size x sample_num x sample_num
    """
    
    eigvals, eigvecs = torch.linalg.eigh(L)

    

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = tuple(zip(*[(eigv[eigmask].tolist(), ((eigvec.T)[eigmask]).T.tolist()) for eigv, eigvec, eigmask in zip(eigvals, eigvecs, keep_mask)]))
    else:
        eigvals, eigvecs = eigvals.tolist(), eigvecs.tolist()
        
        
    return eigvals, eigvecs

def get_spectral_eigv(W, thres=0.9):
    """
    calculate the Sum of Eigenvalues of the Graph Laplacian of the symmetric similarity matrix of sampling responses, detaled refer to the paper "Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models"

    params:
    W: torch_tensor size batch_size x sample_num x sample_num
    """
    L = get_L_mat(W)
    eigvs, _ = get_eig(L, thres=thres)
    return [(1 - np.array(eigv)).clip(0).sum() for eigv in eigvs]

def get_eccentricity(W, thres=0.9):
    """
    calculate the eccentricity of row vector of eigenvectors  of of the Graph Laplacian of the symmetric similarity matrix of sampling responses, detaled refer to the paper "Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models"
    W: torch_tensor size batch_size x sample_num x sample_num
    """
    L = get_L_mat(W)
    _, eigvecs = get_eig(L, thres=thres)
    ds = np.asarray([np.linalg.norm(np.array(x) -np.array(x).mean(0)[None, :],2) for x in eigvecs]).tolist()
    return ds




