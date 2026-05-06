from itertools import chain
from uncertainty.utils import LLM
from uncertainty.utils import get_logits, reshape_sequences
from uncertainty.generation_evaluation.metrics.semantic_similarity import NLIEvaluator
import torch
import numpy as np

DEFAULT_SEMANTIC_DENSITY_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "model": "nli/deberta_nli",
    "generation_config": {
        "do_sample": False,
        "num_responses_per_prompt": 10,
        "num_beams": 10,
        "num_beam_groups": 10,
        "diversity_penalty": 1.0,
        "temperature": 1.0,
        "output_scores": False,
        "return_normalized_transition_scores": False,
        "batch_size": 5,
        "trust_remote_code": True
    }
}

def get_sd_loglikelihood(model_name, responses, prompt_ids, batch_size = 10):
    model_key = LLM.initial_lm(model_name, None)
    model, tokenizer = LLM.loaded_llms[model_key]
    response_ids = [tokenizer(rs, padding=False, add_special_tokens=False)["input_ids"] for rs in responses]
    input_ids = list(chain(*[ [p_id + r_id for r_id in r_ids ] for p_id, r_ids in zip(prompt_ids, response_ids)]))
    input_len = list(chain(*[[len(ids) for _ in range(len(response_ids[0]))] for ids in prompt_ids]))
    loglikelihood = get_logits(input_ids, model, tokenizer, batch_size=batch_size, return_transition_scores=True, num_scores_returned=10,prompt_lens=input_len, temperature=0.1)["transition_scores"]
    return reshape_sequences(loglikelihood, len(response_ids[0]))

def evaluate_semantic_distance(original_responses, sampled_responses, queries_for_similarity, semantic_model_name, semantic_batch_size=256):
    if isinstance(sampled_responses, str):
        sampled_responses = [sampled_responses]

    if isinstance(original_responses, str):
        original_responses = [original_responses]
    if isinstance(sampled_responses[0], list):
        n = len(sampled_responses[0])
    else: 
        n = 1
        sampled_responses = [[r] for r in sampled_responses]
    assert len(original_responses) == len(sampled_responses), f"the given origianl_responses (length {len(original_responses)}) do not match the sampled response (length {len(sampled_responses)})"
    num_records = len(original_responses)
    evaluator = NLIEvaluator(model_name=semantic_model_name, return_type="distribution")
    original_responses_flat = [item for item in original_responses for _ in range(n)]
    sampled_responses_flat = list(chain.from_iterable(sampled_responses))
   
    queries_flat = [item for item in queries_for_similarity for _ in range(n)]
    prepend_text = queries_flat + queries_flat

    res_hyps = original_responses_flat + sampled_responses_flat
    res_refs = sampled_responses_flat + original_responses_flat

    hyps = [q.strip()+" "+h.strip() for q,h in zip(prepend_text, res_hyps)]
    refs = [q.strip()+" "+r.strip() for q,r in zip(prepend_text, res_refs)]
    concated_sents = [hyp + "[$usedforsep$]" + ref for hyp, ref in zip(hyps, refs)]
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
    sents = dedup_and_sort(concated_sents)
    compact_hyps, compact_refs = list(zip(*[s.split("[$usedforsep$]") for s in sents]))

    compact_scores =  evaluator(compact_hyps, compact_refs, batch_size=semantic_batch_size)
    evaluator.release_model()
    stat_dict = {k:v for k,v in zip(sents, compact_scores.tolist())}
    scores = torch.tensor([stat_dict[k] for k in concated_sents], dtype=compact_scores.dtype, device=compact_scores.device)
    distance = (scores[:,0] + 0.5 * scores[:,2]).reshape(2, num_records*n).mean(dim=0).reshape(num_records, n).tolist()
    return distance

def cal_semantic_density_score(responses, distances, loglikelihoods):
    unique_response_set = []
    score = 0.0
    total_prob = 0.0
    for r, d, l in zip(responses, distances, loglikelihoods):
        if r not in unique_response_set:
            unique_response_set.append(r)
            prob = np.exp(np.mean(l))
            
            score += (1-d) * prob
            total_prob += prob

        else:
            continue
    return score / total_prob


