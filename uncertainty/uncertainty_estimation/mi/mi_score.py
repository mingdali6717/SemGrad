from .mi_score_utils import mu_cached_key
from collections import defaultdict
import numpy as np
DEFAULT_MI_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "cached_mu2_path": None,
    "save_mu2_path": None,
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 10,
        "temperature": 0.9,
        "top_p": 1.0,
        "output_scores": False,
        "return_normalized_transition_scores": True,
        "batch_size": 5

    }
}
MIN_LOG_VALUE = 1e-100

def calculate_mi_score(query_list, mu_1_probs_dict_list, cached_mu_2_result):
    scores = []
    for query, mu_1_probs_dict in zip(query_list, mu_1_probs_dict_list):
        mu_2_probs_dict = defaultdict(dict)
        distinct_responses = list(mu_1_probs_dict.keys())
        if len(distinct_responses) == 1:
            scores.append(0.0)
            
            continue

        for as1 in distinct_responses:

            for as2 in distinct_responses:
                
                key = mu_cached_key(query, as1, as2)
                mu_2_probs_dict[as1][as2] = cached_mu_2_result[key]
        for k,v in mu_2_probs_dict.items():
            total_probs = sum(list(v.values()))
            for k2, v2 in v.items():
                mu_2_probs_dict[k][k2] = v2 / total_probs
        mi_score = 0
        for as1 in distinct_responses:
            for as2 in distinct_responses:
                joint_dist = mu_1_probs_dict[as1] * mu_2_probs_dict[as1][as2]
                marginal_as2 = 0
                for as3 in distinct_responses:
                    marginal_as2 += mu_1_probs_dict[as3] * mu_2_probs_dict[as3][as2]
                product_dist = mu_1_probs_dict[as1] * marginal_as2
                point_wise_mi = joint_dist * np.log(max(joint_dist / product_dist, MIN_LOG_VALUE))
                mi_score += point_wise_mi
        
        scores.append(mi_score)

        
    return scores
                   
    