from itertools import chain
import numpy as np


DEFAULT_ICE_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "sim_batch_size":256,
    "paraphrase_path": None,
    "paraphrase_num": 5,
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 10,
        "temperature": 1.0,
        "top_p": 0.99,
        "output_scores": False,
        "return_normalized_transition_scores": False,
        "batch_size": 5,
    }
}

def calculate_ice(semantic_cluster_ids):
    """
    semantic_cluster_ids: List[List[list[int]]]: query_num x paraphrased_num x sample_sum: cluster id of generated answers.
    """
    num_paraphrase = len(semantic_cluster_ids[0])
    num_samples = len(semantic_cluster_ids[0][0])
    scores = []
    for ex_ids in semantic_cluster_ids:
        average_total_frequence = np.bincount(list(chain(*ex_ids)))/num_paraphrase
        probs = average_total_frequence/num_samples
        assert np.isclose(probs.sum(), 1)
        total_uncertainty = - (probs * np.log(probs)).sum()
        scores.append(total_uncertainty.item())
    return {"ice": scores}

               
        