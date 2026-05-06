from .methods import average_entropy, max_entropy, average_prob, min_prob, unnormalized_prob
from ..generation_evaluation import SemSimCalculator
from .esi import ESI_ESTIMATOR, DEFAULT_ESI_CONFIG, load_paraphrase
from .semantic_entropy import semantic_entropy, DEFAULT_SEMANTIC_ENTROPY_CONFIG
from .self_consistency import self_consistency,  DEFAULT_SELF_CON_CONFIG,spectral_clustering_metrics, spectral_clustering_metrics_plus_doc
from .semantic_density import DEFAULT_SEMANTIC_DENSITY_CONFIG, get_sd_loglikelihood, evaluate_semantic_distance, cal_semantic_density_score
from .predictive_entropy import predictive_entropy
from .inside import DEFAULT_INSIDE_CONFIG, calculate_inside
from .sar import DEFAULT_SAR_CONFIG, sar
from .p_true import calculate_p_true, DEFAULT_P_TRUE_CONFIG
from .ice import calculate_ice, DEFAULT_ICE_CONFIG

from .mi import DEFAULT_MI_CONFIG, read_mu_1, read_mu_2, calculate_mi_score
from ..response_generator import LLM_RESULTS, construct_hash, StandardGenerator, reshape_sequences
import copy
import itertools
import torch
from loguru import logger
from ..utils import LLM, load_data, get_gpu_memory, PromptTemplate
from .semantic_entropy.semantic_entropy_utils import EntailmentDeberta, get_semantic_ids
from .utils import load_sampling_results
from tqdm import tqdm
from itertools import chain
import json
import gc
import numpy as np

import os

LOGPROB_BASED_METHODS = {
    "mean_logprob": average_prob,
    "logprob": unnormalized_prob
}

AVAILABLE_ESTIMATION_METHODS = ["semantic_entropy", "self_consistency", "sar",  "inside", "mi", "p_true", "semantic_density"]




def run_estimation(outputs,         
                   estimation_methods=AVAILABLE_ESTIMATION_METHODS, is_logits=True, temperature=1.0, device_name=None, se_config=DEFAULT_SEMANTIC_ENTROPY_CONFIG, self_con_config=DEFAULT_SELF_CON_CONFIG, sar_config=DEFAULT_SAR_CONFIG, inside_config=DEFAULT_INSIDE_CONFIG,  mi_config=DEFAULT_MI_CONFIG, sd_config=DEFAULT_SEMANTIC_DENSITY_CONFIG,
                   pt_config=DEFAULT_P_TRUE_CONFIG, 
                   sampling_only=False):
    """
    input: 
    logits - List[List[List[float]]], batch_size x token_num x vocab_size: the logits or probs of each vocab. if is_logits is False, the normalized probability should be given.
    transition_logprobs - List[List[float]], batch_size x token_num: the log prob of the generated token.
    skip_first: int or bool: if True, will skip the score of the first token. if a int number n is given, the first n token will be skipped. for example, llama-7b-chat will always generate '_' for the first token.
    skip_last: int or bool: if True, will skip the score of the last token. if a int number n is given, the last n token will be skipped. for example, llama-7b-chat will always generate '</s>' for the last token.
    """
    LLM.ddp = outputs.raw_config["ddp"]
    logits = outputs.logits["scores"]
    indexes = outputs.logits["ids"]
    transition_logprobs = outputs.transition_scores
    tokenizer = LLM.initial_tokenizer(outputs["model_name"])
    vocab_size = tokenizer.vocab_size
    estimation_scores = dict()

    for method in estimation_methods:

        logger.info(f"start to calculate scores of method '{method}'")
        assert method in AVAILABLE_ESTIMATION_METHODS, f"given estiamtion method {method} is not supported"

        
        if method == "inside":
            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(inside_config["generation_config"])
            tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])
            prompts = outputs.prompts
            model_name = outputs.model_name

            inside_scores = calculate_inside(outputs.prompts, outputs.model_name, generation_kwargs, tokenization_kwargs, cached_path = inside_config["cached_path"], save_path = inside_config["save_path"])
            estimation_scores.update(inside_scores)

        elif method == "semantic_entropy":


            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(se_config["generation_config"])
            sampling_outputs = None
            if se_config["cached_path"] is not None:
                sampling_outputs = load_sampling_results(se_config["cached_path"], generation_kwargs)
            else: 
                logger.info("No cached path given, start to sample from scratch")
            
            se_save_path = se_config["save_path"]
            
            if se_save_path is not None:
                logger.info(f"save sampling results to {se_save_path}")
                
            if sampling_outputs is None:
            
                prompts = outputs.prompts
                tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])


                sampling_outputs = LLM_RESULTS.from_dict(LLM.lm_generate(outputs.model_name, prompts, generation_kwargs, tokenization_kwargs, device_name = device_name, verbose=True))
            
                LLM.release_all()
            
                sampling_outputs.queries = outputs.queries
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
                sampling_outputs.ground_truth = outputs.ground_truth

                if se_save_path is not None:
                    sampling_outputs.save(os.path.abspath(se_save_path))
            else:
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
            if sampling_only:
                continue

            if outputs.queries_for_similarity is not None:
                se_queries = outputs.queries_for_similarity
            else:
                se_queries = outputs.queries

            if sampling_outputs.semantic_cluster_ids is None:
                entailment_model = EntailmentDeberta()
                se_inputs = [[ f'{q} {r}' for r in rs] for q, rs in zip(se_queries, sampling_outputs.responses)]
                semantic_ids_list = get_semantic_ids(se_inputs, entailment_model, batch_size=se_config["sim_batch_size"])
                sampling_outputs.semantic_cluster_ids = semantic_ids_list
                if se_save_path is not None:
                    sampling_outputs.save(os.path.abspath(se_save_path))
                del entailment_model
            
            estimation_scores.update(semantic_entropy(sampling_outputs.transition_scores, sampling_outputs.semantic_cluster_ids))

            del sampling_outputs
            
            gc.collect()

        elif method == "self_consistency":
            scores = []

            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(self_con_config["generation_config"])

            n = generation_kwargs["num_responses_per_prompt"]
            sampling_outputs = None
            if self_con_config["cached_path"] is not None:
                
                sampling_outputs = load_sampling_results(self_con_config["cached_path"], generation_kwargs)
            else: 
                logger.info("No cached path given, start to sample from scratch")

             

            # setting save path

            self_con_save_path = self_con_config["save_path"]
            if self_con_save_path is not None:
                logger.info(f"results will be saved to {self_con_save_path}")
            
            if sampling_outputs is None:

                prompts = outputs.prompts
                tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])

                sampling_outputs = LLM_RESULTS.from_dict(LLM.lm_generate(outputs.model_name, prompts, generation_kwargs, tokenization_kwargs, device_name = device_name, verbose=True))
                LLM.release_all()

                sampling_outputs.queries = outputs.queries
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
                sampling_outputs.ground_truth = outputs.ground_truth
                if self_con_save_path is not None:
                    sampling_outputs.save(os.path.abspath(self_con_save_path))
            else:
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
            
            if sampling_only:
                continue

                
            
            if sampling_outputs.sim_matrix is None or (self_con_config["model"] not in sampling_outputs.sim_matrix):
                
                sampling_outputs.evaluate_response_similarity(model=self_con_config["model"], device_name=device_name, batch_size=self_con_config["sim_batch_size"])
                
                if self_con_save_path is not None:
                    sampling_outputs.save(os.path.abspath(self_con_save_path))
            
            if outputs.queries_for_similarity is not None:
                sc_queries = outputs.queries_for_similarity
            else:
                sc_queries = outputs.queries
            
            key = construct_hash(outputs.responses + sampling_outputs.responses + sc_queries)

            if (sampling_outputs.sim_to_original is None) or (self_con_config["model"] not in sampling_outputs.sim_to_original) or (key not in sampling_outputs.sim_to_original[self_con_config["model"]]):
                
                
                sampling_outputs.evaluate_similarity_to_original_answers(outputs.responses, model=self_con_config["model"], device_name=device_name, batch_size=self_con_config["sim_batch_size"], queries=sc_queries)
                
                if self_con_save_path is not None:
                    sampling_outputs.save(os.path.abspath(self_con_save_path))
                
            

            # scores = self_consistency(outputs.responses,sampling_outputs.responses, self_con_config["model"], queries=outputs.queries,device_name=device_name )
            sim_scores = torch.tensor(sampling_outputs.sim_to_original[self_con_config["model"]][key])

            identifier = (sim_scores > 0.5).int()


            estimation_scores[method] = (1 - identifier.sum(dim=-1)/n).tolist()

            spectral_scores = spectral_clustering_metrics(sampling_outputs.sim_matrix[self_con_config["model"]])
            estimation_scores.update(spectral_scores)

            del sampling_outputs
            gc.collect()

        elif method == "sar":
            scores = []

            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(sar_config["generation_config"])

            n = generation_kwargs["num_responses_per_prompt"]
            sampling_outputs = None
            to_cache_result = False
            if sar_config["cached_path"] is not None:
                
                sampling_outputs = load_sampling_results(sar_config["cached_path"], generation_kwargs)
            else: 
                logger.info("No cached path given, start to sample from scratch")

            # setting save path
            sar_save_path = sar_config["save_path"] 
                
            if sar_save_path is not None:
                logger.info(f"sampling results will be saved to {sar_save_path}")
            
            if sampling_outputs is None:
                prompts = outputs.prompts
                tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])

                sampling_outputs = LLM_RESULTS.from_dict(LLM.lm_generate(outputs.model_name, prompts, generation_kwargs, tokenization_kwargs, device_name = device_name, verbose=True))
                LLM.release_all()

                sampling_outputs.queries = outputs.queries
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
                sampling_outputs.ground_truth = outputs.ground_truth

                if sar_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sar_save_path))
            else:
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
            
            if sampling_only:
                continue
            
            if (sampling_outputs.token_importance) is None or (sar_config["token_importance_model"] not in sampling_outputs.token_importance):
                
                
                sampling_outputs.evaluate_token_importance(model=sar_config["token_importance_model"],device_name=device_name, batch_size=sar_config["sim_batch_size"])
                if sar_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sar_save_path))
                
            
            
            if sampling_outputs.sim_matrix is None or (sar_config["sentence_similarity_model"] not in sampling_outputs.sim_matrix):
                
                sampling_outputs.evaluate_response_similarity(model=sar_config["sentence_similarity_model"], device_name=device_name, batch_size=sar_config["sim_batch_size"])
                if sar_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sar_save_path))
                # print(f"after computing sar responses similarity: Current GPU Memory Usage: {float(get_gpu_memory())/1024:.2f} GB")
                
            
            token_importance = sampling_outputs.token_importance[sar_config["token_importance_model"]]
            sim_matrix = sampling_outputs.sim_matrix[sar_config["sentence_similarity_model"]]
            

            estimation_scores.update(sar(sampling_outputs.transition_scores, token_importance, sim_matrix))

            # logprobs_list = [[np.mean(ts) for ts in tss] for tss in sampling_outputs.transition_scores]
            # full_logprobs_list = [np.mean([np.sum(ts) for ts in tss]).item() for tss in sampling_outputs.transition_scores]
            # predictive_entropy_score = {"ln_predictive_entropy": [predictive_entropy(logprobs).item() for logprobs in logprobs_list], "predictive_entropy": full_logprobs_list}
            # estimation_scores.update(predictive_entropy_score)

            del sampling_outputs
            gc.collect()
        
        
        elif method == "mi":


            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(mi_config["generation_config"])
            sampling_outputs = None
            
        

            param_name =  str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"])

            if mi_config["save_path"] is not None:
                
                if os.path.isdir(mi_config["save_path"]):
                    mi_save_path = os.path.join(mi_config["save_path"], "mi_sampling_results" + "_" + param_name + ".json")
                else:
                    assert mi_config["save_path"].endswith(".json"), "save file should be end with '.json'"
                    mi_save_path = mi_config["save_path"] 
                logger.info(f"save sampling results to {mi_save_path}")
            elif  mi_config["cached_path"] is not None:
                mi_cache_dir = os.path.dirname(mi_config["cached_path"])
                mi_save_path = os.path.join(mi_cache_dir, "mi_sampling_results" + "_" + param_name + ".json")
                logger.info(f"save sampling results to {mi_save_path}")
            
            if mi_config["save_mu2_path"] is not None:
                
                if os.path.isdir(mi_config["save_mu2_path"]):
                    mu2_save_path = os.path.join(mi_config["save_mu2_path"], "mi_mu2_results" + "_" + param_name + ".json")
                else:
                    assert mi_config["save_mu2_path"].endswith(".json"), "save file should be end with '.json'"
                    mu2_save_path = mi_config["save_mu2_path"] 
                logger.info(f"save condititonal prob mu2 results to {mu2_save_path}")
            elif  mi_config["cached_mu2_path"] is not None:
                mi_cache_dir = os.path.dirname(mi_config["cached_mu2_path"])
                mu2_save_path = os.path.join(mi_cache_dir, "mi_mu2_results" + "_" + param_name + ".json")
                logger.info(f"save condititonal prob mu2 results to {mu2_save_path}")

            if mi_config["cached_path"] is not None and (not os.path.exists(mi_config["cached_path"])):
                logger.info(f"cached path {mi_config['cached_path']} does not exist, evaluate MI from scratch.")
                mi_config["cached_path"] = None
            
            if mi_config["cached_mu2_path"] is not None and (not os.path.exists(mi_config["cached_mu2_path"])):
                logger.info(f"cached path {mi_config['cached_mu2_path']} does not exist, evaluate MI from scratch.")
                mi_config["cached_mu2_path"] = None

            if mi_config["cached_path"] is not None:
                sampling_outputs = load_sampling_results(mi_config["cached_path"], generation_kwargs)
            else: 
                logger.info("No cached path given, start to sample from scratch")
            
            if mi_config["cached_mu2_path"] is not None:
                with open(mi_config["cached_mu2_path"], "r", encoding="utf-8") as f:
                    cached_mu2_result = json.load(f)
            else: 
                cached_mu2_result = None
                logger.info("No cached mu2 results given, start to read from scratch")
            param_name = str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"])
            
                
            if sampling_outputs is None:
            
                prompts = outputs.prompts
                tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])


                sampling_outputs = LLM_RESULTS.from_dict(LLM.lm_generate(outputs.model_name, prompts, generation_kwargs, tokenization_kwargs, device_name = device_name, verbose=True))
            
                LLM.release_all()
            
                sampling_outputs.queries = outputs.queries
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
                sampling_outputs.ground_truth = outputs.ground_truth

                
                sampling_outputs.save(os.path.abspath(mi_save_path))
            else:
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
            

            raw_prob_list =[np.exp([np.sum(ts) for ts in ts_scores]) for ts_scores in sampling_outputs.transition_scores]
            mu_1_probs_dict_list = read_mu_1(sampling_outputs.responses, raw_prob_list)

            cached_mu2_result = read_mu_2(mu_1_probs_dict_list, sampling_outputs.queries, sampling_outputs.model_name, cached_mu_2_result=cached_mu2_result, device_name=device_name, batch_size=mi_config["generation_config"]["batch_size"]*5)

            
            
            with open(mu2_save_path, "w", encoding="utf-8") as f:
                json.dump(cached_mu2_result, f, indent=4)
            
            logger.info(f" condititonal prob mu2 results saved to {mu2_save_path}")

            if sampling_only:
                continue
            
            estimation_scores.update({"mi_score": calculate_mi_score(sampling_outputs.queries, mu_1_probs_dict_list, cached_mu2_result)})


            del sampling_outputs
            
            gc.collect()
        
        elif method == "semantic_density":
        
            scores = []

            generation_kwargs = copy.deepcopy(outputs.config["generation_config"])
            generation_kwargs.update(sd_config["generation_config"])
            
            sd_batch_size = sd_config["generation_config"]["batch_size"]

            n = generation_kwargs["num_responses_per_prompt"]
            sampling_outputs = None
            if sd_config["cached_path"] is not None:
                
                sampling_outputs = load_sampling_results(sd_config["cached_path"], generation_kwargs)
            else: 
                logger.info("No cached path given, start to sample from scratch")

            # setting save path
            

            sd_save_path = sd_config["save_path"]
            
            logger.info(f"results will be saved to {sd_save_path}")
            
            if sampling_outputs is None:

                prompts = outputs.prompts
                tokenization_kwargs = copy.deepcopy(outputs.config["tokenization_config"])

                sampling_outputs = LLM_RESULTS.from_dict(LLM.lm_generate(outputs.model_name, prompts, generation_kwargs, tokenization_kwargs, device_name = device_name, verbose=True))
                
                sampling_outputs.queries = outputs.queries
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
                sampling_outputs.ground_truth = outputs.ground_truth
                if sd_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sd_save_path))
            else:
                sampling_outputs.queries_for_similarity = outputs.queries_for_similarity
            
            if sampling_only:
                continue
        
            if sampling_outputs.transition_scores is None:
                sampling_outputs.transition_scores = get_sd_loglikelihood(outputs.model_name, sampling_outputs.responses, sampling_outputs.prompt_ids, batch_size=sd_batch_size*2)
                if sd_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sd_save_path))

            LLM.release_all()

                
            
            if outputs.queries_for_similarity is not None:
                sd_queries = outputs.queries_for_similarity
            else:
                sd_queries = outputs.queries
            
            key = construct_hash(outputs.responses + sampling_outputs.responses + sd_queries)

            if (sampling_outputs.sim_to_original is None) or (sd_config["model"] not in sampling_outputs.sim_to_original) or (key not in sampling_outputs.sim_to_original[sd_config["model"]]):
                
                
                distance_score = evaluate_semantic_distance(outputs.responses, sampling_outputs.responses, sd_queries, sd_config["model"],semantic_batch_size=sd_config["sim_batch_size"])
                sampling_outputs.sim_to_original= {sd_config["model"]: {key: distance_score}}
                if sd_save_path is not None:
                    sampling_outputs.save(os.path.abspath(sd_save_path))
            
            score_list = [1 - cal_semantic_density_score(r, d, l) for r, d, l in zip(sampling_outputs.responses, sampling_outputs.sim_to_original[sd_config["model"]][key], sampling_outputs.transition_scores)]

                
                
            


            estimation_scores[method] = score_list

            del sampling_outputs
            gc.collect()

        elif method == "p_true":
            
            
            if pt_config["model_name"] != outputs.model_name:

                logger.info(f"calculating P(True) with a third-party model {pt_config['model_name']}")
                LLM.release_all()
            
            pt_keys = [q + "[ptset]" + r for q, r in zip(outputs.queries, outputs.responses)]
                
            if pt_config["cached_path"] is not None:
                with open(pt_config["cached_path"], "r") as f:
                    pt_scores = json.load(f)
                pt_scores = [pt_scores[k] for k in pt_keys]
            else:
                
                pt_scores = calculate_p_true(outputs.queries, outputs.responses, pt_config["model_name"])
                if pt_config["save_path"] is not None:
                    with open(pt_config["save_path"], "w") as f:
                        json.dump(dict(zip(pt_keys, pt_scores)), f, indent=4)
                    logger.info(f"save p_true scores to {pt_config['save_path']}")

            estimation_scores.update({"p_true" + f"_{pt_config['model_name']}": pt_scores})
            
            LLM.release_all()
                
            gc.collect()

    

    if not sampling_only:
        return estimation_scores   
    else:
        return None




    

