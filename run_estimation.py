from uncertainty.uncertainty_estimation import run_estimation
from uncertainty.uncertainty_evaluation import Uncertainty_Evaluator 
from uncertainty.utils import normalize_cache_path   
import argparse
import torch
import json
import os
import numpy as np
import random
import debugpy
import copy
from uncertainty.uncertainty_estimation import DEFAULT_SEMANTIC_ENTROPY_CONFIG, DEFAULT_SAR_CONFIG, DEFAULT_SELF_CON_CONFIG, DEFAULT_INSIDE_CONFIG, DEFAULT_MI_CONFIG, DEFAULT_SEMANTIC_DENSITY_CONFIG, DEFAULT_P_TRUE_CONFIG
from uncertainty import LLM_RESULTS
from loguru import logger
import time
from tqdm import tqdm
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='a project on uncertainty estimation')
    parser.add_argument('-d', '--debug', action='store_true',help='use valid dataset to debug your system')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('-c', '--cached_result_path', type=str, help='path cached the outputs of generation')
    parser.add_argument('-o', '--output_dir', type=str, help='the directory to save the result')
    

    parser.add_argument("-m", "--correctness_metric", type=str, default="bem", help='metrics to evaluate correctness')
    parser.add_argument("-t", "--correctness_threshold", type=float, default=0.7, help='threshold to evaluate correctness')

    
    parser.add_argument("--test_num", type=int, help='num of examples used to debug ')

   
    
    parser.add_argument("--num_scores_returned", type=int, default=100, help='num of logits cached')
    parser.add_argument('--store_score', action='store_true', help='whether to store the uncertainty scores')
    parser.add_argument('--sampling_only', action='store_true', help='whether to sampling responses only and remain score calculation for further computation')

    parser.add_argument('--evaluate_method', type=str, help='the single method name used to evaluate uncertainty score')
    parser.add_argument('--sim_batch_size', type=int, default=256, help='batch size for computing semantic similarity')
    
    #args for semantic entropy
    parser.add_argument("--se_cached_path", type=str, help='path cached the sampling results for Semantic entropy ')
    parser.add_argument("--se_save_path", type=str, help='path to save the sampling outputs of Semantic entropy')
    parser.add_argument("--se_temperature", type=float, help='temperature for semantic entropy sampling')
    parser.add_argument("--se_n", type=int, help='number of sampling sentence for computing semantic entropy')
    parser.add_argument("--se_batch_size", type=int, help='batch_size when sampling')

    #args for semantic density
    parser.add_argument("--sd_cached_path", type=str, help='path cached the sampling results for Semantic Density')
    parser.add_argument("--sd_save_path", type=str, help='path to save the sampling outputs of Semantic Density')
    parser.add_argument("--sd_temperature", type=float, help='temperature for semantic density sampling')
    parser.add_argument("--sd_n", type=int, help='number of sampling sentence for computing semantic density')
    parser.add_argument("--sd_model", type=str, help='model used to compute semantic similarity for computing semantic density')
    parser.add_argument("--sd_batch_size", type=int, help='batch_size when sampling')


    #args for self-con
    parser.add_argument("--sc_cached_path", type=str, help='path cached the sampling results for self-consistency')
    parser.add_argument("--sc_save_path", type=str, help='path to save the sampling outputs of self-consistency')
    parser.add_argument("--sc_temperature", type=float, help='temperature for self-consistency sampling')
    parser.add_argument("--sc_n", type=int, help='number of sampling sentence for computing self-consistency')
    parser.add_argument("--sc_model", type=str, help='model used to compute semantic similarity for computing self-consistency')
    parser.add_argument("--sc_batch_size", type=int, help='batch size when sampling sentences for computing semantic entropy')
    #args for sar
    parser.add_argument("--sar_cached_path", type=str, help='path cached the sampling results for SAR')
    parser.add_argument("--sar_save_path", type=str, help='path to save the sampling outputs of SAR')
    parser.add_argument("--sar_temperature", type=float, help='temperature for SAR sampling')
    parser.add_argument("--sar_n", type=int, help='number of sampling sentence for computing SAR')
    parser.add_argument("--sar_token_model", type=str, help='model used to compute token_wise_importance for computing SAR')
    parser.add_argument("--sar_sentence_model", type=str, help='model used to compute sentence similarity for computing SAR')
    parser.add_argument("--sar_batch_size", type=int, help='batch size when sampling sentences for computing sar')

    #args for inside
    parser.add_argument("--inside_cached_path", type=str, help='path cached the sampling hidden states for INSIDE')
    parser.add_argument("--inside_save_path", type=str, help='path to save the sampling hidden states of INSIDE')
    parser.add_argument("--inside_n", type=int, help='number of sampling responses for computing INSIDE')
    parser.add_argument("--inside_batch_size", type=int, help='batch size when sampling sentences for computing inside')

    #args for mi
    parser.add_argument("--mi_cached_path", type=str, help='path cached the sampling results for MI ')
    parser.add_argument("--mi_cached_mu2_path", type=str, help='path cached the conditional prob mu2 for MI ')
    parser.add_argument("--mi_save_path", type=str, help='path to save the sampling outputs of MI')
    parser.add_argument("--mi_temperature", type=float, help='temperature for MI sampling')
    parser.add_argument("--mi_n", type=int, help='number of sampling sentence for computing MI')
    parser.add_argument("--mi_batch_size", type=int, help='batch_size when sampling')

    #args for p_true
    parser.add_argument("--pt_model_name", type=str, help='the model name used to calulcate p_true')
    parser.add_argument("--pt_zero_shot", action='store_true', help='if true, will not use few-shot for p_true')
    parser.add_argument("--pt_cached_path", type=str, help='path cached the p(True) score for p(True)')
    parser.add_argument("--pt_save_path", type=str, help='path to save the p(True) score for p(True)') 
    

    
    parser.add_argument("-l", "--log_name", type=str, help='name of the log file')

    args, _ = parser.parse_known_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", 14328))
        print("listen ready")
        debugpy.wait_for_client()
    
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # read generation results
    cached_results = LLM_RESULTS.load(args.cached_result_path)
    if args.test_num is not None:
        cached_results = LLM_RESULTS.from_records(cached_results.to_records()[:args.test_num])
    model_name = cached_results.model_name

    
    if args.evaluate_method is None:
        
        
        # estimation_methods = ["sar","semantic_entropy",  "inside", "mi", "p_true", "semantic_density", "self_consistency" ]
        estimation_methods = ["sar","semantic_entropy", "mi", "semantic_density", "self_consistency", "inside", "p_true"]



            
    else:
        assert args.evaluate_method in [ "sar", "semantic_entropy", "self_consistency", "inside", "mi", "p_true", "semantic_density"], f"the given method {args.evaluate_method} is not supported"
        estimation_methods = [args.evaluate_method]

    # set config for sar methods
    if "sar" in estimation_methods:
        sar_config = copy.deepcopy(DEFAULT_SAR_CONFIG)
        sar_config["sim_batch_size"] = args.sim_batch_size
        if args.sar_token_model is not None:
            sar_config["token_importance_model"] = args.sar_token_model
        if args.sar_sentence_model is not None:
            sar_config["sentence_similarity_model"] = args.sar_sentence_model
        if args.sar_n is not None:
            sar_config["generation_config"]["num_responses_per_prompt"] = args.sar_n
        if args.sar_temperature is not None:
            sar_config["generation_config"]["temperature"] = args.sar_temperature
        if args.sar_batch_size is not None:
            sar_config["generation_config"]["batch_size"] = args.sar_batch_size

        sar_file_name = "sar_sampling_results" + "_" + str(sar_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(sar_config["generation_config"]["temperature"]) + ".json"
        cached_path_for_sar, save_path_for_sar = normalize_cache_path(args.sar_cached_path, args.sar_save_path, "SAR", sar_file_name)

        sar_config["cached_path"] = cached_path_for_sar
        print(cached_path_for_sar)
        sar_config["save_path"] = save_path_for_sar
    else:
        sar_config = copy.deepcopy(DEFAULT_SAR_CONFIG)


    # set config for semantic entropy methods
    if "semantic_entropy" in estimation_methods:
        se_config = copy.deepcopy(DEFAULT_SEMANTIC_ENTROPY_CONFIG)
        se_config["sim_batch_size"] = args.sim_batch_size
        if args.se_n is not None:
            se_config["generation_config"]["num_responses_per_prompt"] = args.se_n
        if args.se_temperature is not None:
            se_config["generation_config"]["temperature"] = args.se_temperature
        if args.se_batch_size is not None:
            se_config["generation_config"]["batch_size"] = args.se_batch_size

        se_file_name = "se_sampling_results" + "_" + str(se_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(se_config["generation_config"]["temperature"]) + ".json"

        cached_path_for_se, save_path_for_se = normalize_cache_path(args.se_cached_path, args.se_save_path, "SEMANTIC ENTROPY", se_file_name)

        se_config["cached_path"] = cached_path_for_se
        se_config["save_path"] = save_path_for_se
    else:
        se_config = copy.deepcopy(DEFAULT_SEMANTIC_ENTROPY_CONFIG)

    # set config for self-consistency methods
    if "self_consistency" in estimation_methods:
        sc_config = copy.deepcopy(DEFAULT_SELF_CON_CONFIG)
        sc_config["sim_batch_size"] = args.sim_batch_size
        if args.sc_model is not None:
            sc_config["model"] = args.sc_model

        if args.sc_n is not None:
            sc_config["generation_config"]["num_responses_per_prompt"] = args.sc_n
        if args.sc_temperature is not None:
            sc_config["generation_config"]["temperature"] = args.sc_temperature
        if args.sc_batch_size is not None:
            sc_config["generation_config"]["batch_size"] = args.sc_batch_size
        
        sc_file_name = "sc_sampling_results" + "_" + str(sc_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(sc_config["generation_config"]["temperature"]) + ".json"
        cached_path_for_sc, save_path_for_sc = normalize_cache_path(args.sc_cached_path, args.sc_save_path, "SELF-CONSISTENCY", sc_file_name)

        sc_config["cached_path"] = cached_path_for_sc
        sc_config["save_path"] = save_path_for_sc
    else:
        sc_config = copy.deepcopy(DEFAULT_SELF_CON_CONFIG)
    
    # set config for semantic density methods
    if "semantic_density" in estimation_methods:
        sd_config = copy.deepcopy(DEFAULT_SEMANTIC_DENSITY_CONFIG)
        sd_config["sim_batch_size"] = args.sim_batch_size
        if args.sd_model is not None:
            sd_config["model"] = args.sd_model

        if args.sd_n is not None:
            sd_config["generation_config"]["num_responses_per_prompt"] = args.sd_n
        if args.sd_temperature is not None:
            sd_config["generation_config"]["temperature"] = args.sd_temperature
        if args.sd_batch_size is not None:
            sd_config["generation_config"]["batch_size"] = args.sd_batch_size
        
        sd_file_name = "sd_sampling_results" + "_" + str(sd_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(sd_config["generation_config"]["temperature"]) + ".json"
        cached_path_for_sd, save_path_for_sd = normalize_cache_path(args.sd_cached_path, args.sd_save_path, "SEMANTIC_DENSITY", sd_file_name)

        sd_config["cached_path"] = cached_path_for_sd
        sd_config["save_path"] = save_path_for_sd
    else:
        sd_config = copy.deepcopy(DEFAULT_SEMANTIC_DENSITY_CONFIG)

    # set config for inside methods
    if "inside" in estimation_methods:
        inside_config = copy.deepcopy(DEFAULT_INSIDE_CONFIG)

        if args.inside_batch_size is not None:
            inside_config["generation_config"]["batch_size"] = args.inside_batch_size
        if args.inside_n is not None:
            inside_config["generation_config"]["num_responses_per_prompt"] = args.inside_n

        inside_file_name = "inside_sampling_results" + "_" + str(inside_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(inside_config["generation_config"]["temperature"]) + ".json"
        cached_path_for_inside, save_path_for_inside = normalize_cache_path(args.inside_cached_path, args.inside_save_path, "INSIDE", inside_file_name)

        inside_config["cached_path"] = cached_path_for_inside
        inside_config["save_path"] = save_path_for_inside
    else:
        inside_config = copy.deepcopy(DEFAULT_INSIDE_CONFIG)

    # set config for p_true methods
    if "p_true" in estimation_methods:
        
        pt_config = copy.deepcopy(DEFAULT_P_TRUE_CONFIG)
        if args.pt_model_name is None:
            pt_config["model_name"] = model_name
        else:
            pt_config["model_name"] = args.pt_model_name
        pt_file_name = "p_true_scores" + "_" + pt_config["model_name"] + ".json"
        cached_path_for_pt, save_path_for_pt = normalize_cache_path(args.pt_cached_path, args.pt_save_path, "P(True)", pt_file_name)

        pt_config["cached_path"] = cached_path_for_pt
        pt_config["save_path"] = save_path_for_pt
        
        if args.pt_zero_shot:
            pt_config["few_shot"] = False
    else:
        pt_config = copy.deepcopy(DEFAULT_P_TRUE_CONFIG)
    


    # set config for MI methods
    if "mi" in estimation_methods:
        mi_config = copy.deepcopy(DEFAULT_MI_CONFIG)
        if args.mi_n is not None:
            mi_config["generation_config"]["num_responses_per_prompt"] = args.mi_n
        if args.mi_temperature is not None:
            mi_config["generation_config"]["temperature"] = args.mi_temperature
        if args.mi_batch_size is not None:
            mi_config["generation_config"]["batch_size"] = args.mi_batch_size
        if args.mi_save_path is not None:
            if os.path.isdir(args.mi_save_path):
                file_name = "mi_sampling_results" + "_" + str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"]) + ".json"
                file_name_for_m2 = "mi_mu2_results" + "_" + str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"]) + ".json"
                save_path_for_mi = os.path.join(args.mi_save_path, file_name)
                mu2_save_path_for_mi = os.path.join(args.mi_save_path, file_name_for_m2)
            else:
                save_path_for_mi = args.mi_save_path 
                file_name_for_m2 = "mi_mu2_results" + "_" + str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"]) + ".json"
                mu2_save_path_for_mi = os.path.join(os.path.dirname(args.mi_save_path), file_name_for_m2)

        else:
            save_path_for_mi = None
            mu2_save_path_for_mi = None

        if args.mi_cached_path is not None:
            if os.path.isdir(args.mi_cached_path):
                file_name = "mi_sampling_results" + "_" + str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"]) + ".json"
                cached_path_for_mi = os.path.join(args.mi_cached_path, file_name)
            else:
                cached_path_for_mi = args.mi_cached_path  

        else:
            cached_path_for_mi = None
        
        if args.mi_cached_mu2_path is not None:
            if os.path.isdir(args.mi_cached_mu2_path):
                file_name = "mi_mu2_results" + "_" + str(mi_config["generation_config"]["num_responses_per_prompt"])+ "_temp_" + str(mi_config["generation_config"]["temperature"]) + ".json"
                mu2_cached_path_for_mi = os.path.join(args.mi_cached_mu2_path, file_name)
            else:
                mu2_cached_path_for_mi = args.mi_cached_mu2_path  

        else:
            mu2_cached_path_for_mi = None
        mi_config["cached_mu2_path"] = mu2_cached_path_for_mi
        mi_config["cached_path"] = cached_path_for_mi
        mi_config["save_path"] = save_path_for_mi
        mi_config["save_mu2_path"] = mu2_save_path_for_mi
    else:
        mi_config = copy.deepcopy(DEFAULT_MI_CONFIG)



    # set logger
    if args.debug:
        level = 'DEBUG'
    else:
        level = 'INFO'
    logger.remove()

    if getattr(args, "log_name"):
        if not os.path.exists("log"):
            os.makedirs("log")
        
        logger.add(os.path.join("log", args.log_name + ".log"), level=level)
    
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)
    
    

    # gpu setting
    if not torch.cuda.is_available():
        device_name = "cpu"
    else:
        device_name = "gpu0"
   

    
    
    logger.info(f"start to compute the uncertainty scores for outputs generated by model '{model_name}' on dataset from '{args.cached_result_path}'")
    estimated_scores = run_estimation(cached_results, estimation_methods=estimation_methods,  device_name=device_name, se_config=se_config, self_con_config=sc_config, sar_config=sar_config, inside_config=inside_config, sd_config=sd_config, sampling_only=args.sampling_only, pt_config=pt_config, mi_config=mi_config)

    if not args.sampling_only:
        if len(estimation_methods) == 1:
            name = estimation_methods[0]
        else:

            name = "full_test"
        uncertainty_score_save_path = os.path.join(os.path.dirname(args.output_dir), name + "_" + "uncertainty_scores.json")
        with open(uncertainty_score_save_path, 'w', encoding='utf-8') as f:
            json.dump(estimated_scores, f, indent=4)
    
        truth_label = (np.array(cached_results.scores[args.correctness_metric]) < args.correctness_threshold).astype(int)
    
        logger.info('start to evaluate error detection performace')
        # evaluator = Uncertainty_Evaluator(metrics="auroc")
        evaluator = Uncertainty_Evaluator(metrics=["auroc", "coverage"])
        evaluator.evaluate(estimated_scores, truth_label)
        if args.output_dir is not None:
    
            evaluator.to_excel(args.output_dir, name=name + f"_{args.correctness_metric}")

    