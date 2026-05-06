from uncertainty import LLM_RESULTS, PromptTemplate
from uncertainty.utils import LLM
from uncertainty.uncertainty_estimation.esi.utils import load_paraphrase
from uncertainty.uncertainty_estimation.gradient import SFTCausalCollator, loss_function, grad_vector_and_weight, grad_norms
from uncertainty.response_generator import StandardGenerator
from uncertainty.uncertainty_evaluation import Uncertainty_Evaluator
from itertools import chain
import copy
from tqdm import tqdm
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import debugpy
import numpy as np
import random
import json
from loguru import logger
from collections import defaultdict

IMPORTANCE_TOKEN = {
    "llama3.1-8b-instruct": -4,
    "llama3-8b-instruct": -4,
    "qwen2.5-14b-instruct": -3,
    "qwen3-4b-instruct": -3,
    "qwen3-30b-instruct": -3,
    "mistral-8b-instruct": -2,
    "mistral-nemo-instruct": -2



}

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='a project on uncertainty estimation')
    parser.add_argument('--debug', action='store_true',help='use valid dataset to debug your system')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('-c', '--cached_result_path', type=str, help='path cached the outputs of generation')
    parser.add_argument('-b', '--batch_size', type=int, default=10,  help='batch size for reading hiddenstates and gradients')
    parser.add_argument('-o', '--output_dir', type=str, help='the directory to save the result')
    parser.add_argument('-d', '--dataset', type=str, help='dataset')
    parser.add_argument('-m', '--model_name', type=str, help='model')
    parser.add_argument('-ne', '--no_entropy_weight', action='store_true', help='wether to use entropy weight when calculating loss')
    parser.add_argument('-t', '--token_importance', action='store_true', help='wether to use entropy weight when calculating loss')
    
    parser.add_argument('-k', '--keep_token_num', type=int, default=5, help='number of token preserved for calculating embedding ')
    parser.add_argument("--test_num", type=int, help='num of examples used to debug ')
    
    parser.add_argument("--correctness_metric", type=str, default="bem", help='metrics to evaluate correctness')
    parser.add_argument("--correctness_threshold", type=float, default=0.7, help='threshold to evaluate correctness')
    
    
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
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True

    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size
    ignore_id = -100
    keep_token_num = args.keep_token_num
    import_token_idx = IMPORTANCE_TOKEN[model_name]
    keep_token_num = max(-1 *import_token_idx,  keep_token_num)
    result_path = os.path.join(os.path.abspath(args.cached_result_path) ,"results.json")
    data = LLM_RESULTS.load(result_path)
   
    if args.test_num is not None:
        data = LLM_RESULTS.from_records(data.to_records()[:args.test_num])
    if args.token_importance:
        token_importance = data.token_importance["cross-encoder/stsb-roberta-large"]

    template_config = getattr(data, "raw_config", None)
    if template_config is None:
        template_config = {
            "verbose": False,
            "system_id": None,
            "template_id":2,
            "generate_kwargs": dict()
        }
    
    template_config["model_name"] = model_name
    if template_config["system_id"] == 0:
        template_config["system_id"] = None
    template = StandardGenerator(template_config).prompt_template
    
    
    if not args.no_entropy_weight:
        entropy_weight = True
    else:
        entropy_weight = False
    
    loss_name = "CE"

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    
    tokenize_kwargs =copy.deepcopy(data.raw_config["tokenize_kwargs"])
    tokenizer = LLM.initial_tokenizer(model_name, tokenizer_kwargs=tokenize_kwargs)
   
    input_ids = [p + r for p, r in zip(data.prompt_ids, data.response_ids)]
    labels = [ [ignore_id]*len(p) + r for p, r in zip(data.prompt_ids, data.response_ids)]
    if args.token_importance:
        token_imp = [ [0]*len(p) + imp for p, imp in zip(data.prompt_ids, token_importance)]
    response_len = [len(r) for r in data.response_ids]
    
    attention_mask = [[1 if _id != tokenizer.pad_token else 0 for _id in ids] for ids in input_ids]
    if args.token_importance:
        inputs= {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "response_len": response_len, "token_importance": token_imp}
    else:
        inputs= {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "response_len": response_len}
    #read gradient
    logger.info(f"start to read gradients for model {model_name} on dataset {dataset}")
    lm_name = LLM.initial_lm(model_name, None)
    model, _ = LLM.loaded_llms[lm_name]
    datacollator = SFTCausalCollator(tokenizer,label_pad_token_id=ignore_id) 
    dataloader = DataLoader(Dataset.from_dict(inputs), batch_size=batch_size, collate_fn=datacollator)
    
    

    total_grads = None
    scores = defaultdict(list)
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        res_lens = batch["response_len"]
        # if args.token_importance:
        #     token_imps = batch["token_importance"].to(model.device)
        last_token_index = -res_lens-1
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states=True)
        layer_num = len(outputs["hidden_states"])
        keep_token_indexes = last_token_index[:, None] - (keep_token_num-1 - torch.arange(keep_token_num)[None, :] )
        for i in range(outputs.hidden_states[0].shape[0]):
            
            target_logit = outputs.logits[i, :, :].unsqueeze(0)
            target_labels = labels[i, :].unsqueeze(0)
            # if args.token_importance:
            #     seq_len = target_logit.shape[1]
            #     timp = torch.cat((torch.zeros(seq_len - token_imps[i].shape[0], device=model.device), token_imps[i]))
            #     loss, mean_entropy = loss_function(target_logit, target_labels, loss_name=loss_name, topk=3, logsoftmax=True, entropy_weight=entropy_weight, token_imps = timp.unsqueeze(0))
            # else:
            loss, mean_entropy = loss_function(target_logit, target_labels, loss_name=loss_name, topk=3, logsoftmax=True, entropy_weight=entropy_weight)

            
            
            
            if entropy_weight:
                grad = torch.autograd.grad(outputs=loss, inputs=outputs["hidden_states"], retain_graph=True, create_graph=False, allow_unused=True)
                hidden_gradient = (torch.cat([ layer.unsqueeze(-2) for layer in grad], dim=-2)[i, keep_token_indexes[i]])[:, :-1, :].detach() # keep_token_num x num_layers  x hidden_size]
    
                
            
                tilhalf_grads_l1_norm_mean = torch.norm(hidden_gradient[import_token_idx, -int((layer_num-1)/2):, :], p=1)/(hidden_gradient[import_token_idx, -int((layer_num-1)/2):, :].numel()) 
            
                scores['semgrad'].append(tilhalf_grads_l1_norm_mean.item())
                

                del grad, hidden_gradient
                torch.cuda.empty_cache()
    
           
            lmh_grad = torch.autograd.grad(outputs=loss, inputs = model.lm_head.weight, retain_graph=True, create_graph=False, allow_unused=True)
            lm_head_grad = torch.cat(lmh_grad).detach()
            
                    
                
            lm_head_l1_norm_mean = torch.norm(lm_head_grad, p=1)/(lm_head_grad.numel())
            
            
            if entropy_weight:
                alpha = np.exp(-mean_entropy.item()).item()
                scores['exgrad_weight'].append(lm_head_l1_norm_mean.item())
                scores["hybridgrad"].append(alpha * lm_head_l1_norm_mean.item() + (1-alpha)*tilhalf_grads_l1_norm_mean.item())
            else:
                scores["exgrad"].append(lm_head_l1_norm_mean.item())
               


                
            del lm_head_grad, lmh_grad,

            torch.cuda.empty_cache()
            
            model.zero_grad()
            torch.cuda.empty_cache()
        
        del outputs, target_logit, loss
        torch.cuda.empty_cache()

    if entropy_weight:
        name = "semgrad"
    else:
        name = "exgrad"
    
    # if args.token_importance:
    #     name = name + "_tokenimp"
    with open(os.path.join(output_dir, f"{name}_scores.json"), 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=4)
    
    truth_label = (np.array(data.scores[args.correctness_metric]) < args.correctness_threshold).astype(int)
    
    logger.info('start to evaluate error detection performace')
    evaluator = Uncertainty_Evaluator(metrics=["auroc", "aucpr", "coverage"])
    evaluator.evaluate(scores, truth_label)
    if output_dir is not None:
    
        evaluator.to_excel(output_dir, name=f"{name}_evaluation_results_{args.correctness_metric}")





