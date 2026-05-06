from ...response_generator import LLM_RESULTS, StandardGenerator, construct_hash
import torch
from ...utils import LLM, get_logits, reshape_sequences, get_gpu_memory
from ..utils import expand_truncated_logits, entropy
from .utils import load_paraphrase
from itertools import chain
from functools import reduce, partial
from . import sub_chars, sub_whole_word_chars, skip_one_char, cap_first_char, sub_spelling_typo, sub_keyboard_typo, sub_synonym, sub_antonym, del_words, swap_words
import os
import json 
from .estimation_func import DistributionDistance
from loguru import logger
from tqdm import tqdm
import numpy as np
import copy
import time

dummy_config = {
    "verbose": False,
    "system_id": None,
    "template_id":2,
    "generate_kwargs": dict()
}
AUG_FUNC_MAPPING = {
    "char": sub_chars,
    "word": sub_whole_word_chars,
    "skip": skip_one_char,
    "cap": cap_first_char,
    "keyboard": sub_keyboard_typo,
    "spell": sub_spelling_typo,
    "synonym": sub_synonym,
    "antonym":sub_antonym,
    "del": del_words,
    "swap": swap_words
}

DEFAULT_ESI_CONFIG = {
    "augfunc": "skip", 
    "sample_num": 10,
    "cached_result_path": None,
    "save_path": None,
    "num_scores_returned":100,
    "percent": 0.3,
    "sim_batch_size":256,

}

class ESI_ESTIMATOR:

    def __init__(self, outputs, sample_num=None, cached_result_path=None):
        assert isinstance(outputs, LLM_RESULTS), "outputs should be an instance of LLM_RESULTS"

        assert getattr(outputs, "prompts", None) is not None, "prompts should be given"
        self.prompts = outputs.prompts
        self.queries = [p.strip() for p in outputs.queries] if outputs.queries is not None else [p.strip() for p in outputs.prompts]
        self.queries_for_similarity = outputs.queries_for_similarity
        self.config = copy.deepcopy(outputs.config)


        assert getattr(outputs, "responses", None) is not None, "generated responses should be given"
        self.responses = outputs.responses

        assert getattr(outputs, "model_name", None) is not None, "the model used to generate the responses should be given."
        self.model_name = outputs.model_name
        self.tokenizer = LLM.initial_tokenizer(self.model_name)

        template_config = getattr(outputs, "raw_config", None)
        if template_config is None:
            template_config = dummy_config
        
        template_config["model_name"] = self.model_name
        if template_config["system_id"] == 0:
            template_config["system_id"] = None
        self.prompt_template = StandardGenerator(template_config).prompt_template

        if (getattr(outputs, "prompts_ids", None) is None):
            self.prompt_ids = self.tokenizer(self.prompts, padding=False, add_special_tokens=True)["input_ids"]
        else:
            self.prompt_ids = outputs.prompt_ids

        if getattr(outputs, "response_ids", None) is None:
            response_ids = self.tokenizer(self.responses, padding=False, add_special_tokens=False)["input_ids"]
            self.response_ids = [r[:-1] if r[-1] == self.tokenizer.eos_token_id else r for r in response_ids]
        else:
            self.response_ids = outputs.response_ids
            
        self.sample_num = sample_num

        if cached_result_path is not None:
            self.cached_dir = os.path.dirname(cached_result_path)
            self.load(cached_result_path)
        else:
            self.cached_result = None
            self.cached_dir = None
    
    def estimate(self, device_name="gpu0", augfunc="skip", batch_size=20, num_scores_returned=100, save_path=None, percent=0.3, sim_batch_size=256, sampling_only=False, paraphrase_path=None):

        logger.info("start to read logits")
        outputs = self.read_logits(device_name=device_name, augfunc=augfunc, batch_size=batch_size, num_scores_returned=num_scores_returned, percent=percent, paraphrase_path = paraphrase_path)

        logger.info("logits readed")
        if save_path is not None:
            logger.info(f"intermediate results are saved to {save_path}")
            self.save(save_path)
        
        if sampling_only:
            return None


        logits = outputs["logits"]["scores"]
        indexes = outputs["logits"]["ids"]
        transition_scores = outputs["transition_scores"]
        
            
        logger.info("start to expand logits")
        
        expanded_logits = [expand_truncated_logits(logit, indexes=index, vocab_size=self.tokenizer.vocab_size, expand_to_max_acceptable_size=True) for logit, index in tqdm(zip(logits,indexes))]
    
        entropy_weight = [entropy(l[0], index = i[0]) for l, i in zip(logits, indexes)]

        scores = dict()  
        
        
        logger.info(f"start to estimate ESI")
    
            
        name = ["esi"] + [augfunc]

        for dm in ["hellinger", "kl", "Bhatacharyya"]:
            
            weight = entropy_weight
            
            
            distribution_scores = [DistributionDistance(logit, wt, distance_measure=dm, grid_sizes=None) for logit, wt in tqdm(zip(expanded_logits, entropy_weight), desc=f"computing distribution scores for {'_'.join(name + [dm])}")]
            
            
            logger.info(f"Estimation Finished for {dm}")
            
            scores.update({'_'.join(name + [dm] + ["mean"]) :  [ds["mean"] for ds in distribution_scores]})
            
    
        return scores


        

    def _prompts_preprocess(self, prompts, aug_func, aug_num, percent=0.5):
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, list):
            assert isinstance(prompts[0], str), "prompts should be a str or a list of strings"

        return aug_func(prompts, augnum=aug_num, percent=percent)
    
    

    def _sample_and_read_logits(self, queries, prompts, responses, prompt_ids, response_ids, device_name=None, augfunc="skip", batch_size=20, num_scores_returned=100, percent=0.3, paraphrase_path=None, **kwargs):
        """
        inputs:
        queries: List[str] - shape(num_prompts)
        prompts: List[str] - shape(num_prompts)
        responses: List[str] - shape(num_prompts)
        prompts_ids: List[List[int]] - shape(num_prompts x prompt_len)
        response_ids: List[List[int]] - shape(num_prompts x response_len)
        device_name: str - "cpu" or "gpu{gpu_id}"
        augfunc: str - the function used to intervene prompts
        batch_size: int - batch for reading logits
        num_scores_returned: int - the top-k logits retained
        percent: float - percentage of words are intervened for non-paraphrase methods
        paraphrase_path: string - path to the pre-generated paraphrased prompts.



        return:
        results will be saved to self.cached_result
        
        {
            "model_name": str,
            "sample_num": int,
            "aug_func": str,
            "data": {
                self.key(p_1,k_1):{
                    "input_ids": List[List[int]], shape((self.sample_num + 1) x input_len),
                    "logits": {
                        "scores": List[List[List[int]]], shape((self.sample_num + 1) x response_len x num_scores_returned),
                        "ids": List[List[List[int]]], shape((self.sample_num + 1) x response_len x num_scores_returned)
                    },
                    "transition_scores": List[List[int]], shape((self.sample_num + 1) x response_len)
                }
            }
        }

        """
        if "_full" in augfunc:
            augfunc_method = augfunc.replace("_full", "")
            full_queries = [self.prompt_template.fill({"query":q}) for q in queries]
            processed_queries = list(chain(*self._prompts_preprocess(full_queries, AUG_FUNC_MAPPING[augfunc_method], self.sample_num, percent=percent)))
            processed_prompts = [self.prompt_template.build_model_specific_prompt(q) for q in processed_queries]
        elif augfunc == "paraphrase":
            assert paraphrase_path is not None, f"paraphrased queries should be given for testaug method {augfunc}"
            paraphrased_queries_mapping = load_paraphrase(paraphrase_path, self.sample_num)
            
            processed_queries = list(chain(*[paraphrased_queries_mapping[q.strip()] for q in queries]))
            processed_prompts = [self.prompt_template.build_prompt({"query":q}) for q in processed_queries]
        else:
            processed_queries = list(chain(*self._prompts_preprocess(queries, AUG_FUNC_MAPPING[augfunc], self.sample_num, percent=percent)))
            processed_prompts = [self.prompt_template.build_prompt({"query":q}) for q in processed_queries]
        processed_prompts_ids = self.tokenizer(processed_prompts, padding=False, add_special_tokens=True)["input_ids"]
        processed_responses_ids = list(chain(*[[r_ids]*self.sample_num for r_ids in response_ids]))
        input_ids = [p + r for p, r in zip(prompt_ids, response_ids)]
        input_lens = [len(p) for p in prompt_ids]
        processed_input_ids = [p + r for p, r in zip(processed_prompts_ids, processed_responses_ids)]
        processed_input_lens = [len(p) for p in processed_prompts_ids]

        # merge and reorder the original and process inputs to make one original texts followed by num_sample processed texts.
        merge_input_ids = list(chain(*[[input_ids[i]] + processed_input_ids[i*self.sample_num: (i+1)*self.sample_num] for i in range(len(input_ids))]))

        merge_input_lens = list(chain(*[[input_lens[i]] + processed_input_lens[i*self.sample_num: (i+1)*self.sample_num] for i in range(len(input_lens))]))


        lm_name = LLM.initial_lm(self.model_name, device_name)

        model, _ = LLM.loaded_llms[lm_name]



        logits = get_logits(merge_input_ids, model, self.tokenizer, batch_size=batch_size, return_transition_scores=True, num_scores_returned=num_scores_returned, prompt_lens=merge_input_lens)

        LLM.release_all()
        logits["input_ids"] = reshape_sequences(logits["input_ids"], self.sample_num + 1)
        logits["transition_scores"] = reshape_sequences(logits["transition_scores"], self.sample_num + 1)

        for k, v in logits["logits"].items():
            logits["logits"][k] = reshape_sequences(v, self.sample_num + 1)

        
        cached_data = dict()
        for i, (p, r) in enumerate(zip(prompts, responses)):
            cached_data[self.key(p, r)] = {
                "input_ids": logits["input_ids"][i],
                "logits": {
                    "scores": logits["logits"]["scores"][i],
                    "ids": logits["logits"]["ids"][i]
                },
                "transition_scores": logits["transition_scores"][i]
            }
        
        if self.cached_result is not None:
            self.cached_result["data"].update(cached_data)
        else:
            self.cached_result = {
                "model_name": self.model_name,
                "sample_num": self.sample_num,
                "aug_func": augfunc,
                "percent": percent,
                "data": cached_data
            }
    
    def read_logits(self, device_name="gpu0", augfunc="char", batch_size=20, num_scores_returned=500, percent=0.5, paraphrase_path = None):
        """
        read logits based on self.prompts and self.responses.
        inputs:
        device_name: str - "cpu" or "gpu{gpu_id}"
        augfunc: str - the function used to intervene prompts
        batch_size: int
        num_scores_returned: int

        return:
        logits - dict. the logits is packed for each prompt in the first dimension. in the second dimension, the first index is the value(logits, tranisiton_score) for the original response, the other self.sample_num indexs are for the intervene and sampled responses.
        {
            "input_ids": List[List[List[int]]], shape((prompts_num, self.sample_num + 1, input_len),
            "logits": {
                "scores": List[List[List[List[int]]]], shape(prompts_num, self.sample_num + 1, response_len, num_scores_returned),
                "ids": List[List[List[List[int]]]], shape(prompts_num, self.sample_num + 1, response_len, num_scores_returned)
            },
            "transition_scores": List[List[List[int]]], shape(prompts_num, self.sample_num + 1, response_len)
        }
        """
        if self.cached_result is None:
            load_from_cache = False
            logger.info(f"No cached result found, start to regenerate from scratch")
            
        else:
            if self.cached_result["aug_func"] != augfunc:
                logger.info(f"cached result used augmentattion method '{self.cached_result['aug_func']}' which is not the same as the given aug_func '{augfunc}', start to regenerate from scratch")
                load_from_cache = False
            elif self.cached_result["percent"] != percent:
                logger.info(f"cached result used augmentattion percent '{self.cached_result['percent']}' which is not the same as the given percent '{percent}', start to regenerate from scratch")
                load_from_cache = False
            else:
                load_from_cache = True

        if not load_from_cache:
            self._sample_and_read_logits(self.queries, self.prompts, self.responses, self.prompt_ids, self.response_ids,device_name=device_name, augfunc=augfunc, batch_size=batch_size, num_scores_returned=num_scores_returned, percent=percent, paraphrase_path=paraphrase_path)
        else:
            non_cached_queries, non_cached_prompts, non_cached_responses, non_cached_prompt_ids, non_cached_response_ids = [], [], [], [], []
            for i in range(len(self.prompts)):
                if self.key(self.prompts[i], self.responses[i]) in self.cached_result["data"]:
                    continue
                else:
                    non_cached_queries.append(self.queries[i])
                    non_cached_prompts.append(self.prompts[i])
                    non_cached_prompt_ids.append(self.prompt_ids[i])
                    non_cached_responses.append(self.responses[i])
                    non_cached_response_ids.append(self.response_ids[i])
            
                
            if non_cached_prompts != []:
                logger.info(f"{len(non_cached_prompts)} non cached prompts found, start to read logits")
                self._sample_and_read_logits(non_cached_queries, non_cached_prompts, non_cached_responses, non_cached_prompt_ids,non_cached_response_ids,device_name=device_name, augfunc=augfunc, batch_size=batch_size, num_scores_returned=num_scores_returned, percent=percent)
            else:
                logger.info(f"all prompts found in cached results.")
    
        return self.read_from_cache(self.prompts, self.responses)
        

    def read_from_cache(self, prompts, responses):
        logits ={
                "input_ids": [],
                "logits": {
                 "scores": [],
                    "ids": []
                    },
                "transition_scores": []
        }
        for p, r in zip(prompts, responses):
            d = self.cached_result["data"][self.key(p,r)]
            for k,v in d.items():
                if isinstance(v, dict):
                    for k_v, v_v in v.items():
                        logits[k][k_v].append(v_v)
                else:
                    logits[k].append(v)
        return logits
                
                

    def check_cached_results(self):
        assert self.cached_result["model_name"] == self.model_name, f"the given cached file is generate by model '{self.cached_result['model_name']}', but the model '{self.model_name}' is given"


        if self.cached_result["sample_num"] < self.sample_num:
            logger.info(f"the cached result only sample {self.cached_result['sample_num']} responses, but {self.sample_num} responses required, hence regenerate from scratch.")
            self.cached_result = None
        elif self.cached_result["sample_num"] > self.sample_num:
            for k, logit in self.cached_result["data"].items():
                for k_l, v_l in logit.items():
                    if isinstance(v_l, dict):
                        for k_v, v_v in v_l.items():
                            self.cached_result["data"][k][k_l][k_v] = v_v[:self.sample_num+1]
                    else:
                        self.cached_result["data"][k][k_l] = v_l[:self.sample_num+1]

            self.cached_result["sample_num"] = self.sample_num
        
    def key(self, prompt, response):
        return prompt+"[sep]"+response+"[sep]" + self.model_name
    
    def load(self, path):
        assert os.path.exists(path), f"the given cached result path '{path}' do not exits."
        logger.info(f"cached result found, start to load from '{path}'")

        with open(path, "r", encoding="utf-8") as f:
            self.cached_result = json.load(f)
        self.check_cached_results()

        logger.info(f"cached result loaded")
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cached_result, f, indent=4)
    
    def get_intervened_prompts(self, outputs):
        """
        extract the intervened prompts from cached input_ids

        return:
        List[List[str]] - query_num x sample_num
        """
        output_input_ids = [ ids[1:] for ids in outputs["input_ids"]]
        response_len = [len(r) for r in self.response_ids]
        intervened_prompts = [self.tokenizer.batch_decode([prompt_ids[:-r_len] for prompt_ids in ids]) for ids, r_len in zip(output_input_ids, response_len)]
        return intervened_prompts
        



        

        

        

      
        

    

    