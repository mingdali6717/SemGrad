from ...response_generator import LLM_RESULTS, StandardGenerator
import torch
from ...utils import LLM
from ..utils import expand_truncated_logits, entropy
from .utils import load_paraphrase

from ...utils.read_logits import MAX_LENGTH, remove_pad
from ...utils import reshape_sequences
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import Dataset
from itertools import chain

from . import sub_chars, sub_whole_word_chars, skip_one_char, cap_first_char, sub_spelling_typo, sub_keyboard_typo, sub_synonym, sub_antonym, del_words, swap_words

from .estimation_func import DistributionDistance
from loguru import logger
from tqdm import tqdm
import numpy as np
import time
import copy
from collections import defaultdict

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

class CLEAN_ESI_ESTIMATOR:

    def __init__(self, outputs, sample_num=None, test_model_name=None):
        assert isinstance(outputs, LLM_RESULTS), "outputs should be an instance of LLM_RESULTS"

        assert getattr(outputs, "queries", None) is not None, "queries should be given"
        self.prompts = outputs.prompts
        self.queries = [p.strip() for p in outputs.queries]
        self.queries_for_similarity = outputs.queries_for_similarity
        self.config = copy.deepcopy(outputs.config)


        assert getattr(outputs, "responses", None) is not None, "generated responses should be given"
        self.responses = outputs.responses

        if test_model_name is None:
            assert getattr(outputs, "model_name", None) is not None, "the model used to generate the responses should be given."
            self.generation_model_name = outputs.model_name
            self.model_name = outputs.model_name 
        else:
            if getattr(outputs, "model_name", None) is None:
                self.generation_model_name = "[NOMODEL]"
            else:
                self.generation_model_name = outputs.model_name
            self.model_name = test_model_name
        
        self.tokenizer = LLM.initial_tokenizer(self.model_name)

        template_config = getattr(outputs, "raw_config", None)
        if template_config is None:
            template_config = dummy_config
        
        template_config["model_name"] = self.model_name
        if template_config["system_id"] == 0:
            template_config["system_id"] = None
        self.prompt_template = StandardGenerator(template_config).prompt_template

        if (getattr(outputs, "prompts_ids", None) is None) or (self.generation_model_name != self.model_name):
            self.prompt_ids = self.tokenizer(self.prompts, padding=False, add_special_tokens=True)["input_ids"]
        else:
            self.prompt_ids = outputs.prompt_ids

        if getattr(outputs, "response_ids", None) is None or (self.generation_model_name != self.model_name):
            response_ids = self.tokenizer(self.responses, padding=False, add_special_tokens=False)["input_ids"]
            self.response_ids = [r[:-1] if r[-1] == self.tokenizer.eos_token_id else r for r in response_ids]
        else:
            self.response_ids = outputs.response_ids
            
        self.sample_num = sample_num
    
    def estimate(self, estimation_methods, device_name="gpu0", augfunc="skip", batch_size=20, num_scores_considered=-1, percent=0.3):
        if isinstance(estimation_methods, str):
            estimation_methods = [estimation_methods]

        logger.info("start to calculate scores")
        scores, inference_time = self.sample_and_read_scores(self.queries, self.prompt_ids, self.response_ids, estimation_methods, device_name=device_name, augfunc=augfunc, batch_size=batch_size, num_scores_returned=num_scores_considered, percent=percent)
  
        return scores, inference_time


        

    def _prompts_preprocess(self, prompts, aug_func, aug_num, percent=0.5):
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, list):
            assert isinstance(prompts[0], str), "prompts should be a str or a list of strings"

        return aug_func(prompts, augnum=aug_num, percent=percent)
    
    

    def sample_and_read_scores(self, queries, prompt_ids, response_ids, estimation_methods, device_name="gpu0", augfunc="skip", batch_size=20, num_scores_considered=100, percent=0.5, paraphrase_path=None,memory_efficiency=False):
        
        if isinstance(estimation_methods, str):
            estimation_methods = [estimation_methods]
        lm_name = LLM.initial_lm(self.model_name, device_name)

        model, _ = LLM.loaded_llms[lm_name]

        start_time = time.time()

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

        ue_scores = self._evaluate_scores(merge_input_ids, estimation_methods, model, self.tokenizer, batch_size=batch_size, num_scores_considered=num_scores_considered, prompt_lens=merge_input_lens, memory_efficiency=memory_efficiency)
        end_time=time.time()

        return ue_scores, end_time - start_time
    
    def _evaluate_scores(self, texts_or_ids, estimation_methods, model, tokenizer, batch_size=10, num_scores_considered=100, prompt_lens=None, memory_efficiency=False):
        
        sample_seq_num = self.sample_num + 1
        vocab_size = len(tokenizer.get_vocab())
        if num_scores_considered >= vocab_size:
            print(f"the number of scores returned per token is set to {num_scores_considered} which is larger than the vocab size {vocab_size}, ALL SCORES WILL BE RETURNED")
            num_scores_considered = vocab_size
            truncate_logits = False
        elif num_scores_considered <= 0:
            print(f"the number of scores returned per token is set to {num_scores_considered} which is less than 0, ALL SCORES WILL BE RETURNED")
            num_scores_considered = vocab_size
            truncate_logits = False
        else:
            truncate_logits = True

        if isinstance(texts_or_ids, str) or (isinstance(texts_or_ids, list) and isinstance(texts_or_ids[0], int)):
            texts_or_ids = [texts_or_ids]
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        max_sequence_length = max(getattr(model.config, "max_position_embeddings", 0), getattr(model.config, "n_positions", 0), getattr(model.config, "seq_length", 0))
        if max_sequence_length == 0:
            print(f"model max input length is not detected, set max_prompt_length to {MAX_LENGTH}")
            max_length = MAX_LENGTH
        if isinstance(texts_or_ids[0], str):
            inputs = tokenizer(texts_or_ids, padding=False, truncation=True, max_length=max_length).data
        elif isinstance(texts_or_ids[0][0], int):

            attention_mask = [[1 if _id != tokenizer.pad_token else 0 for _id in ids] for ids in texts_or_ids]

            inputs = {"input_ids": texts_or_ids, "attention_mask": attention_mask}
        else:
            raise ValueError("given texts_or_ids should be a str;List[str];List[int] or List[List[int]]")

        if prompt_lens is None:
            prompt_lens = [1] * len(inputs["input_ids"])
        assert len(prompt_lens) == len(inputs["input_ids"]), f"prompt_len should be with the same length of num of prompts '{len(inputs['input_ids'])}', but {len(prompt_lens)} prompt_len are given"
        inputs["prompt_lens"] = prompt_lens
        ue_scores = defaultdict(list)
        if truncate_logits:
            remain_scores = []
            remain_ids = []
        else:
            remain_scores = []
        
        with torch.no_grad():
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if model.dtype==torch.float16 else None)
            dataloader = DataLoader(Dataset.from_dict(inputs), batch_size=batch_size, collate_fn=data_collator)
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                seq_lens = attention_mask.sum(dim=1).tolist()
                start_time = time.time()
                outputs = model(input_ids = input_ids, attention_mask = attention_mask).logits
                forward_time = time.time() - start_time
                logger.info(f"forward time: {forward_time}")
                if truncate_logits:
                    
                    trun_batch_scores, trun_batch_ids = torch.topk(outputs, num_scores_considered, dim=-1, sorted=True) # shape(batch_size, sequence_length, num_scores_returend_per_token)
                    trun_batch_scores = remove_pad(trun_batch_scores.tolist(), seq_lens)
                    trun_batch_ids = remove_pad(trun_batch_ids.tolist(), seq_lens)

                    remain_scores.extend([s[l-1:-1] for s, l in zip(trun_batch_scores, batch["prompt_lens"].tolist())])
                    remain_ids.extend([s[l-1:-1] for s, l in zip(trun_batch_ids, batch["prompt_lens"].tolist())])
                    if memory_efficiency:
                        start_time = time.time()
                        chunked_scores, remain_scores = chunk_batch(remain_scores, sample_seq_num)
                        chunked_ids, remain_ids = chunk_batch(remain_ids, sample_seq_num)
                        if remain_ids is None:
                            remain_ids = []
                        if chunked_scores is None:
                            continue
                        else:
                            if remain_scores is None:
                                remain_scores = []
                            for ex_scores, ex_ids in zip(chunked_scores, chunked_ids):
                                batch_scores = expand_truncated_logits(ex_scores, ex_ids, expand_to_max_acceptable_size=True)
                                entropy_weight =  entropy(ex_scores[0], index = ex_ids[0])
                                for m in estimation_methods:
                                    score = DistributionDistance(batch_scores, entropy_weight,  distance_measure=m, )
                                    ue_scores[m].append(score["mean"])
                        logger.info(f"calculation time: {time.time() - start_time}")
               
                else:
                    batch_scores = remove_pad(outputs.tolist(), seq_lens)
                    remain_scores.extend([s[l-1:-1] for s, l in zip(batch_scores, batch["prompt_lens"].tolist())])
                    chunked_scores, remain_scores = chunk_batch(remain_scores, sample_seq_num)
                    if chunked_scores is None:
                        continue
                    else:
                        if remain_scores is None:
                            remain_scores = []
                        for ex_scores in chunked_scores:
                            entropy_weight =  entropy(ex_scores[0])
                            for m in estimation_methods:
                                score = DistributionDistance(ex_scores, 
                                entropy_weight,distance_measure=m)
                                ue_scores[m].append(score["mean"])
                score_calculation_time = time.time() - forward_time - start_time
                logger.info(f"calculation time: {score_calculation_time}")

        if truncate_logits and not memory_efficiency:
            start_time = time.time()
            chunked_scores = reshape_sequences(remain_scores, sample_seq_num)
            chunked_ids = reshape_sequences(remain_ids, sample_seq_num)
            
            for ex_scores, ex_ids in zip(chunked_scores, chunked_ids):
                batch_scores = expand_truncated_logits(ex_scores, ex_ids, expand_to_max_acceptable_size=True)
                entropy_weight =  entropy(ex_scores[0], index = ex_ids[0])
                for m in estimation_methods:
                    score = DistributionDistance(batch_scores, entropy_weight, distance_measure=m)
                    ue_scores[m].append(score["mean"])
            
            logger.info(f"calculation time: {time.time() - start_time}")
        return ue_scores
        
    
    def get_augmented_prompts(self, outputs):
        """
        extract the augmented prompts from cached input_ids

        return:
        List[List[str]] - query_num x sample_num
        """
        output_input_ids = [ ids[1:] for ids in outputs["input_ids"]]
        response_len = [len(r) for r in self.response_ids]
        augmented_prompts = [self.tokenizer.batch_decode([prompt_ids[:-r_len] for prompt_ids in ids]) for ids, r_len in zip(output_input_ids, response_len)]
        return augmented_prompts

def chunk_batch(batch_data, chunk_size):
    """
    chunk a batch of data to semi-batch with size chunk_size, return the chunked batches and the remain unchunkable smaller batch

    batch_data: torch.Tensor or List - size batch_size x arbitraty dimensions
    chunk_size: int

    return:
    chunked_data: None or torch.Tensor or List, the chunked data with size chunk_num x chunk_size. None if chunk_size > batch_size, i.e. unchunkable

    remain_data: the tail of the original data with size less than chunk_size, shape (batch_size - chunk_size * chunk_num), None if batch_data % chunk_size = 0.
    """
    if isinstance(batch_data, torch.Tensor):
        is_Tensor=True
        batch_size = batch_data.shape[0]
    else:
        is_Tensor = False
        batch_size = len(batch_data)
    
    chunk_num = batch_size//chunk_size
    if chunk_num == 0:
        return None, batch_data
    if is_Tensor:
        chunked_data = torch.cat([batch_data[i*chunk_size: (i+1)*chunk_size].unsqueeze(0) for i in range(chunk_num)], dim=0)
        
    else:
        chunked_data = [batch_data[i*chunk_size: (i+1)*chunk_size] for i in range(chunk_num)]
        
    if batch_size%chunk_size == 0:
        remain_data = None
    else:
        remain_data = batch_data[chunk_size*chunk_num:]
    
    return chunked_data, remain_data



        

        

        

      
        

    

    