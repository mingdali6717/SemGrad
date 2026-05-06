from uncertainty.generation_evaluation import SemSimCalculator, Evaluator, SUPPORTED_METRIC_NAMES
from itertools import chain
import json
import time
from loguru import logger
import pandas as pd
from itertools import chain
import torch
import os
from uncertainty.utils import LLM

import hashlib

class LLM_RESULTS:

    def __init__(self, generated_sequences=None, prompts=None, prompt_ids=None,  generated_sequences_ids=None, logits=None, transition_scores=None, model_name=None, raw_config=None, generation_config=None, tokenization_config=None, queries=None, ground_truth=None, scores=None, token_importance=None, sim_matrix=None, sim_to_original=None, queries_for_similarity=None, semantic_cluster_ids=None):
        self.responses = generated_sequences
        self.response_ids = generated_sequences_ids
        self.prompts = prompts
        self.prompt_ids = prompt_ids
        self.logits = logits
        self.transition_scores = transition_scores
        self.num_records = len(self.responses)
        self.model_name = model_name
        self.raw_config = raw_config
        self._generation_config = generation_config
        self._tokenization_config = tokenization_config
        self.queries =  queries
        self.ground_truth = ground_truth
        self.scores = scores
        self.token_importance = token_importance
        self.sim_matrix = sim_matrix
        self.sim_to_original = sim_to_original
        self.semantic_cluster_ids = semantic_cluster_ids
        self.queries_for_similarity = queries_for_similarity

        if self.scores is not None:
            self.metrics = list(self.scores.keys())
        else:
            self.metrics = None
        
    

    
    def __getitem__(self, key):
        return self.to_dict()[key]
    
    @property
    def config(self):
        return {"generation_config": self._generation_config,
        "tokenization_config": self._tokenization_config}
    
    @config.setter
    def config(self, config):
        if config is not None:
            self._generation_config = config["generation_config"]
            self._tokenization_config = config["tokenization_config"]
        else:
            self._generation_config = None
            self._tokenization_config = None


    def to_dict(self):
        return {

            "prompts": self.prompts, 
            "prompt_ids": self.prompt_ids, 
            "responses": self.responses, 
            "response_ids": self.response_ids,
            "logits": self.logits,
            "transition_scores": self.transition_scores,
            "queries": self.queries,
            "ground_truth": self.ground_truth,
            "model_name": self.model_name,
            "raw_config": self.raw_config,
            "config": self.config,
            "scores": self.scores,
            "token_importance": self.token_importance,
            "sim_matrix": self.sim_matrix,
            "sim_to_original": self.sim_to_original,
            "semantic_cluster_ids": self.semantic_cluster_ids,
            "queries_for_similarity": self.queries_for_similarity
        }
    
    def to_records(self):
        
        records = list()
        for i in range(self.num_records):
            record = dict()
            for k, v in self.to_dict().items():
                if v is None:
                    record[k] = v
                    continue
                elif k == "model_name" or k == "config" or k == "raw_config":
                    record[k] = v
                elif k == "logits" or  k=="scores" or k == "token_importance" or k == "sim_matrix":
                    new_v = dict()
                    for k_l, v_l in v.items():
                        new_v[k_l] = v_l[i]
                    record[k] = new_v
                elif k == "sim_to_original":
                    new_v = dict()
                    for k_l,v_l in v.items():
                        new_v_l = dict()
                        for k_v_l, v_v_l in v_l.items():
                            new_v_l[k_v_l] = v_v_l[i]
                        new_v[k_l] = new_v_l
                    
                    record[k] = new_v
                else: record[k] = v[i]
            records.append(record)
        return records
    
    @classmethod
    def from_records(cls, records):
        d = records[0]
        for k,v in d.items():
            if k == "model_name" or k == "config" or k == "raw_config":
                continue
            elif isinstance(v, str) or isinstance(v, list):
                d[k] = [v]
            elif isinstance(v, dict):
                new_v = dict()
                for k_l, v_l in v.items():
                    if isinstance(v_l, dict):
                        new_v_l = dict()
                        for k_v_l, v_v_l in v_l.items():
                            new_v_l[k_v_l] = [v_v_l]
                        new_v[k_l] = new_v_l
                    else:
                        new_v[k_l] = [v_l]
                d[k] = new_v

        for i in range(1, len(records)):
            for k,v in records[i].items():
                if v is None:
                    continue
                elif k == "model_name" or k == "config" or k == "raw_config":
                    continue
                elif k == "logits" or  k == "scores" or k == "token_importance" or k == "sim_matrix":
                    for k_l,v_l in v.items():
                        d[k][k_l].append(v_l)
                elif k == "sim_to_original":
                    for k_l,v_l in v.items():
                        for k_v_l, v_v_l in v_l.items():
                            d[k][k_l][k_v_l].append(v_v_l)      
                else:
                    d[k].append(v)
        
        return cls.from_dict(d)    

    @classmethod
    def from_dict(cls, d):
        try:
            responses = d["responses"]
        except:
            raise KeyError("givn dict should contain responses")
        prompts = d.get("prompts", None)
        prompt_ids = d.get("prompt_ids", None)
        response_ids = d.get("response_ids", None)
        logits = d.get("logits", None)
        transition_scores = d.get("transition_scores", None)
        model_name = d.get("model_name", None)
        raw_config = d.get("raw_config", None)
        if "config" in d:
            generation_config = d["config"].get("generation_config", None)
            tokenization_config = d["config"].get("tokenization_config", None)
        else:
            generation_config = None
            tokenization_config = None

            
        queries = d.get("queries", None)
        queries_for_similarity = d.get('queries_for_similarity', None)
        ground_truth = d.get("ground_truth", None)
        scores = d.get("scores", None)
        token_importance = d.get("token_importance")
        
        sim_matrix = d.get("sim_matrix")
        sim_to_original = d.get("sim_to_original")
        semantic_cluster_ids = d.get("semantic_cluster_ids")
        if scores is None:
            new_scores = dict()
            ks = list(d.keys())
            for k in ks:
                if k in SUPPORTED_METRIC_NAMES or "semantic/" in k:
                    new_scores[k] = d[k]
            
            if new_scores != dict():

                scores = new_scores 
            
        return cls(generated_sequences=responses, prompts=prompts, prompt_ids=prompt_ids,  generated_sequences_ids=response_ids, logits=logits, transition_scores=transition_scores, model_name=model_name, raw_config=raw_config, generation_config=generation_config, tokenization_config=tokenization_config, queries=queries, ground_truth=ground_truth, scores=scores, token_importance = token_importance, sim_matrix=sim_matrix, sim_to_original=sim_to_original, queries_for_similarity=queries_for_similarity, semantic_cluster_ids=semantic_cluster_ids)

    @classmethod
    def merge_results(cls, saver1, saver2):
        if isinstance(saver1, dict):
            saver1 = LLM_RESULTS.from_dict(saver1)
        elif not isinstance(saver1, LLM_RESULTS):
            raise KeyError("given input should be a dict or an instance of CLASS LLM_SAVER")
        
        if isinstance(saver2, dict):
            saver2 = LLM_RESULTS.from_dict(saver2)
        elif not isinstance(saver2, LLM_RESULTS):
            raise KeyError("given input should be a dict or an instance of CLASS LLM_SAVER")
    

        assert saver1.model_name == saver2.model_name and saver1.config == saver2.config, "given results are not generated by the same config or model"

        
        return  LLM_RESULTS.from_records((saver1.to_records() + saver2.to_records()))    

    def save(self, save_file, to_records=False):
        if to_records:
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(self.to_records(), f, indent=4)
        else:
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
    
    def to_excel(self, save_file):
        keys_to_save = ["queries", "prompts", "responses", "ground_truth", "model_name", "scores"]
        if isinstance(self.responses[0], list):
            n = len(self.responses[0])
            to_save = dict()
            for k,v in self.to_dict().items():
                if k not in keys_to_save:
                    continue
                elif v is None:
                    continue
                elif k == "model_name":
                    to_save[k] = [v] * self.num_records * n
                elif k == "queries" or k == "prompts":
                    to_save[k] = list(chain(*[[q for _ in range(n)] for q in v]))
                elif k == "scores":
                    for k_s, v_s in v.items():
                        to_save[k_s] = list(chain(*v_s))
                elif k =="ground_truth":
                    gts = [" ;".join(v_i) if isinstance(v_i, list) else v_i for v_i in v]
                    to_save[k] = list(chain(*[[gt for _ in range(n)] for gt in gts]))
                else:
                    to_save[k] = list(chain(*v))
        else:
            to_save = dict()
            for k,v in self.to_dict().items():
                if k not in keys_to_save:
                    continue
                elif v is None:
                    continue
                elif k == "model_name":
                    to_save[k] = [v] * self.num_records
                elif k == "scores":
                    for k_s, v_s in v.items():
                        to_save[k_s] = v_s
                elif k =="ground_truth":
                    to_save[k] = [" ;".join(v_i) if isinstance(v_i, list) else v_i for v_i in v]
                else:
                    to_save[k] = v
        pd.DataFrame.from_dict(to_save).to_excel(save_file)

    
    @classmethod
    def load(cls, load_file):
        with open(load_file, "r", encoding="utf-8") as f:
           data = json.load(f)
        results = cls.from_dict(data)
        if "config" in data:
            results.config = data["config"]
        return results

    def evaluate_correctness(self, evaluation_metrics, queries=None, ground_truth=None, models_for_sim=None, force_evaluate=False, **kwargs):
        if isinstance(evaluation_metrics, str):
            evaluation_metrics = [evaluation_metrics]
        
        if queries is None:
            assert self.queries is not None, f"please provide the quries for evaluating correctness, 'queries' are not found in the cached results"
            queries = self.queries
        
        

        
        if ground_truth is None:
            assert self.ground_truth is not None, f"please provide the ground_truth for evaluating correctness"
            ground_truth = self.ground_truth
        
        evaluator = Evaluator(metrics = evaluation_metrics, models_for_sim=models_for_sim)
        metrics_to_evaluate = []
        metrics_evaluated = []
        if self.scores is None:
            self.scores = dict()
        if self.metrics is not None and not force_evaluate:
            for m in evaluator.metrics:
                if m not in self.metrics:
                    metrics_to_evaluate.append(m)
                else:
                    metrics_evaluated.append(m)
        else:
            metrics_to_evaluate = evaluator.metrics
            
        
        if metrics_to_evaluate == []:
            print("all metrics have been evaluated and saved in the saver, do not need for re-evaluation")
            return None
        elif metrics_evaluated != []:
            print(f"metrics '{'|'.join(metrics_evaluated)} have been evaluated, only need to evaluate metrics '{'|'.join(metrics_to_evaluate)}'")
        else:
            print(f"no evaluated metrics found, start to evaluate all metrics given from scratch")

        evaluator.metrics = metrics_to_evaluate
        queries_to_evaluate = [q.strip() for q in queries]
        if isinstance(self.responses[0], list):
            responses_to_evalute = [[r.strip() for r in rs]for rs in self.responses]
            n = len(self.responses[0])
            queries_to_evaluate = list(chain(*[[q for _ in range(n)] for q in queries_to_evaluate]))
            responses_to_evalute = list(chain(*responses_to_evalute))
            ground_truth_to_evaluate = list(chain(*[[g for _ in range(n)] for g in ground_truth]))
        else: 
            n = 1
            responses_to_evalute = [r.strip() for r in self.responses]
            ground_truth_to_evaluate = ground_truth
        
        to_evaluate = {
            "queries": queries_to_evaluate,
            "responses": responses_to_evalute,
            "ground_truth_answers": ground_truth_to_evaluate
            }
        evaluator.evaluate(to_evaluate, model_evaluated=self.model_name, **kwargs)
        extract_keys = {"questions": queries_to_evaluate, "answers": responses_to_evalute}
        for metric in metrics_to_evaluate:
            score = evaluator.result[extract_keys][metric].tolist()
            if n > 1:
                self.scores[metric] = reshape_sequences(score, n)
            else:
                self.scores[metric] = score
        self.metrics = list(self.scores.keys())
    
    def evaluate_token_importance(self, method="semantic_importance", model="cross-encoder/stsb-roberta-large", device_name=None, batch_size=256, return_inference_time=False):
        """
        return the importance value of each response ids

        return:
        List[List[float]] or List[List[List[float]]]
        shape (num_records x (sample_num) x num_response_ids)
        """
        if self.queries_for_similarity is not None:
            queries = self.queries_for_similarity
        else:
            queries = self.queries
        logger.info(f"start to evaluate the token importance with similarity model '{model}'")
        if self.token_importance is None:
            self.token_importance = dict()
        if method == "semantic_importance":

            if return_inference_time:
                data_process_start_time = time.time()
            if isinstance(self.responses[0], list):

                n = len(self.responses[0])
                queries_to_evaluate = list(chain(*[[q for _ in range(n)] for q in queries]))
                responses_ids_to_evalute = list(chain(*self.response_ids))

            else: 
                n = 1
                queries_to_evaluate = queries
                responses_ids_to_evalute = self.response_ids
            
            if return_inference_time:
                data_process_time = time.time() - data_process_start_time


            tokenizer = LLM.initial_tokenizer(self.model_name, tokenizer_kwargs=self.config["tokenization_config"])
            token_wise_importance, sim_eval_time = get_tokenwise_importance(queries_to_evaluate, responses_ids_to_evalute, tokenizer, model, device_name=device_name, batch_size=batch_size, return_inference_time=True)

            if n > 1:
                scores = reshape_sequences(token_wise_importance, n)
                
            else:
                scores = token_wise_importance
            
        else:
            raise NotImplementedError(f"given method '{method}' is not supported.")

        self.token_importance.update({model: scores})
        if return_inference_time:
            return scores, data_process_time + sim_eval_time
        else:
            return scores
    
    def evaluate_response_similarity(self, model="cross-encoder/stsb-roberta-large", device_name=None, batch_size=512, prepend_query=True, queries=None, responses=None, return_inference_time=False):
        """
        output a similarity matrix with shape (self.num_records, n, n), n is the sample n for each query.
        the diagonal value is 1.0, and the value in i,j is the similarity score of response i and j.

        return:
        List[List[List[float]]] shape(self.num_records x sample_num x sample_num)
        """
        logger.info(f"start to evaluate the similarity matrix with similarity model '{model}'")
        if self.sim_matrix is None:
            self.sim_matrix = dict()
        if responses is None:
            responses = self.responses
        
        if isinstance(responses[0], list):

            n = len(responses[0])
            
        else: 
            n = 1
            logger.warning("there is no sampling generation.can not evaluate inter similarity")
            return None
        
        flatten_cands = []
        flatten_refs = []
        num_records = len(responses)
        
        measure_model = SemSimCalculator(model_name=model, device_name=device_name)

        if return_inference_time:
            start_time = time.time()
        if prepend_query:
            if queries is None:
                if self.queries_for_similarity is not None:
                    queries = self.queries_for_similarity
                else:
                    queries = self.queries
            flatten_queries = []
            for q, r in zip(queries, responses):
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        flatten_cands.append(r[i])
                        flatten_refs.append(r[j])
                        flatten_queries.append(q)
            prepend_text = flatten_queries
        else:
            for r in responses:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        flatten_cands.append(r[i])
                        flatten_refs.append(r[j])
                        
            prepend_text = None
        sim_scores = measure_model(flatten_cands, flatten_refs, prepend_text=prepend_text, batch_size=batch_size).reshape(num_records, n, n-1)
        measure_model.release_model()
        index = torch.arange(n).unsqueeze(0).unsqueeze(0).expand(num_records, n, -1)
        non_diagonal_mask = ~torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(num_records,n,n)
        scatter_index = index[non_diagonal_mask].reshape(num_records, n, n-1)

        sim_matrix = torch.ones((num_records, n, n), dtype=sim_scores.dtype)
        sim_matrix.scatter_(-1, scatter_index, sim_scores)
        sim_matrix = (sim_matrix + torch.transpose(sim_matrix,-1,-2))/2

        if return_inference_time:
            end_time = time.time()

        self.sim_matrix.update({model: sim_matrix.tolist()})
        if return_inference_time:
            return sim_matrix.tolist(), end_time - start_time
        else:
            return sim_matrix
    
    def evaluate_similarity_to_original_answers(self, original_responses, model="cross-encoder/stsb-roberta-large",  device_name=None, batch_size=512, prepend_query=True, queries=None, sampled_responses=None):
        """
        output the similarity scores between the sampled responses and the given responses, with shape (self.num_records, n), n is the sample n for each query.
        

        return:
        List[List[float]] shape(self.num_records x sample_num)
        """
        
        logger.info(f"start to evaluate the similarity to given responses with similarity model '{model}'")
        if self.sim_to_original is None:
            self.sim_to_original = dict()
        if sampled_responses is None:
            sampled_responses = self.responses
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
        
        measure_model = SemSimCalculator(model_name=model, device_name=device_name)
        original_responses_flat = [item for item in original_responses for _ in range(n)]
        sampled_responses_flat = list(chain.from_iterable(sampled_responses))

        if prepend_query:
            if queries is None:
                if self.queries_for_similarity is not None:
                    queries = self.queries_for_similarity
                else:
                    queries = self.queries
            
            queries_flat = [item for item in queries for _ in range(n)]
            prepend_text = queries_flat + queries_flat
        else:          
            prepend_text = None
            queries = None
        
        cands = original_responses_flat + sampled_responses_flat
        refs = sampled_responses_flat + original_responses_flat

        sim_scores = measure_model(cands, refs, prepend_text=prepend_text, batch_size=batch_size).reshape(2, num_records * n).mean(dim=0).reshape(num_records, n).tolist()
        measure_model.release_model()

        key = construct_hash(original_responses + sampled_responses) if queries is None else construct_hash(original_responses + sampled_responses + queries)
        if model not in self.sim_to_original:

            self.sim_to_original[model] = {key: sim_scores}
        else:
            self.sim_to_original[model].update({key: sim_scores})
        
        return sim_scores
    




        


def reshape_sequences(sequences, n):
    """
    reshape input sequences List[] with len m*n to List[List[]],each sublist is with len n
    """
    reshaped = []
    assert len(
        sequences) % n == 0, f"length of sequences should be a multiple of {n}, but the length of given sequences is {len(sequences)}"
    m = int(len(sequences) / n)
    start_id = 0
    for _ in range(m):
        end_id = start_id + n
        reshaped.append(sequences[start_id: end_id])
        start_id = end_id
    return reshaped


def get_tokenwise_importance(questions, response_ids, tokenizer, sim_models, device_name="gpu0", batch_size=256, return_inference_time=False):

    """
    question: List[str] or str
    responses: List[str] or str
    tokenier:  transformers.tokenizer
    sim_models: str a semantic similarity model supported by the class SemSimCakculator. all supported models can be found in uncertainty.generation_evaluation.metrics.semantic_similarity.ALL_SUPPORTED_MODELS
    device_name: str - "cpu" or "gpu{rank}", rank is the int number of current gpu device.

    return: List[List[int]] response_num x token_num, token num is the number of tokens after tokenizing the response with the tokenizer.
    """
    if isinstance(response_ids, list) and (not isinstance(response_ids[0], list)):
        response_ids = [response_ids]
    if isinstance(questions, str):
        questions = [questions]
    measure_model = SemSimCalculator(model_name=sim_models, device_name=device_name)
    if return_inference_time:
        start_time = time.time()
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    token_nums = [len(r) for r in response_ids]
    rm_token_responses = list(chain(*[rm_tokens(r, tokenizer) for r in response_ids]))
    expand_questions = list(chain(*[[q]*n for q,n in zip(questions, token_nums)]))
    expand_responses = list(chain(*[[r]*n for r,n in zip(responses, token_nums)]))
    original_inputs = [q.strip() + " " + r.strip() for q,r in zip(expand_questions, expand_responses)]
    rmtoken_inputs = [q.strip() + " " + r.strip() for q,r in zip(expand_questions, rm_token_responses)]

    flat_scores = (1 - measure_model(original_inputs, rmtoken_inputs, batch_size = batch_size)).tolist()

    if return_inference_time:
        end_time = time.time()
    scores = []
    pos = 0
    for l in token_nums:
        scores.append(flat_scores[pos:pos+l])
        pos+=l
    if return_inference_time:
        return scores, end_time - start_time
    else:
        return scores
    
    

                                                                
def rm_tokens(sentence_id, tokenizer):
    rmed_sentences = []
    for i in range(len(sentence_id)):
        pre_part = sentence_id[:i]
        latter_part = sentence_id[i+1:]
        rm_tok = tokenizer.decode(sentence_id[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)     
        if rm_tok.startswith(" "):
            rmed_sentences.append(tokenizer.decode(pre_part, skip_special_tokens=True, clean_up_tokenization_spaces=True) + " " + tokenizer.decode(latter_part, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        else:
            rmed_sentences.append(tokenizer.decode(pre_part + latter_part, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return rmed_sentences    

def construct_hash(tuple_or_list):
    hash_object = hashlib.sha256()  
    hash_object.update(str(tuple(tuple_or_list)).encode()) 
    return hash_object.hexdigest() 

def list_directories(path):
    """List all directories under the given path using os module."""
    directories = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def collect_results(path):
    dirs = list_directories(path)
    raw_data = []
    for p in dirs:
        with open(os.path.join(p, "output.json"), "r",encoding="utf-8") as f:
            raw_data.append(json.load(f))
    results = functools.reduce(LLM_RESULTS.merge_results, raw_data)
            
    return results