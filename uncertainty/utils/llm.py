from typing import List, Optional
import torch
import json
import time
from tqdm import tqdm
from itertools import chain
import transformers
from transformers import DataCollatorForLanguageModeling
 
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from torch.nn.functional import pad
import gc
import copy
from uncertainty.utils.utils import reshape_sequences
DEFAULT_TOKENIZER_CONFIG = {
    "padding": "longest",
    "truncation": True,
    "padding_side": "right",
    "truncation_side": "right"
}

DEFAULT_GENERATE_CONFIG = {
}



MAX_LENGTH = int(10000)

class LLM:
    

    LLM_MODEL_CONFIG = {
        
        "bertscore/deberta":{
            "model_name": "bertscore/deberta",
            "model_path": "microsoft/deberta-v2-xlarge-mnli",
            "model_class": "AutoModel",
            "fp16": False,
            "tokenizer_class": "AutoTokenizer"
        },
        "nli/deberta_nli":{
          "model_name": "nli/deberta_nli",
          "model_path": "microsoft/deberta-v2-xlarge-mnli",
          "model_class": "AutoModelForSequenceClassification",
          "fp16": False,
          "tokenizer_class": "AutoTokenizer"
        },
        "llama3.1-8b-instruct": {
            "model_name": "llama3.1-8b-instruct",
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "model_path": "meta-llama/Llama-3.1-8B-Instruct",
            "fp16": True
        },
        "llama3.1-70b-instruct": {
            "model_name": "llama3.1-70b-instruct",
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "model_path": "meta-llama/Meta-Llama-3-70B-Instruct",
            "fp16": True
        },
        "mistral-nemo-instruct": {
            "model_name": "mistral-nemo-instruct",
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "model_path": "mistralai/Mistral-Nemo-Instruct-2407",
            "fp16": True
        },
        "qwen3-4b-instruct":{
            "model_name": "qwen3-4b-instruct",
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "model_path": "/Qwen/Qwen3-4B-Instruct-2507",
            "fp16": True
        },
        "qwen3-30b-instruct":{
            "model_name": "qwen3-30b-instruct",
            "model_class": "AutoModelForCausalLM",
            "tokenizer_class": "AutoTokenizer",
            "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "fp16": True
        },
    }
    
    support_models = list(LLM_MODEL_CONFIG.keys())
    loaded_llms = {}
    openai_usage_log = None
    gpu_ids = []
    ddp = False
    
    @classmethod
    def get_llm_config(cls, model_name):
        return cls.LLM_MODEL_CONFIG[model_name] 
    
    @classmethod
    def set_llm_config(cls, configs):
        for k,v in configs.items():
            cls.LLM_MODEL_CONFIG[k] = v 
    
    @classmethod
    def to_torch_device(cls, device_name):
        if device_name is None:
            if not torch.cuda.is_available():
                device_name = "cpu"
            else:
                device_name = [f"gpu{i}" for i in range(torch.cuda.device_count())][0]
        if "gpu" in device_name:
            device = torch.device(f"cuda:{device_name.replace('gpu','')}")
        else:
            device = torch.device("cpu")
        return device
    
    @classmethod
    def initial_lm(cls, model_name, device_name, verbose=True, tokenizer_kwargs=None):
        """
        model_name: should be one of supported model in LLMConfig model_name.
        device_name: in the format of f'gpu{device_id}' or 'cpu' or None
        """
        device = cls.to_torch_device(device_name)

        if device_name is None:
            if not torch.cuda.is_available():
                device_name = "cpu"
            else:
                device_name = [f"gpu{i}" for i in range(torch.cuda.device_count())][0]

        

        if device_name + ":" + model_name in cls.loaded_llms.keys():
            if verbose:
                logger.info(f"***************{device_name}: REUSE LOADED MODEL '{device_name+':'+model_name}*************")
            if tokenizer_kwargs is not None:
                tokenizer = cls.initial_tokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
                cls.loaded_llms[device_name + ":" + model_name] = (cls.loaded_llms[device_name + ":" + model_name][0], tokenizer)
            return device_name + ":" + model_name
        else: 
            if verbose:
                logger.info(f"***************{device_name}: LOAD MODEL '{device_name+':'+model_name} FROM SCRATCH*************")
    
        
        llm_config = cls.LLM_MODEL_CONFIG[model_name]
        
    
        model_class = getattr(transformers, llm_config["model_class"])

        if llm_config["fp16"]:
            if ("llama" in model_name.lower()) or ("mistral" in model_name.lower()) or ("deepseek" in model_name.lower()) or ("qwen" in model_name.lower()):
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

        else:
            dtype = torch.float32

        if ("llama" in model_name.lower() or "deepseek" in model_name.lower() or "qwen" in model_name.lower() or "mistral" in model_name.lower())  and (not getattr(cls, "ddp", False)):
            model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True, local_files_only=True,
                                                torch_dtype=dtype,
                                                device_map="auto")
        else:
            model = model_class.from_pretrained(llm_config["model_path"], trust_remote_code=True, local_files_only=True,
                                                torch_dtype=dtype,
                                                device_map=device)
        
        tokenizer = cls.initial_tokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        if getattr(cls, "ddp", False):
            model = DDP(model, device_ids=[int(device_name.replace("gpu", ""))])
            if verbose:
                logger.info(f"***************DDP model setting finished for {device_name}*************")
        
        model.eval()
        
        
    
        cls.loaded_llms[device_name + ":" + model_name] = (model, tokenizer)
        
        print(f"Successfully initial {device_name + ':' + model_name}")
        
        return device_name + ":" + model_name

    @classmethod
    def initial_tokenizer(cls, model_name, tokenizer_kwargs=None, verbose=True):
        """
        model_name: should be one of supported model in LLMConfig model_name.
        device_name: in the format of f'gpu{device_id}' or 'cpu'
        """
    
        
        llm_config = cls.LLM_MODEL_CONFIG[model_name]
        
        tokenizer_class = getattr(transformers, llm_config["tokenizer_class"])
        tokenizer_path = llm_config.get("tokenizer_path", llm_config["model_path"])
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, local_files_only=True,trust_remote_code=True)


        if tokenizer.pad_token is None:
        
            logger.info(f"pad_token is not given for the tokenizer of model '{tokenizer.name_or_path}', pad_token is set to eos_token '{tokenizer.eos_token}'")
            tokenizer.pad_token = tokenizer.eos_token

        # setting tokenizer_kwargs
        if tokenizer_kwargs is not None and "truncation_side" in tokenizer_kwargs:
            if tokenizer_kwargs["truncation_side"] is not None:
                assert tokenizer_kwargs["truncation_side"] in ["left", "right"], f"truncation side should be 'left' or 'right', but {tokenizer_kwargs['truncation_side']} is given"
                tokenizer.truncation_side = tokenizer_kwargs["truncation_side"]
                tokenizer_kwargs.pop("truncation_side")

        if tokenizer_kwargs is not None and "padding_side" in tokenizer_kwargs:
            if tokenizer_kwargs["padding_side"] is not None:
                assert tokenizer_kwargs["padding_side"] in ["left", "right"], f"padding side should be 'left' or 'right', but {tokenizer_kwargs['padding_side']} is given"
                tokenizer.padding_side = tokenizer_kwargs["padding_side"]
                tokenizer_kwargs.pop("padding_side")
        
        if tokenizer_kwargs is not None and ("chat" in model_name.lower() or "instruct" in model_name.lower() or "it" in model_name.lower()):
            tokenizer_kwargs.update({"add_special_tokens": False})

        

        
        return tokenizer

    
    
    @classmethod
    def release_one(cls, model_name):
        del cls.loaded_llms[model_name]
    
        gc.collect()
        torch.cuda.empty_cache()
    
    @classmethod
    def release_all(cls):
        for llm in cls.loaded_llms.values():
            model, tokenizer = llm
            del model
            del tokenizer
        
        torch.cuda.empty_cache()
        
        cls.loaded_llms = {}
        gc.collect()
        
        if cls.openai_usage_log is not None:
            cls.openai_usage_log.close()        
    
    @classmethod
    def lm_generate(cls, model_or_model_name, prompts, generate_kwargs=None,  tokenizer_kwargs=None, tokenizer=None, device_name = None, verbose=False, return_inference_time=False):
        """
        input:
        model_or_model_name: Str or torch.nn.Module. if an OpenAI model name is given, wil call the openai API. otherwise will initialize based on LLM_MODEL_CONFIG.
        prompts: List[str]
        generate_kwargs: Dict
            {
            batch_size: int,
            temperature: float,
            top_p: float,
            top_k: int,
            do_sample: bool,
            max_new_tokens: int,
            output_scores: bool, if True, will return all output logits for each token position.
            num_score_returned: int, default 100. top-k scores will be returned for each token position, k equal the 'num_scores_returned'. if less or equal to zero, all scores will be returned
            num_responses_per_prompt: int, the number of reponses generated for each prompt, only activate when do_sample=True.
            return_normalized_transition_scores: bool, if true, will return the log softmax of the logits of the generated tokens.
            }
        tokenizer_kwargs: Dict
            {
            padding: str,
            truncation: bool,
            padding_side: str,
            truncation_side: str
            }
        device_name: 'cpu' or 'gpu{gpu_id}'
        verbose: bool
        tokenizer: if model_or_model_name is a nn.module model, the correponding tokenizer is required

        return:

        result - dict:
            keys: 
                - prompts: (List[str]), shape(prompts_num) - the input prompts

                - prompts_ids: (List[List[int]]), shape(prompts_nume, sequence_length) - the corresponding tokenized ids.

                - responses: (List[str] or List[List[str]] if n>1), shape(prompts_num) if n=1 else shape(prompts_num, n) - generated responses. if n = 1, will return a list of shape batch_size

                - response_ids: (List[List[int]] or List[List[List[int]]]), shape(prompts_num, n, max_seq_length) if n > 1 else (prompts_num, max_seq_length)) - the tokenized ids of responses.

                - logits [optional]: dict
                    - scores: (List[List[List[float]]] or List[List[List[List[float]]]]), shape (prompts_num, n, seq_length, num_scores_returned) if n > 1 else (prompts_num, seq_length, num_scores_returned). top-k logits at each position. will be returned if set "output_scores" to True. k is determined by "num_scores_returne"
                    - ids: (List[List[List[int]]] or List[List[List[List[int]]]]), shape (prompts_num, n, seq_length, num_scores_returned) if n > 1 else (prompts_num, seq_length, num_scores_returned). corresponding token it of logit in "scores". will be returned if set "output_scores" to True. k is determined by "num_scores_returne"

                - transition_scores [optional]: (List[List[float]] or List[List[List[float]]]) shape (prompts_num, n, seq_length) if n > 1 else (prompts_num, seq_length). logprob of each generated tokens, will be returned if "return_normalized_transition_scores" is True
        """
        if generate_kwargs is None:
            generate_kwargs = copy.deepcopy(DEFAULT_GENERATE_CONFIG)
        if tokenizer_kwargs is None:
            tokenizer_kwargs = copy.deepcopy(DEFAULT_TOKENIZER_CONFIG)
        config = {
           "generation_config": copy.deepcopy(generate_kwargs),
           "tokenization_config": copy.deepcopy(tokenizer_kwargs)
        }

        
        # load model and tokenizer
        if isinstance(model_or_model_name, str):
            assert model_or_model_name in cls.LLM_MODEL_CONFIG.keys(), f"model '{model_or_model_name}' is not supported!"
            model_key = cls.initial_lm(model_or_model_name, device_name, verbose, tokenizer_kwargs=tokenizer_kwargs)
            model, tokenizer = cls.loaded_llms[model_key]
            model_name = model_or_model_name
        else:
            assert isinstance(model_or_model_name, torch.nn.Module), f"given model should be name string or a nn.Module, but {type(model_or_model_name)} is given"
            assert tokenizer is not None, "please provide a corresponding tokenizer!!!" 
            model = model_or_model_name
            model_name = model.config._name_or_path
        
        # setting generate_kwargs
        if ("num_responses_per_prompt" in generate_kwargs) :
            if generate_kwargs["do_sample"] and (generate_kwargs["num_responses_per_prompt"] > 1):
                n = generate_kwargs["num_responses_per_prompt"]
            else:
                if generate_kwargs["num_responses_per_prompt"] > 1:
                    if generate_kwargs.get("num_beams", None) is not None:
                        assert generate_kwargs["num_beams"] >= generate_kwargs["num_responses_per_prompt"], f"num_beams should be larger than or equal to num_responses_per_prompt, but {generate_kwargs['num_beams']} < {generate_kwargs['num_responses_per_prompt']}"
                        n = generate_kwargs["num_responses_per_prompt"]
                    else:
                        logger.warning(f"num_responses_per_prompt is set to {generate_kwargs['num_responses_per_prompt']} > 1, but do_sample is set to False and num_beams is not given, hence only 1 response will be generated for each prompt")
                        n = 1
                else:
                    n=1
            generate_kwargs.pop("num_responses_per_prompt")
        else:
            n = 1
        generate_kwargs["num_return_sequences"] = n
        if ("return_normalized_transition_scores" in generate_kwargs) :

            return_normalized_transition_scores = generate_kwargs.pop("return_normalized_transition_scores") 
            if return_normalized_transition_scores is None:
                return_normalized_transition_scores = False
        else:
            return_normalized_transition_scores=False
        if generate_kwargs.get("eos_token_id", None) is not None:
            print(f"other eos_tokens are provided: {generate_kwargs['eos_token_id']}")
            if isinstance(generate_kwargs["eos_token_id"], str):
                generate_kwargs["eos_token_id"] = [generate_kwargs["eos_token_id"]]
            single_eos_token_ids, other_eos_token_ids = process_eos_token(generate_kwargs["eos_token_id"], tokenizer)
            
            generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(chain(*single_eos_token_ids))
        else:
            other_eos_token_ids = None
        

            
        if ("llama3" in model_name.lower() or "llama-3" in model_name.lower()) and "instruct" in model_name.lower():
            if generate_kwargs.get("eos_token_id", None) is not None:
                if tokenizer.eos_token_id not in generate_kwargs["eos_token_id"]:
                    generate_kwargs["eos_token_id"].append(tokenizer.eos_token_id)
                if tokenizer.convert_tokens_to_ids("<|eot_id|>") not in generate_kwargs["eos_token_id"]:
                    generate_kwargs["eos_token_id"].append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            else:
                generate_kwargs.update({"eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]})
        if generate_kwargs.get("eos_token_id", None) is not None:
            logger.info(f"additional eos tokens are given:\nsingle eos_token_ids: {generate_kwargs['eos_token_id']}\nother eos_token_ids : {other_eos_token_ids}")
        
        result = cls._frozen_lm_generate(model, tokenizer, prompts, tokenizer_kwargs, generate_kwargs, return_normalized_transition_scores=return_normalized_transition_scores, additional_eos_token_ids=other_eos_token_ids, return_inference_time=return_inference_time)
        if return_inference_time:
            inference_time = result[1]
            result = result[0]
        result["config"] = config
        
        result["model_name"] = model_name
        logger.info(F"generation finished for LLM {model_name}")
        if not return_inference_time:
            return result
        else:
            return result, inference_time
    
    @classmethod
    def _frozen_lm_generate(cls, model, tokenizer, prompts, tokenize_kwargs, generate_kwargs,return_normalized_transition_scores=False, additional_eos_token_ids=None, return_inference_time=False):
        
        if type(prompts) is str:
            prompts = [prompts]
        
        output_scores = generate_kwargs.get("output_scores", False) if generate_kwargs.get("output_scores", False) is not None else False

        num_scores_returned = generate_kwargs.get("num_scores_returned", 100) if generate_kwargs.get("num_scores_returned", 100) is not None else 100
        if "num_scores_returned" in generate_kwargs:
            generate_kwargs.pop("num_scores_returned")
        terminators = generate_kwargs.get("eos_token_id")
        vocab_size = tokenizer.vocab_size
        if num_scores_returned >= vocab_size:
            logger.info(f"the number of scores returned per token is set to {num_scores_returned} which is larger than the vocab size {vocab_size}, ALL SCORES WILL BE RETURNED")
            num_scores_returned = vocab_size
        elif num_scores_returned <= 0:
            logger.info(f"the number of scores returned per token is set to {num_scores_returned} which is less than 0, ALL SCORES WILL BE RETURNED")
            num_scores_returned = vocab_size

        if return_normalized_transition_scores:
            generate_kwargs["output_scores"] = True
        
        
        n = generate_kwargs["num_return_sequences"]

        
        module = model.module if LLM.ddp else model
        max_sequence_length = max(getattr(module.config, "max_position_embeddings", 0), getattr(module.config, "n_positions", 0), getattr(module.config, "seq_length", 0))
        max_prompt_length, max_new_tokens = adjust_length_to_model(generate_kwargs["max_new_tokens"], max_sequence_length) 
        generate_kwargs["max_new_tokens"] = max_new_tokens
        tokenize_kwargs["max_length"] = max_prompt_length
        tokenize_kwargs["truncation"] = True
        #tokenize_kwargs["return_tensors"] = "pt"
        padding = tokenize_kwargs.pop("padding")
        tokenize_kwargs["padding"] = False
        
        if return_inference_time:
            start_time = time.time()
        tokenize_prompt = tokenizer(prompts, **tokenize_kwargs)
        prompts_ids = tokenize_prompt.data["input_ids"]
        
        batch_size = generate_kwargs.pop("batch_size", 10)
        
        with torch.no_grad():
            
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if module.dtype==torch.float16 else None)
            dataloader = DataLoader(Dataset.from_dict(tokenize_prompt.data), batch_size=batch_size, collate_fn=data_collator)
            if terminators is not None:
                eos_token_id = terminators
            else:
                if hasattr(model, "generation_config"):
                    eos_token_id = module.generation_config.eos_token_id
                else:
                    eos_token_id = module.config.eos_token_id
            generated_sequences = []
            scores = []
            ids = []
            generated_ids = []
            transition_scores = []
            
            for batch in tqdm(dataloader, desc=f"generation with model '{tokenizer.name_or_path}'"):
                input_ids = batch["input_ids"].to(module.device)
                attention_mask = batch["attention_mask"].to(module.device)
                outputs = module.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, **generate_kwargs)
                # print_gpu_usage(input_ids.get_device())
                output_seq = outputs.sequences[:, input_ids.shape[1]:] # shape(batch_size*n, sequence_length)
                seq_lens = get_sequence_length(output_seq.tolist(), eos_token_id)
                if additional_eos_token_ids is not None: 
                    truncate_lens = [truncate_eos_tokens(r_ids, additional_eos_token_ids) for r_ids in output_seq.tolist()]
                    seq_lens = [min(t_l, s_l) for t_l, s_l in zip(truncate_lens, seq_lens)]
                
                if output_scores:
                    score = torch.cat([l.unsqueeze(1) for l in outputs["scores"]], dim=1) # shape(batch_size*n, sequence_length, vocab_size)

                    trun_scores, trun_ids = torch.topk(score, num_scores_returned, dim=-1, sorted=True) # shape(batch_size*n, sequence_length, num_scores_returend_per_token)

                        

                    scores.extend(remove_pad(trun_scores.tolist(), seq_lens))
                    ids.extend(remove_pad(trun_ids.tolist(), seq_lens))
                    

                if return_normalized_transition_scores:
                    transition_score = module.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)  # shape(batch_size*n , seq_length)
                    transition_scores.extend(remove_pad(transition_score.tolist(), seq_lens))
                gen_ids = remove_pad(output_seq.tolist(), seq_lens)
                output_str = [tokenizer.decode(r_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for r_ids in gen_ids]
                generated_ids.extend(gen_ids)
                generated_sequences.extend(output_str)
            
            logger.info("batch generation finished!! start to process results")
            

            if n > 1:
                generated_sequences = reshape_sequences(generated_sequences, n)
                generated_ids = reshape_sequences(generated_ids, n)
                if output_scores:
                    scores = reshape_sequences(scores, n)
                    ids = reshape_sequences(ids, n)
                if return_normalized_transition_scores:
                    transition_scores = reshape_sequences(transition_scores, n)
            
            if return_inference_time:
                end_time = time.time()

            result = {"prompts": prompts, "prompt_ids": prompts_ids, "responses": generated_sequences, "response_ids": generated_ids}
            if scores is not None:
                result["logits"] = {"scores": scores, "ids": ids}

            if return_normalized_transition_scores:
                result["transition_scores"] = transition_scores
            if return_inference_time:
                return result, end_time - start_time
            else:
                return result
    


def adjust_length_to_model(max_new_tokens, max_sequence_length=0):
    if max_new_tokens < max_sequence_length:
        max_prompt_length = max_sequence_length - max_new_tokens
        if max_prompt_length < int(0.2 * max_sequence_length):
            logger.warning(f"the max_prompt_length {max_prompt_length} is a bit small, less than 20 percent of the total acceptable length {max_sequence_length}, it is recommended to reduce the max_new_tokens to support longer input length.")
    elif max_new_tokens >= max_sequence_length and max_sequence_length > 0:
        print(f"model max input length is {max_sequence_length}, but given max_new_tokens are {max_new_tokens} which is intractable, max_new_tokens will be set to {int(0.3 * max_sequence_length)}")
        max_prompt_length = int(0.7 * max_sequence_length)
        max_new_tokens = max_sequence_length - max_prompt_length
    else:
        print(f"model max input length is not detected, set max_prompt_length to {MAX_LENGTH}")
        max_prompt_length = MAX_LENGTH

    return max_prompt_length, max_new_tokens

def get_sequence_length(seq_ids, eos_token_id):
    """
    get the seqence length of the generated sequence given the eos_ids

    input:
    seq_ids: List[List[int]]
    eos_token_id: int or List[int]
    """
    if not isinstance(eos_token_id, list):
        eos_token_id = [eos_token_id]
    seq_lengths = []
    for i in range(len(seq_ids)):
        indexes = []
        for e_token in eos_token_id:
            try :
                idx = seq_ids[i].index(e_token)+1
            except:
                idx = len(seq_ids[i])
            indexes.append(idx)
        seq_lengths.append(min(indexes))
    return seq_lengths

def remove_pad(data_list, seq_lengths):
    """
    remove the padding position in the second dimenson of data_list following the seqence length given in seq_lengths
    input: 
        data_list: List[List[****]] batch_size x padded_sequence_length x *****
        seq_length: List, batchsize.

    """
    assert len(data_list) == len(seq_lengths), f"data_list (length {len(data_list)}) and seq_lengths {len(seq_lengths)} are in different lens."
    for i in range(len(data_list)):
        data_list[i] = data_list[i][:seq_lengths[i]]
    return data_list

def expand_cat(tensor_a, tensor_b, cat_dim=0, pad_value=0, pad_dim=-1):
    
    tensor_size = len(tensor_a.shape)
    if pad_dim >= 0:
        pad_dim =  pad_dim - tensor_size
    max_width = max(tensor_a.shape[pad_dim], tensor_b.shape[pad_dim])
    padding_a = -1*pad_dim*[0, 0]
    padding_b = -1*pad_dim*[0, 0]
    padding_a[-1] = max_width - tensor_a.shape[pad_dim]
    padding_b[-1] = max_width - tensor_b.shape[pad_dim]

    
    # Calculate padding (only right side, hence last two values in the padding tuple)
    

    padded_a = pad(tensor_a, padding_a, "constant", value=pad_value)  # Apply padding
    padded_b = pad(tensor_b, padding_b, "constant", value=pad_value)  # Apply padding

    # Step 3: Concatenate the tensors
    return torch.cat((padded_a, padded_b), dim=cat_dim)

def process_eos_token(eos_tokens, tokenizer):
    eos_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(eos_token)) if isinstance(eos_token, str) else eos_token for eos_token in eos_tokens]
    single_eos_token_ids = [_id for _id in eos_token_ids if len(_id) == 1]
    other_eos_token_ids = [_id for _id in eos_token_ids if len(_id) > 1]
    if other_eos_token_ids == []:
        other_eos_token_ids = None
    
    return single_eos_token_ids, other_eos_token_ids

def truncate_eos_tokens(response_ids, eos_token_lists, protect_first=True):
    if isinstance(eos_token_lists[0], int):
        eos_token_lists = [eos_token_lists]
    all_start_positions = []
    for eos_token in eos_token_lists:
        
        length = len(eos_token)

        start_position = len(response_ids)
        for i in range(len(response_ids) - length + 1):
            # Check if the elements from i to i+length match the sublist
            if response_ids[i:i+length] == eos_token:
                if i == 0 and protect_first:
                    continue
                else:
                    start_position = i
                    break
        
        all_start_positions.append(start_position)
    
    return min(all_start_positions)

def find_eos_start_position(response_ids, eos_token, protect_first=True):
    length = len(eos_token)
    for i in range(len(response_ids) - length + 1):
        # Check if the elements from i to i+length match the sublist
        if response_ids[i:i+length] == eos_token:
            if i == 0 and protect_first:
                continue
            else:
                start_position = i
            break
